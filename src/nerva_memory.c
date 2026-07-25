// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_memory.h"
#include "nerva_mutation.h"
#include "nerva_trace.h"

#include <string.h>

static uint32_t nerva_memory_count_traces_for_query(const NervaEngine *e, uint32_t query_tag) {
    if (!e || query_tag == 0) {
        return 0;
    }

    uint32_t count = 0;
    uint32_t n = e->trace_count < e->trace_cap ? e->trace_count : e->trace_cap;
    for (uint32_t age = 0; age < n; ++age) {
        NervaTrace *t = nerva_trace_recent((NervaEngine *)e, age);
        if (!t) {
            continue;
        }
        if (t->query_tag == query_tag && (t->flags & NERVA_TRACE_USED_PATH)) {
            count++;
        }
    }
    return count;
}

static int nerva_memory_has_open_episode(const NervaEngine *e) {
    if (!e) {
        return 0;
    }
    for (uint32_t i = 0; i < e->memory_count; ++i) {
        if (e->memory[i].state & NERVA_MEM_STATE_OPEN) {
            return 1;
        }
    }
    return 0;
}

static void nerva_memory_refresh_state(NervaEngine *e, NervaMemoryBlock *m) {
    if (!e || !m) {
        return;
    }

    if (m->charge > e->cfg.memory_store_threshold) {
        m->state |= NERVA_MEM_STATE_CONSOLIDATED;
    }
    if (m->charge >= e->cfg.memory_forget_threshold) {
        m->low_charge_since = NERVA_MEM_LOW_CHARGE_UNSET;
    } else if (m->low_charge_since == NERVA_MEM_LOW_CHARGE_UNSET) {
        m->low_charge_since = e->tick;
    }
}

uint32_t nerva_memory_begin_episode(NervaEngine *e, uint32_t query_tag) {
    if (!e || e->memory_count >= e->memory_cap) {
        return UINT32_MAX;
    }

    NervaMemoryBlock *m = &e->memory[e->memory_count++];
    memset(m, 0, sizeof(*m));
    m->id = e->memory_count - 1u;
    m->type = NERVA_MEM_TYPE_EPISODIC;
    m->state = NERVA_MEM_STATE_PENDING | NERVA_MEM_STATE_OPEN;
    m->created_tick = e->tick;
    m->last_access_tick = e->tick;
    m->low_charge_since = NERVA_MEM_LOW_CHARGE_UNSET;
    m->query_tag = query_tag;
    return m->id;
}

void nerva_memory_end_episode(NervaEngine *e, uint32_t mem_id) {
    if (!e || mem_id >= e->memory_count) {
        return;
    }

    NervaMemoryBlock *m = &e->memory[mem_id];
    m->state &= (uint8_t)~NERVA_MEM_STATE_OPEN;
    m->trace_count = nerva_memory_count_traces_for_query(e, m->query_tag);
    m->last_access_tick = e->tick;
}

void nerva_memory_charge_update(NervaEngine *e, uint32_t mem_id, float useful, float surprise,
                                float repetition, float correction) {
    if (!e || mem_id >= e->memory_count) {
        return;
    }

    NervaMemoryBlock *m = &e->memory[mem_id];
    if (m->state & NERVA_MEM_STATE_MARK_DELETE) {
        return;
    }

    m->charge += useful + surprise + repetition + correction;
    m->last_access_tick = e->tick;
    if (useful > 0.0f || repetition > 0.0f) {
        m->flags |= NERVA_MEM_FLAG_USEFUL;
    }
    nerva_memory_refresh_state(e, m);
}

static void nerva_memory_decay_pass(NervaEngine *e) {
    for (uint32_t i = 0; i < e->memory_count; ++i) {
        NervaMemoryBlock *m = &e->memory[i];
        if (m->state & (NERVA_MEM_STATE_MARK_DELETE | NERVA_MEM_STATE_OPEN)) {
            continue;
        }
        m->charge -= e->cfg.memory_decay_per_idle * m->charge;
        if (m->charge < 0.0f) {
            m->charge = 0.0f;
        }
        nerva_memory_refresh_state(e, m);
    }
}

static void nerva_memory_forget_pass(NervaEngine *e) {
    for (uint32_t i = 0; i < e->memory_count; ++i) {
        NervaMemoryBlock *m = &e->memory[i];
        if (m->state & (NERVA_MEM_STATE_MARK_DELETE | NERVA_MEM_STATE_OPEN)) {
            continue;
        }
        if (m->charge >= e->cfg.memory_forget_threshold ||
            m->low_charge_since == NERVA_MEM_LOW_CHARGE_UNSET) {
            continue;
        }
        if (e->tick - m->low_charge_since < e->cfg.memory_hold_period_ticks) {
            continue;
        }
        m->state |= NERVA_MEM_STATE_MARK_DELETE;
        e->debug.memory_forgotten++;
    }
}

static void nerva_memory_replay_top_k(NervaEngine *e, uint16_t k) {
    if (!e || k == 0) {
        return;
    }

    uint32_t picked[8];
    uint32_t picked_count = 0;
    if (k > (uint16_t)(sizeof(picked) / sizeof(picked[0]))) {
        k = (uint16_t)(sizeof(picked) / sizeof(picked[0]));
    }

    for (uint16_t pass = 0; pass < k; ++pass) {
        uint32_t best = UINT32_MAX;
        float best_charge = -1.0f;
        for (uint32_t i = 0; i < e->memory_count; ++i) {
            const NervaMemoryBlock *m = &e->memory[i];
            if (m->state & (NERVA_MEM_STATE_MARK_DELETE | NERVA_MEM_STATE_OPEN)) {
                continue;
            }

            int already = 0;
            for (uint32_t j = 0; j < picked_count; ++j) {
                if (picked[j] == i) {
                    already = 1;
                    break;
                }
            }
            if (already) {
                continue;
            }

            if (m->charge > best_charge) {
                best_charge = m->charge;
                best = i;
            }
        }
        if (best == UINT32_MAX || best_charge <= 0.0f) {
            break;
        }

        picked[picked_count++] = best;
        NervaMemoryBlock *m = &e->memory[best];
        m->flags |= NERVA_MEM_FLAG_REPLAYED;
        m->last_access_tick = e->tick;
        e->debug.memory_replayed++;
    }
}

void nerva_consolidate_idle(NervaEngine *e) {
    if (!e) {
        return;
    }
    if (nerva_memory_has_open_episode(e)) {
        e->idle_ticks = 0;
        return;
    }

    nerva_apply_mutations(e);
    nerva_memory_decay_pass(e);
    nerva_memory_replay_top_k(e, e->cfg.memory_replay_top_k);
    nerva_memory_forget_pass(e);
    e->debug.memory_consolidations++;
    e->idle_ticks = 0;
}

void nerva_memory_on_tick_end(NervaEngine *e) {
    if (!e) {
        return;
    }

    if (e->event_count > 0) {
        e->idle_ticks = 0;
        return;
    }

    e->idle_ticks++;
    if (e->idle_ticks >= e->cfg.idle_consolidate_ticks) {
        nerva_consolidate_idle(e);
    }
}

int nerva_memory_is_consolidated(const NervaEngine *e, uint32_t mem_id) {
    if (!e || mem_id >= e->memory_count) {
        return 0;
    }
    return (e->memory[mem_id].state & NERVA_MEM_STATE_CONSOLIDATED) != 0;
}

int nerva_memory_is_marked_delete(const NervaEngine *e, uint32_t mem_id) {
    if (!e || mem_id >= e->memory_count) {
        return 0;
    }
    return (e->memory[mem_id].state & NERVA_MEM_STATE_MARK_DELETE) != 0;
}

int nerva_memory_is_episode_open(const NervaEngine *e, uint32_t mem_id) {
    if (!e || mem_id >= e->memory_count) {
        return 0;
    }
    return (e->memory[mem_id].state & NERVA_MEM_STATE_OPEN) != 0;
}

const NervaMemoryBlock *nerva_memory_get(const NervaEngine *e, uint32_t mem_id) {
    if (!e || mem_id >= e->memory_count) {
        return NULL;
    }
    return &e->memory[mem_id];
}

uint32_t nerva_memory_count(const NervaEngine *e) {
    return e ? e->memory_count : 0u;
}

void nerva_memory_print_blocks(const NervaEngine *e, FILE *out) {
    if (!e || !out) {
        return;
    }

    for (uint32_t i = 0; i < e->memory_count; ++i) {
        const NervaMemoryBlock *m = &e->memory[i];
        if (m->low_charge_since == NERVA_MEM_LOW_CHARGE_UNSET) {
            fprintf(out,
                    "id=%u query=%u charge=%.3f state=%u flags=%u traces=%u low_since=unset\n",
                    m->id, m->query_tag, m->charge, (unsigned)m->state, (unsigned)m->flags,
                    m->trace_count);
        } else {
            fprintf(out,
                    "id=%u query=%u charge=%.3f state=%u flags=%u traces=%u low_since=%llu\n",
                    m->id, m->query_tag, m->charge, (unsigned)m->state, (unsigned)m->flags,
                    m->trace_count, (unsigned long long)m->low_charge_since);
        }
    }
}

int nerva_memory_save_log(const NervaEngine *e, const char *path) {
    if (!e || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        return -1;
    }

    nerva_memory_print_blocks(e, out);
    fclose(out);
    return 0;
}
