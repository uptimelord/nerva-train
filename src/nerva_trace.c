// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_trace.h"
#include "nerva_graph.h"
#include "nerva_math.h"

#include <stdio.h>
#include <string.h>

uint32_t nerva_make_path_tag(const NervaEngine *e, uint32_t edge_id) {
    if (!e) {
        return 0;
    }
    return (uint32_t)((e->tick << 16) ^ (edge_id + 1u));
}

void nerva_trace_record(NervaEngine *e, const NervaEvent *ev, uint16_t flags,
                        nerva_q8_8_t target_v_before, nerva_q8_8_t target_v_after,
                        int fired_after_apply) {
    if (!e || !ev || e->trace_cap == 0) {
        return;
    }

    NervaTrace *t = &e->traces[e->trace_head];
    memset(t, 0, sizeof(*t));
    t->tick = e->tick;
    t->source = ev->source;
    t->target = ev->target;
    t->edge_id = ev->edge_id;
    t->trace_tag = ev->trace_tag;
    t->query_tag = e->active_query_tag;
    t->signal = ev->signal;
    t->pre = NERVA_UQ0_16_ONE;
    t->post = NERVA_UQ0_16_ONE;
    t->relation = ev->relation;
    t->flags = flags;
    t->target_v_before = target_v_before;
    t->target_v_after = target_v_after;
    t->fired_after_apply = (uint16_t)(fired_after_apply ? 1 : 0);

    if (ev->target < e->node_count) {
        e->nodes[ev->target].trace_head = e->trace_head;
    }

    e->trace_head = (e->trace_head + 1) % e->trace_cap;
    if (e->trace_count < e->trace_cap) {
        e->trace_count++;
    }
    e->debug.traces_recorded++;
}

static NervaTrace *nerva_trace_at_age(NervaEngine *e, uint32_t age) {
    if (!e || age >= e->trace_count || e->trace_count == 0) {
        return NULL;
    }
    uint32_t idx = (e->trace_head + e->trace_cap - 1u - age) % e->trace_cap;
    return &e->traces[idx];
}

NervaTrace *nerva_trace_recent(NervaEngine *e, uint32_t age) {
    return nerva_trace_at_age(e, age);
}

void nerva_trace_decay(NervaEngine *e) {
    if (!e) {
        return;
    }

    uint32_t n = e->trace_count;
    if (n > e->cfg.trace_decay_scan_limit) {
        n = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = 0; age < n; ++age) {
        NervaTrace *t = nerva_trace_at_age(e, age);
        if (!t) {
            continue;
        }
        t->pre = (nerva_uq0_16_t)(((uint32_t)t->pre * e->cfg.trace_pre_decay_q0_16) >> 16);
        t->post = (nerva_uq0_16_t)(((uint32_t)t->post * e->cfg.trace_post_decay_q0_16) >> 16);
        if (e->tick - t->tick > e->cfg.trace_window_ticks) {
            t->flags |= NERVA_TRACE_DECAYED;
        }
    }
}

uint32_t nerva_trace_count_used_path(const NervaEngine *e) {
    if (!e) {
        return 0;
    }

    uint32_t count = 0;
    uint32_t limit = e->trace_count;
    if (limit > e->cfg.trace_decay_scan_limit) {
        limit = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = 0; age < limit; ++age) {
        uint32_t idx = (e->trace_head + e->trace_cap - 1u - age) % e->trace_cap;
        const NervaTrace *t = &e->traces[idx];
        if ((t->flags & NERVA_TRACE_USED_PATH) && !(t->flags & NERVA_TRACE_DECAYED)) {
            count++;
        }
    }
    return count;
}

uint32_t nerva_trace_count_edge_flags(const NervaEngine *e, uint32_t edge_id, uint16_t flags) {
    if (!e) {
        return 0;
    }

    uint32_t count = 0;
    uint32_t limit = e->trace_count;
    if (limit > e->cfg.trace_decay_scan_limit) {
        limit = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = 0; age < limit; ++age) {
        uint32_t idx = (e->trace_head + e->trace_cap - 1u - age) % e->trace_cap;
        const NervaTrace *t = &e->traces[idx];
        if (t->edge_id != edge_id) {
            continue;
        }
        if ((t->flags & flags) == flags) {
            count++;
        }
    }
    return count;
}

int nerva_trace_find_edge_in_recent(const NervaEngine *e, uint32_t edge_id,
                                    uint16_t required_flags, uint32_t trace_tag_filter) {
    if (!e) {
        return 0;
    }

    uint32_t limit = e->trace_count;
    if (limit > e->cfg.trace_decay_scan_limit) {
        limit = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = 0; age < limit; ++age) {
        uint32_t idx = (e->trace_head + e->trace_cap - 1u - age) % e->trace_cap;
        const NervaTrace *t = &e->traces[idx];
        if (t->edge_id != edge_id) {
            continue;
        }
        if (trace_tag_filter != 0 && t->trace_tag != trace_tag_filter) {
            continue;
        }
        if ((t->flags & required_flags) != required_flags) {
            continue;
        }
        if (t->flags & NERVA_TRACE_DECAYED) {
            continue;
        }
        return 1;
    }
    return 0;
}

void nerva_trace_clear(NervaEngine *e) {
    if (!e) {
        return;
    }
    e->trace_head = 0;
    e->trace_count = 0;
}

static const char *nerva_trace_node_label(const NervaEngine *e, uint32_t node_id) {
    if (!e || node_id >= e->node_count) {
        return "?";
    }
    const char *name = nerva_name_by_id(e, e->nodes[node_id].name_id);
    return (name && name[0] != '\0') ? name : "?";
}

void nerva_trace_print_recent(const NervaEngine *e, FILE *out, uint32_t limit) {
    if (!e || !out) {
        return;
    }

    if (limit == 0 || limit > e->trace_count) {
        limit = e->trace_count;
    }
    if (limit > e->cfg.trace_decay_scan_limit) {
        limit = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = 0; age < limit; ++age) {
        uint32_t idx = (e->trace_head + e->trace_cap - 1u - age) % e->trace_cap;
        const NervaTrace *t = &e->traces[idx];
        fprintf(out,
                "tick=%llu edge=%u %s->%s relation=%u signal=%d v_before=%d v_after=%d "
                "fired=%u trace_tag=0x%08x flags=0x%04x pre=%u post=%u\n",
                (unsigned long long)t->tick, t->edge_id,
                nerva_trace_node_label(e, t->source), nerva_trace_node_label(e, t->target),
                (unsigned)t->relation, (int)t->signal, (int)t->target_v_before,
                (int)t->target_v_after, (unsigned)t->fired_after_apply, t->trace_tag,
                (unsigned)t->flags, (unsigned)t->pre, (unsigned)t->post);
    }
}

void nerva_trace_print_path(const NervaEngine *e, FILE *out, uint32_t limit) {
    if (!e || !out) {
        return;
    }

    if (limit == 0 || limit > e->trace_count) {
        limit = e->trace_count;
    }
    if (limit > e->cfg.trace_decay_scan_limit) {
        limit = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = limit; age > 0; --age) {
        uint32_t idx = (e->trace_head + e->trace_cap - age) % e->trace_cap;
        const NervaTrace *t = &e->traces[idx];
        fprintf(out,
                "tick=%llu edge=%u %s->%s relation=%u signal=%d v_before=%d v_after=%d "
                "fired=%u trace_tag=0x%08x flags=0x%04x pre=%u post=%u\n",
                (unsigned long long)t->tick, t->edge_id,
                nerva_trace_node_label(e, t->source), nerva_trace_node_label(e, t->target),
                (unsigned)t->relation, (int)t->signal, (int)t->target_v_before,
                (int)t->target_v_after, (unsigned)t->fired_after_apply, t->trace_tag,
                (unsigned)t->flags, (unsigned)t->pre, (unsigned)t->post);
    }
}

int nerva_trace_save_path(const NervaEngine *e, const char *path, uint32_t limit) {
    if (!e || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        return -1;
    }

    nerva_trace_print_path(e, out, limit);
    fclose(out);
    return 0;
}

int nerva_trace_save_recent(const NervaEngine *e, const char *path, uint32_t limit) {
    if (!e || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        return -1;
    }

    nerva_trace_print_recent(e, out, limit);
    fclose(out);
    return 0;
}
