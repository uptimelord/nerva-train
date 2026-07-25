// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_learning.h"
#include "nerva_mutation.h"
#include "nerva_trace.h"
#include "nerva_math.h"

#include <string.h>

bool nerva_queue_weight_delta(NervaEngine *e, uint32_t edge_id, nerva_q8_8_t delta,
                              uint32_t reason, uint32_t trace_tag) {
    if (!e || edge_id >= e->edge_count) {
        return false;
    }

    NervaMutation m;
    memset(&m, 0, sizeof(m));
    m.tick = e->tick;
    m.type = NERVA_MUT_MODIFY_WEIGHT;
    m.edge_id = edge_id;
    m.delta_weight = delta;
    m.new_gate = NERVA_GATE_UNCHANGED;
    m.trace_tag = trace_tag;
    m.debug_reason = reason;
    return nerva_mutation_queue_push(e, m);
}

bool nerva_queue_gate_close(NervaEngine *e, uint32_t edge_id, uint32_t reason,
                            uint32_t trace_tag) {
    if (!e || edge_id >= e->edge_count) {
        return false;
    }

    NervaEdge *ed = &e->edges[edge_id];
    uint32_t step = e->cfg.gate_close_step_q0_16;
    nerva_uq0_16_t new_gate = 0;
    if ((uint32_t)ed->gate > step) {
        new_gate = (nerva_uq0_16_t)((uint32_t)ed->gate - step);
    }

    NervaMutation m;
    memset(&m, 0, sizeof(m));
    m.tick = e->tick;
    m.type = NERVA_MUT_MODIFY_GATE;
    m.edge_id = edge_id;
    m.new_gate = new_gate;
    m.trace_tag = trace_tag;
    m.debug_reason = reason;
    return nerva_mutation_queue_push(e, m);
}

bool nerva_queue_create_edge(NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation,
                             uint32_t reason) {
    if (!e || source >= e->node_count || target >= e->node_count || relation == NERVA_REL_NONE) {
        return false;
    }

    NervaMutation m;
    memset(&m, 0, sizeof(m));
    m.tick = e->tick;
    m.type = NERVA_MUT_CREATE_EDGE;
    m.source = source;
    m.target = target;
    m.relation = relation;
    m.edge_flags = 0;
    m.new_gate = NERVA_GATE_UNCHANGED;
    m.debug_reason = reason;
    return nerva_mutation_queue_push(e, m);
}

bool nerva_hebbian_cofire(NervaEngine *e, uint32_t src, uint32_t dst, uint16_t relation,
                          nerva_q8_8_t delta, uint32_t reason) {
    if (!e || src >= e->node_count || dst >= e->node_count || relation == NERVA_REL_NONE) {
        return false;
    }

    const NervaNode *s = &e->nodes[src];
    const NervaNode *d = &e->nodes[dst];
    /* Co-activity: src has fired within the trace window and dst currently holds charge. */
    if (s->activation_count == 0u || d->v <= 0) {
        return false;
    }
    if ((e->tick - s->last_fired_tick) > (nerva_tick_t)e->cfg.trace_window_ticks) {
        return false;
    }

    uint32_t edge_id = NERVA_INVALID_ID;
    for (uint32_t i = 0; i < e->edge_count; ++i) {
        const NervaEdge *ed = &e->edges[i];
        if (!(ed->flags & NERVA_EDGE_DELETED) && ed->source == src && ed->target == dst &&
            ed->relation == relation) {
            edge_id = i;
            break;
        }
    }

    if (edge_id == NERVA_INVALID_ID) {
        return nerva_queue_create_edge(e, src, dst, relation, reason);
    }
    return nerva_queue_weight_delta(e, edge_id, delta, reason, 0);
}

static int nerva_feedback_seen_edge(uint32_t edge_id, const uint32_t *seen, uint32_t seen_count) {
    for (uint32_t i = 0; i < seen_count; ++i) {
        if (seen[i] == edge_id) {
            return 1;
        }
    }
    return 0;
}

static uint32_t nerva_feedback_process(NervaEngine *e, int wrong, uint32_t trace_tag_filter) {
    if (!e) {
        return 0;
    }

    uint32_t limit = e->trace_count;
    if (limit > e->cfg.trace_decay_scan_limit) {
        limit = e->cfg.trace_decay_scan_limit;
    }

    uint32_t seen[128];
    uint32_t seen_count = 0;
    uint32_t queued = 0;

    for (uint32_t age = limit; age > 0; --age) {
        NervaTrace *t = nerva_trace_recent(e, age - 1);
        if (!t) {
            continue;
        }
        if (!(t->flags & NERVA_TRACE_USED_PATH) || (t->flags & NERVA_TRACE_DECAYED)) {
            continue;
        }
        if (trace_tag_filter != 0 && t->trace_tag != trace_tag_filter) {
            continue;
        }
        if (e->active_query_tag != 0 && t->query_tag != e->active_query_tag) {
            continue;
        }
        if (t->edge_id == NERVA_INVALID_ID || t->edge_id >= e->edge_count) {
            continue;
        }
        if (nerva_feedback_seen_edge(t->edge_id, seen, seen_count)) {
            continue;
        }
        if (seen_count < (sizeof(seen) / sizeof(seen[0]))) {
            seen[seen_count++] = t->edge_id;
        }

        if (wrong) {
            if (nerva_queue_weight_delta(e, t->edge_id, e->cfg.ltd_delta_q8_8,
                                         NERVA_REASON_FEEDBACK_WRONG, t->trace_tag)) {
                queued++;
            }
            t->flags |= NERVA_TRACE_WRONG;
            {
                NervaEdge *ed = &e->edges[t->edge_id];
                uint32_t projected = (uint32_t)ed->wrong_feedback_count + 1u;
                if (projected >= e->cfg.feedback_wrong_gate_threshold) {
                    nerva_queue_gate_close(e, t->edge_id, NERVA_REASON_FEEDBACK_WRONG, t->trace_tag);
                }
            }
        } else {
            if (nerva_queue_weight_delta(e, t->edge_id, e->cfg.ltp_delta_q8_8,
                                         NERVA_REASON_FEEDBACK_CORRECT, t->trace_tag)) {
                queued++;
            }
            t->flags |= NERVA_TRACE_CORRECT;
        }
    }

    return queued;
}

uint32_t nerva_feedback_correct_tag(NervaEngine *e, uint32_t trace_tag) {
    return nerva_feedback_process(e, 0, trace_tag);
}

uint32_t nerva_feedback_wrong_tag(NervaEngine *e, uint32_t trace_tag) {
    return nerva_feedback_process(e, 1, trace_tag);
}

uint32_t nerva_feedback_correct(NervaEngine *e) {
    return nerva_feedback_correct_tag(e, 0);
}

uint32_t nerva_feedback_wrong(NervaEngine *e) {
    return nerva_feedback_wrong_tag(e, 0);
}
