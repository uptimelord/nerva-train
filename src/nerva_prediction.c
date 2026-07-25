// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_prediction.h"
#include "nerva_graph.h"
#include "nerva_learning.h"
#include "nerva_trace.h"
#include "nerva_event.h"
#include "nerva_math.h"

#include <stdio.h>
#include <string.h>

static void nerva_prediction_mark_trace(NervaEngine *e, uint32_t edge_id, uint16_t add_flags);

void nerva_set_prediction_mode(NervaEngine *e, int enabled) {
    if (!e) {
        return;
    }
    e->prediction_mode = enabled ? 1 : 0;
}

void nerva_prediction_clear(NervaEngine *e) {
    if (!e) {
        return;
    }
    e->expectation_count = 0;
}

static void nerva_prediction_resolve_miss(NervaEngine *e, NervaExpectation *exp) {
    nerva_queue_weight_delta(e, exp->edge_id, e->cfg.ltd_delta_q8_8,
                             NERVA_REASON_PREDICTION_MISSED, exp->trace_tag);
    nerva_prediction_mark_trace(e, exp->edge_id,
                                (uint16_t)(NERVA_TRACE_PRED_MISSED | NERVA_TRACE_SURPRISE));
    exp->flags &= (uint16_t)~NERVA_EXP_PENDING;
    e->debug.predictions_missed++;
}

void nerva_prediction_expire_stale(NervaEngine *e) {
    if (!e || e->cfg.prediction_window_ticks == 0) {
        return;
    }

    for (uint32_t i = 0; i < e->expectation_count; ++i) {
        NervaExpectation *exp = &e->expectations[i];
        if (!(exp->flags & NERVA_EXP_PENDING)) {
            continue;
        }
        if ((e->tick - exp->tick) > e->cfg.prediction_window_ticks) {
            nerva_prediction_resolve_miss(e, exp);
        }
    }
}

static uint32_t nerva_prediction_select_edge(const NervaEngine *e, uint32_t source_node_id) {
    if (!e || source_node_id >= e->node_count || !e->adjacency_valid) {
        return NERVA_INVALID_ID;
    }

    const NervaNode *n = &e->nodes[source_node_id];
    uint32_t best_edge = NERVA_INVALID_ID;
    uint32_t best_score = 0;

    for (uint32_t k = 0; k < n->out_count; ++k) {
        uint32_t slot = n->first_out + k;
        if (slot >= e->edge_count) {
            break;
        }

        uint32_t edge_id = e->sorted_edges[slot];
        const NervaEdge *ed = &e->edges[edge_id];
        if (ed->flags & NERVA_EDGE_DELETED) {
            continue;
        }
        if (ed->stability < e->cfg.prediction_min_stability) {
            continue;
        }

        uint32_t score = (uint32_t)ed->stability;
        int32_t w = nerva_abs32((int32_t)ed->weight);
        score += (uint32_t)(w >> 4);
        if (score >= best_score) {
            best_score = score;
            best_edge = edge_id;
        }
    }

    return best_edge;
}

static void nerva_prediction_precharge(NervaEngine *e, NervaNode *n) {
    nerva_q8_8_t cap = (nerva_q8_8_t)(n->theta_fire - 1);
    if (cap < 0) {
        cap = 0;
    }
    nerva_q8_8_t next = nerva_q8_8_saturating_add(n->v, e->cfg.prediction_pre_charge_q8_8);
    if (next > cap) {
        next = cap;
    }
    n->v = next;
}

static void nerva_prediction_record_expected(NervaEngine *e, uint32_t edge_id, const NervaEdge *ed,
                                             nerva_q8_8_t v_before, nerva_q8_8_t v_after) {
    NervaEvent ev;
    memset(&ev, 0, sizeof(ev));
    ev.due_tick = e->tick;
    ev.source = ed->source;
    ev.target = ed->target;
    ev.edge_id = edge_id;
    ev.signal = 0;
    ev.relation = ed->relation;
    ev.trace_tag = ed->trace_tag;
    ev.type_flags = NERVA_EVT_EXPECTED;
    nerva_trace_record(e, &ev, NERVA_TRACE_EXPECTED, v_before, v_after, 0);
}

void nerva_prediction_on_fire(NervaEngine *e, uint32_t source_node_id) {
    if (!e || !e->prediction_mode) {
        return;
    }

    uint32_t edge_id = nerva_prediction_select_edge(e, source_node_id);
    if (edge_id == NERVA_INVALID_ID || edge_id >= e->edge_count) {
        return;
    }

    NervaEdge *ed = &e->edges[edge_id];
    if (ed->target >= e->node_count) {
        return;
    }

    if (e->expectation_count >= e->expectation_cap) {
        return;
    }

    NervaNode *target = &e->nodes[ed->target];
    nerva_q8_8_t v_before = target->v;
    nerva_prediction_precharge(e, target);
    nerva_mark_active(e, ed->target);

    if (ed->trace_tag == 0) {
        ed->trace_tag = nerva_make_path_tag(e, edge_id);
    }

    NervaExpectation *exp = &e->expectations[e->expectation_count++];
    exp->tick = e->tick;
    exp->source = ed->source;
    exp->target = ed->target;
    exp->edge_id = edge_id;
    exp->query_tag = e->active_query_tag;
    exp->trace_tag = ed->trace_tag;
    exp->relation = ed->relation;
    exp->flags = NERVA_EXP_PENDING;

    nerva_prediction_record_expected(e, edge_id, ed, v_before, target->v);
    e->debug.predictions_emitted++;
}

static NervaExpectation *nerva_prediction_find_pending(NervaEngine *e, uint32_t query_tag) {
    for (uint32_t i = 0; i < e->expectation_count; ++i) {
        NervaExpectation *exp = &e->expectations[i];
        if (!(exp->flags & NERVA_EXP_PENDING)) {
            continue;
        }
        if (query_tag != 0 && exp->query_tag != query_tag) {
            continue;
        }
        return exp;
    }
    return NULL;
}

static void nerva_prediction_mark_trace(NervaEngine *e, uint32_t edge_id, uint16_t add_flags) {
    uint32_t limit = e->trace_count;
    if (limit > e->cfg.trace_decay_scan_limit) {
        limit = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = 0; age < limit; ++age) {
        NervaTrace *t = nerva_trace_recent(e, age);
        if (!t || t->edge_id != edge_id) {
            continue;
        }
        if (!(t->flags & NERVA_TRACE_EXPECTED)) {
            continue;
        }
        if (e->active_query_tag != 0 && t->query_tag != e->active_query_tag) {
            continue;
        }
        t->flags |= add_flags;
        return;
    }
}

void nerva_prediction_on_actual(NervaEngine *e, const NervaEvent *ev) {
    if (!e || !ev || ev->edge_id == NERVA_INVALID_ID) {
        return;
    }

    /* Single-hop scope: match the first pending expectation for this query. */
    NervaExpectation *exp = nerva_prediction_find_pending(e, e->active_query_tag);
    if (!exp) {
        exp = nerva_prediction_find_pending(e, 0);
    }
    if (!exp) {
        return;
    }

    if (ev->edge_id == exp->edge_id && ev->target == exp->target) {
        nerva_queue_weight_delta(e, exp->edge_id, e->cfg.ltp_delta_q8_8,
                                 NERVA_REASON_PREDICTION_CONFIRMED, exp->trace_tag);
        nerva_prediction_mark_trace(e, exp->edge_id, NERVA_TRACE_PRED_CONFIRMED);
        exp->flags &= (uint16_t)~NERVA_EXP_PENDING;
        e->debug.predictions_confirmed++;
        return;
    }

    nerva_prediction_resolve_miss(e, exp);
}

bool nerva_inject_edge_event(NervaEngine *e, uint32_t edge_id, nerva_q8_8_t signal) {
    if (!e || edge_id >= e->edge_count) {
        return false;
    }

    const NervaEdge *ed = &e->edges[edge_id];
    if (ed->flags & NERVA_EDGE_DELETED) {
        return false;
    }

    NervaEvent ev;
    memset(&ev, 0, sizeof(ev));
    ev.due_tick = e->tick;
    ev.source = ed->source;
    ev.target = ed->target;
    ev.edge_id = edge_id;
    ev.signal = signal;
    ev.relation = ed->relation;
    ev.trace_tag = ed->trace_tag;
    ev.type_flags = NERVA_EVT_ACTUAL;
    return nerva_event_push(e, ev);
}

const NervaExpectation *nerva_prediction_pending_for_query(const NervaEngine *e, uint32_t query_tag) {
    if (!e) {
        return NULL;
    }
    for (uint32_t i = 0; i < e->expectation_count; ++i) {
        const NervaExpectation *exp = &e->expectations[i];
        if (!(exp->flags & NERVA_EXP_PENDING)) {
            continue;
        }
        if (query_tag != 0 && exp->query_tag != query_tag) {
            continue;
        }
        return exp;
    }
    return NULL;
}

uint32_t nerva_prediction_count_pending(const NervaEngine *e) {
    if (!e) {
        return 0;
    }

    uint32_t count = 0;
    for (uint32_t i = 0; i < e->expectation_count; ++i) {
        if (e->expectations[i].flags & NERVA_EXP_PENDING) {
            count++;
        }
    }
    return count;
}

void nerva_prediction_print_trace(const NervaEngine *e, FILE *out, uint32_t limit) {
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
        if (!(t->flags & (NERVA_TRACE_EXPECTED | NERVA_TRACE_PRED_CONFIRMED | NERVA_TRACE_PRED_MISSED |
                          NERVA_TRACE_SURPRISE))) {
            continue;
        }
        const char *kind = "expected";
        if (t->flags & NERVA_TRACE_PRED_CONFIRMED) {
            kind = "confirmed";
        } else if (t->flags & NERVA_TRACE_PRED_MISSED) {
            kind = "missed";
        }
        fprintf(out,
                "tick=%llu %s edge=%u source=%u target=%u flags=0x%04x v_before=%d v_after=%d\n",
                (unsigned long long)t->tick, kind, t->edge_id, t->source, t->target,
                (unsigned)t->flags, (int)t->target_v_before, (int)t->target_v_after);
    }
}

int nerva_prediction_save_trace(const NervaEngine *e, const char *path, uint32_t limit) {
    if (!e || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        return -1;
    }

    nerva_prediction_print_trace(e, out, limit);
    fclose(out);
    return 0;
}
