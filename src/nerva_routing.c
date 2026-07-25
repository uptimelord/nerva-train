// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_routing.h"
#include "nerva_graph.h"
#include "nerva_schema.h"
#include "nerva_trace.h"
#include "nerva_math.h"

#include <string.h>

static int nerva_routing_query_has_exception(const NervaEngine *e) {
    if (!e || e->active_query_tag == 0) {
        return 0;
    }

    uint32_t n = e->trace_count < e->trace_cap ? e->trace_count : e->trace_cap;
    if (n > e->cfg.trace_decay_scan_limit) {
        n = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = 0; age < n; ++age) {
        NervaTrace *t = nerva_trace_recent((NervaEngine *)e, age);
        if (!t || t->query_tag != e->active_query_tag) {
            continue;
        }
        if ((t->flags & (NERVA_TRACE_EXCEPTION | NERVA_TRACE_BLOCKER)) ==
            (uint16_t)(NERVA_TRACE_EXCEPTION | NERVA_TRACE_BLOCKER)) {
            return 1;
        }
    }
    return 0;
}

static void nerva_routing_refresh_query_signals(NervaEngine *e) {
    if (!e || !e->router.query_active) {
        return;
    }

    if (nerva_routing_query_has_exception(e)) {
        e->router.contradiction_count = 1;
        e->router.unresolved_constraints = 1;
        e->router.confidence_q8_8 = (nerva_q8_8_t)(NERVA_Q8_8_ONE / 4);
    }
}

nerva_q8_8_t nerva_routing_compute_difficulty(const NervaEngine *e) {
    if (!e) {
        return 0;
    }

    int32_t d = 0;
    d += (int32_t)e->cfg.diff_novelty_w_q8_8 * (int32_t)e->router.novelty_count;
    d += (int32_t)e->cfg.diff_contradiction_w_q8_8 * (int32_t)e->router.contradiction_count;
    d += (int32_t)e->cfg.diff_unresolved_w_q8_8 * (int32_t)e->router.unresolved_constraints;
    d -= ((int32_t)e->cfg.diff_confidence_w_q8_8 * (int32_t)e->router.confidence_q8_8) >> 8;
    return nerva_q8_8_clip(d);
}

static void nerva_routing_decay_threshold(NervaEngine *e) {
    int32_t decay =
        ((int32_t)e->router.fluid_threshold_q8_8 - (int32_t)e->cfg.fluid_threshold_base_q8_8) >>
        (int32_t)e->cfg.fatigue_decay_shift;
    e->router.fluid_threshold_q8_8 =
        nerva_q8_8_clip((int32_t)e->router.fluid_threshold_q8_8 - decay);
}

static void nerva_routing_admit_top_active(NervaEngine *e) {
    e->router.fluid_count = 0;

    for (uint32_t pass = 0; pass < NERVA_FLUID_ACTIVE_MAX; ++pass) {
        uint32_t best = UINT32_MAX;
        nerva_q8_8_t best_v = 0;
        for (uint32_t i = 0; i < e->active_count; ++i) {
            uint32_t node_id = e->active_nodes[i];
            if (node_id >= e->node_count) {
                continue;
            }

            int already = 0;
            for (uint32_t j = 0; j < e->router.fluid_count; ++j) {
                if (e->router.fluid_nodes[j] == node_id) {
                    already = 1;
                    break;
                }
            }
            if (already) {
                continue;
            }

            nerva_q8_8_t v = e->nodes[node_id].v;
            if (best == UINT32_MAX || v > best_v) {
                best = node_id;
                best_v = v;
            }
        }
        if (best == UINT32_MAX) {
            break;
        }
        e->router.fluid_nodes[e->router.fluid_count++] = best;
    }
}

static void nerva_routing_apply_lateral_inhibition(NervaEngine *e) {
    if (!e || e->router.fluid_count == 0) {
        return;
    }

    uint32_t winner = e->router.fluid_nodes[0];
    nerva_q8_8_t winner_v = e->nodes[winner].v;
    for (uint32_t i = 1; i < e->router.fluid_count; ++i) {
        uint32_t node_id = e->router.fluid_nodes[i];
        nerva_q8_8_t v = e->nodes[node_id].v;
        if (v > winner_v) {
            winner = node_id;
            winner_v = v;
        }
    }

    for (uint32_t i = 0; i < e->router.fluid_count; ++i) {
        uint32_t node_id = e->router.fluid_nodes[i];
        if (node_id == winner) {
            continue;
        }
        NervaNode *n = &e->nodes[node_id];
        if (n->v < e->cfg.theta_compete_q8_8) {
            continue;
        }
        int32_t v = (int32_t)n->v;
        v = (v * (65535 - (int32_t)e->cfg.ambiguity_inhibition_q0_16)) >> 16;
        n->v = nerva_q8_8_clip(v);
    }
}

static int nerva_routing_path_reachable(NervaEngine *e, uint32_t source, uint32_t target,
                                        uint16_t relation) {
    if (!e || relation == NERVA_REL_NONE || source >= e->node_count ||
        target >= e->node_count) {
        return 0;
    }
    if (!e->adjacency_valid && e->edge_count > 0) {
        nerva_graph_rebuild_adjacency(e);
    }
    return nerva_graph_reachable(e, source, target, relation);
}

static int nerva_routing_try_alternate_path(NervaEngine *e) {
    if (!e || !e->router.query_active || e->router.query_relation == NERVA_REL_NONE) {
        return 0;
    }

    if (nerva_routing_path_reachable(e, e->router.query_source, e->router.query_target,
                                     e->router.query_relation)) {
        return 0;
    }

    if (e->router.query_relation == NERVA_REL_KIND_OF &&
        nerva_schema_find_promoted(e, NERVA_REL_KIND_OF, NERVA_REL_KIND_OF) != NULL) {
        return 1;
    }
    return 0;
}

static void nerva_routing_fluid_workspace_step(NervaEngine *e) {
    nerva_routing_admit_top_active(e);
    nerva_routing_apply_lateral_inhibition(e);
    (void)nerva_routing_try_alternate_path(e);
    e->debug.fluid_workspace_steps++;
}

void nerva_routing_reset(NervaEngine *e) {
    if (!e) {
        return;
    }
    memset(&e->router, 0, sizeof(e->router));
    e->router.fluid_threshold_q8_8 = e->cfg.fluid_threshold_base_q8_8;
}

void nerva_routing_begin_query(NervaEngine *e, uint32_t source, uint32_t target,
                               uint16_t relation) {
    if (!e || source >= e->node_count || target >= e->node_count) {
        return;
    }

    e->router.query_active = 1;
    e->router.query_routing_decided = 0;
    e->router.fluid_active = 0;
    e->router.crystallized = 0;
    e->router.query_source = source;
    e->router.query_target = target;
    e->router.query_relation = relation;
    e->router.novelty_count = 0;
    e->router.contradiction_count = 0;
    e->router.unresolved_constraints = 0;
    e->router.fluid_count = 0;

    if (relation == NERVA_REL_NONE ||
        !nerva_routing_path_reachable(e, source, target, relation)) {
        e->router.novelty_count = 1;
        e->router.unresolved_constraints = 1;
        e->router.confidence_q8_8 = 0;
    } else {
        e->router.confidence_q8_8 = (nerva_q8_8_t)(NERVA_Q8_8_ONE * 2);
    }

    e->router.difficulty_q8_8 = nerva_routing_compute_difficulty(e);
}

void nerva_routing_on_tick_end(NervaEngine *e) {
    if (!e || !e->router.query_active || e->router.query_routing_decided) {
        return;
    }
    if (e->event_count > 0) {
        return;
    }

    nerva_routing_refresh_query_signals(e);
    e->router.difficulty_q8_8 = nerva_routing_compute_difficulty(e);
    nerva_routing_decay_threshold(e);

    if (e->router.difficulty_q8_8 > e->router.fluid_threshold_q8_8) {
        e->router.fluid_active = 1;
        e->router.fluid_threshold_q8_8 = nerva_q8_8_saturating_add(
            e->router.fluid_threshold_q8_8, e->cfg.fatigue_increment_q8_8);
        e->debug.fluid_activations++;
        nerva_routing_fluid_workspace_step(e);
    } else {
        e->router.crystallized = 1;
        e->debug.crystallized_queries++;
    }

    e->router.query_routing_decided = 1;
}

int nerva_routing_was_crystallized_last_query(const NervaEngine *e) {
    if (!e) {
        return 0;
    }
    return e->router.crystallized != 0;
}

int nerva_routing_fluid_active(const NervaEngine *e) {
    if (!e) {
        return 0;
    }
    return e->router.fluid_active != 0;
}

void nerva_routing_print_snapshot(const NervaEngine *e, FILE *out, const char *label) {
    if (!e || !out) {
        return;
    }

    fprintf(out,
            "case=%s difficulty=%d threshold=%d novelty=%u contradiction=%u "
            "crystallized=%u fluid=%u fluid_nodes=%u\n",
            label ? label : "query", (int)e->router.difficulty_q8_8,
            (int)e->router.fluid_threshold_q8_8, (unsigned)e->router.novelty_count,
            (unsigned)e->router.contradiction_count, (unsigned)e->router.crystallized,
            (unsigned)e->router.fluid_active, (unsigned)e->router.fluid_count);
}

void nerva_routing_print_state(const NervaEngine *e, FILE *out) {
    nerva_routing_print_snapshot(e, out, "query");
}

int nerva_routing_save_log(const NervaEngine *e, const char *path) {
    if (!e || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        return -1;
    }

    nerva_routing_print_state(e, out);
    fclose(out);
    return 0;
}
