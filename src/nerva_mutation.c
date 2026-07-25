// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_mutation.h"
#include "nerva_graph.h"
#include "nerva_math.h"

#include <stdio.h>
#include <string.h>

static void nerva_mutation_log_applied(NervaEngine *e, const NervaMutation *m,
                                       nerva_q8_8_t old_weight, nerva_q8_8_t new_weight,
                                       nerva_uq0_16_t old_gate, nerva_uq0_16_t new_gate) {
    if (!e || e->mutation_log_count >= e->mutation_log_cap) {
        return;
    }

    NervaMutationRecord *rec = &e->mutation_log[e->mutation_log_count++];
    rec->tick = m->tick;
    rec->type = m->type;
    rec->edge_id = m->edge_id;
    rec->old_weight = old_weight;
    rec->new_weight = new_weight;
    rec->old_gate = old_gate;
    rec->new_gate = new_gate;
    rec->debug_reason = m->debug_reason;
}

bool nerva_mutation_queue_push(NervaEngine *e, NervaMutation m) {
    if (!e || e->mutation_cap == 0) {
        return false;
    }
    if (e->mutation_count >= e->mutation_cap) {
        e->debug.mutations_overflow++;
        return false;
    }

    e->mutations[e->mutation_tail] = m;
    e->mutation_tail = (e->mutation_tail + 1) % e->mutation_cap;
    e->mutation_count++;
    e->debug.mutations_queued++;
    return true;
}

void nerva_apply_mutations(NervaEngine *e) {
    if (!e) {
        return;
    }

    int adjacency_dirty = 0;
    while (e->mutation_count > 0) {
        NervaMutation m = e->mutations[e->mutation_head];
        e->mutation_head = (e->mutation_head + 1) % e->mutation_cap;
        e->mutation_count--;

        nerva_q8_8_t old_weight = 0;
        nerva_q8_8_t new_weight = 0;
        nerva_uq0_16_t old_gate = 0;
        nerva_uq0_16_t new_gate = 0;

        switch (m.type) {
        case NERVA_MUT_MODIFY_WEIGHT:
            if (m.edge_id < e->edge_count) {
                old_weight = e->edges[m.edge_id].weight;
                old_gate = e->edges[m.edge_id].gate;
            }
            if (nerva_graph_modify_weight(e, m.edge_id, m.delta_weight, &old_weight, &new_weight)) {
                new_gate = e->edges[m.edge_id].gate;
                if (m.debug_reason == NERVA_REASON_FEEDBACK_WRONG) {
                    NervaEdge *ed = &e->edges[m.edge_id];
                    if (ed->wrong_feedback_count < UINT16_MAX) {
                        ed->wrong_feedback_count++;
                    }
                } else if (m.debug_reason == NERVA_REASON_FEEDBACK_CORRECT ||
                           m.debug_reason == NERVA_REASON_PREDICTION_CONFIRMED) {
                    e->edges[m.edge_id].wrong_feedback_count = 0;
                } else if (m.debug_reason == NERVA_REASON_PREDICTION_MISSED) {
                    NervaEdge *ed = &e->edges[m.edge_id];
                    if (ed->wrong_feedback_count < UINT16_MAX) {
                        ed->wrong_feedback_count++;
                    }
                }
                nerva_mutation_log_applied(e, &m, old_weight, new_weight, old_gate, new_gate);
            }
            break;
        case NERVA_MUT_MODIFY_GATE:
            if (m.edge_id < e->edge_count) {
                old_weight = e->edges[m.edge_id].weight;
                old_gate = e->edges[m.edge_id].gate;
            }
            if (nerva_graph_modify_gate(e, m.edge_id, m.new_gate, &old_gate, &new_gate)) {
                new_weight = e->edges[m.edge_id].weight;
                nerva_mutation_log_applied(e, &m, old_weight, new_weight, old_gate, new_gate);
            }
            break;
        case NERVA_MUT_CREATE_EDGE: {
            uint32_t edge_id = nerva_graph_create_edge(e, m.source, m.target, m.relation);
            if (edge_id != UINT32_MAX) {
                adjacency_dirty = 1;
                NervaMutation logged = m;
                logged.edge_id = edge_id;
                nerva_mutation_log_applied(e, &logged, 0, e->edges[edge_id].weight,
                                           (nerva_uq0_16_t)NERVA_UQ0_16_ONE,
                                           (nerva_uq0_16_t)NERVA_UQ0_16_ONE);
            }
            break;
        }
        case NERVA_MUT_ADD_BLOCKER_EDGE: {
            uint32_t edge_id =
                nerva_graph_create_blocker_edge(e, m.source, m.target, m.relation);
            if (edge_id != UINT32_MAX) {
                e->debug.blockers_applied++;
                adjacency_dirty = 1;
                NervaMutation logged = m;
                logged.edge_id = edge_id;
                nerva_mutation_log_applied(e, &logged, 0, e->edges[edge_id].weight,
                                           (nerva_uq0_16_t)NERVA_UQ0_16_ONE,
                                           (nerva_uq0_16_t)NERVA_UQ0_16_ONE);
            }
            break;
        }
        case NERVA_MUT_DELETE_EDGE:
            adjacency_dirty = 1;
            break;
        default:
            break;
        }

        e->debug.mutations_applied++;
    }

    if (adjacency_dirty) {
        nerva_graph_rebuild_adjacency(e);
    }
}

void nerva_mutation_clear_log(NervaEngine *e) {
    if (!e) {
        return;
    }
    e->mutation_log_count = 0;
}

void nerva_mutation_print_log(const NervaEngine *e, FILE *out) {
    if (!e || !out) {
        return;
    }

    for (uint32_t i = 0; i < e->mutation_log_count; ++i) {
        const NervaMutationRecord *rec = &e->mutation_log[i];
        fprintf(out,
                "tick=%llu type=%u edge=%u reason=%u weight %d->%d gate %u->%u\n",
                (unsigned long long)rec->tick, (unsigned)rec->type, rec->edge_id, rec->debug_reason,
                (int)rec->old_weight, (int)rec->new_weight, (unsigned)rec->old_gate,
                (unsigned)rec->new_gate);
    }
}

int nerva_mutation_save_log(const NervaEngine *e, const char *path) {
    if (!e || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        return -1;
    }

    nerva_mutation_print_log(e, out);
    fclose(out);
    return 0;
}
