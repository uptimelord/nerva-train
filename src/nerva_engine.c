// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_engine.h"
#include "nerva_event.h"
#include "nerva_trace.h"
#include "nerva_prediction.h"
#include "nerva_exception.h"
#include "nerva_memory.h"
#include "nerva_routing.h"
#include "nerva_math.h"

#include <stdlib.h>
#include <string.h>

int nerva_engine_init(NervaEngine *e, NervaConfig cfg) {
    if (!e || cfg.max_nodes == 0 || cfg.max_edges == 0 || cfg.max_names == 0 ||
        cfg.max_events == 0 || cfg.max_active_nodes == 0 || cfg.max_fire_log == 0 ||
        cfg.max_traces == 0 || cfg.max_mutations == 0 || cfg.max_expectations == 0 ||
        cfg.max_schemas == 0 || cfg.max_memory_blocks == 0) {
        return -1;
    }

    memset(e, 0, sizeof(*e));
    e->cfg = cfg;
    e->node_cap = cfg.max_nodes;
    e->edge_cap = cfg.max_edges;
    e->name_cap = cfg.max_names;
    e->event_cap = cfg.max_events;
    e->active_cap = cfg.max_active_nodes;
    e->fire_log_cap = cfg.max_fire_log;
    e->trace_cap = cfg.max_traces;
    e->mutation_cap = cfg.max_mutations;
    e->mutation_log_cap = cfg.max_mutation_log;
    e->expectation_cap = cfg.max_expectations;
    e->schema_cap = cfg.max_schemas;
    e->memory_cap = cfg.max_memory_blocks;

    e->nodes = calloc(e->node_cap, sizeof(NervaNode));
    e->edges = calloc(e->edge_cap, sizeof(NervaEdge));
    e->sorted_edges = calloc(e->edge_cap, sizeof(uint32_t));
    e->blocker_in_edges = calloc(e->edge_cap, sizeof(uint32_t));
    e->events = calloc(e->event_cap, sizeof(NervaEvent));
    e->active_nodes = calloc(e->active_cap, sizeof(uint32_t));
    e->fire_log = calloc(e->fire_log_cap, sizeof(NervaFireRecord));
    e->traces = calloc(e->trace_cap, sizeof(NervaTrace));
    e->mutations = calloc(e->mutation_cap, sizeof(NervaMutation));
    e->mutation_log = calloc(e->mutation_log_cap, sizeof(NervaMutationRecord));
    e->expectations = calloc(e->expectation_cap, sizeof(NervaExpectation));
    e->schemas = calloc(e->schema_cap, sizeof(NervaSchema));
    e->memory = calloc(e->memory_cap, sizeof(NervaMemoryBlock));
    e->names = calloc(e->name_cap, sizeof(char *));
    if (!e->nodes || !e->edges || !e->sorted_edges || !e->blocker_in_edges || !e->events ||
        !e->active_nodes || !e->fire_log || !e->traces || !e->mutations || !e->mutation_log ||
        !e->expectations || !e->schemas || !e->memory || !e->names) {
        nerva_engine_free(e);
        return -1;
    }

    e->adjacency_valid = 0;
    nerva_routing_reset(e);
    return 0;
}

void nerva_engine_free(NervaEngine *e) {
    if (!e) {
        return;
    }

    for (uint32_t i = 0; i < e->name_count; ++i) {
        free(e->names[i]);
    }

    free(e->names);
    free(e->fire_log);
    free(e->traces);
    free(e->mutations);
    free(e->mutation_log);
    free(e->expectations);
    free(e->schemas);
    free(e->memory);
    free(e->active_nodes);
    free(e->events);
    free(e->sorted_edges);
    free(e->blocker_in_edges);
    free(e->edges);
    free(e->nodes);
    memset(e, 0, sizeof(*e));
}

void nerva_tick(NervaEngine *e) {
    if (!e) {
        return;
    }

    e->debug.tick_events = 0;
    e->debug.tick_fired = 0;
    e->active_count = 0;

    nerva_prediction_expire_stale(e);

    NervaEvent ev;
    while (nerva_event_pop(e, &ev)) {
        e->debug.tick_events++;
        if (ev.target >= e->node_count) {
            continue;
        }

        NervaNode *n = &e->nodes[ev.target];
        nerva_q8_8_t v_before = n->v;
        nerva_apply_event_to_node(e, n, &ev);
        uint16_t suppress_flags = nerva_exception_apply_suppression(e, n, &ev);
        nerva_mark_active(e, ev.target);

        if (ev.edge_id != NERVA_INVALID_ID) {
            nerva_prediction_on_actual(e, &ev);
        }

        int should_fire = nerva_node_should_fire(e, n);
        if (ev.edge_id != NERVA_INVALID_ID) {
            uint16_t trace_flags = NERVA_TRACE_USED_PATH;
            if (e->edges[ev.edge_id].flags & NERVA_EDGE_BLOCKER) {
                trace_flags |= NERVA_TRACE_BLOCKER;
            }
            if (e->edges[ev.edge_id].gate_ctrl != NERVA_INVALID_ID) {
                trace_flags |= NERVA_TRACE_GATED;
            }
            trace_flags |= suppress_flags;
            nerva_trace_record(e, &ev, trace_flags, v_before, n->v, should_fire);
        }

        if (should_fire) {
            nerva_fire_node(e, ev.target);
        }
    }

    for (uint32_t k = 0; k < e->active_count; ++k) {
        uint32_t node_id = e->active_nodes[k];
        NervaNode *n = &e->nodes[node_id];
        nerva_apply_leak(e, n);
    }

    nerva_trace_decay(e);
    nerva_routing_on_tick_end(e);
    nerva_memory_on_tick_end(e);

    e->debug.total_fired += e->debug.tick_fired;
    e->tick++;
}

void nerva_tick_n(NervaEngine *e, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        nerva_tick(e);
    }
}

