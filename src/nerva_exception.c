// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_exception.h"
#include "nerva_graph.h"
#include "nerva_math.h"
#include "nerva_mutation.h"
#include "nerva_trace.h"

#include <string.h>

bool nerva_queue_blocker_edge(NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation) {
    if (!e || source >= e->node_count || target >= e->node_count) {
        return false;
    }

    NervaMutation m;
    memset(&m, 0, sizeof(m));
    m.tick = e->tick;
    m.type = NERVA_MUT_ADD_BLOCKER_EDGE;
    m.source = source;
    m.target = target;
    m.relation = relation != NERVA_REL_NONE ? relation : NERVA_REL_BLOCKS;
    m.edge_flags = (uint16_t)(NERVA_EDGE_INHIBITORY | NERVA_EDGE_BLOCKER);
    m.delta_weight = (nerva_q8_8_t)(-e->cfg.default_weight_q8_8);
    m.debug_reason = NERVA_REASON_EXCEPTION_BLOCKER;
    return nerva_mutation_queue_push(e, m);
}

static int nerva_exception_blocker_source_active(const NervaEngine *e, uint32_t source_id) {
    if (!e || source_id >= e->node_count) {
        return 0;
    }

    const NervaNode *src = &e->nodes[source_id];
    if (src->flags & NERVA_NODE_DELETED) {
        return 0;
    }
    if (src->activation_count == 0) {
        return 0;
    }
    return src->last_fired_tick >= e->last_query_start_tick;
}

uint16_t nerva_exception_apply_suppression(NervaEngine *e, NervaNode *n, const NervaEvent *ev) {
    if (!e || !n || !ev || ev->signal <= 0 || !e->adjacency_valid) {
        return 0;
    }

    for (uint32_t k = 0; k < n->blocker_in_count; ++k) {
        uint32_t slot = n->first_blocker_in + k;
        if (slot >= e->blocker_in_count) {
            break;
        }

        uint32_t blocker_edge_id = e->blocker_in_edges[slot];
        if (blocker_edge_id >= e->edge_count) {
            continue;
        }

        const NervaEdge *blocker = &e->edges[blocker_edge_id];
        if (blocker->flags & NERVA_EDGE_DELETED) {
            continue;
        }
        if (!(blocker->flags & NERVA_EDGE_BLOCKER)) {
            continue;
        }
        if (!nerva_exception_blocker_source_active(e, blocker->source)) {
            continue;
        }

        int32_t v = (int32_t)n->v;
        v = (v * (65535 - (int32_t)e->cfg.suppress_q0_16)) >> 16;
        n->v = nerva_q8_8_clip(v);
        e->debug.exceptions_suppressed++;
        return (uint16_t)(NERVA_TRACE_EXCEPTION | NERVA_TRACE_BLOCKER);
    }

    return 0;
}

uint32_t nerva_exception_count_blocker_traces(const NervaEngine *e) {
    if (!e) {
        return 0;
    }

    uint32_t count = 0;
    uint32_t limit = e->trace_count;
    if (limit > e->cfg.trace_decay_scan_limit) {
        limit = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = 0; age < limit; ++age) {
        NervaTrace *t = nerva_trace_recent((NervaEngine *)e, age);
        if (t && (t->flags & NERVA_TRACE_BLOCKER)) {
            count++;
        }
    }
    return count;
}

int nerva_exception_trace_has_suppressed_path(const NervaEngine *e, uint32_t edge_id) {
    if (!e) {
        return 0;
    }

    uint32_t limit = e->trace_count;
    if (limit > e->cfg.trace_decay_scan_limit) {
        limit = e->cfg.trace_decay_scan_limit;
    }

    for (uint32_t age = 0; age < limit; ++age) {
        NervaTrace *t = nerva_trace_recent((NervaEngine *)e, age);
        if (!t || t->edge_id != edge_id) {
            continue;
        }
        if ((t->flags & (NERVA_TRACE_EXCEPTION | NERVA_TRACE_BLOCKER)) ==
            (uint16_t)(NERVA_TRACE_EXCEPTION | NERVA_TRACE_BLOCKER)) {
            return 1;
        }
    }
    return 0;
}
