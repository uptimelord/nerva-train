// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_branch.h"

#include <stdlib.h>
#include <string.h>

static int g_pcw_enabled = 0;

void nerva_pcw_set_enabled(int enabled) { g_pcw_enabled = enabled ? 1 : 0; }
int nerva_pcw_enabled(void) { return g_pcw_enabled; }

static uint64_t fnv1a(const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 14695981039346656037ull;
    size_t i;
    for (i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

int nerva_branch_clone(const NervaEngine *src, NervaEngine *dst) {
    if (!src || !dst)
        return -1;
    memset(dst, 0, sizeof(*dst));
    dst->cfg = src->cfg;
    dst->tick = src->tick;
    dst->node_count = src->node_count;
    dst->node_cap = src->node_cap;
    dst->edge_count = src->edge_count;
    dst->edge_cap = src->edge_cap;
    dst->event_count = src->event_count;
    dst->event_cap = src->event_cap;
    if (src->node_cap) {
        dst->nodes = (NervaNode *)calloc(src->node_cap, sizeof(NervaNode));
        if (!dst->nodes)
            return -1;
        memcpy(dst->nodes, src->nodes, (size_t)src->node_count * sizeof(NervaNode));
    }
    if (src->edge_cap) {
        dst->edges = (NervaEdge *)calloc(src->edge_cap, sizeof(NervaEdge));
        if (!dst->edges) {
            free(dst->nodes);
            return -1;
        }
        memcpy(dst->edges, src->edges, (size_t)src->edge_count * sizeof(NervaEdge));
    }
    if (src->event_cap) {
        dst->events = (NervaEvent *)calloc(src->event_cap, sizeof(NervaEvent));
        if (!dst->events) {
            free(dst->nodes);
            free(dst->edges);
            return -1;
        }
        memcpy(dst->events, src->events, (size_t)src->event_count * sizeof(NervaEvent));
    }
    return 0;
}

void nerva_branch_free_clone(NervaEngine *dst) {
    if (!dst)
        return;
    free(dst->nodes);
    free(dst->edges);
    free(dst->events);
    free(dst->sorted_edges);
    free(dst->blocker_in_edges);
    free(dst->active_nodes);
    free(dst->fire_log);
    free(dst->traces);
    free(dst->mutations);
    free(dst->mutation_log);
    free(dst->expectations);
    free(dst->schemas);
    free(dst->memory);
    /* names not deep-copied */
    memset(dst, 0, sizeof(*dst));
}

uint64_t nerva_branch_fingerprint(const NervaEngine *e) {
    uint64_t h = 14695981039346656037ull;
    uint32_t i;
    if (!e)
        return 0;
    h = fnv1a(&e->tick, sizeof(e->tick)) ^ (h * 1099511628211ull);
    for (i = 0; i < e->node_count; ++i) {
        h ^= fnv1a(&e->nodes[i].v, sizeof(e->nodes[i].v));
        h *= 1099511628211ull;
        h ^= fnv1a(&e->nodes[i].theta_fire, sizeof(e->nodes[i].theta_fire));
        h *= 1099511628211ull;
    }
    for (i = 0; i < e->edge_count; ++i) {
        h ^= fnv1a(&e->edges[i].weight, sizeof(e->edges[i].weight));
        h *= 1099511628211ull;
        h ^= fnv1a(&e->edges[i].gate, sizeof(e->edges[i].gate));
        h *= 1099511628211ull;
    }
    return h;
}
