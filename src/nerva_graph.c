// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_graph.h"
#include "nerva_config.h"
#include "nerva_math.h"

#include <stdlib.h>
#include <string.h>

static const NervaEngine *g_sort_engine;

static int nerva_edge_cmp_ids(uint32_t edge_a, uint32_t edge_b) {
    const NervaEdge *a = &g_sort_engine->edges[edge_a];
    const NervaEdge *b = &g_sort_engine->edges[edge_b];

    if (a->source != b->source) {
        return (a->source < b->source) ? -1 : 1;
    }
    if (a->relation != b->relation) {
        return (a->relation < b->relation) ? -1 : 1;
    }
    if (a->target != b->target) {
        return (a->target < b->target) ? -1 : 1;
    }
    if (edge_a != edge_b) {
        return (edge_a < edge_b) ? -1 : 1;
    }
    return 0;
}

static int nerva_edge_index_cmp(const void *pa, const void *pb) {
    return nerva_edge_cmp_ids(*(const uint32_t *)pa, *(const uint32_t *)pb);
}

static void nerva_sort_edge_indices(NervaEngine *e) {
    for (uint32_t i = 0; i < e->edge_count; ++i) {
        e->sorted_edges[i] = i;
    }
    if (e->edge_count < 2) {
        return;
    }

    g_sort_engine = e;
    qsort(e->sorted_edges, e->edge_count, sizeof(uint32_t), nerva_edge_index_cmp);
    g_sort_engine = NULL;
}

static void nerva_clear_adjacency(NervaEngine *e) {
    for (uint32_t i = 0; i < e->node_count; ++i) {
        e->nodes[i].first_out = 0;
        e->nodes[i].out_count = 0;
        e->nodes[i].first_blocker_in = 0;
        e->nodes[i].blocker_in_count = 0;
    }
}

static int nerva_blocker_edge_target_cmp(const void *pa, const void *pb) {
    uint32_t edge_a = *(const uint32_t *)pa;
    uint32_t edge_b = *(const uint32_t *)pb;
    const NervaEdge *a = &g_sort_engine->edges[edge_a];
    const NervaEdge *b = &g_sort_engine->edges[edge_b];

    if (a->target != b->target) {
        return (a->target < b->target) ? -1 : 1;
    }
    if (edge_a != edge_b) {
        return (edge_a < edge_b) ? -1 : 1;
    }
    return 0;
}

static void nerva_rebuild_blocker_in(NervaEngine *e) {
    if (!e || !e->blocker_in_edges) {
        return;
    }

    uint32_t count = 0;
    for (uint32_t i = 0; i < e->edge_count; ++i) {
        const NervaEdge *ed = &e->edges[i];
        if (ed->flags & NERVA_EDGE_DELETED) {
            continue;
        }
        if (!(ed->flags & NERVA_EDGE_BLOCKER)) {
            continue;
        }
        e->blocker_in_edges[count++] = i;
    }
    e->blocker_in_count = count;

    if (count < 2) {
        if (count == 1) {
            uint32_t target = e->edges[e->blocker_in_edges[0]].target;
            if (target < e->node_count) {
                e->nodes[target].first_blocker_in = 0;
                e->nodes[target].blocker_in_count = 1;
            }
        }
        return;
    }

    g_sort_engine = e;
    qsort(e->blocker_in_edges, count, sizeof(uint32_t), nerva_blocker_edge_target_cmp);
    g_sort_engine = NULL;

    for (uint32_t i = 0; i < e->node_count; ++i) {
        e->nodes[i].first_blocker_in = 0;
        e->nodes[i].blocker_in_count = 0;
    }

    for (uint32_t i = 0; i < count; ) {
        uint32_t edge_id = e->blocker_in_edges[i];
        uint32_t target = e->edges[edge_id].target;
        uint32_t start = i;
        ++i;
        while (i < count && e->edges[e->blocker_in_edges[i]].target == target) {
            ++i;
        }
        if (target >= e->node_count || (e->nodes[target].flags & NERVA_NODE_DELETED)) {
            continue;
        }
        e->nodes[target].first_blocker_in = start;
        e->nodes[target].blocker_in_count = (uint16_t)(i - start);
    }
}

static void nerva_apply_node_defaults(NervaEngine *e, NervaNode *n) {
    n->v = e->cfg.v_rest_q8_8;
    n->v_rest = e->cfg.v_rest_q8_8;
    n->v_reset = e->cfg.v_reset_q8_8;
    n->theta_fire = e->cfg.theta_fire_q8_8;
    n->refractory_max = e->cfg.refractory_ticks;
}

uint32_t nerva_intern_name(NervaEngine *e, const char *name) {
    if (!e || !name || name[0] == '\0') {
        return 0;
    }

    for (uint32_t i = 0; i < e->name_count; ++i) {
        if (strcmp(e->names[i], name) == 0) {
            return i + 1;
        }
    }

    if (e->name_count >= e->name_cap) {
        return 0;
    }

    char *copy = strdup(name);
    if (!copy) {
        return 0;
    }

    e->names[e->name_count] = copy;
    e->name_count++;
    return e->name_count;
}

const char *nerva_name_by_id(const NervaEngine *e, uint32_t name_id) {
    if (!e || name_id == 0 || name_id > e->name_count) {
        return NULL;
    }
    return e->names[name_id - 1];
}

uint32_t nerva_find_node_by_name(const NervaEngine *e, const char *name) {
    if (!e || !name || name[0] == '\0') {
        return UINT32_MAX;
    }

    uint32_t name_id = 0;
    for (uint32_t i = 0; i < e->name_count; ++i) {
        if (strcmp(e->names[i], name) == 0) {
            name_id = i + 1;
            break;
        }
    }
    if (name_id == 0) {
        return UINT32_MAX;
    }

    for (uint32_t i = 0; i < e->node_count; ++i) {
        const NervaNode *n = &e->nodes[i];
        if (!(n->flags & NERVA_NODE_DELETED) && n->name_id == name_id) {
            return n->id;
        }
    }

    return UINT32_MAX;
}

uint32_t nerva_get_or_create_node(NervaEngine *e, const char *name) {
    if (!e || !name || name[0] == '\0') {
        return UINT32_MAX;
    }

    uint32_t existing = nerva_find_node_by_name(e, name);
    if (existing != UINT32_MAX) {
        return existing;
    }

    uint32_t name_id = nerva_intern_name(e, name);
    if (name_id == 0) {
        return UINT32_MAX;
    }

    return nerva_graph_create_node(e, name_id);
}

uint16_t nerva_relation_from_string(const char *name) {
    if (!name) {
        return NERVA_REL_NONE;
    }
    if (strcmp(name, "kind_of") == 0) {
        return NERVA_REL_KIND_OF;
    }
    if (strcmp(name, "usually_has_property") == 0) {
        return NERVA_REL_USUALLY_HAS_PROPERTY;
    }
    if (strcmp(name, "blocks") == 0) {
        return NERVA_REL_BLOCKS;
    }
    if (strcmp(name, "inside") == 0) {
        return NERVA_REL_INSIDE;
    }
    if (strcmp(name, "moved_to") == 0) {
        return NERVA_REL_MOVED_TO;
    }
    if (strcmp(name, "located_at") == 0) {
        return NERVA_REL_LOCATED_AT;
    }
    return NERVA_REL_NONE;
}

uint32_t nerva_graph_create_node(NervaEngine *e, uint32_t name_id) {
    if (!e || e->node_count >= e->node_cap) {
        return UINT32_MAX;
    }

    if (name_id != 0) {
        for (uint32_t i = 0; i < e->node_count; ++i) {
            NervaNode *n = &e->nodes[i];
            if (!(n->flags & NERVA_NODE_DELETED) && n->name_id == name_id) {
                return n->id;
            }
        }
    }

    uint32_t id = e->node_count;
    NervaNode *n = &e->nodes[id];
    memset(n, 0, sizeof(*n));
    n->id = id;
    n->name_id = name_id;
    n->memory_block = UINT32_MAX;
    nerva_apply_node_defaults(e, n);
    e->node_count++;
    e->adjacency_valid = 0;
    return id;
}

uint32_t nerva_graph_create_edge(NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation) {
    if (!e || source >= e->node_count || target >= e->node_count || relation == NERVA_REL_NONE) {
        return UINT32_MAX;
    }
    if (e->nodes[source].flags & NERVA_NODE_DELETED || e->nodes[target].flags & NERVA_NODE_DELETED) {
        return UINT32_MAX;
    }
    if (e->edge_count >= e->edge_cap) {
        return UINT32_MAX;
    }

    for (uint32_t i = 0; i < e->edge_count; ++i) {
        const NervaEdge *ed = &e->edges[i];
        if (!(ed->flags & NERVA_EDGE_DELETED) &&
            ed->source == source &&
            ed->target == target &&
            ed->relation == relation) {
            return i;
        }
    }

    uint32_t id = e->edge_count;
    NervaEdge *ed = &e->edges[id];
    memset(ed, 0, sizeof(*ed));
    ed->source = source;
    ed->target = target;
    ed->relation = relation;
    ed->weight = e->cfg.default_weight_q8_8;
    ed->gate = (nerva_uq0_16_t)NERVA_UQ0_16_ONE;
    ed->delay_ticks = e->cfg.edge_delay_ticks;
    ed->memory_block = UINT32_MAX;
    ed->gate_ctrl = NERVA_INVALID_ID;
    e->edge_count++;
    e->adjacency_valid = 0;
    return id;
}

uint32_t nerva_graph_create_blocker_edge(NervaEngine *e, uint32_t source, uint32_t target,
                                         uint16_t relation) {
    if (relation == NERVA_REL_NONE) {
        relation = NERVA_REL_BLOCKS;
    }

    uint32_t id = nerva_graph_create_edge(e, source, target, relation);
    if (id == UINT32_MAX) {
        return UINT32_MAX;
    }

    NervaEdge *ed = &e->edges[id];
    ed->flags |= (uint16_t)(NERVA_EDGE_INHIBITORY | NERVA_EDGE_BLOCKER);
    ed->weight = e->cfg.default_weight_q8_8;
    return id;
}

void nerva_graph_rebuild_adjacency(NervaEngine *e) {
    if (!e) {
        return;
    }

    nerva_sort_edge_indices(e);
    nerva_clear_adjacency(e);

    for (uint32_t i = 0; i < e->edge_count; ) {
        uint32_t edge_id = e->sorted_edges[i];
        const NervaEdge *ed = &e->edges[edge_id];
        if (ed->source >= e->node_count) {
            ++i;
            continue;
        }

        uint32_t source = ed->source;
        uint32_t start = i;
        ++i;
        while (i < e->edge_count && e->edges[e->sorted_edges[i]].source == source) {
            ++i;
        }

        if (e->nodes[source].flags & NERVA_NODE_DELETED) {
            continue;
        }

        e->nodes[source].first_out = start;
        e->nodes[source].out_count = i - start;
    }

    nerva_rebuild_blocker_in(e);
    e->adjacency_valid = 1;
}

int nerva_graph_modify_weight(NervaEngine *e, uint32_t edge_id, nerva_q8_8_t delta,
                              nerva_q8_8_t *old_weight, nerva_q8_8_t *new_weight) {
    if (!e || edge_id >= e->edge_count) {
        return 0;
    }

    NervaEdge *ed = &e->edges[edge_id];
    if (ed->flags & NERVA_EDGE_DELETED) {
        return 0;
    }

    nerva_q8_8_t before = ed->weight;
    int32_t w = (int32_t)before + (int32_t)delta;
    if (w > (int32_t)e->cfg.weight_max_q8_8) {
        w = (int32_t)e->cfg.weight_max_q8_8;
    }
    if (w < (int32_t)e->cfg.weight_min_q8_8) {
        w = (int32_t)e->cfg.weight_min_q8_8;
    }

    ed->weight = (nerva_q8_8_t)w;
    if (old_weight) {
        *old_weight = before;
    }
    if (new_weight) {
        *new_weight = ed->weight;
    }
    return 1;
}

int nerva_graph_modify_gate(NervaEngine *e, uint32_t edge_id, nerva_uq0_16_t new_gate,
                            nerva_uq0_16_t *old_gate, nerva_uq0_16_t *new_gate_out) {
    if (!e || edge_id >= e->edge_count) {
        return 0;
    }

    NervaEdge *ed = &e->edges[edge_id];
    if (ed->flags & NERVA_EDGE_DELETED) {
        return 0;
    }

    nerva_uq0_16_t before = ed->gate;
    ed->gate = new_gate;
    if (old_gate) {
        *old_gate = before;
    }
    if (new_gate_out) {
        *new_gate_out = ed->gate;
    }
    return 1;
}

int nerva_graph_set_gate_ctrl(NervaEngine *e, uint32_t edge_id, uint32_t ctrl_node) {
    if (!e || edge_id >= e->edge_count) {
        return 0;
    }
    NervaEdge *ed = &e->edges[edge_id];
    if (ed->flags & NERVA_EDGE_DELETED) {
        return 0;
    }
    if (ctrl_node != NERVA_INVALID_ID && ctrl_node >= e->node_count) {
        return 0;
    }
    ed->gate_ctrl = ctrl_node;
    return 1;
}

int nerva_graph_has_edge(const NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation) {
    if (!e || source >= e->node_count || target >= e->node_count || relation == NERVA_REL_NONE) {
        return 0;
    }

    for (uint32_t i = 0; i < e->edge_count; ++i) {
        const NervaEdge *ed = &e->edges[i];
        if (ed->flags & NERVA_EDGE_DELETED) {
            continue;
        }
        if (ed->source == source && ed->target == target && ed->relation == relation) {
            return 1;
        }
    }
    return 0;
}

int nerva_graph_reachable(const NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation) {
    if (!e || source >= e->node_count || target >= e->node_count || relation == NERVA_REL_NONE) {
        return 0;
    }
    if (!e->adjacency_valid) {
        return 0;
    }
    if (e->nodes[source].flags & NERVA_NODE_DELETED || e->nodes[target].flags & NERVA_NODE_DELETED) {
        return 0;
    }
    if (source == target) {
        return 1;
    }

    uint8_t *visited = calloc(e->node_count, 1);
    uint32_t *queue = malloc(e->node_count * sizeof(uint32_t));
    if (!visited || !queue) {
        free(visited);
        free(queue);
        return 0;
    }

    uint32_t head = 0;
    uint32_t tail = 0;
    queue[tail++] = source;
    visited[source] = 1;

    int found = 0;
    while (head < tail) {
        uint32_t node_id = queue[head++];
        const NervaNode *n = &e->nodes[node_id];

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
            if (ed->relation != relation) {
                continue;
            }

            uint32_t next = ed->target;
            if (next == target) {
                found = 1;
                break;
            }
            if (!visited[next]) {
                visited[next] = 1;
                queue[tail++] = next;
            }
        }

        if (found) {
            break;
        }
    }

    free(visited);
    free(queue);
    return found;
}

int nerva_graph_reachable_named(const NervaEngine *e, const char *source_name,
                                const char *target_name, const char *relation_name) {
    if (!e) {
        return 0;
    }

    uint32_t source = nerva_find_node_by_name(e, source_name);
    uint32_t target = nerva_find_node_by_name(e, target_name);
    uint16_t relation = nerva_relation_from_string(relation_name);
    if (source == UINT32_MAX || target == UINT32_MAX || relation == NERVA_REL_NONE) {
        return 0;
    }
    return nerva_graph_reachable(e, source, target, relation);
}
