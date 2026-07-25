// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_GRAPH_H
#define NERVA_GRAPH_H

#include "nerva_types.h"

uint32_t nerva_intern_name(NervaEngine *e, const char *name);
const char *nerva_name_by_id(const NervaEngine *e, uint32_t name_id);
uint32_t nerva_find_node_by_name(const NervaEngine *e, const char *name);
uint32_t nerva_get_or_create_node(NervaEngine *e, const char *name);

uint16_t nerva_relation_from_string(const char *name);

uint32_t nerva_graph_create_node(NervaEngine *e, uint32_t name_id);
uint32_t nerva_graph_create_edge(NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation);
uint32_t nerva_graph_create_blocker_edge(NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation);

void nerva_graph_rebuild_adjacency(NervaEngine *e);

int nerva_graph_modify_weight(NervaEngine *e, uint32_t edge_id, nerva_q8_8_t delta,
                              nerva_q8_8_t *old_weight, nerva_q8_8_t *new_weight);
int nerva_graph_modify_gate(NervaEngine *e, uint32_t edge_id, nerva_uq0_16_t new_gate,
                            nerva_uq0_16_t *old_gate, nerva_uq0_16_t *new_gate_out);

/* Conditional gate (Stage 0e): bind edge's signal modulation to control node's
 * charge. ctrl_node == NERVA_INVALID_ID clears it (unconditional). */
int nerva_graph_set_gate_ctrl(NervaEngine *e, uint32_t edge_id, uint32_t ctrl_node);

int nerva_graph_reachable(const NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation);
int nerva_graph_has_edge(const NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation);
int nerva_graph_reachable_named(const NervaEngine *e, const char *source_name,
                                const char *target_name, const char *relation_name);

#endif
