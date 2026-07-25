// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_EVENT_H
#define NERVA_EVENT_H

#include "nerva_types.h"
#include <stdbool.h>

bool nerva_event_push(NervaEngine *e, NervaEvent ev);
bool nerva_event_pop(NervaEngine *e, NervaEvent *out);
bool nerva_event_merge_or_push(NervaEngine *e, NervaEvent ev);
bool nerva_event_overflow_admit(NervaEngine *e, NervaEvent ev);

void nerva_apply_event_to_node(NervaEngine *e, NervaNode *n, const NervaEvent *ev);
bool nerva_node_should_fire(const NervaEngine *e, const NervaNode *n);
uint32_t nerva_node_refractory_remaining(const NervaEngine *e, const NervaNode *n);
void nerva_fire_node(NervaEngine *e, uint32_t node_id);
nerva_q8_8_t nerva_compute_edge_signal(const NervaEngine *e, uint32_t source, const NervaEdge *ed);
void nerva_apply_leak(NervaEngine *e, NervaNode *n);
void nerva_mark_active(NervaEngine *e, uint32_t node_id);

bool nerva_activate_node(NervaEngine *e, uint32_t node_id, nerva_q8_8_t signal);

#endif
