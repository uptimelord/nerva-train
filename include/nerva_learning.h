// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_LEARNING_H
#define NERVA_LEARNING_H

#include "nerva_types.h"
#include <stdbool.h>

uint32_t nerva_feedback_correct(NervaEngine *e);
uint32_t nerva_feedback_wrong(NervaEngine *e);
uint32_t nerva_feedback_correct_tag(NervaEngine *e, uint32_t trace_tag);
uint32_t nerva_feedback_wrong_tag(NervaEngine *e, uint32_t trace_tag);

bool nerva_queue_weight_delta(NervaEngine *e, uint32_t edge_id, nerva_q8_8_t delta,
                              uint32_t reason, uint32_t trace_tag);
bool nerva_queue_gate_close(NervaEngine *e, uint32_t edge_id, uint32_t reason, uint32_t trace_tag);

bool nerva_queue_create_edge(NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation,
                             uint32_t reason);

/* Hebbian co-fire: if src fired within the trace window and dst currently holds
 * charge (co-active), queue a strengthening of the directed src->dst association,
 * creating the edge if absent. The association is LEARNED from activity, not wired.
 * Returns true if a create/strengthen mutation was queued. */
bool nerva_hebbian_cofire(NervaEngine *e, uint32_t src, uint32_t dst, uint16_t relation,
                          nerva_q8_8_t delta, uint32_t reason);

#endif
