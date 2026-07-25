// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_PREDICTION_H
#define NERVA_PREDICTION_H

#include "nerva_types.h"
#include <stdio.h>
#include <stdbool.h>

void nerva_set_prediction_mode(NervaEngine *e, int enabled);
void nerva_prediction_clear(NervaEngine *e);
void nerva_prediction_expire_stale(NervaEngine *e);

void nerva_prediction_on_fire(NervaEngine *e, uint32_t source_node_id);
void nerva_prediction_on_actual(NervaEngine *e, const NervaEvent *ev);

bool nerva_inject_edge_event(NervaEngine *e, uint32_t edge_id, nerva_q8_8_t signal);

const NervaExpectation *nerva_prediction_pending_for_query(const NervaEngine *e, uint32_t query_tag);
uint32_t nerva_prediction_count_pending(const NervaEngine *e);

void nerva_prediction_print_trace(const NervaEngine *e, FILE *out, uint32_t limit);
int nerva_prediction_save_trace(const NervaEngine *e, const char *path, uint32_t limit);

#endif
