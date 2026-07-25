// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_ROUTING_H
#define NERVA_ROUTING_H

#include "nerva_types.h"
#include <stdio.h>

void nerva_routing_reset(NervaEngine *e);
void nerva_routing_begin_query(NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation);
void nerva_routing_on_tick_end(NervaEngine *e);

nerva_q8_8_t nerva_routing_compute_difficulty(const NervaEngine *e);
int nerva_routing_was_crystallized_last_query(const NervaEngine *e);
int nerva_routing_fluid_active(const NervaEngine *e);

void nerva_routing_print_snapshot(const NervaEngine *e, FILE *out, const char *label);
void nerva_routing_print_state(const NervaEngine *e, FILE *out);
int nerva_routing_save_log(const NervaEngine *e, const char *path);

#endif
