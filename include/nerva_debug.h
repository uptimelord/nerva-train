// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_DEBUG_H
#define NERVA_DEBUG_H

#include "nerva_types.h"
#include <stdio.h>

void nerva_debug_log_fire(NervaEngine *e, uint32_t node_id);
void nerva_debug_clear_fire_log(NervaEngine *e);
void nerva_debug_print_fire_trace(const NervaEngine *e, FILE *out);
void nerva_debug_print_path_line(const NervaEngine *e, FILE *out, const uint32_t *node_ids,
                                 uint32_t count);
int nerva_debug_save_fire_trace(const NervaEngine *e, const char *path);
int nerva_debug_save_fire_trace_with_path(const NervaEngine *e, const char *path,
                                          const uint32_t *node_ids, uint32_t count);
int nerva_debug_fire_sequence_matches(const NervaEngine *e, const uint32_t *node_ids, uint32_t count);

#endif
