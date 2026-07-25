// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_TRACE_H
#define NERVA_TRACE_H

#include "nerva_types.h"
#include <stdio.h>

uint32_t nerva_make_path_tag(const NervaEngine *e, uint32_t edge_id);

void nerva_trace_record(NervaEngine *e, const NervaEvent *ev, uint16_t flags,
                        nerva_q8_8_t target_v_before, nerva_q8_8_t target_v_after,
                        int fired_after_apply);
void nerva_trace_decay(NervaEngine *e);

NervaTrace *nerva_trace_recent(NervaEngine *e, uint32_t age);
uint32_t nerva_trace_count_used_path(const NervaEngine *e);
uint32_t nerva_trace_count_edge_flags(const NervaEngine *e, uint32_t edge_id, uint16_t flags);
int nerva_trace_find_edge_in_recent(const NervaEngine *e, uint32_t edge_id,
                                    uint16_t required_flags, uint32_t trace_tag_filter);

void nerva_trace_clear(NervaEngine *e);
void nerva_trace_print_recent(const NervaEngine *e, FILE *out, uint32_t limit);
void nerva_trace_print_path(const NervaEngine *e, FILE *out, uint32_t limit);
int nerva_trace_save_recent(const NervaEngine *e, const char *path, uint32_t limit);
int nerva_trace_save_path(const NervaEngine *e, const char *path, uint32_t limit);

#endif
