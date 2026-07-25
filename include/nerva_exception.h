// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_EXCEPTION_H
#define NERVA_EXCEPTION_H

#include "nerva_types.h"
#include <stdbool.h>

bool nerva_queue_blocker_edge(NervaEngine *e, uint32_t source, uint32_t target, uint16_t relation);

uint16_t nerva_exception_apply_suppression(NervaEngine *e, NervaNode *n, const NervaEvent *ev);

uint32_t nerva_exception_count_blocker_traces(const NervaEngine *e);
int nerva_exception_trace_has_suppressed_path(const NervaEngine *e, uint32_t edge_id);

#endif
