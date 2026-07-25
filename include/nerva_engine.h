// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_ENGINE_H
#define NERVA_ENGINE_H

#include "nerva_types.h"

int nerva_engine_init(NervaEngine *e, NervaConfig cfg);
void nerva_engine_free(NervaEngine *e);

void nerva_tick(NervaEngine *e);
void nerva_tick_n(NervaEngine *e, uint32_t n);

#endif
