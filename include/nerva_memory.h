// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_MEMORY_H
#define NERVA_MEMORY_H

#include "nerva_types.h"
#include <stdio.h>

uint32_t nerva_memory_begin_episode(NervaEngine *e, uint32_t query_tag);
void nerva_memory_end_episode(NervaEngine *e, uint32_t mem_id);
void nerva_memory_charge_update(NervaEngine *e, uint32_t mem_id, float useful, float surprise,
                                float repetition, float correction);

void nerva_consolidate_idle(NervaEngine *e);
void nerva_memory_on_tick_end(NervaEngine *e);

int nerva_memory_is_consolidated(const NervaEngine *e, uint32_t mem_id);
int nerva_memory_is_marked_delete(const NervaEngine *e, uint32_t mem_id);
int nerva_memory_is_episode_open(const NervaEngine *e, uint32_t mem_id);
const NervaMemoryBlock *nerva_memory_get(const NervaEngine *e, uint32_t mem_id);
uint32_t nerva_memory_count(const NervaEngine *e);

void nerva_memory_print_blocks(const NervaEngine *e, FILE *out);
int nerva_memory_save_log(const NervaEngine *e, const char *path);

#endif
