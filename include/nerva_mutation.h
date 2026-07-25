// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_MUTATION_H
#define NERVA_MUTATION_H

#include "nerva_types.h"
#include <stdbool.h>
#include <stdio.h>

bool nerva_mutation_queue_push(NervaEngine *e, NervaMutation m);
void nerva_apply_mutations(NervaEngine *e);

void nerva_mutation_clear_log(NervaEngine *e);
void nerva_mutation_print_log(const NervaEngine *e, FILE *out);
int nerva_mutation_save_log(const NervaEngine *e, const char *path);

#endif
