// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_SCHEMA_H
#define NERVA_SCHEMA_H

#include "nerva_types.h"

uint16_t nerva_schema_output_relation(uint16_t rel_a, uint16_t rel_b);

/* Learnable composition table (Stage 0b): register rel_a∘rel_b→rel_out as a
 * candidate schema on this engine.  Promotion stays MDL-gated (support +
 * exception limit + compression benefit); registration only declares the
 * composition, it grants nothing.  The builtin table remains as fallback. */
int nerva_schema_register_composition(NervaEngine *e, uint16_t rel_a, uint16_t rel_b,
                                      uint16_t rel_out);
uint16_t nerva_schema_lookup_output(const NervaEngine *e, uint16_t rel_a, uint16_t rel_b);

void nerva_schema_observe_triple(NervaEngine *e, uint32_t a, uint16_t rel_a, uint32_t b,
                                 uint16_t rel_b, uint32_t c);

int nerva_schema_promote_if_ready(NervaEngine *e, uint32_t schema_id);
int nerva_schema_apply(NervaEngine *e, uint32_t a, uint16_t rel_a, uint32_t b, uint16_t rel_b,
                       uint32_t c);

const NervaSchema *nerva_schema_find_promoted(const NervaEngine *e, uint16_t rel_a, uint16_t rel_b);
uint32_t nerva_schema_count(const NervaEngine *e);

#endif
