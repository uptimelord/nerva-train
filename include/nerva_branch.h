// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II
//
// ERG branch/checkpoint: full-clone reference path for paired causal work.
// When PCW is compiled in but disabled (default), these APIs are inert no-ops
// for the hot path — T0 requires PCW-off fingerprints match legacy.

#ifndef NERVA_BRANCH_H
#define NERVA_BRANCH_H

#include "nerva_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Global enable (0 = off, default). T0 seals inertness when 0. */
void nerva_pcw_set_enabled(int enabled);
int nerva_pcw_enabled(void);

/* Deep-clone dynamic engine state (nodes/edges/events/tick). Returns 0 ok. */
int nerva_branch_clone(const NervaEngine *src, NervaEngine *dst);

/* Free a clone created by nerva_branch_clone (not the original). */
void nerva_branch_free_clone(NervaEngine *dst);

/* Fingerprint of node voltages + edge weights for T0 / reconvergence. */
uint64_t nerva_branch_fingerprint(const NervaEngine *e);

#ifdef __cplusplus
}
#endif

#endif
