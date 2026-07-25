// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_MATH_H
#define NERVA_MATH_H

#include "nerva_types.h"

#define NERVA_Q8_8_ONE 256
#define NERVA_UQ0_16_ONE 65535u
#define NERVA_INVALID_ID UINT32_MAX

#define NERVA_EVT_ACTIVATION 0x0001u
/* Injected confirm/miss events use the same queue tag as normal propagation. */
#define NERVA_EVT_ACTUAL NERVA_EVT_ACTIVATION
#define NERVA_EVT_EXPECTED 0x0002u

static inline int32_t nerva_abs32(int32_t x) {
    return (x < 0) ? -x : x;
}

static inline nerva_q8_8_t nerva_q8_8_clip(int32_t x) {
    if (x > 32767) {
        return 32767;
    }
    if (x < -32768) {
        return -32768;
    }
    return (nerva_q8_8_t)x;
}

static inline nerva_q8_8_t nerva_q8_8_saturating_add(nerva_q8_8_t a, nerva_q8_8_t b) {
    return nerva_q8_8_clip((int32_t)a + (int32_t)b);
}

#define NERVA_I32_SCORE_MIN ((int32_t)-2147483647)
#define NERVA_I32_SCORE_MAX ((int32_t)2147483647)

static inline int32_t nerva_i32_saturating_add(int32_t acc, int32_t delta) {
    int64_t sum = (int64_t)acc + (int64_t)delta;
    if (sum > (int64_t)NERVA_I32_SCORE_MAX) {
        return NERVA_I32_SCORE_MAX;
    }
    if (sum < (int64_t)NERVA_I32_SCORE_MIN) {
        return NERVA_I32_SCORE_MIN;
    }
    return (int32_t)sum;
}

static inline int32_t nerva_i32_max(int32_t a, int32_t b) {
    return a > b ? a : b;
}

#endif
