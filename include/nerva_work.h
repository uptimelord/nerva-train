// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II
//
// Work receipts and seal ledgers for PCW. R-grad updates are never applied
// through this path — use nerva_work_set_rgrad_updates(0) (default).

#ifndef NERVA_WORK_H
#define NERVA_WORK_H

#include "nerva_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NervaWorkReceipt {
    double W_plus;
    double W_minus;
    uint64_t touches_plus;
    uint64_t touches_minus;
    uint32_t candidate_id;
    int killed;
    const char *kill_reason;
} NervaWorkReceipt;

typedef struct NervaWorkLedger {
    double discovery_bits;
    double seal_bits;
    uint32_t mint_count;
    uint64_t total_touches;
} NervaWorkLedger;

void nerva_work_set_rgrad_updates(int enabled);
int nerva_work_rgrad_updates_enabled(void);
uint64_t nerva_work_pcw_apply_count(void);
uint64_t nerva_work_rgrad_refuse_count(void);
void nerva_work_reset_counters(void);

void nerva_work_ledger_init(NervaWorkLedger *L);
void nerva_work_ledger_add_discovery(NervaWorkLedger *L, double bits, uint64_t touches);
/* Seal only post-candidacy evidence (A6). */
void nerva_work_ledger_add_seal(NervaWorkLedger *L, double bits);
int nerva_work_try_mint(NervaWorkLedger *L, double L_code, double conf, double work_min);

/* Apply a PCW-measured weight delta (no R-grad). Returns 1 if applied. */
int nerva_work_apply_weight_delta(NervaEngine *e, uint32_t edge_id, nerva_q8_8_t delta,
                                  uint32_t reason);

#ifdef __cplusplus
}
#endif

#endif
