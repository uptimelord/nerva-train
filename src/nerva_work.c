// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_work.h"

#include "nerva_learning.h"
#include "nerva_mutation.h"

#include <math.h>

static int g_rgrad_updates = 0; /* default OFF — replacement path */
static uint64_t g_pcw_apply_count = 0;
static uint64_t g_rgrad_refuse_count = 0;

void nerva_work_set_rgrad_updates(int enabled) { g_rgrad_updates = enabled ? 1 : 0; }
int nerva_work_rgrad_updates_enabled(void) { return g_rgrad_updates; }

uint64_t nerva_work_pcw_apply_count(void) { return g_pcw_apply_count; }
uint64_t nerva_work_rgrad_refuse_count(void) { return g_rgrad_refuse_count; }
void nerva_work_reset_counters(void) {
    g_pcw_apply_count = 0;
    g_rgrad_refuse_count = 0;
}

void nerva_work_ledger_init(NervaWorkLedger *L) {
    if (!L)
        return;
    L->discovery_bits = 0;
    L->seal_bits = 0;
    L->mint_count = 0;
    L->total_touches = 0;
}

void nerva_work_ledger_add_discovery(NervaWorkLedger *L, double bits, uint64_t touches) {
    if (!L)
        return;
    L->discovery_bits += bits;
    L->total_touches += touches;
}

void nerva_work_ledger_add_seal(NervaWorkLedger *L, double bits) {
    if (!L)
        return;
    L->seal_bits += bits;
}

int nerva_work_try_mint(NervaWorkLedger *L, double L_code, double conf, double work_min) {
    if (!L)
        return 0;
    /* Mint gate: seal evidence beats code length + conf; work count threshold. */
    if (L->seal_bits >= L_code + conf && (double)L->total_touches >= work_min) {
        L->mint_count++;
        return 1;
    }
    return 0;
}

int nerva_work_apply_weight_delta(NervaEngine *e, uint32_t edge_id, nerva_q8_8_t delta,
                                  uint32_t reason) {
    if (!e)
        return 0;
    if (g_rgrad_updates) {
        /* Replacement path forbids reverse-assisted updates. */
        g_rgrad_refuse_count++;
        return 0;
    }
    if (!nerva_queue_weight_delta(e, edge_id, delta, reason, 0))
        return 0;
    nerva_apply_mutations(e); /* commit PCW-measured delta immediately */
    g_pcw_apply_count++;
    return 1;
}
