/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Paul Odantabao II
 *
 * Product PCW fact session: teach/edit via paired causal work (R-grad off);
 * ask is pure forward nerva_tick — no weight updates.
 */
#ifndef NERVA_PCW_SESSION_H
#define NERVA_PCW_SESSION_H

#include "nerva_engine.h"
#include "nerva_work.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NERVA_PCW_MAX_KEYS 48
#define NERVA_PCW_MAX_VALS 96
#define NERVA_PCW_STR 64
#define NERVA_PCW_SETTLE 3
#define NERVA_PCW_DELTA 48
#define NERVA_PCW_TEACH_ROUNDS 3
#define NERVA_PCW_MAX_SEQ 16
#define NERVA_PCW_MAX_GEN 12
#define NERVA_PCW_SEQ_TEACH_PASSES 4

typedef struct NervaPcwSession {
    NervaEngine eng;
    int ready;
    char keys[NERVA_PCW_MAX_KEYS][NERVA_PCW_STR];
    char vals[NERVA_PCW_MAX_VALS][NERVA_PCW_STR];
    int n_keys, n_vals;
    uint32_t key_node[NERVA_PCW_MAX_KEYS];
    uint32_t val_node[NERVA_PCW_MAX_VALS];
    uint32_t edge[NERVA_PCW_MAX_KEYS][NERVA_PCW_MAX_VALS];
    NervaWorkLedger ledger;
    uint64_t teach_count;
    uint64_t ask_count;
    uint64_t gen_count;
    uint64_t seq_teach_count;
    uint64_t applies_at_last_ask; /* snapshot of apply counter after last ask */
    uint64_t query_apply_delta;   /* must stay 0 if ask/generate never updates */
} NervaPcwSession;

/* Init forces R-grad updates OFF. Returns 0 ok, -1 fail. */
int nerva_pcw_session_init(NervaPcwSession *s);
void nerva_pcw_session_free(NervaPcwSession *s);

/* Stream teach/edit: key→value via PCW (all-edge candidates, stream+Q).
 * Returns 1 if at least one weight apply happened, 0 no-apply still ok, -1 fail. */
int nerva_pcw_session_teach(NervaPcwSession *s, const char *key, const char *value);

/* Pure forward ask. Writes predicted value to out. Returns 1 hit / 0 unknown / -1 fail.
 * Never applies weight updates (query_apply_delta tracks counter drift). */
int nerva_pcw_session_ask(NervaPcwSession *s, const char *key, char *out, size_t out_cap);

/*
 * Multi-token sequence (next-token chain over PCW weights, not templates):
 * teach_sequence: for toks[0..n-1], PCW-teach each bigram toks[i]→toks[i+1].
 * generate: pure forward chain from seed (not including seed); ≥1 steps, no weight updates.
 */
int nerva_pcw_session_teach_sequence(NervaPcwSession *s, const char *const *toks, int n);
/* Whitespace-separated tokens in line; n must be ≥2. */
int nerva_pcw_session_teach_sequence_line(NervaPcwSession *s, const char *line);
/* Writes generated tokens into out[0..*n_out-1]. Returns 0 ok, -1 fail.
 * *n_out is number of tokens produced (0 if none). max_out caps length. */
int nerva_pcw_session_generate(NervaPcwSession *s, const char *seed,
                               char out[][NERVA_PCW_STR], int max_out, int *n_out);

/* Match: want[0..] appears as a contiguous prefix of gen[0..n_gen-1]. Returns 1/0. */
int nerva_pcw_seq_match_prefix(char gen[][NERVA_PCW_STR], int n_gen, const char *const *want,
                               int n_want);

/* Composite product key: "subj|rel" (normalized lower, no spaces in parts). */
void nerva_pcw_fact_key(char *out, size_t cap, const char *subj, const char *rel);

int nerva_pcw_session_rgrad_off(const NervaPcwSession *s);
uint64_t nerva_pcw_session_apply_count(void);
uint64_t nerva_pcw_session_refuse_count(void);

#ifdef __cplusplus
}
#endif

#endif
