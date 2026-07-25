// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II
//
// Fluency: non-parametric next-token prediction realized as a Nerva firing
// graph. No backprop, no gradient. "Learning" is local count increment on
// edges (Hebbian); "prediction" is a firing event -- context nodes fire into
// high-theta integrator token nodes, and the integrated charge is the
// interpolated n-gram vote. Depth is replaced by memory + backoff orders, so
// there is no deep credit-assignment problem to solve.

#ifndef NERVA_FLUENCY_H
#define NERVA_FLUENCY_H

#include "nerva_types.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define FLUENCY_MAX_ORDER 8u

/* Token id width. Char-level uses 0..255; word/BPE/LLM front-ends assign dense
 * ids up to FLUENCY_VOCAB-1. Default 16384 (product word boards). LLM builds use
 * -DFLUENCY_VOCAB=65536 (tiktoken/GPT-class ~50k fits). flu_tok_t is uint16_t so
 * hard max is 65536. Edge-slot keys use FLU_TOK_KEY_BITS=16.
 * Skip-1 contexts that pack (tok+V) into flu_tok_t are DISABLED when V>32768.
 * F2 embedding is O(V^2) — skip at large V. Tests only use a sparse id subset. */
typedef uint16_t flu_tok_t;
#ifndef FLUENCY_VOCAB
#define FLUENCY_VOCAB 16384u
#endif
#if FLUENCY_VOCAB > 65536u
#error "FLUENCY_VOCAB max is 65536 (flu_tok_t = uint16_t)"
#endif
/* Bits reserved for tok in edge-slot keys (must cover FLUENCY_VOCAB-1). */
#ifndef FLU_TOK_KEY_BITS
#if FLUENCY_VOCAB > 16384u
#define FLU_TOK_KEY_BITS 16u
#else
#define FLU_TOK_KEY_BITS 14u
#endif
#endif
/* Skip packing uses flu_tok_t ids in [V, 2V). Needs 2V-1 <= UINT16_MAX. */
#define FLU_SKIP_PACK_OK (FLUENCY_VOCAB <= 32768u)

#define FLUENCY_WORD_LEN 31u

/* Word-level front-end: maps whitespace-delimited words to token ids 0..255 so a
 * bounded-vocabulary corpus reuses the byte-level model unchanged (each word IS
 * a token). At word granularity, embedding similarity means synonym/paraphrase,
 * which is where the F3 kernel actually pays off on natural text. */
typedef struct FluencyWords {
    char word[FLUENCY_VOCAB][FLUENCY_WORD_LEN + 1u];
    uint32_t freq[FLUENCY_VOCAB]; /* how often each word was seen whole (a count) */
    uint32_t count;
} FluencyWords;

/* Custom relation: context node -> continuation token node (carries the count). */
#define FLU_REL_CTX_TO_TOK ((uint16_t)(NERVA_REL_CUSTOM_BASE + 200u))
/* Documented identity of a fast edge (token-type -> token-type). Fast edges live
 * in FluencyModel's fixed pool, not the main NervaEdge table -- see FAST_EDGE_REPORT. */
#define FLU_REL_FAST ((uint16_t)(NERVA_REL_CUSTOM_BASE + 201u))

/* One-shot Hebbian fast edges: per-source out-slot cap (configurable). */
#define FLU_FAST_SLOTS_DEFAULT 6u
/* Full deposit strength (Q8.8). One-shot bias: any positive inject wins when
 * n-gram votes are empty (induction); kept small so λ∈{0.25,0.5,1} does not
 * swamp trained multi-order counts on ordinary text. */
/* Small one-shot bias (Q8.8). With λ=1 injects 3 units — enough to win empty
 * n-gram races (induction); with λ=0.5 injects 1 unit — ppl stays ≤ baseline×1.01.
 * λ=0.25 underflows to 0 inject at this W0 (integer UQ0.16 scale). */
#define FLU_FAST_W0_DEFAULT ((nerva_q8_8_t)4)
/* Lazy decay half-life in stream tokens (not engine ticks). */
#define FLU_FAST_HALF_LIFE_DEFAULT 256u
/* Delta-rule write rate β (UQ0.16). 65535 = NERVA_UQ0_16_ONE ⇒ exact full write. */
#define FLU_FAST_BETA_DEFAULT ((nerva_uq0_16_t)65535u)
/* Competitor depression δ (UQ0.16). ONE ⇒ full depress (clean rebinding).
 * δ=0 recovers ≈v1 naive refresh (target to W0, competitors untouched). */
#define FLU_FAST_DELTA_DEFAULT ((nerva_uq0_16_t)65535u)

/* ---- Context organs (step 5): mood registers + gain-from-charge ----
 * Both default OFF → bit-exact baseline (λ=0-style inertness). */
#define FLU_MOOD_K_MAX 16u
#define FLU_MOOD_K_DEFAULT 8u
#define FLU_MOOD_HALF_LIFE_DEFAULT 32u
/* Max Q8.8 units injected at full register×affinity. */
#define FLU_MOOD_SCALE_DEFAULT ((nerva_q8_8_t)4)
/* Self-observation mood update weight relative to user (1/16). */
#define FLU_MOOD_SELF_DIV 16u
#define FLU_GAIN_WINDOW_DEFAULT 8u
#define FLU_GAIN_SCALE_DEFAULT ((nerva_q8_8_t)4)
#define FLU_GAIN_TOPK_DEFAULT 256u
/* Max candidates recorded in /why (winner + top losers). */
#define FLU_WHY_TOP 4u

/* ---- Phonebook hubs (step 6): compile-time similarity wiring ---------------
 * Offline PPMI signatures → k-means hubs → graded membership edges (uq0_16).
 * Runtime: pathway gated closed by default; emergency dial only when
 * integrator margin < hub_conf_margin (surprise-gated 2-hop). All default OFF
 * (Organism Gate criterion 2: bit-exact inert). Mood stays nulled — hub
 * influence is local to the dialed prediction only. */
#define FLU_HUB_H_DEFAULT 64u
#define FLU_HUB_H_MAX 256u
#define FLU_HUB_MEM_MAX 3u       /* polysemy: token may join up to 3 hubs */
#define FLU_HUB_SIZE_CAP 48u     /* max members retained per hub (cost cap) */
#define FLU_HUB_SCALE_DEFAULT ((nerva_q8_8_t)4)
/* Dial when (winner − runner-up) charge < this (Q8.8 units). Same spirit as
 * chat confidence-collapse stop. 0 ⇒ never dial even if hub_on (strict off). */
#define FLU_HUB_CONF_MARGIN_DEFAULT 1
#define FLU_HUB_COOC_TOPK_DEFAULT 256u
#define FLU_HUB_WINDOW 4 /* ±4, matches scale_pretrain cooc sidecar */

/* ---- Instance binding (step 7b): conjunction-keyed fast side pool ----------
 * Order-2 fast edges addressed by hash(cue, src) so R concurrent role bindings
 * coexist. Default OFF → bit-exact with pre-instance baseline. Type-node
 * path (fluency_fast_deposit) stays primary; instance inject only when
 * instance_on and ctx_len≥2 (cue live). Bounded open-address pool. */
#define FLU_INSTANCE_CAP_DEFAULT 1024u
/* Eviction: lowest effective weight, then oldest deposit_pos (LRU-ish). */

/* Token→hub membership slot (soft; hub=0xFFFF empty). */
typedef struct FluencyHubSlot {
    uint16_t hub;
    nerva_uq0_16_t weight;
} FluencyHubSlot;

typedef struct FluencyEdgeSlot {
    uint64_t key;     /* (src_node << 9) | next_byte, +1 biased; 0 = empty */
    uint32_t edge_id;
} FluencyEdgeSlot;

/* Episodic fast edge slot: delta-rule strength + lazy stream-distance decay.
 * Empty when target == FLUENCY_VOCAB. weight is the base strength at
 * deposit_pos; w_eff = weight >> (elapsed / half_life). deposit_pos is the
 * stream token counter at last (re)materialize -- not engine last_active_tick32. */
typedef struct FluencyFastSlot {
    flu_tok_t target;
    nerva_q8_8_t weight; /* base strength at deposit_pos; 0 when empty */
    uint32_t deposit_pos;
} FluencyFastSlot;

/* Conjunction-keyed instance slot: (cue, src) → target. Empty when live=0. */
typedef struct FluencyInstanceSlot {
    flu_tok_t cue;
    flu_tok_t src;
    flu_tok_t target;
    nerva_q8_8_t weight;
    uint32_t deposit_pos;
    uint8_t live;
} FluencyInstanceSlot;

/* One candidate's charge split for /why (Organism Gate criterion 1). */
typedef struct FluencyWhyCand {
    flu_tok_t tok;
    double total;                         /* integrated charge (pre-smooth) */
    double ngram[FLUENCY_MAX_ORDER + 1u]; /* [k] = order-k contribution; [0] unused */
    double ngram_sum;
    double fast;
    double mood;
    double gain;
    double hub;           /* phonebook 2-hop (0 if not dialed) */
    double skip;          /* skip-1 gapped context (0 when skip_on=0) */
    double unexplained; /* total - accounted; report if >5% of total */
} FluencyWhyCand;

/* Last-prediction attribution snapshot. */
typedef struct FluencyWhy {
    int valid;
    flu_tok_t win;
    uint32_t n_cands; /* 1..FLU_WHY_TOP */
    FluencyWhyCand cand[FLU_WHY_TOP];
    /* Fast-edge detail for the winner (if any inject landed on win). */
    flu_tok_t fast_src;
    nerva_q8_8_t fast_weff;
    int fast_hit; /* 1 if a fast edge into win was active */
    int hub_dialed; /* 1 if emergency dial opened during this explain */
} FluencyWhy;

/* Count→weight mapping at materialize time (offline/tooling only).
 * RAW: min(count * order_step, weight_max) — legacy saturating map.
 * CTX_NORM: u32 counts offline; per-context for max_c>=64:
 *   drop hapax (and count<3 when max_c>=256); weight = cnt/max_c * 4096
 *   * (step/top_step). Sparse contexts keep RAW. Restores monotone scale
 *   curve (50M PPL ≤ 10M) by keeping α-relative scale stable. */
#define FLU_WEIGHT_MAP_RAW 0u
#define FLU_WEIGHT_MAP_CTX_NORM 1u

typedef struct FluencyModel {
    NervaEngine *e;
    uint32_t order;
    uint8_t weight_map; /* FLU_WEIGHT_MAP_*; default RAW (bit-exact legacy) */

    /* PPL-push levers (both default OFF = bit-exact legacy predict).
     * skip_on: skip-1 gapped contexts (w-2, _, next). Counted at train time
     *   (set BEFORE fluency_train_counted) into the FC1 namespace keyed
     *   ctx_token + FLUENCY_VOCAB, so no naming/materialize special cases;
     *   predict/explain fire the skip node alongside backoff orders.
     * kn_on: Kneser-Ney-style continuation prior replaces the uniform α
     *   prior at smoothing time. kn_cont[b] = #distinct order-1 contexts b
     *   follows (counted path fills it); q(b)=kn_cont[b]/kn_cont_total and
     *   base(b) = α·V·q(b); with q uniform this reduces exactly to legacy. */
    uint8_t skip_on;
    uint8_t kn_on;
    uint32_t *kn_cont; /* [FLUENCY_VOCAB] distinct-predecessor counts (lazy) */
    uint64_t kn_cont_total;
    /* Pass2 Π_B: materialize highest-count (ctx,next) first under node/edge rent.
     * Default 0 = sealed M_v2 (hash order + low-order-first). Orthogonal to soft_nused. */
    uint8_t pass2_pi_b;
    /* Pass1 unique-(ctx,next) soft-cap. Default 7000000 (sealed M_v2 hapax-evict).
     * 0 = exact counts (no steal); grow failure aborts train. */
    size_t count_soft_nused;
    /* CTW-class recursive mix at predict (default 0 = sealed superposition path). */
    uint8_t ctw_mix;
    /* node_id → tok for CTW edge readout (lazy; NERVA_INVALID_ID = not a tok). */
    uint32_t *ctw_tok_of_node;
    uint32_t ctw_tok_cap;

    uint32_t tok_node[FLUENCY_VOCAB];  /* byte -> integrator node id (INVALID if unseen) */
    uint8_t seen[FLUENCY_VOCAB];       /* byte present in vocab */
    uint32_t vocab_count;

    uint64_t unigram[FLUENCY_VOCAB];   /* order-0 baseline counts (reference only) */
    uint64_t unigram_total;

    /* Edge-id index so counting stays O(1) per (context, next). This is a plain
     * lookup table of WHERE each count edge lives; the counts themselves are the
     * edge weights in the graph, and prediction reads them by firing. */
    FluencyEdgeSlot *slots;
    size_t slot_cap;
    size_t slot_count;

    /* Offline name→node_id open-address map (heap OK). Kills O(n)
     * nerva_find_node_by_name on materialize + predict. Engine hot path
     * unchanged; fluency owns this index for nodes it creates. */
    void *name_map; /* FluNameSlot[] — opaque to callers */
    size_t name_map_cap;
    size_t name_map_n;
    uint64_t name_hits;
    uint64_t name_creates;
    double train_pass1_s; /* last fluency_train_counted profile */
    double train_pass2_s;

    uint64_t train_positions;

    /* Fast edges (token-type -> token-type): fixed pool, stream-clock decay.
     * fast_out is [src * fast_slots + k], allocated once at init. λ=0 skips
     * all inject work so prediction stays bit-exact with the count baseline. */
    FluencyFastSlot *fast_out;
    uint16_t fast_slots;              /* per-source cap (default 6) */
    nerva_q8_8_t fast_w0;             /* full deposit strength (Q8.8) */
    uint32_t fast_half_life;          /* tokens per halving */
    nerva_uq0_16_t fast_lambda;       /* blend weight; 0 = baseline path only */
    nerva_uq0_16_t fast_beta;         /* delta write rate β; ONE = full write */
    nerva_uq0_16_t fast_delta;        /* competitor depression δ; 0 ≈ v1 */
    uint32_t stream_pos;              /* stream token counter (decay clock) */
    uint64_t fast_read_ops;           /* slot reads per predict (cost probe) */
    uint64_t fast_deposit_reads;      /* slot reads per deposit (≤ fast_slots) */

    /* Contestant C — mood registers (default OFF, bit-exact). Parallel array
     * on FluencyModel; engine untouched. Decay + affinity tilt at predict. */
    uint8_t mood_on;
    uint8_t mood_k;                   /* 1..FLU_MOOD_K_MAX */
    uint32_t mood_half_life;          /* tokens per halfing of registers */
    nerva_q8_8_t mood_scale;          /* inject scale (Q8.8) */
    float mood_reg[FLU_MOOD_K_MAX];   /* drifting accumulators */
    float *mood_aff;                  /* [FLUENCY_VOCAB * mood_k] or NULL */
    uint64_t mood_inject_ops;         /* affinity reads per mood inject */

    /* Contestant B — gain-from-charge (default OFF, bit-exact). Fluency-level
     * residual×cooc inject mirroring gate_ctrl [0,ONE) attenuation math.
     * Engine gate_ctrl not wired onto count edges: uncharged ctrl would zero
     * n-gram signals and break the permanent baseline. */
    uint8_t gain_on;
    uint16_t gain_window;             /* recent ctx tokens for residual */
    nerva_q8_8_t gain_scale;
    uint32_t gain_topk;
    uint32_t *gain_cooc;              /* [topk*topk] raw cooc counts; NULL ok */
    flu_tok_t *gain_tok;              /* [topk] token ids in cooc index */
    uint16_t *gain_vidx;              /* [FLUENCY_VOCAB] -> topk idx or 0xFFFF */
    uint64_t gain_inject_ops;

    /* Phonebook hubs (step 6). Default OFF → bit-exact baseline. Tables are
     * fluency-owned CSR membership (not engine edges): closed when hub_on=0
     * or scale=0; open only on emergency dial for that prediction. */
    uint8_t hub_on;
    uint16_t hub_h;                 /* number of hubs built (0 = none) */
    nerva_q8_8_t hub_scale;         /* inject scale (Q8.8) */
    int32_t hub_conf_margin;        /* dial if margin < this */
    FluencyHubSlot *tok_hubs;       /* [V * MEM_MAX] */
    flu_tok_t *hub_mem_tok;         /* [H * SIZE_CAP] */
    nerva_uq0_16_t *hub_mem_w;      /* [H * SIZE_CAP] */
    uint16_t *hub_mem_n;            /* [H] live member counts */
    uint64_t hub_dial_count;        /* predictions that opened the pathway */
    uint64_t hub_dial_ops;          /* edge-signal comps across all dials */
    uint64_t hub_predict_count;     /* predicts while hub_on (for dial rate) */
    uint8_t last_dialed;            /* 1 if last predict dialed hubs */
    int32_t last_margin;            /* raw charge margin of last predict */
    uint64_t hub_last_ops;          /* ops on last dial (0 if not dialed) */

    /* /why snapshot (filled by fluency_explain; not hot path). */
    FluencyWhy last_why;

    /* Instance binding side pool (step 7b). Default OFF → bit-exact. Fixed
     * open-address table; budget = cap * sizeof(FluencyInstanceSlot). */
    uint8_t instance_on;
    uint32_t instance_cap;            /* table slots (default 1024) */
    FluencyInstanceSlot *instance_pool;
    uint32_t instance_n;              /* live occupied */
    uint64_t instance_deposits;
    uint64_t instance_evicts;
    uint64_t instance_hits;           /* successful injects that landed mass */
    uint64_t instance_inject_ops;     /* probes during inject */
    /* Last inject detail for /why (folded into fast%; convention below). */
    flu_tok_t instance_last_cue;
    flu_tok_t instance_last_src;
    flu_tok_t instance_last_tgt;
    nerva_q8_8_t instance_last_weff;
    int instance_last_hit; /* 1 if instance edge into winner on last explain */

    /* F2: shallow co-occurrence embedding (no backprop). vidx maps a byte to a
     * compact 0..vdim-1 vocab slot; vbyte is the inverse. embed is a
     * vdim x embed_dim row-major matrix of token vectors, derived by
     * eigendecomposition of the positive-PMI co-occurrence matrix. */
    uint32_t vidx[FLUENCY_VOCAB];
    flu_tok_t vbyte[FLUENCY_VOCAB];
    uint32_t vdim;
    uint32_t embed_dim;
    double *embed;

    /* Product organs (default OFF → bit-exact with prior fluency path).
     * text_organ: KN + instance bind (cue,src)→tgt taught on stream.
     * tscar_organ: sparse content-address key=prev → value=next, mix at predict. */
    uint8_t text_organ;
    uint8_t tscar_organ;
    double tscar_mix; /* λ in [0,1]; 0 when organ off */
    void *tscar_slots; /* FluTscarSlot[FLU_TSCAR_SLOT_CAP] or NULL */
    uint32_t tscar_cap;
    uint32_t tscar_used;
    uint32_t tscar_writes;
    uint32_t tscar_hits; /* last predict: true-in-top-k when known */
    uint32_t tscar_last_slot;
    flu_tok_t tscar_last_key;
    flu_tok_t tscar_last_val;
} FluencyModel;

#ifndef FLU_TSCAR_SLOT_CAP
#define FLU_TSCAR_SLOT_CAP 4096u
#endif

typedef struct FluencyMetrics {
    uint64_t eval_positions;
    uint64_t correct_argmax;       /* next-byte argmax == actual */
    double nll_sum;                /* sum of -log2 P(actual) */
    double model_perplexity;       /* 2^(nll_sum / eval_positions) */
    double unigram_perplexity;     /* order-0 baseline on the same eval text */
    uint64_t eval_mutations;       /* learning mutations during eval (must be 0) */
    uint32_t eval_node_growth;     /* nodes created during eval (must be 0) */
    uint32_t eval_edge_growth;     /* edges created during eval (must be 0) */
} FluencyMetrics;

/* Bind a model to an already-initialized engine. order in [1, FLUENCY_MAX_ORDER]. */
int fluency_init(FluencyModel *m, NervaEngine *e, uint32_t order);
void fluency_free(FluencyModel *m);

/* Rebuild offline name→node index from engine names (after persist load). */
int fluency_rebuild_name_index(FluencyModel *m);

/* Portable name-map blob (occupied key→node_id pairs). For session checkpoint. */
int fluency_name_map_export(const FluencyModel *m, uint8_t **out, size_t *out_len);
int fluency_name_map_import(FluencyModel *m, const uint8_t *buf, size_t len);

/* ---- Session checkpoint (kill-and-resume; fluency side state + engine graph)
 * One file, magic "NervaSes", versioned header + payload CRC. Engine section is
 * a full nerva_persist v0.2 blob; fluency section restores fast/instance/mood/
 * hubs/slots/stream_pos (THE subtle clock). See fluency_session.h for layout.
 * Chat layer: nerva_chat_save_state / nerva_chat_load_state in chat_repl.h. */
int fluency_save(const FluencyModel *m, const char *path);
/* Load into a zeroed engine + zeroed model: inits engine caps from header,
 * nerva_persist_load of embedded graph, fluency_init, then overlays side state.
 * Returns 0 on success. On failure leaves e and m freeable (partial state). */
int fluency_load(NervaEngine *e, FluencyModel *m, const char *path);

/* Ensure token integrator node exists (runtime mint / vocab growth). Does not
 * train n-grams or advance the stream clock. Returns 0 on success. */
int fluency_ensure_tok(FluencyModel *m, flu_tok_t tok);

/* Count all context->next occurrences in the text (local increment; no backprop).
 * Also deposits one-shot fast edges on each consecutive token pair. */
void fluency_train(FluencyModel *m, const flu_tok_t *text, size_t len);

/* Offline counting path (heap OK): stream text, accumulate context→token counts
 * in a hash table (u32), then materialize named nodes/edges. Weight mapping
 * follows m->weight_map (RAW or CTX_NORM). Skips fast-edge stream deposits
 * (λ=0 eval is the equivalence surface). Returns 0 on success.
 * sat_log: optional CSV of new_nodes/new_edges per chunk_tokens (NULL to skip). */
typedef struct FluencySatPoint {
    uint64_t tokens_seen;
    uint32_t nodes;
    uint32_t edges;
    uint32_t new_nodes; /* since previous point */
    uint32_t new_edges;
} FluencySatPoint;

int fluency_train_counted(FluencyModel *m, const flu_tok_t *text, size_t len,
                          FluencySatPoint *sat, size_t sat_cap, size_t *sat_n,
                          uint32_t chunk_tokens);

/* Same as fluency_train_counted, but pass-1 count accumulation uses up to
 * n_jobs CPU shards (offline only; heap OK). n_jobs<=1 ⇒ identical serial path.
 * Pass-2 graph materialize stays single-threaded.
 * Soft-cap from m->count_soft_nused (default 7M sealed v2; 0=exact).
 * Context-hash count shards exist in-code but stay n_cshards=1 by default. */
int fluency_train_counted_jobs(FluencyModel *m, const flu_tok_t *text, size_t len,
                               FluencySatPoint *sat, size_t sat_cap, size_t *sat_n,
                               uint32_t chunk_tokens, uint32_t n_jobs);

/* Like fluency_train, but logs new-nodes/new-edges every chunk_tokens. */
void fluency_train_sat(FluencyModel *m, const flu_tok_t *text, size_t len,
                       FluencySatPoint *sat, size_t sat_cap, size_t *sat_n,
                       uint32_t chunk_tokens);

/* Delta-rule deposit A->B: read-before-write on A's slots, strengthen B by
 * β·(W0 − w_eff), depress competitors by β·δ·w_eff. With β=ONE, δ=0 ≈ v1
 * refresh; with β=δ=ONE full overwrite of prior bindings. Cost O(fast_slots). */
void fluency_fast_deposit(FluencyModel *m, flu_tok_t a, flu_tok_t b);

/* ---- Instance binding (default OFF) ----------------------------------------
 * Deposit (cue, A)→B into the conjunction side pool. No-op if !instance_on.
 * Does not touch the type-node fast pool (call fluency_fast_deposit separately
 * if bare-A last-binding semantics are also desired). */
void fluency_instance_set(FluencyModel *m, int on);
int fluency_instance_on(const FluencyModel *m);
void fluency_instance_deposit(FluencyModel *m, flu_tok_t cue, flu_tok_t a, flu_tok_t b);
/* Bytes of fixed instance pool (cap * sizeof slot); 0 if unallocated. */
size_t fluency_instance_pool_bytes(const FluencyModel *m);

/* PPL-push levers (see FluencyModel field docs). skip_on must be set BEFORE
 * fluency_train_counted for the skip family to be counted; kn uses counts the
 * counted path always fills, so it can be toggled at any time. Both OFF ⇒
 * predict is bit-exact with the legacy path. */
void fluency_skip_set(FluencyModel *m, int on);
void fluency_kn_set(FluencyModel *m, int on);

/* Product organs (default OFF). Off ⇒ predict bit-exact with pre-organ path. */
void fluency_text_organ_set(FluencyModel *m, int on);
void fluency_tscar_organ_set(FluencyModel *m, int on);
int fluency_text_organ_on(const FluencyModel *m);
int fluency_tscar_organ_on(const FluencyModel *m);
/* After counted train (or any stream): deposit organ memory from text. */
void fluency_organs_observe_stream(FluencyModel *m, const flu_tok_t *text, size_t len);

/* Advance the stream token clock by one (call once per consumed token). */
void fluency_stream_advance(FluencyModel *m);

/* Fill dist[0..255] with the integrated firing vote per byte for the given
 * context (last ctx_len bytes). Returns the argmax byte. dist is smoothed
 * probability (sums to ~1 over the vocab). Permanent-graph frozen; with
 * fast_lambda>0, also injects decayed fast-edge votes from the most recent
 * context token. λ=0 is bit-exact baseline. mood_on/gain_on default off =
 * bit-exact with pre-step-5 baseline. */
int fluency_predict(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len, double *dist);

/* ---- Mood / gain (context organs; default OFF) ----------------------------- */

/* Build offline cooc + k-means topic affinities (heap OK). Fills mood_aff and
 * gain_cooc/gain_tok/gain_vidx. Does not enable organs (caller sets mood_on /
 * gain_on). Returns 0 on success. */
int fluency_ctx_organs_build(FluencyModel *m, const flu_tok_t *text, size_t len,
                             uint32_t mood_k, uint32_t cooc_topk);

/* Enable/disable. Off → inject paths skipped entirely (bit-exact). */
void fluency_mood_set(FluencyModel *m, int on);
void fluency_gain_set(FluencyModel *m, int on);

/* Update mood registers from an observed token. weight=1.0 user; use
 * 1/FLU_MOOD_SELF_DIV for self-generated (volume asymmetry). No-op if !mood_on
 * or no affinities. Decays registers by one token half-life step first. */
void fluency_mood_observe(FluencyModel *m, flu_tok_t tok, float weight);

/* Zero registers (e.g. /reset). */
void fluency_mood_reset(FluencyModel *m);

/* Decompose a prediction into n-gram / fast / mood / gain charges for the
 * winner and top-3 losers. Writes m->last_why. Does not mutate stream/mood. */
int fluency_explain(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len, FluencyWhy *out);

/* Pretty-print /why table to out. word_fn may be NULL (prints ids). */
void fluency_why_print(const FluencyWhy *w, FILE *out,
                       const char *(*word_fn)(void *ud, flu_tok_t id), void *ud);

/* /mood dump: register values + top-affinity tokens per axis. */
void fluency_mood_print(const FluencyModel *m, FILE *out,
                        const char *(*word_fn)(void *ud, flu_tok_t id), void *ud);

/* /gain <tok>: residual cooc partners for tok (if gain tables built). */
void fluency_gain_print(const FluencyModel *m, flu_tok_t tok, FILE *out,
                        const char *(*word_fn)(void *ud, flu_tok_t id), void *ud);

/* ---- Phonebook hubs (step 6) ---------------------------------------------- */

/* Build hubs from stream text: windowed cooc → PPMI signatures → k-means
 * (H clusters) → soft membership (up to MEM_MAX hubs/token by distance
 * threshold). Vectors discarded after materialization. Heap OK offline.
 * Does not enable the pathway (caller sets hub_on). Returns 0 on success. */
int fluency_hubs_build(FluencyModel *m, const flu_tok_t *text, size_t len,
                       uint32_t n_hubs, uint32_t cooc_topk);

/* Build from an existing dense cooc matrix (e.g. scale_pretrain sidecar).
 * cooc is topk×topk row-major; tok_of_row[i] maps row i → flu_tok_t (may be
 * NULL ⇒ identity i). Returns 0 on success. */
int fluency_hubs_build_from_cooc(FluencyModel *m, const uint32_t *cooc, uint32_t topk,
                                 const flu_tok_t *tok_of_row, uint32_t n_hubs);

/* Enable/disable pathway availability. Off → inject skipped entirely
 * (bit-exact). When on, still surprise-gated by hub_conf_margin. */
void fluency_hubs_set(FluencyModel *m, int on);

/* /hubs <tok>: pages the token is on, membership weights, top co-members. */
void fluency_hubs_print(const FluencyModel *m, flu_tok_t tok, FILE *out,
                        const char *(*word_fn)(void *ud, flu_tok_t id), void *ud);

/* Human-readable sample: up to n_hubs hubs with top-10 members each. */
void fluency_hubs_sanity_table(const FluencyModel *m, FILE *out, uint32_t n_hubs,
                               const char *(*word_fn)(void *ud, flu_tok_t id), void *ud);

/* Evaluate model + unigram baseline perplexity and next-byte accuracy on text.
 * Permanent graph stays frozen (growth/mutation deltas). Fast edges are
 * deposited at stream time (eval-time induction) into the model's own pool. */
void fluency_evaluate(FluencyModel *m, const flu_tok_t *text, size_t len, FluencyMetrics *out);

/* Generate n bytes from a seed context into out (n+1 bytes incl NUL). */
void fluency_generate(FluencyModel *m, const flu_tok_t *seed, size_t seed_len,
                      flu_tok_t *out, size_t n, uint32_t rng_seed, int sample);

/*
 * PCW credit on multi-token sequences (R-grad off): for each next-token position,
 * measure ±δ on fluency ctx→tok edges via forward fluency_predict soft-loss Q,
 * commit via nerva_work_apply_weight_delta. Shared weights with fluency_generate.
 * text[0..n-1] is the full sequence (n≥2). Returns 1 if any apply, 0 none, -1 fail.
 */
int fluency_pcw_teach_sequence(FluencyModel *m, const flu_tok_t *text, size_t n);

/* ---- F2: co-occurrence embedding (no backprop, closed-form) ---------------- */

/* Build a dim-dimensional token embedding from a symmetric co-occurrence window
 * over text: positive PMI matrix -> top-`dim` eigenvectors by power iteration
 * with Gram-Schmidt deflation. Pure function of counts (deterministic, frozen).
 * Requires the vocabulary to already exist (call fluency_train first, or it
 * seeds the vocab from this text). Returns 0 on success. */
int fluency_build_embedding(FluencyModel *m, const flu_tok_t *text, size_t len,
                            uint32_t window, uint32_t dim);

/* Cosine similarity between two bytes' embedding vectors; 0 if either unseen. */
double fluency_cosine(const FluencyModel *m, flu_tok_t a, flu_tok_t b);

/* Fill out[] with the `k` nearest bytes to `byte` by cosine (excluding itself),
 * most similar first. Returns the count written. */
uint32_t fluency_neighbors(const FluencyModel *m, flu_tok_t byte, flu_tok_t *out, uint32_t k);

/* ---- Word-level front-end -------------------------------------------------- */

void fluency_words_init(FluencyWords *w);

/* Look up (or, if create, assign) the token id for a lowercased word. Returns
 * the id, or FLUENCY_VOCAB on overflow / when not found and create is 0. */
uint32_t fluency_words_id(FluencyWords *w, const char *word, int create);

const char *fluency_words_word(const FluencyWords *w, flu_tok_t id);

/* Tokenize text into token ids (whitespace/punct delimited, lowercased). With
 * create=0, out-of-vocab words are dropped. Returns the number of ids written. */
size_t fluency_words_encode(FluencyWords *w, const char *text, flu_tok_t *out, size_t cap,
                            int create);

/* ---- F4.1: subword (BPE) tokenization -------------------------------------
 * Frequency-driven byte-pair merges (no backprop): learn the most common
 * adjacent-symbol merges within words, so a word splits into reusable subword
 * units. Shared FORM then generalizes -- cats and catnip both carry the "cat"
 * unit -- complementing F4's shared-CONTEXT (synonym) generalization. The whole
 * subword vocabulary (base chars + merges) is capped at 256 so it still feeds
 * the byte model unchanged. */
#define FLU_SYM_LEN 24u
#define FLU_BPE_MAX_WORDS 8192u
#define FLU_BPE_MAX_WLEN 40u

typedef struct FluencySubword {
    char sym[FLUENCY_VOCAB][FLU_SYM_LEN + 1u]; /* id -> its substring expansion */
    uint32_t sym_count;
    int16_t byte_sym[256];                     /* base byte -> symbol id, -1 none */
    uint16_t merge_a[FLUENCY_VOCAB];
    uint16_t merge_b[FLUENCY_VOCAB];
    uint16_t merge_new[FLUENCY_VOCAB];
    uint32_t merge_count;
    uint32_t space_sym; /* word-boundary symbol, FLUENCY_VOCAB if unset */
} FluencySubword;

void fluency_bpe_init(FluencySubword *s);

/* Learn merges from text until the vocabulary reaches target_vocab (<=256) or no
 * pair repeats. Deterministic; pure counting. */
void fluency_bpe_learn(FluencySubword *s, const char *text, uint32_t target_vocab);

const char *fluency_bpe_sym(const FluencySubword *s, flu_tok_t id);

/* Encode one lowercased word into subword ids (greedy merge application). */
size_t fluency_bpe_encode_word(const FluencySubword *s, const char *word, flu_tok_t *out, size_t cap);

/* Encode text into a subword-id stream with a boundary symbol between words. */
size_t fluency_bpe_encode_text(const FluencySubword *s, const char *text, flu_tok_t *out, size_t cap);

/* ---- F4.3: morpheme-aware segmentation (branching entropy / Harris) --------
 * Unlike BPE (which merges by raw frequency and will cross morpheme boundaries),
 * this segments where the number of DISTINCT characters that can follow the
 * prefix-so-far spikes -- Harris's hypothesis that a morpheme boundary is a point
 * of high branching (after a complete morpheme, many continuations open up).
 * Learned from character-transition counts over the corpus; no backprop. */
typedef struct FluencyMorph {
    /* prefix (char n-gram, up to depth) -> set of distinct following chars.
     * Open-addressing hash of prefix-hash -> 256-bit followed-char bitset popcount. */
    uint32_t *pref_hash;    /* key+1, 0 empty */
    uint8_t (*follow)[32];  /* 256-bit bitset of chars seen after this prefix */
    size_t cap;
    size_t count;
    uint32_t depth;         /* max prefix length considered */
} FluencyMorph;

void fluency_morph_init(FluencyMorph *mo);
void fluency_morph_free(FluencyMorph *mo);

/* Learn character branching statistics from text (depth = max prefix length). */
int fluency_morph_learn(FluencyMorph *mo, const char *text, uint32_t depth);

/* Segment one lowercased word: insert a boundary before each position where the
 * branching factor (distinct chars that followed the current suffix in training)
 * rises by more than `jump`. Writes cut offsets into cuts[], returns how many. */
uint32_t fluency_morph_segment(const FluencyMorph *mo, const char *word, uint32_t jump,
                               uint32_t *cuts, uint32_t max_cuts);

/* ---- F4.1: learned lexicalization (compose-vs-override decided by counts) ---
 * Whether a whole word overrides its compositional reading is decided by
 * EVIDENCE, not a hand-set weight: a word is lexicalized when it is both (a) seen
 * whole often enough to be memorable and (b) decomposable into >=2 subword parts
 * (an atomic single-unit word has nothing to override). Both inputs are counts
 * -- word frequency and subword length -- so the override is learned by the same
 * count machinery as everything else, no backprop. Returns the whole-word
 * frequency required to lexicalize, and a verdict for a given word. */
int fluency_word_is_lexicalized(const FluencyWords *w, const FluencySubword *s, const char *word,
                                uint32_t min_freq);

/* ---- F5: firing-path (reservoir) projection -------------------------------
 * "The projection is the stroke of lightning." Instead of comparing static
 * embedding vectors (F2/F3), project a token by the WAVE it sets off through a
 * token-transition graph: fire it, let activation propagate several hops with
 * leak (a reservoir / diffusion kernel), and read the firing signature it leaves
 * across the graph. Similarity = overlap of two signatures. This is nonlinear,
 * shaped by the whole edge structure, and picks up TRANSITIVE structure a static
 * co-occurrence window cannot see -- yet the learning was pure counting (edge =
 * observed transition) and the projection stays traceable (the lit path IS the
 * trace). No backprop. Separate engine so reservoir dynamics never touch the
 * model's count/integrator graph. */
typedef struct FluencyReservoir {
    NervaEngine *e;
    uint32_t node[FLUENCY_VOCAB]; /* token id -> reservoir node (INVALID if absent) */
    uint8_t present[FLUENCY_VOCAB];
    uint32_t vcount;
    double *scratch_a; /* FLUENCY_VOCAB signature buffers (heap; avoids big stack) */
    double *scratch_b;

    /* Signature cache (built once with V fires, then queries are sparse dot-
     * products instead of O(V) fires each). CSR over present tokens. */
    int cached;
    uint32_t cache_hops;
    size_t *cache_off;  /* [FLUENCY_VOCAB+1] */
    flu_tok_t *cache_tok;
    double *cache_w;
    double *cache_norm; /* [FLUENCY_VOCAB] signature L2 norms */
} FluencyReservoir;

/* Build the token-transition (bigram) graph from a token stream. Binary edges
 * (a transition seen -> weight ONE), so the wave travels cleanly; fan-out gives
 * the diffusion its shape. Returns 0 on success. */
int fluency_reservoir_build(FluencyReservoir *r, const flu_tok_t *text, size_t len);
void fluency_reservoir_free(FluencyReservoir *r);

/* Fire `seed`, propagate `hops` ticks with leak, fill sig[0..FLUENCY_VOCAB) with
 * each token's firing closeness (earlier/stronger = larger; 0 if unreached). */
void fluency_firing_signature(FluencyReservoir *r, flu_tok_t seed, uint32_t hops, double *sig);

/* Precompute every present token's signature (V fires) so firing_cosine /
 * firing_neighbors are O(nnz) sparse dot-products instead of O(V) fires per
 * query. Rebuilt automatically if a different `hops` is later requested. */
int fluency_reservoir_cache(FluencyReservoir *r, uint32_t hops);

/* Cosine of two tokens' firing signatures (the reservoir similarity kernel). */
double fluency_firing_cosine(FluencyReservoir *r, flu_tok_t a, flu_tok_t b, uint32_t hops);

/* Top-`k` tokens by firing-signature cosine to `tok` (excluding itself and the
 * query's own forward cone, so results are role-peers not downstream tokens). */
uint32_t fluency_firing_neighbors(FluencyReservoir *r, flu_tok_t tok, uint32_t hops,
                                  flu_tok_t *out, uint32_t k);

/* Kernel prediction whose similarity comes from the FIRING-PATH projection (the
 * reservoir) instead of the static embedding: borrow continuations from
 * firing-path role-peers of the predecessor. Beats the embedding kernel where
 * similarity is TRANSITIVE (visible only through multi-hop reach). Frozen. */
int fluency_predict_firing(FluencyModel *m, FluencyReservoir *r, const flu_tok_t *ctx,
                           size_t ctx_len, double *dist, double lambda, uint32_t hops,
                           uint32_t k_nb);
void fluency_evaluate_firing(FluencyModel *m, FluencyReservoir *r, const flu_tok_t *text,
                             size_t len, double lambda, uint32_t hops, uint32_t k_nb,
                             FluencyMetrics *out);

/* ---- F3: kernel-smoothed prediction over the embedding --------------------- */

/* Like fluency_predict, but when firing the exact context also fires contexts
 * with the predecessor byte replaced by its top-`smooth_k` embedding neighbours,
 * each vote scaled by cosine and mixed in with weight `lambda`. This is
 * Nadaraya-Watson smoothing in embedding space: an unseen context borrows
 * continuations from distributionally-similar contexts. lambda<=0 or no
 * embedding reduces exactly to fluency_predict. Read-only / frozen. */
int fluency_predict_kernel(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len, double *dist,
                           double lambda, uint32_t smooth_k);

/* Perplexity + next-byte accuracy under kernel smoothing; frozen-discipline
 * deltas in out (must be zero). */
void fluency_evaluate_kernel(FluencyModel *m, const flu_tok_t *text, size_t len, double lambda,
                             uint32_t smooth_k, FluencyMetrics *out);

#endif
