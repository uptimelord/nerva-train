/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Paul Odantabao II
 *
 * nerva-train product CLI — one graph, one generate path.
 *   teach  = fluency_pcw_teach_sequence  (PCW on fluency edges, R-grad OFF)
 *   reply  = fluency_generate            (real LM tokens — not templates)
 *
 *   make && ./build/nerva --selfcheck
 *   ./build/nerva
 */

#include "fluency.h"
#include "fluency_session.h"
#include "nerva_config.h"
#include "nerva_engine.h"
#include "nerva_words_io.h"
#include "nerva_work.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOK 256
#define GEN_MAX 16
#define HIST_MAX 128
#define SMOKE_CORPUS "worlds/fluency/corpus/train.txt"
#define DEFAULT_CKPT "checkpoints/tinystories.sess"
#define DEFAULT_WORDS "checkpoints/tinystories.words"

/* Full pretrain fills node/edge caps exactly; teach needs spare capacity to
 * mint novel contexts. Chat boot only — pretrain resume keeps stored caps. */
#define TEACH_HEADROOM_NODES 16384u
#define TEACH_HEADROOM_EDGES 65536u
#define TEACH_HEADROOM_NAMES 16384u

typedef struct {
    NervaEngine eng;
    FluencyModel model;
    FluencyWords words;
    flu_tok_t train_ids[4096];
    size_t train_n;
    flu_tok_t hist[HIST_MAX];
    size_t hist_n;
    /* Last predict context for /why (order-truncated). */
    flu_tok_t last_ctx[FLUENCY_MAX_ORDER];
    size_t last_ctx_len;
    int last_pred_valid;
    size_t gen_default;
    int ready;
    int loaded_ckpt; /* 1 = frozen checkpoint (no pretrain this boot) */
    int organs_present; /* tables in side blob */
    int sample_on;   /* 0 = greedy (default); 1 = sample from fluency_generate dist */
    uint32_t rng_counter; /* incremented per sampled generate for varied seeds */
} App;

static void usage(const char *prog) {
    printf("Usage: %s [--selfcheck] [--script PATH] [--ckpt PATH] [--words PATH]\n", prog);
    printf("       [--force-smoke-train] [--sample] [--help]\n");
    printf("  nerva-train: one graph, one generate path (fluency + PCW).\n");
    printf("  Default boot: load frozen TinyStories checkpoint if present (no retrain).\n");
    printf("  Pretrain once: make pretrain  →  checkpoints/tinystories.sess\n");
    printf("  Organs once:   ./build/pretrain --organs\n");
    printf("  /seq /teach /gen /sample /hist /clear /why /mood /hubs /organs /quit | free text\n");
    printf("  --selfcheck   PASS/FAIL (uses ckpt if present, else smoke train)\n");
    printf("  --force-smoke-train  ignore ckpt; tiny train.txt boot only\n");
    printf("  --sample      enable multinomial sampling (default: greedy)\n");
}

static int organs_tables_present(const FluencyModel *m) {
    int mood = (m->mood_aff != NULL && m->mood_k > 0u);
    int gain = (m->gain_cooc && m->gain_tok && m->gain_vidx && m->gain_topk > 0u);
    int hubs = (m->tok_hubs && m->hub_mem_tok && m->hub_mem_w && m->hub_mem_n && m->hub_h > 0u);
    return (mood && gain && hubs) ? 1 : 0;
}

static size_t organs_table_bytes(const FluencyModel *m) {
    size_t b = 0;
    if (m->mood_aff && m->mood_k > 0u)
        b += sizeof(float) * (size_t)FLUENCY_VOCAB * (size_t)m->mood_k;
    if (m->gain_cooc && m->gain_tok && m->gain_vidx && m->gain_topk > 0u) {
        b += sizeof(uint32_t) * (size_t)m->gain_topk * (size_t)m->gain_topk;
        b += sizeof(flu_tok_t) * (size_t)m->gain_topk;
        b += sizeof(uint16_t) * (size_t)FLUENCY_VOCAB;
    }
    if (m->tok_hubs && m->hub_mem_tok && m->hub_mem_w && m->hub_mem_n && m->hub_h > 0u) {
        b += sizeof(FluencyHubSlot) * (size_t)FLUENCY_VOCAB * (size_t)FLU_HUB_MEM_MAX;
        b += sizeof(flu_tok_t) * (size_t)m->hub_h * (size_t)FLU_HUB_SIZE_CAP;
        b += sizeof(nerva_uq0_16_t) * (size_t)m->hub_h * (size_t)FLU_HUB_SIZE_CAP;
        b += sizeof(uint16_t) * (size_t)m->hub_h;
    }
    return b;
}

static void organs_enable_if_present(App *a) {
    a->organs_present = organs_tables_present(&a->model);
    if (!a->organs_present)
        return;
    fluency_mood_set(&a->model, 1);
    fluency_gain_set(&a->model, 1);
    fluency_hubs_set(&a->model, 1);
    fluency_mood_reset(&a->model);
    printf("organs=present mood_on=%u gain_on=%u hub_on=%u mood_k=%u gain_topk=%u "
           "hub_h=%u table_bytes=%zu hub_conf_margin=%d\n",
           (unsigned)a->model.mood_on, (unsigned)a->model.gain_on, (unsigned)a->model.hub_on,
           (unsigned)a->model.mood_k, (unsigned)a->model.gain_topk, (unsigned)a->model.hub_h,
           organs_table_bytes(&a->model), (int)a->model.hub_conf_margin);
}

static const char *app_word_fn(void *ud, flu_tok_t id) {
    return fluency_words_word((const FluencyWords *)ud, id);
}

/* User tokens: full mood weight. Self tokens: 1/FLU_MOOD_SELF_DIV. */
static void mood_observe_toks(App *a, const flu_tok_t *toks, size_t n, float weight) {
    size_t i;
    if (!a || !toks || n == 0)
        return;
    for (i = 0; i < n; ++i)
        fluency_mood_observe(&a->model, toks[i], weight);
}

static void snapshot_last_ctx(App *a, const flu_tok_t *seed, size_t seed_n) {
    size_t take;
    if (!a || !seed || seed_n < 1) {
        a->last_ctx_len = 0;
        a->last_pred_valid = 0;
        return;
    }
    take = seed_n < a->model.order ? seed_n : (size_t)a->model.order;
    if (take > FLUENCY_MAX_ORDER)
        take = FLUENCY_MAX_ORDER;
    memcpy(a->last_ctx, seed + (seed_n - take), take * sizeof(flu_tok_t));
    a->last_ctx_len = take;
    a->last_pred_valid = 1;
}

/* Load frozen weights + word lexicon. Engine must be zeroed (not inited). */
static int app_boot_ckpt(App *a, const char *ckpt, const char *words_path) {
    FILE *cf;
    long sz = 0;
    memset(&a->eng, 0, sizeof(a->eng));
    memset(&a->model, 0, sizeof(a->model));
    fluency_words_init(&a->words);
    if (fluency_session_load_ex(&a->eng, &a->model, ckpt, NULL, 0, NULL, TEACH_HEADROOM_NODES,
                                TEACH_HEADROOM_EDGES, TEACH_HEADROOM_NAMES) != 0) {
        fprintf(stderr, "boot: fluency_session_load_ex failed path=%s\n", ckpt);
        return -1;
    }
    if (nerva_words_load(&a->words, words_path) != 0) {
        fprintf(stderr, "boot: words load failed path=%s\n", words_path);
        fluency_free(&a->model);
        nerva_engine_free(&a->eng);
        return -1;
    }
    cf = fopen(ckpt, "rb");
    if (cf) {
        fseek(cf, 0, SEEK_END);
        sz = ftell(cf);
        fclose(cf);
    }
    /* Lab chat organism settings. Checkpoint stores λ≈0.15 / w0=4, which make
     * inject = weff*λ>>16 round to 0 (fast path a no-op). */
    a->model.fast_lambda = (nerva_uq0_16_t)(0.5 * 65535.0);
    a->model.fast_w0 = (nerva_q8_8_t)2048;
    nerva_work_set_rgrad_updates(0);
    a->ready = 1;
    a->loaded_ckpt = 1;
    a->train_n = 0;
    printf("boot_mode=LOAD_CHECKPOINT (no pretrain this launch)\n");
    printf("ckpt=%s ckpt_bytes=%ld words=%s vocab=%u\n", ckpt, sz, words_path, a->words.count);
    printf("boot: order=%u nodes=%u/%u edges=%u/%u (teach headroom loaded)\n", a->model.order,
           a->eng.node_count, a->eng.node_cap, a->eng.edge_count, a->eng.edge_cap);
    printf("fast_lambda=%u fast_w0=%d (lab chat; ckpt λ/w0 were no-op)\n",
           (unsigned)a->model.fast_lambda, (int)a->model.fast_w0);
    printf("graph=fluency  generate=fluency_generate  credit=fluency_pcw_teach_sequence\n");
    organs_enable_if_present(a);
    return 0;
}

/* Tiny smoke train — NOT TinyStories full; only when no ckpt / forced. */
static int app_boot_smoke(App *a) {
    NervaConfig cfg = nerva_config_test();
    char *text = NULL;
    size_t text_len = 0;
    FILE *f;
    long sz;

    cfg.max_nodes = 4096;
    cfg.max_edges = 65536;
    cfg.max_names = 4096;
    cfg.max_events = 512;
    cfg.max_active_nodes = 256;
    cfg.max_fire_log = 64;
    cfg.max_traces = 256;
    cfg.max_mutations = 1024;
    cfg.max_mutation_log = 1024;
    cfg.max_expectations = 8;
    cfg.max_schemas = 8;
    cfg.max_memory_blocks = 8;

    if (nerva_engine_init(&a->eng, cfg) != 0)
        return -1;
    if (fluency_init(&a->model, &a->eng, 3u) != 0) {
        nerva_engine_free(&a->eng);
        return -1;
    }
    fluency_words_init(&a->words);

    f = fopen(SMOKE_CORPUS, "rb");
    if (f) {
        if (fseek(f, 0, SEEK_END) == 0) {
            sz = ftell(f);
            if (sz > 0 && sz < 200000) {
                text = (char *)malloc((size_t)sz + 1u);
                if (text) {
                    fseek(f, 0, SEEK_SET);
                    text_len = fread(text, 1, (size_t)sz, f);
                    text[text_len] = 0;
                }
            }
        }
        fclose(f);
    }
    if (!text) {
        const char *fallback =
            "the cat sat on the mat . the dog ran in the park . "
            "birds sing in the morning sun .";
        text = (char *)malloc(strlen(fallback) + 1);
        if (!text) {
            fluency_free(&a->model);
            nerva_engine_free(&a->eng);
            return -1;
        }
        strcpy(text, fallback);
    }

    a->train_n = fluency_words_encode(&a->words, text, a->train_ids, 4096, 1);
    free(text);
    if (a->train_n < 8) {
        fluency_free(&a->model);
        nerva_engine_free(&a->eng);
        return -1;
    }
    fluency_train(&a->model, a->train_ids, a->train_n);
    a->model.fast_lambda = (nerva_uq0_16_t)(0.25 * 65535.0);
    nerva_work_set_rgrad_updates(0);
    a->ready = 1;
    a->loaded_ckpt = 0;
    a->organs_present = 0;
    printf("boot_mode=SMOKE_TRAIN (not full TinyStories — run make pretrain for ckpt)\n");
    printf("boot: words=%u train_toks=%zu order=%u nodes=%u edges=%u\n", a->words.count, a->train_n,
           a->model.order, a->eng.node_count, a->eng.edge_count);
    printf("graph=fluency  generate=fluency_generate  credit=fluency_pcw_teach_sequence\n");
    /* Organs absent on smoke path — stay off silently. */
    return 0;
}

static int app_boot(App *a, const char *ckpt, const char *words_path, int force_smoke) {
    FILE *cf;
    memset(a, 0, sizeof(*a));
    a->gen_default = 4;
    if (!force_smoke && ckpt && words_path) {
        cf = fopen(ckpt, "rb");
        if (cf) {
            fclose(cf);
            if (app_boot_ckpt(a, ckpt, words_path) == 0)
                return 0;
            printf("boot: ckpt load failed; falling back to smoke train\n");
        }
    }
    return app_boot_smoke(a);
}

static void app_free(App *a) {
    if (!a)
        return;
    if (a->ready) {
        fluency_free(&a->model);
        nerva_engine_free(&a->eng);
    }
    memset(a, 0, sizeof(*a));
}

static int encode_line(App *a, const char *line, flu_tok_t *out, size_t cap, size_t *n_out) {
    size_t n = fluency_words_encode(&a->words, line, out, cap, 1);
    if (n_out)
        *n_out = n;
    return n > 0 ? 0 : -1;
}

static void print_tokens(App *a, const flu_tok_t *toks, size_t n, FILE *out) {
    size_t i;
    for (i = 0; i < n; ++i) {
        const char *w = fluency_words_word(&a->words, toks[i]);
        fprintf(out, "%s%s", i ? " " : "", w ? w : "?");
    }
}

static void hist_push(App *a, const flu_tok_t *toks, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (a->hist_n >= HIST_MAX) {
            memmove(a->hist, a->hist + 1, (HIST_MAX - 1) * sizeof(flu_tok_t));
            a->hist_n = HIST_MAX - 1;
        }
        a->hist[a->hist_n++] = toks[i];
    }
}

static int do_seq(App *a, const char *line, FILE *out) {
    flu_tok_t ids[MAX_TOK];
    size_t n = 0;
    int rc;
    if (encode_line(a, line, ids, MAX_TOK, &n) != 0 || n < 2) {
        fprintf(out, "seq: need ≥2 words\n");
        return -1;
    }
    rc = fluency_pcw_teach_sequence(&a->model, ids, n);
    fprintf(out, "taught (fluency_pcw_teach_sequence) n=%zu rc=%d: ", n, rc);
    print_tokens(a, ids, n, out);
    fprintf(out, "\n");
    mood_observe_toks(a, ids, n, 1.0f);
    /* Condition hist with taught sequence so free-chat can continue. */
    hist_push(a, ids, n);
    return rc < 0 ? -1 : 0;
}

/* fluency_generate only — product multi-token path. */
static int generate_from(App *a, const flu_tok_t *seed, size_t seed_n, size_t n_gen, flu_tok_t *gen,
                         uint64_t *apply_delta) {
    uint64_t a0, a1;
    uint32_t rng_seed = 1u;
    int sample = 0;
    if (!seed || seed_n < 1 || n_gen < 1)
        return -1;
    if (a->sample_on) {
        sample = 1;
        a->rng_counter++;
        rng_seed = a->rng_counter ? a->rng_counter : 1u;
    }
    snapshot_last_ctx(a, seed, seed_n);
    a0 = nerva_work_pcw_apply_count();
    fluency_generate(&a->model, seed, seed_n, gen, n_gen, rng_seed, sample);
    a1 = nerva_work_pcw_apply_count();
    if (apply_delta)
        *apply_delta = a1 - a0;
    return 0;
}

static int do_gen(App *a, const char *seed_word, size_t n_gen, FILE *out) {
    flu_tok_t seed[1];
    flu_tok_t gen[GEN_MAX + 1];
    uint32_t sid;
    uint64_t d = 0;

    if (!seed_word || !seed_word[0] || n_gen < 1 || n_gen > GEN_MAX) {
        fprintf(out, "usage: /gen <seed_word> [n]\n");
        return -1;
    }
    sid = fluency_words_id(&a->words, seed_word, 1);
    if (sid >= FLUENCY_VOCAB) {
        fprintf(out, "gen: bad seed\n");
        return -1;
    }
    seed[0] = (flu_tok_t)sid;
    if (generate_from(a, seed, 1, n_gen, gen, &d) != 0)
        return -1;
    fprintf(out, "nerva> ");
    print_tokens(a, gen, n_gen, out);
    fprintf(out, "\n");
    fprintf(out, "  [reply_source=fluency_generate n=%zu apply_delta=%llu rgrad_off=%d sample=%d]\n",
            n_gen, (unsigned long long)d, !nerva_work_rgrad_updates_enabled(), a->sample_on);
    mood_observe_toks(a, seed, 1, 1.0f);
    mood_observe_toks(a, gen, n_gen, 1.0f / (float)FLU_MOOD_SELF_DIV);
    hist_push(a, seed, 1);
    hist_push(a, gen, n_gen);
    return 0;
}

/* Free-chat: user line → hist → fluency_generate continuation. */
static int do_freechat(App *a, const char *line, FILE *out) {
    flu_tok_t ids[MAX_TOK];
    flu_tok_t seed[FLUENCY_MAX_ORDER];
    flu_tok_t gen[GEN_MAX + 1];
    size_t n = 0, seed_n = 0, take, n_gen;
    uint64_t d = 0;

    if (encode_line(a, line, ids, MAX_TOK, &n) != 0 || n < 1) {
        fprintf(out, "nerva> (no tokens)\n");
        return 0;
    }
    mood_observe_toks(a, ids, n, 1.0f);
    hist_push(a, ids, n);
    take = a->hist_n < a->model.order ? a->hist_n : (size_t)a->model.order;
    if (take < 1)
        take = 1;
    seed_n = take;
    memcpy(seed, a->hist + (a->hist_n - take), take * sizeof(flu_tok_t));
    n_gen = a->gen_default;
    if (n_gen > GEN_MAX)
        n_gen = GEN_MAX;
    if (generate_from(a, seed, seed_n, n_gen, gen, &d) != 0) {
        fprintf(out, "gen failed\n");
        return -1;
    }
    fprintf(out, "nerva> ");
    print_tokens(a, gen, n_gen, out);
    fprintf(out, "\n");
    fprintf(out, "  [reply_source=fluency_generate n=%zu apply_delta=%llu rgrad_off=%d sample=%d]\n",
            n_gen, (unsigned long long)d, !nerva_work_rgrad_updates_enabled(), a->sample_on);
    mood_observe_toks(a, gen, n_gen, 1.0f / (float)FLU_MOOD_SELF_DIV);
    hist_push(a, gen, n_gen);
    return 0;
}

static int match_prefix(const flu_tok_t *gen, size_t n_gen, const flu_tok_t *want, size_t n_want) {
    size_t i;
    if (n_gen < n_want)
        return 0;
    for (i = 0; i < n_want; ++i)
        if (gen[i] != want[i])
            return 0;
    return 1;
}

/*
 * Selfcheck runs with organs in their booted state (on when tables present).
 * Mood register is reset before the teach/generate probe so conversation drift
 * cannot poison seq_match; bars themselves are unchanged.
 */
static int run_selfcheck(App *a, FILE *out) {
    const char *phrase = "alice likes green tea";
    flu_tok_t ids[MAX_TOK];
    flu_tok_t gen[GEN_MAX + 1];
    flu_tok_t seed[1];
    size_t n = 0, n_want;
    int ok = 1, match = 0, rgrad_ok = 0;
    uint64_t a0, a1, r0, d = 0;

    /* Selfcheck requires deterministic greedy generation regardless of --sample. */
    a->sample_on = 0;
    a->hist_n = 0;
    a->last_pred_valid = 0;
    fluency_mood_reset(&a->model);

    fprintf(out, "nerva-train selfcheck — single stack\n");
    fprintf(out, "AUDIT reply_source=fluency_generate (not template)\n");
    fprintf(out, "AUDIT learn=fluency_pcw_teach_sequence; rgrad OFF\n");
    fprintf(out, "AUDIT organs_booted mood_on=%u gain_on=%u hub_on=%u present=%d "
                 "(selfcheck keeps booted state; mood reset before probe)\n",
            (unsigned)a->model.mood_on, (unsigned)a->model.gain_on, (unsigned)a->model.hub_on,
            a->organs_present);

    nerva_work_set_rgrad_updates(0);
    if (a->eng.edge_count > 0) {
        r0 = nerva_work_rgrad_refuse_count();
        nerva_work_set_rgrad_updates(1);
        (void)nerva_work_apply_weight_delta(&a->eng, 0, 1, 0);
        nerva_work_set_rgrad_updates(0);
        rgrad_ok = (nerva_work_rgrad_refuse_count() > r0) ? 1 : 0;
    }
    fprintf(out, "rgrad_updates_enabled=%d rgrad_refuse_probe=%s\n",
            nerva_work_rgrad_updates_enabled(), rgrad_ok ? "OK" : "FAIL");

    if (encode_line(a, phrase, ids, MAX_TOK, &n) != 0 || n < 3) {
        fprintf(out, "FAIL encode phrase\n");
        return 1;
    }
    n_want = n - 1;
    fprintf(out, "teach_sequence n=%zu: ", n);
    print_tokens(a, ids, n, out);
    fprintf(out, "\n");

    if (fluency_pcw_teach_sequence(&a->model, ids, n) < 0) {
        fprintf(out, "FAIL fluency_pcw_teach_sequence\n");
        return 1;
    }

    seed[0] = ids[0];
    a0 = nerva_work_pcw_apply_count();
    if (generate_from(a, seed, 1, n_want, gen, &d) != 0) {
        fprintf(out, "FAIL generate\n");
        return 1;
    }
    a1 = nerva_work_pcw_apply_count();

    fprintf(out, "generate path=fluency_generate seed=");
    print_tokens(a, seed, 1, out);
    fprintf(out, " n_tokens=%zu tokens: ", n_want);
    print_tokens(a, gen, n_want, out);
    fprintf(out, "\n");
    fprintf(out, "generate_no_update: applies_before=%llu after=%llu delta=%llu\n",
            (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)(a1 - a0));

    match = match_prefix(gen, n_want, ids + 1, n_want);
    fprintf(out, "seq_match_prefix match=%d want=", match);
    print_tokens(a, ids + 1, n_want, out);
    fprintf(out, "\n");

    /* Free-chat path also uses fluency_generate only */
    {
        flu_tok_t g2[GEN_MAX + 1];
        uint64_t d2 = 0;
        a->hist_n = 0;
        hist_push(a, ids, 1);
        if (generate_from(a, ids, 1, 3, g2, &d2) != 0)
            ok = 0;
        fprintf(out, "freechat_path_check reply_source=fluency_generate n=3 apply_delta=%llu\n",
                (unsigned long long)d2);
        if (d2 != 0)
            ok = 0;
    }

    if (n_want < 2)
        ok = 0;
    if (!match)
        ok = 0;
    if (a1 != a0 || d != 0)
        ok = 0;
    if (!rgrad_ok || nerva_work_rgrad_updates_enabled())
        ok = 0;
    if (nerva_work_pcw_apply_count() < 1)
        ok = 0;

    fprintf(out, "token_len=%zu match=%d rgrad_off=%d applies=%llu\n", n_want, match,
            !nerva_work_rgrad_updates_enabled(),
            (unsigned long long)nerva_work_pcw_apply_count());
    fprintf(out,
            "\nMACHINE mode=nerva_train reply_source=fluency_generate "
            "token_len=%zu seq_match=%d rgrad_updates=OFF pcw_applies=%llu "
            "query_apply_delta=%llu verdict=%s\n",
            n_want, match, (unsigned long long)nerva_work_pcw_apply_count(),
            (unsigned long long)(a1 - a0), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int handle_line(App *a, char *line, FILE *out) {
    size_t L;
    while (*line == ' ' || *line == '\t')
        line++;
    L = strlen(line);
    while (L > 0 && (line[L - 1] == '\n' || line[L - 1] == '\r' || line[L - 1] == ' '))
        line[--L] = 0;
    if (L == 0)
        return 1;
    if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0)
        return 0;
    if (strcmp(line, "/help") == 0) {
        usage("nerva");
        return 1;
    }
    if (strcmp(line, "/hist") == 0) {
        fprintf(out, "hist n=%zu: ", a->hist_n);
        print_tokens(a, a->hist, a->hist_n, out);
        fprintf(out, "\n");
        return 1;
    }
    if (strcmp(line, "/clear") == 0) {
        a->hist_n = 0;
        a->last_pred_valid = 0;
        fluency_mood_reset(&a->model);
        fprintf(out, "hist cleared; mood reset\n");
        return 1;
    }
    if (strcmp(line, "/sample") == 0) {
        a->sample_on = a->sample_on ? 0 : 1;
        fprintf(out, "sample=%s\n", a->sample_on ? "on" : "off");
        return 1;
    }
    if (strcmp(line, "/why") == 0) {
        FluencyWhy w;
        if (!a->last_pred_valid || a->last_ctx_len == 0u) {
            fprintf(out, "why: (no prediction to explain — generate first)\n");
            return 1;
        }
        if (fluency_explain(&a->model, a->last_ctx, a->last_ctx_len, &w) != 0) {
            fprintf(out, "why: explain failed\n");
            return 1;
        }
        fluency_why_print(&w, out, app_word_fn, &a->words);
        return 1;
    }
    if (strcmp(line, "/mood") == 0) {
        fluency_mood_print(&a->model, out, app_word_fn, &a->words);
        return 1;
    }
    if (strncmp(line, "/hubs ", 6) == 0) {
        uint32_t sid = fluency_words_id(&a->words, line + 6, 0);
        if (sid >= FLUENCY_VOCAB) {
            fprintf(out, "hubs: unknown word\n");
            return 1;
        }
        fluency_hubs_print(&a->model, (flu_tok_t)sid, out, app_word_fn, &a->words);
        return 1;
    }
    if (strcmp(line, "/hubs") == 0) {
        fprintf(out, "usage: /hubs <word>\n");
        return 1;
    }
    if (strncmp(line, "/organs", 7) == 0) {
        const char *arg = line + 7;
        while (*arg == ' ' || *arg == '\t')
            arg++;
        if (*arg == '\0') {
            fprintf(out,
                    "organs present=%d mood_on=%u gain_on=%u hub_on=%u mood_k=%u "
                    "gain_topk=%u hub_h=%u table_bytes=%zu\n",
                    a->organs_present, (unsigned)a->model.mood_on, (unsigned)a->model.gain_on,
                    (unsigned)a->model.hub_on, (unsigned)a->model.mood_k,
                    (unsigned)a->model.gain_topk, (unsigned)a->model.hub_h,
                    organs_table_bytes(&a->model));
            return 1;
        }
        if (!a->organs_present) {
            fprintf(out, "organs: tables absent (build with pretrain --organs)\n");
            return 1;
        }
        if (strcmp(arg, "on") == 0) {
            fluency_mood_set(&a->model, 1);
            fluency_gain_set(&a->model, 1);
            fluency_hubs_set(&a->model, 1);
            fprintf(out, "organs=on\n");
        } else if (strcmp(arg, "off") == 0) {
            fluency_mood_set(&a->model, 0);
            fluency_gain_set(&a->model, 0);
            fluency_hubs_set(&a->model, 0);
            fprintf(out, "organs=off\n");
        } else {
            fprintf(out, "usage: /organs [on|off]\n");
        }
        return 1;
    }
    if (strncmp(line, "/seq ", 5) == 0) {
        (void)do_seq(a, line + 5, out);
        return 1;
    }
    if (strncmp(line, "/teach ", 7) == 0) {
        (void)do_seq(a, line + 7, out);
        return 1;
    }
    if (strncmp(line, "/gen ", 5) == 0) {
        char seed[64];
        unsigned n = (unsigned)a->gen_default;
        if (sscanf(line + 5, "%63s %u", seed, &n) < 1) {
            fprintf(out, "usage: /gen <seed> [n]\n");
            return 1;
        }
        if (n < 1)
            n = 1;
        if (n > GEN_MAX)
            n = GEN_MAX;
        (void)do_gen(a, seed, n, out);
        return 1;
    }
    if (strcmp(line, "/seq") == 0 || strcmp(line, "/teach") == 0) {
        fprintf(out, "usage: /seq w1 w2 w3...\n");
        return 1;
    }
    if (strcmp(line, "/gen") == 0) {
        fprintf(out, "usage: /gen <seed> [n]\n");
        return 1;
    }
    if (line[0] == '/') {
        fprintf(out, "unknown command (try /help)\n");
        return 1;
    }
    /* Free chat — only fluency_generate */
    (void)do_freechat(a, line, out);
    return 1;
}

static int run_script(App *a, const char *path, FILE *out) {
    FILE *f = fopen(path, "rb");
    char line[512];
    if (!f) {
        fprintf(out, "script open failed: %s\n", path);
        return 1;
    }
    while (fgets(line, sizeof(line), f)) {
        if (!handle_line(a, line, out))
            break;
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    App *app;
    int selfcheck = 0;
    int force_smoke = 0;
    int sample_flag = 0;
    const char *script = NULL;
    const char *ckpt = DEFAULT_CKPT;
    const char *words_path = DEFAULT_WORDS;
    int i;
    char line[512];
    int rc = 0;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--selfcheck") == 0) {
            selfcheck = 1;
        } else if (strcmp(argv[i], "--force-smoke-train") == 0) {
            force_smoke = 1;
        } else if (strcmp(argv[i], "--sample") == 0) {
            sample_flag = 1;
        } else if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) {
            ckpt = argv[++i];
        } else if (strcmp(argv[i], "--words") == 0 && i + 1 < argc) {
            words_path = argv[++i];
        } else if (strcmp(argv[i], "--script") == 0 && i + 1 < argc) {
            script = argv[++i];
        } else {
            fprintf(stderr, "unknown: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    app = (App *)calloc(1, sizeof(*app));
    if (!app) {
        fprintf(stderr, "OOM\n");
        return 1;
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    printf("nerva-train — teach + chat (fluency + PCW)\n");
    if (app_boot(app, ckpt, words_path, force_smoke) != 0) {
        fprintf(stderr, "boot failed\n");
        free(app);
        return 1;
    }
    if (sample_flag)
        app->sample_on = 1;
    printf("rgrad_updates_enabled=%d (must be 0) loaded_ckpt=%d sample=%d\n",
           nerva_work_rgrad_updates_enabled(), app->loaded_ckpt, app->sample_on);
    printf("commands: /seq /teach /gen /sample /hist /clear /why /mood /hubs /organs /quit  |  "
           "free text\n");

    if (selfcheck) {
        rc = run_selfcheck(app, stdout);
        printf("boot_was_load_ckpt=%d\n", app->loaded_ckpt);
        app_free(app);
        free(app);
        return rc;
    }
    if (script) {
        rc = run_script(app, script, stdout);
        app_free(app);
        free(app);
        return rc;
    }
    while (1) {
        printf("you> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin))
            break;
        if (!handle_line(app, line, stdout))
            break;
    }
    app_free(app);
    free(app);
    return 0;
}
