/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Paul Odantabao II
 *
 * Full TinyStories pretrain → frozen checkpoint (fluency session + words).
 *
 *   make pretrain
 *   ./build/pretrain.exe
 *
 * Streams the entire train file (does not load whole text into one string).
 * Writes:
 *   checkpoints/tinystories.sess   fluency_save
 *   checkpoints/tinystories.words  word lexicon
 *   checkpoints/tinystories.meta   human-readable stats
 */

#include "fluency.h"
#include "fluency_session.h"
#include "nerva_config.h"
#include "nerva_engine.h"
#include "nerva_words_io.h"
#include "nerva_work.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#define WORD_LEN FLUENCY_WORD_LEN
#define CHUNK_TOKS 50000u
#define DEFAULT_CORPUS "data/tinystories/TinyStories-train.txt"
#define DEFAULT_CKPT "checkpoints/tinystories.sess"
#define DEFAULT_WORDS "checkpoints/tinystories.words"
#define DEFAULT_META "checkpoints/tinystories.meta"
#define VOCAB_CAP FLUENCY_VOCAB
/* Periodic partial checkpoint so a cancelled run keeps its progress. */
#ifndef SAVE_EVERY_TOKS
#define SAVE_EVERY_TOKS 25000000ull
#endif
/* Open-addressing: load ~0.5 → 400k uniques need ~1M slots. */
#define FREQ_SLOTS (1u << 20)
#define ID_SLOTS (1u << 15) /* 32k slots, vocab ≤ 16k */
#define IO_BUF (1u << 20)

typedef struct {
    char w[WORD_LEN + 1];
    uint32_t n;
    uint8_t used;
} FreqSlot;

typedef struct {
    char w[WORD_LEN + 1];
    flu_tok_t id;
    uint8_t used;
} IdSlot;

typedef struct {
    FILE *f;
    unsigned char *buf;
    size_t cap;
    size_t len;
    size_t pos;
    int eof;
} BufIn;

static size_t process_rss(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (size_t)pmc.WorkingSetSize;
#endif
    return 0;
}

static double wall_sec(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int init;
    LARGE_INTEGER c;
    if (!init) {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

static int buf_open(BufIn *b, FILE *f) {
    b->f = f;
    b->cap = IO_BUF;
    b->buf = (unsigned char *)malloc(b->cap);
    b->len = b->pos = 0;
    b->eof = 0;
    return b->buf ? 0 : -1;
}

static void buf_close(BufIn *b) {
    free(b->buf);
    b->buf = NULL;
}

static int buf_getc(BufIn *b) {
    if (b->pos >= b->len) {
        if (b->eof)
            return EOF;
        b->len = fread(b->buf, 1, b->cap, b->f);
        b->pos = 0;
        if (b->len == 0) {
            b->eof = 1;
            return EOF;
        }
    }
    return (int)b->buf[b->pos++];
}

static void buf_ungetc(BufIn *b) {
    if (b->pos > 0)
        b->pos--;
}

/* Alpha / apostrophe tokens, lowercased. */
static int next_word(BufIn *b, char *out) {
    int c;
    size_t L = 0;
    do {
        c = buf_getc(b);
        if (c == EOF)
            return 0;
    } while (!isalpha(c));
    while (c != EOF && L < WORD_LEN) {
        if (isalpha(c)) {
            out[L++] = (char)tolower((unsigned char)c);
            c = buf_getc(b);
        } else if (c == '\'' && L > 0) {
            out[L++] = '\'';
            c = buf_getc(b);
        } else {
            if (c != EOF)
                buf_ungetc(b);
            break;
        }
    }
    out[L] = 0;
    return L > 0;
}

static uint32_t str_hash(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h ^= (uint32_t)(unsigned char)*s++;
        h *= 16777619u;
    }
    return h;
}

/* Returns 0 if counted, -1 if table full (new unique dropped). */
static int freq_add(FreqSlot *tab, uint32_t *nused, const char *w) {
    uint32_t h = str_hash(w);
    uint32_t i;
    for (i = 0; i < FREQ_SLOTS; ++i) {
        uint32_t idx = (h + i) & (FREQ_SLOTS - 1u);
        if (!tab[idx].used) {
            if (*nused >= (FREQ_SLOTS * 7u) / 10u)
                return -1;
            strncpy(tab[idx].w, w, WORD_LEN);
            tab[idx].w[WORD_LEN] = 0;
            tab[idx].n = 1;
            tab[idx].used = 1;
            (*nused)++;
            return 0;
        }
        if (strcmp(tab[idx].w, w) == 0) {
            tab[idx].n++;
            return 0;
        }
    }
    return -1;
}

static int cmp_freq_desc(const void *a, const void *b) {
    const FreqSlot *x = (const FreqSlot *)a;
    const FreqSlot *y = (const FreqSlot *)b;
    if (x->used != y->used)
        return (int)y->used - (int)x->used;
    if (!x->used)
        return 0;
    if (y->n < x->n)
        return -1;
    if (y->n > x->n)
        return 1;
    return strcmp(x->w, y->w);
}

static void idmap_put(IdSlot *tab, const char *w, flu_tok_t id) {
    uint32_t h = str_hash(w);
    uint32_t i;
    for (i = 0; i < ID_SLOTS; ++i) {
        uint32_t idx = (h + i) & (ID_SLOTS - 1u);
        if (!tab[idx].used) {
            strncpy(tab[idx].w, w, WORD_LEN);
            tab[idx].w[WORD_LEN] = 0;
            tab[idx].id = id;
            tab[idx].used = 1;
            return;
        }
        if (strcmp(tab[idx].w, w) == 0) {
            tab[idx].id = id;
            return;
        }
    }
}

static flu_tok_t idmap_get(const IdSlot *tab, const char *w) {
    uint32_t h = str_hash(w);
    uint32_t i;
    for (i = 0; i < ID_SLOTS; ++i) {
        uint32_t idx = (h + i) & (ID_SLOTS - 1u);
        if (!tab[idx].used)
            return 0;
        if (strcmp(tab[idx].w, w) == 0)
            return tab[idx].id;
    }
    return 0;
}

#define DEFAULT_ORGANS_TOKS 50000000ull
/* Re-registered after 2M proved infeasible (~10ms/predict × 2M × 2 ≈ 11h).
 * Decision made before any eval numbers existed. Last N of corpus; disjoint
 * from the 50M organ-build slice at the start. */
#define DEFAULT_EVAL_TOKS 200000ull
#define EVAL_PROGRESS_EVERY 20000ull

static IdSlot *idmap_from_words(const FluencyWords *W) {
    IdSlot *idmap;
    uint32_t k;
    idmap = (IdSlot *)calloc(ID_SLOTS, sizeof(IdSlot));
    if (!idmap)
        return NULL;
    for (k = 1; k < W->count; ++k)
        idmap_put(idmap, W->word[k], (flu_tok_t)k);
    return idmap;
}

/* Encode up to `cap` tokens starting after `skip` corpus words. OOV → 0. */
static size_t stream_encode(const char *corpus, const IdSlot *idmap, uint64_t skip, flu_tok_t *out,
                            size_t cap, uint64_t *oov_out) {
    FILE *f;
    BufIn bin;
    char wbuf[WORD_LEN + 1];
    uint64_t skipped = 0, oov = 0;
    size_t n = 0;
    f = fopen(corpus, "rb");
    if (!f)
        return 0;
    if (buf_open(&bin, f) != 0) {
        fclose(f);
        return 0;
    }
    while (skipped < skip && next_word(&bin, wbuf))
        skipped++;
    while (n < cap && next_word(&bin, wbuf)) {
        flu_tok_t id = idmap_get(idmap, wbuf);
        if (id == 0)
            oov++;
        out[n++] = id;
    }
    buf_close(&bin);
    fclose(f);
    if (oov_out)
        *oov_out = oov;
    return n;
}

static uint64_t count_corpus_words(const char *corpus) {
    FILE *f;
    BufIn bin;
    char wbuf[WORD_LEN + 1];
    uint64_t n = 0;
    f = fopen(corpus, "rb");
    if (!f)
        return 0;
    if (buf_open(&bin, f) != 0) {
        fclose(f);
        return 0;
    }
    while (next_word(&bin, wbuf))
        n++;
    buf_close(&bin);
    fclose(f);
    return n;
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

static long file_size_bytes(const char *path) {
    FILE *f = fopen(path, "rb");
    long sz = -1;
    if (!f)
        return -1;
    if (fseek(f, 0, SEEK_END) == 0)
        sz = ftell(f);
    fclose(f);
    return sz;
}

/* Offline organs build: load frozen ckpt, stream budget tokens, build mood+gain+hubs,
 * atomic replace of ckpt (tmp → verify load → rename). Does not retrain n-grams. */
static int run_organs_build(const char *corpus, const char *ckpt, const char *words_path,
                            uint64_t budget) {
    NervaEngine eng;
    FluencyModel model;
    FluencyWords *W = NULL;
    IdSlot *idmap = NULL;
    flu_tok_t *toks = NULL;
    size_t n = 0;
    uint64_t oov = 0;
    double t0, wall;
    char tmp_path[1100];
    int rc_org, rc_hub;
    size_t table_bytes;
    long ckpt_bytes;
    uint32_t nodes0, edges0;

    memset(&eng, 0, sizeof(eng));
    memset(&model, 0, sizeof(model));
    snprintf(tmp_path, sizeof(tmp_path), "%s.organs.tmp", ckpt);

    printf("=== organs build (no n-gram retrain) ===\n");
    printf("ckpt=%s corpus=%s budget_toks=%llu\n", ckpt, corpus, (unsigned long long)budget);
    t0 = wall_sec();
    printf("loading checkpoint (zero headroom)...\n");
    if (fluency_session_load(&eng, &model, ckpt, NULL, 0, NULL) != 0) {
        fprintf(stderr, "FAIL fluency_session_load %s\n", ckpt);
        return 1;
    }
    nodes0 = eng.node_count;
    edges0 = eng.edge_count;
    printf("loaded nodes=%u edges=%u order=%u wall_s=%.1f\n", nodes0, edges0, model.order,
           wall_sec() - t0);

    W = (FluencyWords *)calloc(1, sizeof(FluencyWords));
    if (!W || nerva_words_load(W, words_path) != 0) {
        fprintf(stderr, "FAIL words load %s\n", words_path);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    idmap = idmap_from_words(W);
    if (!idmap) {
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }

    toks = (flu_tok_t *)malloc(sizeof(flu_tok_t) * (size_t)budget);
    if (!toks) {
        fprintf(stderr, "FAIL OOM tok buffer (%llu)\n", (unsigned long long)budget);
        free(idmap);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    printf("encoding organ-build slice (first %llu tokens)...\n", (unsigned long long)budget);
    n = stream_encode(corpus, idmap, 0, toks, (size_t)budget, &oov);
    printf("encoded n=%zu oov=%llu rss_MiB=%.1f\n", n, (unsigned long long)oov,
           (double)process_rss() / (1024.0 * 1024.0));
    if (n < 10000u) {
        fprintf(stderr, "FAIL organ slice too small n=%zu\n", n);
        free(toks);
        free(idmap);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }

    printf("fluency_ctx_organs_build mood_k=%u cooc_topk=%u...\n",
           (unsigned)FLU_MOOD_K_DEFAULT, (unsigned)FLU_GAIN_TOPK_DEFAULT);
    rc_org = fluency_ctx_organs_build(&model, toks, n, FLU_MOOD_K_DEFAULT, FLU_GAIN_TOPK_DEFAULT);
    if (rc_org != 0) {
        fprintf(stderr, "FAIL fluency_ctx_organs_build rc=%d\n", rc_org);
        free(toks);
        free(idmap);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    /* Hubs use ±FLU_HUB_WINDOW cooc (≠ gain ±2); second pass over same buffer. */
    printf("fluency_hubs_build n_hubs=%u cooc_topk=%u...\n", (unsigned)FLU_HUB_H_DEFAULT,
           (unsigned)FLU_HUB_COOC_TOPK_DEFAULT);
    rc_hub = fluency_hubs_build(&model, toks, n, FLU_HUB_H_DEFAULT, FLU_HUB_COOC_TOPK_DEFAULT);
    if (rc_hub != 0) {
        fprintf(stderr, "FAIL fluency_hubs_build rc=%d\n", rc_hub);
        free(toks);
        free(idmap);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    free(toks);
    toks = NULL;
    /* Leave organs OFF in the file; product enables on boot when tables present. */
    fluency_mood_set(&model, 0);
    fluency_gain_set(&model, 0);
    fluency_hubs_set(&model, 0);
    fluency_mood_reset(&model);
    table_bytes = organs_table_bytes(&model);

    printf("saving tmp=%s (nodes=%u edges=%u — must match load)...\n", tmp_path, eng.node_count,
           eng.edge_count);
    if (eng.node_count != nodes0 || eng.edge_count != edges0) {
        fprintf(stderr, "FAIL graph mutated during organs build nodes %u→%u edges %u→%u\n", nodes0,
                eng.node_count, edges0, eng.edge_count);
        free(idmap);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    if (fluency_session_save(&model, tmp_path, NULL, 0) != 0) {
        fprintf(stderr, "FAIL fluency_session_save %s\n", tmp_path);
        free(idmap);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    fluency_free(&model);
    nerva_engine_free(&eng);

    /* Verify tmp loads before replacing the 5h pretrain artifact. */
    {
        NervaEngine e2;
        FluencyModel m2;
        memset(&e2, 0, sizeof(e2));
        memset(&m2, 0, sizeof(m2));
        printf("verify-load %s...\n", tmp_path);
        if (fluency_session_load(&e2, &m2, tmp_path, NULL, 0, NULL) != 0) {
            fprintf(stderr, "FAIL verify load of organs tmp\n");
            remove(tmp_path);
            free(idmap);
            free(W);
            return 1;
        }
        if (e2.node_count != nodes0 || e2.edge_count != edges0 || !m2.mood_aff || !m2.gain_cooc ||
            m2.hub_h == 0u) {
            fprintf(stderr,
                    "FAIL verify: nodes=%u edges=%u mood_aff=%p gain=%p hub_h=%u\n", e2.node_count,
                    e2.edge_count, (void *)m2.mood_aff, (void *)m2.gain_cooc, (unsigned)m2.hub_h);
            fluency_free(&m2);
            nerva_engine_free(&e2);
            remove(tmp_path);
            free(idmap);
            free(W);
            return 1;
        }
        fluency_free(&m2);
        nerva_engine_free(&e2);
    }

    remove(ckpt);
    if (rename(tmp_path, ckpt) != 0) {
        fprintf(stderr, "FAIL rename %s → %s (tmp left intact)\n", tmp_path, ckpt);
        free(idmap);
        free(W);
        return 1;
    }
    ckpt_bytes = file_size_bytes(ckpt);
    wall = wall_sec() - t0;
    printf("organs ckpt replaced path=%s ckpt_bytes=%ld table_bytes=%zu\n", ckpt, ckpt_bytes,
           table_bytes);
    printf("MACHINE organs_build tokens=%zu mood_k=%u cooc_topk=%u n_hubs=%u table_bytes=%zu "
           "wall_s=%.3f ckpt_bytes=%ld nodes=%u edges=%u verdict=PASS\n",
           n, (unsigned)FLU_MOOD_K_DEFAULT, (unsigned)FLU_GAIN_TOPK_DEFAULT,
           (unsigned)FLU_HUB_H_DEFAULT, table_bytes, wall, ckpt_bytes, nodes0, edges0);
    free(idmap);
    free(W);
    return 0;
}

/* Same semantics as fluency_evaluate, with progress every EVAL_PROGRESS_EVERY. */
static void organs_evaluate_progress(FluencyModel *m, const flu_tok_t *text, size_t len,
                                     FluencyMetrics *out, const char *label) {
    double *dist;
    double uni_denom;
    double t_pass;
    uint64_t mut_before;
    uint32_t nodes_before, edges_before;
    size_t i;

    if (!m || !text || !out)
        return;
    memset(out, 0, sizeof(*out));
    mut_before = m->e->debug.mutations_applied;
    nodes_before = m->e->node_count;
    edges_before = m->e->edge_count;
    dist = (double *)calloc((size_t)FLUENCY_VOCAB, sizeof(double));
    if (!dist)
        return;
    uni_denom = (double)m->unigram_total + (double)m->vocab_count;
    t_pass = wall_sec();

    for (i = 0; i < len; ++i) {
        flu_tok_t actual = text[i];
        size_t ctx_len = i;
        int pred = fluency_predict(m, text, ctx_len, dist);
        double p = dist[actual];
        double p_uni;

        if (p <= 0.0)
            p = 0.5 / (uni_denom > 0.0 ? uni_denom : 1.0);
        out->nll_sum += -log2(p);
        p_uni = ((double)m->unigram[actual] + 1.0) / (uni_denom > 0.0 ? uni_denom : 1.0);
        out->eval_positions++;
        if (pred == (int)actual)
            out->correct_argmax++;
        out->unigram_perplexity += -log2(p_uni);

        if (i > 0u)
            fluency_fast_deposit(m, text[i - 1u], actual);
        fluency_stream_advance(m);

        if (((i + 1u) % (size_t)EVAL_PROGRESS_EVERY) == 0u || (i + 1u) == len) {
            double npos = (double)(out->eval_positions ? out->eval_positions : 1u);
            double run_ppl = pow(2.0, out->nll_sum / npos);
            double run_acc = (double)out->correct_argmax / npos;
            printf("PROGRESS organs_eval label=%s done=%zu/%zu run_ppl=%.4f run_acc=%.4f "
                   "wall_s=%.1f\n",
                   label ? label : "?", i + 1u, len, run_ppl, run_acc, wall_sec() - t_pass);
            fflush(stdout);
        }
    }

    {
        double npos = (double)(out->eval_positions ? out->eval_positions : 1u);
        out->model_perplexity = pow(2.0, out->nll_sum / npos);
        out->unigram_perplexity = pow(2.0, out->unigram_perplexity / npos);
    }
    out->eval_mutations = m->e->debug.mutations_applied - mut_before;
    out->eval_node_growth = m->e->node_count - nodes_before;
    out->eval_edge_growth = m->e->edge_count - edges_before;
    free(dist);
}

/* eval_pass: 0=off, 1=on, 2=both (contaminated — prefer separate processes). */
static int run_organs_eval(const char *corpus, const char *ckpt, const char *words_path,
                           uint64_t eval_toks, int eval_pass) {
    NervaEngine eng;
    FluencyModel model;
    FluencyWords *W = NULL;
    IdSlot *idmap = NULL;
    flu_tok_t *toks = NULL;
    size_t n = 0;
    uint64_t total, skip, oov = 0;
    double t0, t_pass, wall;
    FluencyMetrics met;
    int32_t hub_margin_save;
    int do_off = (eval_pass == 0 || eval_pass == 2);
    int do_on = (eval_pass == 1 || eval_pass == 2);
    const char *pass_name =
        (eval_pass == 0) ? "off" : (eval_pass == 1) ? "on" : "both";

    memset(&eng, 0, sizeof(eng));
    memset(&model, 0, sizeof(model));
    printf("=== organs eval (held-out last %llu toks, eval_pass=%s) ===\n",
           (unsigned long long)eval_toks, pass_name);
    if (eval_pass == 2) {
        printf("WARN CONTAMINATED: eval_pass=both runs ON after OFF on the same in-memory "
               "model; fast-edge deposits from OFF warm the ON pass. Registered protocol is "
               "two separate processes (--eval-pass off then --eval-pass on).\n");
        fflush(stdout);
    }
    t0 = wall_sec();
    if (fluency_session_load(&eng, &model, ckpt, NULL, 0, NULL) != 0) {
        fprintf(stderr, "FAIL load %s\n", ckpt);
        return 1;
    }
    if (!model.mood_aff || !model.gain_cooc || model.hub_h == 0u) {
        fprintf(stderr, "FAIL checkpoint missing organ tables (run --organs first)\n");
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    W = (FluencyWords *)calloc(1, sizeof(FluencyWords));
    if (!W || nerva_words_load(W, words_path) != 0) {
        fprintf(stderr, "FAIL words %s\n", words_path);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    idmap = idmap_from_words(W);
    if (!idmap) {
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    printf("counting corpus words...\n");
    total = count_corpus_words(corpus);
    if (total <= eval_toks) {
        fprintf(stderr, "FAIL corpus too small total=%llu need >%llu\n", (unsigned long long)total,
                (unsigned long long)eval_toks);
        free(idmap);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    skip = total - eval_toks;
    toks = (flu_tok_t *)malloc(sizeof(flu_tok_t) * (size_t)eval_toks);
    if (!toks) {
        free(idmap);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }
    printf("encoding held-out slice skip=%llu...\n", (unsigned long long)skip);
    n = stream_encode(corpus, idmap, skip, toks, (size_t)eval_toks, &oov);
    printf("eval_n=%zu oov=%llu\n", n, (unsigned long long)oov);
    if (n < 1000u) {
        fprintf(stderr, "FAIL eval slice too small\n");
        free(toks);
        free(idmap);
        free(W);
        fluency_free(&model);
        nerva_engine_free(&eng);
        return 1;
    }

    /* Match product fast path so eval is not λ≈0 no-op. */
    model.fast_lambda = (nerva_uq0_16_t)(0.5 * 65535.0);
    model.fast_w0 = (nerva_q8_8_t)2048;
    nerva_work_set_rgrad_updates(0);

    if (do_off) {
        fluency_mood_set(&model, 0);
        fluency_gain_set(&model, 0);
        fluency_hubs_set(&model, 0);
        fluency_mood_reset(&model);
        model.mood_inject_ops = 0;
        model.gain_inject_ops = 0;
        model.hub_dial_count = 0;
        model.hub_dial_ops = 0;
        model.hub_predict_count = 0;
        printf("eval organs=OFF (fresh load this process)...\n");
        fflush(stdout);
        t_pass = wall_sec();
        organs_evaluate_progress(&model, toks, n, &met, "OFF");
        printf("MACHINE organs_eval organs=OFF ppl=%.6f acc=%.6f positions=%llu "
               "mood_inject_ops=%llu gain_inject_ops=%llu hub_dial_count=%llu "
               "hub_predict_count=%llu wall_s=%.3f eval_pass=%s\n",
               met.model_perplexity,
               met.eval_positions ? (double)met.correct_argmax / (double)met.eval_positions : 0.0,
               (unsigned long long)met.eval_positions, (unsigned long long)model.mood_inject_ops,
               (unsigned long long)model.gain_inject_ops, (unsigned long long)model.hub_dial_count,
               (unsigned long long)model.hub_predict_count, wall_sec() - t_pass, pass_name);
        fflush(stdout);
    }

    if (do_on) {
        if (eval_pass == 2) {
            printf("WARN: starting ON after OFF without reload — fast tables warmed on "
                   "eval slice (CONTAMINATED).\n");
            fflush(stdout);
        }
        fluency_mood_set(&model, 1);
        fluency_gain_set(&model, 1);
        fluency_hubs_set(&model, 1);
        fluency_mood_reset(&model);
        model.mood_inject_ops = 0;
        model.gain_inject_ops = 0;
        model.hub_dial_count = 0;
        model.hub_dial_ops = 0;
        model.hub_predict_count = 0;
        printf("eval organs=ON hub_conf_margin=%d (fresh load this process=%s)...\n",
               (int)model.hub_conf_margin, eval_pass == 1 ? "yes" : "NO");
        fflush(stdout);
        t_pass = wall_sec();
        organs_evaluate_progress(&model, toks, n, &met, "ON");
        printf("MACHINE organs_eval organs=ON ppl=%.6f acc=%.6f positions=%llu "
               "mood_inject_ops=%llu gain_inject_ops=%llu hub_dial_count=%llu "
               "hub_predict_count=%llu wall_s=%.3f eval_pass=%s\n",
               met.model_perplexity,
               met.eval_positions ? (double)met.correct_argmax / (double)met.eval_positions : 0.0,
               (unsigned long long)met.eval_positions, (unsigned long long)model.mood_inject_ops,
               (unsigned long long)model.gain_inject_ops, (unsigned long long)model.hub_dial_count,
               (unsigned long long)model.hub_predict_count, wall_sec() - t_pass, pass_name);
        fflush(stdout);

        /* Exploratory only on single ON pass: if hubs never dialed, raise margin. */
        if (eval_pass == 1 && model.hub_dial_count == 0ull) {
            FluencyMetrics met_x;
            hub_margin_save = model.hub_conf_margin;
            model.hub_conf_margin = 256;
            fluency_mood_reset(&model);
            model.mood_inject_ops = 0;
            model.gain_inject_ops = 0;
            model.hub_dial_count = 0;
            model.hub_dial_ops = 0;
            model.hub_predict_count = 0;
            printf("EXPLORATORY organs=ON hub_conf_margin=%d (not registered)...\n",
                   (int)model.hub_conf_margin);
            fflush(stdout);
            t_pass = wall_sec();
            organs_evaluate_progress(&model, toks, n, &met_x, "ON_EXPLORATORY_HUB");
            printf("MACHINE organs_eval organs=ON_EXPLORATORY_HUB_MARGIN ppl=%.6f acc=%.6f "
                   "positions=%llu mood_inject_ops=%llu gain_inject_ops=%llu hub_dial_count=%llu "
                   "hub_predict_count=%llu hub_conf_margin=%d wall_s=%.3f\n",
                   met_x.model_perplexity,
                   met_x.eval_positions
                       ? (double)met_x.correct_argmax / (double)met_x.eval_positions
                       : 0.0,
                   (unsigned long long)met_x.eval_positions,
                   (unsigned long long)model.mood_inject_ops,
                   (unsigned long long)model.gain_inject_ops,
                   (unsigned long long)model.hub_dial_count,
                   (unsigned long long)model.hub_predict_count, (int)model.hub_conf_margin,
                   wall_sec() - t_pass);
            model.hub_conf_margin = hub_margin_save;
            fflush(stdout);
        }
    }

    wall = wall_sec() - t0;
    printf("organs_eval done wall_s=%.3f eval_pass=%s\n", wall, pass_name);
    free(toks);
    free(idmap);
    free(W);
    fluency_free(&model);
    nerva_engine_free(&eng);
    return 0;
}

static void usage(const char *p) {
    printf("Usage: %s [--corpus PATH] [--ckpt PATH] [--words PATH] [--meta PATH] [--fresh]\n",
           p);
    printf("       %s --organs [--organs-toks N] [...paths]\n", p);
    printf("       %s --organs-eval [--eval-toks N] [--eval-pass off|on|both] [...paths]\n", p);
    printf("  Full TinyStories stream pretrain → frozen checkpoint.\n");
    printf("  --organs       build mood/gain/hub tables into existing ckpt (no retrain).\n");
    printf("  --organs-eval  held-out last N toks (default %llu).\n",
           (unsigned long long)DEFAULT_EVAL_TOKS);
    printf("  --eval-pass    off|on|both (default both). Registered: separate processes "
           "off then on.\n");
    printf("  Auto-resumes from <ckpt>.partial if present; --fresh ignores it.\n");
    printf("  default corpus=%s\n", DEFAULT_CORPUS);
    printf("  default ckpt=%s\n", DEFAULT_CKPT);
}

int main(int argc, char **argv) {
    const char *corpus = DEFAULT_CORPUS;
    const char *ckpt = DEFAULT_CKPT;
    const char *words_path = DEFAULT_WORDS;
    const char *meta_path = DEFAULT_META;
    int i;
    FILE *f;
    FreqSlot *tab = NULL;
    IdSlot *idmap = NULL;
    uint32_t nused = 0;
    uint64_t total_words = 0, dropped = 0;
    char wbuf[WORD_LEN + 1];
    double t0, t1;
    FluencyWords *W = NULL;
    NervaEngine eng;
    FluencyModel model;
    NervaConfig cfg;
    flu_tok_t *chunk = NULL;
    size_t cn = 0;
    uint64_t trained = 0, oov = 0;
    uint64_t last_ckpt = 0;
    char part_path[1024];
    char part_meta[1040];
    int resume = 0, force_fresh = 0;
    uint64_t resume_toks = 0;
    long corpus_bytes = 0;
    BufIn bin;
    int do_organs = 0, do_organs_eval = 0;
    uint64_t organs_toks = DEFAULT_ORGANS_TOKS;
    uint64_t eval_toks = DEFAULT_EVAL_TOKS;
    int eval_pass = 2; /* 0=off, 1=on, 2=both */

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc)
            corpus = argv[++i];
        else if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc)
            ckpt = argv[++i];
        else if (strcmp(argv[i], "--words") == 0 && i + 1 < argc)
            words_path = argv[++i];
        else if (strcmp(argv[i], "--meta") == 0 && i + 1 < argc)
            meta_path = argv[++i];
        else if (strcmp(argv[i], "--fresh") == 0)
            force_fresh = 1;
        else if (strcmp(argv[i], "--organs") == 0)
            do_organs = 1;
        else if (strcmp(argv[i], "--organs-eval") == 0)
            do_organs_eval = 1;
        else if (strcmp(argv[i], "--organs-toks") == 0 && i + 1 < argc)
            organs_toks = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--eval-toks") == 0 && i + 1 < argc)
            eval_toks = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--eval-pass") == 0 && i + 1 < argc) {
            const char *ep = argv[++i];
            if (strcmp(ep, "off") == 0)
                eval_pass = 0;
            else if (strcmp(ep, "on") == 0)
                eval_pass = 1;
            else if (strcmp(ep, "both") == 0)
                eval_pass = 2;
            else {
                fprintf(stderr, "unknown --eval-pass %s (want off|on|both)\n", ep);
                usage(argv[0]);
                return 2;
            }
        } else {
            fprintf(stderr, "unknown %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    if (do_organs && do_organs_eval) {
        fprintf(stderr, "FAIL: use --organs or --organs-eval, not both\n");
        return 2;
    }
    if (do_organs)
        return run_organs_build(corpus, ckpt, words_path, organs_toks ? organs_toks
                                                                      : DEFAULT_ORGANS_TOKS);
    if (do_organs_eval)
        return run_organs_eval(corpus, ckpt, words_path, eval_toks ? eval_toks : DEFAULT_EVAL_TOKS,
                               eval_pass);

    setvbuf(stdout, NULL, _IONBF, 0);
    snprintf(part_path, sizeof(part_path), "%s.partial", ckpt);
    snprintf(part_meta, sizeof(part_meta), "%s.partial.meta", ckpt);
    if (!force_fresh) {
        FILE *pm = fopen(part_meta, "rb");
        if (pm) {
            unsigned long long t = 0;
            if (fscanf(pm, "trained_toks=%llu", &t) == 1 && t > 0) {
                FILE *pf = fopen(part_path, "rb");
                FILE *wf = fopen(words_path, "rb");
                if (pf && wf) {
                    resume = 1;
                    resume_toks = (uint64_t)t;
                }
                if (pf)
                    fclose(pf);
                if (wf)
                    fclose(wf);
            }
            fclose(pm);
        }
    }
    printf("=== TinyStories FULL pretrain ===\n");
    if (resume)
        printf("RESUME from %s at trained_toks=%llu (use --fresh to restart)\n", part_path,
               (unsigned long long)resume_toks);
    printf("corpus=%s\n", corpus);
    printf("ckpt=%s\nwords=%s\n", ckpt, words_path);

    f = fopen(corpus, "rb");
    if (!f) {
        fprintf(stderr, "FAIL open corpus (need full TinyStories train file)\n");
        return 1;
    }
    if (fseek(f, 0, SEEK_END) == 0) {
        corpus_bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
    }
    printf("corpus_bytes=%ld (%.2f MiB) — FULL file will be streamed twice\n", corpus_bytes,
           (double)corpus_bytes / (1024.0 * 1024.0));
    if (corpus_bytes > 0 && corpus_bytes < 1000000) {
        fprintf(stderr, "FAIL: corpus too small to be full TinyStories (~1.8GiB expected)\n");
        fclose(f);
        return 1;
    }

    if (!resume) {
        tab = (FreqSlot *)calloc(FREQ_SLOTS, sizeof(FreqSlot));
        if (!tab) {
            fclose(f);
            return 1;
        }
        if (buf_open(&bin, f) != 0) {
            free(tab);
            fclose(f);
            return 1;
        }
        t0 = wall_sec();
        printf("pass1: counting words over full corpus (hash)...\n");
        while (next_word(&bin, wbuf)) {
            total_words++;
            if (freq_add(tab, &nused, wbuf) != 0)
                dropped++;
            if ((total_words % 5000000ull) == 0ull)
                printf("  pass1 words=%llu unique=%u rss_MiB=%.1f\n",
                       (unsigned long long)total_words, nused,
                       (double)process_rss() / (1024.0 * 1024.0));
        }
        buf_close(&bin);
        fclose(f);
        printf("pass1 done words=%llu unique=%u dropped_new=%llu wall_s=%.1f\n",
               (unsigned long long)total_words, nused, (unsigned long long)dropped,
               wall_sec() - t0);

        qsort(tab, FREQ_SLOTS, sizeof(FreqSlot), cmp_freq_desc);
    } else {
        fclose(f);
    }

    W = (FluencyWords *)calloc(1, sizeof(*W));
    if (!W) {
        free(tab);
        return 1;
    }
    if (!resume) {
        fluency_words_init(W);
        strncpy(W->word[0], "<unk>", FLUENCY_WORD_LEN);
        W->freq[0] = 0;
        W->count = 1;
        {
            uint32_t k, limit = VOCAB_CAP - 1u;
            if (nused < limit)
                limit = nused;
            for (k = 0; k < limit; ++k) {
                uint32_t id;
                if (!tab[k].used)
                    break;
                id = W->count;
                strncpy(W->word[id], tab[k].w, FLUENCY_WORD_LEN);
                W->word[id][FLUENCY_WORD_LEN] = 0;
                W->freq[id] = tab[k].n;
                W->count++;
            }
        }
        free(tab);
        tab = NULL;
        printf("vocab sealed count=%u (incl <unk>)\n", W->count);
        /* Words written now so a partial checkpoint is loadable mid-run. */
        if (nerva_words_save(W, words_path) != 0)
            fprintf(stderr, "WARN early words save failed %s\n", words_path);
    } else {
        if (nerva_words_load(W, words_path) != 0) {
            fprintf(stderr, "FAIL resume: cannot load words %s\n", words_path);
            free(W);
            return 1;
        }
        printf("vocab loaded count=%u from %s\n", W->count, words_path);
    }

    idmap = (IdSlot *)calloc(ID_SLOTS, sizeof(IdSlot));
    if (!idmap) {
        free(W);
        return 1;
    }
    {
        uint32_t k;
        for (k = 1; k < W->count; ++k)
            idmap_put(idmap, W->word[k], (flu_tok_t)k);
    }

    if (!resume) {
        cfg = nerva_config_default();
        cfg.weight_max_q8_8 = INT16_MAX;
        cfg.max_nodes = 800000u;
        cfg.max_edges = 6000000u;
        cfg.max_names = 800000u;
        cfg.max_events = (uint32_t)FLUENCY_VOCAB * 2u + 4096u;
        cfg.max_active_nodes = 8192u;
        cfg.max_fire_log = 4096u;
        cfg.max_traces = 16384u;
        cfg.max_mutations = 4096u;
        cfg.max_mutation_log = 4096u;
        cfg.max_schemas = 64u;
        cfg.max_memory_blocks = 64u;
        cfg.max_expectations = 64u;

        printf("engine caps nodes=%u edges=%u names=%u\n", cfg.max_nodes, cfg.max_edges,
               cfg.max_names);
        if (nerva_engine_init(&eng, cfg) != 0) {
            fprintf(stderr, "FAIL engine init\n");
            free(idmap);
            free(W);
            return 1;
        }
        if (fluency_init(&model, &eng, 3u) != 0) {
            fprintf(stderr, "FAIL fluency init\n");
            nerva_engine_free(&eng);
            free(idmap);
            free(W);
            return 1;
        }
        model.fast_lambda = (nerva_uq0_16_t)(0.15 * 65535.0);
    } else {
        memset(&eng, 0, sizeof(eng));
        memset(&model, 0, sizeof(model));
        if (fluency_load(&eng, &model, part_path) != 0) {
            fprintf(stderr, "FAIL resume: cannot load partial ckpt %s\n", part_path);
            free(idmap);
            free(W);
            return 1;
        }
        printf("resume: model loaded order=%u nodes=%u edges=%u\n", model.order, eng.node_count,
               eng.edge_count);
    }
    nerva_work_set_rgrad_updates(0);

    chunk = (flu_tok_t *)malloc(sizeof(flu_tok_t) * CHUNK_TOKS);
    if (!chunk) {
        fluency_free(&model);
        nerva_engine_free(&eng);
        free(idmap);
        free(W);
        return 1;
    }

    f = fopen(corpus, "rb");
    if (!f) {
        free(chunk);
        fluency_free(&model);
        nerva_engine_free(&eng);
        free(idmap);
        free(W);
        return 1;
    }
    if (buf_open(&bin, f) != 0) {
        fclose(f);
        free(chunk);
        fluency_free(&model);
        nerva_engine_free(&eng);
        free(idmap);
        free(W);
        return 1;
    }

    t0 = wall_sec();
    printf("pass2: streaming train over FULL corpus (online fluency_train)...\n");
    if (resume) {
        uint64_t skipped = 0;
        printf("resume: skipping %llu already-trained tokens...\n",
               (unsigned long long)resume_toks);
        while (skipped < resume_toks && next_word(&bin, wbuf))
            skipped++;
        if (skipped != resume_toks) {
            fprintf(stderr, "FAIL resume: corpus ended at %llu < resume point %llu\n",
                    (unsigned long long)skipped, (unsigned long long)resume_toks);
            buf_close(&bin);
            fclose(f);
            free(chunk);
            fluency_free(&model);
            nerva_engine_free(&eng);
            free(idmap);
            free(W);
            return 1;
        }
        trained = resume_toks;
        last_ckpt = resume_toks;
        printf("resume: skip done wall_s=%.1f\n", wall_sec() - t0);
    }
    cn = 0;
    while (next_word(&bin, wbuf)) {
        flu_tok_t id = idmap_get(idmap, wbuf);
        if (id == 0)
            oov++;
        chunk[cn++] = id;
        if (cn >= CHUNK_TOKS) {
            fluency_train(&model, chunk, cn);
            trained += cn;
            cn = 0;
            if ((trained % 2000000ull) == 0ull) {
                printf("  trained_toks=%llu nodes=%u edges=%u rss_MiB=%.1f wall_s=%.1f\n",
                       (unsigned long long)trained, eng.node_count, eng.edge_count,
                       (double)process_rss() / (1024.0 * 1024.0), wall_sec() - t0);
            }
            if (trained - last_ckpt >= SAVE_EVERY_TOKS) {
                double ts = wall_sec();
                if (fluency_save(&model, part_path) == 0) {
                    FILE *pm = fopen(part_meta, "wb");
                    if (pm) {
                        fprintf(pm, "trained_toks=%llu\n", (unsigned long long)trained);
                        fclose(pm);
                    }
                    printf("  partial ckpt trained_toks=%llu path=%s save_s=%.1f\n",
                           (unsigned long long)trained, part_path, wall_sec() - ts);
                } else {
                    fprintf(stderr, "WARN partial ckpt save failed %s\n", part_path);
                }
                last_ckpt = trained;
            }
        }
    }
    if (cn > 0) {
        fluency_train(&model, chunk, cn);
        trained += cn;
    }
    buf_close(&bin);
    fclose(f);
    free(chunk);
    free(idmap);
    idmap = NULL;
    t1 = wall_sec() - t0;
    printf("pass2 done trained_toks=%llu oov=%llu nodes=%u edges=%u wall_s=%.1f "
           "rss_MiB=%.1f\n",
           (unsigned long long)trained, (unsigned long long)oov, eng.node_count, eng.edge_count, t1,
           (double)process_rss() / (1024.0 * 1024.0));

    if (trained < total_words / 2 && total_words > 1000) {
        fprintf(stderr, "FAIL: trained_toks far below pass1 word count — incomplete stream\n");
        fluency_free(&model);
        nerva_engine_free(&eng);
        free(W);
        return 1;
    }

    printf("saving checkpoint...\n");
    if (fluency_save(&model, ckpt) == 0) {
        remove(part_path);
        remove(part_meta);
    } else {
        fprintf(stderr, "FAIL fluency_save %s\n", ckpt);
        fluency_free(&model);
        nerva_engine_free(&eng);
        free(W);
        return 1;
    }
    if (nerva_words_save(W, words_path) != 0) {
        fprintf(stderr, "FAIL words save %s\n", words_path);
        fluency_free(&model);
        nerva_engine_free(&eng);
        free(W);
        return 1;
    }

    {
        FILE *mf = fopen(meta_path, "wb");
        long ckpt_sz = 0, words_sz = 0;
        FILE *cf = fopen(ckpt, "rb");
        if (cf) {
            fseek(cf, 0, SEEK_END);
            ckpt_sz = ftell(cf);
            fclose(cf);
        }
        cf = fopen(words_path, "rb");
        if (cf) {
            fseek(cf, 0, SEEK_END);
            words_sz = ftell(cf);
            fclose(cf);
        }
        if (mf) {
            fprintf(mf, "corpus=%s\n", corpus);
            fprintf(mf, "corpus_bytes=%ld\n", corpus_bytes);
            fprintf(mf, "pass1_words=%llu\n", (unsigned long long)total_words);
            fprintf(mf, "trained_toks=%llu\n", (unsigned long long)trained);
            fprintf(mf, "oov=%llu\n", (unsigned long long)oov);
            fprintf(mf, "vocab=%u\n", W->count);
            fprintf(mf, "nodes=%u\nedges=%u\n", eng.node_count, eng.edge_count);
            fprintf(mf, "ckpt=%s\nckpt_bytes=%ld\n", ckpt, ckpt_sz);
            fprintf(mf, "words=%s\nwords_bytes=%ld\n", words_path, words_sz);
            fprintf(mf, "train_wall_s=%.3f\n", t1);
            fprintf(mf, "full_tinystories=YES\n");
            fclose(mf);
        }
        printf("ckpt_bytes=%ld words_bytes=%ld meta=%s\n", ckpt_sz, words_sz, meta_path);
        if (ckpt_sz < 10000) {
            fprintf(stderr, "FAIL: checkpoint suspiciously small\n");
            fluency_free(&model);
            nerva_engine_free(&eng);
            free(W);
            return 1;
        }
    }

    printf("MACHINE pretrain=tinystories_full corpus_bytes=%ld trained_toks=%llu "
           "vocab=%u nodes=%u edges=%u ckpt=%s ckpt_bytes_ok=YES verdict=PASS\n",
           corpus_bytes, (unsigned long long)trained, W->count, eng.node_count, eng.edge_count,
           ckpt);

    fluency_free(&model);
    nerva_engine_free(&eng);
    free(W);
    return 0;
}
