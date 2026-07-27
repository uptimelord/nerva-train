// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II
//
// Session checkpoint: engine (nerva_persist v0.2) + FluencyModel side state
// + optional chat blob. See fluency_session.h for on-disk layout.

#include "fluency_session.h"

#include "nerva_config.h"
#include "nerva_engine.h"
#include "nerva_persist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- CRC helpers (reuse engine CRC32) ------------------------------------ */

static void sess_crc_update(uint32_t *crc, const void *data, size_t len) {
    if (!crc || !data || len == 0u) {
        return;
    }
    /* Incremental: fold via full-buffer recompute is O(n²); use byte loop with
     * the public API by XOR-composing is not exposed. Recompute over streaming
     * by maintaining running CRC with the same poly as nerva_persist_crc32.
     * We call nerva_persist_crc32 on chunks by combining: final = crc32(all).
     * For write path we accumulate into a growable buffer is heavy; instead
     * mirror the update used in nerva_persist (table not public). Simplest
     * correct approach: write blobs, then CRC the payload range by re-read.
     * Write path: two-pass — write payload first, CRC, then rewrite header. */
    (void)crc;
    (void)data;
    (void)len;
}

static int write_all(FILE *f, const void *data, size_t n) {
    if (n == 0u) {
        return 0;
    }
    return fwrite(data, 1, n, f) == n ? 0 : -1;
}

static int read_all(FILE *f, void *data, size_t n) {
    if (n == 0u) {
        return 0;
    }
    return fread(data, 1, n, f) == n ? 0 : -1;
}

static int path_tmp(const char *path, char *out, size_t out_cap, const char *suffix) {
    if (!path || !out || out_cap < 8u) {
        return -1;
    }
    size_t n = strlen(path);
    size_t sn = strlen(suffix);
    if (n + sn + 1u >= out_cap) {
        return -1;
    }
    memcpy(out, path, n);
    memcpy(out + n, suffix, sn + 1u);
    return 0;
}

static uint8_t *read_file_bytes(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf && sz > 0) {
        fclose(f);
        return NULL;
    }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    if (out_len) {
        *out_len = (size_t)sz;
    }
    return buf;
}

/* ---- Fluency side blob --------------------------------------------------- */

#define FLU_SIDE_MAGIC "FluSide1"

typedef struct FluSideScalars {
    uint32_t order;
    uint8_t weight_map;
    uint8_t _pad0[3];
    uint32_t vocab_count;
    uint64_t unigram_total;
    uint64_t train_positions;
    uint16_t fast_slots;
    uint16_t _pad1;
    nerva_q8_8_t fast_w0;
    uint32_t fast_half_life;
    nerva_uq0_16_t fast_lambda;
    nerva_uq0_16_t fast_beta;
    nerva_uq0_16_t fast_delta;
    uint16_t _pad2;
    uint32_t stream_pos; /* THE subtle clock */
    uint8_t mood_on;
    uint8_t mood_k;
    uint16_t _pad3;
    uint32_t mood_half_life;
    nerva_q8_8_t mood_scale;
    uint8_t gain_on;
    uint8_t hub_on;
    uint8_t instance_on;
    uint8_t has_mood_aff;
    uint16_t gain_window;
    nerva_q8_8_t gain_scale;
    uint32_t gain_topk;
    uint8_t has_gain;
    uint8_t has_hubs;
    uint16_t hub_h;
    nerva_q8_8_t hub_scale;
    int32_t hub_conf_margin;
    uint32_t instance_cap;
    uint32_t instance_n;
    uint64_t instance_deposits;
    uint64_t instance_evicts;
    uint64_t instance_hits;
    size_t slot_cap;
    size_t slot_count;
    float mood_reg[FLU_MOOD_K_MAX];
    /* Engine query-state scalars that nerva_persist v0.2 resets on load
     * (src/nerva_persist.c reset block). last_query_start_tick feeds the
     * exception path (src fired-this-query test); an unrestored value makes
     * loaded propagation diverge from live on previously-fired nodes. */
    uint64_t eng_last_query_start_tick;
    uint32_t eng_idle_ticks;
    uint32_t eng_active_query_tag;
    /* nerva_persist_load unconditionally rebuilds adjacency (valid=1), but a
     * live model that created nodes/edges after its last rebuild is running
     * with adjacency_valid=0 (fires do not propagate). The flag must
     * round-trip or loaded propagation diverges from live. */
    uint8_t eng_adjacency_valid;
    uint8_t _pad4[7];
    /* PPL-push levers (skip contexts live in the engine graph; only the
     * flags and KN continuation table need side-blob carriage). */
    uint8_t skip_on;
    uint8_t kn_on;
    uint8_t has_kn_cont;
    uint8_t _pad5[5];
    uint64_t kn_cont_total;
} FluSideScalars;

static int fluency_side_encode(const FluencyModel *m, uint8_t **out, size_t *out_len) {
    if (!m || !m->e || !m->fast_out || !m->slots || !out || !out_len) {
        return -1;
    }
    *out = NULL;
    *out_len = 0;

    FluSideScalars sc;
    memset(&sc, 0, sizeof(sc));
    sc.order = m->order;
    sc.weight_map = m->weight_map;
    sc.vocab_count = m->vocab_count;
    sc.unigram_total = m->unigram_total;
    sc.train_positions = m->train_positions;
    sc.fast_slots = m->fast_slots;
    sc.fast_w0 = m->fast_w0;
    sc.fast_half_life = m->fast_half_life;
    sc.fast_lambda = m->fast_lambda;
    sc.fast_beta = m->fast_beta;
    sc.fast_delta = m->fast_delta;
    sc.stream_pos = m->stream_pos;
    sc.mood_on = m->mood_on;
    sc.mood_k = m->mood_k;
    sc.mood_half_life = m->mood_half_life;
    sc.mood_scale = m->mood_scale;
    sc.gain_on = m->gain_on;
    sc.hub_on = m->hub_on;
    sc.instance_on = m->instance_on;
    sc.has_mood_aff = (m->mood_aff != NULL) ? 1u : 0u;
    sc.gain_window = m->gain_window;
    sc.gain_scale = m->gain_scale;
    sc.gain_topk = m->gain_topk;
    sc.has_gain = (m->gain_cooc && m->gain_tok && m->gain_vidx && m->gain_topk > 0u) ? 1u : 0u;
    sc.has_hubs = (m->tok_hubs && m->hub_mem_tok && m->hub_mem_w && m->hub_mem_n && m->hub_h > 0u)
                      ? 1u
                      : 0u;
    sc.hub_h = m->hub_h;
    sc.hub_scale = m->hub_scale;
    sc.hub_conf_margin = m->hub_conf_margin;
    sc.instance_cap = m->instance_cap;
    sc.instance_n = m->instance_n;
    sc.instance_deposits = m->instance_deposits;
    sc.instance_evicts = m->instance_evicts;
    sc.instance_hits = m->instance_hits;
    sc.slot_cap = m->slot_cap;
    sc.slot_count = m->slot_count;
    memcpy(sc.mood_reg, m->mood_reg, sizeof(sc.mood_reg));
    sc.eng_last_query_start_tick = (uint64_t)m->e->last_query_start_tick;
    sc.eng_idle_ticks = m->e->idle_ticks;
    sc.eng_active_query_tag = m->e->active_query_tag;
    sc.eng_adjacency_valid = (uint8_t)m->e->adjacency_valid;
    sc.skip_on = m->skip_on;
    sc.kn_on = m->kn_on;
    sc.has_kn_cont = (m->kn_cont != NULL) ? 1u : 0u;
    sc.kn_cont_total = m->kn_cont_total;

    uint8_t *name_blob = NULL;
    size_t name_len = 0;
    if (fluency_name_map_export(m, &name_blob, &name_len) != 0) {
        return -1;
    }

    size_t nfast = (size_t)FLUENCY_VOCAB * (size_t)m->fast_slots;
    size_t need = 8u + sizeof(sc);
    need += sizeof(uint32_t) * (size_t)FLUENCY_VOCAB;
    need += (size_t)FLUENCY_VOCAB;
    need += sizeof(uint64_t) * (size_t)FLUENCY_VOCAB;
    need += sizeof(uint64_t) + name_len; /* name blob length prefix + body */
    need += sizeof(FluencyEdgeSlot) * m->slot_cap;
    need += sizeof(FluencyFastSlot) * nfast;
    if (m->instance_pool && m->instance_cap > 0u) {
        need += sizeof(FluencyInstanceSlot) * (size_t)m->instance_cap;
    }
    if (sc.has_mood_aff) {
        need += sizeof(float) * (size_t)FLUENCY_VOCAB * (size_t)m->mood_k;
    }
    if (sc.has_gain) {
        need += sizeof(uint32_t) * (size_t)m->gain_topk * (size_t)m->gain_topk;
        need += sizeof(flu_tok_t) * (size_t)m->gain_topk;
        need += sizeof(uint16_t) * (size_t)FLUENCY_VOCAB;
    }
    if (sc.has_hubs) {
        need += sizeof(FluencyHubSlot) * (size_t)FLUENCY_VOCAB * (size_t)FLU_HUB_MEM_MAX;
        need += sizeof(flu_tok_t) * (size_t)m->hub_h * (size_t)FLU_HUB_SIZE_CAP;
        need += sizeof(nerva_uq0_16_t) * (size_t)m->hub_h * (size_t)FLU_HUB_SIZE_CAP;
        need += sizeof(uint16_t) * (size_t)m->hub_h;
    }
    if (sc.has_kn_cont) {
        need += sizeof(uint32_t) * (size_t)FLUENCY_VOCAB;
    }

    uint8_t *buf = (uint8_t *)malloc(need);
    if (!buf) {
        free(name_blob);
        return -1;
    }
    size_t o = 0;
    memcpy(buf + o, FLU_SIDE_MAGIC, 8);
    o += 8;
    memcpy(buf + o, &sc, sizeof(sc));
    o += sizeof(sc);
    memcpy(buf + o, m->tok_node, sizeof(uint32_t) * (size_t)FLUENCY_VOCAB);
    o += sizeof(uint32_t) * (size_t)FLUENCY_VOCAB;
    memcpy(buf + o, m->seen, (size_t)FLUENCY_VOCAB);
    o += (size_t)FLUENCY_VOCAB;
    memcpy(buf + o, m->unigram, sizeof(uint64_t) * (size_t)FLUENCY_VOCAB);
    o += sizeof(uint64_t) * (size_t)FLUENCY_VOCAB;
    {
        uint64_t nl = (uint64_t)name_len;
        memcpy(buf + o, &nl, sizeof(nl));
        o += sizeof(nl);
        if (name_len) {
            memcpy(buf + o, name_blob, name_len);
            o += name_len;
        }
    }
    free(name_blob);
    memcpy(buf + o, m->slots, sizeof(FluencyEdgeSlot) * m->slot_cap);
    o += sizeof(FluencyEdgeSlot) * m->slot_cap;
    memcpy(buf + o, m->fast_out, sizeof(FluencyFastSlot) * nfast);
    o += sizeof(FluencyFastSlot) * nfast;
    if (m->instance_pool && m->instance_cap > 0u) {
        memcpy(buf + o, m->instance_pool, sizeof(FluencyInstanceSlot) * (size_t)m->instance_cap);
        o += sizeof(FluencyInstanceSlot) * (size_t)m->instance_cap;
    }
    if (sc.has_mood_aff) {
        size_t na = (size_t)FLUENCY_VOCAB * (size_t)m->mood_k;
        memcpy(buf + o, m->mood_aff, sizeof(float) * na);
        o += sizeof(float) * na;
    }
    if (sc.has_gain) {
        size_t gc = (size_t)m->gain_topk * (size_t)m->gain_topk;
        memcpy(buf + o, m->gain_cooc, sizeof(uint32_t) * gc);
        o += sizeof(uint32_t) * gc;
        memcpy(buf + o, m->gain_tok, sizeof(flu_tok_t) * (size_t)m->gain_topk);
        o += sizeof(flu_tok_t) * (size_t)m->gain_topk;
        memcpy(buf + o, m->gain_vidx, sizeof(uint16_t) * (size_t)FLUENCY_VOCAB);
        o += sizeof(uint16_t) * (size_t)FLUENCY_VOCAB;
    }
    if (sc.has_hubs) {
        size_t th = (size_t)FLUENCY_VOCAB * (size_t)FLU_HUB_MEM_MAX;
        size_t hm = (size_t)m->hub_h * (size_t)FLU_HUB_SIZE_CAP;
        memcpy(buf + o, m->tok_hubs, sizeof(FluencyHubSlot) * th);
        o += sizeof(FluencyHubSlot) * th;
        memcpy(buf + o, m->hub_mem_tok, sizeof(flu_tok_t) * hm);
        o += sizeof(flu_tok_t) * hm;
        memcpy(buf + o, m->hub_mem_w, sizeof(nerva_uq0_16_t) * hm);
        o += sizeof(nerva_uq0_16_t) * hm;
        memcpy(buf + o, m->hub_mem_n, sizeof(uint16_t) * (size_t)m->hub_h);
        o += sizeof(uint16_t) * (size_t)m->hub_h;
    }
    if (sc.has_kn_cont) {
        memcpy(buf + o, m->kn_cont, sizeof(uint32_t) * (size_t)FLUENCY_VOCAB);
        o += sizeof(uint32_t) * (size_t)FLUENCY_VOCAB;
    }
    if (o != need) {
        free(buf);
        return -1;
    }
    *out = buf;
    *out_len = need;
    return 0;
}

static int fluency_side_decode(FluencyModel *m, const uint8_t *buf, size_t len) {
    if (!m || !m->e || !buf || len < 8u + sizeof(FluSideScalars)) {
        return -1;
    }
    if (memcmp(buf, FLU_SIDE_MAGIC, 8) != 0) {
        return -1;
    }
    size_t o = 8;
    FluSideScalars sc;
    memcpy(&sc, buf + o, sizeof(sc));
    o += sizeof(sc);

    if (sc.order == 0u || sc.order > FLUENCY_MAX_ORDER || sc.fast_slots == 0u) {
        return -1;
    }
    if (sc.mood_k == 0u || sc.mood_k > FLU_MOOD_K_MAX) {
        return -1;
    }

    /* Reconfigure pools if sizes differ from fluency_init defaults. */
    m->order = sc.order;
    m->weight_map = sc.weight_map;
    m->vocab_count = sc.vocab_count;
    m->unigram_total = sc.unigram_total;
    m->train_positions = sc.train_positions;
    m->fast_w0 = sc.fast_w0;
    m->fast_half_life = sc.fast_half_life;
    m->fast_lambda = sc.fast_lambda;
    m->fast_beta = sc.fast_beta;
    m->fast_delta = sc.fast_delta;
    m->stream_pos = sc.stream_pos;
    m->mood_on = sc.mood_on;
    m->mood_k = sc.mood_k;
    m->mood_half_life = sc.mood_half_life;
    m->mood_scale = sc.mood_scale;
    m->gain_on = sc.gain_on;
    m->hub_on = sc.hub_on;
    m->instance_on = sc.instance_on;
    m->gain_window = sc.gain_window;
    m->gain_scale = sc.gain_scale;
    m->gain_topk = sc.gain_topk;
    m->hub_h = sc.hub_h;
    m->hub_scale = sc.hub_scale;
    m->hub_conf_margin = sc.hub_conf_margin;
    m->instance_n = sc.instance_n;
    m->instance_deposits = sc.instance_deposits;
    m->instance_evicts = sc.instance_evicts;
    m->instance_hits = sc.instance_hits;
    memcpy(m->mood_reg, sc.mood_reg, sizeof(m->mood_reg));
    /* Restore engine query state zeroed by nerva_persist_load. */
    m->e->last_query_start_tick = (nerva_tick_t)sc.eng_last_query_start_tick;
    m->e->idle_ticks = sc.eng_idle_ticks;
    m->e->active_query_tag = sc.eng_active_query_tag;
    m->e->adjacency_valid = sc.eng_adjacency_valid;
    m->skip_on = sc.skip_on;
    m->kn_on = sc.kn_on;
    m->kn_cont_total = sc.kn_cont_total;

    size_t nfast = (size_t)FLUENCY_VOCAB * (size_t)sc.fast_slots;
    if (sc.fast_slots != m->fast_slots || !m->fast_out) {
        free(m->fast_out);
        m->fast_out = (FluencyFastSlot *)calloc(nfast, sizeof(FluencyFastSlot));
        if (!m->fast_out) {
            return -1;
        }
        m->fast_slots = sc.fast_slots;
    }
    if (sc.instance_cap != m->instance_cap || !m->instance_pool) {
        free(m->instance_pool);
        m->instance_pool = NULL;
        m->instance_cap = sc.instance_cap;
        if (sc.instance_cap > 0u) {
            m->instance_pool =
                (FluencyInstanceSlot *)calloc((size_t)sc.instance_cap, sizeof(FluencyInstanceSlot));
            if (!m->instance_pool) {
                return -1;
            }
        }
    }

    /* Slots table. */
    free(m->slots);
    m->slots = NULL;
    m->slot_cap = 0;
    m->slot_count = 0;
    if (sc.slot_cap == 0u) {
        return -1;
    }
    m->slots = (FluencyEdgeSlot *)calloc(sc.slot_cap, sizeof(FluencyEdgeSlot));
    if (!m->slots) {
        return -1;
    }
    m->slot_cap = sc.slot_cap;
    m->slot_count = sc.slot_count;

    /* Remaining length checked incrementally below (name blob is variable). */
    if (o + sizeof(uint32_t) * (size_t)FLUENCY_VOCAB + (size_t)FLUENCY_VOCAB +
            sizeof(uint64_t) * (size_t)FLUENCY_VOCAB + sizeof(uint64_t) >
        len) {
        return -1;
    }

    memcpy(m->tok_node, buf + o, sizeof(uint32_t) * (size_t)FLUENCY_VOCAB);
    o += sizeof(uint32_t) * (size_t)FLUENCY_VOCAB;
    memcpy(m->seen, buf + o, (size_t)FLUENCY_VOCAB);
    o += (size_t)FLUENCY_VOCAB;
    memcpy(m->unigram, buf + o, sizeof(uint64_t) * (size_t)FLUENCY_VOCAB);
    o += sizeof(uint64_t) * (size_t)FLUENCY_VOCAB;
    {
        if (o + sizeof(uint64_t) > len) {
            return -1;
        }
        uint64_t nl = 0;
        memcpy(&nl, buf + o, sizeof(nl));
        o += sizeof(nl);
        if (o + (size_t)nl > len) {
            return -1;
        }
        if (fluency_name_map_import(m, buf + o, (size_t)nl) != 0) {
            /* Fallback: rebuild from engine names. */
            if (fluency_rebuild_name_index(m) != 0) {
                return -1;
            }
        }
        o += (size_t)nl;
    }
    {
        size_t need_slots = sizeof(FluencyEdgeSlot) * sc.slot_cap + sizeof(FluencyFastSlot) * nfast;
        if (sc.instance_cap > 0u) {
            need_slots += sizeof(FluencyInstanceSlot) * (size_t)sc.instance_cap;
        }
        if (o + need_slots > len) {
            return -1;
        }
    }
    memcpy(m->slots, buf + o, sizeof(FluencyEdgeSlot) * sc.slot_cap);
    o += sizeof(FluencyEdgeSlot) * sc.slot_cap;
    memcpy(m->fast_out, buf + o, sizeof(FluencyFastSlot) * nfast);
    o += sizeof(FluencyFastSlot) * nfast;
    if (sc.instance_cap > 0u) {
        if (!m->instance_pool || o + sizeof(FluencyInstanceSlot) * (size_t)sc.instance_cap > len) {
            return -1;
        }
        memcpy(m->instance_pool, buf + o, sizeof(FluencyInstanceSlot) * (size_t)sc.instance_cap);
        o += sizeof(FluencyInstanceSlot) * (size_t)sc.instance_cap;
    }

    free(m->mood_aff);
    m->mood_aff = NULL;
    if (sc.has_mood_aff) {
        size_t na = (size_t)FLUENCY_VOCAB * (size_t)sc.mood_k;
        if (o + sizeof(float) * na > len) {
            return -1;
        }
        m->mood_aff = (float *)malloc(sizeof(float) * na);
        if (!m->mood_aff) {
            return -1;
        }
        memcpy(m->mood_aff, buf + o, sizeof(float) * na);
        o += sizeof(float) * na;
    }

    free(m->gain_cooc);
    free(m->gain_tok);
    free(m->gain_vidx);
    m->gain_cooc = NULL;
    m->gain_tok = NULL;
    m->gain_vidx = NULL;
    if (sc.has_gain) {
        size_t gc = (size_t)sc.gain_topk * (size_t)sc.gain_topk;
        size_t need_g = sizeof(uint32_t) * gc + sizeof(flu_tok_t) * (size_t)sc.gain_topk +
                        sizeof(uint16_t) * (size_t)FLUENCY_VOCAB;
        if (o + need_g > len) {
            return -1;
        }
        m->gain_cooc = (uint32_t *)malloc(sizeof(uint32_t) * gc);
        m->gain_tok = (flu_tok_t *)malloc(sizeof(flu_tok_t) * (size_t)sc.gain_topk);
        m->gain_vidx = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)FLUENCY_VOCAB);
        if (!m->gain_cooc || !m->gain_tok || !m->gain_vidx) {
            return -1;
        }
        memcpy(m->gain_cooc, buf + o, sizeof(uint32_t) * gc);
        o += sizeof(uint32_t) * gc;
        memcpy(m->gain_tok, buf + o, sizeof(flu_tok_t) * (size_t)sc.gain_topk);
        o += sizeof(flu_tok_t) * (size_t)sc.gain_topk;
        memcpy(m->gain_vidx, buf + o, sizeof(uint16_t) * (size_t)FLUENCY_VOCAB);
        o += sizeof(uint16_t) * (size_t)FLUENCY_VOCAB;
    }

    free(m->tok_hubs);
    free(m->hub_mem_tok);
    free(m->hub_mem_w);
    free(m->hub_mem_n);
    m->tok_hubs = NULL;
    m->hub_mem_tok = NULL;
    m->hub_mem_w = NULL;
    m->hub_mem_n = NULL;
    if (sc.has_hubs) {
        size_t th = (size_t)FLUENCY_VOCAB * (size_t)FLU_HUB_MEM_MAX;
        size_t hm = (size_t)sc.hub_h * (size_t)FLU_HUB_SIZE_CAP;
        size_t need_h = sizeof(FluencyHubSlot) * th + sizeof(flu_tok_t) * hm +
                        sizeof(nerva_uq0_16_t) * hm + sizeof(uint16_t) * (size_t)sc.hub_h;
        if (o + need_h > len) {
            return -1;
        }
        m->tok_hubs = (FluencyHubSlot *)malloc(sizeof(FluencyHubSlot) * th);
        m->hub_mem_tok = (flu_tok_t *)malloc(sizeof(flu_tok_t) * hm);
        m->hub_mem_w = (nerva_uq0_16_t *)malloc(sizeof(nerva_uq0_16_t) * hm);
        m->hub_mem_n = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)sc.hub_h);
        if (!m->tok_hubs || !m->hub_mem_tok || !m->hub_mem_w || !m->hub_mem_n) {
            return -1;
        }
        memcpy(m->tok_hubs, buf + o, sizeof(FluencyHubSlot) * th);
        o += sizeof(FluencyHubSlot) * th;
        memcpy(m->hub_mem_tok, buf + o, sizeof(flu_tok_t) * hm);
        o += sizeof(flu_tok_t) * hm;
        memcpy(m->hub_mem_w, buf + o, sizeof(nerva_uq0_16_t) * hm);
        o += sizeof(nerva_uq0_16_t) * hm;
        memcpy(m->hub_mem_n, buf + o, sizeof(uint16_t) * (size_t)sc.hub_h);
        o += sizeof(uint16_t) * (size_t)sc.hub_h;
    }

    free(m->kn_cont);
    m->kn_cont = NULL;
    if (sc.has_kn_cont) {
        size_t need_k = sizeof(uint32_t) * (size_t)FLUENCY_VOCAB;
        if (o + need_k > len) {
            return -1;
        }
        m->kn_cont = (uint32_t *)malloc(need_k);
        if (!m->kn_cont) {
            return -1;
        }
        memcpy(m->kn_cont, buf + o, need_k);
        o += need_k;
    }

    if (o != len) {
        /* Trailing bytes = corrupt / version skew. */
        return -1;
    }
    return 0;
}

/* ---- Session save/load --------------------------------------------------- */

int fluency_session_save(const FluencyModel *m, const char *path, const uint8_t *chat,
                         size_t chat_len) {
    if (!m || !m->e || !path) {
        return -1;
    }
    if (chat_len > 0u && !chat) {
        return -1;
    }

    char eng_path[1024];
    if (path_tmp(path, eng_path, sizeof(eng_path), ".eng.tmp") != 0) {
        return -1;
    }
    if (nerva_persist_save(m->e, eng_path) != 0) {
        return -1;
    }
    size_t eng_len = 0;
    uint8_t *eng = read_file_bytes(eng_path, &eng_len);
    remove(eng_path);
    if (!eng && eng_len > 0u) {
        return -1;
    }

    uint8_t *flu = NULL;
    size_t flu_len = 0;
    if (fluency_side_encode(m, &flu, &flu_len) != 0) {
        free(eng);
        return -1;
    }

    NervaSessionHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, NERVA_SESSION_MAGIC, 8);
    h.version_major = NERVA_SESSION_VERSION_MAJOR;
    h.version_minor = NERVA_SESSION_VERSION_MINOR;
    h.header_size = (uint16_t)sizeof(NervaSessionHeader);
    h.endian_marker = NERVA_SESSION_ENDIAN_MARKER;
    h.flags = (chat_len > 0u) ? NERVA_SESSION_FLAG_CHAT : 0u;
    h.fluency_vocab = (uint32_t)FLUENCY_VOCAB;
    h.max_nodes = m->e->cfg.max_nodes;
    h.max_edges = m->e->cfg.max_edges;
    h.max_names = m->e->cfg.max_names;
    h.max_memory = m->e->cfg.max_memory_blocks;
    h.max_schemas = m->e->cfg.max_schemas;
    h.weight_max_q8_8 = m->e->cfg.weight_max_q8_8;
    h.engine_offset = sizeof(NervaSessionHeader);
    h.engine_size = (uint64_t)eng_len;
    h.fluency_offset = h.engine_offset + h.engine_size;
    h.fluency_size = (uint64_t)flu_len;
    h.chat_offset = h.fluency_offset + h.fluency_size;
    h.chat_size = (uint64_t)chat_len;
    h.file_size = h.chat_offset + h.chat_size;

    /* Payload CRC over concatenated sections. */
    size_t pay_len = eng_len + flu_len + chat_len;
    uint8_t *pay = (uint8_t *)malloc(pay_len ? pay_len : 1u);
    if (!pay) {
        free(eng);
        free(flu);
        return -1;
    }
    size_t po = 0;
    if (eng_len) {
        memcpy(pay + po, eng, eng_len);
        po += eng_len;
    }
    if (flu_len) {
        memcpy(pay + po, flu, flu_len);
        po += flu_len;
    }
    if (chat_len) {
        memcpy(pay + po, chat, chat_len);
        po += chat_len;
    }
    h.payload_crc32 = nerva_persist_crc32(pay, pay_len);
    h.header_crc32 = 0;
    h.header_crc32 = nerva_persist_crc32((const uint8_t *)&h, sizeof(h));

    FILE *out = fopen(path, "wb");
    if (!out) {
        free(eng);
        free(flu);
        free(pay);
        return -1;
    }
    int rc = 0;
    if (write_all(out, &h, sizeof(h)) != 0 || write_all(out, pay, pay_len) != 0) {
        rc = -1;
    }
    fclose(out);
    free(eng);
    free(flu);
    free(pay);
    return rc;
}

static int header_valid(const NervaSessionHeader *h, uint64_t file_size) {
    if (!h) {
        return 0;
    }
    if (memcmp(h->magic, NERVA_SESSION_MAGIC, 8) != 0) {
        return 0;
    }
    if (h->version_major != NERVA_SESSION_VERSION_MAJOR ||
        h->version_minor != NERVA_SESSION_VERSION_MINOR) {
        return 0;
    }
    if (h->header_size != (uint16_t)sizeof(NervaSessionHeader)) {
        return 0;
    }
    if (h->endian_marker != NERVA_SESSION_ENDIAN_MARKER) {
        return 0;
    }
    if (h->fluency_vocab != (uint32_t)FLUENCY_VOCAB) {
        return 0;
    }
    if (h->file_size != file_size || file_size < sizeof(NervaSessionHeader)) {
        return 0;
    }
    if (h->engine_offset != sizeof(NervaSessionHeader)) {
        return 0;
    }
    if (h->fluency_offset != h->engine_offset + h->engine_size) {
        return 0;
    }
    if (h->chat_offset != h->fluency_offset + h->fluency_size) {
        return 0;
    }
    if (h->file_size != h->chat_offset + h->chat_size) {
        return 0;
    }
    NervaSessionHeader chk = *h;
    uint32_t expect = chk.header_crc32;
    chk.header_crc32 = 0;
    if (nerva_persist_crc32((const uint8_t *)&chk, sizeof(chk)) != expect) {
        return 0;
    }
    return 1;
}

int fluency_session_load(NervaEngine *e, FluencyModel *m, const char *path, uint8_t *chat_out,
                         size_t chat_cap, size_t *chat_len_out) {
    if (!e || !m || !path) {
        return -1;
    }
    if (chat_len_out) {
        *chat_len_out = 0;
    }

    FILE *in = fopen(path, "rb");
    if (!in) {
        fprintf(stderr, "session: cannot open %s\n", path);
        return -1;
    }
    if (fseek(in, 0, SEEK_END) != 0) {
        fclose(in);
        return -1;
    }
    long fsz = ftell(in);
    if (fsz < (long)sizeof(NervaSessionHeader)) {
        fprintf(stderr, "session: truncated header\n");
        fclose(in);
        return -1;
    }
    if (fseek(in, 0, SEEK_SET) != 0) {
        fclose(in);
        return -1;
    }

    NervaSessionHeader h;
    if (read_all(in, &h, sizeof(h)) != 0) {
        fclose(in);
        return -1;
    }
    if (!header_valid(&h, (uint64_t)fsz)) {
        fprintf(stderr, "session: bad header (magic/version/CRC/layout)\n");
        fclose(in);
        return -1;
    }

    size_t pay_len = (size_t)(h.file_size - sizeof(NervaSessionHeader));
    uint8_t *pay = (uint8_t *)malloc(pay_len ? pay_len : 1u);
    if (!pay) {
        fclose(in);
        return -1;
    }
    if (read_all(in, pay, pay_len) != 0) {
        free(pay);
        fclose(in);
        fprintf(stderr, "session: truncated payload\n");
        return -1;
    }
    fclose(in);

    if (nerva_persist_crc32(pay, pay_len) != h.payload_crc32) {
        free(pay);
        fprintf(stderr, "session: payload CRC mismatch\n");
        return -1;
    }

    const uint8_t *eng = pay;
    size_t eng_len = (size_t)h.engine_size;
    const uint8_t *flu = pay + eng_len;
    size_t flu_len = (size_t)h.fluency_size;
    const uint8_t *chat = flu + flu_len;
    size_t chat_len = (size_t)h.chat_size;

    if (eng_len + flu_len + chat_len != pay_len) {
        free(pay);
        return -1;
    }

    /* Init engine with saved caps (or reuse if already large enough). */
    int eng_owned = 0;
    if (e->nodes == NULL) {
        NervaConfig cfg = nerva_config_default();
        cfg.max_nodes = h.max_nodes ? h.max_nodes : cfg.max_nodes;
        cfg.max_edges = h.max_edges ? h.max_edges : cfg.max_edges;
        cfg.max_names = h.max_names ? h.max_names : cfg.max_names;
        cfg.max_memory_blocks = h.max_memory ? h.max_memory : cfg.max_memory_blocks;
        cfg.max_schemas = h.max_schemas ? h.max_schemas : cfg.max_schemas;
        cfg.weight_max_q8_8 = (nerva_q8_8_t)h.weight_max_q8_8;
        if (cfg.weight_max_q8_8 < NERVA_WEIGHT_MAX_Q8_8) {
            /* Chat uses INT16_MAX; keep stored value. */
        }
        if (nerva_engine_init(e, cfg) != 0) {
            free(pay);
            fprintf(stderr, "session: engine init failed\n");
            return -1;
        }
        eng_owned = 1;
    }

    char eng_path[1024];
    if (path_tmp(path, eng_path, sizeof(eng_path), ".eng.tmp") != 0) {
        if (eng_owned) {
            nerva_engine_free(e);
        }
        free(pay);
        return -1;
    }
    FILE *ef = fopen(eng_path, "wb");
    if (!ef || write_all(ef, eng, eng_len) != 0) {
        if (ef) {
            fclose(ef);
        }
        remove(eng_path);
        if (eng_owned) {
            nerva_engine_free(e);
        }
        free(pay);
        fprintf(stderr, "session: cannot stage engine blob\n");
        return -1;
    }
    fclose(ef);

    if (nerva_persist_load(e, eng_path) != 0) {
        remove(eng_path);
        if (eng_owned) {
            nerva_engine_free(e);
        }
        free(pay);
        fprintf(stderr, "session: engine load failed\n");
        return -1;
    }
    remove(eng_path);

    /* Fluency init then overlay side state. */
    if (m->e != NULL || m->fast_out != NULL) {
        fluency_free(m);
    }
    memset(m, 0, sizeof(*m));
    /* order patched from side blob; temporary order=1 for init. */
    if (fluency_init(m, e, 1u) != 0) {
        if (eng_owned) {
            nerva_engine_free(e);
        }
        free(pay);
        fprintf(stderr, "session: fluency_init failed\n");
        return -1;
    }
    if (fluency_side_decode(m, flu, flu_len) != 0) {
        fluency_free(m);
        if (eng_owned) {
            nerva_engine_free(e);
        }
        free(pay);
        fprintf(stderr, "session: fluency side decode failed\n");
        return -1;
    }

    if (chat_len > 0u && chat_out) {
        if (chat_cap < chat_len) {
            fluency_free(m);
            if (eng_owned) {
                nerva_engine_free(e);
            }
            free(pay);
            fprintf(stderr, "session: chat section present but buffer too small\n");
            return -1;
        }
        memcpy(chat_out, chat, chat_len);
        if (chat_len_out) {
            *chat_len_out = chat_len;
        }
    } else if (chat_len_out) {
        *chat_len_out = chat_len;
    }

    free(pay);
    (void)sess_crc_update;
    return 0;
}

int fluency_save(const FluencyModel *m, const char *path) {
    return fluency_session_save(m, path, NULL, 0);
}

int fluency_load(NervaEngine *e, FluencyModel *m, const char *path) {
    return fluency_session_load(e, m, path, NULL, 0, NULL);
}
