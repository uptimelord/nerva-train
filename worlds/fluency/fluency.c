// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "fluency.h"

#include "nerva_config.h"
#include "nerva_engine.h"
#include "nerva_event.h"
#include "nerva_graph.h"
#include "nerva_learning.h"
#include "nerva_math.h"
#include "nerva_trace.h"
#include "nerva_work.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* Backoff weighting: a higher-order match emits more charge than a lower-order
 * one, so a specific long context outvotes generic short-context smoothing.
 * Pure counts still live on the edges; this only scales the per-observation
 * increment by order. Capped so a small corpus stays inside int16 weights. */
static int32_t fluency_order_step(uint32_t k) {
    uint32_t shift = 3u * (k - 1u);
    if (shift > 12u) {
        shift = 12u; /* cap single-order increment at 4096 */
    }
    return (int32_t)(1u << shift);
}

static void fluency_tok_name(char *buf, size_t cap, flu_tok_t tok) {
    /* 4 hex digits: covers FLUENCY_VOCAB up to 16384. */
    snprintf(buf, cap, "FT:%04x", (unsigned)tok);
}

static void fluency_ctx_name(char *buf, size_t cap, const flu_tok_t *bytes, uint32_t k) {
    size_t used = (size_t)snprintf(buf, cap, "FC%u:", (unsigned)k);
    for (uint32_t i = 0; i < k && used + 4u < cap; ++i) {
        used += (size_t)snprintf(buf + used, cap - used, "%04x", (unsigned)bytes[i]);
    }
}

/* ---- offline name→node_id index (materialize + predict O(1)) -------------- */
/* Root cause of SMART wall: nerva_find_node_by_name / nerva_intern_name are
 * linear scans. Option (a): fluency-layer hash only; engine untouched.
 * Creates append names directly when the index misses (index is complete for
 * fluency-owned nodes), so create is O(1) amortized too — not O(n²). */

#define FLU_NAME_MAX (8u + 5u * FLUENCY_MAX_ORDER)

typedef struct FluNameSlot {
    char name[FLU_NAME_MAX];
    uint32_t node_id;
} FluNameSlot;

static uint32_t fluency_name_hash(const char *s) {
    uint64_t h = 14695981039346656037ull;
    for (; s && *s; ++s) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ull;
    }
    return (uint32_t)h | 1u;
}

static int fluency_name_grow(FluencyModel *m) {
    size_t ncap = m->name_map_cap ? m->name_map_cap * 2u : 4096u;
    FluNameSlot *ns = (FluNameSlot *)calloc(ncap, sizeof(*ns));
    if (!ns) {
        return -1;
    }
    FluNameSlot *old = (FluNameSlot *)m->name_map;
    for (size_t i = 0; i < m->name_map_cap; ++i) {
        if (!old[i].name[0]) {
            continue;
        }
        size_t idx = (size_t)(fluency_name_hash(old[i].name) % (uint32_t)ncap);
        while (ns[idx].name[0]) {
            idx = (idx + 1u) % ncap;
        }
        ns[idx] = old[i];
    }
    free(old);
    m->name_map = ns;
    m->name_map_cap = ncap;
    return 0;
}

static uint32_t fluency_name_lookup(const FluencyModel *m, const char *name) {
    if (!m || !name || !m->name_map_cap || !m->name_map) {
        return NERVA_INVALID_ID;
    }
    const FluNameSlot *slots = (const FluNameSlot *)m->name_map;
    size_t idx = (size_t)(fluency_name_hash(name) % (uint32_t)m->name_map_cap);
    for (;;) {
        if (!slots[idx].name[0]) {
            return NERVA_INVALID_ID;
        }
        if (strcmp(slots[idx].name, name) == 0) {
            return slots[idx].node_id;
        }
        idx = (idx + 1u) % m->name_map_cap;
    }
}

static int fluency_name_put(FluencyModel *m, const char *name, uint32_t id) {
    if (!m->name_map_cap || m->name_map_n * 10u >= m->name_map_cap * 7u) {
        if (fluency_name_grow(m) != 0) {
            return -1;
        }
    }
    FluNameSlot *slots = (FluNameSlot *)m->name_map;
    size_t idx = (size_t)(fluency_name_hash(name) % (uint32_t)m->name_map_cap);
    for (;;) {
        if (!slots[idx].name[0]) {
            strncpy(slots[idx].name, name, FLU_NAME_MAX - 1u);
            slots[idx].name[FLU_NAME_MAX - 1u] = '\0';
            slots[idx].node_id = id;
            m->name_map_n++;
            return 0;
        }
        if (strcmp(slots[idx].name, name) == 0) {
            slots[idx].node_id = id;
            return 0;
        }
        idx = (idx + 1u) % m->name_map_cap;
    }
}

/* Append node without create_node's O(n) name_id scan. Fluency name index
 * already guarantees uniqueness of the names we mint. */
static uint32_t fluency_append_node(NervaEngine *e, uint32_t name_id) {
    if (!e || e->node_count >= e->node_cap) {
        return NERVA_INVALID_ID;
    }
    uint32_t id = e->node_count;
    NervaNode *n = &e->nodes[id];
    memset(n, 0, sizeof(*n));
    n->id = id;
    n->name_id = name_id;
    n->memory_block = UINT32_MAX;
    /* Match nerva_apply_node_defaults used by create_node. */
    n->v = e->cfg.v_rest_q8_8;
    n->v_rest = e->cfg.v_rest_q8_8;
    n->v_reset = e->cfg.v_reset_q8_8;
    n->theta_fire = e->cfg.theta_fire_q8_8;
    n->refractory_max = e->cfg.refractory_ticks;
    e->node_count++;
    e->adjacency_valid = 0;
    return id;
}

/* Append edge without create_edge's O(E) duplicate scan. Caller (slot index)
 * guarantees (src,tgt,rel) is new for fluency count edges. */
static uint32_t fluency_append_edge(NervaEngine *e, uint32_t source, uint32_t target,
                                   uint16_t relation) {
    if (!e || source >= e->node_count || target >= e->node_count || relation == NERVA_REL_NONE) {
        return NERVA_INVALID_ID;
    }
    if (e->edge_count >= e->edge_cap) {
        return NERVA_INVALID_ID;
    }
    uint32_t id = e->edge_count;
    NervaEdge *ed = &e->edges[id];
    memset(ed, 0, sizeof(*ed));
    ed->source = source;
    ed->target = target;
    ed->relation = relation;
    ed->weight = e->cfg.default_weight_q8_8;
    ed->gate = (nerva_uq0_16_t)NERVA_UQ0_16_ONE;
    ed->delay_ticks = e->cfg.edge_delay_ticks;
    ed->memory_block = UINT32_MAX;
    ed->gate_ctrl = NERVA_INVALID_ID;
    e->edge_count++;
    e->adjacency_valid = 0;
    return id;
}

/* Index is authoritative for fluency-created names. Lookup hit → O(1).
 * Miss + create: append engine name + node (no linear scans) + index. */
static uint32_t fluency_named_node(FluencyModel *m, const char *name, int create) {
    uint32_t id = fluency_name_lookup(m, name);
    if (id != NERVA_INVALID_ID) {
        m->name_hits++;
        return id;
    }
    if (!create) {
        return NERVA_INVALID_ID;
    }
    NervaEngine *e = m->e;
    if (!e || e->name_count >= e->name_cap) {
        return NERVA_INVALID_ID;
    }
    char *copy = (char *)malloc(strlen(name) + 1u);
    if (!copy) {
        return NERVA_INVALID_ID;
    }
    memcpy(copy, name, strlen(name) + 1u);
    e->names[e->name_count] = copy;
    e->name_count++;
    uint32_t name_id = e->name_count; /* 1-based, matches nerva_intern_name */
    id = fluency_append_node(e, name_id);
    if (id == NERVA_INVALID_ID) {
        return NERVA_INVALID_ID;
    }
    if (fluency_name_put(m, name, id) != 0) {
        return NERVA_INVALID_ID;
    }
    m->name_creates++;
    return id;
}

/* ---- edge-id index (where each count edge lives) --------------------------- */

static uint64_t fluency_slot_key(uint32_t src, flu_tok_t tok) {
    /* FLU_TOK_KEY_BITS: 14 for V≤16k, 16 for LLM V≤65k; +1 so 0 stays empty. */
    return (((uint64_t)src << FLU_TOK_KEY_BITS) | (uint64_t)tok) + 1u;
}

static int fluency_slots_grow(FluencyModel *m, size_t want) {
    size_t cap = m->slot_cap ? m->slot_cap : 1024u;
    while (cap < want) {
        cap *= 2u;
    }
    FluencyEdgeSlot *slots = calloc(cap, sizeof(*slots));
    if (!slots) {
        return -1;
    }
    for (size_t i = 0; i < m->slot_cap; ++i) {
        if (m->slots[i].key == 0u) {
            continue;
        }
        size_t h = (size_t)(m->slots[i].key % cap);
        while (slots[h].key != 0u) {
            h = (h + 1u) % cap;
        }
        slots[h] = m->slots[i];
    }
    free(m->slots);
    m->slots = slots;
    m->slot_cap = cap;
    return 0;
}

static uint32_t *fluency_slot_ref(FluencyModel *m, uint64_t key) {
    if (m->slot_count * 10u >= m->slot_cap * 7u) {
        if (fluency_slots_grow(m, m->slot_cap ? m->slot_cap * 2u : 1024u) != 0) {
            return NULL;
        }
    }
    size_t h = (size_t)(key % m->slot_cap);
    while (m->slots[h].key != 0u) {
        if (m->slots[h].key == key) {
            return &m->slots[h].edge_id;
        }
        h = (h + 1u) % m->slot_cap;
    }
    m->slots[h].key = key;
    m->slots[h].edge_id = NERVA_INVALID_ID;
    m->slot_count++;
    return &m->slots[h].edge_id;
}

/* ---- nodes ----------------------------------------------------------------- */

static uint32_t fluency_get_tok(FluencyModel *m, flu_tok_t byte) {
    if (m->tok_node[byte] != NERVA_INVALID_ID) {
        return m->tok_node[byte];
    }
    char name[16];
    fluency_tok_name(name, sizeof(name), byte);
    uint32_t id = fluency_named_node(m, name, 1);
    if (id == NERVA_INVALID_ID) {
        return NERVA_INVALID_ID;
    }
    /* Integrator: never spikes, so it accumulates weighted votes we can read. */
    m->e->nodes[id].theta_fire = INT16_MAX;
    m->e->nodes[id].v_rest = 0;
    m->e->nodes[id].v_reset = 0;
    m->tok_node[byte] = id;
    if (!m->seen[byte]) {
        m->seen[byte] = 1u;
        m->vocab_count++;
    }
    return id;
}

int fluency_ensure_tok(FluencyModel *m, flu_tok_t tok) {
    if (!m || tok >= FLUENCY_VOCAB) {
        return -1;
    }
    return fluency_get_tok(m, tok) == NERVA_INVALID_ID ? -1 : 0;
}

static uint32_t fluency_get_ctx(FluencyModel *m, const flu_tok_t *bytes, uint32_t k) {
    char name[8u + 5u * FLUENCY_MAX_ORDER];
    fluency_ctx_name(name, sizeof(name), bytes, k);
    uint32_t id = fluency_named_node(m, name, 1);
    if (id != NERVA_INVALID_ID) {
        m->e->nodes[id].theta_fire = NERVA_Q8_8_ONE; /* one activation makes it fire */
    }
    return id;
}

int fluency_init(FluencyModel *m, NervaEngine *e, uint32_t order) {
    if (!m || !e || order == 0u || order > FLUENCY_MAX_ORDER) {
        return -1;
    }
    memset(m, 0, sizeof(*m));
    m->e = e;
    m->order = order;
    m->count_soft_nused = 7000000u; /* sealed M_v2; 0 = exact */
    m->weight_map = FLU_WEIGHT_MAP_RAW; /* legacy; scale tools opt into CTX_NORM */
    m->fast_slots = FLU_FAST_SLOTS_DEFAULT;
    m->fast_w0 = FLU_FAST_W0_DEFAULT;
    m->fast_half_life = FLU_FAST_HALF_LIFE_DEFAULT;
    m->fast_lambda = 0; /* baseline bit-exact until caller enables blend */
    m->fast_beta = FLU_FAST_BETA_DEFAULT;
    m->fast_delta = FLU_FAST_DELTA_DEFAULT;
    /* Context organs default OFF (Organism Gate criterion 2: bit-exact inert). */
    m->mood_on = 0;
    m->mood_k = (uint8_t)FLU_MOOD_K_DEFAULT;
    m->mood_half_life = FLU_MOOD_HALF_LIFE_DEFAULT;
    m->mood_scale = FLU_MOOD_SCALE_DEFAULT;
    m->mood_aff = NULL;
    m->mood_inject_ops = 0;
    m->gain_on = 0;
    m->gain_window = (uint16_t)FLU_GAIN_WINDOW_DEFAULT;
    m->gain_scale = FLU_GAIN_SCALE_DEFAULT;
    m->gain_topk = 0;
    m->gain_cooc = NULL;
    m->gain_tok = NULL;
    m->gain_vidx = NULL;
    m->gain_inject_ops = 0;
    /* Phonebook hubs default OFF (bit-exact inert). */
    m->hub_on = 0;
    m->hub_h = 0;
    m->hub_scale = FLU_HUB_SCALE_DEFAULT;
    m->hub_conf_margin = FLU_HUB_CONF_MARGIN_DEFAULT;
    m->tok_hubs = NULL;
    m->hub_mem_tok = NULL;
    m->hub_mem_w = NULL;
    m->hub_mem_n = NULL;
    m->hub_dial_count = 0;
    m->hub_dial_ops = 0;
    m->hub_predict_count = 0;
    m->last_dialed = 0;
    m->last_margin = 0;
    m->hub_last_ops = 0;
    memset(&m->last_why, 0, sizeof(m->last_why));
    /* Instance binding default OFF (bit-exact inert). Fixed side pool. */
    m->instance_on = 0;
    m->instance_cap = FLU_INSTANCE_CAP_DEFAULT;
    m->instance_pool = NULL;
    m->instance_n = 0;
    m->instance_deposits = 0;
    m->instance_evicts = 0;
    m->instance_hits = 0;
    m->instance_inject_ops = 0;
    m->instance_last_cue = (flu_tok_t)FLUENCY_VOCAB;
    m->instance_last_src = (flu_tok_t)FLUENCY_VOCAB;
    m->instance_last_tgt = (flu_tok_t)FLUENCY_VOCAB;
    m->instance_last_weff = 0;
    m->instance_last_hit = 0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        m->tok_node[b] = NERVA_INVALID_ID;
        m->vidx[b] = UINT32_MAX;
    }
    if (fluency_name_grow(m) != 0) {
        return -1;
    }
    if (fluency_slots_grow(m, 1024u) != 0) {
        free(m->name_map);
        m->name_map = NULL;
        m->name_map_cap = 0;
        return -1;
    }
    /* Fixed fast-edge pool: V * slots slots, allocated once (not hot-path). */
    size_t nfast = (size_t)FLUENCY_VOCAB * (size_t)m->fast_slots;
    m->fast_out = calloc(nfast, sizeof(*m->fast_out));
    if (!m->fast_out) {
        free(m->slots);
        m->slots = NULL;
        return -1;
    }
    for (size_t i = 0; i < nfast; ++i) {
        m->fast_out[i].target = (flu_tok_t)FLUENCY_VOCAB; /* empty */
    }
    /* Fixed instance side pool (budget-printed by callers/tests). */
    m->instance_pool = calloc((size_t)m->instance_cap, sizeof(*m->instance_pool));
    if (!m->instance_pool) {
        free(m->fast_out);
        m->fast_out = NULL;
        free(m->slots);
        m->slots = NULL;
        return -1;
    }
    return 0;
}

void fluency_free(FluencyModel *m) {
    if (!m) {
        return;
    }
    free(m->tscar_slots);
    m->tscar_slots = NULL;
    free(m->kn_cont);
    m->kn_cont = NULL;
    m->kn_cont_total = 0;
    free(m->ctw_tok_of_node);
    m->ctw_tok_of_node = NULL;
    m->ctw_tok_cap = 0;
    free(m->slots);
    m->slots = NULL;
    m->slot_cap = 0;
    m->slot_count = 0;
    free(m->name_map);
    m->name_map = NULL;
    m->name_map_cap = 0;
    m->name_map_n = 0;
    free(m->fast_out);
    m->fast_out = NULL;
    free(m->mood_aff);
    m->mood_aff = NULL;
    free(m->gain_cooc);
    m->gain_cooc = NULL;
    free(m->gain_tok);
    m->gain_tok = NULL;
    free(m->gain_vidx);
    m->gain_vidx = NULL;
    free(m->tok_hubs);
    m->tok_hubs = NULL;
    free(m->hub_mem_tok);
    m->hub_mem_tok = NULL;
    free(m->hub_mem_w);
    m->hub_mem_w = NULL;
    free(m->hub_mem_n);
    m->hub_mem_n = NULL;
    m->hub_h = 0;
    free(m->instance_pool);
    m->instance_pool = NULL;
    m->instance_n = 0;
    m->instance_cap = 0;
    free(m->embed);
    m->embed = NULL;
    m->embed_dim = 0;
    m->vdim = 0;
}

int fluency_rebuild_name_index(FluencyModel *m) {
    if (!m || !m->e) {
        return -1;
    }
    free(m->name_map);
    m->name_map = NULL;
    m->name_map_cap = 0;
    m->name_map_n = 0;
    if (fluency_name_grow(m) != 0) {
        return -1;
    }
    /* Ample load factor for node_count names. */
    while (m->name_map_cap < (size_t)m->e->node_count * 2u + 64u) {
        if (fluency_name_grow(m) != 0) {
            return -1;
        }
    }
    for (uint32_t i = 0; i < m->e->node_count; ++i) {
        uint32_t nid = m->e->nodes[i].name_id;
        if (nid == 0u || nid > m->e->name_count) {
            continue;
        }
        const char *nm = m->e->names[nid - 1u];
        if (!nm || !nm[0]) {
            continue;
        }
        if (fluency_name_put(m, nm, i) != 0) {
            return -1;
        }
    }
    return 0;
}

int fluency_name_map_export(const FluencyModel *m, uint8_t **out, size_t *out_len) {
    if (!m || !out || !out_len) {
        return -1;
    }
    *out = NULL;
    *out_len = 0;
    const FluNameSlot *slots = (const FluNameSlot *)m->name_map;
    uint32_t n = 0;
    for (size_t i = 0; i < m->name_map_cap; ++i) {
        if (slots && slots[i].name[0]) {
            n++;
        }
    }
    /* u32 count + n * (u8 len + name + u32 id), len < FLU_NAME_MAX */
    size_t need = sizeof(uint32_t);
    for (size_t i = 0; i < m->name_map_cap; ++i) {
        if (!slots || !slots[i].name[0]) {
            continue;
        }
        size_t L = strlen(slots[i].name);
        if (L > 255u) {
            L = 255u;
        }
        need += 1u + L + sizeof(uint32_t);
    }
    uint8_t *buf = (uint8_t *)malloc(need ? need : 1u);
    if (!buf) {
        return -1;
    }
    size_t o = 0;
    memcpy(buf + o, &n, sizeof(n));
    o += sizeof(n);
    for (size_t i = 0; i < m->name_map_cap; ++i) {
        if (!slots || !slots[i].name[0]) {
            continue;
        }
        size_t L = strlen(slots[i].name);
        if (L > 255u) {
            L = 255u;
        }
        buf[o++] = (uint8_t)L;
        memcpy(buf + o, slots[i].name, L);
        o += L;
        memcpy(buf + o, &slots[i].node_id, sizeof(uint32_t));
        o += sizeof(uint32_t);
    }
    if (o != need) {
        free(buf);
        return -1;
    }
    *out = buf;
    *out_len = need;
    return 0;
}

int fluency_name_map_import(FluencyModel *m, const uint8_t *buf, size_t len) {
    if (!m || !buf || len < sizeof(uint32_t)) {
        return -1;
    }
    uint32_t n = 0;
    size_t o = 0;
    memcpy(&n, buf + o, sizeof(n));
    o += sizeof(n);
    free(m->name_map);
    m->name_map = NULL;
    m->name_map_cap = 0;
    m->name_map_n = 0;
    if (fluency_name_grow(m) != 0) {
        return -1;
    }
    while (m->name_map_cap < (size_t)n * 2u + 64u) {
        if (fluency_name_grow(m) != 0) {
            return -1;
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (o >= len) {
            return -1;
        }
        uint8_t L = buf[o++];
        if (o + L + sizeof(uint32_t) > len) {
            return -1;
        }
        char name[FLU_NAME_MAX];
        if (L >= FLU_NAME_MAX) {
            return -1;
        }
        memcpy(name, buf + o, L);
        name[L] = '\0';
        o += L;
        uint32_t id = 0;
        memcpy(&id, buf + o, sizeof(id));
        o += sizeof(id);
        if (fluency_name_put(m, name, id) != 0) {
            return -1;
        }
    }
    if (o != len) {
        return -1;
    }
    return 0;
}

/* ---- fast edges: delta-rule deposit, lazy stream decay, λ-blend inject ----- */

void fluency_stream_advance(FluencyModel *m) {
    if (m) {
        m->stream_pos++;
    }
}

/* Scale x by UQ0.16 factor. ONE (65535) is exact 1.0 so β=1 preserves one-shot. */
static int32_t fluency_scale_uq0_16(nerva_uq0_16_t f, int32_t x) {
    if (f == 0u || x == 0) {
        return 0;
    }
    if (f >= NERVA_UQ0_16_ONE) {
        return x;
    }
    return (int32_t)(((int64_t)f * (int64_t)x) / (int64_t)NERVA_UQ0_16_ONE);
}

/* Effective weight: w_eff = weight >> (elapsed / half_life). */
static nerva_q8_8_t fluency_fast_weff(const FluencyModel *m, nerva_q8_8_t base_w,
                                      uint32_t deposit_pos) {
    if (!m || m->fast_half_life == 0u || base_w <= 0) {
        return 0;
    }
    uint32_t elapsed = (m->stream_pos >= deposit_pos) ? (m->stream_pos - deposit_pos) : UINT32_MAX;
    uint32_t shifts = elapsed / m->fast_half_life;
    if (shifts >= 15u) {
        return 0;
    }
    return (nerva_q8_8_t)((uint16_t)base_w >> shifts);
}

/* Delta-rule write (Schlag/Irie/Schmidhuber): W += β(v − v̄)⊗k on sparse slots.
 * Read-before-write over A's own slots only (O(fast_slots)). */
void fluency_fast_deposit(FluencyModel *m, flu_tok_t a, flu_tok_t b) {
    if (!m || !m->fast_out || a >= FLUENCY_VOCAB || b >= FLUENCY_VOCAB) {
        return;
    }
    FluencyFastSlot *base = &m->fast_out[(size_t)a * (size_t)m->fast_slots];
    int target_i = -1;
    int free_i = -1;
    int weak_i = 0;
    nerva_q8_8_t weak_w = INT16_MAX;
    nerva_q8_8_t target_weff = 0;

    /* Pass 1: read all of A's slots (deposit cost ≤ fast_slots), depress competitors. */
    for (uint16_t i = 0; i < m->fast_slots; ++i) {
        m->fast_deposit_reads++;
        if (base[i].target >= FLUENCY_VOCAB) {
            if (free_i < 0) {
                free_i = (int)i;
            }
            continue;
        }
        nerva_q8_8_t weff = fluency_fast_weff(m, base[i].weight, base[i].deposit_pos);
        if (weff <= 0) {
            /* Reclaim zeroed (decayed) slot. */
            base[i].target = (flu_tok_t)FLUENCY_VOCAB;
            base[i].weight = 0;
            if (free_i < 0) {
                free_i = (int)i;
            }
            continue;
        }
        if (base[i].target == b) {
            target_i = (int)i;
            target_weff = weff;
            continue;
        }
        /* Depress competitors from pre-write weff. δ=0 ⇒ sub=0, leave age intact. */
        int32_t sub = fluency_scale_uq0_16(
            m->fast_delta, fluency_scale_uq0_16(m->fast_beta, (int32_t)weff));
        if (sub > 0) {
            int32_t nw = (int32_t)weff - sub;
            if (nw <= 0) {
                base[i].target = (flu_tok_t)FLUENCY_VOCAB;
                base[i].weight = 0;
                if (free_i < 0) {
                    free_i = (int)i;
                }
                continue;
            }
            base[i].weight = (nerva_q8_8_t)nw;
            base[i].deposit_pos = m->stream_pos; /* rematerialize at new strength */
            weff = (nerva_q8_8_t)nw;
        }
        if (weff < weak_w) {
            weak_w = weff;
            weak_i = (int)i;
        }
    }

    /* Pass 2: strengthen target by error toward W0. Missing ⇒ weff=0 ⇒ near-full write. */
    int32_t w0 = (int32_t)m->fast_w0;
    int32_t err = w0 - (int32_t)target_weff;
    if (err < 0) {
        err = 0;
    }
    int32_t add = fluency_scale_uq0_16(m->fast_beta, err);
    int32_t new_w = (int32_t)target_weff + add;
    if (new_w > w0) {
        new_w = w0;
    }
    if (new_w <= 0) {
        /* β=0 and no prior mass: nothing to store. */
        if (target_i >= 0) {
            base[target_i].target = (flu_tok_t)FLUENCY_VOCAB;
            base[target_i].weight = 0;
        }
        return;
    }

    int use;
    if (target_i >= 0) {
        use = target_i;
    } else if (free_i >= 0) {
        use = free_i;
    } else {
        use = weak_i;
    }
    base[use].target = b;
    base[use].weight = (nerva_q8_8_t)new_w;
    base[use].deposit_pos = m->stream_pos;
}

/* Inject λ · w_eff from the most recent context token into integrators.
 * Cost: O(fast_slots), independent of history length. Guarded by λ != 0. */
static void fluency_fast_inject(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len) {
    if (!m || !m->fast_out || m->fast_lambda == 0u || ctx_len == 0u) {
        return;
    }
    flu_tok_t src = ctx[ctx_len - 1u];
    if (src >= FLUENCY_VOCAB) {
        return;
    }
    FluencyFastSlot *base = &m->fast_out[(size_t)src * (size_t)m->fast_slots];
    for (uint16_t i = 0; i < m->fast_slots; ++i) {
        m->fast_read_ops++;
        flu_tok_t tgt = base[i].target;
        if (tgt >= FLUENCY_VOCAB) {
            continue;
        }
        nerva_q8_8_t weff = fluency_fast_weff(m, base[i].weight, base[i].deposit_pos);
        if (weff <= 0) {
            continue;
        }
        /* inject = (weff * λ) in Q8.8, λ is UQ0.16 */
        int32_t delta = ((int32_t)weff * (int32_t)m->fast_lambda) >> 16;
        if (delta <= 0) {
            continue;
        }
        uint32_t nid = m->tok_node[tgt];
        if (nid == NERVA_INVALID_ID || !m->seen[tgt]) {
            continue; /* never grow graph on the predict path */
        }
        m->e->nodes[nid].v = nerva_q8_8_saturating_add(m->e->nodes[nid].v, (nerva_q8_8_t)delta);
    }
}

/* ---- instance binding: conjunction-keyed side pool (default OFF) ---------- */

void fluency_instance_set(FluencyModel *m, int on) {
    if (m) {
        m->instance_on = on ? 1u : 0u;
    }
}

void fluency_skip_set(FluencyModel *m, int on) {
    if (m) {
        m->skip_on = on ? 1u : 0u;
    }
}

typedef struct FluTscarSlot {
    flu_tok_t key;
    flu_tok_t val;
    uint16_t str;
    uint8_t used;
} FluTscarSlot;

void fluency_text_organ_set(FluencyModel *m, int on) {
    if (!m) {
        return;
    }
    m->text_organ = on ? 1u : 0u;
    if (on) {
        fluency_kn_set(m, 1);
        fluency_instance_set(m, 1);
        /* Instance inject requires fast_lambda > 0. */
        m->fast_lambda = (nerva_uq0_16_t)32768u; /* 0.5 */
    } else {
        fluency_kn_set(m, 0);
        fluency_instance_set(m, 0);
        m->fast_lambda = 0;
    }
}

int fluency_text_organ_on(const FluencyModel *m) {
    return m && m->text_organ;
}

void fluency_tscar_organ_set(FluencyModel *m, int on) {
    if (!m) {
        return;
    }
    m->tscar_organ = on ? 1u : 0u;
    if (on) {
        m->tscar_mix = 0.25;
        if (!m->tscar_slots) {
            m->tscar_cap = FLU_TSCAR_SLOT_CAP;
            m->tscar_slots = calloc((size_t)m->tscar_cap, sizeof(FluTscarSlot));
            m->tscar_used = 0;
            m->tscar_writes = 0;
        }
    } else {
        m->tscar_mix = 0.0;
        /* Keep table allocated for reuse; empty path when mix=0 / organ off. */
    }
}

int fluency_tscar_organ_on(const FluencyModel *m) {
    return m && m->tscar_organ;
}

static void fluency_tscar_write(FluencyModel *m, flu_tok_t key, flu_tok_t val) {
    if (!m || !m->tscar_organ || !m->tscar_slots || m->tscar_cap == 0u) {
        return;
    }
    FluTscarSlot *slots = (FluTscarSlot *)m->tscar_slots;
    int hit = -1, empty = -1;
    for (uint32_t i = 0; i < m->tscar_cap; ++i) {
        if (!slots[i].used) {
            if (empty < 0) {
                empty = (int)i;
            }
            continue;
        }
        if (slots[i].key == key) {
            hit = (int)i;
            break;
        }
    }
    int idx = hit >= 0 ? hit : (empty >= 0 ? empty : (int)(m->tscar_writes % m->tscar_cap));
    if (!slots[idx].used) {
        m->tscar_used++;
    }
    slots[idx].used = 1;
    slots[idx].key = key;
    slots[idx].val = val;
    if (slots[idx].str < 60000u) {
        slots[idx].str++;
    }
    m->tscar_writes++;
    m->tscar_last_slot = (uint32_t)idx;
    m->tscar_last_key = key;
    m->tscar_last_val = val;
}

/* Mix TSCAR mass into dist (already normalized). No-op if organ off / mix=0. */
static void fluency_tscar_mix_dist(FluencyModel *m, flu_tok_t key, double *dist) {
    if (!m || !m->tscar_organ || m->tscar_mix <= 0.0 || !m->tscar_slots || !dist) {
        return;
    }
    FluTscarSlot *slots = (FluTscarSlot *)m->tscar_slots;
    double mass[FLUENCY_VOCAB];
    memset(mass, 0, sizeof(mass));
    double sum = 0.0;
    uint32_t best_i = 0;
    uint16_t best_str = 0;
    int found = 0;
    for (uint32_t i = 0; i < m->tscar_cap; ++i) {
        if (!slots[i].used || slots[i].key != key) {
            continue;
        }
        if (slots[i].val < FLUENCY_VOCAB) {
            double w = (double)slots[i].str;
            mass[slots[i].val] += w;
            sum += w;
            if (slots[i].str >= best_str) {
                best_str = slots[i].str;
                best_i = i;
                found = 1;
            }
        }
    }
    if (!found || sum <= 0.0) {
        return;
    }
    m->tscar_last_slot = best_i;
    m->tscar_last_key = key;
    m->tscar_last_val = slots[best_i].val;
    double lam = m->tscar_mix;
    if (lam > 1.0) {
        lam = 1.0;
    }
    double out_sum = 0.0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        double pscar = mass[b] / sum;
        dist[b] = (1.0 - lam) * dist[b] + lam * pscar;
        out_sum += dist[b];
    }
    if (out_sum > 0.0) {
        for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
            dist[b] /= out_sum;
        }
    }
}

void fluency_organs_observe_stream(FluencyModel *m, const flu_tok_t *text, size_t len) {
    if (!m || !text || len < 2u) {
        return;
    }
    if (!m->text_organ && !m->tscar_organ) {
        return;
    }
    for (size_t i = 1; i < len; ++i) {
        flu_tok_t prev = text[i - 1u];
        flu_tok_t next = text[i];
        if (m->tscar_organ) {
            fluency_tscar_write(m, prev, next);
        }
        if (m->text_organ && i >= 2u) {
            /* (cue=t-2, src=t-1) → next : name-bind / induction memory */
            fluency_instance_deposit(m, text[i - 2u], prev, next);
        }
        fluency_stream_advance(m);
    }
}

void fluency_kn_set(FluencyModel *m, int on) {
    if (m) {
        m->kn_on = on ? 1u : 0u;
    }
}

/* Fire the skip-1 gapped-context node (w-2, _) if built. Skip contexts live
 * in the FC1 namespace keyed ctx_token + FLUENCY_VOCAB, so lookup reuses the
 * ordinary ctx-name machinery. Returns 1 if a node was activated. */
static int fluency_skip_fire(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len) {
    if (!m->skip_on || ctx_len < 2u) {
        return 0;
    }
#if !FLU_SKIP_PACK_OK
    /* Skip packs (tok+V) into flu_tok_t; impossible for LLM V>32768. */
    (void)ctx;
    return 0;
#else
    flu_tok_t sctx = (flu_tok_t)(ctx[ctx_len - 2u] + FLUENCY_VOCAB);
    char name[8u + 5u * FLUENCY_MAX_ORDER];
    fluency_ctx_name(name, sizeof(name), &sctx, 1u);
    uint32_t id = fluency_name_lookup(m, name);
    if (id == NERVA_INVALID_ID) {
        return 0;
    }
    nerva_activate_node(m->e, id, NERVA_Q8_8_ONE);
    return 1;
#endif
}

int fluency_instance_on(const FluencyModel *m) {
    return m && m->instance_on ? 1 : 0;
}

size_t fluency_instance_pool_bytes(const FluencyModel *m) {
    if (!m || !m->instance_pool || m->instance_cap == 0u) {
        return 0u;
    }
    return (size_t)m->instance_cap * sizeof(FluencyInstanceSlot);
}

static uint32_t fluency_instance_hash(flu_tok_t cue, flu_tok_t src) {
    uint32_t h = 2166136261u;
    h ^= (uint32_t)cue;
    h *= 16777619u;
    h ^= (uint32_t)src;
    h *= 16777619u;
    return h;
}

/* Reinsert live slots after a delete (tombstone-free open address). */
static void fluency_instance_rehash(FluencyModel *m) {
    if (!m || !m->instance_pool || m->instance_cap == 0u) {
        return;
    }
    size_t cap = (size_t)m->instance_cap;
    FluencyInstanceSlot *tmp =
        (FluencyInstanceSlot *)malloc(cap * sizeof(FluencyInstanceSlot));
    if (!tmp) {
        return; /* leave table as-is; next insert may fail soft */
    }
    memcpy(tmp, m->instance_pool, cap * sizeof(FluencyInstanceSlot));
    memset(m->instance_pool, 0, cap * sizeof(FluencyInstanceSlot));
    m->instance_n = 0;
    for (size_t i = 0; i < cap; ++i) {
        if (!tmp[i].live) {
            continue;
        }
        uint32_t h = fluency_instance_hash(tmp[i].cue, tmp[i].src);
        size_t idx = (size_t)(h % (uint32_t)cap);
        for (size_t probe = 0; probe < cap; ++probe) {
            size_t j = (idx + probe) % cap;
            if (!m->instance_pool[j].live) {
                m->instance_pool[j] = tmp[i];
                m->instance_n++;
                break;
            }
        }
    }
    free(tmp);
}

/* Find slot index for (cue,src); is_empty=1 if landed on free for insert. */
static int fluency_instance_find(FluencyModel *m, flu_tok_t cue, flu_tok_t src, int *is_empty) {
    if (is_empty) {
        *is_empty = 0;
    }
    if (!m || !m->instance_pool || m->instance_cap == 0u) {
        return -1;
    }
    size_t cap = (size_t)m->instance_cap;
    size_t idx = (size_t)(fluency_instance_hash(cue, src) % (uint32_t)cap);
    for (size_t probe = 0; probe < cap; ++probe) {
        size_t j = (idx + probe) % cap;
        m->instance_inject_ops++; /* shared counter for find/inject cost */
        if (!m->instance_pool[j].live) {
            if (is_empty) {
                *is_empty = 1;
            }
            return (int)j;
        }
        if (m->instance_pool[j].cue == cue && m->instance_pool[j].src == src) {
            return (int)j;
        }
    }
    return -1; /* full, key absent */
}

static void fluency_instance_evict_one(FluencyModel *m) {
    if (!m || !m->instance_pool || m->instance_n == 0u) {
        return;
    }
    size_t cap = (size_t)m->instance_cap;
    int victim = -1;
    nerva_q8_8_t best_w = INT16_MAX;
    uint32_t best_pos = UINT32_MAX;
    for (size_t i = 0; i < cap; ++i) {
        if (!m->instance_pool[i].live) {
            continue;
        }
        nerva_q8_8_t weff =
            fluency_fast_weff(m, m->instance_pool[i].weight, m->instance_pool[i].deposit_pos);
        if (weff <= 0) {
            /* Prefer fully decayed. */
            victim = (int)i;
            break;
        }
        if (weff < best_w || (weff == best_w && m->instance_pool[i].deposit_pos < best_pos)) {
            best_w = weff;
            best_pos = m->instance_pool[i].deposit_pos;
            victim = (int)i;
        }
    }
    if (victim < 0) {
        return;
    }
    memset(&m->instance_pool[victim], 0, sizeof(m->instance_pool[victim]));
    m->instance_n = m->instance_n > 0u ? m->instance_n - 1u : 0u;
    m->instance_evicts++;
    fluency_instance_rehash(m);
}

void fluency_instance_deposit(FluencyModel *m, flu_tok_t cue, flu_tok_t a, flu_tok_t b) {
    if (!m || !m->instance_on || !m->instance_pool || m->instance_cap == 0u) {
        return;
    }
    if (cue >= FLUENCY_VOCAB || a >= FLUENCY_VOCAB || b >= FLUENCY_VOCAB) {
        return;
    }
    m->instance_deposits++;
    int is_empty = 0;
    int slot = fluency_instance_find(m, cue, a, &is_empty);
    if (slot < 0 || (is_empty && m->instance_n >= m->instance_cap)) {
        fluency_instance_evict_one(m);
        is_empty = 0;
        slot = fluency_instance_find(m, cue, a, &is_empty);
        if (slot < 0) {
            return; /* hard stop: still full after evict (should not happen) */
        }
    }
    FluencyInstanceSlot *s = &m->instance_pool[slot];
    if (!s->live) {
        s->live = 1;
        s->cue = cue;
        s->src = a;
        m->instance_n++;
    }
    /* One-shot full write to W0 (same strength as fresh fast deposit). */
    s->target = b;
    s->weight = m->fast_w0 > 0 ? m->fast_w0 : FLU_FAST_W0_DEFAULT;
    s->deposit_pos = m->stream_pos;
}

/* Inject from hash(prev_cue, A) when cue is live (ctx_len≥2). Local only.
 * Returns 1 if mass landed (caller may then skip type-path fast inject so the
 * cue-selected binding is not tied with bare-A last-binding). */
static int fluency_instance_inject(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len) {
    if (!m || !m->instance_on || !m->instance_pool || m->fast_lambda == 0u || ctx_len < 2u) {
        return 0;
    }
    flu_tok_t cue = ctx[ctx_len - 2u];
    flu_tok_t src = ctx[ctx_len - 1u];
    if (cue >= FLUENCY_VOCAB || src >= FLUENCY_VOCAB) {
        return 0;
    }
    int is_empty = 0;
    int slot = fluency_instance_find(m, cue, src, &is_empty);
    if (slot < 0 || is_empty || !m->instance_pool[slot].live) {
        return 0;
    }
    FluencyInstanceSlot *s = &m->instance_pool[slot];
    flu_tok_t tgt = s->target;
    if (tgt >= FLUENCY_VOCAB || !m->seen[tgt] || m->tok_node[tgt] == NERVA_INVALID_ID) {
        return 0;
    }
    nerva_q8_8_t weff = fluency_fast_weff(m, s->weight, s->deposit_pos);
    if (weff <= 0) {
        return 0;
    }
    int32_t delta = ((int32_t)weff * (int32_t)m->fast_lambda) >> 16;
    if (delta <= 0) {
        return 0;
    }
    m->e->nodes[m->tok_node[tgt]].v =
        nerva_q8_8_saturating_add(m->e->nodes[m->tok_node[tgt]].v, (nerva_q8_8_t)delta);
    m->instance_hits++;
    m->instance_last_cue = cue;
    m->instance_last_src = src;
    m->instance_last_tgt = tgt;
    m->instance_last_weff = weff;
    return 1;
}

/* Mood inject: global tilt from registers × per-token cluster affinity. */
static void fluency_mood_inject(FluencyModel *m) {
    if (!m || !m->mood_on || !m->mood_aff || m->mood_k == 0u || m->mood_scale <= 0) {
        return;
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (!m->seen[b] || m->tok_node[b] == NERVA_INVALID_ID) {
            continue;
        }
        float dot = 0.0f;
        const float *row = m->mood_aff + (size_t)b * (size_t)m->mood_k;
        for (uint8_t k = 0; k < m->mood_k; ++k) {
            m->mood_inject_ops++;
            dot += m->mood_reg[k] * row[k];
        }
        if (dot <= 0.0f) {
            continue;
        }
        /* Clamp inject to mood_scale (Q8.8). */
        if (dot > 1.0f) {
            dot = 1.0f;
        }
        int32_t delta = (int32_t)((float)m->mood_scale * dot + 0.5f);
        if (delta > 0) {
            m->e->nodes[m->tok_node[b]].v =
                nerva_q8_8_saturating_add(m->e->nodes[m->tok_node[b]].v, (nerva_q8_8_t)delta);
        }
    }
}

/* Phonebook 2-hop inject: ctx tokens → hubs → members (graded uq0_16),
 * plus soft-fire of order-1 context pages for co-members of the last token
 * (cat knowledge reaches the race when surface form is kitten).
 * Cost ≤ ~2 × ctx × MEM_MAX × SIZE_CAP edge-signal ops. Fluency pool only. */
static void fluency_hub_inject(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len,
                               uint64_t *ops_out) {
    uint64_t ops = 0;
    if (ops_out) {
        *ops_out = 0;
    }
    if (!m || m->hub_h == 0u || !m->tok_hubs || !m->hub_mem_tok || !m->hub_mem_w || !m->hub_mem_n ||
        m->hub_scale <= 0 || !ctx || ctx_len == 0u) {
        return;
    }
    size_t win = (size_t)m->order;
    if (win > 4u) {
        win = 4u;
    }
    if (win > ctx_len) {
        win = ctx_len;
    }
    size_t start = ctx_len - win;
    for (size_t i = start; i < ctx_len; ++i) {
        flu_tok_t c = ctx[i];
        if (c >= FLUENCY_VOCAB) {
            continue;
        }
        const FluencyHubSlot *slots = m->tok_hubs + (size_t)c * (size_t)FLU_HUB_MEM_MAX;
        for (uint32_t s = 0; s < FLU_HUB_MEM_MAX; ++s) {
            if (slots[s].hub == 0xFFFFu || slots[s].weight == 0u) {
                continue;
            }
            uint16_t h = slots[s].hub;
            if (h >= m->hub_h) {
                continue;
            }
            nerva_uq0_16_t w1 = slots[s].weight;
            uint16_t nmem = m->hub_mem_n[h];
            if (nmem > FLU_HUB_SIZE_CAP) {
                nmem = FLU_HUB_SIZE_CAP;
            }
            size_t base = (size_t)h * (size_t)FLU_HUB_SIZE_CAP;
            for (uint16_t j = 0; j < nmem; ++j) {
                ops++;
                flu_tok_t t = m->hub_mem_tok[base + j];
                if (t >= FLUENCY_VOCAB || !m->seen[t] || m->tok_node[t] == NERVA_INVALID_ID) {
                    continue;
                }
                nerva_uq0_16_t w2 = m->hub_mem_w[base + j];
                int32_t delta =
                    (int32_t)(((int64_t)m->hub_scale * (int64_t)w1 * (int64_t)w2) /
                              ((int64_t)NERVA_UQ0_16_ONE * (int64_t)NERVA_UQ0_16_ONE));
                if (delta > 0) {
                    m->e->nodes[m->tok_node[t]].v = nerva_q8_8_saturating_add(
                        m->e->nodes[m->tok_node[t]].v, (nerva_q8_8_t)delta);
                }
            }
        }
    }
    /* Alias hop: last token's hub siblings soft-fire their o1 context pages. */
    flu_tok_t last = ctx[ctx_len - 1u];
    if (last < FLUENCY_VOCAB) {
        const FluencyHubSlot *slots = m->tok_hubs + (size_t)last * (size_t)FLU_HUB_MEM_MAX;
        int any = 0;
        for (uint32_t s = 0; s < FLU_HUB_MEM_MAX; ++s) {
            if (slots[s].hub == 0xFFFFu || slots[s].weight == 0u) {
                continue;
            }
            uint16_t h = slots[s].hub;
            if (h >= m->hub_h) {
                continue;
            }
            nerva_uq0_16_t w1 = slots[s].weight;
            uint16_t nmem = m->hub_mem_n[h];
            if (nmem > FLU_HUB_SIZE_CAP) {
                nmem = FLU_HUB_SIZE_CAP;
            }
            size_t base = (size_t)h * (size_t)FLU_HUB_SIZE_CAP;
            for (uint16_t j = 0; j < nmem; ++j) {
                ops++;
                flu_tok_t sib = m->hub_mem_tok[base + j];
                if (sib == last || sib >= FLUENCY_VOCAB) {
                    continue;
                }
                char name[8u + 5u * FLUENCY_MAX_ORDER];
                fluency_ctx_name(name, sizeof(name), &sib, 1u);
                uint32_t id = fluency_name_lookup(m, name);
                if (id == NERVA_INVALID_ID) {
                    continue;
                }
                nerva_uq0_16_t w2 = m->hub_mem_w[base + j];
                int32_t act =
                    (int32_t)(((int64_t)NERVA_Q8_8_ONE * (int64_t)w1 * (int64_t)w2) /
                              ((int64_t)NERVA_UQ0_16_ONE * (int64_t)NERVA_UQ0_16_ONE));
                if (act <= 0) {
                    act = 1;
                }
                nerva_activate_node(m->e, id, (nerva_q8_8_t)act);
                any = 1;
            }
        }
        if (any) {
            nerva_tick_n(m->e, 3u);
        }
    }
    if (ops_out) {
        *ops_out = ops;
    }
}

/* Raw charge margin (winner − runner-up) among seen tokens. */
static int32_t fluency_charge_margin(const FluencyModel *m, const double *score, int *argmax_out) {
    double best = -1.0, second = -1.0;
    int argmax = -1;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (!m->seen[b]) {
            continue;
        }
        double v = score[b];
        if (v > best) {
            second = best;
            best = v;
            argmax = (int)b;
        } else if (v > second) {
            second = v;
        }
    }
    if (argmax_out) {
        *argmax_out = argmax;
    }
    if (best < 0.0) {
        return 0;
    }
    if (second < 0.0) {
        return best > 0.0 ? (int32_t)(best + 0.5) : 0;
    }
    return (int32_t)(best - second + 0.5);
}

/* Gain inject: recent context residual × cooc boosts co-occurring candidates.
 * residual attenuates with age like gate_ctrl charge in [0,ONE). */
static void fluency_gain_inject(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len) {
    if (!m || !m->gain_on || !m->gain_cooc || !m->gain_vidx || !m->gain_tok || m->gain_topk == 0u ||
        ctx_len == 0u || m->gain_scale <= 0) {
        return;
    }
    size_t win = (size_t)m->gain_window;
    if (win > ctx_len) {
        win = ctx_len;
    }
    size_t start = ctx_len - win;
    for (size_t i = start; i < ctx_len; ++i) {
        flu_tok_t c = ctx[i];
        if (c >= FLUENCY_VOCAB) {
            continue;
        }
        uint16_t ci = m->gain_vidx[c];
        if (ci == 0xFFFFu) {
            continue;
        }
        /* Age: most recent = full residual ONE, older halves. */
        uint32_t age = (uint32_t)(ctx_len - 1u - i);
        int32_t residual = (int32_t)NERVA_Q8_8_ONE >> (age > 14u ? 14 : (int)age);
        if (residual <= 0) {
            continue;
        }
        const uint32_t *row = m->gain_cooc + (size_t)ci * (size_t)m->gain_topk;
        for (uint32_t j = 0; j < m->gain_topk; ++j) {
            m->gain_inject_ops++;
            uint32_t cnt = row[j];
            if (cnt == 0u) {
                continue;
            }
            flu_tok_t t = m->gain_tok[j];
            if (t >= FLUENCY_VOCAB || !m->seen[t] || m->tok_node[t] == NERVA_INVALID_ID) {
                continue;
            }
            /* log1p-ish via shifts: strength ~ min(cnt, 16) */
            int32_t strength = (int32_t)(cnt > 16u ? 16u : cnt);
            int32_t delta =
                (int32_t)(((int64_t)m->gain_scale * (int64_t)residual * (int64_t)strength) /
                          ((int64_t)NERVA_Q8_8_ONE * 16));
            if (delta > 0) {
                m->e->nodes[m->tok_node[t]].v =
                    nerva_q8_8_saturating_add(m->e->nodes[m->tok_node[t]].v, (nerva_q8_8_t)delta);
            }
        }
    }
}

/* ---- training: local count increment (no backprop) ------------------------- */

/* Forward decl: on-graph CTX_NORM routes through counted materialize. */
int fluency_train_counted(FluencyModel *m, const flu_tok_t *text, size_t len,
                          FluencySatPoint *sat, size_t sat_cap, size_t *sat_n,
                          uint32_t chunk_tokens);

/* Open-addressing multiset: stores full (order, ctx, next) so decode is exact
 * even when packed keys would overflow 64 bits at high order. */
typedef struct FluCountSlot {
    uint8_t used;
    uint8_t order;
    flu_tok_t next;
    flu_tok_t ctx[FLUENCY_MAX_ORDER];
    uint32_t count;
} FluCountSlot;

/* Context-only hash: used to shard pass1 so all (ctx,*) land together (CTX_NORM). */
static uint64_t fluency_count_ctx_hash(uint32_t k, const flu_tok_t *ctx) {
    uint64_t h = 14695981039346656037ull ^ (uint64_t)k;
    for (uint32_t i = 0; i < k; ++i) {
        h ^= (uint64_t)ctx[i] + 0x9e3779b97f4a7c15ull;
        h *= 1099511628211ull;
    }
    return h | 1ull;
}

static uint64_t fluency_count_hash(uint32_t k, const flu_tok_t *ctx, flu_tok_t next) {
    uint64_t h = fluency_count_ctx_hash(k, ctx);
    h ^= (uint64_t)next + 0x9e3779b97f4a7c15ull;
    h *= 1099511628211ull;
    return h | 1ull; /* never 0 */
}

static int fluency_count_same(const FluCountSlot *s, uint32_t k, const flu_tok_t *ctx,
                              flu_tok_t next) {
    if (!s->used || s->order != (uint8_t)k || s->next != next) {
        return 0;
    }
    for (uint32_t i = 0; i < k; ++i) {
        if (s->ctx[i] != ctx[i]) {
            return 0;
        }
    }
    return 1;
}

static int fluency_count_grow(FluCountSlot **slots, size_t *cap, size_t nused,
                              size_t soft_nused) {
    size_t ncap = *cap ? *cap * 2u : 4096u;
    while (ncap < nused * 2u + 16u) {
        ncap *= 2u;
    }
    /* Soft-cap path: 12M slots (~sealed v2). Exact soft=0: 48M (~1–1.5GB table). */
    const size_t kMaxCap =
        (soft_nused == 0u) ? (48u * 1024u * 1024u) : (12u * 1024u * 1024u);
    if (ncap > kMaxCap) {
        ncap = kMaxCap;
        if (nused * 2u + 16u > ncap) {
            return -1; /* cannot grow further */
        }
    }
    FluCountSlot *ns = (FluCountSlot *)calloc(ncap, sizeof(*ns));
    if (!ns) {
        return -1;
    }
    for (size_t i = 0; i < *cap; ++i) {
        if (!(*slots)[i].used) {
            continue;
        }
        uint64_t h =
            fluency_count_hash((*slots)[i].order, (*slots)[i].ctx, (*slots)[i].next);
        size_t idx = (size_t)(h % ncap);
        while (ns[idx].used) {
            idx = (idx + 1u) % ncap;
        }
        ns[idx] = (*slots)[i];
    }
    free(*slots);
    *slots = ns;
    *cap = ncap;
    return 0;
}

/* soft_nused==0 → no soft-cap (context-hash shards bound peak RAM instead). */
static FluCountSlot *fluency_count_ref(FluCountSlot **slots, size_t *cap, size_t *nused,
                                       uint32_t k, const flu_tok_t *ctx, flu_tok_t next,
                                       size_t soft_nused) {
    int allow_insert = (soft_nused == 0u) || (*nused < soft_nused);
    if (allow_insert && *nused * 10u >= *cap * 7u) {
        if (fluency_count_grow(slots, cap, *nused + 1u, soft_nused) != 0) {
            allow_insert = 0;
        }
    }
    if (*cap == 0u) {
        return NULL;
    }
    uint64_t h = fluency_count_hash(k, ctx, next);
    size_t idx = (size_t)(h % *cap);
    size_t probes = 0;
    size_t hapax_idx = (size_t)-1;
    while ((*slots)[idx].used) {
        if (fluency_count_same(&(*slots)[idx], k, ctx, next)) {
            return &(*slots)[idx];
        }
        if (soft_nused > 0u && (*slots)[idx].count <= 1u) {
            hapax_idx = idx;
        }
        idx = (idx + 1u) % *cap;
        if (++probes >= *cap) {
            return NULL;
        }
    }
    if (!allow_insert) {
        if (hapax_idx == (size_t)-1) {
            return NULL;
        }
        idx = hapax_idx;
    } else {
        (*nused)++;
    }
    (*slots)[idx].used = 1;
    (*slots)[idx].order = (uint8_t)k;
    (*slots)[idx].next = next;
    for (uint32_t i = 0; i < k; ++i) {
        (*slots)[idx].ctx[i] = ctx[i];
    }
    for (uint32_t i = k; i < FLUENCY_MAX_ORDER; ++i) {
        (*slots)[idx].ctx[i] = 0;
    }
    (*slots)[idx].count = 0;
    return &(*slots)[idx];
}

/* Offline pass-1 over text[lo, hi). If n_shards>1, only count ngrams whose
 * context-hash falls in `shard` (exact full-corpus counts across shards).
 * soft_nused==0 disables soft-cap. do_unigram=0 when unigram already filled. */
static int fluency_count_pass1_range(const flu_tok_t *text, size_t lo, size_t hi,
                                     uint32_t order, int skip_on, FluCountSlot **slots,
                                     size_t *cap, size_t *nused, uint64_t *unigram,
                                     uint64_t *unigram_total, uint64_t *train_positions,
                                     uint32_t shard, uint32_t n_shards, size_t soft_nused,
                                     int do_unigram) {
    if (n_shards == 0u) {
        n_shards = 1u;
    }
    for (size_t i = lo; i < hi; ++i) {
        flu_tok_t next = text[i];
        if (next >= FLUENCY_VOCAB) {
            continue;
        }
        if (do_unigram && unigram && unigram_total) {
            unigram[next]++;
            (*unigram_total)++;
        }
        for (uint32_t k = 1u; k <= order; ++k) {
            if ((size_t)k > i) {
                break;
            }
            if (n_shards > 1u) {
                uint64_t ch = fluency_count_ctx_hash(k, text + (i - k));
                if ((uint32_t)(ch % (uint64_t)n_shards) != shard) {
                    continue;
                }
            }
            FluCountSlot *s =
                fluency_count_ref(slots, cap, nused, k, text + (i - k), next, soft_nused);
            if (!s) {
                if (soft_nused == 0u) {
                    fprintf(stderr,
                            "fluency: soft=0 count table full/OOM nused=%zu cap=%zu\n", *nused,
                            *cap);
                    return -1;
                }
                continue;
            }
            if (s->count < UINT32_MAX) {
                s->count++;
            }
        }
#if FLU_SKIP_PACK_OK
        if (skip_on && i >= 2u && text[i - 2u] < FLUENCY_VOCAB) {
            flu_tok_t sctx = (flu_tok_t)(text[i - 2u] + FLUENCY_VOCAB);
            int take = 1;
            if (n_shards > 1u) {
                uint64_t ch = fluency_count_ctx_hash(1u, &sctx);
                if ((uint32_t)(ch % (uint64_t)n_shards) != shard) {
                    take = 0;
                }
            }
            if (take) {
                FluCountSlot *s =
                    fluency_count_ref(slots, cap, nused, 1u, &sctx, next, soft_nused);
                if (!s) {
                    if (soft_nused == 0u) {
                        fprintf(stderr,
                                "fluency: soft=0 count table full/OOM nused=%zu cap=%zu\n",
                                *nused, *cap);
                        return -1;
                    }
                } else if (s->count < UINT32_MAX) {
                    s->count++;
                }
            }
        }
#else
        (void)skip_on;
#endif
        if (train_positions) {
            (*train_positions)++;
        }
    }
    return 0;
}

static int fluency_count_merge_into(FluCountSlot **dst, size_t *dcap, size_t *dnused,
                                    const FluCountSlot *src, size_t scap,
                                    size_t soft_nused) {
    for (size_t hi = 0; hi < scap; ++hi) {
        if (!src[hi].used || src[hi].count == 0u) {
            continue;
        }
        FluCountSlot *s = fluency_count_ref(dst, dcap, dnused, src[hi].order, src[hi].ctx,
                                            src[hi].next, soft_nused);
        if (!s) {
            return -1;
        }
        uint64_t sum = (uint64_t)s->count + (uint64_t)src[hi].count;
        s->count = sum > UINT32_MAX ? UINT32_MAX : (uint32_t)sum;
    }
    return 0;
}

typedef struct FluCountJob {
    const flu_tok_t *text;
    size_t lo;
    size_t hi;
    uint32_t order;
    int skip_on;
    FluCountSlot *slots;
    size_t cap;
    size_t nused;
    uint64_t *unigram;
    uint64_t unigram_total;
    uint64_t train_positions;
    int err;
} FluCountJob;

#if defined(_WIN32)
static DWORD WINAPI fluency_count_job_main(void *arg) {
#else
static void *fluency_count_job_main(void *arg) {
#endif
    FluCountJob *j = (FluCountJob *)arg;
    j->err = 0;
    j->slots = NULL;
    j->cap = 0;
    j->nused = 0;
    j->unigram_total = 0;
    j->train_positions = 0;
    if (fluency_count_grow(&j->slots, &j->cap, 4096u, 0u) != 0) {
        j->err = -1;
#if defined(_WIN32)
        return 0;
#else
        return NULL;
#endif
    }
    j->unigram = (uint64_t *)calloc((size_t)FLUENCY_VOCAB, sizeof(uint64_t));
    if (!j->unigram) {
        free(j->slots);
        j->slots = NULL;
        j->err = -1;
#if defined(_WIN32)
        return 0;
#else
        return NULL;
#endif
    }
    /* Position-parallel jobs: single context-shard, no soft-cap. */
    if (fluency_count_pass1_range(j->text, j->lo, j->hi, j->order, j->skip_on, &j->slots, &j->cap,
                                  &j->nused, j->unigram, &j->unigram_total, &j->train_positions,
                                  0u, 1u, 0u, 1) != 0) {
        j->err = -1;
    }
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

/* Materialize one count table into the graph. Does not free slots; does not
 * rebuild adjacency (caller does once). Tok nodes must already exist. */
static FluCountSlot *g_pi_b_sort_slots;

static int fluency_pi_b_idx_cmp(const void *a, const void *b) {
    size_t ia = *(const size_t *)a;
    size_t ib = *(const size_t *)b;
    uint32_t ca = g_pi_b_sort_slots[ia].count;
    uint32_t cb = g_pi_b_sort_slots[ib].count;
    if (ca != cb) {
        return (cb > ca) - (cb < ca); /* descending count */
    }
    uint32_t oa = g_pi_b_sort_slots[ia].order;
    uint32_t ob = g_pi_b_sort_slots[ib].order;
    if (oa != ob) {
        return (oa > ob) - (oa < ob); /* lower order first on ties */
    }
    return (ia > ib) - (ia < ib);
}

/* Context-create sort keys (Π_B): highest max_count first. Typed after FluCtxCache. */
typedef struct {
    uint8_t used;
    uint8_t order;
    flu_tok_t ctx[FLUENCY_MAX_ORDER];
    uint32_t max_count;
    uint32_t node_id;
} FluCtxCache;

static FluCtxCache *g_pi_b_ctxcache;

static int fluency_pi_b_ctx_cmp(const void *a, const void *b) {
    size_t ia = *(const size_t *)a;
    size_t ib = *(const size_t *)b;
    uint32_t ca = g_pi_b_ctxcache[ia].max_count;
    uint32_t cb = g_pi_b_ctxcache[ib].max_count;
    if (ca != cb) {
        return (cb > ca) - (cb < ca);
    }
    uint8_t oa = g_pi_b_ctxcache[ia].order;
    uint8_t ob = g_pi_b_ctxcache[ib].order;
    if (oa != ob) {
        return (oa > ob) - (oa < ob);
    }
    return (ia > ib) - (ia < ib);
}

static int fluency_materialize_counts(FluencyModel *m, FluCountSlot *slots, size_t cap,
                                      size_t nused) {
    nerva_q8_8_t wmax = m->e->cfg.weight_max_q8_8;
    int use_ctx_norm = (m->weight_map == FLU_WEIGHT_MAP_CTX_NORM);
    int use_pi_b = m->pass2_pi_b ? 1 : 0;

    size_t *idx_list = (size_t *)malloc(nused * sizeof(size_t));
    if (!idx_list) {
        return -1;
    }
    size_t nlist = 0;
    for (size_t hi = 0; hi < cap; ++hi) {
        if (slots[hi].used && slots[hi].count > 0u) {
            if (nlist < nused) {
                idx_list[nlist++] = hi;
            }
        }
    }
    if (use_pi_b && nlist > 1u) {
        g_pi_b_sort_slots = slots;
        qsort(idx_list, nlist, sizeof(size_t), fluency_pi_b_idx_cmp);
        g_pi_b_sort_slots = NULL;
        fprintf(stderr, "fluency: pass2_pi_b=1 sorted %zu count slots by N desc\n", nlist);
    }

    if (nlist > 0u) {
        size_t want_slots = ((size_t)m->e->edge_count + nlist) * 2u + 16u;
        if (m->slot_cap < want_slots) {
            if (fluency_slots_grow(m, want_slots) != 0) {
                free(idx_list);
                return -1;
            }
        }
        size_t want_names = (size_t)m->e->node_count + nlist + 1024u;
        while (m->name_map_cap < want_names * 2u + 16u) {
            if (fluency_name_grow(m) != 0) {
                free(idx_list);
                return -1;
            }
        }
    }

    size_t cmax_cap = nlist * 2u + 16u;
    FluCtxCache *ctxcache =
        nlist > 0u ? (FluCtxCache *)calloc(cmax_cap, sizeof(*ctxcache)) : NULL;
    uint32_t *max_of = use_ctx_norm && nlist > 0u ? (uint32_t *)malloc(nlist * sizeof(uint32_t))
                                                    : NULL;
    uint32_t *ctx_of = nlist > 0u ? (uint32_t *)malloc(nlist * sizeof(uint32_t)) : NULL;
    if ((nlist > 0u && (!ctxcache || !ctx_of)) || (use_ctx_norm && nlist > 0u && !max_of)) {
        free(idx_list);
        free(max_of);
        free(ctx_of);
        free(ctxcache);
        return -1;
    }

    if (nlist > 0u) {
        for (size_t li = 0; li < nlist; ++li) {
            const FluCountSlot *s = &slots[idx_list[li]];
            uint32_t k = s->order;
            uint64_t h = fluency_count_ctx_hash(k, s->ctx);
            size_t idx = (size_t)(h % cmax_cap);
            size_t probes = 0;
            for (;;) {
                if (!ctxcache[idx].used) {
                    ctxcache[idx].used = 1;
                    ctxcache[idx].order = (uint8_t)k;
                    for (uint32_t i = 0; i < FLUENCY_MAX_ORDER; ++i) {
                        ctxcache[idx].ctx[i] = (i < k) ? s->ctx[i] : 0;
                    }
                    ctxcache[idx].max_count = s->count;
                    ctxcache[idx].node_id = NERVA_INVALID_ID;
                    break;
                }
                if (ctxcache[idx].order == (uint8_t)k) {
                    int same = 1;
                    for (uint32_t i = 0; i < k; ++i) {
                        if (ctxcache[idx].ctx[i] != s->ctx[i]) {
                            same = 0;
                            break;
                        }
                    }
                    if (same) {
                        if (s->count > ctxcache[idx].max_count) {
                            ctxcache[idx].max_count = s->count;
                        }
                        break;
                    }
                }
                idx = (idx + 1u) % cmax_cap;
                if (++probes >= cmax_cap) {
                    break;
                }
            }
        }

        size_t nctx = 0;
        for (size_t idx = 0; idx < cmax_cap; ++idx) {
            if (ctxcache[idx].used) {
                nctx++;
            }
        }
        size_t *corder = nctx > 0u ? (size_t *)malloc(nctx * sizeof(size_t)) : NULL;
        if (nctx > 0u && !corder) {
            free(idx_list);
            free(max_of);
            free(ctx_of);
            free(ctxcache);
            return -1;
        }
        nctx = 0;
        for (size_t idx = 0; idx < cmax_cap; ++idx) {
            if (ctxcache[idx].used) {
                corder[nctx++] = idx;
            }
        }
        if (use_pi_b && nctx > 1u) {
            /* Context create order: highest max_count first (rent keeps mass). */
            g_pi_b_ctxcache = ctxcache;
            qsort(corder, nctx, sizeof(size_t), fluency_pi_b_ctx_cmp);
            g_pi_b_ctxcache = NULL;
        } else {
            size_t buck[FLUENCY_MAX_ORDER + 1u];
            size_t *tmp = (size_t *)malloc(nctx * sizeof(size_t));
            if (!tmp) {
                free(corder);
                free(idx_list);
                free(max_of);
                free(ctx_of);
                free(ctxcache);
                return -1;
            }
            memset(buck, 0, sizeof(buck));
            for (size_t i = 0; i < nctx; ++i) {
                uint8_t o = ctxcache[corder[i]].order;
                if (o > FLUENCY_MAX_ORDER) {
                    o = (uint8_t)FLUENCY_MAX_ORDER;
                }
                buck[o]++;
            }
            size_t run = 0;
            for (uint32_t o = 0; o <= FLUENCY_MAX_ORDER; ++o) {
                size_t c = buck[o];
                buck[o] = run;
                run += c;
            }
            for (size_t i = 0; i < nctx; ++i) {
                uint8_t o = ctxcache[corder[i]].order;
                if (o > FLUENCY_MAX_ORDER) {
                    o = (uint8_t)FLUENCY_MAX_ORDER;
                }
                tmp[buck[o]++] = corder[i];
            }
            memcpy(corder, tmp, nctx * sizeof(size_t));
            free(tmp);
        }
        uint32_t ctx_skip = 0;
        for (size_t ci = 0; ci < nctx; ++ci) {
            size_t idx = corder[ci];
            uint32_t id = fluency_get_ctx(m, ctxcache[idx].ctx, ctxcache[idx].order);
            if (id == NERVA_INVALID_ID) {
                ctx_skip++;
                ctxcache[idx].node_id = NERVA_INVALID_ID;
                continue;
            }
            ctxcache[idx].node_id = id;
        }
        free(corder);
        if (ctx_skip > 0u) {
            fprintf(stderr,
                    "fluency: pass2 graph-full, skipped %u context nodes "
                    "(nodes=%u/%u edges=%u/%u pi_b=%d)\n",
                    ctx_skip, m->e->node_count, m->e->cfg.max_nodes, m->e->edge_count,
                    m->e->cfg.max_edges, use_pi_b);
        }
        for (size_t li = 0; li < nlist; ++li) {
            const FluCountSlot *s = &slots[idx_list[li]];
            uint32_t k = s->order;
            uint64_t h = fluency_count_ctx_hash(k, s->ctx);
            size_t idx = (size_t)(h % cmax_cap);
            size_t probes = 0;
            uint32_t max_c = s->count;
            uint32_t cid = NERVA_INVALID_ID;
            while (ctxcache[idx].used && probes < cmax_cap) {
                if (ctxcache[idx].order == (uint8_t)k) {
                    int same = 1;
                    for (uint32_t i = 0; i < k; ++i) {
                        if (ctxcache[idx].ctx[i] != s->ctx[i]) {
                            same = 0;
                            break;
                        }
                    }
                    if (same) {
                        max_c = ctxcache[idx].max_count;
                        cid = ctxcache[idx].node_id;
                        break;
                    }
                }
                idx = (idx + 1u) % cmax_cap;
                probes++;
            }
            if (use_ctx_norm) {
                max_of[li] = max_c ? max_c : 1u;
            }
            ctx_of[li] = cid;
        }
    }
    free(ctxcache);

    uint32_t edge_skip = 0;
    for (size_t li = 0; li < nlist; ++li) {
        size_t hi = idx_list[li];
        uint32_t k = slots[hi].order;
        int64_t step = (int64_t)fluency_order_step(k);
        int64_t raw = step * (int64_t)slots[hi].count;
        int64_t w;
        if (use_ctx_norm) {
            uint32_t max_c = max_of[li] ? max_of[li] : 1u;
            uint32_t cnt = slots[hi].count;
            if (max_c >= 64u) {
                if (cnt == 1u) {
                    continue;
                }
                if (max_c >= 256u && cnt < 3u) {
                    continue;
                }
                {
                    int64_t target = 4096;
                    if (target > (int64_t)wmax) {
                        target = (int64_t)wmax;
                    }
                    int64_t top = (int64_t)fluency_order_step(m->order);
                    if (top < 1) {
                        top = 1;
                    }
                    w = ((int64_t)cnt * target * step) / ((int64_t)max_c * top);
                }
            } else {
                w = raw;
            }
        } else {
            w = raw;
        }
        if (w > (int64_t)wmax) {
            w = (int64_t)wmax;
        }
        if (w < 1) {
            continue;
        }

        flu_tok_t next = slots[hi].next;
        uint32_t tok = fluency_get_tok(m, next);
        uint32_t ctx_id = ctx_of[li];
        if (tok == NERVA_INVALID_ID || ctx_id == NERVA_INVALID_ID) {
            continue;
        }
        uint32_t *slot = fluency_slot_ref(m, fluency_slot_key(ctx_id, next));
        if (!slot) {
            free(idx_list);
            free(max_of);
            free(ctx_of);
            return -1;
        }
        if (*slot == NERVA_INVALID_ID) {
            uint32_t edge = fluency_append_edge(m->e, ctx_id, tok, FLU_REL_CTX_TO_TOK);
            if (edge == NERVA_INVALID_ID) {
                edge_skip++;
                continue;
            }
            m->e->edges[edge].weight = 0;
            *slot = edge;
        }
        if ((int32_t)m->e->edges[*slot].weight < (int32_t)w) {
            m->e->edges[*slot].weight = (nerva_q8_8_t)w;
        }
    }
    if (edge_skip > 0u) {
        fprintf(stderr,
                "fluency: pass2 edge-full, skipped %u new edges "
                "(nodes=%u/%u edges=%u/%u pi_b=%d)\n",
                edge_skip, m->e->node_count, m->e->cfg.max_nodes, m->e->edge_count,
                m->e->cfg.max_edges, use_pi_b);
    }
    free(idx_list);
    free(max_of);
    free(ctx_of);
    return 0;
}

static void fluency_kn_add_from_slots(FluencyModel *m, const FluCountSlot *slots, size_t cap) {
    if (!m->kn_cont) {
        return;
    }
    for (size_t hi = 0; hi < cap; ++hi) {
        if (slots[hi].used && slots[hi].order == 1u && slots[hi].ctx[0] < FLUENCY_VOCAB) {
            m->kn_cont[slots[hi].next]++;
            m->kn_cont_total++;
        }
    }
}

int fluency_train_counted(FluencyModel *m, const flu_tok_t *text, size_t len,
                          FluencySatPoint *sat, size_t sat_cap, size_t *sat_n,
                          uint32_t chunk_tokens) {
    return fluency_train_counted_jobs(m, text, len, sat, sat_cap, sat_n, chunk_tokens, 1u);
}

int fluency_train_counted_jobs(FluencyModel *m, const flu_tok_t *text, size_t len,
                               FluencySatPoint *sat, size_t sat_cap, size_t *sat_n,
                               uint32_t chunk_tokens, uint32_t n_jobs) {
    if (!m || !text || len == 0u) {
        return -1;
    }
    if (sat_n) {
        *sat_n = 0;
    }
    if (chunk_tokens == 0u) {
        chunk_tokens = 100000u;
    }
    if (n_jobs == 0u) {
        n_jobs = 1u;
    }
    if (n_jobs > 8u) {
        n_jobs = 8u;
    }
    if (len < 100000u && n_jobs > 1u) {
        n_jobs = 1u;
    }

    /* Soft-cap from model (default 7M sealed v2). soft=0 = exact; n_cshards=1
     * (v3 multi-shard per-shard materialize Killed — do not re-enable here). */
    uint32_t n_cshards = 1u;
    size_t soft_nused = m->count_soft_nused;
    if (n_cshards > 1u) {
        n_jobs = 1u; /* parallel jobs × shards would multiply RAM */
        soft_nused = 0u;
    }

    uint32_t nodes0 = m->e->node_count;
    uint32_t edges0 = m->e->edge_count;
    m->name_hits = 0;
    m->name_creates = 0;
    m->train_pass1_s = 0.0;
    m->train_pass2_s = 0.0;

    /* Unigram + train_positions once (not × shards). */
    clock_t t_uni = clock();
    m->train_positions = 0;
    for (size_t i = 0; i < len; ++i) {
        flu_tok_t next = text[i];
        if (next >= FLUENCY_VOCAB) {
            continue;
        }
        m->unigram[next]++;
        m->unigram_total++;
        m->train_positions++;
    }
    m->train_pass1_s += (double)(clock() - t_uni) / (double)CLOCKS_PER_SEC;

    for (uint32_t t = 0; t < FLUENCY_VOCAB; ++t) {
        if (m->unigram[t] > 0u) {
            if (fluency_get_tok(m, (flu_tok_t)t) == NERVA_INVALID_ID) {
                return -1;
            }
        }
    }

    if (!m->kn_cont) {
        m->kn_cont = (uint32_t *)calloc((size_t)FLUENCY_VOCAB, sizeof(uint32_t));
    } else {
        memset(m->kn_cont, 0, sizeof(uint32_t) * (size_t)FLUENCY_VOCAB);
    }
    m->kn_cont_total = 0;

    fprintf(stderr, "fluency: count_shards=%u soft_nused=%zu%s\n", n_cshards, soft_nused,
            soft_nused == 0u ? " (exact)" : "");

    size_t sat_si = 0;
    uint32_t prev_edges_snap = 0;

    for (uint32_t shard = 0; shard < n_cshards; ++shard) {
        FluCountSlot *slots = NULL;
        size_t cap = 0, nused = 0;
        if (fluency_count_grow(&slots, &cap, 4096u, soft_nused) != 0) {
            return -1;
        }

        clock_t t_pass1 = clock();
        if (n_jobs <= 1u) {
            if (fluency_count_pass1_range(text, 0, len, m->order, m->skip_on ? 1 : 0, &slots, &cap,
                                          &nused, NULL, NULL, NULL, shard, n_cshards, soft_nused,
                                          0) != 0) {
                free(slots);
                return -1;
            }
            if (sat && sat_n && sat_cap > 0 && chunk_tokens > 0u && n_cshards == 1u) {
                /* Mid-stream sat only for single-shard serial (legacy). */
                for (size_t i = chunk_tokens; i < len && sat_si + 1u < sat_cap;
                     i += (size_t)chunk_tokens) {
                    uint32_t e_now = (uint32_t)(nused > UINT32_MAX ? UINT32_MAX : nused);
                    sat[sat_si].tokens_seen = (uint64_t)i;
                    sat[sat_si].nodes = 0;
                    sat[sat_si].edges = e_now;
                    sat[sat_si].new_nodes = 0;
                    sat[sat_si].new_edges = e_now - prev_edges_snap;
                    prev_edges_snap = e_now;
                    sat_si++;
                }
            }
        } else {
            /* Position-parallel (small/medium streams, single context-shard). */
            FluCountJob *jobs = (FluCountJob *)calloc((size_t)n_jobs, sizeof(FluCountJob));
            if (!jobs) {
                free(slots);
                return -1;
            }
            size_t chunk = (len + (size_t)n_jobs - 1u) / (size_t)n_jobs;
#if defined(_WIN32)
            HANDLE *ths = (HANDLE *)calloc((size_t)n_jobs, sizeof(HANDLE));
            if (!ths) {
                free(jobs);
                free(slots);
                return -1;
            }
#else
            pthread_t *ths = (pthread_t *)calloc((size_t)n_jobs, sizeof(pthread_t));
            if (!ths) {
                free(jobs);
                free(slots);
                return -1;
            }
#endif
            for (uint32_t j = 0; j < n_jobs; ++j) {
                size_t lo = (size_t)j * chunk;
                size_t hi = lo + chunk;
                if (lo >= len) {
                    lo = len;
                    hi = len;
                }
                if (hi > len) {
                    hi = len;
                }
                jobs[j].text = text;
                jobs[j].lo = lo;
                jobs[j].hi = hi;
                jobs[j].order = m->order;
                jobs[j].skip_on = m->skip_on ? 1 : 0;
#if defined(_WIN32)
                ths[j] = CreateThread(NULL, 0, fluency_count_job_main, &jobs[j], 0, NULL);
                if (!ths[j]) {
                    jobs[j].err = -1;
                    fluency_count_job_main(&jobs[j]);
                }
#else
                if (pthread_create(&ths[j], NULL, fluency_count_job_main, &jobs[j]) != 0) {
                    jobs[j].err = -1;
                    fluency_count_job_main(&jobs[j]);
                }
#endif
            }
            int fail = 0;
            for (uint32_t j = 0; j < n_jobs; ++j) {
#if defined(_WIN32)
                if (ths[j]) {
                    WaitForSingleObject(ths[j], INFINITE);
                    CloseHandle(ths[j]);
                }
#else
                pthread_join(ths[j], NULL);
#endif
                if (jobs[j].err != 0) {
                    fail = 1;
                }
            }
            free(ths);
            if (!fail) {
                for (uint32_t j = 0; j < n_jobs; ++j) {
                    if (jobs[j].hi <= jobs[j].lo) {
                        free(jobs[j].unigram);
                        free(jobs[j].slots);
                        continue;
                    }
                    if (fluency_count_merge_into(&slots, &cap, &nused, jobs[j].slots, jobs[j].cap,
                                                 soft_nused) != 0) {
                        fail = 1;
                    }
                    /* Unigram already filled globally; ignore job unigrams. */
                    free(jobs[j].unigram);
                    free(jobs[j].slots);
                }
            } else {
                for (uint32_t j = 0; j < n_jobs; ++j) {
                    free(jobs[j].unigram);
                    free(jobs[j].slots);
                }
            }
            free(jobs);
            if (fail) {
                free(slots);
                return -1;
            }
        }
        m->train_pass1_s += (double)(clock() - t_pass1) / (double)CLOCKS_PER_SEC;

        fluency_kn_add_from_slots(m, slots, cap);

        clock_t t_pass2 = clock();
        if (fluency_materialize_counts(m, slots, cap, nused) != 0) {
            free(slots);
            return -1;
        }
        free(slots);
        m->train_pass2_s += (double)(clock() - t_pass2) / (double)CLOCKS_PER_SEC;
    }

    nerva_graph_rebuild_adjacency(m->e);
    fluency_organs_observe_stream(m, text, len);

    if (sat && sat_n && sat_cap > 0) {
        if (sat_si >= sat_cap) {
            sat_si = sat_cap - 1u;
        }
        sat[sat_si].tokens_seen = (uint64_t)len;
        sat[sat_si].nodes = m->e->node_count;
        sat[sat_si].edges = m->e->edge_count;
        sat[sat_si].new_nodes = m->e->node_count - nodes0;
        sat[sat_si].new_edges = m->e->edge_count - edges0;
        sat_si++;
        *sat_n = sat_si;
    }
    return 0;
}

static void fluency_fast_deposit_stream(FluencyModel *m, const flu_tok_t *text, size_t len) {
    if (!m || !text || len == 0u) {
        return;
    }
    for (size_t i = 0; i < len; ++i) {
        if (i > 0u) {
            fluency_fast_deposit(m, text[i - 1u], text[i]);
        }
        fluency_stream_advance(m);
    }
}

void fluency_train(FluencyModel *m, const flu_tok_t *text, size_t len) {
    if (!m || !text) {
        return;
    }
    /* CTX_NORM needs wide counts before normalize — share counted materialize
     * so on-graph and counted paths stay identical at λ=0. */
    if (m->weight_map == FLU_WEIGHT_MAP_CTX_NORM) {
        if (fluency_train_counted(m, text, len, NULL, 0, NULL, 0) != 0) {
            return;
        }
        fluency_fast_deposit_stream(m, text, len);
        return;
    }
    for (size_t i = 0; i < len; ++i) {
        flu_tok_t next = text[i];
        m->unigram[next]++;
        m->unigram_total++;
        uint32_t tok = fluency_get_tok(m, next);
        if (tok == NERVA_INVALID_ID) {
            fluency_stream_advance(m);
            continue;
        }
        for (uint32_t k = 1u; k <= m->order; ++k) {
            if ((size_t)k > i) {
                break;
            }
            const flu_tok_t *ctx_bytes = text + (i - k);
            uint32_t ctx = fluency_get_ctx(m, ctx_bytes, k);
            if (ctx == NERVA_INVALID_ID) {
                continue;
            }
            uint32_t *slot = fluency_slot_ref(m, fluency_slot_key(ctx, next));
            if (!slot) {
                continue;
            }
            if (*slot == NERVA_INVALID_ID) {
                uint32_t edge = fluency_append_edge(m->e, ctx, tok, FLU_REL_CTX_TO_TOK);
                if (edge == NERVA_INVALID_ID) {
                    continue;
                }
                m->e->edges[edge].weight = 0;
                *slot = edge;
            }
            nerva_q8_8_t old_w = 0, new_w = 0;
            nerva_graph_modify_weight(m->e, *slot, (nerva_q8_8_t)fluency_order_step(k), &old_w,
                                      &new_w);
        }
        /* One-shot Hebbian: deposit A->B on every observed transition. */
        if (i > 0u) {
            fluency_fast_deposit(m, text[i - 1u], next);
        }
        fluency_stream_advance(m);
        m->train_positions++;
    }
    nerva_graph_rebuild_adjacency(m->e);
}

/* Incremental on-graph train with saturation sampling every chunk_tokens. */
void fluency_train_sat(FluencyModel *m, const flu_tok_t *text, size_t len,
                       FluencySatPoint *sat, size_t sat_cap, size_t *sat_n,
                       uint32_t chunk_tokens) {
    if (sat_n) {
        *sat_n = 0;
    }
    if (!m || !text) {
        return;
    }
    if (chunk_tokens == 0u) {
        chunk_tokens = 100000u;
    }
    /* Full-stream counts required for CTX_NORM; chunked on-graph would lose ratios. */
    if (m->weight_map == FLU_WEIGHT_MAP_CTX_NORM) {
        if (fluency_train_counted(m, text, len, sat, sat_cap, sat_n, chunk_tokens) != 0) {
            return;
        }
        fluency_fast_deposit_stream(m, text, len);
        return;
    }
    uint32_t prev_n = m->e->node_count;
    uint32_t prev_e = m->e->edge_count;
    size_t si = 0;
    size_t start = 0;
    while (start < len) {
        size_t n = (size_t)chunk_tokens;
        if (start + n > len) {
            n = len - start;
        }
        fluency_train(m, text + start, n);
        if (sat && si < sat_cap) {
            sat[si].tokens_seen = (uint64_t)(start + n);
            sat[si].nodes = m->e->node_count;
            sat[si].edges = m->e->edge_count;
            sat[si].new_nodes = m->e->node_count - prev_n;
            sat[si].new_edges = m->e->edge_count - prev_e;
            prev_n = m->e->node_count;
            prev_e = m->e->edge_count;
            si++;
        }
        start += n;
    }
    if (sat_n) {
        *sat_n = si;
    }
}

/* ---- prediction: a firing event over the count graph ----------------------- */

static void fluency_reset_charges(NervaEngine *e) {
    for (uint32_t i = 0; i < e->node_count; ++i) {
        e->nodes[i].v = e->nodes[i].v_rest;
    }
    e->event_count = 0;
    e->active_count = 0;
    nerva_trace_clear(e);
}

/* CTW-class recursive mix over materialized ctx→tok edges (Willems-style β).
 * Default path (ctw_mix=0) stays charge-superposition — bit-exact sealed. */
static int fluency_ctw_ensure_inv(FluencyModel *m) {
    uint32_t need = m->e->cfg.max_nodes;
    if (m->ctw_tok_of_node && m->ctw_tok_cap >= need) {
        return 0;
    }
    free(m->ctw_tok_of_node);
    m->ctw_tok_of_node = (uint32_t *)malloc((size_t)need * sizeof(uint32_t));
    if (!m->ctw_tok_of_node) {
        m->ctw_tok_cap = 0;
        return -1;
    }
    m->ctw_tok_cap = need;
    for (uint32_t i = 0; i < need; ++i) {
        m->ctw_tok_of_node[i] = NERVA_INVALID_ID;
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        uint32_t id = m->tok_node[b];
        if (id != NERVA_INVALID_ID && id < need) {
            m->ctw_tok_of_node[id] = b;
        }
    }
    return 0;
}

static int fluency_predict_ctw_mix(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len,
                                  double *dist) {
    const double alpha = 0.5;
    const double gamma = 256.0; /* β = W/(W+γ); Schematic CTW-class, not binary KT */
    if (fluency_ctw_ensure_inv(m) != 0) {
        return -1;
    }
    if (!m->e->adjacency_valid) {
        nerva_graph_rebuild_adjacency(m->e);
    }

    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        dist[b] = 0.0;
    }

    int use_kn = (m->kn_on && m->kn_cont && m->kn_cont_total > 0u);
    double kn_scale = use_kn ? (1.0 / (double)m->kn_cont_total) : 0.0;
    double mass0 = 0.0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (!m->seen[b]) {
            continue;
        }
        double p0 = use_kn ? (kn_scale * (double)m->kn_cont[b]) : (1.0 / (double)m->vocab_count);
        dist[b] = p0;
        mass0 += p0;
    }
    if (mass0 > 0.0) {
        for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
            if (m->seen[b]) {
                dist[b] /= mass0;
            }
        }
    }

    double *pk = (double *)calloc((size_t)FLUENCY_VOCAB, sizeof(double));
    if (!pk) {
        return -1;
    }

    for (uint32_t k = 1u; k <= m->order; ++k) {
        if ((size_t)k > ctx_len) {
            break;
        }
        const flu_tok_t *bytes = ctx + (ctx_len - k);
        char name[8u + 5u * FLUENCY_MAX_ORDER];
        fluency_ctx_name(name, sizeof(name), bytes, k);
        uint32_t cid = fluency_name_lookup(m, name);
        if (cid == NERVA_INVALID_ID || cid >= m->e->node_count) {
            continue;
        }
        const NervaNode *n = &m->e->nodes[cid];
        double W = 0.0;
        memset(pk, 0, (size_t)FLUENCY_VOCAB * sizeof(double));
        for (uint32_t ei = 0; ei < n->out_count; ++ei) {
            uint32_t slot = n->first_out + ei;
            if (slot >= m->e->edge_count) {
                break;
            }
            uint32_t edge_id = m->e->sorted_edges[slot];
            const NervaEdge *ed = &m->e->edges[edge_id];
            if (ed->flags & NERVA_EDGE_DELETED) {
                continue;
            }
            if (ed->relation != FLU_REL_CTX_TO_TOK) {
                continue;
            }
            if (ed->target >= m->ctw_tok_cap) {
                continue;
            }
            uint32_t tok = m->ctw_tok_of_node[ed->target];
            if (tok >= FLUENCY_VOCAB || !m->seen[tok]) {
                continue;
            }
            double w = (double)((ed->weight > 0) ? ed->weight : 0);
            if (w <= 0.0) {
                continue;
            }
            pk[tok] += w;
            W += w;
        }
        if (W <= 0.0) {
            continue;
        }
        double beta = W / (W + gamma);
        for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
            if (!m->seen[b]) {
                continue;
            }
            double p_emp = pk[b] / W;
            dist[b] = beta * p_emp + (1.0 - beta) * dist[b];
        }
    }
    free(pk);

    /* Light add-α floor so zeros stay defined under Gibbs. */
    double denom = 0.0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (!m->seen[b]) {
            continue;
        }
        dist[b] = dist[b] + alpha / (double)m->vocab_count;
        denom += dist[b];
    }
    int argmax = -1;
    double best = -1.0;
    if (denom > 0.0) {
        for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
            if (!m->seen[b]) {
                continue;
            }
            dist[b] /= denom;
            if (dist[b] > best) {
                best = dist[b];
                argmax = (int)b;
            }
        }
    }
    return argmax;
}

int fluency_predict(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len, double *dist) {
    if (!m || !dist) {
        return -1;
    }
    if (m->ctw_mix) {
        return fluency_predict_ctw_mix(m, ctx, ctx_len, dist);
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        dist[b] = 0.0;
    }

    fluency_reset_charges(m->e);

    /* Fire every available backoff-order context node at once; their count
     * edges superpose on the continuation integrators (linear interpolation). */
    int any = 0;
    for (uint32_t k = 1u; k <= m->order; ++k) {
        if ((size_t)k > ctx_len) {
            break;
        }
        const flu_tok_t *bytes = ctx + (ctx_len - k);
        char name[8u + 5u * FLUENCY_MAX_ORDER];
        fluency_ctx_name(name, sizeof(name), bytes, k);
        uint32_t id = fluency_name_lookup(m, name);
        if (id != NERVA_INVALID_ID) {
            nerva_activate_node(m->e, id, NERVA_Q8_8_ONE);
            any = 1;
        }
    }
    if (fluency_skip_fire(m, ctx, ctx_len)) {
        any = 1;
    }
    if (any) {
        nerva_tick_n(m->e, 3u); /* fire context, deliver one hop, read */
    }

    /* Fast-edge inject: third current into integrators. λ=0 skips entirely
     * (bit-exact baseline). Instance path: if a live (cue,src) binding hits,
     * it replaces bare-A type inject for this prediction (cue selects among
     * coexisting roles). Miss / off → type path only (bit-exact when off). */
    if (!fluency_instance_inject(m, ctx, ctx_len)) {
        fluency_fast_inject(m, ctx, ctx_len);
    }

    /* Context organs (default off → no work, bit-exact). Additive injects so
     * /why can sum components. */
    fluency_gain_inject(m, ctx, ctx_len);
    fluency_mood_inject(m);

    /* Read integrated votes; emergency-dial hubs only if margin is low.
     * Heap score[] so LLM V=65k does not blow the stack. */
    const double alpha = 0.5;
    double total = 0.0;
    double *score = (double *)calloc((size_t)FLUENCY_VOCAB, sizeof(double));
    if (!score) {
        return -1;
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (m->seen[b] && m->tok_node[b] != NERVA_INVALID_ID) {
            int32_t v = (int32_t)m->e->nodes[m->tok_node[b]].v;
            if (v > 0) {
                score[b] = (double)v;
            }
        }
        total += score[b];
    }

    m->last_dialed = 0;
    m->hub_last_ops = 0;
    m->last_margin = fluency_charge_margin(m, score, NULL);
    if (m->hub_on) {
        m->hub_predict_count++;
    }
    /* Surprise-gated emergency dial: pathway closed unless margin collapses. */
    if (m->hub_on && m->hub_h > 0u && m->hub_scale > 0 && m->hub_conf_margin > 0 &&
        m->last_margin < m->hub_conf_margin) {
        uint64_t ops = 0;
        fluency_hub_inject(m, ctx, ctx_len, &ops);
        m->last_dialed = 1;
        m->hub_last_ops = ops;
        m->hub_dial_count++;
        m->hub_dial_ops += ops;
        total = 0.0;
        for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
            score[b] = 0.0;
            if (m->seen[b] && m->tok_node[b] != NERVA_INVALID_ID) {
                int32_t v = (int32_t)m->e->nodes[m->tok_node[b]].v;
                if (v > 0) {
                    score[b] = (double)v;
                }
            }
            total += score[b];
        }
        m->last_margin = fluency_charge_margin(m, score, NULL);
    }

    double denom = total + alpha * (double)m->vocab_count;
    /* KN continuation prior: base(b) = α·V·q(b), q(b) = distinct-predecessor
     * share. With q uniform (or kn off) this is exactly the legacy α. */
    int use_kn = (m->kn_on && m->kn_cont && m->kn_cont_total > 0u);
    double kn_scale = use_kn ? (alpha * (double)m->vocab_count / (double)m->kn_cont_total) : 0.0;
    int argmax = -1;
    double best = -1.0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (!m->seen[b]) {
            continue;
        }
        double base = use_kn ? (kn_scale * (double)m->kn_cont[b]) : alpha;
        double p = (score[b] + base) / (denom > 0.0 ? denom : 1.0);
        dist[b] = p;
        if (p > best) {
            best = p;
            argmax = (int)b;
        }
    }
    free(score);
    /* TSCAR product organ mix (default off → no change). */
    if (m->tscar_organ && m->tscar_mix > 0.0 && ctx_len >= 1u) {
        fluency_tscar_mix_dist(m, ctx[ctx_len - 1u], dist);
        best = -1.0;
        argmax = -1;
        for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
            if (!m->seen[b]) {
                continue;
            }
            if (dist[b] > best) {
                best = dist[b];
                argmax = (int)b;
            }
        }
    }
    return argmax;
}

void fluency_evaluate(FluencyModel *m, const flu_tok_t *text, size_t len, FluencyMetrics *out) {
    if (!m || !text || !out) {
        return;
    }
    memset(out, 0, sizeof(*out));

    uint64_t mut_before = m->e->debug.mutations_applied;
    uint32_t nodes_before = m->e->node_count;
    uint32_t edges_before = m->e->edge_count;

    double *dist = (double *)calloc((size_t)FLUENCY_VOCAB, sizeof(double));
    if (!dist) {
        return;
    }
    double uni_denom = (double)m->unigram_total + (double)m->vocab_count;

    for (size_t i = 0; i < len; ++i) {
        flu_tok_t actual = text[i];
        size_t ctx_len = i; /* preceding bytes available as context */
        int pred = fluency_predict(m, text, ctx_len, dist);

        double p = dist[actual];
        if (p <= 0.0) {
            p = 0.5 / (uni_denom > 0.0 ? uni_denom : 1.0); /* floor for unseen */
        }
        out->nll_sum += -log2(p);

        double p_uni = ((double)m->unigram[actual] + 1.0) / (uni_denom > 0.0 ? uni_denom : 1.0);
        out->eval_positions++;
        if (pred == (int)actual) {
            out->correct_argmax++;
        }
        /* accumulate unigram nll in a separate field via reuse */
        out->model_perplexity += 0.0; /* placeholder; computed below */
        (void)p_uni;
        out->unigram_perplexity += -log2(p_uni);

        /* Eval-time deposit: induction memory for later A->? probes in-stream.
         * Lives in the model pool; permanent e->edge_count is unchanged. */
        if (i > 0u) {
            fluency_fast_deposit(m, text[i - 1u], actual);
        }
        fluency_stream_advance(m);
    }

    double n = (double)(out->eval_positions ? out->eval_positions : 1u);
    out->model_perplexity = pow(2.0, out->nll_sum / n);
    out->unigram_perplexity = pow(2.0, out->unigram_perplexity / n);

    out->eval_mutations = m->e->debug.mutations_applied - mut_before;
    out->eval_node_growth = m->e->node_count - nodes_before;
    out->eval_edge_growth = m->e->edge_count - edges_before;
    free(dist);
}

/* ---- generation ------------------------------------------------------------ */

void fluency_generate(FluencyModel *m, const flu_tok_t *seed, size_t seed_len,
                      flu_tok_t *out, size_t n, uint32_t rng_seed, int sample) {
    if (!m || !out) {
        return;
    }
    /* Context = the last min(seed_len, order) bytes of the seed. */
    flu_tok_t ctx[FLUENCY_MAX_ORDER];
    size_t ctx_len = 0;
    size_t start = seed_len > m->order ? seed_len - m->order : 0;
    for (size_t i = start; i < seed_len; ++i) {
        ctx[ctx_len++] = seed[i];
    }

    uint32_t rng = rng_seed ? rng_seed : 0x9e3779b9u;
    double dist[FLUENCY_VOCAB];

    for (size_t step = 0; step < n; ++step) {
        int pick = fluency_predict(m, ctx, ctx_len, dist);
        if (sample) {
            rng = rng * 1664525u + 1013904223u;
            double r = ((double)(rng >> 8) / 16777216.0);
            double acc = 0.0;
            int chosen = pick;
            for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
                acc += dist[b];
                if (r <= acc) {
                    chosen = (int)b;
                    break;
                }
            }
            pick = chosen;
        }
        if (pick < 0) {
            break;
        }
        out[step] = (flu_tok_t)pick;
        if (ctx_len < m->order) {
            ctx[ctx_len++] = (flu_tok_t)pick;
        } else {
            memmove(ctx, ctx + 1, m->order - 1u);
            ctx[m->order - 1u] = (flu_tok_t)pick;
        }
    }
    out[n] = '\0';
}

/* ---- PCW credit on fluency next-token sequences (shared generate weights) - */

#define FLU_PCW_DELTA ((int16_t)48)
#define FLU_PCW_PASSES 4
#define FLU_PCW_MAX_CAND 48
#define FLU_PCW_ESC_MAX_ROUNDS 24
#define FLU_PCW_ESC_DELTA_CAP 8192

static double fluency_pcw_soft_loss(const double *dist, flu_tok_t gold) {
    double p;
    if (!dist || gold >= FLUENCY_VOCAB)
        return 40.0;
    p = dist[gold];
    if (p < 1e-12)
        p = 1e-12;
    return -log(p) / log(2.0);
}

/* Ensure ctx→gold edge exists; return edge id or NERVA_INVALID_ID. */
static uint32_t fluency_pcw_ensure_edge(FluencyModel *m, uint32_t ctx, flu_tok_t gold) {
    uint32_t tok_nid = fluency_get_tok(m, gold);
    uint32_t *slot;
    if (tok_nid == NERVA_INVALID_ID || ctx == NERVA_INVALID_ID)
        return NERVA_INVALID_ID;
    slot = fluency_slot_ref(m, fluency_slot_key(ctx, gold));
    if (!slot)
        return NERVA_INVALID_ID;
    if (*slot == NERVA_INVALID_ID) {
        uint32_t edge = fluency_append_edge(m->e, ctx, tok_nid, FLU_REL_CTX_TO_TOK);
        if (edge == NERVA_INVALID_ID)
            return NERVA_INVALID_ID;
        m->e->edges[edge].weight = 16;
        *slot = edge;
        nerva_graph_rebuild_adjacency(m->e);
    }
    return *slot;
}

static nerva_q8_8_t fluency_pcw_clamp_w(int32_t w) {
    if (w > 32767)
        return (nerva_q8_8_t)32767;
    if (w < -32768)
        return (nerva_q8_8_t)-32768;
    return (nerva_q8_8_t)w;
}

/* One next-token position: PCW ±δ on outgoing ctx edges under fluency_predict Q. */
static int fluency_pcw_teach_pos(FluencyModel *m, const flu_tok_t *text, size_t i) {
    uint32_t k, ctx, gold_e;
    flu_tok_t gold;
    double *dist;
    double l0, loss, best_pos_imp, best_neg_imp;
    uint32_t best_pos_eid = NERVA_INVALID_ID, best_neg_eid = NERVA_INVALID_ID;
    int16_t half = (int16_t)(FLU_PCW_DELTA / 2);
    int applied = 0;
    uint32_t c, n_cand = 0;
    uint32_t cands[FLU_PCW_MAX_CAND];

    if (!m || !text || i == 0)
        return 0;
    gold = text[i];
    if (fluency_ensure_tok(m, gold) != 0)
        return -1;
    for (size_t j = 0; j < i; ++j) {
        if (fluency_ensure_tok(m, text[j]) != 0)
            return -1;
    }

    k = m->order;
    if ((size_t)k > i)
        k = (uint32_t)i;
    if (k == 0)
        return 0;
    ctx = fluency_get_ctx(m, text + (i - k), k);
    if (ctx == NERVA_INVALID_ID)
        return -1;
    gold_e = fluency_pcw_ensure_edge(m, ctx, gold);
    if (gold_e == NERVA_INVALID_ID)
        return -1;

    dist = (double *)calloc((size_t)FLUENCY_VOCAB, sizeof(double));
    if (!dist)
        return -1;

    (void)fluency_predict(m, text + (i - k), (size_t)k, dist);
    l0 = fluency_pcw_soft_loss(dist, gold);

    /* Candidates: all outgoing from ctx (cap), always include gold edge. */
    cands[n_cand++] = gold_e;
    nerva_graph_rebuild_adjacency(m->e);
    {
        NervaNode *cn = &m->e->nodes[ctx];
        uint32_t oi;
        for (oi = 0; oi < cn->out_count && n_cand < FLU_PCW_MAX_CAND; ++oi) {
            uint32_t eid = m->e->sorted_edges[cn->first_out + oi];
            if (eid >= m->e->edge_count || eid == gold_e)
                continue;
            cands[n_cand++] = eid;
        }
    }

    best_pos_imp = 0.0;
    best_neg_imp = 0.0;
    for (c = 0; c < n_cand; ++c) {
        uint32_t eid = cands[c];
        nerva_q8_8_t wsave = m->e->edges[eid].weight;
        /* +δ */
        m->e->edges[eid].weight = fluency_pcw_clamp_w((int32_t)wsave + (int32_t)FLU_PCW_DELTA);
        (void)fluency_predict(m, text + (i - k), (size_t)k, dist);
        loss = fluency_pcw_soft_loss(dist, gold);
        m->e->edges[eid].weight = wsave;
        if (loss < l0 - 1e-15 && (l0 - loss) > best_pos_imp) {
            best_pos_imp = l0 - loss;
            best_pos_eid = eid;
        }
        /* −δ */
        m->e->edges[eid].weight = fluency_pcw_clamp_w((int32_t)wsave - (int32_t)FLU_PCW_DELTA);
        (void)fluency_predict(m, text + (i - k), (size_t)k, dist);
        loss = fluency_pcw_soft_loss(dist, gold);
        m->e->edges[eid].weight = wsave;
        if (loss < l0 - 1e-15 && (l0 - loss) > best_neg_imp) {
            best_neg_imp = l0 - loss;
            best_neg_eid = eid;
        }
    }

    if (best_pos_eid != NERVA_INVALID_ID && half != 0) {
        if (nerva_work_apply_weight_delta(m->e, best_pos_eid, (nerva_q8_8_t)half,
                                         NERVA_REASON_HEBBIAN_COFIRE))
            applied = 1;
    }
    if (best_neg_eid != NERVA_INVALID_ID && half != 0) {
        if (nerva_work_apply_weight_delta(m->e, best_neg_eid, (nerva_q8_8_t)(-half),
                                         NERVA_REASON_HEBBIAN_COFIRE))
            applied = 1;
    }
    free(dist);
    return applied;
}

/*
 * Escalating PCW after the fixed ±24×4 passes.
 *
 * A one-shot taught fact must override a saturated pretrain habit; ±24×4 cannot
 * close a ~32k q8.8 gap. Doubling probes commit the smallest measured-useful
 * deltas (multiscale quanta). Demoting the measured argmax competitor is the
 * surgical edit — only this position's own context edges are touched.
 */
static int fluency_pcw_teach_pos_escalate(FluencyModel *m, const flu_tok_t *text, size_t i) {
    uint32_t k, ctx, gold_e;
    flu_tok_t gold;
    double *dist;
    double l0;
    int applied = 0;
    int round;
    int32_t delta_i;

    if (!m || !text || i == 0)
        return 0;
    gold = text[i];
    if (fluency_ensure_tok(m, gold) != 0)
        return -1;
    for (size_t j = 0; j < i; ++j) {
        if (fluency_ensure_tok(m, text[j]) != 0)
            return -1;
    }

    k = m->order;
    if ((size_t)k > i)
        k = (uint32_t)i;
    if (k == 0)
        return 0;
    ctx = fluency_get_ctx(m, text + (i - k), k);
    if (ctx == NERVA_INVALID_ID)
        return -1;
    gold_e = fluency_pcw_ensure_edge(m, ctx, gold);
    if (gold_e == NERVA_INVALID_ID)
        return -1;

    dist = (double *)calloc((size_t)FLUENCY_VOCAB, sizeof(double));
    if (!dist)
        return -1;

    delta_i = (int32_t)FLU_PCW_DELTA;
    for (round = 0; round < FLU_PCW_ESC_MAX_ROUNDS; ++round) {
        int pick;
        int16_t d;
        int do_pos = 0, do_neg = 0;
        uint32_t *comp_slot;
        uint32_t comp_e = NERVA_INVALID_ID;
        nerva_q8_8_t wsave;
        double loss;

        pick = fluency_predict(m, text + (i - k), (size_t)k, dist);
        if (pick == (int)gold)
            break;

        if (delta_i > FLU_PCW_ESC_DELTA_CAP)
            delta_i = FLU_PCW_ESC_DELTA_CAP;
        d = (int16_t)delta_i;
        l0 = fluency_pcw_soft_loss(dist, gold);

        /* (b) look up argmax competitor — never mint a missing edge. */
        if (pick >= 0 && (flu_tok_t)pick != gold && (flu_tok_t)pick < FLUENCY_VOCAB) {
            comp_slot = fluency_slot_ref(m, fluency_slot_key(ctx, (flu_tok_t)pick));
            if (comp_slot && *comp_slot != NERVA_INVALID_ID)
                comp_e = *comp_slot;
        }

        /* Probe both sides against the same baseline, then commit via work path. */
        wsave = m->e->edges[gold_e].weight;
        m->e->edges[gold_e].weight = fluency_pcw_clamp_w((int32_t)wsave + (int32_t)d);
        (void)fluency_predict(m, text + (i - k), (size_t)k, dist);
        loss = fluency_pcw_soft_loss(dist, gold);
        m->e->edges[gold_e].weight = wsave;
        if (loss < l0 - 1e-15)
            do_pos = 1;

        if (comp_e != NERVA_INVALID_ID) {
            wsave = m->e->edges[comp_e].weight;
            m->e->edges[comp_e].weight = fluency_pcw_clamp_w((int32_t)wsave - (int32_t)d);
            (void)fluency_predict(m, text + (i - k), (size_t)k, dist);
            loss = fluency_pcw_soft_loss(dist, gold);
            m->e->edges[comp_e].weight = wsave;
            if (loss < l0 - 1e-15)
                do_neg = 1;
        }

        if (do_pos) {
            if (nerva_work_apply_weight_delta(m->e, gold_e, (nerva_q8_8_t)d,
                                             NERVA_REASON_HEBBIAN_COFIRE))
                applied = 1;
        }
        if (do_neg) {
            if (nerva_work_apply_weight_delta(m->e, comp_e, (nerva_q8_8_t)(-d),
                                             NERVA_REASON_HEBBIAN_COFIRE))
                applied = 1;
        }

        if (delta_i < FLU_PCW_ESC_DELTA_CAP) {
            delta_i *= 2;
            if (delta_i > FLU_PCW_ESC_DELTA_CAP)
                delta_i = FLU_PCW_ESC_DELTA_CAP;
        }
    }

    free(dist);
    return applied ? 1 : 0;
}

int fluency_pcw_teach_sequence(FluencyModel *m, const flu_tok_t *text, size_t n) {
    int applied = 0;
    int pass;
    size_t i;
    if (!m || !text || n < 2)
        return -1;
    nerva_work_set_rgrad_updates(0);
    for (pass = 0; pass < FLU_PCW_PASSES; ++pass) {
        for (i = 1; i < n; ++i) {
            int rc = fluency_pcw_teach_pos(m, text, i);
            if (rc < 0)
                return -1;
            if (rc > 0)
                applied = 1;
        }
    }
    /* Escalation: grow deltas until gold is argmax on each position's context. */
    for (i = 1; i < n; ++i) {
        int rc = fluency_pcw_teach_pos_escalate(m, text, i);
        if (rc < 0)
            return -1;
        if (rc > 0)
            applied = 1;
    }
    nerva_graph_rebuild_adjacency(m->e);
    /* Match fluency_train: episodic fast deposits on every taught transition. */
    fluency_fast_deposit_stream(m, text, n);
    return applied ? 1 : 0;
}

/* ---- Context organs: mood + gain + /why (step 5) --------------------------- */

void fluency_mood_set(FluencyModel *m, int on) {
    if (m) {
        m->mood_on = on ? 1u : 0u;
    }
}

void fluency_gain_set(FluencyModel *m, int on) {
    if (m) {
        m->gain_on = on ? 1u : 0u;
    }
}

void fluency_mood_reset(FluencyModel *m) {
    if (!m) {
        return;
    }
    for (uint8_t k = 0; k < FLU_MOOD_K_MAX; ++k) {
        m->mood_reg[k] = 0.0f;
    }
}

void fluency_mood_observe(FluencyModel *m, flu_tok_t tok, float weight) {
    if (!m || !m->mood_on || !m->mood_aff || m->mood_k == 0u || tok >= FLUENCY_VOCAB) {
        return;
    }
    /* Decay registers one token step: factor = 0.5^(1/half_life) ≈ 1 - ln2/hl. */
    float decay = 1.0f;
    if (m->mood_half_life > 0u) {
        decay = 1.0f - 0.693147f / (float)m->mood_half_life;
        if (decay < 0.5f) {
            decay = 0.5f;
        }
    }
    for (uint8_t k = 0; k < m->mood_k; ++k) {
        m->mood_reg[k] *= decay;
    }
    if (weight == 0.0f || !m->seen[tok]) {
        return;
    }
    const float *row = m->mood_aff + (size_t)tok * (size_t)m->mood_k;
    for (uint8_t k = 0; k < m->mood_k; ++k) {
        m->mood_reg[k] += weight * row[k];
        /* Bound: self-babble must not runaway (poisoning CI). */
        if (m->mood_reg[k] > 4.0f) {
            m->mood_reg[k] = 4.0f;
        } else if (m->mood_reg[k] < -4.0f) {
            m->mood_reg[k] = -4.0f;
        }
    }
}

/* Offline: cooc topk + k-means affinities for mood; cooc matrix for gain. */
int fluency_ctx_organs_build(FluencyModel *m, const flu_tok_t *text, size_t len,
                             uint32_t mood_k, uint32_t cooc_topk) {
    if (!m || !text || len == 0u) {
        return -1;
    }
    if (mood_k == 0u || mood_k > FLU_MOOD_K_MAX) {
        mood_k = FLU_MOOD_K_DEFAULT;
    }
    if (cooc_topk == 0u) {
        cooc_topk = FLU_GAIN_TOPK_DEFAULT;
    }
    if (cooc_topk > FLUENCY_VOCAB) {
        cooc_topk = FLUENCY_VOCAB;
    }

    /* Rank tokens by unigram (already filled by train) for topk. */
    typedef struct {
        uint32_t id;
        uint64_t n;
    } Rank;
    Rank *rank = (Rank *)malloc((size_t)FLUENCY_VOCAB * sizeof(Rank));
    if (!rank) {
        return -1;
    }
    uint32_t nseen = 0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (m->seen[b] && m->unigram[b] > 0u) {
            rank[nseen].id = b;
            rank[nseen].n = m->unigram[b];
            nseen++;
        }
    }
    /* Partial selection sort for top cooc_topk (offline; nseen ≤ V). */
    uint32_t topk = cooc_topk < nseen ? cooc_topk : nseen;
    for (uint32_t i = 0; i < topk; ++i) {
        uint32_t best = i;
        for (uint32_t j = i + 1u; j < nseen; ++j) {
            if (rank[j].n > rank[best].n) {
                best = j;
            }
        }
        Rank tmp = rank[i];
        rank[i] = rank[best];
        rank[best] = tmp;
    }
    if (topk == 0u) {
        free(rank);
        return -1;
    }

    free(m->gain_cooc);
    free(m->gain_tok);
    free(m->gain_vidx);
    free(m->mood_aff);
    m->gain_cooc = (uint32_t *)calloc((size_t)topk * (size_t)topk, sizeof(uint32_t));
    m->gain_tok = (flu_tok_t *)malloc((size_t)topk * sizeof(flu_tok_t));
    m->gain_vidx = (uint16_t *)malloc((size_t)FLUENCY_VOCAB * sizeof(uint16_t));
    m->mood_aff = (float *)calloc((size_t)FLUENCY_VOCAB * (size_t)mood_k, sizeof(float));
    if (!m->gain_cooc || !m->gain_tok || !m->gain_vidx || !m->mood_aff) {
        free(m->gain_cooc);
        m->gain_cooc = NULL;
        free(m->gain_tok);
        m->gain_tok = NULL;
        free(m->gain_vidx);
        m->gain_vidx = NULL;
        free(m->mood_aff);
        m->mood_aff = NULL;
        free(rank);
        return -1;
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        m->gain_vidx[b] = 0xFFFFu;
    }
    for (uint32_t i = 0; i < topk; ++i) {
        m->gain_tok[i] = (flu_tok_t)rank[i].id;
        m->gain_vidx[rank[i].id] = (uint16_t)i;
    }
    m->gain_topk = topk;
    m->mood_k = (uint8_t)mood_k;
    free(rank);

    /* Window cooc ±2 among topk. */
    const int W = 2;
    for (size_t i = 0; i < len; ++i) {
        flu_tok_t a = text[i];
        if (a >= FLUENCY_VOCAB || m->gain_vidx[a] == 0xFFFFu) {
            continue;
        }
        uint16_t ai = m->gain_vidx[a];
        for (int d = -W; d <= W; ++d) {
            if (d == 0) {
                continue;
            }
            int64_t j = (int64_t)i + d;
            if (j < 0 || (size_t)j >= len) {
                continue;
            }
            flu_tok_t b = text[(size_t)j];
            if (b >= FLUENCY_VOCAB || m->gain_vidx[b] == 0xFFFFu) {
                continue;
            }
            m->gain_cooc[(size_t)ai * topk + m->gain_vidx[b]]++;
        }
    }

    /* K-means-ish on L1-normalized cooc rows → mood affinities. */
    float *cent = (float *)calloc((size_t)mood_k * (size_t)topk, sizeof(float));
    uint32_t *assign = (uint32_t *)calloc(topk, sizeof(uint32_t));
    if (!cent || !assign) {
        free(cent);
        free(assign);
        return -1;
    }
    /* Init centroids: pick every topk/k-th row. */
    for (uint32_t k = 0; k < mood_k; ++k) {
        uint32_t src = (topk * k) / mood_k;
        if (src >= topk) {
            src = topk - 1u;
        }
        double sum = 0.0;
        for (uint32_t j = 0; j < topk; ++j) {
            float v = (float)m->gain_cooc[(size_t)src * topk + j];
            cent[(size_t)k * topk + j] = v;
            sum += (double)v;
        }
        if (sum > 0.0) {
            for (uint32_t j = 0; j < topk; ++j) {
                cent[(size_t)k * topk + j] = (float)((double)cent[(size_t)k * topk + j] / sum);
            }
        }
    }
    for (int iter = 0; iter < 8; ++iter) {
        for (uint32_t i = 0; i < topk; ++i) {
            double row_sum = 0.0;
            for (uint32_t j = 0; j < topk; ++j) {
                row_sum += (double)m->gain_cooc[(size_t)i * topk + j];
            }
            uint32_t best_k = 0;
            double best_d = 1e300;
            for (uint32_t k = 0; k < mood_k; ++k) {
                double d = 0.0;
                for (uint32_t j = 0; j < topk; ++j) {
                    double rv =
                        row_sum > 0.0
                            ? (double)m->gain_cooc[(size_t)i * topk + j] / row_sum
                            : 0.0;
                    double diff = rv - (double)cent[(size_t)k * topk + j];
                    d += diff * diff;
                }
                if (d < best_d) {
                    best_d = d;
                    best_k = k;
                }
            }
            assign[i] = best_k;
        }
        /* Recompute centroids. */
        memset(cent, 0, (size_t)mood_k * (size_t)topk * sizeof(float));
        uint32_t *cnt = (uint32_t *)calloc(mood_k, sizeof(uint32_t));
        if (!cnt) {
            break;
        }
        for (uint32_t i = 0; i < topk; ++i) {
            uint32_t k = assign[i];
            cnt[k]++;
            double row_sum = 0.0;
            for (uint32_t j = 0; j < topk; ++j) {
                row_sum += (double)m->gain_cooc[(size_t)i * topk + j];
            }
            for (uint32_t j = 0; j < topk; ++j) {
                float rv =
                    row_sum > 0.0
                        ? (float)((double)m->gain_cooc[(size_t)i * topk + j] / row_sum)
                        : 0.0f;
                cent[(size_t)k * topk + j] += rv;
            }
        }
        for (uint32_t k = 0; k < mood_k; ++k) {
            if (cnt[k] == 0u) {
                continue;
            }
            for (uint32_t j = 0; j < topk; ++j) {
                cent[(size_t)k * topk + j] /= (float)cnt[k];
            }
        }
        free(cnt);
    }

    /* Soft affinity: 1 / (1 + dist) normalized over k for each topk token;
     * non-topk tokens get uniform tiny affinity. */
    for (uint32_t i = 0; i < topk; ++i) {
        flu_tok_t tok = m->gain_tok[i];
        double row_sum = 0.0;
        for (uint32_t j = 0; j < topk; ++j) {
            row_sum += (double)m->gain_cooc[(size_t)i * topk + j];
        }
        float soft[FLU_MOOD_K_MAX];
        float ssum = 0.0f;
        for (uint32_t k = 0; k < mood_k; ++k) {
            double d = 0.0;
            for (uint32_t j = 0; j < topk; ++j) {
                double rv =
                    row_sum > 0.0 ? (double)m->gain_cooc[(size_t)i * topk + j] / row_sum : 0.0;
                double diff = rv - (double)cent[(size_t)k * topk + j];
                d += diff * diff;
            }
            soft[k] = 1.0f / (1.0f + (float)d);
            ssum += soft[k];
        }
        float *out = m->mood_aff + (size_t)tok * (size_t)mood_k;
        for (uint32_t k = 0; k < mood_k; ++k) {
            out[k] = ssum > 0.0f ? soft[k] / ssum : (1.0f / (float)mood_k);
        }
    }
    float uni = 1.0f / (float)mood_k;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (!m->seen[b] || m->gain_vidx[b] != 0xFFFFu) {
            continue;
        }
        float *out = m->mood_aff + (size_t)b * (size_t)mood_k;
        for (uint32_t k = 0; k < mood_k; ++k) {
            out[k] = uni;
        }
    }

    free(cent);
    free(assign);
    fluency_mood_reset(m);
    return 0;
}

/* ---- Phonebook hubs: PPMI k-means → graded membership (step 6) ------------- */

void fluency_hubs_set(FluencyModel *m, int on) {
    if (m) {
        m->hub_on = on ? 1u : 0u;
    }
}

static void fluency_hubs_clear_tables(FluencyModel *m) {
    free(m->tok_hubs);
    m->tok_hubs = NULL;
    free(m->hub_mem_tok);
    m->hub_mem_tok = NULL;
    free(m->hub_mem_w);
    m->hub_mem_w = NULL;
    free(m->hub_mem_n);
    m->hub_mem_n = NULL;
    m->hub_h = 0;
}

/* Materialize hubs from dense topk×topk cooc (row = context signature). */
int fluency_hubs_build_from_cooc(FluencyModel *m, const uint32_t *cooc, uint32_t topk,
                                 const flu_tok_t *tok_of_row, uint32_t n_hubs) {
    if (!m || !cooc || topk < 2u) {
        return -1;
    }
    if (n_hubs == 0u) {
        n_hubs = FLU_HUB_H_DEFAULT;
    }
    if (n_hubs > FLU_HUB_H_MAX) {
        n_hubs = FLU_HUB_H_MAX;
    }
    if (n_hubs > topk) {
        n_hubs = topk;
    }

    /* PPMI row vectors (heap offline). */
    double *row_sum = (double *)calloc(topk, sizeof(double));
    double *col_sum = (double *)calloc(topk, sizeof(double));
    double *ppmi = (double *)calloc((size_t)topk * (size_t)topk, sizeof(double));
    if (!row_sum || !col_sum || !ppmi) {
        free(row_sum);
        free(col_sum);
        free(ppmi);
        return -1;
    }
    double grand = 0.0;
    for (uint32_t i = 0; i < topk; ++i) {
        for (uint32_t j = 0; j < topk; ++j) {
            double c = (double)cooc[(size_t)i * topk + j];
            row_sum[i] += c;
            col_sum[j] += c;
            grand += c;
        }
    }
    if (grand <= 0.0) {
        free(row_sum);
        free(col_sum);
        free(ppmi);
        return -1;
    }
    for (uint32_t i = 0; i < topk; ++i) {
        for (uint32_t j = 0; j < topk; ++j) {
            double c = (double)cooc[(size_t)i * topk + j];
            if (c <= 0.0 || row_sum[i] <= 0.0 || col_sum[j] <= 0.0) {
                continue;
            }
            double pmi = log((c * grand) / (row_sum[i] * col_sum[j]));
            if (pmi > 0.0) {
                ppmi[(size_t)i * topk + j] = pmi;
            }
        }
    }
    /* L2-normalize rows for cosine k-means. */
    for (uint32_t i = 0; i < topk; ++i) {
        double n2 = 0.0;
        for (uint32_t j = 0; j < topk; ++j) {
            double v = ppmi[(size_t)i * topk + j];
            n2 += v * v;
        }
        if (n2 > 0.0) {
            double inv = 1.0 / sqrt(n2);
            for (uint32_t j = 0; j < topk; ++j) {
                ppmi[(size_t)i * topk + j] *= inv;
            }
        }
    }

    float *cent = (float *)calloc((size_t)n_hubs * (size_t)topk, sizeof(float));
    uint32_t *assign = (uint32_t *)calloc(topk, sizeof(uint32_t));
    double *dist_to = (double *)calloc((size_t)topk * (size_t)n_hubs, sizeof(double));
    if (!cent || !assign || !dist_to) {
        free(cent);
        free(assign);
        free(dist_to);
        free(row_sum);
        free(col_sum);
        free(ppmi);
        return -1;
    }
    /* Deterministic init: every topk/n_hubs-th row. */
    for (uint32_t k = 0; k < n_hubs; ++k) {
        uint32_t src = (topk * k) / n_hubs;
        if (src >= topk) {
            src = topk - 1u;
        }
        for (uint32_t j = 0; j < topk; ++j) {
            cent[(size_t)k * topk + j] = (float)ppmi[(size_t)src * topk + j];
        }
    }
    for (int iter = 0; iter < 12; ++iter) {
        for (uint32_t i = 0; i < topk; ++i) {
            uint32_t best_k = 0;
            double best_d = 1e300;
            for (uint32_t k = 0; k < n_hubs; ++k) {
                double d = 0.0;
                for (uint32_t j = 0; j < topk; ++j) {
                    double diff = ppmi[(size_t)i * topk + j] - (double)cent[(size_t)k * topk + j];
                    d += diff * diff;
                }
                dist_to[(size_t)i * n_hubs + k] = d;
                if (d < best_d) {
                    best_d = d;
                    best_k = k;
                }
            }
            assign[i] = best_k;
        }
        memset(cent, 0, (size_t)n_hubs * (size_t)topk * sizeof(float));
        uint32_t *cnt = (uint32_t *)calloc(n_hubs, sizeof(uint32_t));
        if (!cnt) {
            break;
        }
        for (uint32_t i = 0; i < topk; ++i) {
            uint32_t k = assign[i];
            cnt[k]++;
            for (uint32_t j = 0; j < topk; ++j) {
                cent[(size_t)k * topk + j] += (float)ppmi[(size_t)i * topk + j];
            }
        }
        for (uint32_t k = 0; k < n_hubs; ++k) {
            if (cnt[k] == 0u) {
                /* Re-seed empty hub from a far row (deterministic: row k). */
                uint32_t src = k % topk;
                for (uint32_t j = 0; j < topk; ++j) {
                    cent[(size_t)k * topk + j] = (float)ppmi[(size_t)src * topk + j];
                }
                continue;
            }
            double n2 = 0.0;
            for (uint32_t j = 0; j < topk; ++j) {
                cent[(size_t)k * topk + j] /= (float)cnt[k];
                n2 += (double)cent[(size_t)k * topk + j] * (double)cent[(size_t)k * topk + j];
            }
            if (n2 > 0.0) {
                float inv = (float)(1.0 / sqrt(n2));
                for (uint32_t j = 0; j < topk; ++j) {
                    cent[(size_t)k * topk + j] *= inv;
                }
            }
        }
        free(cnt);
    }

    /* Soft membership: nearest hub + hubs within 1.5× best distance, ≤ MEM_MAX.
     * weight ∝ 1/(1+d) normalized over accepted hubs → uq0_16. */
    fluency_hubs_clear_tables(m);
    m->tok_hubs =
        (FluencyHubSlot *)malloc((size_t)FLUENCY_VOCAB * (size_t)FLU_HUB_MEM_MAX * sizeof(FluencyHubSlot));
    m->hub_mem_tok = (flu_tok_t *)malloc((size_t)n_hubs * (size_t)FLU_HUB_SIZE_CAP * sizeof(flu_tok_t));
    m->hub_mem_w =
        (nerva_uq0_16_t *)malloc((size_t)n_hubs * (size_t)FLU_HUB_SIZE_CAP * sizeof(nerva_uq0_16_t));
    m->hub_mem_n = (uint16_t *)calloc(n_hubs, sizeof(uint16_t));
    if (!m->tok_hubs || !m->hub_mem_tok || !m->hub_mem_w || !m->hub_mem_n) {
        fluency_hubs_clear_tables(m);
        free(cent);
        free(assign);
        free(dist_to);
        free(row_sum);
        free(col_sum);
        free(ppmi);
        return -1;
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        for (uint32_t s = 0; s < FLU_HUB_MEM_MAX; ++s) {
            m->tok_hubs[(size_t)b * FLU_HUB_MEM_MAX + s].hub = 0xFFFFu;
            m->tok_hubs[(size_t)b * FLU_HUB_MEM_MAX + s].weight = 0;
        }
    }
    for (uint32_t i = 0; i < (size_t)n_hubs * FLU_HUB_SIZE_CAP; ++i) {
        m->hub_mem_tok[i] = (flu_tok_t)FLUENCY_VOCAB;
        m->hub_mem_w[i] = 0;
    }

    /* Per-hub: collect (tok, strength) then keep top SIZE_CAP. */
    typedef struct {
        flu_tok_t tok;
        float str;
    } MemCand;
    MemCand *bucket = (MemCand *)malloc((size_t)n_hubs * (size_t)topk * sizeof(MemCand));
    uint32_t *bcnt = (uint32_t *)calloc(n_hubs, sizeof(uint32_t));
    if (!bucket || !bcnt) {
        free(bucket);
        free(bcnt);
        fluency_hubs_clear_tables(m);
        free(cent);
        free(assign);
        free(dist_to);
        free(row_sum);
        free(col_sum);
        free(ppmi);
        return -1;
    }

    for (uint32_t i = 0; i < topk; ++i) {
        flu_tok_t tok = tok_of_row ? tok_of_row[i] : (flu_tok_t)i;
        if (tok >= FLUENCY_VOCAB) {
            continue;
        }
        /* Rank hubs by distance. */
        uint32_t order[FLU_HUB_H_MAX];
        for (uint32_t k = 0; k < n_hubs; ++k) {
            order[k] = k;
        }
        for (uint32_t a = 0; a < n_hubs; ++a) {
            for (uint32_t b = a + 1u; b < n_hubs; ++b) {
                if (dist_to[(size_t)i * n_hubs + order[b]] <
                    dist_to[(size_t)i * n_hubs + order[a]]) {
                    uint32_t tmp = order[a];
                    order[a] = order[b];
                    order[b] = tmp;
                }
            }
        }
        double best_d = dist_to[(size_t)i * n_hubs + order[0]];
        double thr = best_d * 1.5 + 1e-9;
        float soft[FLU_HUB_MEM_MAX];
        uint16_t hubs[FLU_HUB_MEM_MAX];
        uint32_t nacc = 0;
        float ssum = 0.0f;
        for (uint32_t r = 0; r < n_hubs && nacc < FLU_HUB_MEM_MAX; ++r) {
            double d = dist_to[(size_t)i * n_hubs + order[r]];
            if (r > 0u && d > thr) {
                break;
            }
            hubs[nacc] = (uint16_t)order[r];
            soft[nacc] = 1.0f / (1.0f + (float)d);
            ssum += soft[nacc];
            nacc++;
        }
        for (uint32_t s = 0; s < nacc; ++s) {
            float w = ssum > 0.0f ? soft[s] / ssum : (1.0f / (float)nacc);
            nerva_uq0_16_t uq = (nerva_uq0_16_t)(w * (float)NERVA_UQ0_16_ONE + 0.5f);
            if (uq == 0u && w > 0.0f) {
                uq = 1u;
            }
            m->tok_hubs[(size_t)tok * FLU_HUB_MEM_MAX + s].hub = hubs[s];
            m->tok_hubs[(size_t)tok * FLU_HUB_MEM_MAX + s].weight = uq;
            uint32_t h = hubs[s];
            uint32_t bi = bcnt[h]++;
            bucket[(size_t)h * topk + bi].tok = tok;
            bucket[(size_t)h * topk + bi].str = w;
        }
    }

    for (uint32_t h = 0; h < n_hubs; ++h) {
        uint32_t n = bcnt[h];
        MemCand *row = bucket + (size_t)h * topk;
        /* Selection-sort top SIZE_CAP by strength. */
        uint32_t keep = n < FLU_HUB_SIZE_CAP ? n : FLU_HUB_SIZE_CAP;
        for (uint32_t a = 0; a < keep; ++a) {
            uint32_t best = a;
            for (uint32_t b = a + 1u; b < n; ++b) {
                if (row[b].str > row[best].str) {
                    best = b;
                }
            }
            MemCand tmp = row[a];
            row[a] = row[best];
            row[best] = tmp;
        }
        m->hub_mem_n[h] = (uint16_t)keep;
        size_t base = (size_t)h * (size_t)FLU_HUB_SIZE_CAP;
        for (uint32_t a = 0; a < keep; ++a) {
            m->hub_mem_tok[base + a] = row[a].tok;
            nerva_uq0_16_t uq =
                (nerva_uq0_16_t)(row[a].str * (float)NERVA_UQ0_16_ONE + 0.5f);
            if (uq == 0u && row[a].str > 0.0f) {
                uq = 1u;
            }
            m->hub_mem_w[base + a] = uq;
        }
    }

    m->hub_h = (uint16_t)n_hubs;
    m->hub_on = 0; /* built but closed */
    m->hub_dial_count = 0;
    m->hub_dial_ops = 0;
    m->hub_predict_count = 0;

    free(bucket);
    free(bcnt);
    free(cent);
    free(assign);
    free(dist_to);
    free(row_sum);
    free(col_sum);
    free(ppmi);
    return 0;
}

int fluency_hubs_build(FluencyModel *m, const flu_tok_t *text, size_t len, uint32_t n_hubs,
                       uint32_t cooc_topk) {
    if (!m || !text || len == 0u) {
        return -1;
    }
    if (cooc_topk == 0u) {
        cooc_topk = FLU_HUB_COOC_TOPK_DEFAULT;
    }
    if (cooc_topk > FLUENCY_VOCAB) {
        cooc_topk = FLUENCY_VOCAB;
    }

    typedef struct {
        uint32_t id;
        uint64_t n;
    } Rank;
    Rank *rank = (Rank *)malloc((size_t)FLUENCY_VOCAB * sizeof(Rank));
    if (!rank) {
        return -1;
    }
    uint32_t nseen = 0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (m->seen[b] && m->unigram[b] > 0u) {
            rank[nseen].id = b;
            rank[nseen].n = m->unigram[b];
            nseen++;
        }
    }
    uint32_t topk = cooc_topk < nseen ? cooc_topk : nseen;
    if (topk < 2u) {
        free(rank);
        return -1;
    }
    for (uint32_t i = 0; i < topk; ++i) {
        uint32_t best = i;
        for (uint32_t j = i + 1u; j < nseen; ++j) {
            if (rank[j].n > rank[best].n) {
                best = j;
            }
        }
        Rank tmp = rank[i];
        rank[i] = rank[best];
        rank[best] = tmp;
    }

    flu_tok_t *tok_of_row = (flu_tok_t *)malloc((size_t)topk * sizeof(flu_tok_t));
    uint16_t *vidx = (uint16_t *)malloc((size_t)FLUENCY_VOCAB * sizeof(uint16_t));
    uint32_t *mat = (uint32_t *)calloc((size_t)topk * (size_t)topk, sizeof(uint32_t));
    if (!tok_of_row || !vidx || !mat) {
        free(tok_of_row);
        free(vidx);
        free(mat);
        free(rank);
        return -1;
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        vidx[b] = 0xFFFFu;
    }
    for (uint32_t i = 0; i < topk; ++i) {
        tok_of_row[i] = (flu_tok_t)rank[i].id;
        vidx[rank[i].id] = (uint16_t)i;
    }
    free(rank);

    const int W = FLU_HUB_WINDOW;
    for (size_t i = 0; i < len; ++i) {
        flu_tok_t a = text[i];
        if (a >= FLUENCY_VOCAB || vidx[a] == 0xFFFFu) {
            continue;
        }
        uint16_t ai = vidx[a];
        for (int d = -W; d <= W; ++d) {
            if (d == 0) {
                continue;
            }
            int64_t j = (int64_t)i + d;
            if (j < 0 || (size_t)j >= len) {
                continue;
            }
            flu_tok_t b = text[(size_t)j];
            if (b >= FLUENCY_VOCAB || vidx[b] == 0xFFFFu) {
                continue;
            }
            mat[(size_t)ai * topk + vidx[b]]++;
        }
    }
    free(vidx);

    int rc = fluency_hubs_build_from_cooc(m, mat, topk, tok_of_row, n_hubs);
    free(mat);
    free(tok_of_row);
    return rc;
}

void fluency_hubs_print(const FluencyModel *m, flu_tok_t tok, FILE *out,
                        const char *(*word_fn)(void *ud, flu_tok_t id), void *ud) {
    if (!m || !out) {
        return;
    }
    fprintf(out, "hubs: on=%u H=%u scale=%d conf_margin=%d last_dialed=%u last_ops=%llu\n",
            (unsigned)m->hub_on, (unsigned)m->hub_h, (int)m->hub_scale, (int)m->hub_conf_margin,
            (unsigned)m->last_dialed, (unsigned long long)m->hub_last_ops);
    if (!m->tok_hubs || m->hub_h == 0u) {
        fprintf(out, "  (no hubs built — fluency_hubs_build)\n");
        return;
    }
    if (tok >= FLUENCY_VOCAB) {
        fprintf(out, "  token id %u out of range\n", (unsigned)tok);
        return;
    }
    const char *name = word_fn ? word_fn(ud, tok) : NULL;
    char idbuf[16];
    if (!name) {
        snprintf(idbuf, sizeof(idbuf), "#%u", (unsigned)tok);
        name = idbuf;
    }
    fprintf(out, "  token '%s' (id %u) pages:\n", name, (unsigned)tok);
    const FluencyHubSlot *slots = m->tok_hubs + (size_t)tok * FLU_HUB_MEM_MAX;
    int any = 0;
    for (uint32_t s = 0; s < FLU_HUB_MEM_MAX; ++s) {
        if (slots[s].hub == 0xFFFFu) {
            continue;
        }
        any = 1;
        uint16_t h = slots[s].hub;
        double w = (double)slots[s].weight / (double)NERVA_UQ0_16_ONE;
        fprintf(out, "    hub %u  membership=%.4f  top co-members:", (unsigned)h, w);
        if (h < m->hub_h && m->hub_mem_tok && m->hub_mem_n) {
            size_t base = (size_t)h * FLU_HUB_SIZE_CAP;
            uint16_t n = m->hub_mem_n[h];
            uint16_t show = n < 10u ? n : 10u;
            for (uint16_t j = 0; j < show; ++j) {
                flu_tok_t t = m->hub_mem_tok[base + j];
                if (t == tok) {
                    continue;
                }
                const char *tn = word_fn ? word_fn(ud, t) : NULL;
                char tbuf[16];
                if (!tn) {
                    snprintf(tbuf, sizeof(tbuf), "#%u", (unsigned)t);
                    tn = tbuf;
                }
                double mw = (double)m->hub_mem_w[base + j] / (double)NERVA_UQ0_16_ONE;
                fprintf(out, " %s(%.2f)", tn, mw);
            }
        }
        fprintf(out, "\n");
    }
    if (!any) {
        fprintf(out, "    (token not on any hub page — outside cooc topk or zero signature)\n");
    }
}

void fluency_hubs_sanity_table(const FluencyModel *m, FILE *out, uint32_t n_hubs,
                               const char *(*word_fn)(void *ud, flu_tok_t id), void *ud) {
    if (!m || !out) {
        return;
    }
    if (!m->hub_mem_tok || m->hub_h == 0u) {
        fprintf(out, "hubs sanity: (none built)\n");
        return;
    }
    uint32_t show = n_hubs && n_hubs < m->hub_h ? n_hubs : m->hub_h;
    if (show > 10u) {
        show = 10u;
    }
    /* Member count distribution. */
    uint32_t min_m = UINT32_MAX, max_m = 0, sum_m = 0;
    for (uint16_t h = 0; h < m->hub_h; ++h) {
        uint32_t n = m->hub_mem_n[h];
        if (n < min_m) {
            min_m = n;
        }
        if (n > max_m) {
            max_m = n;
        }
        sum_m += n;
    }
    if (min_m == UINT32_MAX) {
        min_m = 0;
    }
    fprintf(out, "hubs: H=%u members/hub min=%u max=%u mean=%.1f size_cap=%u mem_max=%u\n",
            (unsigned)m->hub_h, min_m, max_m,
            m->hub_h ? (double)sum_m / (double)m->hub_h : 0.0, (unsigned)FLU_HUB_SIZE_CAP,
            (unsigned)FLU_HUB_MEM_MAX);
    fprintf(out, "sample hubs (top-10 members by membership strength):\n");
    for (uint32_t h = 0; h < show; ++h) {
        fprintf(out, "  hub %u (n=%u):", h, (unsigned)m->hub_mem_n[h]);
        size_t base = (size_t)h * FLU_HUB_SIZE_CAP;
        uint16_t n = m->hub_mem_n[h];
        uint16_t k = n < 10u ? n : 10u;
        for (uint16_t j = 0; j < k; ++j) {
            flu_tok_t t = m->hub_mem_tok[base + j];
            const char *tn = word_fn ? word_fn(ud, t) : NULL;
            char tbuf[16];
            if (!tn) {
                snprintf(tbuf, sizeof(tbuf), "#%u", (unsigned)t);
                tn = tbuf;
            }
            fprintf(out, " %s", tn);
        }
        fprintf(out, "\n");
    }
}

static void fluency_read_charges(const FluencyModel *m, double *score) {
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        score[b] = 0.0;
        if (m->seen[b] && m->tok_node[b] != NERVA_INVALID_ID) {
            int32_t v = (int32_t)m->e->nodes[m->tok_node[b]].v;
            if (v > 0) {
                score[b] = (double)v;
            }
        }
    }
}

/* Fire only order-k context (for /why decomposition). */
static void fluency_fire_order(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len, uint32_t k) {
    if ((size_t)k > ctx_len || k == 0u) {
        return;
    }
    const flu_tok_t *bytes = ctx + (ctx_len - k);
    char name[8u + 5u * FLUENCY_MAX_ORDER];
    fluency_ctx_name(name, sizeof(name), bytes, k);
    uint32_t id = fluency_name_lookup(m, name);
    if (id != NERVA_INVALID_ID) {
        nerva_activate_node(m->e, id, NERVA_Q8_8_ONE);
        nerva_tick_n(m->e, 3u);
    }
}

int fluency_explain(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len, FluencyWhy *out) {
    if (!m || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));

    /* Heap: V=8192 × several scoreboards exceeds default Windows stack. */
    size_t vbytes = (size_t)FLUENCY_VOCAB * sizeof(double);
    double *fast_sc = (double *)calloc(1, vbytes);
    double *mood_sc = (double *)calloc(1, vbytes);
    double *gain_sc = (double *)calloc(1, vbytes);
    double *hub_sc = (double *)calloc(1, vbytes);
    double *skip_sc = (double *)calloc(1, vbytes);
    double *total_sc = (double *)calloc(1, vbytes);
    double *pre_hub_sc = (double *)calloc(1, vbytes);
    double *ngram_flat =
        (double *)calloc((size_t)(FLUENCY_MAX_ORDER + 1u) * (size_t)FLUENCY_VOCAB, sizeof(double));
    if (!fast_sc || !mood_sc || !gain_sc || !hub_sc || !skip_sc || !total_sc || !pre_hub_sc ||
        !ngram_flat) {
        free(fast_sc);
        free(mood_sc);
        free(gain_sc);
        free(hub_sc);
        free(skip_sc);
        free(total_sc);
        free(pre_hub_sc);
        free(ngram_flat);
        return -1;
    }
#define NGRAM_ROW(k) (ngram_flat + (size_t)(k) * (size_t)FLUENCY_VOCAB)

    uint8_t save_mood = m->mood_on;
    uint8_t save_gain = m->gain_on;
    uint8_t save_hub = m->hub_on;
    nerva_uq0_16_t save_lam = m->fast_lambda;

    m->mood_on = 0;
    m->gain_on = 0;
    m->hub_on = 0;
    m->fast_lambda = 0;
    for (uint32_t k = 1u; k <= m->order; ++k) {
        if ((size_t)k > ctx_len) {
            break;
        }
        fluency_reset_charges(m->e);
        fluency_fire_order(m, ctx, ctx_len, k);
        fluency_read_charges(m, NGRAM_ROW(k));
    }

    /* Skip column: isolate the gapped-context contribution. */
    if (m->skip_on) {
        fluency_reset_charges(m->e);
        if (fluency_skip_fire(m, ctx, ctx_len)) {
            nerva_tick_n(m->e, 3u);
        }
        fluency_read_charges(m, skip_sc);
    }

    fluency_reset_charges(m->e);
    m->fast_lambda = save_lam;
    /* Convention: instance charge folds into fast% (same inject family).
     * Mirror predict: instance hit replaces type inject for this step. */
    if (!fluency_instance_inject(m, ctx, ctx_len)) {
        fluency_fast_inject(m, ctx, ctx_len);
    }
    fluency_read_charges(m, fast_sc);

    flu_tok_t fsrc = (ctx_len > 0u) ? ctx[ctx_len - 1u] : (flu_tok_t)FLUENCY_VOCAB;
    nerva_q8_8_t fweff = 0;
    int fhit = 0;
    m->instance_last_hit = 0;

    fluency_reset_charges(m->e);
    m->fast_lambda = 0;
    m->mood_on = save_mood;
    fluency_mood_inject(m);
    fluency_read_charges(m, mood_sc);

    fluency_reset_charges(m->e);
    m->mood_on = 0;
    m->gain_on = save_gain;
    fluency_gain_inject(m, ctx, ctx_len);
    fluency_read_charges(m, gain_sc);

    /* Hub column: isolate inject only if dial would fire for this context. */
    m->gain_on = 0;
    m->mood_on = 0;
    m->hub_on = 0;
    fluency_reset_charges(m->e);
    {
        int any = 0;
        for (uint32_t k = 1u; k <= m->order; ++k) {
            if ((size_t)k > ctx_len) {
                break;
            }
            const flu_tok_t *bytes = ctx + (ctx_len - k);
            char name[8u + 5u * FLUENCY_MAX_ORDER];
            fluency_ctx_name(name, sizeof(name), bytes, k);
            uint32_t id = fluency_name_lookup(m, name);
            if (id != NERVA_INVALID_ID) {
                nerva_activate_node(m->e, id, NERVA_Q8_8_ONE);
                any = 1;
            }
        }
        if (fluency_skip_fire(m, ctx, ctx_len)) {
            any = 1;
        }
        if (any) {
            nerva_tick_n(m->e, 3u);
        }
    }
    if (!fluency_instance_inject(m, ctx, ctx_len)) {
        fluency_fast_inject(m, ctx, ctx_len);
    }
    m->gain_on = save_gain;
    m->mood_on = save_mood;
    fluency_gain_inject(m, ctx, ctx_len);
    fluency_mood_inject(m);
    fluency_read_charges(m, pre_hub_sc);
    int32_t pre_margin = fluency_charge_margin(m, pre_hub_sc, NULL);
    int will_dial = (save_hub && m->hub_h > 0u && m->hub_scale > 0 && m->hub_conf_margin > 0 &&
                     pre_margin < m->hub_conf_margin);
    if (will_dial) {
        fluency_reset_charges(m->e);
        m->hub_on = 1;
        fluency_hub_inject(m, ctx, ctx_len, NULL);
        fluency_read_charges(m, hub_sc);
    }

    m->mood_on = save_mood;
    m->gain_on = save_gain;
    m->hub_on = save_hub;
    m->fast_lambda = save_lam;
    fluency_reset_charges(m->e);
    {
        int any = 0;
        for (uint32_t k = 1u; k <= m->order; ++k) {
            if ((size_t)k > ctx_len) {
                break;
            }
            const flu_tok_t *bytes = ctx + (ctx_len - k);
            char name[8u + 5u * FLUENCY_MAX_ORDER];
            fluency_ctx_name(name, sizeof(name), bytes, k);
            uint32_t id = fluency_name_lookup(m, name);
            if (id != NERVA_INVALID_ID) {
                nerva_activate_node(m->e, id, NERVA_Q8_8_ONE);
                any = 1;
            }
        }
        if (fluency_skip_fire(m, ctx, ctx_len)) {
            any = 1;
        }
        if (any) {
            nerva_tick_n(m->e, 3u);
        }
    }
    if (!fluency_instance_inject(m, ctx, ctx_len)) {
        fluency_fast_inject(m, ctx, ctx_len);
    }
    fluency_gain_inject(m, ctx, ctx_len);
    fluency_mood_inject(m);
    if (will_dial) {
        fluency_hub_inject(m, ctx, ctx_len, NULL);
    }
    fluency_read_charges(m, total_sc);

    m->mood_on = save_mood;
    m->gain_on = save_gain;
    m->hub_on = save_hub;
    m->fast_lambda = save_lam;

    int top_i[FLU_WHY_TOP];
    double top_v[FLU_WHY_TOP];
    for (uint32_t i = 0; i < FLU_WHY_TOP; ++i) {
        top_i[i] = -1;
        top_v[i] = -1.0;
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (!m->seen[b]) {
            continue;
        }
        double v = total_sc[b];
        for (uint32_t i = 0; i < FLU_WHY_TOP; ++i) {
            if (v > top_v[i]) {
                for (uint32_t j = FLU_WHY_TOP - 1u; j > i; --j) {
                    top_i[j] = top_i[j - 1u];
                    top_v[j] = top_v[j - 1u];
                }
                top_i[i] = (int)b;
                top_v[i] = v;
                break;
            }
        }
    }

    if (top_i[0] < 0) {
        out->valid = 0;
        m->last_why = *out;
        free(fast_sc);
        free(mood_sc);
        free(gain_sc);
        free(hub_sc);
        free(skip_sc);
        free(total_sc);
        free(pre_hub_sc);
        free(ngram_flat);
        return -1;
    }
    out->valid = 1;
    out->win = (flu_tok_t)top_i[0];
    out->hub_dialed = will_dial;

    if (fsrc < FLUENCY_VOCAB && m->fast_out && save_lam != 0u) {
        FluencyFastSlot *base = &m->fast_out[(size_t)fsrc * (size_t)m->fast_slots];
        for (uint16_t i = 0; i < m->fast_slots; ++i) {
            if (base[i].target == out->win) {
                fweff = fluency_fast_weff(m, base[i].weight, base[i].deposit_pos);
                if (fweff > 0) {
                    fhit = 1;
                    break;
                }
            }
        }
    }
    /* Instance edge into winner counts as fast_hit (folded convention). */
    if (!fhit && m->instance_on && ctx_len >= 2u && save_lam != 0u && m->instance_pool) {
        flu_tok_t cue = ctx[ctx_len - 2u];
        flu_tok_t isrc = ctx[ctx_len - 1u];
        int is_empty = 0;
        int islot = fluency_instance_find(m, cue, isrc, &is_empty);
        if (islot >= 0 && !is_empty && m->instance_pool[islot].live &&
            m->instance_pool[islot].target == out->win) {
            nerva_q8_8_t iweff = fluency_fast_weff(m, m->instance_pool[islot].weight,
                                                   m->instance_pool[islot].deposit_pos);
            if (iweff > 0) {
                fhit = 1;
                fweff = iweff;
                fsrc = isrc;
                m->instance_last_hit = 1;
                m->instance_last_cue = cue;
                m->instance_last_src = isrc;
                m->instance_last_tgt = out->win;
                m->instance_last_weff = iweff;
            }
        }
    }
    out->fast_src = fsrc;
    out->fast_weff = fweff;
    out->fast_hit = fhit;

    out->n_cands = 0;
    for (uint32_t i = 0; i < FLU_WHY_TOP; ++i) {
        if (top_i[i] < 0) {
            break;
        }
        flu_tok_t t = (flu_tok_t)top_i[i];
        FluencyWhyCand *c = &out->cand[out->n_cands++];
        memset(c, 0, sizeof(*c));
        c->tok = t;
        c->total = total_sc[t];
        c->fast = fast_sc[t];
        c->mood = mood_sc[t];
        c->gain = gain_sc[t];
        c->hub = hub_sc[t];
        c->skip = skip_sc[t];
        for (uint32_t k = 1u; k <= m->order; ++k) {
            c->ngram[k] = NGRAM_ROW(k)[t];
            c->ngram_sum += NGRAM_ROW(k)[t];
        }
        double accounted = c->ngram_sum + c->fast + c->mood + c->gain + c->hub + c->skip;
        c->unexplained = c->total - accounted;
        if (c->unexplained > -0.5 && c->unexplained < 0.5) {
            c->unexplained = 0.0;
        }
    }

    m->last_why = *out;
    free(fast_sc);
    free(mood_sc);
    free(gain_sc);
    free(hub_sc);
    free(skip_sc);
    free(total_sc);
    free(pre_hub_sc);
    free(ngram_flat);
#undef NGRAM_ROW
    return 0;
}

void fluency_why_print(const FluencyWhy *w, FILE *out,
                       const char *(*word_fn)(void *ud, flu_tok_t id), void *ud) {
    if (!w || !out) {
        return;
    }
    if (!w->valid || w->n_cands == 0u) {
        fprintf(out, "why: (no prediction to explain — generate first)\n");
        return;
    }
    fprintf(out, "why: winner + top losers (integrator charge split)%s\n",
            w->hub_dialed ? " [hub dialed]" : "");
    fprintf(out, "%-12s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", "token", "total", "ngram%",
            "fast%", "mood%", "gain%", "hub%", "skip%", "unexpl%", "rel%");
    double win_tot = w->cand[0].total > 0.0 ? w->cand[0].total : 1.0;
    for (uint32_t i = 0; i < w->n_cands; ++i) {
        const FluencyWhyCand *c = &w->cand[i];
        const char *name = word_fn ? word_fn(ud, c->tok) : NULL;
        char idbuf[16];
        if (!name) {
            snprintf(idbuf, sizeof(idbuf), "#%u", (unsigned)c->tok);
            name = idbuf;
        }
        double t = c->total > 0.0 ? c->total : 1.0;
        double ng = 100.0 * c->ngram_sum / t;
        double fa = 100.0 * c->fast / t;
        double mo = 100.0 * c->mood / t;
        double ga = 100.0 * c->gain / t;
        double hu = 100.0 * c->hub / t;
        double sk = 100.0 * c->skip / t;
        double un = 100.0 * c->unexplained / t;
        double mark = 100.0 * c->total / win_tot;
        fprintf(out, "%-12s %8.1f %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%%",
                name, c->total, ng, fa, mo, ga, hu, sk, un, mark);
        if (i == 0u) {
            fprintf(out, "  << win");
        }
        fprintf(out, "\n");
        /* Per-order n-gram breakdown for winner. */
        if (i == 0u) {
            fprintf(out, "  n-gram by order:");
            for (uint32_t k = 1u; k <= FLUENCY_MAX_ORDER; ++k) {
                if (c->ngram[k] != 0.0) {
                    fprintf(out, " o%u=%.1f", k, c->ngram[k]);
                }
            }
            if (c->ngram_sum == 0.0) {
                fprintf(out, " (none)");
            }
            fprintf(out, "\n");
            if (w->fast_hit) {
                const char *sname = word_fn ? word_fn(ud, w->fast_src) : NULL;
                char sbuf[16];
                if (!sname) {
                    snprintf(sbuf, sizeof(sbuf), "#%u", (unsigned)w->fast_src);
                    sname = sbuf;
                }
                fprintf(out, "  fast edge: %s -> %s  w_eff=%d\n", sname, name, (int)w->fast_weff);
                /* Instance detail when last explain used a conjunction key
                 * (folded into fast%; caller may also check instance_last_hit). */
            } else {
                fprintf(out, "  fast edge: (none into winner)\n");
            }
            if (c->hub != 0.0) {
                fprintf(out, "  hub: charge=%.1f (emergency dial pathway)\n", c->hub);
            } else {
                fprintf(out, "  hub: (closed or not dialed)\n");
            }
            if (un > 5.0 || un < -5.0) {
                fprintf(out, "  NOTE: unexplained |%.1f%%| > 5%% (organ interaction / superpos)\n",
                        un);
            }
        }
    }
}

void fluency_mood_print(const FluencyModel *m, FILE *out,
                        const char *(*word_fn)(void *ud, flu_tok_t id), void *ud) {
    if (!m || !out) {
        return;
    }
    fprintf(out, "mood: on=%u k=%u half_life=%u scale=%d\n", (unsigned)m->mood_on,
            (unsigned)m->mood_k, m->mood_half_life, (int)m->mood_scale);
    if (!m->mood_aff || m->mood_k == 0u) {
        fprintf(out, "  (no affinities built — train + fluency_ctx_organs_build)\n");
        return;
    }
    fprintf(out, "  registers:");
    for (uint8_t k = 0; k < m->mood_k; ++k) {
        fprintf(out, " r%u=%.3f", (unsigned)k, (double)m->mood_reg[k]);
    }
    fprintf(out, "\n");
    for (uint8_t k = 0; k < m->mood_k; ++k) {
        /* Top-3 affinity tokens for this axis. */
        int ti[3] = {-1, -1, -1};
        float tv[3] = {-1.0f, -1.0f, -1.0f};
        for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
            if (!m->seen[b]) {
                continue;
            }
            float a = m->mood_aff[(size_t)b * (size_t)m->mood_k + k];
            for (int i = 0; i < 3; ++i) {
                if (a > tv[i]) {
                    for (int j = 2; j > i; --j) {
                        ti[j] = ti[j - 1];
                        tv[j] = tv[j - 1];
                    }
                    ti[i] = (int)b;
                    tv[i] = a;
                    break;
                }
            }
        }
        fprintf(out, "  axis%u top:", (unsigned)k);
        for (int i = 0; i < 3; ++i) {
            if (ti[i] < 0) {
                break;
            }
            const char *nm = word_fn ? word_fn(ud, (flu_tok_t)ti[i]) : NULL;
            char buf[16];
            if (!nm) {
                snprintf(buf, sizeof(buf), "#%u", (unsigned)ti[i]);
                nm = buf;
            }
            fprintf(out, " %s(%.2f)", nm, (double)tv[i]);
        }
        fprintf(out, "\n");
    }
}

void fluency_gain_print(const FluencyModel *m, flu_tok_t tok, FILE *out,
                        const char *(*word_fn)(void *ud, flu_tok_t id), void *ud) {
    if (!m || !out) {
        return;
    }
    const char *nm = word_fn ? word_fn(ud, tok) : NULL;
    char buf[16];
    if (!nm) {
        snprintf(buf, sizeof(buf), "#%u", (unsigned)tok);
        nm = buf;
    }
    fprintf(out, "gain: on=%u window=%u scale=%d tok='%s'\n", (unsigned)m->gain_on,
            (unsigned)m->gain_window, (int)m->gain_scale, nm);
    if (!m->gain_cooc || !m->gain_vidx || m->gain_topk == 0u) {
        fprintf(out, "  (no cooc table — fluency_ctx_organs_build)\n");
        return;
    }
    if (tok >= FLUENCY_VOCAB || m->gain_vidx[tok] == 0xFFFFu) {
        fprintf(out, "  (token not in cooc topk)\n");
        return;
    }
    uint16_t ci = m->gain_vidx[tok];
    /* Top-10 partners. */
    uint32_t best_j[10], best_n[10];
    for (int i = 0; i < 10; ++i) {
        best_j[i] = 0;
        best_n[i] = 0;
    }
    for (uint32_t j = 0; j < m->gain_topk; ++j) {
        uint32_t c = m->gain_cooc[(size_t)ci * m->gain_topk + j];
        for (int i = 0; i < 10; ++i) {
            if (c > best_n[i]) {
                for (int k = 9; k > i; --k) {
                    best_n[k] = best_n[k - 1];
                    best_j[k] = best_j[k - 1];
                }
                best_n[i] = c;
                best_j[i] = j;
                break;
            }
        }
    }
    fprintf(out, "  top cooc partners:\n");
    for (int i = 0; i < 10; ++i) {
        if (best_n[i] == 0u) {
            break;
        }
        flu_tok_t t = m->gain_tok[best_j[i]];
        const char *tn = word_fn ? word_fn(ud, t) : NULL;
        char tbuf[16];
        if (!tn) {
            snprintf(tbuf, sizeof(tbuf), "#%u", (unsigned)t);
            tn = tbuf;
        }
        fprintf(out, "    %s %u\n", tn, best_n[i]);
    }
}

/* ---- Word-level front-end -------------------------------------------------- */

void fluency_words_init(FluencyWords *w) {
    if (w) {
        w->count = 0u;
    }
}

uint32_t fluency_words_id(FluencyWords *w, const char *word, int create) {
    if (!w || !word || word[0] == '\0') {
        return FLUENCY_VOCAB;
    }
    for (uint32_t i = 0; i < w->count; ++i) {
        if (strcmp(w->word[i], word) == 0) {
            if (create) {
                w->freq[i]++; /* seen whole once more -> more lexicalized */
            }
            return i;
        }
    }
    if (!create || w->count >= FLUENCY_VOCAB) {
        return FLUENCY_VOCAB;
    }
    uint32_t id = w->count++;
    size_t n = strlen(word);
    if (n > FLUENCY_WORD_LEN) {
        n = FLUENCY_WORD_LEN;
    }
    memcpy(w->word[id], word, n);
    w->word[id][n] = '\0';
    w->freq[id] = 1u;
    return id;
}

int fluency_word_is_lexicalized(const FluencyWords *w, const FluencySubword *s, const char *word,
                                uint32_t min_freq) {
    if (!w || !s || !word) {
        return 0;
    }
    /* (a) seen whole often enough to be memorable. */
    uint32_t id = FLUENCY_VOCAB;
    for (uint32_t i = 0; i < w->count; ++i) {
        if (strcmp(w->word[i], word) == 0) {
            id = i;
            break;
        }
    }
    if (id >= FLUENCY_VOCAB || w->freq[id] < min_freq) {
        return 0;
    }
    /* (b) BPE collapsed the whole word into ONE dedicated multi-char unit. That
     * merge only happens when the word recurs whole often enough to beat its
     * parts, so the single whole-word unit IS the memorized lexical entry. */
    flu_tok_t parts[FLU_BPE_MAX_WLEN];
    size_t np = fluency_bpe_encode_word(s, word, parts, sizeof(parts));
    if (np != 1u || strlen(s->sym[parts[0]]) <= 1u) {
        return 0; /* not a single multi-char unit -> compositional / atomic */
    }
    /* (c) It must HAVE a compositional alternative to override: the string splits
     * into >=2 OTHER known whole words (hotdog = hot + dog). An atomic word like
     * "hot" has a whole-word unit but no sub-word spelling, so there is nothing
     * to suppress -- it is a lexical entry, not a lexicalized *compound*. All
     * three signals are counts; the override is learned, not hand-wired. */
    size_t L = strlen(word);
    for (size_t cut = 1u; cut < L; ++cut) {
        char a[FLUENCY_WORD_LEN + 1u], b[FLUENCY_WORD_LEN + 1u];
        if (cut > FLUENCY_WORD_LEN || L - cut > FLUENCY_WORD_LEN) {
            continue;
        }
        memcpy(a, word, cut);
        a[cut] = '\0';
        memcpy(b, word + cut, L - cut);
        b[L - cut] = '\0';
        if (fluency_words_id((FluencyWords *)w, a, 0) < FLUENCY_VOCAB &&
            fluency_words_id((FluencyWords *)w, b, 0) < FLUENCY_VOCAB) {
            return 1;
        }
    }
    return 0;
}

const char *fluency_words_word(const FluencyWords *w, flu_tok_t id) {
    if (!w || id >= w->count) {
        return "";
    }
    return w->word[id];
}

static int fluency_word_char(int c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '\'';
}

size_t fluency_words_encode(FluencyWords *w, const char *text, flu_tok_t *out, size_t cap,
                            int create) {
    if (!w || !text || !out) {
        return 0u;
    }
    size_t n = 0u;
    char tok[FLUENCY_WORD_LEN + 1u];
    size_t tl = 0u;
    for (const char *p = text;; ++p) {
        int c = (unsigned char)*p;
        int lc = (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
        if (fluency_word_char(lc)) {
            if (tl < FLUENCY_WORD_LEN) {
                tok[tl++] = (char)lc;
            }
        } else {
            if (tl > 0u) {
                tok[tl] = '\0';
                uint32_t id = fluency_words_id(w, tok, create);
                if (id < FLUENCY_VOCAB && n < cap) {
                    out[n++] = (flu_tok_t)id;
                }
                tl = 0u;
            }
            if (c == '\0') {
                break;
            }
        }
    }
    return n;
}

/* ---- F4.1: subword (BPE) tokenization -------------------------------------- */

void fluency_bpe_init(FluencySubword *s) {
    if (!s) {
        return;
    }
    s->sym_count = 0u;
    s->merge_count = 0u;
    s->space_sym = FLUENCY_VOCAB;
    for (uint32_t b = 0; b < 256u; ++b) {
        s->byte_sym[b] = -1;
    }
}

static uint32_t fluency_bpe_base_sym(FluencySubword *s, uint8_t byte) {
    if (s->byte_sym[byte] >= 0) {
        return (uint32_t)s->byte_sym[byte];
    }
    if (s->sym_count >= FLUENCY_VOCAB) {
        return FLUENCY_VOCAB;
    }
    uint32_t id = s->sym_count++;
    s->sym[id][0] = (char)byte;
    s->sym[id][1] = '\0';
    s->byte_sym[byte] = (int16_t)id;
    return id;
}

const char *fluency_bpe_sym(const FluencySubword *s, flu_tok_t id) {
    if (!s || id >= s->sym_count) {
        return "";
    }
    return s->sym[id];
}

void fluency_bpe_learn(FluencySubword *s, const char *text, uint32_t target_vocab) {
    if (!s || !text) {
        return;
    }
    if (target_vocab > FLUENCY_VOCAB) {
        target_vocab = FLUENCY_VOCAB;
    }
    fluency_bpe_init(s);
    s->space_sym = fluency_bpe_base_sym(s, ' '); /* reserve boundary symbol */

    /* Collect distinct words as symbol-id sequences, with frequencies. */
    static uint16_t seq[FLU_BPE_MAX_WORDS][FLU_BPE_MAX_WLEN];
    static uint32_t len[FLU_BPE_MAX_WORDS];
    static uint32_t freq[FLU_BPE_MAX_WORDS];
    static char wtext[FLU_BPE_MAX_WORDS][FLU_BPE_MAX_WLEN + 1u];
    uint32_t nwords = 0u;

    char tok[FLU_BPE_MAX_WLEN + 1u];
    size_t tl = 0u;
    for (const char *p = text;; ++p) {
        int c = (unsigned char)*p;
        int lc = (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
        if (fluency_word_char(lc)) {
            if (tl < FLU_BPE_MAX_WLEN) {
                tok[tl++] = (char)lc;
            }
        } else {
            if (tl > 0u) {
                tok[tl] = '\0';
                uint32_t wi = nwords;
                for (uint32_t i = 0; i < nwords; ++i) {
                    if (strcmp(wtext[i], tok) == 0) {
                        wi = i;
                        break;
                    }
                }
                if (wi == nwords && nwords < FLU_BPE_MAX_WORDS) {
                    memcpy(wtext[nwords], tok, tl + 1u);
                    len[nwords] = (uint32_t)tl;
                    for (uint32_t i = 0; i < tl; ++i) {
                        seq[nwords][i] = (uint16_t)fluency_bpe_base_sym(s, (uint8_t)tok[i]);
                    }
                    freq[nwords] = 0u;
                    nwords++;
                }
                if (wi < nwords) {
                    freq[wi]++;
                }
                tl = 0u;
            }
            if (c == '\0') {
                break;
            }
        }
    }

    /* Greedy BPE: merge the most frequent adjacent pair until target reached.
     * Pair key = a*FLUENCY_VOCAB + b covers the full symbol range (post-widening
     * ids exceed a byte), counted in an open-addressing hash so there is no fixed
     * (a<<8|b) overflow and no V^2 array to clear. */
#define FLU_PC_SLOTS 65536u
    while (s->sym_count < target_vocab) {
        uint32_t best = 0u;
        uint16_t ba = 0, bb = 0;
        static uint32_t pk[FLU_PC_SLOTS]; /* key+1, 0 = empty */
        static uint32_t pv[FLU_PC_SLOTS];
        memset(pk, 0, sizeof(pk));
        memset(pv, 0, sizeof(pv));
        for (uint32_t wi = 0; wi < nwords; ++wi) {
            for (uint32_t i = 0; i + 1u < len[wi]; ++i) {
                /* uint64 pair key — V=65k overflows uint32 (V*V). */
                uint64_t key =
                    (uint64_t)seq[wi][i] * (uint64_t)FLUENCY_VOCAB + (uint64_t)seq[wi][i + 1u];
                uint32_t h = (uint32_t)((key * 2654435761ull) & (uint64_t)(FLU_PC_SLOTS - 1u));
                while (pk[h] != 0u && pk[h] != (uint32_t)(key + 1u)) {
                    /* ponytail: 32-bit pk slot can't hold full key+1 for V=65k;
                     * use truncated key for hash only — collision-safe enough for BPE learn. */
                    h = (h + 1u) & (FLU_PC_SLOTS - 1u);
                }
                pk[h] = (uint32_t)(key + 1u); /* truncated; fine for small V */
                pv[h] += freq[wi] ? freq[wi] : 1u;
            }
        }
        for (uint32_t h = 0; h < FLU_PC_SLOTS; ++h) {
            if (pk[h] != 0u && pv[h] > best) {
                best = pv[h];
                uint32_t key = pk[h] - 1u;
                ba = (uint16_t)(key / FLUENCY_VOCAB);
                bb = (uint16_t)(key % FLUENCY_VOCAB);
            }
        }
        if (best < 2u || s->sym_count >= FLUENCY_VOCAB) {
            break; /* no repeated pair left */
        }
        uint32_t nid = s->sym_count++;
        char merged[2u * (FLU_SYM_LEN + 1u)];
        snprintf(merged, sizeof(merged), "%s%s", s->sym[ba], s->sym[bb]);
        merged[FLU_SYM_LEN] = '\0';
        memcpy(s->sym[nid], merged, FLU_SYM_LEN + 1u);
        s->merge_a[s->merge_count] = ba;
        s->merge_b[s->merge_count] = bb;
        s->merge_new[s->merge_count] = (uint16_t)nid;
        s->merge_count++;
        /* Apply the merge in every word. */
        for (uint32_t wi = 0; wi < nwords; ++wi) {
            uint32_t w = 0u;
            for (uint32_t i = 0; i < len[wi];) {
                if (i + 1u < len[wi] && seq[wi][i] == ba && seq[wi][i + 1u] == bb) {
                    seq[wi][w++] = (uint16_t)nid;
                    i += 2u;
                } else {
                    seq[wi][w++] = seq[wi][i];
                    i += 1u;
                }
            }
            len[wi] = w;
        }
    }
}

size_t fluency_bpe_encode_word(const FluencySubword *s, const char *word, flu_tok_t *out,
                               size_t cap) {
    if (!s || !word || !out) {
        return 0u;
    }
    uint16_t buf[FLU_BPE_MAX_WLEN];
    uint32_t n = 0u;
    for (const char *p = word; *p && n < FLU_BPE_MAX_WLEN; ++p) {
        int c = (unsigned char)*p;
        int lc = (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
        if (s->byte_sym[lc] < 0) {
            continue; /* unknown char: skip */
        }
        buf[n++] = (uint16_t)s->byte_sym[lc];
    }
    /* Apply merges in learned order, repeatedly, until none fire. */
    for (uint32_t mi = 0; mi < s->merge_count; ++mi) {
        uint16_t a = s->merge_a[mi], b = s->merge_b[mi], nw = s->merge_new[mi];
        uint32_t w = 0u;
        for (uint32_t i = 0; i < n;) {
            if (i + 1u < n && buf[i] == a && buf[i + 1u] == b) {
                buf[w++] = nw;
                i += 2u;
            } else {
                buf[w++] = buf[i];
                i += 1u;
            }
        }
        n = w;
    }
    size_t k = 0u;
    for (uint32_t i = 0; i < n && k < cap; ++i) {
        out[k++] = (flu_tok_t)buf[i];
    }
    return k;
}

size_t fluency_bpe_encode_text(const FluencySubword *s, const char *text, flu_tok_t *out,
                               size_t cap) {
    if (!s || !text || !out) {
        return 0u;
    }
    size_t k = 0u;
    char tok[FLU_BPE_MAX_WLEN + 1u];
    size_t tl = 0u;
    int first = 1;
    for (const char *p = text;; ++p) {
        int c = (unsigned char)*p;
        int lc = (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
        if (fluency_word_char(lc)) {
            if (tl < FLU_BPE_MAX_WLEN) {
                tok[tl++] = (char)lc;
            }
        } else {
            if (tl > 0u) {
                tok[tl] = '\0';
                if (!first && s->space_sym < FLUENCY_VOCAB && k < cap) {
                    out[k++] = (flu_tok_t)s->space_sym; /* boundary between words */
                }
                first = 0;
                k += fluency_bpe_encode_word(s, tok, out + k, cap - k);
                tl = 0u;
            }
            if (c == '\0') {
                break;
            }
        }
    }
    return k;
}

/* ---- F4.3: morpheme-aware segmentation (branching entropy / Harris) --------- */

static uint32_t fluency_morph_hash(const char *bytes, uint32_t n) {
    uint32_t h = 2166136261u; /* FNV-1a */
    for (uint32_t i = 0; i < n; ++i) {
        h = (h ^ (uint8_t)bytes[i]) * 16777619u;
    }
    return h ? h : 1u;
}

void fluency_morph_init(FluencyMorph *mo) {
    if (mo) {
        memset(mo, 0, sizeof(*mo));
    }
}

void fluency_morph_free(FluencyMorph *mo) {
    if (!mo) {
        return;
    }
    free(mo->pref_hash);
    free(mo->follow);
    mo->pref_hash = NULL;
    mo->follow = NULL;
    mo->cap = 0;
    mo->count = 0;
}

/* Record that `next` followed prefix bytes[0..n). */
static void fluency_morph_add(FluencyMorph *mo, const char *bytes, uint32_t n, uint8_t next) {
    uint32_t key = fluency_morph_hash(bytes, n);
    size_t h = (size_t)((key * 2654435761u) & (mo->cap - 1u));
    while (mo->pref_hash[h] != 0u && mo->pref_hash[h] != key) {
        h = (h + 1u) & (mo->cap - 1u);
    }
    if (mo->pref_hash[h] == 0u) {
        mo->pref_hash[h] = key;
        mo->count++;
    }
    mo->follow[h][next >> 3] |= (uint8_t)(1u << (next & 7u));
}

static uint32_t fluency_morph_branch(const FluencyMorph *mo, const char *bytes, uint32_t n) {
    uint32_t key = fluency_morph_hash(bytes, n);
    size_t h = (size_t)((key * 2654435761u) & (mo->cap - 1u));
    while (mo->pref_hash[h] != 0u) {
        if (mo->pref_hash[h] == key) {
            uint32_t pop = 0;
            for (int b = 0; b < 32; ++b) {
                uint8_t x = mo->follow[h][b];
                while (x) {
                    pop += (x & 1u);
                    x >>= 1;
                }
            }
            return pop;
        }
        h = (h + 1u) & (mo->cap - 1u);
    }
    return 0u;
}

int fluency_morph_learn(FluencyMorph *mo, const char *text, uint32_t depth) {
    if (!mo || !text || depth == 0u) {
        return -1;
    }
    fluency_morph_free(mo);
    mo->cap = 1u << 16;
    mo->depth = depth;
    mo->pref_hash = calloc(mo->cap, sizeof(uint32_t));
    mo->follow = calloc(mo->cap, sizeof(*mo->follow));
    if (!mo->pref_hash || !mo->follow) {
        fluency_morph_free(mo);
        return -1;
    }

    char word[FLU_BPE_MAX_WLEN + 1u];
    uint32_t wl = 0u;
    for (const char *p = text;; ++p) {
        int c = (unsigned char)*p;
        int lc = (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
        if (fluency_word_char(lc)) {
            if (wl < FLU_BPE_MAX_WLEN) {
                word[wl++] = (char)lc;
            }
        } else {
            /* For each position, record the following char for every suffix
             * prefix up to `depth` chars back -- these are the branching stats. */
            for (uint32_t i = 0; i + 1u <= wl; ++i) {
                if (i == wl) {
                    break;
                }
                uint8_t next = (uint8_t)word[i];
                uint32_t dmax = i < depth ? i : depth;
                for (uint32_t d = 1u; d <= dmax; ++d) {
                    fluency_morph_add(mo, word + (i - d), d, next);
                }
                /* Grow if load high. */
                if ((mo->count + 1u) * 10u >= mo->cap * 7u) {
                    size_t ncap = mo->cap * 2u;
                    uint32_t *nk = calloc(ncap, sizeof(uint32_t));
                    uint8_t (*nf)[32] = calloc(ncap, sizeof(*nf));
                    if (!nk || !nf) {
                        free(nk);
                        free(nf);
                        return -1;
                    }
                    for (size_t t = 0; t < mo->cap; ++t) {
                        if (mo->pref_hash[t] == 0u) {
                            continue;
                        }
                        size_t h = (size_t)((mo->pref_hash[t] * 2654435761u) & (ncap - 1u));
                        while (nk[h] != 0u) {
                            h = (h + 1u) & (ncap - 1u);
                        }
                        nk[h] = mo->pref_hash[t];
                        memcpy(nf[h], mo->follow[t], 32);
                    }
                    free(mo->pref_hash);
                    free(mo->follow);
                    mo->pref_hash = nk;
                    mo->follow = nf;
                    mo->cap = ncap;
                }
            }
            wl = 0u;
            if (c == '\0') {
                break;
            }
        }
    }
    return 0;
}

uint32_t fluency_morph_segment(const FluencyMorph *mo, const char *word, uint32_t jump,
                               uint32_t *cuts, uint32_t max_cuts) {
    if (!mo || !word || !cuts || max_cuts == 0u) {
        return 0u;
    }
    uint32_t wl = (uint32_t)strlen(word);
    uint32_t nc = 0u;
    uint32_t prev = 0u;
    /* Walk left to right; at each interior position measure the branching factor
     * of the current suffix (up to depth). A boundary is where branching rises
     * sharply relative to the previous position (a complete morpheme opens up
     * many possible continuations). */
    for (uint32_t i = 1u; i < wl && nc < max_cuts; ++i) {
        uint32_t d = i < mo->depth ? i : mo->depth;
        uint32_t br = fluency_morph_branch(mo, word + (i - d), d);
        if (i >= 2u && br > prev + jump) {
            cuts[nc++] = i;
        }
        prev = br;
    }
    return nc;
}

/* ---- F5: firing-path (reservoir) projection -------------------------------- */

#define FLU_RES_REL ((uint16_t)(NERVA_REL_CUSTOM_BASE + 210u))

static uint32_t fluency_res_node(FluencyReservoir *r, flu_tok_t t) {
    if (r->node[t] != NERVA_INVALID_ID) {
        return r->node[t];
    }
    char name[16];
    snprintf(name, sizeof(name), "RES:%03x", (unsigned)t);
    uint32_t id = nerva_get_or_create_node(r->e, name);
    if (id == NERVA_INVALID_ID) {
        return NERVA_INVALID_ID;
    }
    /* Threshold BELOW a main-path transition probability but ABOVE a hub's tiny
     * per-successor probability: a dominant transition (P > 1/4) fires and the
     * wave travels; a high-fanout hub (each successor << 1/4) is absorbed. This
     * is the echo-state operating point paired with the row-stochastic weights. */
    r->e->nodes[id].theta_fire = (nerva_q8_8_t)(NERVA_Q8_8_ONE / 4);
    r->e->nodes[id].v_rest = 0;
    r->e->nodes[id].v_reset = 0;
    r->node[t] = id;
    if (!r->present[t]) {
        r->present[t] = 1u;
        r->vcount++;
    }
    return id;
}

int fluency_reservoir_build(FluencyReservoir *r, const flu_tok_t *text, size_t len) {
    if (!r || !text) {
        return -1;
    }
    memset(r, 0, sizeof(*r));
    for (uint32_t t = 0; t < FLUENCY_VOCAB; ++t) {
        r->node[t] = NERVA_INVALID_ID;
    }
    NervaConfig cfg = nerva_config_default();
    cfg.weight_max_q8_8 = INT16_MAX;
    cfg.leak_shift = 2u; /* strong leak: the wave decays, does not flood */
    cfg.max_nodes = FLUENCY_VOCAB + 16u;
    cfg.max_edges = (uint32_t)(FLUENCY_VOCAB * 64u);
    cfg.max_names = FLUENCY_VOCAB + 16u;
    r->e = calloc(1, sizeof(NervaEngine));
    r->scratch_a = calloc(FLUENCY_VOCAB, sizeof(double));
    r->scratch_b = calloc(FLUENCY_VOCAB, sizeof(double));
    if (!r->e || !r->scratch_a || !r->scratch_b || nerva_engine_init(r->e, cfg) != 0) {
        fluency_reservoir_free(r);
        return -1;
    }

    /* Probability-weighted transition edges: weight(a->b) = P(b|a) * ONE. Because
     * these sum to ONE over each source's successors, the outgoing weights are
     * ROW-STOCHASTIC -- each fired node distributes exactly ONE unit of charge to
     * its successors, so activation is conserved per hop (spectral radius ~1, the
     * edge of chaos). This is the echo-state tuning, and it falls out of using
     * transition probabilities: a hub like "the" splits its ONE across a hundred
     * successors (each ~0.01), so it damps rather than floods. The BREAK sentinel
     * (FLUENCY_VOCAB-1) still severs structured chains; ordinary tokenizers never
     * emit it. */
    const flu_tok_t BREAK = (flu_tok_t)(FLUENCY_VOCAB - 1u);

    /* First pass: transition counts (open-addressing hash) + per-source totals. */
    size_t ccap = 1u << 16;
    uint64_t *ck = calloc(ccap, sizeof(uint64_t)); /* (a*V+b)+1 */
    uint32_t *cv = calloc(ccap, sizeof(uint32_t));
    uint32_t *outc = calloc(FLUENCY_VOCAB, sizeof(uint32_t));
    if (!ck || !cv || !outc) {
        free(ck); free(cv); free(outc);
        fluency_reservoir_free(r);
        return -1;
    }
    size_t cn = 0;
    for (size_t i = 0; i + 1u < len; ++i) {
        flu_tok_t a = text[i], b = text[i + 1u];
        if (a == BREAK || b == BREAK || a >= FLUENCY_VOCAB || b >= FLUENCY_VOCAB) {
            continue;
        }
        (void)fluency_res_node(r, a);
        (void)fluency_res_node(r, b);
        uint64_t key = (uint64_t)a * FLUENCY_VOCAB + b;
        if ((cn + 1u) * 10u >= ccap * 7u) {
            size_t ncap = ccap * 2u;
            uint64_t *nk = calloc(ncap, sizeof(uint64_t));
            uint32_t *nv = calloc(ncap, sizeof(uint32_t));
            if (!nk || !nv) { free(nk); free(nv); free(ck); free(cv); free(outc);
                fluency_reservoir_free(r); return -1; }
            for (size_t t = 0; t < ccap; ++t) {
                if (!ck[t]) continue;
                size_t h = (size_t)((ck[t] * 2654435761u) & (ncap - 1u));
                while (nk[h]) h = (h + 1u) & (ncap - 1u);
                nk[h] = ck[t]; nv[h] = cv[t];
            }
            free(ck); free(cv); ck = nk; cv = nv; ccap = ncap;
        }
        size_t h = (size_t)(((key + 1u) * 2654435761u) & (ccap - 1u));
        while (ck[h] && ck[h] != key + 1u) h = (h + 1u) & (ccap - 1u);
        if (!ck[h]) { ck[h] = key + 1u; cn++; }
        cv[h]++;
        outc[a]++;
    }

    /* Second pass: create one weighted edge per distinct transition. */
    for (size_t t = 0; t < ccap; ++t) {
        if (!ck[t]) {
            continue;
        }
        uint64_t key = ck[t] - 1u;
        flu_tok_t a = (flu_tok_t)(key / FLUENCY_VOCAB), b = (flu_tok_t)(key % FLUENCY_VOCAB);
        uint32_t sa = r->node[a], sb = r->node[b];
        if (sa == NERVA_INVALID_ID || sb == NERVA_INVALID_ID || outc[a] == 0u) {
            continue;
        }
        int32_t wq = (int32_t)(((int64_t)cv[t] * NERVA_Q8_8_ONE) / outc[a]);
        if (wq < 1) {
            wq = 1; /* a present transition always delivers something */
        }
        uint32_t ed = nerva_graph_create_edge(r->e, sa, sb, FLU_RES_REL);
        if (ed != NERVA_INVALID_ID) {
            r->e->edges[ed].weight = (nerva_q8_8_t)wq;
        }
    }
    free(ck);
    free(cv);
    free(outc);
    nerva_graph_rebuild_adjacency(r->e);
    return 0;
}

void fluency_reservoir_free(FluencyReservoir *r) {
    if (!r) {
        return;
    }
    if (r->e) {
        nerva_engine_free(r->e);
        free(r->e);
        r->e = NULL;
    }
    free(r->scratch_a);
    free(r->scratch_b);
    free(r->cache_off);
    free(r->cache_tok);
    free(r->cache_w);
    free(r->cache_norm);
    r->scratch_a = NULL;
    r->scratch_b = NULL;
    r->cache_off = NULL;
    r->cache_tok = NULL;
    r->cache_w = NULL;
    r->cache_norm = NULL;
    r->cached = 0;
}

void fluency_firing_signature(FluencyReservoir *r, flu_tok_t seed, uint32_t hops, double *sig) {
    for (uint32_t t = 0; t < FLUENCY_VOCAB; ++t) {
        sig[t] = 0.0;
    }
    if (!r || !r->e || seed >= FLUENCY_VOCAB || r->node[seed] == NERVA_INVALID_ID) {
        return;
    }
    for (uint32_t i = 0; i < r->e->node_count; ++i) {
        r->e->nodes[i].v = r->e->nodes[i].v_rest;
    }
    r->e->event_count = 0;
    r->e->active_count = 0;
    nerva_trace_clear(r->e);

    nerva_tick_t t0 = r->e->tick;
    nerva_activate_node(r->e, r->node[seed], NERVA_Q8_8_ONE);
    nerva_tick_n(r->e, hops);

    for (uint32_t t = 0; t < FLUENCY_VOCAB; ++t) {
        if (t == seed || r->node[t] == NERVA_INVALID_ID) {
            continue;
        }
        NervaNode *n = &r->e->nodes[r->node[t]];
        if (n->activation_count > 0u && n->last_fired_tick > t0 &&
            n->last_fired_tick <= t0 + (nerva_tick_t)hops) {
            /* Weight by DEPTH reached: transitive similarity lives in shared deep
             * reachability (two tokens that funnel to the same place through
             * disjoint short paths), so a late arrival counts more than a
             * chain-specific immediate successor. */
            uint32_t age = (uint32_t)(n->last_fired_tick - t0);
            sig[t] = (double)(age * age);
        }
    }
}

static double fluency_sig_cosine(const double *a, const double *b) {
    double dot = 0, na = 0, nb = 0;
    for (uint32_t i = 0; i < FLUENCY_VOCAB; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) {
        return 0.0;
    }
    return dot / (sqrt(na) * sqrt(nb));
}

int fluency_reservoir_cache(FluencyReservoir *r, uint32_t hops) {
    if (!r || !r->e) {
        return -1;
    }
    if (r->cached && r->cache_hops == hops) {
        return 0;
    }
    free(r->cache_off);
    free(r->cache_tok);
    free(r->cache_w);
    free(r->cache_norm);
    r->cache_off = calloc((size_t)FLUENCY_VOCAB + 1u, sizeof(size_t));
    r->cache_norm = calloc(FLUENCY_VOCAB, sizeof(double));
    size_t cap = 4096, nn = 0;
    r->cache_tok = malloc(cap * sizeof(flu_tok_t));
    r->cache_w = malloc(cap * sizeof(double));
    if (!r->cache_off || !r->cache_norm || !r->cache_tok || !r->cache_w) {
        r->cached = 0;
        return -1;
    }
    for (uint32_t t = 0; t < FLUENCY_VOCAB; ++t) {
        r->cache_off[t] = nn;
        if (r->node[t] == NERVA_INVALID_ID) {
            continue;
        }
        fluency_firing_signature(r, (flu_tok_t)t, hops, r->scratch_a);
        double nrm = 0.0;
        for (uint32_t j = 0; j < FLUENCY_VOCAB; ++j) {
            if (r->scratch_a[j] <= 0.0) {
                continue;
            }
            if (nn + 1u >= cap) {
                cap *= 2u;
                flu_tok_t *nt = realloc(r->cache_tok, cap * sizeof(flu_tok_t));
                double *nw = realloc(r->cache_w, cap * sizeof(double));
                if (!nt || !nw) {
                    free(nt ? nt : r->cache_tok);
                    free(nw ? nw : r->cache_w);
                    r->cache_tok = NULL;
                    r->cache_w = NULL;
                    r->cached = 0;
                    return -1;
                }
                r->cache_tok = nt;
                r->cache_w = nw;
            }
            r->cache_tok[nn] = (flu_tok_t)j;
            r->cache_w[nn] = r->scratch_a[j];
            nn++;
            nrm += r->scratch_a[j] * r->scratch_a[j];
        }
        r->cache_norm[t] = sqrt(nrm);
    }
    r->cache_off[FLUENCY_VOCAB] = nn;
    r->cached = 1;
    r->cache_hops = hops;
    return 0;
}

/* Cosine of two cached signatures via a scatter/gather over the dense scratch. */
static double fluency_cache_cos(FluencyReservoir *r, flu_tok_t a, flu_tok_t b) {
    if (r->cache_norm[a] < 1e-12 || r->cache_norm[b] < 1e-12) {
        return 0.0;
    }
    for (size_t e = r->cache_off[a]; e < r->cache_off[a + 1u]; ++e) {
        r->scratch_a[r->cache_tok[e]] = r->cache_w[e];
    }
    double dot = 0.0;
    for (size_t e = r->cache_off[b]; e < r->cache_off[b + 1u]; ++e) {
        dot += r->scratch_a[r->cache_tok[e]] * r->cache_w[e];
    }
    for (size_t e = r->cache_off[a]; e < r->cache_off[a + 1u]; ++e) {
        r->scratch_a[r->cache_tok[e]] = 0.0; /* leave scratch clean */
    }
    return dot / (r->cache_norm[a] * r->cache_norm[b]);
}

double fluency_firing_cosine(FluencyReservoir *r, flu_tok_t a, flu_tok_t b, uint32_t hops) {
    if (!r || a >= FLUENCY_VOCAB || b >= FLUENCY_VOCAB) {
        return 0.0;
    }
    if (fluency_reservoir_cache(r, hops) == 0) {
        return fluency_cache_cos(r, a, b);
    }
    /* Fallback (cache alloc failed): compute on demand. */
    fluency_firing_signature(r, a, hops, r->scratch_a);
    fluency_firing_signature(r, b, hops, r->scratch_b);
    return fluency_sig_cosine(r->scratch_a, r->scratch_b);
}

uint32_t fluency_firing_neighbors(FluencyReservoir *r, flu_tok_t tok, uint32_t hops,
                                  flu_tok_t *out, uint32_t k) {
    if (!r || !out || k == 0u || tok >= FLUENCY_VOCAB || r->node[tok] == NERVA_INVALID_ID) {
        return 0u;
    }
    if (fluency_reservoir_cache(r, hops) != 0) {
        return 0u;
    }
    /* Scatter the query's own signature: entries mark its forward cone (skip
     * those -- role-peers are outside the cone), and serve as the dot vector. */
    for (size_t e = r->cache_off[tok]; e < r->cache_off[tok + 1u]; ++e) {
        r->scratch_a[r->cache_tok[e]] = r->cache_w[e];
    }
    uint8_t *used = calloc(FLUENCY_VOCAB, 1);
    uint32_t written = 0u;
    while (used && written < k) {
        double best = -2.0;
        int best_t = -1;
        for (uint32_t t = 0; t < FLUENCY_VOCAB; ++t) {
            if (t == tok || used[t] || r->node[t] == NERVA_INVALID_ID) {
                continue;
            }
            if (r->scratch_a[t] > 0.0) {
                continue; /* t is in the query's forward cone */
            }
            if (r->cache_norm[tok] < 1e-12 || r->cache_norm[t] < 1e-12) {
                continue;
            }
            double dot = 0.0;
            for (size_t e = r->cache_off[t]; e < r->cache_off[t + 1u]; ++e) {
                dot += r->scratch_a[r->cache_tok[e]] * r->cache_w[e];
            }
            double c = dot / (r->cache_norm[tok] * r->cache_norm[t]);
            if (c > best) {
                best = c;
                best_t = (int)t;
            }
        }
        if (best_t < 0) {
            break;
        }
        used[best_t] = 1u;
        out[written++] = (flu_tok_t)best_t;
    }
    for (size_t e = r->cache_off[tok]; e < r->cache_off[tok + 1u]; ++e) {
        r->scratch_a[r->cache_tok[e]] = 0.0; /* leave scratch clean */
    }
    free(used);
    return written;
}

/* ---- F2: co-occurrence embedding (no backprop, closed-form) ----------------
 *
 * Representation, not a meaning decision (per the fluency-pivot invariant): a
 * distributed token code derived purely from counts, so it is deterministic and
 * its provenance is the co-occurrence matrix. Two tokens that occur in similar
 * neighbourhoods land near each other -- the generalization across surface forms
 * that raw n-gram counts cannot give.
 *
 * Method (all by hand, no gradient):
 *   1. Symmetric windowed co-occurrence count C[i][j].
 *   2. Positive PMI  M[i][j] = max(0, log( C[i][j] * T / (row_i * col_j) )).
 *      (Levy & Goldberg: SGNS implicitly factorizes shifted PMI; PPMI is the
 *      k=1 closed form, no sampling and no backprop.)
 *   3. Top-`dim` eigenvectors of the symmetric M by power iteration with
 *      Gram-Schmidt deflation; embedding row = eigvec * sqrt(|eigval|).
 */

static void fluency_build_vocab_index(FluencyModel *m) {
    m->vdim = 0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (m->seen[b]) {
            m->vidx[b] = m->vdim;
            m->vbyte[m->vdim] = (flu_tok_t)b;
            m->vdim++;
        } else {
            m->vidx[b] = UINT32_MAX;
        }
    }
}

int fluency_build_embedding(FluencyModel *m, const flu_tok_t *text, size_t len,
                            uint32_t window, uint32_t dim) {
    if (!m || !text || window == 0u || dim == 0u) {
        return -1;
    }

    /* Seed vocab from this text if training has not populated it yet. */
    for (size_t i = 0; i < len; ++i) {
        if (!m->seen[text[i]]) {
            m->seen[text[i]] = 1u;
            m->vocab_count++;
        }
    }
    fluency_build_vocab_index(m);
    uint32_t V = m->vdim;
    if (V == 0u) {
        return -1;
    }
    if (dim > V) {
        dim = V;
    }

    /* Sparse co-occurrence: an open-addressing hash of (i*V + j) -> count. Real
     * text co-occurrence is sparse (most pairs never occur), so this is O(nnz),
     * not O(V^2) -- the dense matrix is what bounded the token cap, and it is
     * gone. The math is unchanged: symmetric windowed counts -> positive PMI ->
     * top-dim eigenvectors by power iteration (now sparse matvec). */
    size_t hcap = 1u << 18;
    uint64_t *hk = calloc(hcap, sizeof(uint64_t)); /* key+1; 0 = empty */
    double *hv = calloc(hcap, sizeof(double));
    if (!hk || !hv) {
        free(hk);
        free(hv);
        return -1;
    }
    size_t hn = 0;
    for (size_t i = 0; i < len; ++i) {
        uint32_t ci = m->vidx[text[i]];
        if (ci == UINT32_MAX) {
            continue;
        }
        size_t lo = i > window ? i - window : 0;
        size_t hi = i + window + 1u < len ? i + window + 1u : len;
        for (size_t j = lo; j < hi; ++j) {
            if (j == i) {
                continue;
            }
            uint32_t cj = m->vidx[text[j]];
            if (cj == UINT32_MAX) {
                continue;
            }
            uint64_t key = (uint64_t)ci * V + cj;
            if ((hn + 1u) * 10u >= hcap * 7u) { /* grow at 0.7 load */
                size_t ncap = hcap * 2u;
                uint64_t *nk = calloc(ncap, sizeof(uint64_t));
                double *nv = calloc(ncap, sizeof(double));
                if (!nk || !nv) {
                    free(nk);
                    free(nv);
                    free(hk);
                    free(hv);
                    return -1;
                }
                for (size_t t = 0; t < hcap; ++t) {
                    if (hk[t] == 0u) {
                        continue;
                    }
                    size_t h = (size_t)((hk[t] * 2654435761u) & (ncap - 1u));
                    while (nk[h] != 0u) {
                        h = (h + 1u) & (ncap - 1u);
                    }
                    nk[h] = hk[t];
                    nv[h] = hv[t];
                }
                free(hk);
                free(hv);
                hk = nk;
                hv = nv;
                hcap = ncap;
            }
            size_t h = (size_t)(((key + 1u) * 2654435761u) & (hcap - 1u));
            while (hk[h] != 0u && hk[h] != key + 1u) {
                h = (h + 1u) & (hcap - 1u);
            }
            if (hk[h] == 0u) {
                hk[h] = key + 1u;
                hn++;
            }
            hv[h] += 1.0;
        }
    }

    double *row = calloc(V, sizeof(double));
    if (!row) {
        free(hk);
        free(hv);
        return -1;
    }
    double total = 0.0;
    for (size_t t = 0; t < hcap; ++t) {
        if (hk[t] == 0u) {
            continue;
        }
        uint32_t i = (uint32_t)((hk[t] - 1u) / V);
        row[i] += hv[t];
        total += hv[t];
    }
    if (total <= 0.0) {
        free(hk);
        free(hv);
        free(row);
        return -1;
    }

    /* Compact to PPMI triplets (drop zero/negative PMI). */
    uint32_t *ti = malloc(hn * sizeof(uint32_t));
    uint32_t *tj = malloc(hn * sizeof(uint32_t));
    double *tv = malloc(hn * sizeof(double));
    if (!ti || !tj || !tv) {
        free(hk);
        free(hv);
        free(row);
        free(ti);
        free(tj);
        free(tv);
        return -1;
    }
    size_t nnz = 0;
    for (size_t t = 0; t < hcap; ++t) {
        if (hk[t] == 0u) {
            continue;
        }
        uint64_t key = hk[t] - 1u;
        uint32_t i = (uint32_t)(key / V), j = (uint32_t)(key % V);
        double c = hv[t];
        if (c > 0.0 && row[i] > 0.0 && row[j] > 0.0) {
            double pmi = log((c * total) / (row[i] * row[j]));
            if (pmi > 0.0) {
                ti[nnz] = i;
                tj[nnz] = j;
                tv[nnz] = pmi;
                nnz++;
            }
        }
    }
    free(hk);
    free(hv);
    free(row);

    free(m->embed);
    m->embed = calloc((size_t)V * dim, sizeof(double));
    double *evec = calloc((size_t)dim * V, sizeof(double));
    double *v = calloc(V, sizeof(double));
    double *w = calloc(V, sizeof(double));
    if (!m->embed || !evec || !v || !w) {
        free(m->embed);
        m->embed = NULL;
        free(evec);
        free(v);
        free(w);
        free(ti);
        free(tj);
        free(tv);
        return -1;
    }
    m->embed_dim = dim;

    uint32_t rng = 0x2545f491u; /* deterministic init */
    for (uint32_t d = 0; d < dim; ++d) {
        for (uint32_t i = 0; i < V; ++i) {
            rng = rng * 1664525u + 1013904223u;
            v[i] = ((double)(rng >> 8) / 16777216.0) - 0.5;
        }
        double eval = 0.0;
        for (uint32_t it = 0; it < 100u; ++it) {
            for (uint32_t i = 0; i < V; ++i) {
                w[i] = 0.0;
            }
            for (size_t t = 0; t < nnz; ++t) { /* sparse matvec w += M v */
                w[ti[t]] += tv[t] * v[tj[t]];
            }
            for (uint32_t p = 0; p < d; ++p) {
                const double *ep = &evec[(size_t)p * V];
                double dot = 0.0;
                for (uint32_t i = 0; i < V; ++i) {
                    dot += w[i] * ep[i];
                }
                for (uint32_t i = 0; i < V; ++i) {
                    w[i] -= dot * ep[i];
                }
            }
            double norm = 0.0;
            for (uint32_t i = 0; i < V; ++i) {
                norm += w[i] * w[i];
            }
            norm = sqrt(norm);
            if (norm < 1e-12) {
                break;
            }
            for (uint32_t i = 0; i < V; ++i) {
                v[i] = w[i] / norm;
            }
            eval = norm;
        }
        double *ed = &evec[(size_t)d * V];
        double scale = sqrt(eval > 0.0 ? eval : 0.0);
        for (uint32_t i = 0; i < V; ++i) {
            ed[i] = v[i];
            m->embed[(size_t)i * dim + d] = v[i] * scale;
        }
    }

    free(evec);
    free(v);
    free(w);
    free(ti);
    free(tj);
    free(tv);
    return 0;
}

double fluency_cosine(const FluencyModel *m, flu_tok_t a, flu_tok_t b) {
    if (!m || !m->embed || m->vidx[a] == UINT32_MAX || m->vidx[b] == UINT32_MAX) {
        return 0.0;
    }
    const double *va = &m->embed[(size_t)m->vidx[a] * m->embed_dim];
    const double *vb = &m->embed[(size_t)m->vidx[b] * m->embed_dim];
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t d = 0; d < m->embed_dim; ++d) {
        dot += va[d] * vb[d];
        na += va[d] * va[d];
        nb += vb[d] * vb[d];
    }
    if (na < 1e-18 || nb < 1e-18) {
        return 0.0;
    }
    return dot / (sqrt(na) * sqrt(nb));
}

/* Fire exactly one context node (the k bytes given) and add its integrated
 * continuation votes into acc[], scaled by `scale`. No-op if the node is unseen.
 * Each call is a genuine firing readout; the caller does the kernel weighting. */
static void fluency_accumulate_context(FluencyModel *m, const flu_tok_t *bytes, uint32_t k,
                                       double scale, double *acc) {
    char name[8u + 5u * FLUENCY_MAX_ORDER];
    fluency_ctx_name(name, sizeof(name), bytes, k);
    uint32_t id = fluency_name_lookup(m, name);
    if (id == NERVA_INVALID_ID) {
        return;
    }
    fluency_reset_charges(m->e);
    nerva_activate_node(m->e, id, NERVA_Q8_8_ONE);
    nerva_tick_n(m->e, 3u);
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (m->seen[b] && m->tok_node[b] != NERVA_INVALID_ID) {
            int32_t v = (int32_t)m->e->nodes[m->tok_node[b]].v;
            if (v > 0) {
                acc[b] += scale * (double)v;
            }
        }
    }
}

int fluency_predict_kernel(FluencyModel *m, const flu_tok_t *ctx, size_t ctx_len, double *dist,
                           double lambda, uint32_t smooth_k) {
    if (!m || !dist) {
        return -1;
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        dist[b] = 0.0;
    }
    int use_kernel = (m->embed != NULL && lambda > 0.0 && smooth_k > 0u);

    double acc[FLUENCY_VOCAB] = {0};    /* exact-context votes (this is F1) */
    double borrow[FLUENCY_VOCAB] = {0}; /* votes from embedding-similar contexts */
    flu_tok_t variant[FLUENCY_MAX_ORDER];

    for (uint32_t k = 1u; k <= m->order; ++k) {
        if ((size_t)k > ctx_len) {
            break;
        }
        const flu_tok_t *bytes = ctx + (ctx_len - k);
        fluency_accumulate_context(m, bytes, k, 1.0, acc);

        if (!use_kernel) {
            continue;
        }
        /* Similarity backoff: also fire contexts with the predecessor byte
         * replaced by its embedding neighbours (scaled by cosine * lambda),
         * collected separately so they can ONLY fill continuation gaps. */
        flu_tok_t last = bytes[k - 1u];
        flu_tok_t nb[8];
        uint32_t got = fluency_neighbors(m, last, nb, smooth_k);
        for (uint32_t i = 0; i + 1u < k; ++i) {
            variant[i] = bytes[i];
        }
        for (uint32_t j = 0; j < got; ++j) {
            double c = fluency_cosine(m, last, nb[j]);
            if (c <= 0.0) {
                continue;
            }
            variant[k - 1u] = nb[j];
            fluency_accumulate_context(m, variant, k, lambda * c, borrow);
        }
    }

    /* Gap-filling only: a similar context may add mass to a continuation the
     * exact context never voted for, but must never perturb one it did. This
     * makes F3 safe -- it reduces to F1 wherever the graph already has an
     * opinion, and only generalizes across surface forms where F1 is blank. */
    if (use_kernel) {
        for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
            if (acc[b] <= 0.0) {
                acc[b] += borrow[b];
            }
        }
    }

    const double alpha = 0.5;
    double total = 0.0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        total += acc[b];
    }
    double denom = total + alpha * (double)m->vocab_count;
    int argmax = -1;
    double best = -1.0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (!m->seen[b]) {
            continue;
        }
        double p = (acc[b] + alpha) / (denom > 0.0 ? denom : 1.0);
        dist[b] = p;
        if (p > best) {
            best = p;
            argmax = (int)b;
        }
    }
    return argmax;
}

int fluency_predict_firing(FluencyModel *m, FluencyReservoir *r, const flu_tok_t *ctx,
                           size_t ctx_len, double *dist, double lambda, uint32_t hops,
                           uint32_t k_nb) {
    if (!m || !dist) {
        return -1;
    }
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        dist[b] = 0.0;
    }
    int use = (r != NULL && lambda > 0.0 && k_nb > 0u);

    double acc[FLUENCY_VOCAB] = {0};
    double borrow[FLUENCY_VOCAB] = {0};
    flu_tok_t variant[FLUENCY_MAX_ORDER];

    for (uint32_t k = 1u; k <= m->order; ++k) {
        if ((size_t)k > ctx_len) {
            break;
        }
        const flu_tok_t *bytes = ctx + (ctx_len - k);
        fluency_accumulate_context(m, bytes, k, 1.0, acc);
        if (!use) {
            continue;
        }
        /* Borrow from firing-path role-peers of the predecessor token. */
        flu_tok_t last = bytes[k - 1u];
        flu_tok_t nb[8];
        uint32_t got = fluency_firing_neighbors(r, last, hops, nb, k_nb < 8u ? k_nb : 8u);
        for (uint32_t i = 0; i + 1u < k; ++i) {
            variant[i] = bytes[i];
        }
        for (uint32_t j = 0; j < got; ++j) {
            double c = fluency_firing_cosine(r, last, nb[j], hops);
            if (c <= 0.0) {
                continue;
            }
            variant[k - 1u] = nb[j];
            fluency_accumulate_context(m, variant, k, lambda * c, borrow);
        }
    }

    if (use) { /* gap-fill only, same safety as F3 */
        for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
            if (acc[b] <= 0.0) {
                acc[b] += borrow[b];
            }
        }
    }

    const double alpha = 0.5;
    double total = 0.0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        total += acc[b];
    }
    double denom = total + alpha * (double)m->vocab_count;
    int argmax = -1;
    double best = -1.0;
    for (uint32_t b = 0; b < FLUENCY_VOCAB; ++b) {
        if (!m->seen[b]) {
            continue;
        }
        double p = (acc[b] + alpha) / (denom > 0.0 ? denom : 1.0);
        dist[b] = p;
        if (p > best) {
            best = p;
            argmax = (int)b;
        }
    }
    return argmax;
}

void fluency_evaluate_firing(FluencyModel *m, FluencyReservoir *r, const flu_tok_t *text,
                             size_t len, double lambda, uint32_t hops, uint32_t k_nb,
                             FluencyMetrics *out) {
    if (!m || !text || !out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    uint64_t mut_before = m->e->debug.mutations_applied;
    uint32_t nodes_before = m->e->node_count, edges_before = m->e->edge_count;
    double dist[FLUENCY_VOCAB];
    double uni_denom = (double)m->unigram_total + (double)m->vocab_count;
    for (size_t i = 0; i < len; ++i) {
        flu_tok_t actual = text[i];
        int pred = fluency_predict_firing(m, r, text, i, dist, lambda, hops, k_nb);
        double p = dist[actual];
        if (p <= 0.0) {
            p = 0.5 / (uni_denom > 0.0 ? uni_denom : 1.0);
        }
        out->nll_sum += -log2(p);
        out->eval_positions++;
        if (pred == (int)actual) {
            out->correct_argmax++;
        }
    }
    double n = (double)(out->eval_positions ? out->eval_positions : 1u);
    out->model_perplexity = pow(2.0, out->nll_sum / n);
    out->eval_mutations = m->e->debug.mutations_applied - mut_before;
    out->eval_node_growth = m->e->node_count - nodes_before;
    out->eval_edge_growth = m->e->edge_count - edges_before;
}

void fluency_evaluate_kernel(FluencyModel *m, const flu_tok_t *text, size_t len, double lambda,
                             uint32_t smooth_k, FluencyMetrics *out) {
    if (!m || !text || !out) {
        return;
    }
    memset(out, 0, sizeof(*out));
    uint64_t mut_before = m->e->debug.mutations_applied;
    uint32_t nodes_before = m->e->node_count;
    uint32_t edges_before = m->e->edge_count;

    double dist[FLUENCY_VOCAB];
    double uni_denom = (double)m->unigram_total + (double)m->vocab_count;

    for (size_t i = 0; i < len; ++i) {
        flu_tok_t actual = text[i];
        int pred = fluency_predict_kernel(m, text, i, dist, lambda, smooth_k);
        double p = dist[actual];
        if (p <= 0.0) {
            p = 0.5 / (uni_denom > 0.0 ? uni_denom : 1.0);
        }
        out->nll_sum += -log2(p);
        double p_uni = ((double)m->unigram[actual] + 1.0) / (uni_denom > 0.0 ? uni_denom : 1.0);
        out->unigram_perplexity += -log2(p_uni);
        out->eval_positions++;
        if (pred == (int)actual) {
            out->correct_argmax++;
        }
    }
    double n = (double)(out->eval_positions ? out->eval_positions : 1u);
    out->model_perplexity = pow(2.0, out->nll_sum / n);
    out->unigram_perplexity = pow(2.0, out->unigram_perplexity / n);
    out->eval_mutations = m->e->debug.mutations_applied - mut_before;
    out->eval_node_growth = m->e->node_count - nodes_before;
    out->eval_edge_growth = m->e->edge_count - edges_before;
}

uint32_t fluency_neighbors(const FluencyModel *m, flu_tok_t byte, flu_tok_t *out, uint32_t k) {
    if (!m || !m->embed || !out || k == 0u || m->vidx[byte] == UINT32_MAX) {
        return 0u;
    }
    uint8_t used[FLUENCY_VOCAB] = {0};
    uint32_t written = 0u;
    while (written < k) {
        double best = -2.0;
        int best_b = -1;
        for (uint32_t vi = 0; vi < m->vdim; ++vi) {
            flu_tok_t cand = m->vbyte[vi];
            if (cand == byte || used[cand]) {
                continue;
            }
            double c = fluency_cosine(m, byte, cand);
            if (c > best) {
                best = c;
                best_b = (int)cand;
            }
        }
        if (best_b < 0) {
            break;
        }
        used[best_b] = 1u;
        out[written++] = (flu_tok_t)best_b;
    }
    return written;
}
