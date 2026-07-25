// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_schema.h"
#include "nerva_graph.h"
#include "nerva_learning.h"
#include "nerva_config.h"
#include "nerva_math.h"

#include <string.h>

uint16_t nerva_schema_output_relation(uint16_t rel_a, uint16_t rel_b) {
    if (rel_a == NERVA_REL_KIND_OF && rel_b == NERVA_REL_KIND_OF) {
        return NERVA_REL_KIND_OF;
    }
    if (rel_a == NERVA_REL_INSIDE && rel_b == NERVA_REL_MOVED_TO) {
        return NERVA_REL_LOCATED_AT;
    }
    return NERVA_REL_NONE;
}

uint16_t nerva_schema_lookup_output(const NervaEngine *e, uint16_t rel_a, uint16_t rel_b) {
    if (e) {
        for (uint32_t i = 0; i < e->schema_count; ++i) {
            const NervaSchema *s = &e->schemas[i];
            if (s->rel_a == rel_a && s->rel_b == rel_b && s->rel_out != NERVA_REL_NONE) {
                return s->rel_out;
            }
        }
    }
    return nerva_schema_output_relation(rel_a, rel_b);
}

static NervaSchema *nerva_schema_find_mutable(NervaEngine *e, uint16_t rel_a, uint16_t rel_b,
                                              uint16_t rel_out) {
    if (!e) {
        return NULL;
    }

    for (uint32_t i = 0; i < e->schema_count; ++i) {
        NervaSchema *s = &e->schemas[i];
        if (s->rel_a == rel_a && s->rel_b == rel_b && s->rel_out == rel_out) {
            return s;
        }
    }
    return NULL;
}

static NervaSchema *nerva_schema_get_or_create(NervaEngine *e, uint16_t rel_a, uint16_t rel_b,
                                               uint16_t rel_out) {
    NervaSchema *found = nerva_schema_find_mutable(e, rel_a, rel_b, rel_out);
    if (found) {
        return found;
    }
    if (!e || e->schema_count >= e->schema_cap) {
        return NULL;
    }

    NervaSchema *s = &e->schemas[e->schema_count++];
    memset(s, 0, sizeof(*s));
    s->id = e->schema_count - 1u;
    s->rel_a = rel_a;
    s->rel_b = rel_b;
    s->rel_out = rel_out;
    s->flags = NERVA_SCHEMA_CANDIDATE;
    s->schema_edge_cost = NERVA_SCHEMA_SHORTCUT_EDGE_COST;
    return s;
}

int nerva_schema_register_composition(NervaEngine *e, uint16_t rel_a, uint16_t rel_b,
                                      uint16_t rel_out) {
    if (!e || rel_a == NERVA_REL_NONE || rel_b == NERVA_REL_NONE ||
        rel_out == NERVA_REL_NONE) {
        return -1;
    }
    /* One output per (rel_a, rel_b): reject conflicting registration. */
    uint16_t existing = nerva_schema_lookup_output(e, rel_a, rel_b);
    if (existing != NERVA_REL_NONE && existing != rel_out) {
        return -1;
    }
    return nerva_schema_get_or_create(e, rel_a, rel_b, rel_out) ? 0 : -1;
}

static int nerva_schema_triple_seen(const NervaSchema *s, uint32_t a, uint32_t b, uint32_t c) {
    for (uint32_t i = 0; i < s->distinct_count; ++i) {
        const NervaSchemaDistinct *ex = &s->distinct[i];
        if (ex->a == a && ex->b == b && ex->c == c) {
            return 1;
        }
    }
    return 0;
}

static int nerva_schema_record_distinct(NervaSchema *s, uint32_t a, uint32_t b, uint32_t c) {
    if (nerva_schema_triple_seen(s, a, b, c)) {
        return 0;
    }
    if (s->distinct_count >= NERVA_SCHEMA_DISTINCT_CAP) {
        return 0;
    }

    NervaSchemaDistinct *ex = &s->distinct[s->distinct_count++];
    ex->a = a;
    ex->b = b;
    ex->c = c;
    return 1;
}

static int nerva_schema_has_compression_benefit(const NervaSchema *s) {
    if (!s || s->distinct_count == 0) {
        return 0;
    }
    return (s->schema_edge_cost + s->exception_cost) < s->raw_hop_cost;
}

static int nerva_schema_premises_present(const NervaEngine *e, uint32_t a, uint16_t rel_a,
                                         uint32_t b, uint16_t rel_b, uint32_t c) {
    if (!e || a >= e->node_count || b >= e->node_count || c >= e->node_count) {
        return 0;
    }
    if (a == b || b == c) {
        return 0;
    }
    return nerva_graph_has_edge(e, a, b, rel_a) && nerva_graph_has_edge(e, b, c, rel_b);
}

void nerva_schema_observe_triple(NervaEngine *e, uint32_t a, uint16_t rel_a, uint32_t b,
                                 uint16_t rel_b, uint32_t c) {
    if (!e || a >= e->node_count || b >= e->node_count || c >= e->node_count) {
        return;
    }

    uint16_t rel_out = nerva_schema_lookup_output(e, rel_a, rel_b);
    if (rel_out == NERVA_REL_NONE) {
        return;
    }

    NervaSchema *s = nerva_schema_get_or_create(e, rel_a, rel_b, rel_out);
    if (!s) {
        return;
    }

    if (!nerva_schema_record_distinct(s, a, b, c)) {
        return;
    }

    s->support_count++;
    s->raw_hop_cost += NERVA_SCHEMA_PREMISE_HOP_COST;
    if (s->support_count >= e->cfg.schema_support_threshold) {
        nerva_schema_promote_if_ready(e, s->id);
    }
}

int nerva_schema_promote_if_ready(NervaEngine *e, uint32_t schema_id) {
    if (!e || schema_id >= e->schema_count) {
        return 0;
    }

    NervaSchema *s = &e->schemas[schema_id];
    if (s->flags & NERVA_SCHEMA_PROMOTED) {
        return 1;
    }
    if (s->support_count < e->cfg.schema_support_threshold) {
        return 0;
    }
    if (s->exception_count > e->cfg.schema_exception_limit) {
        return 0;
    }
    if (!nerva_schema_has_compression_benefit(s)) {
        return 0;
    }

    s->flags |= NERVA_SCHEMA_PROMOTED;
    s->promoted_tick = e->tick;
    e->debug.schemas_promoted++;
    return 1;
}

int nerva_schema_apply(NervaEngine *e, uint32_t a, uint16_t rel_a, uint32_t b, uint16_t rel_b,
                       uint32_t c) {
    if (!e || a >= e->node_count || b >= e->node_count || c >= e->node_count) {
        return 0;
    }

    uint16_t rel_out = nerva_schema_lookup_output(e, rel_a, rel_b);
    if (rel_out == NERVA_REL_NONE) {
        return 0;
    }

    NervaSchema *s = nerva_schema_find_mutable(e, rel_a, rel_b, rel_out);
    if (!s || !(s->flags & NERVA_SCHEMA_PROMOTED)) {
        return 0;
    }
    if (!nerva_schema_premises_present(e, a, rel_a, b, rel_b, c)) {
        return 0;
    }

    s->total_predictions++;
    if (!nerva_queue_create_edge(e, a, c, rel_out, NERVA_REASON_SCHEMA_APPLY)) {
        return 0;
    }

    s->correct_predictions++;
    e->debug.schemas_applied++;
    return 1;
}

const NervaSchema *nerva_schema_find_promoted(const NervaEngine *e, uint16_t rel_a,
                                              uint16_t rel_b) {
    if (!e) {
        return NULL;
    }

    uint16_t rel_out = nerva_schema_lookup_output(e, rel_a, rel_b);
    if (rel_out == NERVA_REL_NONE) {
        return NULL;
    }

    for (uint32_t i = 0; i < e->schema_count; ++i) {
        const NervaSchema *s = &e->schemas[i];
        if (s->rel_a == rel_a && s->rel_b == rel_b && s->rel_out == rel_out &&
            (s->flags & NERVA_SCHEMA_PROMOTED)) {
            return s;
        }
    }
    return NULL;
}

uint32_t nerva_schema_count(const NervaEngine *e) {
    return e ? e->schema_count : 0u;
}
