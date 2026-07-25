// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_persist.h"
#include "nerva_graph.h"
#include "nerva_routing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(NervaFileHeader) == 128, "NervaFileHeader must be 128 bytes");
#endif

static uint32_t g_crc32_table[256];
static int g_crc32_ready = 0;

typedef struct NervaPersistStaging {
    NervaNode *nodes;
    NervaEdge *edges;
    NervaMemoryBlock *memory;
    NervaSchema *schemas;
    char **names;
    uint32_t name_count;
} NervaPersistStaging;

static void nerva_persist_crc32_init(void) {
    if (g_crc32_ready) {
        return;
    }
    for (uint32_t i = 0; i < 256u; ++i) {
        uint32_t c = i;
        for (int j = 0; j < 8; ++j) {
            if (c & 1u) {
                c = 0xEDB88320u ^ (c >> 1);
            } else {
                c >>= 1;
            }
        }
        g_crc32_table[i] = c;
    }
    g_crc32_ready = 1;
}

uint32_t nerva_persist_crc32(const uint8_t *data, size_t len) {
    nerva_persist_crc32_init();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc = g_crc32_table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

static int nerva_persist_header_magic_valid(const NervaFileHeader *h) {
    if (!h) {
        return 0;
    }
    if (memcmp(h->magic, NERVA_FILE_MAGIC, 8) != 0) {
        return 0;
    }
    if (h->version_major != NERVA_FILE_VERSION_MAJOR ||
        h->version_minor != NERVA_FILE_VERSION_MINOR) {
        return 0;
    }
    if (h->header_size != (uint16_t)sizeof(NervaFileHeader)) {
        return 0;
    }
    if (h->endian_marker != NERVA_FILE_ENDIAN_MARKER) {
        return 0;
    }
    return 1;
}

static int nerva_persist_header_crc_valid(const NervaFileHeader *h) {
    NervaFileHeader chk = *h;
    uint32_t expected = chk.header_crc32;
    chk.header_crc32 = 0;
    return nerva_persist_crc32((const uint8_t *)&chk, sizeof(chk)) == expected;
}

static int nerva_persist_header_layout_valid(const NervaFileHeader *h, uint64_t file_size) {
    if (!h || file_size < sizeof(NervaFileHeader)) {
        return 0;
    }
    if (h->payload_offset < (uint64_t)sizeof(NervaFileHeader)) {
        return 0;
    }
    if (h->nodes_offset != h->payload_offset) {
        return 0;
    }

    uint64_t off = h->nodes_offset;
    off += (uint64_t)h->node_count * (uint64_t)sizeof(NervaNode);
    if (h->edges_offset != off) {
        return 0;
    }
    off += (uint64_t)h->edge_count * (uint64_t)sizeof(NervaEdge);
    if (h->memory_offset != off) {
        return 0;
    }
    off += (uint64_t)h->memory_count * (uint64_t)sizeof(NervaMemoryBlock);
    if (h->schemas_offset != off) {
        return 0;
    }
    off += (uint64_t)h->schema_count * (uint64_t)sizeof(NervaSchema);
    if (h->strings_offset != off) {
        return 0;
    }
    if (h->file_size != file_size || h->file_size < h->strings_offset) {
        return 0;
    }
    if (h->relations_offset != h->file_size) {
        return 0;
    }
    return 1;
}

static void nerva_persist_clear_runtime(NervaEngine *e) {
    if (!e) {
        return;
    }

    e->event_count = 0;
    e->active_count = 0;
    e->fire_log_count = 0;
    e->trace_head = 0;
    e->trace_count = 0;
    e->mutation_head = 0;
    e->mutation_tail = 0;
    e->mutation_count = 0;
    e->mutation_log_count = 0;
    e->expectation_count = 0;
    e->idle_ticks = 0;
    e->prediction_mode = 0;
    e->active_query_tag = 0;
    e->last_query_start_tick = 0;
    e->adjacency_valid = 0;
    memset(&e->debug, 0, sizeof(e->debug));
    nerva_routing_reset(e);
}

static void nerva_persist_crc32_update(uint32_t *crc, const uint8_t *data, size_t len) {
    nerva_persist_crc32_init();
    for (size_t i = 0; i < len; ++i) {
        *crc = g_crc32_table[(*crc ^ data[i]) & 0xFFu] ^ (*crc >> 8);
    }
}

static int nerva_persist_write_blob(FILE *out, uint32_t *crc, const void *data, size_t size,
                                    size_t count) {
    if (count == 0) {
        return 0;
    }
    if (!data) {
        return -1;
    }
    if (fwrite(data, size, count, out) != count) {
        return -1;
    }
    nerva_persist_crc32_update(crc, data, size * count);
    return 0;
}

static int nerva_persist_write_strings(const NervaEngine *e, FILE *out, uint32_t *crc) {
    if (nerva_persist_write_blob(out, crc, &e->name_count, sizeof(e->name_count), 1) != 0) {
        return -1;
    }
    for (uint32_t i = 0; i < e->name_count; ++i) {
        const char *name = e->names[i] ? e->names[i] : "";
        size_t len = strlen(name);
        if (len > 65535u) {
            return -1;
        }
        uint16_t len16 = (uint16_t)len;
        if (nerva_persist_write_blob(out, crc, &len16, sizeof(len16), 1) != 0) {
            return -1;
        }
        if (len > 0 && nerva_persist_write_blob(out, crc, name, 1, len) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nerva_persist_counts_fit(const NervaEngine *e, const NervaFileHeader *h) {
    if (!e || !h) {
        return 0;
    }
    if (h->node_count > e->node_cap || h->edge_count > e->edge_cap ||
        h->memory_count > e->memory_cap || h->schema_count > e->schema_cap ||
        h->string_count > e->name_cap) {
        return 0;
    }
    return 1;
}

static int nerva_persist_read_blob(const uint8_t **cursor, size_t *remaining, void *dst,
                                   size_t elem_size, size_t count) {
    if (count == 0) {
        return 0;
    }
    if (!cursor || !*cursor || !remaining || !dst) {
        return -1;
    }
    size_t need = elem_size * count;
    if (*remaining < need) {
        return -1;
    }
    memcpy(dst, *cursor, need);
    *cursor += need;
    *remaining -= need;
    return 0;
}

static void nerva_persist_staging_free(NervaPersistStaging *st) {
    if (!st) {
        return;
    }
    if (st->names) {
        for (uint32_t i = 0; i < st->name_count; ++i) {
            free(st->names[i]);
        }
    }
    free(st->nodes);
    free(st->edges);
    free(st->memory);
    free(st->schemas);
    free(st->names);
    memset(st, 0, sizeof(*st));
}

static int nerva_persist_staging_parse(const NervaFileHeader *h, const uint8_t *payload,
                                       size_t payload_len, NervaPersistStaging *st) {
    if (!h || !payload || !st) {
        return -1;
    }

    memset(st, 0, sizeof(*st));

    st->nodes = calloc(h->node_count, sizeof(NervaNode));
    st->edges = calloc(h->edge_count, sizeof(NervaEdge));
    st->memory = calloc(h->memory_count, sizeof(NervaMemoryBlock));
    st->schemas = calloc(h->schema_count, sizeof(NervaSchema));
    if ((h->node_count > 0 && !st->nodes) || (h->edge_count > 0 && !st->edges) ||
        (h->memory_count > 0 && !st->memory) || (h->schema_count > 0 && !st->schemas)) {
        nerva_persist_staging_free(st);
        return -1;
    }

    const uint8_t *cursor = payload;
    size_t remaining = payload_len;

    if (nerva_persist_read_blob(&cursor, &remaining, st->nodes, sizeof(NervaNode),
                                h->node_count) != 0 ||
        nerva_persist_read_blob(&cursor, &remaining, st->edges, sizeof(NervaEdge), h->edge_count) !=
            0 ||
        nerva_persist_read_blob(&cursor, &remaining, st->memory, sizeof(NervaMemoryBlock),
                                h->memory_count) != 0 ||
        nerva_persist_read_blob(&cursor, &remaining, st->schemas, sizeof(NervaSchema),
                                h->schema_count) != 0) {
        nerva_persist_staging_free(st);
        return -1;
    }

    uint32_t stored_count = 0;
    if (nerva_persist_read_blob(&cursor, &remaining, &stored_count, sizeof(stored_count), 1) != 0 ||
        stored_count != h->string_count) {
        nerva_persist_staging_free(st);
        return -1;
    }

    if (stored_count > 0) {
        st->names = calloc(stored_count, sizeof(char *));
        if (!st->names) {
            nerva_persist_staging_free(st);
            return -1;
        }
        st->name_count = stored_count;
        for (uint32_t i = 0; i < stored_count; ++i) {
            uint16_t len16 = 0;
            if (nerva_persist_read_blob(&cursor, &remaining, &len16, sizeof(len16), 1) != 0) {
                nerva_persist_staging_free(st);
                return -1;
            }
            char *copy = calloc((size_t)len16 + 1u, 1);
            if (!copy) {
                nerva_persist_staging_free(st);
                return -1;
            }
            if (len16 > 0 &&
                nerva_persist_read_blob(&cursor, &remaining, copy, 1, len16) != 0) {
                free(copy);
                nerva_persist_staging_free(st);
                return -1;
            }
            st->names[i] = copy;
        }
    }

    if (remaining != 0) {
        nerva_persist_staging_free(st);
        return -1;
    }

    return 0;
}

static int nerva_persist_apply_staging(NervaEngine *e, const NervaFileHeader *h,
                                       const NervaPersistStaging *st) {
    if (!e || !h || !st) {
        return -1;
    }

    for (uint32_t i = 0; i < e->name_count; ++i) {
        free(e->names[i]);
        e->names[i] = NULL;
    }
    e->name_count = 0;

    if (h->node_count > 0) {
        memcpy(e->nodes, st->nodes, (size_t)h->node_count * sizeof(NervaNode));
    }
    if (h->edge_count > 0) {
        memcpy(e->edges, st->edges, (size_t)h->edge_count * sizeof(NervaEdge));
    }
    if (h->memory_count > 0) {
        memcpy(e->memory, st->memory, (size_t)h->memory_count * sizeof(NervaMemoryBlock));
    }
    if (h->schema_count > 0) {
        memcpy(e->schemas, st->schemas, (size_t)h->schema_count * sizeof(NervaSchema));
    }

    for (uint32_t i = 0; i < st->name_count; ++i) {
        e->names[e->name_count++] = st->names[i];
    }

    e->node_count = h->node_count;
    e->edge_count = h->edge_count;
    e->memory_count = h->memory_count;
    e->schema_count = h->schema_count;
    e->tick = h->tick;

    nerva_persist_clear_runtime(e);
    nerva_graph_rebuild_adjacency(e);
    return 0;
}

int nerva_persist_save(NervaEngine *e, const char *path) {
    if (!e || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w+b");
    if (!out) {
        return -1;
    }

    uint32_t crc = 0xFFFFFFFFu;

    NervaFileHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, NERVA_FILE_MAGIC, 8);
    h.version_major = NERVA_FILE_VERSION_MAJOR;
    h.version_minor = NERVA_FILE_VERSION_MINOR;
    h.header_size = (uint16_t)sizeof(NervaFileHeader);
    h.endian_marker = NERVA_FILE_ENDIAN_MARKER;
    h.node_count = e->node_count;
    h.edge_count = e->edge_count;
    h.memory_count = e->memory_count;
    h.schema_count = e->schema_count;
    h.string_count = e->name_count;
    h.relation_count = 0;
    h.tick = e->tick;

    if (fwrite(&h, sizeof(h), 1, out) != 1) {
        fclose(out);
        return -1;
    }

    h.nodes_offset = (uint64_t)ftell(out);
    if (nerva_persist_write_blob(out, &crc, e->nodes, sizeof(NervaNode), h.node_count) != 0) {
        fclose(out);
        return -1;
    }

    h.edges_offset = (uint64_t)ftell(out);
    if (nerva_persist_write_blob(out, &crc, e->edges, sizeof(NervaEdge), h.edge_count) != 0) {
        fclose(out);
        return -1;
    }

    h.memory_offset = (uint64_t)ftell(out);
    if (nerva_persist_write_blob(out, &crc, e->memory, sizeof(NervaMemoryBlock), h.memory_count) !=
        0) {
        fclose(out);
        return -1;
    }

    h.schemas_offset = (uint64_t)ftell(out);
    if (nerva_persist_write_blob(out, &crc, e->schemas, sizeof(NervaSchema), h.schema_count) != 0) {
        fclose(out);
        return -1;
    }

    h.strings_offset = (uint64_t)ftell(out);
    if (nerva_persist_write_strings(e, out, &crc) != 0) {
        fclose(out);
        return -1;
    }

    h.relations_offset = (uint64_t)ftell(out);
    h.file_size = (uint64_t)ftell(out);
    h.payload_offset = h.nodes_offset;
    h.payload_crc32 = crc ^ 0xFFFFFFFFu;
    h.header_crc32 = 0;
    h.header_crc32 = nerva_persist_crc32((const uint8_t *)&h, sizeof(h));

    if (fseek(out, 0, SEEK_SET) != 0 || fwrite(&h, sizeof(h), 1, out) != 1) {
        fclose(out);
        return -1;
    }

    fclose(out);
    return 0;
}

int nerva_persist_load(NervaEngine *e, const char *path) {
    if (!e || !path) {
        return -1;
    }

    FILE *in = fopen(path, "rb");
    if (!in) {
        return -1;
    }

    NervaFileHeader h;
    if (fread(&h, sizeof(h), 1, in) != 1 || !nerva_persist_header_magic_valid(&h)) {
        fclose(in);
        return -1;
    }

    if (fseek(in, 0, SEEK_END) != 0) {
        fclose(in);
        return -1;
    }
    uint64_t file_size = (uint64_t)ftell(in);
    if (h.file_size == 0) {
        h.file_size = file_size;
    } else if (h.file_size != file_size) {
        fclose(in);
        return -1;
    }

    if (!nerva_persist_header_crc_valid(&h) ||
        !nerva_persist_header_layout_valid(&h, file_size) ||
        !nerva_persist_counts_fit(e, &h)) {
        fclose(in);
        return -1;
    }

    if (h.file_size <= h.payload_offset) {
        fclose(in);
        return -1;
    }

    size_t payload_len = (size_t)(h.file_size - h.payload_offset);
    uint8_t *payload = malloc(payload_len);
    if (!payload) {
        fclose(in);
        return -1;
    }

    if (fseek(in, (long)h.payload_offset, SEEK_SET) != 0 ||
        fread(payload, 1, payload_len, in) != payload_len) {
        free(payload);
        fclose(in);
        return -1;
    }
    fclose(in);

    if (nerva_persist_crc32(payload, payload_len) != h.payload_crc32) {
        free(payload);
        return -1;
    }

    NervaPersistStaging staging;
    if (nerva_persist_staging_parse(&h, payload, payload_len, &staging) != 0) {
        free(payload);
        return -1;
    }
    free(payload);

    if (nerva_persist_apply_staging(e, &h, &staging) != 0) {
        nerva_persist_staging_free(&staging);
        return -1;
    }

    staging.names = NULL;
    staging.name_count = 0;
    nerva_persist_staging_free(&staging);
    return 0;
}

void nerva_persist_print_summary(const NervaEngine *e, FILE *out, const char *label) {
    if (!e || !out) {
        return;
    }

    fprintf(out,
            "label=%s tick=%llu nodes=%u edges=%u memory=%u schemas=%u strings=%u "
            "adjacency_valid=%u\n",
            label ? label : "engine", (unsigned long long)e->tick, e->node_count, e->edge_count,
            e->memory_count, e->schema_count, e->name_count, (unsigned)e->adjacency_valid);
}
