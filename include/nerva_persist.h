// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_PERSIST_H
#define NERVA_PERSIST_H

#include "nerva_types.h"
#include <stdio.h>

#define NERVA_FILE_MAGIC "Nervav001"
#define NERVA_FILE_VERSION_MAJOR 0u
#define NERVA_FILE_VERSION_MINOR 2u
#define NERVA_FILE_ENDIAN_MARKER 0x0102u

typedef struct NervaFileHeader {
    uint8_t magic[8];
    uint16_t version_major;
    uint16_t version_minor;
    uint16_t header_size;
    uint16_t endian_marker;
    uint32_t flags;
    uint32_t node_count;
    uint32_t edge_count;
    uint32_t memory_count;
    uint32_t schema_count;
    uint32_t string_count;
    uint32_t relation_count;
    uint32_t reserved0;
    uint64_t tick;
    uint64_t nodes_offset;
    uint64_t edges_offset;
    uint64_t memory_offset;
    uint64_t schemas_offset;
    uint64_t strings_offset;
    uint64_t relations_offset;
    uint64_t file_size;
    uint64_t payload_offset;
    uint32_t payload_crc32;
    uint32_t header_crc32;
} NervaFileHeader;

int nerva_persist_save(NervaEngine *e, const char *path);
int nerva_persist_load(NervaEngine *e, const char *path);

uint32_t nerva_persist_crc32(const uint8_t *data, size_t len);
void nerva_persist_print_summary(const NervaEngine *e, FILE *out, const char *label);

#endif
