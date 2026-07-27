// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II
//
// Session checkpoint format (kill-and-resume / fast boot).
//
// One file, versioned magic + CRC (same spirit as nerva_persist v0.2).
//
// Layout (host endian, guarded by endian_marker = 0x0102):
//
//   NervaSessionHeader  (fixed size; header_crc32 over all other header fields
//                        with header_crc32 zeroed; no timestamps)
//   engine_blob         [engine_offset, engine_size)  full nerva_persist v0.2 file
//   fluency_blob        [fluency_offset, fluency_size) FluencyModel side state
//   chat_blob           [chat_offset, chat_size)       optional NervaChat state
//
// Magic: "NervaSes" (8 bytes, no NUL required in file)
// Version: major=0, minor=1
// Payload CRC: CRC32 of engine_blob || fluency_blob || chat_blob
//
// Fluency blob (v1):
//   "FluSide1" (8) then fixed scalars, then variable tables:
//     tok_node[V], seen[V], unigram[V]
//     edge slots (cap, count, FluencyEdgeSlot[cap])
//     fast_out[V * fast_slots]  — includes deposit_pos (age for w_eff)
//     stream_pos (THE subtle clock — unrestored → silent decay/resurrect)
//     instance_pool[instance_cap]
//     mood_reg[K], optional mood_aff[V*K]
//     optional gain tables (cooc/tok/vidx)
//     optional hub tables (tok_hubs, hub_mem_*)
//   name_map is NOT stored; rebuilt from engine names after load.
//   F2 embed is NOT stored (chat/predict path does not need it).
//
// Chat blob (v1) when flags has NERVA_SESSION_FLAG_CHAT:
//   "ChatSd1\0" (8) + vocab strings + mint ledger + hist + flags/stats.
//
// Rationale for one file vs sidecar: single atomic path for --save-state /
// --load-state; engine stays nerva_persist v0.2 so core src/ is untouched.

#ifndef NERVA_FLUENCY_SESSION_H
#define NERVA_FLUENCY_SESSION_H

#include "fluency.h"

#include <stddef.h>
#include <stdint.h>

#define NERVA_SESSION_MAGIC "NervaSes"
#define NERVA_SESSION_VERSION_MAJOR 0u
#define NERVA_SESSION_VERSION_MINOR 1u
#define NERVA_SESSION_ENDIAN_MARKER 0x0102u
#define NERVA_SESSION_FLAG_CHAT 1u

typedef struct NervaSessionHeader {
    uint8_t magic[8];
    uint16_t version_major;
    uint16_t version_minor;
    uint16_t header_size;
    uint16_t endian_marker;
    uint32_t flags;
    uint32_t fluency_vocab; /* must match FLUENCY_VOCAB at load */
    uint32_t max_nodes;
    uint32_t max_edges;
    uint32_t max_names;
    uint32_t max_memory;
    uint32_t max_schemas;
    int32_t weight_max_q8_8;
    uint64_t engine_offset;
    uint64_t engine_size;
    uint64_t fluency_offset;
    uint64_t fluency_size;
    uint64_t chat_offset;
    uint64_t chat_size;
    uint64_t file_size;
    uint32_t payload_crc32;
    uint32_t header_crc32;
} NervaSessionHeader;

/* Internal: save/load with optional chat payload (chat_repl uses these). */
int fluency_session_save(const FluencyModel *m, const char *path, const uint8_t *chat,
                         size_t chat_len);
/* e/m zeroed on entry. chat_out may be NULL if chat_cap==0 (chat section ignored). */
int fluency_session_load(NervaEngine *e, FluencyModel *m, const char *path, uint8_t *chat_out,
                         size_t chat_cap, size_t *chat_len_out);

#endif
