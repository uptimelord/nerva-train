// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_debug.h"
#include "nerva_graph.h"

#include <stdio.h>

static const char *nerva_debug_node_label(const NervaEngine *e, uint32_t node_id) {
    if (!e || node_id >= e->node_count) {
        return "?";
    }
    const char *name = nerva_name_by_id(e, e->nodes[node_id].name_id);
    return (name && name[0] != '\0') ? name : "?";
}

void nerva_debug_log_fire(NervaEngine *e, uint32_t node_id) {
    if (!e || e->fire_log_count >= e->fire_log_cap) {
        return;
    }

    NervaFireRecord *rec = &e->fire_log[e->fire_log_count++];
    rec->tick = e->tick;
    rec->node_id = node_id;
}

void nerva_debug_clear_fire_log(NervaEngine *e) {
    if (!e) {
        return;
    }
    e->fire_log_count = 0;
}

void nerva_debug_print_fire_trace(const NervaEngine *e, FILE *out) {
    if (!e || !out) {
        return;
    }

    for (uint32_t i = 0; i < e->fire_log_count; ++i) {
        const NervaFireRecord *rec = &e->fire_log[i];
        fprintf(out, "tick %llu: %s fired\n",
                (unsigned long long)rec->tick, nerva_debug_node_label(e, rec->node_id));
    }
}

void nerva_debug_print_path_line(const NervaEngine *e, FILE *out, const uint32_t *node_ids,
                                 uint32_t count) {
    if (!e || !out || !node_ids || count == 0) {
        return;
    }

    fputs("path:", out);
    for (uint32_t i = 0; i < count; ++i) {
        if (i == 0) {
            fputc(' ', out);
        } else {
            fputs(" -> ", out);
        }
        fputs(nerva_debug_node_label(e, node_ids[i]), out);
    }
    fputc('\n', out);
}

int nerva_debug_save_fire_trace(const NervaEngine *e, const char *path) {
    if (!e || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        return -1;
    }

    nerva_debug_print_fire_trace(e, out);
    fclose(out);
    return 0;
}

int nerva_debug_save_fire_trace_with_path(const NervaEngine *e, const char *path,
                                          const uint32_t *node_ids, uint32_t count) {
    if (!e || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        return -1;
    }

    nerva_debug_print_fire_trace(e, out);
    nerva_debug_print_path_line(e, out, node_ids, count);
    fclose(out);
    return 0;
}

int nerva_debug_fire_sequence_matches(const NervaEngine *e, const uint32_t *node_ids,
                                      uint32_t count) {
    if (!e || !node_ids) {
        return 0;
    }
    if (e->fire_log_count != count) {
        return 0;
    }

    for (uint32_t i = 0; i < count; ++i) {
        if (e->fire_log[i].node_id != node_ids[i]) {
            return 0;
        }
        if (i > 0 && e->fire_log[i].tick < e->fire_log[i - 1].tick) {
            return 0;
        }
    }
    return 1;
}
