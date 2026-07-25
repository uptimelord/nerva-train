// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_parse.h"
#include "nerva_engine.h"
#include "nerva_graph.h"
#include "nerva_event.h"
#include "nerva_learning.h"
#include "nerva_mutation.h"
#include "nerva_exception.h"
#include "nerva_routing.h"
#include "nerva_persist.h"
#include "nerva_schema.h"
#include "nerva_math.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void nerva_parse_strip_comment(char *line) {
    if (!line) {
        return;
    }
    char *hash = strchr(line, '#');
    if (hash) {
        *hash = '\0';
    }
}

static char *nerva_parse_trim(char *s) {
    if (!s) {
        return NULL;
    }
    while (*s && isspace((unsigned char)*s)) {
        s++;
    }
    if (*s == '\0') {
        return s;
    }
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
    return s;
}

static int nerva_parse_tokenize(char *line, char *tokens[], int max_tokens) {
    int count = 0;
    char *p = line;
    while (*p && count < max_tokens) {
        while (*p && isspace((unsigned char)*p)) {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        tokens[count++] = p;
        while (*p && !isspace((unsigned char)*p)) {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        *p++ = '\0';
    }
    return count;
}

static void nerva_parse_copy_arg(char *dst, size_t cap, const char *src) {
    if (!dst || cap == 0) {
        return;
    }
    if (!src) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, cap, "%s", src);
}

static void nerva_parse_ensure_adjacency(NervaEngine *e) {
    if (e && !e->adjacency_valid && e->edge_count > 0) {
        nerva_graph_rebuild_adjacency(e);
    }
}

static int nerva_parse_require_node(NervaEngine *e, const char *name, uint32_t *out) {
    if (!e || !name || !out) {
        return -1;
    }
    uint32_t id = nerva_find_node_by_name(e, name);
    if (id == UINT32_MAX) {
        return -1;
    }
    *out = id;
    return 0;
}

int nerva_parse_line(const char *line, NervaCommand *out) {
    if (!out) {
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->type = NERVA_CMD_NOOP;
    out->count = 1u;

    if (!line) {
        return 0;
    }

    char buf[NERVA_PARSE_LINE_MAX];
    snprintf(buf, sizeof(buf), "%s", line);
    nerva_parse_strip_comment(buf);
    char *trimmed = nerva_parse_trim(buf);
    if (!trimmed || trimmed[0] == '\0') {
        return 0;
    }

    char *tokens[8];
    int n = nerva_parse_tokenize(trimmed, tokens, 8);
    if (n == 0) {
        return 0;
    }

    if (strcmp(tokens[0], "NODE") == 0) {
        if (n != 2) {
            return -1;
        }
        out->type = NERVA_CMD_NODE;
        nerva_parse_copy_arg(out->arg0, sizeof(out->arg0), tokens[1]);
        return 1;
    }
    if (strcmp(tokens[0], "EDGE") == 0) {
        if (n != 4) {
            return -1;
        }
        out->type = NERVA_CMD_EDGE;
        nerva_parse_copy_arg(out->arg0, sizeof(out->arg0), tokens[1]);
        nerva_parse_copy_arg(out->arg1, sizeof(out->arg1), tokens[2]);
        nerva_parse_copy_arg(out->arg2, sizeof(out->arg2), tokens[3]);
        return 1;
    }
    if (strcmp(tokens[0], "QUERY") == 0) {
        if (n != 4) {
            return -1;
        }
        out->type = NERVA_CMD_QUERY;
        nerva_parse_copy_arg(out->arg0, sizeof(out->arg0), tokens[1]);
        nerva_parse_copy_arg(out->arg1, sizeof(out->arg1), tokens[2]);
        nerva_parse_copy_arg(out->arg2, sizeof(out->arg2), tokens[3]);
        return 1;
    }
    if (strcmp(tokens[0], "ACTIVATE") == 0) {
        if (n != 2) {
            return -1;
        }
        out->type = NERVA_CMD_ACTIVATE;
        nerva_parse_copy_arg(out->arg0, sizeof(out->arg0), tokens[1]);
        return 1;
    }
    if (strcmp(tokens[0], "FEEDBACK") == 0) {
        if (n != 2) {
            return -1;
        }
        out->type = NERVA_CMD_FEEDBACK;
        nerva_parse_copy_arg(out->arg0, sizeof(out->arg0), tokens[1]);
        return 1;
    }
    if (strcmp(tokens[0], "BLOCK") == 0) {
        if (n != 3) {
            return -1;
        }
        out->type = NERVA_CMD_BLOCK;
        nerva_parse_copy_arg(out->arg0, sizeof(out->arg0), tokens[1]);
        nerva_parse_copy_arg(out->arg1, sizeof(out->arg1), tokens[2]);
        return 1;
    }
    if (strcmp(tokens[0], "TICK") == 0) {
        if (n != 1 && n != 2) {
            return -1;
        }
        out->type = NERVA_CMD_TICK;
        if (n == 2) {
            out->count = (uint32_t)strtoul(tokens[1], NULL, 10);
            if (out->count == 0) {
                return -1;
            }
        }
        return 1;
    }
    if (strcmp(tokens[0], "APPLY") == 0) {
        if (n != 1) {
            return -1;
        }
        out->type = NERVA_CMD_APPLY;
        return 1;
    }
    if (strcmp(tokens[0], "SAVE") == 0) {
        if (n != 2) {
            return -1;
        }
        out->type = NERVA_CMD_SAVE;
        nerva_parse_copy_arg(out->arg0, sizeof(out->arg0), tokens[1]);
        return 1;
    }
    if (strcmp(tokens[0], "LOAD") == 0) {
        if (n != 2) {
            return -1;
        }
        out->type = NERVA_CMD_LOAD;
        nerva_parse_copy_arg(out->arg0, sizeof(out->arg0), tokens[1]);
        return 1;
    }
    if (strcmp(tokens[0], "SCHEMA") == 0) {
        if (n != 7) {
            return -1;
        }
        if (strcmp(tokens[1], "OBSERVE") != 0 && strcmp(tokens[1], "APPLY") != 0) {
            return -1;
        }
        out->type = NERVA_CMD_SCHEMA;
        nerva_parse_copy_arg(out->sub, sizeof(out->sub), tokens[1]);
        nerva_parse_copy_arg(out->arg0, sizeof(out->arg0), tokens[2]);
        nerva_parse_copy_arg(out->arg1, sizeof(out->arg1), tokens[3]);
        nerva_parse_copy_arg(out->arg2, sizeof(out->arg2), tokens[4]);
        nerva_parse_copy_arg(out->arg3, sizeof(out->arg3), tokens[5]);
        nerva_parse_copy_arg(out->arg4, sizeof(out->arg4), tokens[6]);
        return 1;
    }

    return -1;
}

int nerva_parse_execute(NervaEngine *e, const NervaCommand *cmd) {
    if (!e || !cmd) {
        return -1;
    }

    switch (cmd->type) {
    case NERVA_CMD_NOOP:
        return 0;
    case NERVA_CMD_NODE:
        if (nerva_get_or_create_node(e, cmd->arg0) == UINT32_MAX) {
            return -1;
        }
        return 0;
    case NERVA_CMD_EDGE: {
        uint32_t src = nerva_get_or_create_node(e, cmd->arg0);
        uint32_t dst = nerva_get_or_create_node(e, cmd->arg2);
        uint16_t rel = nerva_relation_from_string(cmd->arg1);
        if (src == UINT32_MAX || dst == UINT32_MAX || rel == NERVA_REL_NONE) {
            return -1;
        }
        if (nerva_graph_create_edge(e, src, dst, rel) == UINT32_MAX) {
            return -1;
        }
        return 0;
    }
    case NERVA_CMD_QUERY: {
        uint32_t src = UINT32_MAX;
        uint32_t dst = UINT32_MAX;
        uint16_t rel = nerva_relation_from_string(cmd->arg1);
        if (rel == NERVA_REL_NONE || nerva_parse_require_node(e, cmd->arg0, &src) != 0 ||
            nerva_parse_require_node(e, cmd->arg2, &dst) != 0) {
            return -1;
        }
        nerva_parse_ensure_adjacency(e);
        nerva_routing_begin_query(e, src, dst, rel);
        if (!nerva_activate_node(e, src, NERVA_Q8_8_ONE)) {
            return -1;
        }
        return 0;
    }
    case NERVA_CMD_ACTIVATE: {
        uint32_t node = UINT32_MAX;
        if (nerva_parse_require_node(e, cmd->arg0, &node) != 0) {
            return -1;
        }
        nerva_parse_ensure_adjacency(e);
        if (!nerva_activate_node(e, node, NERVA_Q8_8_ONE)) {
            return -1;
        }
        return 0;
    }
    case NERVA_CMD_FEEDBACK:
        if (strcmp(cmd->arg0, "correct") == 0) {
            (void)nerva_feedback_correct(e);
            return 0;
        }
        if (strcmp(cmd->arg0, "wrong") == 0) {
            (void)nerva_feedback_wrong(e);
            return 0;
        }
        return -1;
    case NERVA_CMD_BLOCK: {
        uint32_t src = UINT32_MAX;
        uint32_t dst = UINT32_MAX;
        if (nerva_parse_require_node(e, cmd->arg0, &src) != 0 ||
            nerva_parse_require_node(e, cmd->arg1, &dst) != 0) {
            return -1;
        }
        if (!nerva_queue_blocker_edge(e, src, dst, NERVA_REL_BLOCKS)) {
            return -1;
        }
        return 0;
    }
    case NERVA_CMD_TICK:
        nerva_parse_ensure_adjacency(e);
        nerva_tick_n(e, cmd->count);
        return 0;
    case NERVA_CMD_APPLY:
        nerva_apply_mutations(e);
        return 0;
    case NERVA_CMD_SAVE:
        return nerva_persist_save(e, cmd->arg0);
    case NERVA_CMD_LOAD:
        return nerva_persist_load(e, cmd->arg0);
    case NERVA_CMD_SCHEMA: {
        uint16_t rel_a = nerva_relation_from_string(cmd->arg1);
        uint16_t rel_b = nerva_relation_from_string(cmd->arg3);
        if (rel_a == NERVA_REL_NONE || rel_b == NERVA_REL_NONE) {
            return -1;
        }
        if (strcmp(cmd->sub, "OBSERVE") == 0) {
            uint32_t a = nerva_get_or_create_node(e, cmd->arg0);
            uint32_t b = nerva_get_or_create_node(e, cmd->arg2);
            uint32_t c = nerva_get_or_create_node(e, cmd->arg4);
            if (a == UINT32_MAX || b == UINT32_MAX || c == UINT32_MAX) {
                return -1;
            }
            nerva_schema_observe_triple(e, a, rel_a, b, rel_b, c);
            return 0;
        }
        if (strcmp(cmd->sub, "APPLY") == 0) {
            uint32_t a = UINT32_MAX;
            uint32_t b = UINT32_MAX;
            uint32_t c = UINT32_MAX;
            if (nerva_parse_require_node(e, cmd->arg0, &a) != 0 ||
                nerva_parse_require_node(e, cmd->arg2, &b) != 0 ||
                nerva_parse_require_node(e, cmd->arg4, &c) != 0) {
                return -1;
            }
            nerva_parse_ensure_adjacency(e);
            if (!nerva_schema_apply(e, a, rel_a, b, rel_b, c)) {
                return -1;
            }
            return 0;
        }
        return -1;
    }
    default:
        return -1;
    }
}

int nerva_parse_run_line(NervaEngine *e, const char *line) {
    NervaCommand cmd;
    int parsed = nerva_parse_line(line, &cmd);
    if (parsed < 0) {
        return -1;
    }
    if (parsed == 0) {
        return 0;
    }
    return nerva_parse_execute(e, &cmd);
}

int nerva_parse_run_file(NervaEngine *e, const char *path) {
    if (!e || !path) {
        return -1;
    }

    FILE *in = fopen(path, "r");
    if (!in) {
        return -1;
    }

    char line[NERVA_PARSE_LINE_MAX];
    while (fgets(line, sizeof(line), in)) {
        if (nerva_parse_run_line(e, line) != 0) {
            fclose(in);
            return -1;
        }
    }

    fclose(in);
    return 0;
}
