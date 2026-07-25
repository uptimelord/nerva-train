// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_PARSE_H
#define NERVA_PARSE_H

#include "nerva_types.h"

#define NERVA_PARSE_ARG_MAX 128u
#define NERVA_PARSE_LINE_MAX 512u

typedef enum NervaCmdType {
    NERVA_CMD_NOOP = 0,
    NERVA_CMD_NODE,
    NERVA_CMD_EDGE,
    NERVA_CMD_QUERY,
    NERVA_CMD_ACTIVATE,
    NERVA_CMD_FEEDBACK,
    NERVA_CMD_BLOCK,
    NERVA_CMD_TICK,
    NERVA_CMD_APPLY,
    NERVA_CMD_SAVE,
    NERVA_CMD_LOAD,
    NERVA_CMD_SCHEMA
} NervaCmdType;

typedef struct NervaCommand {
    NervaCmdType type;
    char sub[16];
    char arg0[NERVA_PARSE_ARG_MAX];
    char arg1[NERVA_PARSE_ARG_MAX];
    char arg2[NERVA_PARSE_ARG_MAX];
    char arg3[NERVA_PARSE_ARG_MAX];
    char arg4[NERVA_PARSE_ARG_MAX];
    uint32_t count;
} NervaCommand;

int nerva_parse_line(const char *line, NervaCommand *out);
int nerva_parse_execute(NervaEngine *e, const NervaCommand *cmd);
int nerva_parse_run_line(NervaEngine *e, const char *line);
int nerva_parse_run_file(NervaEngine *e, const char *path);

#endif
