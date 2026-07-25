// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_BENCH_H
#define NERVA_BENCH_H

#include <stdint.h>

#define NERVA_BENCH_COUNT 9u

typedef struct NervaBenchResult {
    const char *name;
    int pass;
    uint32_t ticks;
    uint32_t peak_queue;
    char notes[160];
} NervaBenchResult;

typedef struct NervaBenchReport {
    NervaBenchResult items[NERVA_BENCH_COUNT];
    uint32_t count;
    int all_pass;
    uint32_t peak_queue;
    double ticks_per_sec;
    double est_ram_mb;
    double fluid_routine_pct;
} NervaBenchReport;

int nerva_bench_run_all(NervaBenchReport *report, const char *bench_log_path,
                        const char *trace_log_path);

#endif
