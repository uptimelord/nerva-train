// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_bench.h"
#include "nerva_config.h"
#include "nerva_debug.h"
#include "nerva_engine.h"
#include "nerva_event.h"
#include "nerva_exception.h"
#include "nerva_graph.h"
#include "nerva_memory.h"
#include "nerva_mutation.h"
#include "nerva_routing.h"
#include "nerva_schema.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static void nerva_bench_setup_poodle(NervaEngine *e, uint32_t *poodle, uint32_t *dog,
                                     uint32_t *animal) {
    *poodle = nerva_get_or_create_node(e, "poodle");
    *dog = nerva_get_or_create_node(e, "dog");
    *animal = nerva_get_or_create_node(e, "animal");
    nerva_graph_create_edge(e, *poodle, *dog, NERVA_REL_KIND_OF);
    nerva_graph_create_edge(e, *dog, *animal, NERVA_REL_KIND_OF);
    nerva_graph_rebuild_adjacency(e);
}

static NervaConfig nerva_bench_config(void) {
    return nerva_config_test();
}

static uint32_t nerva_bench_peak_queue(const NervaEngine *e) {
    return e ? e->debug.event_depth_max : 0u;
}

static void nerva_bench_update_peak(NervaEngine *e, uint32_t *peak) {
    uint32_t q = nerva_bench_peak_queue(e);
    if (q > *peak) {
        *peak = q;
    }
}

static void nerva_bench_result_init(NervaBenchResult *r, const char *name) {
    memset(r, 0, sizeof(*r));
    r->name = name;
}

static int nerva_bench_fire_log_contains(const NervaEngine *e, uint32_t node_id) {
    for (uint32_t i = 0; i < e->fire_log_count; ++i) {
        if (e->fire_log[i].node_id == node_id) {
            return 1;
        }
    }
    return 0;
}

static uint32_t nerva_bench_count_ordered_fires(const NervaEngine *e, const uint32_t *nodes,
                                                uint32_t count) {
    uint32_t found = 0;
    for (uint32_t i = 0; i < e->fire_log_count && found < count; ++i) {
        if (e->fire_log[i].node_id == nodes[found]) {
            found++;
        }
    }
    return found;
}

static int nerva_bench_node_fires_within(NervaEngine *e, uint32_t node_id, uint32_t max_ticks,
                                       uint32_t *ticks_used) {
    for (uint32_t t = 0; t <= max_ticks; ++t) {
        if (nerva_bench_fire_log_contains(e, node_id)) {
            if (ticks_used) {
                *ticks_used = t;
            }
            return 1;
        }
        if (t < max_ticks) {
            nerva_tick(e);
        }
    }
    if (ticks_used) {
        *ticks_used = max_ticks;
    }
    return 0;
}

static void nerva_bench_append_trace(NervaEngine *e, FILE *trace_out, const char *label) {
    if (!trace_out) {
        return;
    }
    fprintf(trace_out, "benchmark=%s tick=%llu\n", label, (unsigned long long)e->tick);
    nerva_debug_print_fire_trace(e, trace_out);
    fputc('\n', trace_out);
}

static int nerva_bench_few_shot_apply(NervaEngine *e, uint32_t n_train) {
    for (uint32_t i = 0; i < n_train; ++i) {
        char a_name[16];
        char b_name[16];
        char c_name[16];
        snprintf(a_name, sizeof(a_name), "fsA%u", i);
        snprintf(b_name, sizeof(b_name), "fsB%u", i);
        snprintf(c_name, sizeof(c_name), "fsC%u", i);
        uint32_t a = nerva_get_or_create_node(e, a_name);
        uint32_t b = nerva_get_or_create_node(e, b_name);
        uint32_t c = nerva_get_or_create_node(e, c_name);
        nerva_schema_observe_triple(e, a, NERVA_REL_KIND_OF, b, NERVA_REL_KIND_OF, c);
    }

    uint32_t tx = nerva_get_or_create_node(e, "fsTestA");
    uint32_t ty = nerva_get_or_create_node(e, "fsTestB");
    uint32_t tz = nerva_get_or_create_node(e, "fsTestC");
    nerva_graph_create_edge(e, tx, ty, NERVA_REL_KIND_OF);
    nerva_graph_create_edge(e, ty, tz, NERVA_REL_KIND_OF);
    nerva_graph_rebuild_adjacency(e);

    if (nerva_schema_find_promoted(e, NERVA_REL_KIND_OF, NERVA_REL_KIND_OF) == NULL) {
        return 0;
    }
    if (!nerva_schema_apply(e, tx, NERVA_REL_KIND_OF, ty, NERVA_REL_KIND_OF, tz)) {
        return 0;
    }
    nerva_apply_mutations(e);
    return nerva_graph_has_edge(e, tx, tz, NERVA_REL_KIND_OF);
}

static int nerva_bench_run_few_shot(NervaBenchResult *r, FILE *trace_out, uint32_t *peak) {
    nerva_bench_result_init(r, "few_shot_relation");

    const uint32_t curve[] = {1u, 3u, 5u, 10u};
    int acc_at_5 = 0;
    uint32_t total_ticks = 0;

    for (size_t i = 0; i < sizeof(curve) / sizeof(curve[0]); ++i) {
        NervaEngine e;
        if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
            return -1;
        }

        int ok = nerva_bench_few_shot_apply(&e, curve[i]);
        total_ticks += e.tick;
        nerva_bench_update_peak(&e, peak);
        if (curve[i] == 5u) {
            acc_at_5 = ok;
        }
        if (curve[i] == 5u && trace_out) {
            nerva_bench_append_trace(&e, trace_out, r->name);
        }
        nerva_engine_free(&e);
    }

    r->ticks = total_ticks;
    r->peak_queue = *peak;
    r->pass = acc_at_5;
    snprintf(r->notes, sizeof(r->notes), "N=1,3,5,10 curve; N=5 held-out apply=%s", acc_at_5 ? "ok" : "fail");
    return 0;
}

static int nerva_bench_run_transitive(NervaBenchResult *r, FILE *trace_out, uint32_t *peak) {
    nerva_bench_result_init(r, "transitive_reasoning");

    NervaEngine e;
    if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
        return -1;
    }

    uint32_t a = nerva_get_or_create_node(&e, "trA");
    uint32_t b = nerva_get_or_create_node(&e, "trB");
    uint32_t c = nerva_get_or_create_node(&e, "trC");
    nerva_graph_create_edge(&e, a, b, NERVA_REL_KIND_OF);
    nerva_graph_create_edge(&e, b, c, NERVA_REL_KIND_OF);
    nerva_graph_rebuild_adjacency(&e);

    nerva_debug_clear_fire_log(&e);
    nerva_activate_node(&e, a, NERVA_Q8_8_ONE);
    uint32_t ticks_used = 100u;
    int fired = nerva_bench_node_fires_within(&e, c, 100u, &ticks_used);

    r->ticks = e.tick;
    r->peak_queue = nerva_bench_peak_queue(&e);
    nerva_bench_update_peak(&e, peak);
    r->pass = fired && ticks_used <= 100u;
    snprintf(r->notes, sizeof(r->notes), "C fired in %u ticks (limit 100)", ticks_used);
    nerva_bench_append_trace(&e, trace_out, r->name);
    nerva_engine_free(&e);
    return 0;
}

static void nerva_bench_setup_penguin(NervaEngine *e, uint32_t *bird, uint32_t *penguin,
                                      uint32_t *fly) {
    *bird = nerva_get_or_create_node(e, "benchBird");
    *penguin = nerva_get_or_create_node(e, "benchPenguin");
    *fly = nerva_get_or_create_node(e, "benchFly");
    nerva_graph_create_edge(e, *bird, *fly, NERVA_REL_USUALLY_HAS_PROPERTY);
    nerva_graph_create_edge(e, *penguin, *bird, NERVA_REL_KIND_OF);
    nerva_graph_rebuild_adjacency(e);
    nerva_queue_blocker_edge(e, *penguin, *fly, NERVA_REL_BLOCKS);
    nerva_apply_mutations(e);
}

static int nerva_bench_run_exception(NervaBenchResult *r, FILE *trace_out, uint32_t *peak) {
    nerva_bench_result_init(r, "exception_handling");

    NervaEngine e;
    if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
        return -1;
    }

    uint32_t bird, penguin, fly;
    nerva_bench_setup_penguin(&e, &bird, &penguin, &fly);

    uint32_t bird_ok = 0;
    for (uint32_t trial = 0; trial < 20u; ++trial) {
        nerva_debug_clear_fire_log(&e);
        nerva_activate_node(&e, bird, NERVA_Q8_8_ONE);
        nerva_tick_n(&e, 4);
        if (nerva_bench_fire_log_contains(&e, fly)) {
            bird_ok++;
        }
    }

    uint32_t penguin_ok = 0;
    for (uint32_t trial = 0; trial < 5u; ++trial) {
        nerva_debug_clear_fire_log(&e);
        nerva_activate_node(&e, penguin, NERVA_Q8_8_ONE);
        nerva_tick_n(&e, 4);
        if (!nerva_bench_fire_log_contains(&e, fly) && e.nodes[fly].v < e.nodes[fly].theta_fire) {
            penguin_ok++;
        }
    }

    r->ticks = e.tick;
    r->peak_queue = nerva_bench_peak_queue(&e);
    nerva_bench_update_peak(&e, peak);
    r->pass = bird_ok >= 19u && penguin_ok >= 5u;
    snprintf(r->notes, sizeof(r->notes), "bird->fly %u/20 penguin blocked %u/5", bird_ok, penguin_ok);

    nerva_debug_clear_fire_log(&e);
    nerva_activate_node(&e, penguin, NERVA_Q8_8_ONE);
    nerva_tick_n(&e, 4);
    nerva_bench_append_trace(&e, trace_out, r->name);
    nerva_engine_free(&e);
    return 0;
}

static void nerva_bench_train_container_schema(NervaEngine *e) {
    for (uint32_t i = 0; i < e->cfg.schema_support_threshold; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "cb%u", i);
        uint32_t ball = nerva_get_or_create_node(e, name);
        snprintf(name, sizeof(name), "cx%u", i);
        uint32_t box = nerva_get_or_create_node(e, name);
        snprintf(name, sizeof(name), "cs%u", i);
        uint32_t shelf = nerva_get_or_create_node(e, name);
        nerva_schema_observe_triple(e, ball, NERVA_REL_INSIDE, box, NERVA_REL_MOVED_TO, shelf);
    }
}

static int nerva_bench_run_container(NervaBenchResult *r, FILE *trace_out, uint32_t *peak) {
    nerva_bench_result_init(r, "container_movement");

    NervaEngine e;
    if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
        return -1;
    }

    nerva_bench_train_container_schema(&e);

    uint32_t ball = nerva_get_or_create_node(&e, "queryBall");
    uint32_t box = nerva_get_or_create_node(&e, "queryBox");
    uint32_t shelf = nerva_get_or_create_node(&e, "queryShelf");
    nerva_graph_create_edge(&e, ball, box, NERVA_REL_INSIDE);
    nerva_graph_create_edge(&e, box, shelf, NERVA_REL_MOVED_TO);
    nerva_graph_rebuild_adjacency(&e);

    int applied = nerva_schema_apply(&e, ball, NERVA_REL_INSIDE, box, NERVA_REL_MOVED_TO, shelf);
    if (applied) {
        nerva_apply_mutations(&e);
    }

    nerva_debug_clear_fire_log(&e);
    nerva_activate_node(&e, ball, NERVA_Q8_8_ONE);
    uint32_t ticks_used = 200u;
    int shelf_fired = nerva_bench_node_fires_within(&e, shelf, 200u, &ticks_used);
    int has_edge = nerva_graph_has_edge(&e, ball, shelf, NERVA_REL_LOCATED_AT);

    r->ticks = e.tick;
    r->peak_queue = nerva_bench_peak_queue(&e);
    nerva_bench_update_peak(&e, peak);
    r->pass = applied && has_edge && shelf_fired && ticks_used <= 200u;
    snprintf(r->notes, sizeof(r->notes), "located_at edge=%s shelf in %u ticks", has_edge ? "yes" : "no",
             ticks_used);
    nerva_bench_append_trace(&e, trace_out, r->name);
    nerva_engine_free(&e);
    return 0;
}

static int nerva_bench_run_ambiguity(NervaBenchResult *r, FILE *trace_out, uint32_t *peak) {
    nerva_bench_result_init(r, "ambiguity_resolution");

    uint32_t wins = 0;
    uint32_t total_ticks = 0;

    for (uint32_t trial = 0; trial < 5u; ++trial) {
        NervaEngine e;
        if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
            return -1;
        }

        uint32_t n1 = nerva_get_or_create_node(&e, "ambA");
        uint32_t n2 = nerva_get_or_create_node(&e, "ambB");
        uint32_t z = nerva_get_or_create_node(&e, "ambZ");

        const nerva_q8_8_t hi = (nerva_q8_8_t)192;
        const nerva_q8_8_t lo = (nerva_q8_8_t)64;

        nerva_routing_begin_query(&e, n1, z, NERVA_REL_KIND_OF);
        nerva_activate_node(&e, n1, hi);
        nerva_activate_node(&e, n2, lo);

        int won = 0;
        for (uint32_t t = 0; t <= 50u; ++t) {
            nerva_tick(&e);
            if (e.nodes[n1].v > e.nodes[n2].v && e.nodes[n1].v >= e.cfg.theta_compete_q8_8) {
                won = 1;
                break;
            }
        }
        if (won) {
            wins++;
        }

        total_ticks += e.tick;
        nerva_bench_update_peak(&e, peak);
        if (trial == 0) {
            nerva_bench_append_trace(&e, trace_out, r->name);
        }
        nerva_engine_free(&e);
    }

    r->ticks = total_ticks;
    r->peak_queue = *peak;
    r->pass = wins >= 4u;
    snprintf(r->notes, sizeof(r->notes), "3:1 bias wins %u/5 within 50 ticks", wins);
    return 0;
}

static int nerva_bench_run_memory_consolidation(NervaBenchResult *r, FILE *trace_out,
                                                uint32_t *peak) {
    nerva_bench_result_init(r, "memory_consolidation");

    NervaEngine e;
    if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
        return -1;
    }

    uint32_t nodes[5];
    const char *names[] = {"mc0", "mc1", "mc2", "mc3", "mc4"};
    for (int i = 0; i < 5; ++i) {
        nodes[i] = nerva_get_or_create_node(&e, names[i]);
        if (i > 0) {
            nerva_graph_create_edge(&e, nodes[i - 1], nodes[i], NERVA_REL_KIND_OF);
        }
    }
    nerva_graph_rebuild_adjacency(&e);

    nerva_debug_clear_fire_log(&e);
    nerva_activate_node(&e, nodes[0], NERVA_Q8_8_ONE);
    uint32_t mem_id = nerva_memory_begin_episode(&e, 0xB007u);
    nerva_tick_n(&e, 12);
    nerva_memory_end_episode(&e, mem_id);
    nerva_memory_charge_update(&e, mem_id, 8.0f, 0.0f, 1.0f, 0.0f);

    for (uint32_t i = 0; i < e.cfg.idle_consolidate_ticks * 3u; ++i) {
        nerva_tick(&e);
    }

    int consolidated = nerva_memory_is_consolidated(&e, mem_id);

    nerva_debug_clear_fire_log(&e);
    nerva_activate_node(&e, nodes[0], NERVA_Q8_8_ONE);
    nerva_tick_n(&e, 20);

    uint32_t follow = nerva_bench_count_ordered_fires(&e, &nodes[1], 4u);

    r->ticks = e.tick;
    r->peak_queue = nerva_bench_peak_queue(&e);
    nerva_bench_update_peak(&e, peak);
    r->pass = consolidated && follow >= 4u;
    snprintf(r->notes, sizeof(r->notes), "consolidated=%s retrieval=%u/4 nodes", consolidated ? "yes" : "no",
             follow);
    nerva_bench_append_trace(&e, trace_out, r->name);
    nerva_engine_free(&e);
    return 0;
}

static int nerva_bench_run_forgetting(NervaBenchResult *r, FILE *trace_out, uint32_t *peak) {
    nerva_bench_result_init(r, "forgetting");

    NervaEngine e;
    if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
        return -1;
    }

    uint32_t poodle, dog, animal;
    nerva_bench_setup_poodle(&e, &poodle, &dog, &animal);
    (void)dog;
    (void)animal;

    uint32_t mem_id = nerva_memory_begin_episode(&e, 1u);
    nerva_activate_node(&e, poodle, NERVA_Q8_8_ONE);
    nerva_tick_n(&e, 4);
    nerva_memory_end_episode(&e, mem_id);
    nerva_memory_charge_update(&e, mem_id, 0.1f, 0.0f, 0.0f, 0.0f);

    uint32_t ticks_before = e.tick;
    for (uint32_t i = 0; i < e.cfg.memory_hold_period_ticks + e.cfg.idle_consolidate_ticks * 4u; ++i) {
        nerva_tick(&e);
    }

    int forgotten = nerva_memory_is_marked_delete(&e, mem_id);
    const NervaMemoryBlock *mem = nerva_memory_get(&e, mem_id);
    float charge = mem ? mem->charge : 0.0f;

    r->ticks = e.tick - ticks_before;
    r->peak_queue = nerva_bench_peak_queue(&e);
    nerva_bench_update_peak(&e, peak);
    r->pass = forgotten || charge < e.cfg.memory_forget_threshold;
    snprintf(r->notes, sizeof(r->notes), "marked_delete=%s charge=%.3f", forgotten ? "yes" : "no", charge);
    nerva_bench_append_trace(&e, trace_out, r->name);
    nerva_engine_free(&e);
    return 0;
}

static int nerva_bench_run_contradiction(NervaBenchResult *r, FILE *trace_out, uint32_t *peak) {
    nerva_bench_result_init(r, "contradiction_repair");

    NervaEngine e;
    if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
        return -1;
    }

    uint32_t bird, penguin, fly;
    nerva_bench_setup_penguin(&e, &bird, &penguin, &fly);
    (void)bird;

    nerva_tick_t start = e.tick;
    nerva_debug_clear_fire_log(&e);
    nerva_activate_node(&e, penguin, NERVA_Q8_8_ONE);

    int trace_within_2 = 0;
    int suppressed_within_10 = 0;
    for (uint32_t t = 0; t < 10u; ++t) {
        nerva_tick(&e);
        if (e.tick - start <= 2u && nerva_exception_count_blocker_traces(&e) >= 1u) {
            trace_within_2 = 1;
        }
        if (e.nodes[fly].v < e.nodes[fly].theta_fire) {
            suppressed_within_10 = 1;
        }
    }

    r->ticks = e.tick - start;
    r->peak_queue = nerva_bench_peak_queue(&e);
    nerva_bench_update_peak(&e, peak);
    r->pass = trace_within_2 && suppressed_within_10 && !nerva_bench_fire_log_contains(&e, fly);
    snprintf(r->notes, sizeof(r->notes), "blocker_trace<=2t=%s fly_suppressed=%s",
             trace_within_2 ? "yes" : "no", suppressed_within_10 ? "yes" : "no");
    nerva_bench_append_trace(&e, trace_out, r->name);
    nerva_engine_free(&e);
    return 0;
}

static double nerva_bench_measure_ticks_per_sec(void) {
    NervaEngine e;
    if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
        return 0.0;
    }

    uint32_t poodle, dog, animal;
    nerva_bench_setup_poodle(&e, &poodle, &dog, &animal);
    (void)dog;
    (void)animal;
    nerva_activate_node(&e, poodle, NERVA_Q8_8_ONE);

    const uint32_t n = 10000u;
    clock_t t0 = clock();
    nerva_tick_n(&e, n);
    clock_t t1 = clock();
    nerva_engine_free(&e);

    if (t1 <= t0) {
        return (double)n;
    }
    return (double)n / (((double)(t1 - t0)) / (double)CLOCKS_PER_SEC);
}

static double nerva_bench_measure_fluid_routine_pct(void) {
    NervaEngine e;
    if (nerva_engine_init(&e, nerva_bench_config()) != 0) {
        return 100.0;
    }

    uint32_t poodle, dog, animal;
    nerva_bench_setup_poodle(&e, &poodle, &dog, &animal);
    (void)dog;

    const uint32_t rounds = 20u;
    uint64_t fluid_before = e.debug.fluid_activations;
    for (uint32_t i = 0; i < rounds; ++i) {
        nerva_routing_begin_query(&e, poodle, animal, NERVA_REL_KIND_OF);
        nerva_activate_node(&e, poodle, NERVA_Q8_8_ONE);
        nerva_tick_n(&e, 4);
    }
    uint64_t fluid = e.debug.fluid_activations - fluid_before;
    nerva_engine_free(&e);
    return ((double)fluid * 100.0) / (double)rounds;
}

static double nerva_bench_estimate_ram_mb(void) {
    NervaConfig cfg = nerva_bench_config();
    size_t bytes = sizeof(NervaEngine);
    bytes += (size_t)cfg.max_nodes * sizeof(NervaNode);
    bytes += (size_t)cfg.max_edges * sizeof(NervaEdge);
    bytes += (size_t)cfg.max_names * sizeof(char *);
    bytes += (size_t)cfg.max_events * sizeof(NervaEvent);
    bytes += (size_t)cfg.max_traces * sizeof(NervaTrace);
    bytes += (size_t)cfg.max_mutations * sizeof(NervaMutation);
    bytes += (size_t)cfg.max_schemas * sizeof(NervaSchema);
    bytes += (size_t)cfg.max_memory_blocks * sizeof(NervaMemoryBlock);
    return (double)bytes / (1024.0 * 1024.0);
}

static int nerva_bench_run_efficiency(NervaBenchReport *report, NervaBenchResult *r, uint32_t *peak) {
    nerva_bench_result_init(r, "compute_efficiency");

    NervaConfig cfg = nerva_bench_config();
    report->ticks_per_sec = nerva_bench_measure_ticks_per_sec();
    report->est_ram_mb = nerva_bench_estimate_ram_mb();
    report->fluid_routine_pct = nerva_bench_measure_fluid_routine_pct();

    int others_pass = 1;
    for (uint32_t i = 0; i + 1u < report->count; ++i) {
        if (!report->items[i].pass) {
            others_pass = 0;
            break;
        }
    }

    r->peak_queue = *peak;
    r->ticks = 10000u;
    r->pass = others_pass && *peak < cfg.max_events && report->est_ram_mb < 7168.0 &&
              report->ticks_per_sec >= 1000.0 && report->fluid_routine_pct < 5.0;
    snprintf(r->notes, sizeof(r->notes), "tps=%.0f ram=%.2fMB fluid=%.1f%% peak_q=%u",
             report->ticks_per_sec, report->est_ram_mb, report->fluid_routine_pct, *peak);
    return 0;
}

static int nerva_bench_write_log(const NervaBenchReport *report, const char *path) {
    if (!report || !path) {
        return -1;
    }

    FILE *out = fopen(path, "w");
    if (!out) {
        return -1;
    }

    for (uint32_t i = 0; i < report->count; ++i) {
        const NervaBenchResult *r = &report->items[i];
        fprintf(out, "benchmark=%s pass=%d ticks=%u peak_queue=%u notes=%s\n", r->name, r->pass, r->ticks,
                r->peak_queue, r->notes);
    }
    fprintf(out, "summary all_pass=%d peak_queue=%u ticks_per_sec=%.0f est_ram_mb=%.2f fluid_routine_pct=%.1f\n",
            report->all_pass, report->peak_queue, report->ticks_per_sec, report->est_ram_mb,
            report->fluid_routine_pct);
    fclose(out);
    return 0;
}

int nerva_bench_run_all(NervaBenchReport *report, const char *bench_log_path,
                        const char *trace_log_path) {
    if (!report) {
        return -1;
    }

    memset(report, 0, sizeof(*report));
    uint32_t peak = 0;

    FILE *trace_out = NULL;
    if (trace_log_path) {
        trace_out = fopen(trace_log_path, "w");
    }

    typedef int (*bench_fn)(NervaBenchResult *, FILE *, uint32_t *);
    bench_fn runners[] = {
        nerva_bench_run_few_shot,       nerva_bench_run_transitive,     nerva_bench_run_exception,
        nerva_bench_run_container,      nerva_bench_run_ambiguity,      nerva_bench_run_memory_consolidation,
        nerva_bench_run_forgetting,     nerva_bench_run_contradiction,
    };

    for (size_t i = 0; i < sizeof(runners) / sizeof(runners[0]); ++i) {
        if (runners[i](&report->items[report->count], trace_out, &peak) != 0) {
            if (trace_out) {
                fclose(trace_out);
            }
            return -1;
        }
        report->count++;
    }

    if (nerva_bench_run_efficiency(report, &report->items[report->count], &peak) != 0) {
        if (trace_out) {
            fclose(trace_out);
        }
        return -1;
    }
    report->count++;

    if (trace_out) {
        fclose(trace_out);
    }

    report->peak_queue = peak;
    report->all_pass = 1;
    for (uint32_t i = 0; i < report->count; ++i) {
        if (!report->items[i].pass) {
            report->all_pass = 0;
        }
    }

    if (bench_log_path) {
        nerva_bench_write_log(report, bench_log_path);
    }
    return report->all_pass ? 0 : 1;
}
