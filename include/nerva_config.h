// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_CONFIG_H
#define NERVA_CONFIG_H

#include "nerva_types.h"
#include "nerva_math.h"

#define NERVA_DEFAULT_MAX_NODES 50000u
#define NERVA_DEFAULT_MAX_EDGES 1000000u
#define NERVA_DEFAULT_MAX_NAMES 50000u
#define NERVA_DEFAULT_MAX_EVENTS 500000u
#define NERVA_DEFAULT_MAX_ACTIVE 1024u
#define NERVA_DEFAULT_FIRE_LOG 1024u
#define NERVA_DEFAULT_MAX_TRACES 500000u
#define NERVA_DEFAULT_MAX_MUTATIONS 65536u
#define NERVA_DEFAULT_MUTATION_LOG 1024u

#define NERVA_TEST_MAX_NODES 1024u
#define NERVA_TEST_MAX_EDGES 10000u
#define NERVA_TEST_MAX_NAMES 1024u
#define NERVA_TEST_MAX_EVENTS 8192u
#define NERVA_TEST_MAX_ACTIVE 256u
#define NERVA_TEST_FIRE_LOG 64u
#define NERVA_TEST_MAX_TRACES 4096u
#define NERVA_TEST_TRACE_DECAY_SCAN 128u
#define NERVA_TEST_MAX_MUTATIONS 512u
#define NERVA_TEST_MUTATION_LOG 64u

#define NERVA_TRACE_PRE_DECAY_Q0_16 61566u
#define NERVA_TRACE_POST_DECAY_Q0_16 63523u
#define NERVA_TRACE_WINDOW_TICKS 256u

#define NERVA_WEIGHT_MIN_Q8_8 (-1024)
#define NERVA_WEIGHT_MAX_Q8_8 1024
#define NERVA_LTP_DELTA_Q8_8 16
#define NERVA_LTD_DELTA_Q8_8 (-12)
#define NERVA_GATE_CLOSE_STEP_Q0_16 16384u
#define NERVA_FEEDBACK_WRONG_GATE_THRESHOLD 2u

#define NERVA_DEFAULT_MAX_EXPECTATIONS 64u
#define NERVA_TEST_MAX_EXPECTATIONS 16u
#define NERVA_PREDICTION_MIN_STABILITY 3u
#define NERVA_PREDICTION_PRE_CHARGE_Q8_8 128
#define NERVA_PREDICTION_WINDOW_TICKS 8u
#define NERVA_SUPPRESS_Q0_16 32768u
#define NERVA_DEFAULT_MAX_SCHEMAS 16384u
#define NERVA_TEST_MAX_SCHEMAS 32u
#define NERVA_SCHEMA_SUPPORT_THRESHOLD 10u
#define NERVA_TEST_SCHEMA_SUPPORT_THRESHOLD 3u
#define NERVA_SCHEMA_EXCEPTION_LIMIT 2u
#define NERVA_SCHEMA_PREMISE_HOP_COST 2u
#define NERVA_SCHEMA_SHORTCUT_EDGE_COST 1u

#define NERVA_DEFAULT_MAX_MEMORY_BLOCKS 1024u
#define NERVA_TEST_MAX_MEMORY_BLOCKS 32u
#define NERVA_MEMORY_STORE_THRESHOLD 10.0f
#define NERVA_MEMORY_FORGET_THRESHOLD 1.0f
#define NERVA_MEMORY_HOLD_PERIOD_TICKS 10000u
#define NERVA_IDLE_CONSOLIDATE_TICKS 100u
#define NERVA_MEMORY_DECAY_PER_IDLE 0.001f
#define NERVA_MEMORY_REPLAY_TOP_K 5u
#define NERVA_TEST_MEMORY_STORE_THRESHOLD 5.0f
#define NERVA_TEST_MEMORY_FORGET_THRESHOLD 1.0f
#define NERVA_TEST_MEMORY_HOLD_PERIOD_TICKS 20u
#define NERVA_TEST_IDLE_CONSOLIDATE_TICKS 2u
#define NERVA_TEST_MEMORY_DECAY_PER_IDLE 0.5f
#define NERVA_TEST_MEMORY_REPLAY_TOP_K 2u

#define NERVA_FLUID_THRESHOLD_BASE_Q8_8 2048
#define NERVA_TEST_FLUID_THRESHOLD_BASE_Q8_8 640
#define NERVA_FATIGUE_INCREMENT_Q8_8 256
#define NERVA_FATIGUE_DECAY_SHIFT 8u
#define NERVA_DIFF_NOVELTY_W_Q8_8 512
#define NERVA_DIFF_CONTRADICTION_W_Q8_8 512
#define NERVA_DIFF_UNRESOLVED_W_Q8_8 256
#define NERVA_DIFF_CONFIDENCE_W_Q8_8 256
#define NERVA_THETA_COMPETE_Q8_8 128
#define NERVA_AMBIGUITY_INHIBITION_Q0_16 8192u

static inline void nerva_config_init_defaults(NervaConfig *cfg) {
    cfg->max_nodes = NERVA_DEFAULT_MAX_NODES;
    cfg->max_edges = NERVA_DEFAULT_MAX_EDGES;
    cfg->max_names = NERVA_DEFAULT_MAX_NAMES;
    cfg->max_events = NERVA_DEFAULT_MAX_EVENTS;
    cfg->max_active_nodes = NERVA_DEFAULT_MAX_ACTIVE;
    cfg->max_fire_log = NERVA_DEFAULT_FIRE_LOG;
    cfg->max_traces = NERVA_DEFAULT_MAX_TRACES;
    cfg->max_mutations = NERVA_DEFAULT_MAX_MUTATIONS;
    cfg->max_mutation_log = NERVA_DEFAULT_MUTATION_LOG;

    cfg->theta_fire_q8_8 = NERVA_Q8_8_ONE;
    cfg->v_rest_q8_8 = 0;
    cfg->v_reset_q8_8 = 0;
    cfg->default_output_q8_8 = NERVA_Q8_8_ONE;

    cfg->leak_shift = 6u;
    cfg->refractory_ticks = 3u;
    cfg->edge_delay_ticks = 1u;

    cfg->stale_ticks = 10u;
    cfg->merge_window_ticks = 0u;

    cfg->trace_pre_decay_q0_16 = NERVA_TRACE_PRE_DECAY_Q0_16;
    cfg->trace_post_decay_q0_16 = NERVA_TRACE_POST_DECAY_Q0_16;
    cfg->trace_window_ticks = NERVA_TRACE_WINDOW_TICKS;
    cfg->trace_decay_scan_limit = 1024u;

    cfg->weight_min_q8_8 = NERVA_WEIGHT_MIN_Q8_8;
    cfg->weight_max_q8_8 = NERVA_WEIGHT_MAX_Q8_8;
    cfg->default_weight_q8_8 = NERVA_Q8_8_ONE;
    cfg->ltp_delta_q8_8 = NERVA_LTP_DELTA_Q8_8;
    cfg->ltd_delta_q8_8 = NERVA_LTD_DELTA_Q8_8;
    cfg->gate_close_step_q0_16 = NERVA_GATE_CLOSE_STEP_Q0_16;
    cfg->feedback_wrong_gate_threshold = (uint16_t)NERVA_FEEDBACK_WRONG_GATE_THRESHOLD;
    cfg->prediction_min_stability = (uint16_t)NERVA_PREDICTION_MIN_STABILITY;
    cfg->prediction_pre_charge_q8_8 = (nerva_q8_8_t)NERVA_PREDICTION_PRE_CHARGE_Q8_8;
    cfg->prediction_window_ticks = NERVA_PREDICTION_WINDOW_TICKS;
    cfg->max_expectations = NERVA_DEFAULT_MAX_EXPECTATIONS;
    cfg->suppress_q0_16 = NERVA_SUPPRESS_Q0_16;
    cfg->max_schemas = NERVA_DEFAULT_MAX_SCHEMAS;
    cfg->schema_support_threshold = (uint16_t)NERVA_SCHEMA_SUPPORT_THRESHOLD;
    cfg->schema_exception_limit = (uint16_t)NERVA_SCHEMA_EXCEPTION_LIMIT;

    cfg->max_memory_blocks = NERVA_DEFAULT_MAX_MEMORY_BLOCKS;
    cfg->memory_store_threshold = NERVA_MEMORY_STORE_THRESHOLD;
    cfg->memory_forget_threshold = NERVA_MEMORY_FORGET_THRESHOLD;
    cfg->memory_hold_period_ticks = NERVA_MEMORY_HOLD_PERIOD_TICKS;
    cfg->idle_consolidate_ticks = NERVA_IDLE_CONSOLIDATE_TICKS;
    cfg->memory_decay_per_idle = NERVA_MEMORY_DECAY_PER_IDLE;
    cfg->memory_replay_top_k = (uint16_t)NERVA_MEMORY_REPLAY_TOP_K;

    cfg->fluid_threshold_base_q8_8 = (nerva_q8_8_t)NERVA_FLUID_THRESHOLD_BASE_Q8_8;
    cfg->fatigue_increment_q8_8 = (nerva_q8_8_t)NERVA_FATIGUE_INCREMENT_Q8_8;
    cfg->fatigue_decay_shift = NERVA_FATIGUE_DECAY_SHIFT;
    cfg->diff_novelty_w_q8_8 = (nerva_q8_8_t)NERVA_DIFF_NOVELTY_W_Q8_8;
    cfg->diff_contradiction_w_q8_8 = (nerva_q8_8_t)NERVA_DIFF_CONTRADICTION_W_Q8_8;
    cfg->diff_unresolved_w_q8_8 = (nerva_q8_8_t)NERVA_DIFF_UNRESOLVED_W_Q8_8;
    cfg->diff_confidence_w_q8_8 = (nerva_q8_8_t)NERVA_DIFF_CONFIDENCE_W_Q8_8;
    cfg->theta_compete_q8_8 = (nerva_q8_8_t)NERVA_THETA_COMPETE_Q8_8;
    cfg->ambiguity_inhibition_q0_16 = NERVA_AMBIGUITY_INHIBITION_Q0_16;
}

static inline NervaConfig nerva_config_default(void) {
    NervaConfig cfg;
    nerva_config_init_defaults(&cfg);
    return cfg;
}

static inline NervaConfig nerva_config_test(void) {
    NervaConfig cfg;
    nerva_config_init_defaults(&cfg);
    cfg.max_nodes = NERVA_TEST_MAX_NODES;
    cfg.max_edges = NERVA_TEST_MAX_EDGES;
    cfg.max_names = NERVA_TEST_MAX_NAMES;
    cfg.max_events = NERVA_TEST_MAX_EVENTS;
    cfg.max_active_nodes = NERVA_TEST_MAX_ACTIVE;
    cfg.max_fire_log = NERVA_TEST_FIRE_LOG;
    cfg.max_traces = NERVA_TEST_MAX_TRACES;
    cfg.trace_decay_scan_limit = NERVA_TEST_TRACE_DECAY_SCAN;
    cfg.max_mutations = NERVA_TEST_MAX_MUTATIONS;
    cfg.max_mutation_log = NERVA_TEST_MUTATION_LOG;
    cfg.max_expectations = NERVA_TEST_MAX_EXPECTATIONS;
    cfg.prediction_min_stability = 2u;
    cfg.max_schemas = NERVA_TEST_MAX_SCHEMAS;
    cfg.schema_support_threshold = (uint16_t)NERVA_TEST_SCHEMA_SUPPORT_THRESHOLD;
    cfg.max_memory_blocks = NERVA_TEST_MAX_MEMORY_BLOCKS;
    cfg.memory_store_threshold = NERVA_TEST_MEMORY_STORE_THRESHOLD;
    cfg.memory_forget_threshold = NERVA_TEST_MEMORY_FORGET_THRESHOLD;
    cfg.memory_hold_period_ticks = NERVA_TEST_MEMORY_HOLD_PERIOD_TICKS;
    cfg.idle_consolidate_ticks = NERVA_TEST_IDLE_CONSOLIDATE_TICKS;
    cfg.memory_decay_per_idle = NERVA_TEST_MEMORY_DECAY_PER_IDLE;
    cfg.memory_replay_top_k = (uint16_t)NERVA_TEST_MEMORY_REPLAY_TOP_K;
    cfg.fluid_threshold_base_q8_8 = (nerva_q8_8_t)NERVA_TEST_FLUID_THRESHOLD_BASE_Q8_8;
    return cfg;
}

#endif
