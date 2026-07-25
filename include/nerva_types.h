// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#ifndef NERVA_TYPES_H
#define NERVA_TYPES_H

#include <stdint.h>

typedef int16_t nerva_q8_8_t;
typedef uint16_t nerva_uq0_16_t;
typedef uint64_t nerva_tick_t;

typedef enum NervaRelationBuiltin {
    NERVA_REL_NONE = 0,
    NERVA_REL_KIND_OF,
    NERVA_REL_USUALLY_HAS_PROPERTY,
    NERVA_REL_BLOCKS,
    NERVA_REL_INSIDE,
    NERVA_REL_MOVED_TO,
    NERVA_REL_LOCATED_AT,
    NERVA_REL_CUSTOM_BASE = 1024
} NervaRelationBuiltin;

enum {
    NERVA_NODE_DELETED = 1u << 0,
    NERVA_EDGE_DELETED = 1u << 0,
    NERVA_EDGE_INHIBITORY = 1u << 1,
    NERVA_EDGE_BLOCKER = 1u << 2
};

enum {
    NERVA_TRACE_USED_PATH = 1u << 0,
    NERVA_TRACE_DECAYED = 1u << 1,
    NERVA_TRACE_CORRECT = 1u << 2,
    NERVA_TRACE_WRONG = 1u << 3,
    NERVA_TRACE_EXPECTED = 1u << 4,
    NERVA_TRACE_PRED_CONFIRMED = 1u << 5,
    NERVA_TRACE_PRED_MISSED = 1u << 6,
    NERVA_TRACE_SURPRISE = 1u << 7,
    NERVA_TRACE_BLOCKER = 1u << 8,
    NERVA_TRACE_EXCEPTION = 1u << 9,
    NERVA_TRACE_GATED = 1u << 10
};

enum {
    NERVA_MUT_CREATE_EDGE = 1,
    NERVA_MUT_MODIFY_WEIGHT = 2,
    NERVA_MUT_MODIFY_GATE = 3,
    NERVA_MUT_ADD_BLOCKER_EDGE = 4,
    NERVA_MUT_DELETE_EDGE = 5,
    NERVA_MUT_PROMOTE_PROVISIONAL_EDGE = 6,
    NERVA_MUT_MARK_CONTRADICTION = 7
};

enum {
    NERVA_REASON_FEEDBACK_CORRECT = 1,
    NERVA_REASON_FEEDBACK_WRONG = 2,
    NERVA_REASON_PREDICTION_CONFIRMED = 3,
    NERVA_REASON_PREDICTION_MISSED = 4,
    NERVA_REASON_EXCEPTION_BLOCKER = 5,
    NERVA_REASON_SCHEMA_APPLY = 6,
    NERVA_REASON_HEBBIAN_COFIRE = 7,
    NERVA_REASON_MEMORY_WRITE = 8
};

enum {
    NERVA_SCHEMA_CANDIDATE = 1u << 0,
    NERVA_SCHEMA_PROMOTED = 1u << 1
};

enum {
    NERVA_EXP_PENDING = 1u << 0
};

enum {
    NERVA_MEM_TYPE_EPISODIC = 1u
};

enum {
    NERVA_MEM_STATE_PENDING = 1u << 0,
    NERVA_MEM_STATE_CONSOLIDATED = 1u << 1,
    NERVA_MEM_STATE_MARK_DELETE = 1u << 2,
    NERVA_MEM_STATE_OPEN = 1u << 3
};

#define NERVA_MEM_LOW_CHARGE_UNSET UINT64_MAX

enum {
    NERVA_MEM_FLAG_USEFUL = 1u << 0,
    NERVA_MEM_FLAG_REPLAYED = 1u << 1
};

#define NERVA_GATE_UNCHANGED 65535u

typedef struct NervaNode {
    uint32_t id;
    uint32_t name_id;
    uint32_t first_out;
    uint32_t out_count;

    nerva_q8_8_t v;
    nerva_q8_8_t v_rest;
    nerva_q8_8_t v_reset;
    nerva_q8_8_t theta_fire;

    uint16_t refractory_max;
    uint16_t flags;

    float memory_charge;
    uint32_t activation_count;
    nerva_tick_t last_fired_tick;

    uint32_t trace_head;
    uint32_t path_tag;
    uint32_t memory_block;
    uint32_t first_blocker_in;
    uint16_t blocker_in_count;
    uint16_t _pad_node;
} NervaNode;

typedef struct NervaEdge {
    uint32_t source;
    uint32_t target;
    uint16_t relation;
    uint16_t flags;

    nerva_q8_8_t weight;
    nerva_uq0_16_t gate;
    uint16_t delay_ticks;
    uint16_t stability;
    uint16_t wrong_feedback_count;

    uint32_t trace_tag;
    uint32_t memory_block;
    uint32_t last_active_tick32;

    /* Conditional gate (Stage 0e): when != NERVA_INVALID_ID, this edge's signal
     * is modulated by the charge of control node gate_ctrl (a role register).
     * NERVA_INVALID_ID means unconditional (legacy behavior). */
    uint32_t gate_ctrl;
} NervaEdge;

typedef struct NervaEvent {
    nerva_tick_t due_tick;
    uint32_t source;
    uint32_t target;
    uint32_t edge_id;

    nerva_q8_8_t signal;
    uint16_t relation;
    uint32_t trace_tag;
    uint16_t type_flags;
    uint16_t generation;
} NervaEvent;

typedef struct NervaFireRecord {
    nerva_tick_t tick;
    uint32_t node_id;
} NervaFireRecord;

typedef struct NervaTrace {
    nerva_tick_t tick;
    uint32_t source;
    uint32_t target;
    uint32_t edge_id;
    uint32_t trace_tag;
    uint32_t query_tag;

    nerva_q8_8_t signal;
    nerva_uq0_16_t pre;
    nerva_uq0_16_t post;
    uint16_t relation;
    uint16_t flags;

    nerva_q8_8_t target_v_before;
    nerva_q8_8_t target_v_after;
    uint16_t fired_after_apply;
    uint16_t _pad;
} NervaTrace;

typedef struct NervaExpectation {
    nerva_tick_t tick;
    uint32_t source;
    uint32_t target;
    uint32_t edge_id;
    uint32_t query_tag;
    uint32_t trace_tag;
    uint16_t flags;
    uint16_t relation;
} NervaExpectation;

typedef struct NervaMutation {
    nerva_tick_t tick;
    uint16_t type;
    uint16_t flags;
    uint32_t edge_id;

    uint32_t source;
    uint32_t target;
    uint16_t relation;
    uint16_t edge_flags;

    nerva_q8_8_t delta_weight;
    nerva_uq0_16_t new_gate;
    uint32_t trace_tag;
    uint32_t debug_reason;
    uint32_t _pad;
} NervaMutation;

typedef struct NervaMutationRecord {
    nerva_tick_t tick;
    uint16_t type;
    uint16_t _pad;
    uint32_t edge_id;
    nerva_q8_8_t old_weight;
    nerva_q8_8_t new_weight;
    nerva_uq0_16_t old_gate;
    nerva_uq0_16_t new_gate;
    uint32_t debug_reason;
} NervaMutationRecord;

#define NERVA_SCHEMA_DISTINCT_CAP 16u

typedef struct NervaSchemaDistinct {
    uint32_t a;
    uint32_t b;
    uint32_t c;
} NervaSchemaDistinct;

typedef struct NervaSchema {
    uint32_t id;
    uint16_t flags;
    uint16_t rel_a;
    uint16_t rel_b;
    uint16_t rel_out;
    uint32_t support_count;
    uint32_t distinct_count;
    uint32_t raw_hop_cost;
    uint32_t schema_edge_cost;
    uint32_t exception_cost;
    uint32_t exception_count;
    uint32_t correct_predictions;
    uint32_t total_predictions;
    nerva_tick_t promoted_tick;
    NervaSchemaDistinct distinct[NERVA_SCHEMA_DISTINCT_CAP];
} NervaSchema;

typedef struct NervaMemoryBlock {
    uint32_t id;
    uint8_t type;
    uint8_t state;
    uint16_t flags;
    float charge;
    nerva_tick_t created_tick;
    nerva_tick_t last_access_tick;
    nerva_tick_t low_charge_since;
    uint32_t query_tag;
    uint32_t trace_count;
} NervaMemoryBlock;

typedef struct NervaDebugCounters {
    uint64_t events_pushed;
    uint64_t events_popped;
    uint64_t events_stale_dropped;
    uint64_t events_merged;
    uint64_t events_overflow_dropped;
    uint64_t events_overflow_admitted;
    uint32_t event_depth_max;
    uint32_t tick_events;
    uint32_t tick_fired;
    uint64_t total_fired;
    uint64_t traces_recorded;
    uint64_t mutations_queued;
    uint64_t mutations_applied;
    uint64_t mutations_overflow;
    uint64_t predictions_emitted;
    uint64_t predictions_confirmed;
    uint64_t predictions_missed;
    uint64_t blockers_applied;
    uint64_t exceptions_suppressed;
    uint64_t schemas_promoted;
    uint64_t schemas_applied;
    uint64_t memory_consolidations;
    uint64_t memory_replayed;
    uint64_t memory_forgotten;
    uint64_t fluid_activations;
    uint64_t crystallized_queries;
    uint64_t fluid_workspace_steps;
} NervaDebugCounters;

typedef struct NervaConfig {
    uint32_t max_nodes;
    uint32_t max_edges;
    uint32_t max_names;
    uint32_t max_events;
    uint32_t max_active_nodes;
    uint32_t max_fire_log;
    uint32_t max_traces;
    uint32_t max_mutations;
    uint32_t max_mutation_log;

    nerva_q8_8_t theta_fire_q8_8;
    nerva_q8_8_t v_rest_q8_8;
    nerva_q8_8_t v_reset_q8_8;
    nerva_q8_8_t default_output_q8_8;

    uint32_t leak_shift;
    uint16_t refractory_ticks;
    uint16_t edge_delay_ticks;

    uint32_t stale_ticks;
    uint32_t merge_window_ticks;

    nerva_uq0_16_t trace_pre_decay_q0_16;
    nerva_uq0_16_t trace_post_decay_q0_16;
    uint32_t trace_window_ticks;
    uint32_t trace_decay_scan_limit;

    nerva_q8_8_t weight_min_q8_8;
    nerva_q8_8_t weight_max_q8_8;
    nerva_q8_8_t default_weight_q8_8;
    nerva_q8_8_t ltp_delta_q8_8;
    nerva_q8_8_t ltd_delta_q8_8;
    nerva_uq0_16_t gate_close_step_q0_16;
    uint16_t feedback_wrong_gate_threshold;
    uint16_t prediction_min_stability;

    nerva_q8_8_t prediction_pre_charge_q8_8;
    uint32_t prediction_window_ticks;
    uint32_t max_expectations;

    nerva_uq0_16_t suppress_q0_16;

    uint32_t max_schemas;
    uint16_t schema_support_threshold;
    uint16_t schema_exception_limit;

    uint32_t max_memory_blocks;
    float memory_store_threshold;
    float memory_forget_threshold;
    float memory_decay_per_idle;
    uint32_t memory_hold_period_ticks;
    uint32_t idle_consolidate_ticks;
    uint16_t memory_replay_top_k;

    nerva_q8_8_t fluid_threshold_base_q8_8;
    nerva_q8_8_t fatigue_increment_q8_8;
    uint32_t fatigue_decay_shift;
    nerva_q8_8_t diff_novelty_w_q8_8;
    nerva_q8_8_t diff_contradiction_w_q8_8;
    nerva_q8_8_t diff_unresolved_w_q8_8;
    nerva_q8_8_t diff_confidence_w_q8_8;
    nerva_q8_8_t theta_compete_q8_8;
    nerva_uq0_16_t ambiguity_inhibition_q0_16;
} NervaConfig;

#define NERVA_FLUID_ACTIVE_MAX 4u

typedef struct NervaRouterState {
    nerva_q8_8_t difficulty_q8_8;
    nerva_q8_8_t fluid_threshold_q8_8;
    nerva_q8_8_t confidence_q8_8;
    uint16_t novelty_count;
    uint16_t contradiction_count;
    uint16_t unresolved_constraints;
    uint32_t query_source;
    uint32_t query_target;
    uint16_t query_relation;
    uint32_t fluid_nodes[NERVA_FLUID_ACTIVE_MAX];
    uint32_t fluid_count;
    uint8_t query_active;
    uint8_t query_routing_decided;
    uint8_t fluid_active;
    uint8_t crystallized;
} NervaRouterState;

typedef struct NervaEngine {
    NervaConfig cfg;
    nerva_tick_t tick;

    NervaNode *nodes;
    uint32_t node_count;
    uint32_t node_cap;

    NervaEdge *edges;
    uint32_t edge_count;
    uint32_t edge_cap;

    uint32_t *sorted_edges;
    uint32_t *blocker_in_edges;
    uint32_t blocker_in_count;

    NervaEvent *events;
    uint32_t event_count;
    uint32_t event_cap;

    uint32_t *active_nodes;
    uint32_t active_count;
    uint32_t active_cap;

    NervaFireRecord *fire_log;
    uint32_t fire_log_count;
    uint32_t fire_log_cap;

    NervaTrace *traces;
    uint32_t trace_head;
    uint32_t trace_count;
    uint32_t trace_cap;

    NervaMutation *mutations;
    uint32_t mutation_head;
    uint32_t mutation_tail;
    uint32_t mutation_count;
    uint32_t mutation_cap;

    NervaMutationRecord *mutation_log;
    uint32_t mutation_log_count;
    uint32_t mutation_log_cap;

    NervaDebugCounters debug;

    NervaRouterState router;

    char **names;
    uint32_t name_count;
    uint32_t name_cap;

    int adjacency_valid;

    nerva_tick_t last_query_start_tick;
    uint32_t active_query_tag;

    NervaExpectation *expectations;
    uint32_t expectation_count;
    uint32_t expectation_cap;
    int prediction_mode;

    NervaSchema *schemas;
    uint32_t schema_count;
    uint32_t schema_cap;

    NervaMemoryBlock *memory;
    uint32_t memory_count;
    uint32_t memory_cap;
    uint32_t idle_ticks;
} NervaEngine;

#endif
