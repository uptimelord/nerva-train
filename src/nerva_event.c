// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Paul Odantabao II

#include "nerva_event.h"
#include "nerva_debug.h"
#include "nerva_graph.h"
#include "nerva_math.h"
#include "nerva_trace.h"
#include "nerva_prediction.h"

#include <stdlib.h>
#include <string.h>

static void heap_swap(NervaEvent *a, NervaEvent *b) {
    NervaEvent tmp = *a;
    *a = *b;
    *b = tmp;
}

static void heap_sift_up(NervaEvent *heap, uint32_t idx) {
    while (idx > 0) {
        uint32_t parent = (idx - 1) / 2;
        if (heap[parent].due_tick <= heap[idx].due_tick) {
            break;
        }
        heap_swap(&heap[parent], &heap[idx]);
        idx = parent;
    }
}

static void heap_sift_down(NervaEvent *heap, uint32_t count, uint32_t idx) {
    for (;;) {
        uint32_t left = idx * 2 + 1;
        uint32_t right = left + 1;
        uint32_t smallest = idx;
        if (left < count && heap[left].due_tick < heap[smallest].due_tick) {
            smallest = left;
        }
        if (right < count && heap[right].due_tick < heap[smallest].due_tick) {
            smallest = right;
        }
        if (smallest == idx) {
            break;
        }
        heap_swap(&heap[idx], &heap[smallest]);
        idx = smallest;
    }
}

static void heap_insert(NervaEvent *heap, uint32_t *count, NervaEvent ev) {
    heap[*count] = ev;
    heap_sift_up(heap, *count);
    (*count)++;
}

static NervaEvent heap_remove_min(NervaEvent *heap, uint32_t *count) {
    NervaEvent min = heap[0];
    (*count)--;
    if (*count > 0) {
        heap[0] = heap[*count];
        heap_sift_down(heap, *count, 0);
    }
    return min;
}

static void heap_fix_position(NervaEvent *heap, uint32_t count, uint32_t idx) {
    heap_sift_up(heap, idx);
    heap_sift_down(heap, count, idx);
}

static uint32_t nerva_find_merge_candidate(NervaEngine *e, const NervaEvent *ev) {
    for (uint32_t i = 0; i < e->event_count; ++i) {
        NervaEvent *cur = &e->events[i];
        if (cur->target != ev->target) {
            continue;
        }
        if (cur->source != ev->source) {
            continue;
        }
        if (cur->type_flags != ev->type_flags) {
            continue;
        }
        if (cur->relation != ev->relation) {
            continue;
        }
        if (cur->due_tick > ev->due_tick + e->cfg.merge_window_ticks) {
            continue;
        }
        if (ev->due_tick > cur->due_tick + e->cfg.merge_window_ticks) {
            continue;
        }
        return i;
    }
    return NERVA_INVALID_ID;
}

bool nerva_event_overflow_admit(NervaEngine *e, NervaEvent ev) {
    if (!e || e->event_count == 0) {
        return false;
    }

    uint32_t weakest = 0;
    int32_t weakest_abs = nerva_abs32(e->events[0].signal);
    for (uint32_t i = 1; i < e->event_count; ++i) {
        int32_t abs_sig = nerva_abs32(e->events[i].signal);
        if (abs_sig < weakest_abs) {
            weakest_abs = abs_sig;
            weakest = i;
        }
    }
    if (nerva_abs32(ev.signal) <= weakest_abs) {
        return false;
    }

    e->events[weakest] = ev;
    heap_fix_position(e->events, e->event_count, weakest);
    return true;
}

bool nerva_event_push(NervaEngine *e, NervaEvent ev) {
    if (!e) {
        return false;
    }
    if (ev.due_tick + e->cfg.stale_ticks < e->tick) {
        e->debug.events_stale_dropped++;
        return false;
    }
    if (e->event_count >= e->event_cap) {
        if (!nerva_event_overflow_admit(e, ev)) {
            e->debug.events_overflow_dropped++;
            return false;
        }
        e->debug.events_overflow_admitted++;
        return true;
    }

    heap_insert(e->events, &e->event_count, ev);
    e->debug.events_pushed++;
    if (e->event_count > e->debug.event_depth_max) {
        e->debug.event_depth_max = e->event_count;
    }
    return true;
}

bool nerva_event_pop(NervaEngine *e, NervaEvent *out) {
    if (!e || !out) {
        return false;
    }

    while (e->event_count > 0) {
        if (e->events[0].due_tick > e->tick) {
            return false;
        }
        *out = heap_remove_min(e->events, &e->event_count);
        if (out->due_tick + e->cfg.stale_ticks < e->tick) {
            e->debug.events_stale_dropped++;
            continue;
        }
        e->debug.events_popped++;
        return true;
    }
    return false;
}

bool nerva_event_merge_or_push(NervaEngine *e, NervaEvent ev) {
    uint32_t i = nerva_find_merge_candidate(e, &ev);
    if (i != NERVA_INVALID_ID) {
        e->events[i].signal = nerva_q8_8_saturating_add(e->events[i].signal, ev.signal);
        if (ev.due_tick < e->events[i].due_tick) {
            e->events[i].due_tick = ev.due_tick;
        }
        e->debug.events_merged++;
        heap_fix_position(e->events, e->event_count, i);
        return true;
    }
    return nerva_event_push(e, ev);
}

void nerva_apply_event_to_node(NervaEngine *e, NervaNode *n, const NervaEvent *ev) {
    (void)e;
    n->v = nerva_q8_8_saturating_add(n->v, ev->signal);
    n->path_tag = ev->trace_tag;
}

static int nerva_node_refractory_active(const NervaEngine *e, const NervaNode *n) {
    if (!e || !n || n->refractory_max == 0 || n->activation_count == 0) {
        return false;
    }
    return (e->tick - n->last_fired_tick) < (nerva_tick_t)n->refractory_max;
}

uint32_t nerva_node_refractory_remaining(const NervaEngine *e, const NervaNode *n) {
    if (!nerva_node_refractory_active(e, n)) {
        return 0;
    }
    return (uint32_t)(n->refractory_max - (e->tick - n->last_fired_tick));
}

bool nerva_node_should_fire(const NervaEngine *e, const NervaNode *n) {
    return !(n->flags & NERVA_NODE_DELETED) &&
           !nerva_node_refractory_active(e, n) &&
           n->v >= n->theta_fire;
}

void nerva_fire_node(NervaEngine *e, uint32_t node_id) {
    if (!e || node_id >= e->node_count) {
        return;
    }

    NervaNode *n = &e->nodes[node_id];
    n->last_fired_tick = e->tick;
    n->activation_count++;
    n->v = n->v_reset;
    e->debug.tick_fired++;
    nerva_debug_log_fire(e, node_id);

    if (!e->adjacency_valid) {
        return;
    }

    if (e->prediction_mode) {
        nerva_prediction_on_fire(e, node_id);
        return;
    }

    for (uint32_t k = 0; k < n->out_count; ++k) {
        uint32_t slot = n->first_out + k;
        if (slot >= e->edge_count) {
            break;
        }

        uint32_t edge_id = e->sorted_edges[slot];
        NervaEdge *ed = &e->edges[edge_id];
        if (ed->flags & NERVA_EDGE_DELETED) {
            continue;
        }

        nerva_q8_8_t sig = nerva_compute_edge_signal(e, node_id, ed);
        if (sig == 0) {
            continue;
        }

        NervaEvent out;
        memset(&out, 0, sizeof(out));
        out.due_tick = e->tick + ed->delay_ticks;
        out.source = ed->source;
        out.target = ed->target;
        out.edge_id = edge_id;
        out.signal = sig;
        out.relation = ed->relation;
        if (ed->trace_tag == 0) {
            ed->trace_tag = nerva_make_path_tag(e, edge_id);
        }
        out.trace_tag = ed->trace_tag;
        out.type_flags = NERVA_EVT_ACTIVATION;
        nerva_event_merge_or_push(e, out);

        if (ed->stability < UINT16_MAX) {
            ed->stability++;
        }
        ed->last_active_tick32 = (uint32_t)e->tick;
    }
}

nerva_q8_8_t nerva_compute_edge_signal(const NervaEngine *e, uint32_t source,
                                       const NervaEdge *ed) {
    (void)source;
    int32_t x = e->cfg.default_output_q8_8;
    x = (x * (int32_t)ed->weight) >> 8;
    x = (x * (int32_t)ed->gate + 32768) >> 16;
    if (ed->gate_ctrl != NERVA_INVALID_ID && ed->gate_ctrl < e->node_count) {
        /* Conditional gate: modulate by control (role-register) charge.
         * <=0 closes the gate; [0,ONE) softly attenuates; >=ONE passes fully. */
        int32_t ctrl_v = (int32_t)e->nodes[ed->gate_ctrl].v;
        if (ctrl_v <= 0) {
            return 0;
        }
        if (ctrl_v < NERVA_Q8_8_ONE) {
            x = (x * ctrl_v) / NERVA_Q8_8_ONE;
        }
    }
    if (ed->flags & (NERVA_EDGE_INHIBITORY | NERVA_EDGE_BLOCKER)) {
        x = -nerva_abs32(x);
    }
    return nerva_q8_8_clip(x);
}

void nerva_apply_leak(NervaEngine *e, NervaNode *n) {
    int32_t delta = (int32_t)n->v - (int32_t)n->v_rest;
    n->v = (nerva_q8_8_t)((int32_t)n->v - (delta >> (int)e->cfg.leak_shift));
}

void nerva_mark_active(NervaEngine *e, uint32_t node_id) {
    for (uint32_t i = 0; i < e->active_count; ++i) {
        if (e->active_nodes[i] == node_id) {
            return;
        }
    }
    if (e->active_count < e->active_cap) {
        e->active_nodes[e->active_count++] = node_id;
    }
}

bool nerva_activate_node(NervaEngine *e, uint32_t node_id, nerva_q8_8_t signal) {
    if (!e || node_id >= e->node_count) {
        return false;
    }

    e->last_query_start_tick = e->tick;
    e->active_query_tag = (uint32_t)((e->tick << 16) ^ (node_id + 1u));
    nerva_prediction_clear(e);

    NervaEvent ev;
    memset(&ev, 0, sizeof(ev));
    ev.due_tick = e->tick;
    ev.source = NERVA_INVALID_ID;
    ev.target = node_id;
    ev.edge_id = NERVA_INVALID_ID;
    ev.signal = signal;
    ev.type_flags = NERVA_EVT_ACTIVATION;
    return nerva_event_push(e, ev);
}
