# nerva-train — product binary + TinyStories pretrain
CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -O2 -Iinclude -Iworlds/fluency
LDFLAGS ?=
LDLIBS ?= -lm
ifeq ($(OS),Windows_NT)
  LDLIBS += -lpsapi
  BIN = build/nerva.exe
  PRETRAIN = build/pretrain.exe
else
  BIN = build/nerva
  PRETRAIN = build/pretrain
endif

LIB_SRCS = \
	src/nerva_graph.c \
	src/nerva_engine.c \
	src/nerva_event.c \
	src/nerva_debug.c \
	src/nerva_trace.c \
	src/nerva_mutation.c \
	src/nerva_learning.c \
	src/nerva_prediction.c \
	src/nerva_exception.c \
	src/nerva_schema.c \
	src/nerva_memory.c \
	src/nerva_routing.c \
	src/nerva_parse.c \
	src/nerva_persist.c \
	src/nerva_bench.c \
	src/nerva_branch.c \
	src/nerva_work.c

LIB_OBJS = $(patsubst src/%.c,build/%.o,$(LIB_SRCS))
FLU_OBJS = build/fluency.o build/fluency_session.o

.PHONY: all product pretrain selfcheck clean checkpoints

all product: $(BIN) $(PRETRAIN)

build:
	mkdir -p build

checkpoints:
	mkdir -p checkpoints

build/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/fluency.o: worlds/fluency/fluency.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/fluency_session.o: worlds/fluency/fluency_session.c | build
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): tools/nerva.c $(LIB_OBJS) $(FLU_OBJS) | build
	$(CC) $(CFLAGS) tools/nerva.c $(LIB_OBJS) $(FLU_OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(PRETRAIN): tools/pretrain.c $(LIB_OBJS) $(FLU_OBJS) | build checkpoints
	$(CC) $(CFLAGS) tools/pretrain.c $(LIB_OBJS) $(FLU_OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

# Full TinyStories stream → checkpoints/tinystories.sess (long run)
pretrain: $(PRETRAIN)
	./$(PRETRAIN)

selfcheck: $(BIN)
	./$(BIN) --selfcheck

clean:
	rm -rf build
