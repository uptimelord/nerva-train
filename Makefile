# nerva-train — single product binary
CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -O2 -Iinclude -Iworlds/fluency
LDFLAGS ?=
LDLIBS ?= -lm
ifeq ($(OS),Windows_NT)
  LDLIBS += -lpsapi
  BIN = build/nerva.exe
else
  BIN = build/nerva
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
FLU_OBJ = build/fluency.o

.PHONY: all product selfcheck clean

all product: $(BIN)

build:
	mkdir -p build

build/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@

$(FLU_OBJ): worlds/fluency/fluency.c | build
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): tools/nerva.c $(LIB_OBJS) $(FLU_OBJ) | build
	$(CC) $(CFLAGS) tools/nerva.c $(LIB_OBJS) $(FLU_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

selfcheck: $(BIN)
	./$(BIN) --selfcheck

clean:
	rm -rf build
