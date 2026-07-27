# nerva-train

Small **C** stack for train / teach / chat without backprop.

- **One product binary** — no second graph, no template replies  
- **Teach** with PCW on fluency weights (R-grad off)  
- **Reply** with real `fluency_generate` tokens  

This is the clean extract from the Nerva lab monorepo (probes, ERG ladder, and extra worlds stay in the lab).

## Build

```bash
make                 # build/nerva + build/pretrain
```

## TinyStories pretrain (once) → frozen checkpoint

Put full train file at `data/tinystories/TinyStories-train.txt` (or pass `--corpus`).

```bash
make pretrain
# writes:
#   checkpoints/tinystories.sess   model weights
#   checkpoints/tinystories.words  word lexicon
#   checkpoints/tinystories.meta   stats
```

Chat CLI **loads the checkpoint by default** and does **not** retrain:

```bash
./build/nerva                 # boot_mode=LOAD_CHECKPOINT
./build/nerva --selfcheck
```

Force tiny smoke train only (dev, not TinyStories):

```bash
./build/nerva --force-smoke-train
```

## Use

```text
./build/nerva --selfcheck
./build/nerva

you> /seq alice likes green tea
you> /gen alice 3
you> hello there
you> /quit
```

## Layout

```text
include/          engine + work headers
src/              engine + nerva_work
worlds/fluency/   fluency model + small train.txt
tools/nerva.c     single CLI
Makefile
```

## What this is not

- Not GPT-scale open chat (yet)  
- Not the full lab (no ERG probe suite, no TagWorld)  
- Lab session-chain toy lives only in the monorepo as `nerva_pcw_chat`

## Relationship to the lab

| Lab monorepo | This repo |
|--------------|-----------|
| Experiments, probes, seals | Product stack only |
| Many binaries | One: `nerva` |
| Edit when doing science | Edit when doing product / scale |

Port improvements **here first**, then optionally back into the lab.

## License

Apache-2.0 (see `LICENSE`).
