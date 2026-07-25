# nerva-train

Small **C** stack for train / teach / chat without backprop.

- **One product binary** — no second graph, no template replies  
- **Teach** with PCW on fluency weights (R-grad off)  
- **Reply** with real `fluency_generate` tokens  

This is the clean extract from the Nerva lab monorepo (probes, ERG ladder, and extra worlds stay in the lab).

## Build

**Linux / MinGW:**
```bash
make
./build/nerva
```

**Windows (PowerShell, with gcc on PATH or use lab’s toolchain path):**
```powershell
# from this repo root
make
.\build\nerva.exe
```

## Use

```text
./build/nerva --selfcheck          # automated PASS/FAIL
./build/nerva                      # interactive

you> /seq alice likes green tea    # PCW-teach multi-word sequence
you> /gen alice 3                  # LM generate
you> hello there                   # free text → same generate path
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
