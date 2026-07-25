# nerva-train agent notes

- **Product only.** Do not add ERG probes, TagWorld, or a second chat graph.
- One CLI: `tools/nerva.c` → `build/nerva`.
- Generate path must stay `fluency_generate`. No template/`sprintf` product replies.
- Teach path: `fluency_pcw_teach_sequence` via `nerva_work` (R-grad off).
- Prefer small diffs. Port science from the lab monorepo only when proven.
