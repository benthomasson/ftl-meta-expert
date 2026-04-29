# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
uv tool install .                        # install CLI as `meta-expert`
uv pip install -e .                      # editable dev install
python -m spacy download en_core_web_md  # required for ask/generate-aliases
```

Prerequisites on PATH: `reasons` (`uv tool install ftl-reasons`), `entry` (`uv tool install ftl-entry`), `claude` CLI.

## Running

```bash
meta-expert init code=../code-expert project=../project-expert product=../product-expert
meta-expert import
meta-expert derive --auto
meta-expert contradictions --auto
meta-expert ask "Is the system ready to ship?"
meta-expert summary
meta-expert update                        # full pipeline: import→derive→contradictions→summary
meta-expert update --skip import           # skip a step (repeatable)
meta-expert status
```

All commands accept `--quiet`, `--model` (claude|gemini), and `--timeout`.

## Testing

No test suite exists yet. Verify changes by running commands against a real knowledge base.

## Architecture

This is a CLI tool that aggregates beliefs from multiple domain-expert repos (code, project, product) into a unified TMS (truth maintenance system) and enables cross-domain reasoning.

### Core flow

1. **Import** (`cli.py` → `reasons import-agent`) — pulls namespaced beliefs from expert repos into a shared `reasons.db`. Prefers `network.json` (lossless) over `beliefs.md` (lossy).
2. **Derive** (`cli.py` + `prompts/derive.py`) — LLM finds cross-domain conclusions requiring antecedents from 2+ agents. Proposals are parsed from structured markdown (DERIVE/GATE format) and applied via `reasons add`.
3. **Ask** (`belief_answer.py`) — spaCy hybrid scoring (60% semantic + 40% keyword overlap) against belief network. Falls back to LLM (`prompts/ask.py`) when no match exceeds threshold.
4. **Contradictions** (`cli.py` + `prompts/contradictions.py`) — LLM detects nogoods across domains, parsed as NOGOOD proposals and applied via `reasons nogood`.
5. **Summary** (`prompts/summary.py`) — LLM executive synthesis across all domains.

### Key modules

- **`cli.py`** — All Click commands plus proposal parsing (`_parse_derive_proposals`, `_parse_nogood_proposals`). Heavy module; both the CLI entry points and the core logic live here.
- **`belief_answer.py`** — spaCy-powered fast-path Q&A. Loads beliefs directly from `reasons.db` (SQLite). Supports pre-generated question aliases (`reasons.aliases.json`) for better matching.
- **`llm.py`** — Thin wrapper that shells out to `claude -p` or `gemini -p` via async subprocess. Strips `CLAUDECODE` env var to allow nested invocation.
- **`prompts/`** — Prompt templates. `common.py` has shared fragments (OUTPUT_FORMAT, TOPICS_INSTRUCTIONS). Each command's prompt is in its own file.
- **`topics.py`** — Investigation queue stored in `.meta-expert/topics.json`. Dataclass-based with parsing from LLM output via regex.

### Data layout (in a knowledge base repo)

- `.meta-expert/config.json` — experts list, ask thresholds, spaCy model name
- `reasons.db` — SQLite TMS database (source of truth)
- `beliefs.md` / `network.json` — exported views of the belief network
- `reasons.aliases.json` — pre-generated question aliases for spaCy matching
- `.meta-expert/topics.json` — investigation queue
- `entries/` — chronological documentation entries

### Belief ID convention

Beliefs are namespaced by agent: `code:belief-id`, `project:belief-id`, `product:belief-id`. Unprefixed IDs are cross-domain derived beliefs. IDs ending in `:active` are internal TMS markers, not user-facing beliefs.

### LLM integration pattern

All LLM calls go through `llm.invoke_sync()` which shells out to external CLIs. The `CLAUDECODE` env var is stripped to avoid recursion when running inside Claude Code. New models are added by extending `MODEL_COMMANDS` dict in `llm.py`.
