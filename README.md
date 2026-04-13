# ftl-meta-expert

Cross-domain expert aggregator — import, derive, and query across expert knowledge bases.

Sits above multiple expert knowledge bases. Uses ftl-reasons (TMS) for multi-agent belief tracking and spaCy semantic matching for fast question answering.

## Install

```bash
uv tool install .
python -m spacy download en_core_web_md
```

Prerequisites: `reasons` (`uv tool install ftl-reasons`), `entry` (`uv tool install ftl-entry`), `claude` CLI.

## Quick Start

```bash
# In your meta-expert knowledge base repo:
meta-expert init code=../code-expert project=../project-expert product=../product-expert
meta-expert import                         # import from all experts
meta-expert derive --auto                  # cross-domain derivation
meta-expert contradictions --auto          # find nogoods across domains
meta-expert generate-aliases               # build aliases for fast matching
meta-expert ask "Is the system ready to ship?"  # query combined beliefs
meta-expert summary                        # executive synthesis
```

## Commands

| Command | Purpose |
|---------|---------|
| `init` | Bootstrap meta-expert knowledge base |
| `import [--expert NAME]` | Import beliefs from expert repos |
| `derive [--auto] [--dry-run]` | Cross-domain derivation |
| `ask "question"` | Query beliefs (spaCy fast path + LLM fallback) |
| `contradictions [--auto]` | Detect cross-domain nogoods |
| `summary` | Executive synthesis across all domains |
| `generate-aliases` | Build question aliases for better matching |
| `status` | Dashboard with per-agent stats |
| `topics` | Cross-domain investigation queue |

## How It Works

1. **Import** pulls beliefs from expert repos via `reasons import-agent`, creating namespaced beliefs (`code:*`, `project:*`, `product:*`)
2. **Derive** uses an LLM to find cross-domain conclusions that require antecedents from multiple agents
3. **Ask** matches questions against the combined belief network using spaCy hybrid scoring (60% semantic + 40% keyword), falling through to LLM for novel questions
4. **Contradictions** uses an LLM to identify nogoods where beliefs from different domains are logically incompatible
5. **Summary** produces an executive synthesis considering all three expert perspectives
