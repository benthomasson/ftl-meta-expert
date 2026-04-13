"""Shared prompt fragments for meta expert."""

TOPICS_INSTRUCTIONS = """\
## Topics to Explore

If your analysis reveals cross-domain tensions or gaps worth investigating,
list them below using this exact format:

- [cross-domain] `target-slug` — Description of what to investigate
- [contradiction] `belief-a vs belief-b` — Why these beliefs conflict
- [derivation] `potential-conclusion` — What could be derived from combining X and Y
- [general] `topic-slug` — General investigation topic
"""

BELIEFS_INSTRUCTIONS = """\
## Beliefs

If your analysis reveals new cross-domain beliefs, list them below:

- `belief-id-in-kebab-case` — One-line factual claim derived from multiple expert domains
"""

OUTPUT_FORMAT = """\
## Output Format

For each proposal, use EXACTLY this format:

### DERIVE belief-id-in-kebab-case
One-line conclusion text
- Antecedents: agent-a:belief-id, agent-b:belief-id
- Label: Why this conclusion follows from the antecedents

### GATE belief-id-in-kebab-case
One-line positive claim (what would be true if blockers were absent)
- Antecedents: agent-a:belief-id, agent-b:belief-id
- Unless: agent-c:blocker-belief-id
- Label: Why this is gated on the blocker being OUT
"""
