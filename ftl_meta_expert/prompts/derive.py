"""Cross-domain derivation prompt for meta expert."""

from .common import OUTPUT_FORMAT


def build_derive_prompt(
    beliefs_by_agent: dict[str, list[dict]],
    derived_beliefs: list[dict],
    budget: int = 300,
) -> str:
    """Build a cross-domain derivation prompt.

    Args:
        beliefs_by_agent: Dict of agent_name -> list of belief dicts with id, text, truth_value.
        derived_beliefs: Existing cross-domain derived beliefs.
        budget: Max beliefs to include per agent.
    """
    # Build beliefs section grouped by agent
    sections = []
    total_in = 0
    for agent_name in sorted(beliefs_by_agent):
        beliefs = beliefs_by_agent[agent_name]
        in_beliefs = [b for b in beliefs if b["truth_value"] == "IN"]
        total_in += len(in_beliefs)

        # Proportional budget
        agent_budget = min(len(in_beliefs), budget)
        selected = in_beliefs[:agent_budget]
        omitted = len(in_beliefs) - len(selected)

        lines = [f"### {agent_name} expert ({len(in_beliefs)} IN beliefs)"]
        for b in selected:
            lines.append(f"- `{b['id']}`: {b['text'][:200]}")
        if omitted:
            lines.append(f"*({omitted} more beliefs omitted)*")
        sections.append("\n".join(lines))

    beliefs_section = "\n\n".join(sections)

    # Derived section
    if derived_beliefs:
        derived_lines = ["### Existing cross-domain derived beliefs"]
        for b in derived_beliefs:
            status = "IN" if b["truth_value"] == "IN" else "OUT"
            derived_lines.append(f"- [{status}] `{b['id']}`: {b['text'][:200]}")
        derived_section = "\n".join(derived_lines)
    else:
        derived_section = "*No cross-domain derived beliefs yet.*"

    statistics = f"Total IN beliefs across agents: {total_in}"

    return f"""\
You are a reasoning architect analyzing a belief network that spans three expert domains:
- **Code**: Architecture, patterns, invariants, and technical debt from codebase analysis
- **Project**: Team capacity, milestone health, process quality, and delivery risks from issue tracking
- **Product**: Feature readiness, user experience, competitive position, and product-market fit

Your task is to find emergent insights that ONLY become visible when combining knowledge across domains.

## Cross-Domain Derivation Patterns

1. **Code+Project**: Technical debt items (code) that explain delivery delays (project)
2. **Code+Product**: Architecture gaps (code) that block feature readiness (product)
3. **Project+Product**: Resource allocation (project) that impacts user-facing features (product)
4. **Code+Project+Product**: Full-stack insights connecting technical, delivery, and market realities
5. **Outlist-gated cross-domain**: Recovery paths GATE'd by domain-specific blockers

Examples:
- "Ship readiness" GATE'd by code:open-bugs AND project:unresolved-blockers
- "Tech debt impact on velocity" DERIVE'd from code:debt-in-X + project:velocity-declining-on-Y
- "Architecture supports product direction" GATE'd by code:missing-capability

## Rules

- Each proposed conclusion MUST reference antecedents from AT LEAST 2 different agents
- Antecedents must be existing belief IDs from the lists below
- Prefer insights that would be INVISIBLE to any single expert
- Don't force connections between unrelated beliefs
- Each conclusion should represent genuine cross-domain emergence

{OUTPUT_FORMAT}

---

## Current Beliefs

{beliefs_section}

## Existing Cross-Domain Derivations

{derived_section}

{statistics}
"""
