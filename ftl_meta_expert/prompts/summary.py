"""Executive synthesis prompt for meta expert."""


def build_summary_prompt(beliefs_text: str, domain: str, agent_stats: dict) -> str:
    """Build an executive summary prompt from combined beliefs."""
    return f"""\
You are synthesizing an executive summary across three expert knowledge bases for: {domain}

## Agent Statistics
- Code expert: {agent_stats.get('code', 0)} beliefs
- Project expert: {agent_stats.get('project', 0)} beliefs
- Product expert: {agent_stats.get('product', 0)} beliefs
- Cross-domain derived: {agent_stats.get('derived', 0)} beliefs
- Contradictions (nogoods): {agent_stats.get('nogoods', 0)}

## Combined Beliefs

{beliefs_text}

## Instructions

Synthesize a single executive summary that a CTO or VP Engineering reads in 5 minutes:

1. **System Overview** — What is this system? What does it do?
2. **Technical Health** — Code architecture strengths and weaknesses (from code expert)
3. **Delivery Health** — Team velocity, blockers, process quality (from project expert)
4. **Product Health** — Feature readiness, user experience, market fit (from product expert)
5. **Cross-Domain Tensions** — Where do the three perspectives conflict or reinforce each other?
6. **Nogoods & Contradictions** — What contradictions exist across domains? What do they mean?
7. **Top Risks** — The 5 most important risks, considering ALL three domains together
8. **Recommendations** — Top 5 actions that address cross-domain concerns

Be concrete. Reference specific belief IDs. Highlight where one domain's strength masks another domain's weakness.
"""
