"""Cross-domain contradiction detection prompt for meta expert."""

CONTRADICTION_DETECTION_PROMPT = """\
You are analyzing beliefs from three expert domains looking for cross-domain contradictions.

A contradiction (nogood) occurs when beliefs from different domains cannot ALL be true simultaneously.

## Types of Cross-Domain Contradictions

1. **Optimism mismatch**: Product says "feature X is ready" but Code shows "critical bugs in X"
2. **Capacity conflict**: Project says "team is at capacity" but Product expects "3 new features this quarter"
3. **Quality disagreement**: Code says "test coverage is comprehensive" but Project shows "regression rate increasing"
4. **Timeline contradiction**: Product says "launch in Q2" but Project shows "6 months of blockers"
5. **Architecture-strategy mismatch**: Code says "monolith is stable" but Product says "need microservices for scale"

## Beliefs from Expert Domains

{beliefs_section}

## Instructions

For each contradiction found, output EXACTLY this format:

### NOGOOD cross-domain-contradiction-id
- Claims: code:belief-id, product:belief-id
- Analysis: Why these cannot both be true
- Severity: High|Medium|Low
- Resolution: What needs to change to resolve the contradiction

Only report genuine contradictions where the claims are logically incompatible.
Do not report tensions or tradeoffs that can coexist — only true contradictions.
"""
