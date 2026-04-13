"""LLM fallback prompt for unmatched questions."""

ASK_FALLBACK_PROMPT = """\
You are answering a question using a combined belief network from three expert domains \
(code, project, product) about: {domain}

## Question
{question}

## Closest Matching Beliefs (below threshold, provided as context)
{closest_beliefs}

## Full Network Summary
{compact_summary}

## Instructions

Answer the question using the belief network as your primary source of truth.
- If the beliefs directly address the question, synthesize an answer from them.
- If the beliefs partially address it, say what you know and what's missing.
- If the beliefs don't address it at all, say so honestly.
- Always cite specific belief IDs when making claims.
- Indicate which expert domain (code/project/product) each piece of evidence comes from.
"""
