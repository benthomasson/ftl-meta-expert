"""Fast-path answering from belief networks using spaCy similarity.

Adapted from mini-analyze's belief_answer.py for cross-domain expert use.

Matches incoming questions against beliefs (from reasons.db) and returns
the belief text + justification as the answer when confidence is high enough.
No LLM call needed.

Supports question-form aliases: pre-generated question phrasings for each
belief that improve matching. Aliases are stored in reasons.aliases.json.
"""

import json
import re
import sqlite3
from pathlib import Path


# Lazy-loaded spaCy model
_spacy_nlp = None


def _get_spacy(model_name: str = "en_core_web_md"):
    """Lazy-load spaCy model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load(model_name)
    return _spacy_nlp


def _load_aliases(source_path: Path) -> dict[str, list[str]]:
    """Load question-form aliases for beliefs.

    Looks for reasons.aliases.json alongside reasons.db.
    """
    stem = source_path.stem
    aliases_path = source_path.parent / f"{stem}.aliases.json"
    if not aliases_path.exists():
        return {}

    with open(aliases_path) as f:
        data = json.load(f)

    # data is [{id: "...", aliases: ["...", ...]}, ...]
    return {item["id"]: item["aliases"] for item in data}


def _load_beliefs_from_reasons(db_path: Path) -> list[dict]:
    """Load beliefs from a reasons.db (TMS) database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    nodes = {}
    for row in conn.execute("SELECT id, text, truth_value FROM nodes"):
        nodes[row["id"]] = {
            "id": row["id"],
            "text": row["text"],
            "truth_value": row["truth_value"],
        }

    for row in conn.execute(
        "SELECT node_id, type, antecedents_json, outlist_json, label FROM justifications"
    ):
        node_id = row["node_id"]
        if node_id in nodes:
            if "justifications" not in nodes[node_id]:
                nodes[node_id]["justifications"] = []
            nodes[node_id]["justifications"].append({
                "type": row["type"],
                "antecedents": json.loads(row["antecedents_json"]),
                "outlist": json.loads(row["outlist_json"]),
                "label": row["label"] or "",
            })

    conn.close()
    return list(nodes.values())


def _agent_label(belief_id: str) -> str:
    """Extract agent domain label from a namespaced belief ID."""
    if ":" in belief_id:
        agent = belief_id.split(":")[0]
        return f"[{agent}]"
    return "[meta]"


def _format_belief_answer(belief: dict, score: float, nodes: dict | None = None) -> str:
    """Format a belief match as a human-readable answer."""
    status = "TRUE" if belief["truth_value"] == "IN" else "FALSE (blocked)"
    agent = _agent_label(belief["id"])
    lines = [
        f"{agent} **{belief['id']}**: {status}",
        "",
        belief["text"],
    ]

    justifications = belief.get("justifications", [])
    if justifications:
        for j in justifications:
            lines.append("")
            if j.get("label"):
                lines.append(f"**Justification:** {j['label']}")

            if j.get("antecedents") and nodes:
                lines.append("")
                lines.append("**Because (IN):**")
                for ant_id in j["antecedents"]:
                    ant = nodes.get(ant_id, {})
                    ant_status = "IN" if ant.get("truth_value") == "IN" else "OUT"
                    ant_text = ant.get("text", ant_id)
                    ant_agent = _agent_label(ant_id)
                    lines.append(f"  - [{ant_status}] {ant_agent} {ant_id}: {ant_text[:120]}")

            if j.get("outlist") and nodes:
                lines.append("")
                lines.append("**Blocked by (OUT list):**")
                for out_id in j["outlist"]:
                    out = nodes.get(out_id, {})
                    out_status = "IN" if out.get("truth_value") == "IN" else "OUT"
                    out_text = out.get("text", out_id)
                    out_agent = _agent_label(out_id)
                    lines.append(f"  - [{out_status}] {out_agent} {out_id}: {out_text[:120]}")

    lines.append("")
    lines.append(f"*(belief match: {score:.0%} confidence)*")

    return "\n".join(lines)


def answer_from_beliefs(
    question: str,
    belief_sources: list[str | Path] | None = None,
    threshold: float = 0.80,
    top_k: int = 3,
    verbose: bool = False,
) -> dict | None:
    """Try to answer a question directly from beliefs.

    Args:
        question: The question to answer.
        belief_sources: List of paths to reasons.db files.
            If None, uses reasons.db in the current directory.
        threshold: Minimum combined score to serve a belief as answer.
        top_k: Number of top matches to include in the answer.
        verbose: Print debug info.

    Returns:
        Dict with answer, beliefs, score, source — or None if no match above threshold.
    """
    import time
    t0 = time.time()

    nlp = _get_spacy()

    if belief_sources is None:
        belief_sources = [Path("reasons.db")]

    # Load all beliefs and aliases
    all_beliefs = []
    all_aliases = {}
    all_nodes = {}
    for src in belief_sources:
        src = Path(src)
        if not src.exists():
            continue

        src_aliases = _load_aliases(src)
        all_aliases.update(src_aliases)

        beliefs = _load_beliefs_from_reasons(src)
        nodes_lookup = {b["id"]: b for b in beliefs}
        all_nodes.update(nodes_lookup)
        all_beliefs.extend(beliefs)

    if not all_beliefs:
        return None

    # Score question against all beliefs
    query_doc = nlp(question.rstrip("?.! ").lower())
    query_lemmas = {t.lemma_ for t in query_doc if not t.is_stop and not t.is_punct}

    def _score_doc(belief_doc):
        """Score a single doc against the query."""
        belief_lemmas = {t.lemma_ for t in belief_doc if not t.is_stop and not t.is_punct}
        semantic = query_doc.similarity(belief_doc)
        if belief_lemmas:
            overlap = len(belief_lemmas & query_lemmas) / len(belief_lemmas)
            return 0.6 * semantic + 0.4 * overlap
        return semantic

    # Batch-process belief texts
    belief_texts = [b["text"].lower()[:500] for b in all_beliefs]
    belief_docs = list(nlp.pipe(belief_texts, batch_size=64))

    # Batch-process alias texts
    alias_entries = []
    for i, belief in enumerate(all_beliefs):
        for alias in all_aliases.get(belief["id"], []):
            alias_entries.append((i, alias.lower()))

    alias_docs = list(nlp.pipe([a[1] for a in alias_entries], batch_size=64)) if alias_entries else []

    # Build alias scores per belief index
    alias_best = {}
    for (bi, _), alias_doc in zip(alias_entries, alias_docs):
        score = _score_doc(alias_doc)
        if bi not in alias_best or score > alias_best[bi]:
            alias_best[bi] = score

    scored = []
    for i, (belief, belief_doc) in enumerate(zip(all_beliefs, belief_docs)):
        text_score = _score_doc(belief_doc)
        alias_score = alias_best.get(i, 0.0)

        # Cross-domain bonus: beliefs spanning multiple agents get a small boost
        combined = max(text_score, alias_score)
        if ":" not in belief.get("id", ""):
            # Unprefixed = cross-domain derived belief, small boost
            combined = min(1.0, combined + 0.03)

        scored.append((combined, belief))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] < threshold:
        if verbose:
            best = scored[0] if scored else (0, {"id": "none"})
            print(f"  Belief answer: no match above {threshold} "
                  f"(best: {best[1]['id']}={best[0]:.3f})", end="")
        return None

    # Build answer from top matches
    top_matches = [(score, b) for score, b in scored[:top_k] if score >= threshold]
    elapsed = time.time() - t0

    # Format answer
    answer_parts = []
    for score, belief in top_matches:
        answer_parts.append(_format_belief_answer(belief, score, all_nodes))

    answer = "\n---\n".join(answer_parts)
    best_score = top_matches[0][0]
    best_belief = top_matches[0][1]

    return {
        "answer": answer,
        "belief_id": best_belief["id"],
        "truth_value": best_belief["truth_value"],
        "score": round(best_score, 3),
        "num_matches": len(top_matches),
        "time": round(elapsed, 3),
        "source": "belief_network",
    }


def generate_aliases(source_path: str | Path, limit: int = 0) -> Path:
    """Generate question-form aliases for beliefs using Claude CLI.

    Args:
        source_path: Path to reasons.db.
        limit: Max beliefs to process (0 = all).

    Returns:
        Path to the generated aliases file.
    """
    import os
    import subprocess

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    beliefs = _load_beliefs_from_reasons(source_path)

    items = [{"id": b["id"], "text": b["text"]} for b in beliefs]
    if limit > 0:
        items = items[:limit]

    prompt = (
        "For each belief below, generate 3-5 natural questions that a person "
        "would ask if they wanted to know about this topic. Output valid JSON: "
        "an array of objects with 'id' (matching the input) and 'aliases' "
        "(array of question strings). Keep questions concise and natural. "
        "Output ONLY the JSON array, no markdown fences."
    )

    env = {**os.environ}
    env.pop("CLAUDECODE", None)

    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        input=json.dumps(items),
        capture_output=True, text=True, env=env,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr}")

    output = json.loads(result.stdout)
    if isinstance(output, dict) and "result" in output:
        text = output["result"]
    else:
        text = result.stdout
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    aliases = json.loads(text.strip())

    stem = source_path.stem
    aliases_path = source_path.parent / f"{stem}.aliases.json"
    with open(aliases_path, "w") as f:
        json.dump(aliases, f, indent=2)

    return aliases_path
