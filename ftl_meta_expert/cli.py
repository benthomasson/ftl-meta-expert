"""Command-line interface for meta expert."""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import click

from .llm import check_model_available, invoke, invoke_sync
from .prompts import (
    ASK_FALLBACK_PROMPT,
    CONTRADICTION_DETECTION_PROMPT,
    build_derive_prompt,
    build_summary_prompt,
)
from .topics import (
    add_topics,
    load_queue,
    parse_topics_from_response,
    pending_count,
)

PROJECT_DIR = ".meta-expert"

DEFAULT_EXPERTS: list[dict] = []


# --- Config helpers ---


def _load_config() -> dict | None:
    """Load .meta-expert/config.json if it exists."""
    config_path = Path.cwd() / PROJECT_DIR / "config.json"
    if config_path.is_file():
        return json.loads(config_path.read_text())
    return None


def _save_config(config: dict) -> None:
    """Save config to .meta-expert/config.json."""
    config_dir = Path.cwd() / PROJECT_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(config, indent=2))


def _require_config() -> dict:
    """Load config or exit with error."""
    config = _load_config()
    if config is None:
        click.echo("Error: not a meta-expert knowledge base", err=True)
        click.echo("Run: meta-expert init", err=True)
        sys.exit(1)
    return config


def _has_reasons() -> bool:
    """Check if reasons CLI is available."""
    return shutil.which("reasons") is not None


def _has_entry() -> bool:
    """Check if entry CLI is available."""
    return shutil.which("entry") is not None


# --- Output helpers ---


def _emit(ctx, text: str) -> None:
    """Print to stdout unless --quiet."""
    if not ctx.obj.get("quiet"):
        click.echo(text)


def _create_entry(topic: str, title: str, content: str) -> None:
    """Create an entry via the entry CLI."""
    try:
        result = subprocess.run(
            ["entry", "create", topic, title, "--content", content],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            click.echo(f"Entry: {result.stdout.strip()}", err=True)
        else:
            result = subprocess.run(
                ["entry", "create", topic, title],
                input=content,
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                click.echo(f"Entry: {result.stdout.strip()}", err=True)
    except FileNotFoundError:
        pass


def _reasons_export() -> None:
    """Re-export beliefs.md and network.json from reasons.db."""
    subprocess.run(
        ["reasons", "export-markdown", "-o", "beliefs.md"],
        capture_output=True,
    )
    with open("network.json", "w") as f:
        subprocess.run(["reasons", "export"], stdout=f, capture_output=False)


def _load_network_json() -> dict:
    """Load network from reasons export (JSON)."""
    result = subprocess.run(
        ["reasons", "export"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo(f"Error: reasons export failed: {result.stderr}", err=True)
        sys.exit(1)
    return json.loads(result.stdout)


def _get_compact(budget: int = 2000) -> str:
    """Get token-budgeted compact summary from reasons."""
    result = subprocess.run(
        ["reasons", "compact", "--budget", str(budget)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return "(compact unavailable)"
    return result.stdout


# --- Proposal parsing (adapted from reasons_lib.derive) ---


def _parse_derive_proposals(response: str) -> list[dict]:
    """Parse DERIVE and GATE proposals from LLM response."""
    proposals = []
    pattern = re.compile(
        r"###\s+(DERIVE|GATE)\s+(\S+)\s*\n"
        r"(.+?)(?=\n###\s+(?:DERIVE|GATE)|\Z)",
        re.DOTALL,
    )

    for match in pattern.finditer(response):
        kind = match.group(1).lower()
        belief_id = match.group(2)
        body = match.group(3).strip()

        # Extract text (first line)
        lines = body.split("\n")
        text = lines[0].strip()

        # Extract antecedents
        antecedents = []
        outlist = []
        label = ""
        for line in lines[1:]:
            line = line.strip().lstrip("- ")
            if line.lower().startswith("antecedents:"):
                ant_str = line.split(":", 1)[1].strip()
                antecedents = [a.strip() for a in ant_str.split(",") if a.strip()]
            elif line.lower().startswith("unless:"):
                unless_str = line.split(":", 1)[1].strip()
                outlist = [u.strip() for u in unless_str.split(",") if u.strip()]
            elif line.lower().startswith("label:"):
                label = line.split(":", 1)[1].strip()

        if antecedents:
            proposals.append({
                "kind": kind,
                "id": belief_id,
                "text": text,
                "antecedents": antecedents,
                "outlist": outlist,
                "label": label,
            })

    return proposals


def _parse_nogood_proposals(response: str) -> list[dict]:
    """Parse NOGOOD proposals from LLM response."""
    proposals = []
    pattern = re.compile(
        r"###\s+NOGOOD\s+(\S+)\s*\n"
        r"(.+?)(?=\n###\s+NOGOOD|\Z)",
        re.DOTALL,
    )

    for match in pattern.finditer(response):
        nogood_id = match.group(1)
        body = match.group(2).strip()

        claims = []
        analysis = ""
        severity = ""
        resolution = ""
        for line in body.split("\n"):
            line = line.strip().lstrip("- ")
            if line.lower().startswith("claims:"):
                claims_str = line.split(":", 1)[1].strip()
                claims = [c.strip() for c in claims_str.split(",") if c.strip()]
            elif line.lower().startswith("analysis:"):
                analysis = line.split(":", 1)[1].strip()
            elif line.lower().startswith("severity:"):
                severity = line.split(":", 1)[1].strip()
            elif line.lower().startswith("resolution:"):
                resolution = line.split(":", 1)[1].strip()

        if len(claims) >= 2:
            proposals.append({
                "id": nogood_id,
                "claims": claims,
                "analysis": analysis,
                "severity": severity,
                "resolution": resolution,
            })

    return proposals


# --- CLI ---


@click.group()
@click.version_option(package_name="ftl-meta-expert")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.option("--model", "-m", default="claude", help="LLM model to use")
@click.option("--timeout", "-t", default=300, type=int, help="LLM timeout in seconds")
@click.pass_context
def cli(ctx, quiet, model, timeout):
    """Cross-domain expert aggregator."""
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["model"] = model
    ctx.obj["timeout"] = timeout


@cli.command()
@click.option("--domain", "-d", default="Cross-domain expert synthesis",
              help="Domain description")
@click.argument("expert_repos", nargs=-1)
@click.pass_context
def init(ctx, domain, expert_repos):
    """Bootstrap a meta-expert knowledge base.

    Pass expert repos as NAME=PATH pairs:

        meta-expert init code=/path/to/code-expert project=/path/to/project-expert
    """
    # Check prerequisites
    for tool in ["entry", "reasons"]:
        if not shutil.which(tool):
            click.echo(f"Error: {tool} not found on PATH", err=True)
            click.echo(f"Install with: uv tool install ftl-{tool if tool == 'reasons' else tool}", err=True)
            sys.exit(1)

    model = ctx.obj["model"]
    if not check_model_available(model):
        click.echo(f"Warning: {model} CLI not found on PATH", err=True)

    # Parse expert repos from NAME=PATH arguments
    experts = []
    for arg in expert_repos:
        if "=" not in arg:
            click.echo(f"Error: expected NAME=PATH, got: {arg}", err=True)
            sys.exit(1)
        name, repo = arg.split("=", 1)
        experts.append({
            "name": name,
            "repo": os.path.abspath(repo),
            "beliefs_file": "beliefs.md",
        })

    # Create project dir and config
    _save_config({
        "domain": domain,
        "experts": experts,
        "ask": {
            "threshold": 0.80,
            "top_k": 3,
            "spacy_model": "en_core_web_md",
        },
        "created": date.today().isoformat(),
    })

    # Create entries dir
    Path("entries").mkdir(exist_ok=True)

    # Init reasons.db
    if not Path("reasons.db").exists():
        subprocess.run(["reasons", "init"], capture_output=True)
        for expert in experts:
            subprocess.run(
                ["reasons", "add-repo", expert["name"], expert["repo"]],
                capture_output=True,
            )

    _emit(ctx, f"Initialized meta-expert knowledge base")
    _emit(ctx, f"  Domain: {domain}")
    if experts:
        _emit(ctx, f"  Experts: {', '.join(e['name'] for e in experts)}")
    else:
        _emit(ctx, "  Experts: none (add via .meta-expert/config.json)")
    _emit(ctx, f"  Config: {PROJECT_DIR}/config.json")
    _emit(ctx, f"  Database: reasons.db")
    _emit(ctx, "")
    _emit(ctx, "Next: meta-expert import")


@cli.command("import")
@click.option("--expert", "-e", help="Import from a specific expert only")
@click.option("--only-in", is_flag=True, help="Import only IN beliefs")
@click.pass_context
def import_beliefs(ctx, expert, only_in):
    """Import beliefs from expert knowledge bases."""
    config = _require_config()
    experts = config.get("experts", DEFAULT_EXPERTS)

    if expert:
        experts = [e for e in experts if e["name"] == expert]
        if not experts:
            click.echo(f"Error: unknown expert '{expert}'", err=True)
            click.echo(f"Available: {', '.join(e['name'] for e in config.get('experts', []))}", err=True)
            sys.exit(1)

    total_imported = 0
    for exp in experts:
        beliefs_path = os.path.join(exp["repo"], exp["beliefs_file"])
        if not os.path.isfile(beliefs_path):
            click.echo(f"Warning: {beliefs_path} not found, skipping {exp['name']}", err=True)
            continue

        cmd = ["reasons", "import-agent", exp["name"], beliefs_path]
        if only_in:
            cmd.append("--only-in")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"Error importing {exp['name']}: {result.stderr}", err=True)
            continue

        _emit(ctx, f"Imported {exp['name']} expert from {beliefs_path}")
        if result.stdout.strip():
            _emit(ctx, f"  {result.stdout.strip()}")
        total_imported += 1

    if total_imported > 0:
        _emit(ctx, "")
        _emit(ctx, "Exporting combined network...")
        _reasons_export()
        _emit(ctx, "  -> beliefs.md")
        _emit(ctx, "  -> network.json")

        # Create entry
        expert_names = ", ".join(e["name"] for e in experts)
        _create_entry(
            "import",
            f"Imported beliefs from {expert_names}",
            f"Imported beliefs from {total_imported} expert(s): {expert_names}\n\n"
            f"Only IN: {only_in}",
        )

    _emit(ctx, "")
    _emit(ctx, f"Done. Imported from {total_imported}/{len(experts)} expert(s).")
    _emit(ctx, "Next: meta-expert derive --auto")


@cli.command()
@click.option("--auto", "auto_apply", is_flag=True, help="Automatically apply proposals")
@click.option("--dry-run", is_flag=True, help="Print prompt without invoking LLM")
@click.option("--budget", "-b", default=300, type=int, help="Max beliefs per agent in prompt")
@click.option("--output", "-o", default="proposed-derivations.md", help="Output file for proposals")
@click.pass_context
def derive(ctx, auto_apply, dry_run, budget, output):
    """Derive cross-domain insights from combined belief network."""
    config = _require_config()

    # Load network
    network = _load_network_json()
    nodes = network.get("nodes", {})

    # Group beliefs by agent
    beliefs_by_agent: dict[str, list[dict]] = {}
    derived_beliefs: list[dict] = []

    for nid, node in nodes.items():
        if nid.endswith(":active"):
            continue

        belief = {"id": nid, "text": node["text"], "truth_value": node["truth_value"]}

        if ":" in nid:
            agent = nid.split(":")[0]
            beliefs_by_agent.setdefault(agent, []).append(belief)
        else:
            derived_beliefs.append(belief)

    if not beliefs_by_agent:
        click.echo("No agent beliefs found. Run: meta-expert import", err=True)
        sys.exit(1)

    # Build prompt
    prompt = build_derive_prompt(beliefs_by_agent, derived_beliefs, budget)

    if dry_run:
        click.echo(prompt)
        return

    # Invoke LLM
    _emit(ctx, "Deriving cross-domain insights...")
    for agent, beliefs in beliefs_by_agent.items():
        in_count = sum(1 for b in beliefs if b["truth_value"] == "IN")
        _emit(ctx, f"  {agent}: {in_count} IN beliefs")
    _emit(ctx, f"  Existing derived: {len(derived_beliefs)}")
    _emit(ctx, "")

    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]
    response = invoke_sync(prompt, model, timeout)

    # Parse proposals
    proposals = _parse_derive_proposals(response)
    _emit(ctx, f"Found {len(proposals)} proposal(s)")

    if not proposals:
        _emit(ctx, "No cross-domain derivations proposed.")
        return

    if auto_apply:
        applied = 0
        for p in proposals:
            cmd = ["reasons", "add", p["id"], p["text"]]
            if p["antecedents"]:
                cmd.extend(["--sl", ",".join(p["antecedents"])])
            if p["outlist"]:
                cmd.extend(["--unless", ",".join(p["outlist"])])
            if p["label"]:
                cmd.extend(["--label", p["label"]])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                kind = p["kind"].upper()
                _emit(ctx, f"  [{kind}] {p['id']}: {p['text'][:80]}")
                applied += 1
            else:
                click.echo(f"  Failed: {p['id']}: {result.stderr.strip()}", err=True)

        _emit(ctx, f"\nApplied {applied}/{len(proposals)} proposals")
        _reasons_export()

        _create_entry(
            "derive",
            f"Derived {applied} cross-domain beliefs",
            f"Applied {applied} cross-domain derivations.\n\n" + response,
        )
    else:
        # Write to file
        with open(output, "w") as f:
            f.write(f"# Cross-Domain Derivation Proposals\n\n")
            f.write(f"Generated: {date.today().isoformat()}\n\n")
            f.write(response)
        _emit(ctx, f"Proposals written to {output}")
        _emit(ctx, "Review and run: meta-expert derive --auto")


@cli.command()
@click.argument("question")
@click.option("--threshold", default=None, type=float, help="Match threshold (default from config)")
@click.option("--top-k", default=None, type=int, help="Number of top matches")
@click.option("--no-fallback", is_flag=True, help="Skip LLM fallback, spaCy only")
@click.option("--verbose", "-v", is_flag=True, help="Show debug info")
@click.pass_context
def ask(ctx, question, threshold, top_k, no_fallback, verbose):
    """Ask a question against the combined belief network."""
    config = _require_config()
    ask_config = config.get("ask", {})

    if threshold is None:
        threshold = ask_config.get("threshold", 0.80)
    if top_k is None:
        top_k = ask_config.get("top_k", 3)

    from .belief_answer import answer_from_beliefs

    db_path = Path("reasons.db")
    if not db_path.exists():
        click.echo("Error: reasons.db not found. Run: meta-expert import", err=True)
        sys.exit(1)

    result = answer_from_beliefs(
        question,
        belief_sources=[db_path],
        threshold=threshold,
        top_k=top_k,
        verbose=verbose,
    )

    if result:
        _emit(ctx, result["answer"])
        if verbose:
            _emit(ctx, f"\n[spaCy match: {result['score']:.0%}, {result['time']:.2f}s]")
        return

    if no_fallback:
        _emit(ctx, f"No belief match above {threshold:.0%} threshold.")
        return

    # LLM fallback
    _emit(ctx, "(No direct belief match, falling through to LLM...)\n")

    # Get closest matches for context even below threshold
    low_result = answer_from_beliefs(
        question,
        belief_sources=[db_path],
        threshold=0.0,
        top_k=5,
        verbose=False,
    )
    closest = low_result["answer"] if low_result else "(no beliefs loaded)"
    compact = _get_compact(2000)

    prompt = ASK_FALLBACK_PROMPT.format(
        domain=config.get("domain", ""),
        question=question,
        closest_beliefs=closest,
        compact_summary=compact,
    )

    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]
    response = invoke_sync(prompt, model, timeout)
    _emit(ctx, response)

    # Create entry for novel questions
    _create_entry(
        "ask",
        f"Question: {question[:60]}",
        f"## Question\n{question}\n\n## Answer\n{response}",
    )


@cli.command("generate-aliases")
@click.option("--limit", default=0, type=int, help="Max beliefs to process (0=all)")
@click.pass_context
def generate_aliases_cmd(ctx, limit):
    """Generate question-form aliases for better spaCy matching."""
    _require_config()

    db_path = Path("reasons.db")
    if not db_path.exists():
        click.echo("Error: reasons.db not found. Run: meta-expert import", err=True)
        sys.exit(1)

    from .belief_answer import generate_aliases

    _emit(ctx, "Generating question aliases...")
    aliases_path = generate_aliases(db_path, limit=limit)
    _emit(ctx, f"Aliases written to {aliases_path}")


@cli.command()
@click.option("--auto", "auto_apply", is_flag=True, help="Automatically apply nogoods")
@click.option("--dry-run", is_flag=True, help="Print prompt without invoking LLM")
@click.option("--budget", "-b", default=200, type=int, help="Max beliefs per agent in prompt")
@click.option("--output", "-o", default="proposed-nogoods.md", help="Output file for proposals")
@click.pass_context
def contradictions(ctx, auto_apply, dry_run, budget, output):
    """Detect cross-domain contradictions (nogoods)."""
    config = _require_config()

    network = _load_network_json()
    nodes = network.get("nodes", {})

    # Group IN beliefs by agent
    sections = []
    for agent_name in sorted(set(
        nid.split(":")[0] for nid in nodes if ":" in nid and not nid.endswith(":active")
    )):
        agent_beliefs = [
            (nid, node) for nid, node in nodes.items()
            if nid.startswith(f"{agent_name}:") and not nid.endswith(":active")
            and node["truth_value"] == "IN"
        ]
        if not agent_beliefs:
            continue

        lines = [f"### {agent_name} expert ({len(agent_beliefs)} IN beliefs)"]
        for nid, node in agent_beliefs[:budget]:
            lines.append(f"- `{nid}`: {node['text'][:200]}")
        if len(agent_beliefs) > budget:
            lines.append(f"*({len(agent_beliefs) - budget} more omitted)*")
        sections.append("\n".join(lines))

    if not sections:
        click.echo("No agent beliefs found. Run: meta-expert import", err=True)
        sys.exit(1)

    beliefs_section = "\n\n".join(sections)
    prompt = CONTRADICTION_DETECTION_PROMPT.format(beliefs_section=beliefs_section)

    if dry_run:
        click.echo(prompt)
        return

    _emit(ctx, "Detecting cross-domain contradictions...")
    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]
    response = invoke_sync(prompt, model, timeout)

    proposals = _parse_nogood_proposals(response)
    _emit(ctx, f"Found {len(proposals)} contradiction(s)")

    if not proposals:
        _emit(ctx, "No cross-domain contradictions detected.")
        return

    if auto_apply:
        applied = 0
        for p in proposals:
            cmd = ["reasons", "nogood"] + p["claims"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                _emit(ctx, f"  [NOGOOD] {p['id']}: {', '.join(p['claims'])}")
                _emit(ctx, f"    {p['analysis'][:100]}")
                applied += 1
            else:
                click.echo(f"  Failed: {p['id']}: {result.stderr.strip()}", err=True)

        _emit(ctx, f"\nApplied {applied}/{len(proposals)} nogoods")
        _reasons_export()

        _create_entry(
            "contradictions",
            f"Found {applied} cross-domain contradictions",
            f"Detected {applied} cross-domain contradictions.\n\n" + response,
        )
    else:
        with open(output, "w") as f:
            f.write(f"# Cross-Domain Contradiction Proposals\n\n")
            f.write(f"Generated: {date.today().isoformat()}\n\n")
            f.write(response)
        _emit(ctx, f"Proposals written to {output}")


@cli.command()
@click.option("--budget", "-b", default=2000, type=int, help="Token budget for belief context")
@click.pass_context
def summary(ctx, budget):
    """Generate executive synthesis across all expert domains."""
    config = _require_config()
    domain = config.get("domain", "")

    # Get beliefs text
    beliefs_text = _get_compact(budget)

    # Get per-agent stats
    network = _load_network_json()
    nodes = network.get("nodes", {})
    nogoods = network.get("nogoods", [])

    agent_stats: dict[str, int] = {"nogoods": len(nogoods), "derived": 0}
    for nid, node in nodes.items():
        if nid.endswith(":active"):
            continue
        if ":" in nid:
            agent = nid.split(":")[0]
            agent_stats[agent] = agent_stats.get(agent, 0) + 1
        else:
            agent_stats["derived"] += 1

    prompt = build_summary_prompt(beliefs_text, domain, agent_stats)

    _emit(ctx, "Generating executive summary...")
    model = ctx.obj["model"]
    timeout = ctx.obj["timeout"]
    response = invoke_sync(prompt, model, timeout)

    click.echo(response)

    _create_entry(
        "summary",
        "Executive synthesis across all expert domains",
        response,
    )


@cli.command()
@click.option("--all", "show_all", is_flag=True, help="Show all topics including done/skipped")
@click.pass_context
def topics(ctx, show_all):
    """Show cross-domain investigation topics."""
    _require_config()
    queue = load_queue()

    if not queue:
        _emit(ctx, "No topics in queue.")
        return

    pending = [t for t in queue if t.status == "pending"]
    done = [t for t in queue if t.status == "done"]
    skipped = [t for t in queue if t.status == "skipped"]

    _emit(ctx, f"Topics: {len(pending)} pending, {len(done)} done, {len(skipped)} skipped\n")

    display = queue if show_all else pending
    for i, topic in enumerate(display):
        status_icon = {"pending": " ", "done": "x", "skipped": "-"}.get(topic.status, "?")
        _emit(ctx, f"  [{status_icon}] {i:3d}. [{topic.kind}] `{topic.target}` — {topic.title}")


@cli.command()
@click.pass_context
def status(ctx, ):
    """Show meta-expert dashboard."""
    config = _require_config()
    domain = config.get("domain", "")
    experts = config.get("experts", [])

    _emit(ctx, f"Meta Expert: {domain}")
    _emit(ctx, f"Created: {config.get('created', 'unknown')}")
    _emit(ctx, "")

    # Expert repos
    _emit(ctx, f"Expert repos ({len(experts)}):")
    for exp in experts:
        beliefs_path = os.path.join(exp["repo"], exp["beliefs_file"])
        exists = "ok" if os.path.isfile(beliefs_path) else "MISSING"
        _emit(ctx, f"  {exp['name']:12s} {exp['repo']} [{exists}]")
    _emit(ctx, "")

    # Reasons stats — use network JSON directly (reasons status is too verbose)
    if Path("reasons.db").exists() and _has_reasons():
        try:
            network = _load_network_json()
            nodes = network.get("nodes", {})
            nogoods = network.get("nogoods", [])

            total_in = sum(1 for n in nodes.values() if n["truth_value"] == "IN")
            total_out = sum(1 for n in nodes.values() if n["truth_value"] != "IN")
            _emit(ctx, f"Belief network: {total_in} IN, {total_out} OUT, {len(nodes)} total")
            _emit(ctx, "")

            agent_counts: dict[str, dict[str, int]] = {}
            derived_count = 0
            for nid, node in nodes.items():
                if nid.endswith(":active"):
                    continue
                if ":" in nid:
                    agent = nid.split(":")[0]
                    if agent not in agent_counts:
                        agent_counts[agent] = {"IN": 0, "OUT": 0}
                    agent_counts[agent][node["truth_value"]] = (
                        agent_counts[agent].get(node["truth_value"], 0) + 1
                    )
                else:
                    derived_count += 1

            if agent_counts:
                _emit(ctx, "Per-agent breakdown:")
                for agent in sorted(agent_counts):
                    c = agent_counts[agent]
                    _emit(ctx, f"  {agent:12s} {c.get('IN', 0)} IN, {c.get('OUT', 0)} OUT")
                _emit(ctx, f"  {'derived':12s} {derived_count}")
                _emit(ctx, f"  {'nogoods':12s} {len(nogoods)}")
                _emit(ctx, "")
        except Exception:
            _emit(ctx, "Belief network: error reading network")
            _emit(ctx, "")
    else:
        _emit(ctx, "Belief network: not initialized")
        _emit(ctx, "")

    # Topics
    pc = pending_count()
    total = len(load_queue())
    _emit(ctx, f"Topics: {pc} pending / {total} total")

    # Aliases
    aliases_path = Path("reasons.aliases.json")
    if aliases_path.exists():
        with open(aliases_path) as f:
            aliases = json.load(f)
        _emit(ctx, f"Aliases: {len(aliases)} beliefs covered")
    else:
        _emit(ctx, "Aliases: not generated (run: meta-expert generate-aliases)")

    # Entries
    entries_dir = Path("entries")
    if entries_dir.exists():
        entry_count = sum(1 for _ in entries_dir.rglob("*.md"))
        _emit(ctx, f"Entries: {entry_count}")
