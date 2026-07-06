"""
main.py — SimuHome TKG Agent CLI
=================================
Usage examples:

  # Run TKG agent on a single episode
  python main.py run --episode simuhome/data/benchmark/qt1_feasible_seed_23.json

  # Run baseline agent on a single episode
  python main.py run --episode simuhome/data/benchmark/qt1_feasible_seed_23.json --baseline

  # Run both agents side-by-side and compare
  python main.py compare --episode simuhome/data/benchmark/qt1_feasible_seed_23.json

  # Test Neo4j connection
  python main.py health
"""
from __future__ import annotations
import os
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

app = typer.Typer(help="SimuHome TKG Agent CLI")
console = Console()


def _load_episode(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Episode file not found: {path}[/red]")
        raise typer.Exit(1)
    with open(p) as f:
        return json.load(f)


def _load_episode_into_simulator(episode: dict) -> bool:
    """POST initial_home_config directly to simulator reset endpoint."""
    import requests, os
    base = os.getenv("SIMULATOR_BASE", "http://localhost:8000")
    try:
        resp = requests.post(
            f"{base}/api/simulation/reset",
            json=episode["initial_home_config"],   # SimulationConfig = initial_home_config directly
            timeout=15,
        )
        if resp.status_code == 200:
            console.print("[dim]  Simulator loaded with episode home config [/dim]")
            return True
        else:
            console.print(f"[yellow]  Simulator reset returned {resp.status_code}: {resp.text[:120]}[/yellow]")
            return False
    except Exception as e:
        console.print(f"[red]  Could not reset simulator: {e}[/red]")
        return False


def _make_llm():
    #from tkg_agent.agent.groq_provider import GroqProvider
    #from tkg_agent.agent.kaggle_provider import KaggleProvider as GroqProvider
    from tkg_agent.agent.gemini_provider import GeminiProvider as GroqProvider
    return GroqProvider()

"""def _make_llm():

    from tkg_agent.agent.kaggle_provider import KaggleProvider

    return KaggleProvider(
        endpoint=os.getenv("KAGGLE_ENDPINT")
    )
"""
def _make_db():
    from tkg_agent.graph.neo4j_client import Neo4jClient
    db = Neo4jClient()
    db.create_constraints()
    return db


def _print_episode_header(episode: dict):
    meta = episode["meta"]
    console.print(Panel(
        f"[bold]Query:[/bold] {episode['query']}\n"
        f"[bold]Type:[/bold]  {meta['query_type']} | "
        f"[bold]Case:[/bold] {meta['case']} | "
        f"[bold]Seed:[/bold] {meta['seed']}\n"
        f"[bold]User location:[/bold] {episode['user_location']}",
        title="Episode",
        border_style="blue",
    ))


def _print_result(result, label: str, color: str = "green"):
    console.print(f"\n[{color}][bold]{label}[/bold][/{color}]")
    for i, step in enumerate(result.steps, 1):
        thought = step.thought or ""
        console.print(f"  Step {step.step}: [cyan]{step.action}[/cyan]")
        if thought:
            console.print(f"    Thought: {thought}{'...' if len(thought)>120 else ''}")
        if step.observation:
            obs_str = json.dumps(step.observation)
            console.print(f"    Obs: {obs_str}{'...' if len(obs_str)>100 else ''}")

    console.print(Panel(
        result.final_answer,
        title=f"{label} — Final Answer",
        border_style=color,
    ))


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def health():
    """Check Neo4j and Groq connectivity."""
    console.print("\n[bold]Checking Neo4j...[/bold]")
    try:
        db = _make_db()
        ok = db.health_check()
        console.print(f"  Neo4j: {'[green] connected[/green]' if ok else '[red] failed[/red]'}")
        db.close()
    except Exception as e:
        console.print(f"  Neo4j: [red] {e}[/red]")

    console.print("\n[bold]Checking Groq...[/bold]")
    try:
        llm = _make_llm()
        resp = llm.generate([{"role": "user", "content": "Reply with: ready"}])
        console.print(f"  Groq:  [green] {resp[:60]}[/green]")
    except Exception as e:
        console.print(f"  Groq:  [red] {e}[/red]")


@app.command()
def run(
    episode: str = typer.Option(..., "--episode", "-e", help="Path to episode JSON"),
    baseline: bool = typer.Option(False, "--baseline", help="Run baseline agent (no TKG)"),
    max_steps: int = typer.Option(20, "--max-steps"),
):
    """Run a single agent (TKG or baseline) on one episode."""
    ep = _load_episode(episode)
    _print_episode_header(ep)
    _load_episode_into_simulator(ep)

    llm = _make_llm()

    if baseline:
        from tkg_agent.agent.base_agent import BaseReActAgent
        agent = BaseReActAgent(llm, max_steps=max_steps)
        label, color = "Baseline ReAct", "yellow"
    else:
        db = _make_db()
        from tkg_agent.agent.tkg_agent import TKGReActAgent
        agent = TKGReActAgent(llm, db, max_steps=max_steps)
        label, color = "TKG Agent", "green"

    console.print(f"\n[bold]Running {label}...[/bold]\n")

    try:
        if baseline:
            result = agent.run(
                ep["query"],
                user_location=ep.get("user_location"),
                current_time=ep["initial_home_config"].get("base_time"),
            )
        else:
            result = agent.run(
                ep["query"],
                user_location=ep.get("user_location"),
                current_time=ep["initial_home_config"].get("base_time"),
                episode=ep,
            )
            metrics = agent.get_tkg_metrics()
            console.print(f"\n[bold]TKG Metrics:[/bold]")
            console.print(f"  Graph hits:    {metrics['graph_hits']}")
            console.print(f"  Graph misses:  {metrics['graph_misses']}")
            console.print(f"  Facts written: {metrics['facts_written']}")
            db.close()

        _print_result(result, label, color)

    except Exception as e:
        console.print(f"[red]Agent failed: {e}[/red]")
        import traceback; traceback.print_exc()


@app.command()
def compare(
    episode: str = typer.Option(..., "--episode", "-e", help="Path to episode JSON"),
    max_steps: int = typer.Option(20, "--max-steps"),
):
    """Run both baseline and TKG agent on the same episode and compare."""
    ep = _load_episode(episode)
    _print_episode_header(ep)
    _load_episode_into_simulator(ep)

    llm = _make_llm()

    # ── Baseline ──────────────────────────────────────────────────────────────
    console.print("\n[yellow][bold]▶ Running Baseline Agent...[/bold][/yellow]")
    from tkg_agent.agent.base_agent import BaseReActAgent
    baseline_agent = BaseReActAgent(llm, max_steps=max_steps)
    try:
        baseline_result = baseline_agent.run(
            ep["query"],
            user_location=ep.get("user_location"),
            current_time=ep["initial_home_config"].get("base_time"),
        )
        baseline_ok = True
    except Exception as e:
        console.print(f"[red]Baseline failed: {e}[/red]")
        baseline_result = None
        baseline_ok = False

    # ── TKG Agent ─────────────────────────────────────────────────────────────
    console.print("\n[green][bold]▶ Running TKG Agent...[/bold][/green]")
    db = _make_db()
    from tkg_agent.agent.tkg_agent import TKGReActAgent
    tkg_agent = TKGReActAgent(llm, db, max_steps=max_steps)
    try:
        tkg_result = tkg_agent.run(
            ep["query"],
            user_location=ep.get("user_location"),
            current_time=ep["initial_home_config"].get("base_time"),
            episode=ep,
        )
        tkg_ok = True
        tkg_metrics = tkg_agent.get_tkg_metrics()
    except Exception as e:
        console.print(f"[red]TKG agent failed: {e}[/red]")
        tkg_result = None
        tkg_ok = False
        tkg_metrics = {}
    finally:
        db.close()

    # ── Comparison table ──────────────────────────────────────────────────────
    console.print("\n")
    table = Table(title="Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Metric",          style="bold", width=25)
    table.add_column("Baseline",        style="yellow", width=20)
    table.add_column("TKG Agent",       style="green",  width=20)

    b_steps = len(baseline_result.steps) if baseline_ok else "—"
    t_steps = len(tkg_result.steps)      if tkg_ok     else "—"
    table.add_row("Steps taken",         str(b_steps), str(t_steps))
    table.add_row("Finished cleanly",
                  "YES" if baseline_ok else "NO",
                  "YES" if tkg_ok else "NO")
    table.add_row("TKG graph hits",      "—", str(tkg_metrics.get("graph_hits", "—")))
    table.add_row("TKG graph misses",    "—", str(tkg_metrics.get("graph_misses", "—")))
    table.add_row("Facts written",       "—", str(tkg_metrics.get("facts_written", "—")))

    console.print(table)

    if baseline_ok:
        _print_result(baseline_result, "Baseline", "yellow")
    if tkg_ok:
        _print_result(tkg_result, "TKG Agent", "green")


if __name__ == "__main__":
    app()