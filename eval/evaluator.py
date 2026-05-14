"""
eval/evaluator.py
=================
Demo batch evaluator with RESUME support.
Results are written to CSV after every single episode run.
On restart, already-completed episodes are skipped automatically.

Usage:
    python -m tkg_agent.eval.evaluator
    python -m tkg_agent.eval.evaluator --qt1-n 5 --qt4-n 3 --delay 8
"""
from __future__ import annotations

import csv
import json
import os
import time
import traceback
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()

SIMUHOME_BENCHMARK = Path(os.getenv("EPISODE_DIR", "../SimuHome/data/benchmark"))
SIMULATOR_BASE     = os.getenv("SIMULATOR_BASE", "http://localhost:8000")
RESULTS_DIR        = Path("experiments/demo")
CSV_PATH           = RESULTS_DIR / "results.csv"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    episode_id:    str
    query_type:    str
    case:          str
    seed:          int
    agent:         str
    success:       bool
    steps:         int
    final_answer:  str
    graph_hits:    int
    graph_misses:  int
    facts_written: int
    error:         str


FIELDNAMES = [f.name for f in fields(EpisodeResult)]


# ── Resume support ────────────────────────────────────────────────────────────

def _load_done() -> set[str]:
    """Return set of 'episode_id::agent' strings already saved to CSV."""
    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add(f"{row['episode_id']}::{row['agent']}")
    return done


def _append_result(result: EpisodeResult):
    """Append one result row to CSV immediately (crash-safe)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_episodes(qt: str, case: str, n: int) -> list[dict]:
    pattern = f"{qt}_{case}*.json"
    files = sorted(SIMUHOME_BENCHMARK.glob(pattern))[:n]
    if not files:
        # fallback: any file containing qt in name
        files = sorted(SIMUHOME_BENCHMARK.glob(f"{qt}*.json"))[:n]
    episodes = []
    for f in files:
        with open(f) as fp:
            episodes.append(json.load(fp))
    console.print(f"  Loaded {len(episodes)} episodes for qt={qt} case={case}")
    return episodes


def _reset_simulator(episode: dict) -> bool:
    try:
        r = requests.post(
            f"{SIMULATOR_BASE}/api/simulation/reset",
            json=episode["initial_home_config"],
            timeout=15,
        )
        return r.status_code == 200
    except Exception as e:
        console.print(f"[red]  Simulator reset error: {e}[/red]")
        return False


def _run_baseline(episode: dict, max_steps: int) -> EpisodeResult:
    from tkg_agent.agent.base_agent import BaseReActAgent
    from tkg_agent.agent.groq_provider import GroqProvider

    meta  = episode["meta"]
    ep_id = f"{meta['query_type']}_{meta['case']}_seed{meta['seed']}"

    try:
        llm    = GroqProvider()
        agent  = BaseReActAgent(llm, max_steps=max_steps)
        result = agent.run(
            episode["query"],
            user_location=episode.get("user_location"),
            current_time=episode["initial_home_config"].get("base_time"),
        )
        success = "Max steps reached" not in result.final_answer and result.final_answer != ""
        return EpisodeResult(
            episode_id=ep_id, query_type=meta["query_type"], case=meta["case"],
            seed=meta["seed"], agent="baseline", success=success,
            steps=len(result.steps), final_answer=result.final_answer[:120],
            graph_hits=0, graph_misses=0, facts_written=0, error="",
        )
    except Exception as e:
        return EpisodeResult(
            episode_id=ep_id, query_type=meta["query_type"], case=meta["case"],
            seed=meta["seed"], agent="baseline", success=False, steps=0,
            final_answer="", graph_hits=0, graph_misses=0, facts_written=0,
            error=str(e)[:150],
        )


def _run_tkg(episode: dict, db, max_steps: int) -> EpisodeResult:
    from tkg_agent.agent.tkg_agent import TKGReActAgent
    from tkg_agent.agent.groq_provider import GroqProvider

    meta  = episode["meta"]
    ep_id = f"{meta['query_type']}_{meta['case']}_seed{meta['seed']}"

    try:
        llm    = GroqProvider()
        agent  = TKGReActAgent(llm, db, max_steps=max_steps)
        result = agent.run(
            episode["query"],
            user_location=episode.get("user_location"),
            current_time=episode["initial_home_config"].get("base_time"),
            episode=episode,
        )
        m = agent.get_tkg_metrics()
        success = "Max steps reached" not in result.final_answer and result.final_answer != ""
        return EpisodeResult(
            episode_id=ep_id, query_type=meta["query_type"], case=meta["case"],
            seed=meta["seed"], agent="tkg", success=success,
            steps=len(result.steps), final_answer=result.final_answer[:120],
            graph_hits=m["graph_hits"], graph_misses=m["graph_misses"],
            facts_written=m["facts_written"], error="",
        )
    except Exception as e:
        return EpisodeResult(
            episode_id=ep_id, query_type=meta["query_type"], case=meta["case"],
            seed=meta["seed"], agent="tkg", success=False, steps=0,
            final_answer="", graph_hits=0, graph_misses=0, facts_written=0,
            error=str(e)[:150],
        )


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_demo(qt1_n=5, qt4_n=3, delay=8.0, max_steps=15):
    from tkg_agent.graph.neo4j_client import Neo4jClient

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db   = Neo4jClient()
    db.create_constraints()
    done = _load_done()

    if done:
        console.print(f"[dim]Resuming — {len(done)} runs already completed, skipping them.[/dim]")

    episode_sets = [
        ("qt1",   "feasible", qt1_n),
        ("qt4-1", "feasible", qt4_n),
    ]

    for qt, case, n in episode_sets:
        console.rule(f"[bold cyan]{qt.upper()} / {case}[/bold cyan]")
        episodes = _load_episodes(qt, case, n)

        for i, ep in enumerate(episodes, 1):
            meta  = ep["meta"]
            ep_id = f"{meta['query_type']}_{meta['case']}_seed{meta['seed']}"
            console.print(f"\n[bold]Episode {i}/{len(episodes)}:[/bold] {ep_id}")
            console.print(f"  [dim]{ep['query'][:90]}...[/dim]")

            # ── Baseline ──────────────────────────────────────────────────
            b_key = f"{ep_id}::baseline"
            if b_key in done:
                console.print("  [dim]Baseline already done — skipping[/dim]")
            else:
                if not _reset_simulator(ep):
                    console.print("[red]  Simulator reset failed — skipping episode[/red]")
                    continue
                console.print("  [yellow]▶ Baseline...[/yellow]", end=" ")
                b = _run_baseline(ep, max_steps)
                _append_result(b)
                done.add(b_key)
                icon = "SUCCESS" if b.success else "FAILED"
                console.print(f"{icon} {b.steps} steps  {b.error or b.final_answer[:60]}")
                console.print(f"  [dim]Waiting {delay}s...[/dim]")
                time.sleep(delay)

            # ── TKG ───────────────────────────────────────────────────────
            t_key = f"{ep_id}::tkg"
            if t_key in done:
                console.print("  [dim]TKG already done — skipping[/dim]")
            else:
                if not _reset_simulator(ep):
                    console.print("[red]  Simulator reset failed — skipping TKG run[/red]")
                    continue
                console.print("  [green]▶ TKG Agent...[/green]", end=" ")
                t = _run_tkg(ep, db, max_steps)
                _append_result(t)
                done.add(t_key)
                icon = "SUCCESS" if t.success else "FAILED"
                console.print(
                    f"{icon} {t.steps} steps  hits={t.graph_hits} "
                    f"facts={t.facts_written}  {t.error or t.final_answer[:50]}"
                )
                console.print(f"  [dim]Waiting {delay}s...[/dim]")
                time.sleep(delay)

    db.close()
    _print_summary()


# ── Summary (reads from CSV so works after resume too) ────────────────────────

def _print_summary():
    if not CSV_PATH.exists():
        console.print("[red]No results CSV found.[/red]")
        return

    results: list[EpisodeResult] = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            results.append(EpisodeResult(
                episode_id=row["episode_id"], query_type=row["query_type"],
                case=row["case"], seed=int(row["seed"]), agent=row["agent"],
                success=row["success"] == "True", steps=int(row["steps"]),
                final_answer=row["final_answer"], graph_hits=int(row["graph_hits"]),
                graph_misses=int(row["graph_misses"]), facts_written=int(row["facts_written"]),
                error=row["error"],
            ))

    baseline = [r for r in results if r.agent == "baseline"]
    tkg      = [r for r in results if r.agent == "tkg"]

    def avg(lst, key):
        vals = [getattr(r, key) for r in lst if not r.error]
        return f"{sum(vals)/len(vals):.1f}" if vals else "—"

    def pct(lst):
        if not lst: return "—"
        ok = sum(1 for r in lst if r.success)
        return f"{ok}/{len(lst)} ({100*ok//len(lst)}%)"

    console.rule("[bold magenta]EVALUATION SUMMARY[/bold magenta]")
    t = Table(show_header=True, header_style="bold magenta", min_width=65)
    t.add_column("Metric",       style="bold",   width=28)
    t.add_column("Baseline",     style="yellow", width=18)
    t.add_column("TKG Agent",    style="green",  width=18)

    t.add_row("Episodes run",       str(len(baseline)),   str(len(tkg)))
    t.add_row("Success rate",       pct(baseline),        pct(tkg))
    t.add_row("Avg steps",          avg(baseline,"steps"),avg(tkg,"steps"))

    for qt in ["qt1", "qt4-1"]:
        bq = [r for r in baseline if r.query_type == qt]
        tq = [r for r in tkg      if r.query_type == qt]
        if bq or tq:
            t.add_row(f"  Success [{qt}]",   pct(bq),              pct(tq))
            t.add_row(f"  Avg steps [{qt}]", avg(bq,"steps"),      avg(tq,"steps"))

    t.add_row("Avg graph hits",     "—",                  avg(tkg,"graph_hits"))
    t.add_row("Avg facts/episode",  "—",                  avg(tkg,"facts_written"))
    console.print(t)

    # Per-episode detail
    console.print("\n[bold]Per-episode detail:[/bold]")
    d = Table(show_header=True, header_style="dim", min_width=95)
    d.add_column("Episode",    width=32)
    d.add_column("B-ok",       width=5)
    d.add_column("B-steps",    width=7)
    d.add_column("T-ok",       width=5)
    d.add_column("T-steps",    width=7)
    d.add_column("Hits",       width=5)
    d.add_column("Facts",      width=6)

    for ep_id in sorted({r.episode_id for r in results}):
        b = next((r for r in results if r.episode_id == ep_id and r.agent == "baseline"), None)
        tg = next((r for r in results if r.episode_id == ep_id and r.agent == "tkg"), None)
        d.add_row(
            ep_id,
            "SUCCESS" if (b  and b.success)  else "FAILED",
            str(b.steps)  if b  else "—",
            "SUCCESS" if (tg and tg.success) else "FAILED",
            str(tg.steps) if tg else "—",
            str(tg.graph_hits)    if tg else "—",
            str(tg.facts_written) if tg else "—",
        )
    console.print(d)
    console.print(f"\n[dim]Full results: {CSV_PATH}[/dim]")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--qt1-n",     type=int,   default=5)
    p.add_argument("--qt4-n",     type=int,   default=3)
    p.add_argument("--delay",     type=float, default=8.0,
                   help="Seconds between runs (increase if hitting rate limits)")
    p.add_argument("--max-steps", type=int,   default=15)
    p.add_argument("--summary",   action="store_true",
                   help="Just print summary from existing CSV, don't run new episodes")
    args = p.parse_args()

    if args.summary:
        _print_summary()
    else:
        run_demo(qt1_n=args.qt1_n, qt4_n=args.qt4_n,
                 delay=args.delay, max_steps=args.max_steps)