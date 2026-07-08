"""
day_sim.py — Full Day Simulation + Natural Query CLI
=====================================================
Loads multiple QT1 episodes sequentially, advancing
simulator time to build a full day of TKG memory.
Then opens an interactive chat loop where you ask
natural language questions answered purely from graph.

Usage:
    python day_sim.py --episodes-dir ../SimuHome/data/benchmark --n 8
    python day_sim.py --load-only      # skip re-loading, just open chat
"""
from __future__ import annotations

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

load_dotenv()

console = Console()

# ── Time simulation ────────────────────────────────────────────────────────

DAY_START = "2025-08-23 06:00:00"   # 6 AM
SLOT_MINUTES = 90                    # each episode = 1.5 hours of sim time

def _make_time_slots(n: int) -> list[str]:
    """Generate n evenly spaced time slots across a day starting at 6 AM."""
    base = datetime.strptime(DAY_START, "%Y-%m-%d %H:%M:%S")
    return [
        (base + timedelta(minutes=i * SLOT_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n)
    ]

# ── Episode loader ─────────────────────────────────────────────────────────

def load_qt1_episodes(episodes_dir: str, n: int) -> list[dict]:
    p = Path(episodes_dir)
    files = sorted(p.glob("qt1_feasible_seed*.json"))[:n]
    if not files:
        console.print(f"[red]No QT1 feasible episodes found in {episodes_dir}[/red]")
        sys.exit(1)
    episodes = []
    for f in files:
        with open(f) as fp:
            ep = json.load(fp)
        episodes.append(ep)
    return episodes

# ── Bootstrap a day into TKG ───────────────────────────────────────────────

def build_day(episodes: list[dict], db, time_slots: list[str]) -> list[dict]:
    """
    Load each episode into Neo4j with its assigned sim time.
    Returns a log of what was loaded.
    """
    from tkg_agent.ingestor.episode_ingestor import EpisodeIngestor

    ingestor = EpisodeIngestor(db)
    log = []

    console.print("\n[bold cyan]Building day simulation...[/bold cyan]\n")

    for i, (ep, sim_time) in enumerate(zip(episodes, time_slots), 1):
        meta = ep["meta"]

        # Override the episode base_time with our simulated slot time
        ep["initial_home_config"]["base_time"] = sim_time

        ep_id = ingestor.bootstrap(ep,clear=False)

        entry = {
            "slot":     i,
            "sim_time": sim_time,
            "episode":  ep_id,
            "query":    ep["query"][:70],
            "rooms":    list(ep["initial_home_config"]["rooms"].keys()),
        }
        log.append(entry)

        console.print(
            f"  [green]Slot {i:2}[/green]  [dim]{sim_time}[/dim]  "
            f"[white]{ep['query'][:55]}...[/white]"
        )

    console.print(f"\n[bold green]✅ Day loaded — {len(episodes)} episodes into TKG[/bold green]")
    return log

# ── Natural query handler ──────────────────────────────────────────────────

def answer_query(question: str, db, llm) -> str:
    """
    Answer a natural language question purely from TKG graph memory.
    No simulator calls — retriever only.
    """
    from tkg_agent.retrieval.retriever import TKGRetriever

    retriever = TKGRetriever(db, window_minutes=1440)  # full 24h window

    # Extract entity hints from the question
    question_lower = question.lower()

    ROOM_KEYWORDS = [
        "bathroom", "bedroom", "kitchen", "living_room",
        "utility_room", "study_room"
    ]
    mentioned_rooms = [r for r in ROOM_KEYWORDS if r.replace("_", " ") in question_lower or r in question_lower]

    # Pull all recent facts across the whole day
    all_facts = db.get_recently_changed("2025-08-23 00:00:00", limit=40)

    # Build context block
    sections = []

    # Room-specific states if mentioned
    for room in mentioned_rooms:
        snap = retriever.get_room_state(room)
        if "No facts" not in snap:
            sections.append(snap)

    # Device-specific: if a device keyword mentioned, find it
    device_rows = [r for r in all_facts if r["entity_type"] == "device"]
    device_ids_seen = set()
    for row in device_rows:
        did = row["entity_id"]
        dev_name = did.replace("device:", "").replace("_", " ")
        if any(word in question_lower for word in dev_name.split()):
            if did not in device_ids_seen:
                snap = retriever.get_device_state(did.replace("device:", ""))
                if "No facts" not in snap:
                    sections.append(snap)
                    device_ids_seen.add(did)

    # All timestamped facts for temporal queries
    if all_facts:
        lines = ["[TKG] Full day timeline:"]
        for row in all_facts:
            lines.append(
                f"  {row['timestamp'][:16]}  {row['entity_id']:<38} "
                f"{row['relation']:<20} → {row['value']}"
            )
        sections.append("\n".join(lines))

    graph_context = "\n\n".join(sections) if sections else "No facts found in graph."

    # Build prompt for LLM
    system = """You are a smart home assistant with access to a Temporal Knowledge Graph.
Answer the user's question using ONLY the graph facts provided below.
Be concise and specific. Include timestamps when relevant.
If the fact is not in the graph, say so clearly — do not guess."""

    user = f"""Graph memory (timestamped facts from today):

{graph_context}

User question: {question}

Answer based only on the graph facts above."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    try:
        return llm.generate(messages)
    except Exception as e:
        return f"[LLM error: {e}]"

# ── CLI chat loop ──────────────────────────────────────────────────────────

def chat_loop(db, llm, day_log: list[dict]):
    console.print("\n[bold]Day loaded. Ask anything about the home today.[/bold]")
    console.print("[dim]Type 'timeline' to see the day, 'exit' to quit.[/dim]\n")

    while True:
        try:
            question = Prompt.ask("[cyan]You[/cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question.strip():
            continue

        if question.strip().lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if question.strip().lower() == "timeline":
            _print_timeline(day_log)
            continue

        if question.strip().lower() == "facts":
            _print_all_facts(db)
            continue

        console.print("[dim]Querying graph...[/dim]")
        answer = answer_query(question, db, llm)
        console.print(Panel(answer, title="[green]Agent[/green]", border_style="green"))
        console.print()

def _print_timeline(day_log: list[dict]):
    t = Table(title="Today's Simulation Timeline", show_header=True)
    t.add_column("Slot", width=5)
    t.add_column("Time",  width=20)
    t.add_column("Episode", width=28)
    t.add_column("Query preview", width=55)
    for entry in day_log:
        t.add_row(
            str(entry["slot"]),
            entry["sim_time"],
            entry["episode"],
            entry["query"],
        )
    console.print(t)

def _print_all_facts(db):
    rows = db.get_recently_changed("2025-08-23 00:00:00", limit=50)
    t = Table(title="All TKG Facts Today", show_header=True)
    t.add_column("Time",     width=17)
    t.add_column("Entity",   width=38)
    t.add_column("Relation", width=22)
    t.add_column("Value",    width=20)
    for row in rows:
        t.add_row(
            row["timestamp"][:16],
            row["entity_id"],
            row["relation"],
            str(row["value"])[:20],
        )
    console.print(t)

# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full Day TKG Simulation + Query CLI")
    parser.add_argument("--episodes-dir", default="../SimuHome/data/benchmark",
                        help="Path to SimuHome benchmark episodes")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of episodes to simulate (default 8 = ~12 hr day)")
    parser.add_argument("--load-only", action="store_true",
                        help="Skip loading, open chat with existing graph data")
    parser.add_argument("--clear", action="store_true",
                        help="Clear all graph data before loading")
    args = parser.parse_args()

    from tkg_agent.graph.neo4j_client import Neo4jClient
    from tkg_agent.agent.groq_provider import GroqProvider

    db = Neo4jClient()
    db.create_constraints()
    llm = GroqProvider()

    day_log = []

    if args.clear:
        console.print("[yellow]Clearing graph...[/yellow]")
        db.run("MATCH (f:Fact) DETACH DELETE f")
        db.run("MATCH (e:Entity) DETACH DELETE e")
        console.print("[green]Graph cleared.[/green]")

    if not args.load_only:
        episodes = load_qt1_episodes(args.episodes_dir, args.n)
        time_slots = _make_time_slots(len(episodes))

        console.print(f"\n[bold]Simulating {len(episodes)} episodes across the day:[/bold]")
        for i, (ep, t) in enumerate(zip(episodes, time_slots), 1):
            console.print(f"  Slot {i}: {t}  →  {ep['query'][:60]}...")

        console.print()
        day_log = build_day(episodes, db, time_slots)
    else:
        console.print("[dim]Skipping load — using existing graph data.[/dim]")

    chat_loop(db, llm, day_log)
    db.close()

if __name__ == "__main__":
    main()
