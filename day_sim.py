"""
day_sim.py — Full Day TKG Simulation + Natural Query CLI
=========================================================
1. Loads N QT1 episodes with advancing simulator time into Neo4j
2. Interactive compare mode: Baseline vs TKG Agent (full loop) + Ground Truth
3. Saves a session report (JSON + Markdown) for presentation

Usage:
    python day_sim.py --n 8 --episodes-dir ../SimuHome/data/benchmark --clear
    python day_sim.py --load-only --mode compare
    python day_sim.py --load-only --mode tkg
"""
from __future__ import annotations

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

load_dotenv()
console = Console()

DAY_START    = "2025-08-23 06:00:00"
SLOT_MINUTES = 90
REPORT_DIR   = Path("experiments/day_sim")

# ── Time slots ──────────────────────────────────────────────────────────────

def _make_time_slots(n: int) -> list[str]:
    base = datetime.strptime(DAY_START, "%Y-%m-%d %H:%M:%S")
    return [(base + timedelta(minutes=i*SLOT_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(n)]

# ── Episode loader ──────────────────────────────────────────────────────────

def load_qt1_episodes(episodes_dir: str, n: int) -> list[dict]:
    files = sorted(Path(episodes_dir).glob("qt1_feasible_seed*.json"))[:n]
    if not files:
        console.print(f"[red]No QT1 episodes in {episodes_dir}[/red]")
        sys.exit(1)
    return [json.load(open(f)) for f in files]

# ── Bootstrap all episodes into TKG ────────────────────────────────────────

def build_day(episodes: list[dict], db, time_slots: list[str]) -> list[dict]:
    from tkg_agent.ingestor.episode_ingestor import EpisodeIngestor
    ingestor = EpisodeIngestor(db)
    log = []
    console.print("\n[bold cyan]Building day simulation...[/bold cyan]\n")
    for i, (ep, sim_time) in enumerate(zip(episodes, time_slots), 1):
        ep["initial_home_config"]["base_time"] = sim_time
        ep_id = ingestor.bootstrap(ep, clear=False)
        log.append({"slot": i, "sim_time": sim_time, "episode_id": ep_id,
                     "query": ep["query"][:80],
                     "rooms": list(ep["initial_home_config"]["rooms"].keys())})
        console.print(f"  [green]Slot {i:2}[/green]  [dim]{sim_time}[/dim]  "
                      f"[white]{ep['query'][:60]}...[/white]")
    total = db.run("MATCH (f:Fact) RETURN count(f) AS c")[0]["c"]
    console.print(f"\n[bold green]✅ Day loaded — {len(episodes)} slots, {total} total facts[/bold green]")
    return log

# ── Ground truth builder ────────────────────────────────────────────────────

def build_ground_truth(episodes: list[dict], time_slots: list[str]) -> list[dict]:
    """Extract verifiable facts straight from episode JSON."""
    gt = []
    for ep, ts in zip(episodes, time_slots):
        for room_id, room_data in ep["initial_home_config"]["rooms"].items():
            state = room_data.get("state", {})
            if state:
                gt.append({"time": ts, "entity": f"room:{room_id}",
                            "relation": "has_temperature",
                            "value": f"{state['temperature']/100:.2f}C"})
                gt.append({"time": ts, "entity": f"room:{room_id}",
                            "relation": "has_humidity",
                            "value": f"{state['humidity']/100:.2f}%"})
                gt.append({"time": ts, "entity": f"room:{room_id}",
                            "relation": "has_illuminance",
                            "value": f"{state['illuminance']:.1f}lux"})
            for device in room_data.get("devices", []):
                attrs = device.get("attributes", {})
                if "1.OnOff.OnOff" in attrs:
                    gt.append({"time": ts,
                                "entity": f"device:{device['device_id']}",
                                "relation": "is_on",
                                "value": str(attrs["1.OnOff.OnOff"])})
    return gt

def find_ground_truth(question: str, gt: list[dict]) -> list[dict]:
    q = question.lower()
    hits = []
    for entry in gt:
        eid  = entry["entity"].lower()
        rel  = entry["relation"].lower()
        match = (
            any(w in eid for w in q.split() if len(w) > 3)
            or ("temperature" in q and rel == "has_temperature")
            or ("humid"       in q and rel == "has_humidity")
            or ("illumin" in q or "bright" in q or "light" in q) and rel == "has_illuminance"
            or ("on" in q or "off" in q or "active" in q) and rel == "is_on"
        )
        if match:
            hits.append(entry)
    return hits[:8]

# ── Graph context builder ───────────────────────────────────────────────────

def _build_graph_context(question: str, db) -> str:
    q = question.lower()
    ROOMS = ["bathroom","bedroom","kitchen","living_room","utility_room","study_room","dining_room"]
    mentioned = [r for r in ROOMS if r.replace("_"," ") in q or r in q]

    all_facts = db.run("""
        MATCH (e:Entity)-[:HAS_FACT]->(f:Fact)
        RETURN e.entity_id AS entity_id, e.entity_type AS entity_type,
               f.relation AS relation, f.value AS value, f.timestamp AS timestamp
        ORDER BY f.timestamp ASC
    """)

    relevant = [r for r in all_facts if (
        any(rm.replace("_"," ") in r["entity_id"] for rm in mentioned)
        or any(w in r["entity_id"] for w in q.split() if len(w) > 3)
        or r["relation"] in {"has_temperature","has_humidity","is_on","has_illuminance"}
    )]
    if not relevant:
        relevant = all_facts[:80]

    lines = ["Timestamped facts from the home today (chronological):"]
    for r in relevant[:100]:
        lines.append(f"  {r['timestamp'][:16]}  {r['entity_id']:<38} "
                     f"{r['relation']:<20} → {r['value']}")
    return "\n".join(lines)

# ── Answer methods ──────────────────────────────────────────────────────────

def answer_baseline(question: str, llm) -> str:
    system = ("You are a smart home assistant with NO real-time data. "
              "Answer from general knowledge only. Be honest when you don't know current state.")
    try:
        return llm.generate([{"role":"system","content":system},
                              {"role":"user","content":question}])
    except Exception as e:
        return f"[Baseline error: {e}]"



# ── Graph-only tool interceptor ─────────────────────────────────────────────

def _graph_tool(tool_name: str, params: dict, db) -> dict:
    """
    Intercepts simulator tool calls and answers from the TKG graph instead.
    Used in day_sim mode where simulator is not running.
    """
    from tkg_agent.retrieval.retriever import TKGRetriever
    retriever = TKGRetriever(db, window_minutes=1440)

    if tool_name == "get_rooms":
        rooms = db.run("MATCH (e:Entity {entity_type:'room'}) RETURN e.entity_id AS eid")
        room_ids = [r["eid"].replace("room:","") for r in rooms]
        return {"status":{"code":200,"message":"OK"},
                "data":{"rooms":[{"room_id":r,"display_name":r.replace("_"," ").title()}
                                  for r in room_ids]}, "error":None}

    if tool_name == "get_room_devices":
        room_id = params.get("room_id","")
        rows = db.run("""
            MATCH (d:Entity {entity_type:'device'})-[:HAS_FACT]->(f:Fact)
            WHERE f.relation='located_in' AND f.value=$rv
            RETURN DISTINCT d.entity_id AS eid
        """, rv=f"room:{room_id}")
        devices = {}
        for r in rows:
            did = r["eid"].replace("device:","")
            dt_rows = db.run("""
                MATCH (e:Entity {entity_id:$eid})-[:HAS_FACT]->(f:Fact {relation:'device_type'})
                RETURN f.value AS dt ORDER BY f.timestamp DESC LIMIT 1
            """, eid=r["eid"])
            dt = dt_rows[0]["dt"] if dt_rows else "unknown"
            devices[did] = {"device_type": dt}
        return {"status":{"code":200},"data":devices,"error":None}

    if tool_name == "get_room_states":
        room_id = params.get("room_id","")
        eid = f"room:{room_id}"
        rows = db.run("""
            MATCH (e:Entity {entity_id:$eid})-[:HAS_FACT]->(f:Fact)
            WHERE f.relation IN ['has_temperature','has_humidity','has_illuminance','has_pm10']
            WITH f.relation AS rel, f ORDER BY f.timestamp DESC
            WITH rel, collect(f)[0] AS latest
            RETURN rel, latest.value AS val
        """, eid=eid)
        state = {}
        rel_map = {"has_temperature":"temperature","has_humidity":"humidity",
                   "has_illuminance":"illuminance","has_pm10":"pm10"}
        for r in rows:
            state[rel_map.get(r["rel"], r["rel"])] = r["val"]
        return {"status":{"code":200},"data":state,"error":None}

    if tool_name == "get_current_time":
        return {"status":{"code":200},"data":{"now":"2025-08-23 16:30:00"},"error":None}

    # Fallback: return graph snapshot for unknown tools
    return {"status":{"code":200},
            "data":{"message":f"Graph-only mode: {tool_name} answered from TKG"},
            "error":None}

def answer_tkg_agent(question: str, db, llm, sim_time: str) -> tuple[str, list[dict]]:
    """
    Run the full TKGReActAgent loop. Returns (final_answer, steps_log).
    Agent sees graph memory injected before each step — uses tools only if needed.
    """
    from tkg_agent.agent.tkg_agent import TKGReActAgent
    import tkg_agent.agent.base_agent as ba

    # Override system prompt: graph-first mode for day queries
    original_prompt = ba.SYSTEM_PROMPT
    ba.SYSTEM_PROMPT = original_prompt + """

[DAY QUERY MODE — READ THIS FIRST]
The TKG graph memory injected into this prompt already contains a FULL DAY of
timestamped home state data (06:00 to 16:30). Every device state, room temperature,
humidity, and illuminance reading is already in the graph.

RULE: Check the TKG GRAPH MEMORY section below BEFORE calling any tool.
- If the answer is present in the graph → respond with finish immediately.
- Only call a simulator tool if the specific fact you need is NOT in the graph.
- Temporal questions ("at 9am", "this morning", "after 2pm") → filter by timestamp in graph."""

    # Monkey-patch call_simulator_tool to use graph instead of HTTP
    import tkg_agent.agent.base_agent as _ba
    _orig_tool = _ba.call_simulator_tool
    _ba.call_simulator_tool = lambda name, params: _graph_tool(name, params, db)

    agent = TKGReActAgent(llm, db, max_steps=8, window_min=1440)
    steps_log = []

    try:
        result = agent.run(
            user_query=question,
            user_location=None,
            current_time=sim_time,
            episode=None,
        )
        for step in result.steps:
            steps_log.append({
                "step":   step.step,
                "action": step.action,
                "thought": step.thought or "",
                "observation": str(step.observation or "")[:200],
            })
            console.print(
                f"  [dim]Step {step.step}[/dim] [cyan]{step.action}[/cyan]"
                + (f"  — {(step.thought or '')[:80]}" if step.thought else "")
            )
            if step.observation and step.action != "finish":
                console.print(f"    [dim]{str(step.observation)[:120]}[/dim]")

        m = agent.get_tkg_metrics()
        console.print(f"  [dim]graph_hits={m['graph_hits']} facts_written={m['facts_written']}[/dim]")
        answer = result.final_answer
    except Exception as e:
        answer = f"[TKG Agent error: {e}]"
    finally:
        ba.SYSTEM_PROMPT = original_prompt   # always restore
        _ba.call_simulator_tool = _orig_tool  # restore tool

    return answer, steps_log

# ── Report saver ────────────────────────────────────────────────────────────

def save_report(session: list[dict], day_log: list[dict]):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = REPORT_DIR / f"session_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({"day_log": day_log, "queries": session}, f, indent=2)

    # Markdown
    md_path = REPORT_DIR / f"session_{ts}.md"
    with open(md_path, "w") as f:
        f.write("# TKG Day Simulation — Query Session Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Episodes loaded:** {len(day_log)}  \n")
        f.write(f"**Queries answered:** {len(session)}  \n\n")

        f.write("## Day Timeline\n\n")
        f.write("| Slot | Time | Query Preview |\n|---|---|---|\n")
        for e in day_log:
            f.write(f"| {e['slot']} | {e['sim_time']} | {e['query'][:60]} |\n")
        f.write("\n")

        for i, q in enumerate(session, 1):
            f.write(f"## Query {i}: {q['question']}\n\n")
            f.write(f"**Baseline:**  \n{q['baseline']}\n\n")
            f.write(f"**TKG Agent:**  \n{q['tkg_answer']}\n\n")
            if q.get("ground_truth"):
                f.write("**Ground Truth (from episode JSON):**\n\n")
                for g in q["ground_truth"]:
                    f.write(f"- `{g['time'][:16]}` `{g['entity']}` "
                            f"`{g['relation']}` → **{g['value']}**\n")
                f.write("\n")
            if q.get("steps"):
                f.write("**Agent Steps:**\n\n")
                for s in q["steps"]:
                    f.write(f"- Step {s['step']} `{s['action']}` — {s['thought'][:80]}\n")
                f.write("\n")
            f.write("---\n\n")

    console.print(f"\n[bold green]Report saved:[/bold green]")
    console.print(f"  JSON: {json_path}")
    console.print(f"  MD:   {md_path}")

# ── Chat loop ───────────────────────────────────────────────────────────────

def chat_loop(db, llm, day_log: list[dict], episodes: list[dict],
              time_slots: list[str], mode: str):

    gt = build_ground_truth(episodes, time_slots) if episodes else []
    latest_sim_time = day_log[-1]["sim_time"] if day_log else "2025-08-23 16:30:00"
    session: list[dict] = []

    console.print(f"\n[bold]Day loaded ({len(day_log)} slots). Ask anything about the home today.[/bold]")
    console.print(f"[dim]Mode: {mode.upper()} | Commands: timeline · facts · "
                  f"mode tkg|baseline|compare · report · exit[/dim]\n")

    current_mode = mode

    while True:
        try:
            question = Prompt.ask("[cyan]You[/cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Saving report and exiting...[/dim]")
            if session:
                save_report(session, day_log)
            break

        q = question.strip()
        if not q:
            continue

        # ── Commands ──────────────────────────────────────────────────────
        if q.lower() in ("exit", "quit", "q"):
            if session:
                save = Prompt.ask("Save session report? [y/n]", default="y")
                if save.lower() == "y":
                    save_report(session, day_log)
            console.print("[dim]Goodbye.[/dim]")
            break

        if q.lower() == "timeline":
            _print_timeline(day_log)
            continue

        if q.lower() == "facts":
            _print_facts(db)
            continue

        if q.lower() == "report":
            if session:
                save_report(session, day_log)
            else:
                console.print("[dim]No queries yet.[/dim]")
            continue

        if q.lower().startswith("mode "):
            current_mode = q.split()[1].lower()
            console.print(f"[dim]Mode → {current_mode.upper()}[/dim]")
            continue

        # ── Answer ────────────────────────────────────────────────────────
        record = {"question": q, "baseline": "", "tkg_answer": "",
                  "ground_truth": [], "steps": []}

        if current_mode in ("compare", "baseline"):
            console.print("[dim]Querying baseline...[/dim]")
            b_ans = answer_baseline(q, llm)
            record["baseline"] = b_ans
            console.print(Panel(b_ans,
                title="[yellow]Baseline — no memory[/yellow]",
                border_style="yellow"))

        if current_mode in ("compare", "tkg"):
            console.print("[dim]Running TKG Agent...[/dim]")
            t_ans, steps = answer_tkg_agent(q, db, llm, latest_sim_time)
            record["tkg_answer"] = t_ans
            record["steps"]      = steps
            console.print(Panel(t_ans,
                title="[green]TKG Agent — graph memory + reasoning loop[/green]",
                border_style="green"))

        # Ground truth panel
        gt_hits = find_ground_truth(q, gt)
        if gt_hits:
            record["ground_truth"] = gt_hits
            gt_text = "\n".join(
                f"  {g['time'][:16]}  {g['entity']:<40} {g['relation']:<20} → {g['value']}"
                for g in gt_hits
            )
            console.print(Panel(gt_text,
                title="[blue]Ground Truth — from episode JSON[/blue]",
                border_style="blue"))

        session.append(record)
        console.print()

# ── Helpers ─────────────────────────────────────────────────────────────────

def _print_timeline(day_log: list[dict]):
    t = Table(title="Day Simulation Timeline", show_header=True)
    t.add_column("Slot", width=5)
    t.add_column("Time", width=20)
    t.add_column("Query", width=70)
    for e in day_log:
        t.add_row(str(e["slot"]), e["sim_time"], e["query"])
    console.print(t)

def _print_facts(db):
    rows = db.run("""
        MATCH (e:Entity)-[:HAS_FACT]->(f:Fact)
        WHERE f.relation IN ['is_on','has_temperature','has_humidity','has_illuminance']
        RETURN f.timestamp AS ts, e.entity_id AS eid,
               f.relation AS rel, f.value AS val
        ORDER BY f.timestamp ASC LIMIT 120
    """)
    t = Table(title="Key Facts Across The Day", show_header=True)
    t.add_column("Time",     width=17)
    t.add_column("Entity",   width=38)
    t.add_column("Relation", width=18)
    t.add_column("Value",    width=16)
    for r in rows:
        t.add_row(r["ts"][:16], r["eid"], r["rel"], str(r["val"])[:16])
    console.print(t)

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", default="../SimuHome/data/benchmark")
    parser.add_argument("--n",            type=int, default=8)
    parser.add_argument("--load-only",    action="store_true")
    parser.add_argument("--clear",        action="store_true")
    parser.add_argument("--mode",         default="compare",
                        choices=["tkg","baseline","compare"])
    args = parser.parse_args()

    from tkg_agent.graph.neo4j_client import Neo4jClient
    from tkg_agent.agent.groq_provider import GroqProvider

    db  = Neo4jClient()
    db.create_constraints()
    llm = GroqProvider()

    episodes, time_slots, day_log = [], [], []

    if args.clear:
        console.print("[yellow]Clearing graph...[/yellow]")
        db.run("MATCH (f:Fact) DETACH DELETE f")
        db.run("MATCH (e:Entity) DETACH DELETE e")
        console.print("[green]Graph cleared.[/green]")

    if not args.load_only:
        episodes   = load_qt1_episodes(args.episodes_dir, args.n)
        time_slots = _make_time_slots(len(episodes))
        console.print(f"\n[bold]Simulating {len(episodes)} episodes:[/bold]")
        for i, (ep, ts) in enumerate(zip(episodes, time_slots), 1):
            console.print(f"  Slot {i}: {ts}  →  {ep['query'][:60]}...")
        console.print()
        day_log = build_day(episodes, db, time_slots)
    else:
        console.print("[dim]Using existing graph data.[/dim]")

    chat_loop(db, llm, day_log, episodes, time_slots, mode=args.mode)
    db.close()

if __name__ == "__main__":
    main()