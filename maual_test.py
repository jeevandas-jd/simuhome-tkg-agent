"""
manual_test.py
==============
Step-by-step manual tests for:
  1. Neo4jClient  — connection, constraints, entity, fact, episode
  2. EpisodeIngestor — bootstrap from real episode file
  3. TKGRetriever — current state, history, grounding block

Run from your repo root:
    python manual_test.py

Each test prints PASS / FAIL with details.
Stop at the first FAIL and fix before continuing.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Adjust this to point at your SimuHome benchmark directory
SIMUHOME_BENCHMARK = Path("../SimuHome/data/benchmark")
QT1_EPISODE = SIMUHOME_BENCHMARK / "qt1_feasible_seed_23.json"
QT4_EPISODE = SIMUHOME_BENCHMARK / "qt4-1_feasible_seed_30.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(msg: str):
    print(f"  {PASS}  {msg}")

def fail(msg: str, exc: Exception | None = None):
    print(f"  {FAIL}  {msg}")
    if exc:
        traceback.print_exc()

def load_episode(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — Neo4jClient
# ══════════════════════════════════════════════════════════════════════════════

def test_neo4j_client():
    section("BLOCK 1 — Neo4jClient")

    from tkg_agent.graph.neo4j_client import Neo4jClient

    # 1.1 Connection + health check
    try:
        db = Neo4jClient()
        assert db.health_check(), "health_check() returned False"
        ok("1.1  Connected to Neo4j and health_check passed")
    except Exception as e:
        fail("1.1  Cannot connect to Neo4j — is it running?", e)
        sys.exit(1)          # nothing else will work without DB

    # 1.2 Create constraints (idempotent)
    try:
        db.create_constraints()
        ok("1.2  create_constraints() ran without error")
    except Exception as e:
        fail("1.2  create_constraints() raised", e)

    # 1.3 Upsert entity
    try:
        db.upsert_entity("room:test_room", "room", "Test Room")
        rows = db.run("MATCH (e:Entity {entity_id: 'room:test_room'}) RETURN e.display_name AS dn")
        assert rows and rows[0]["dn"] == "Test Room", f"Got: {rows}"
        ok("1.3  upsert_entity() created node correctly")
    except Exception as e:
        fail("1.3  upsert_entity() failed", e)

    # 1.4 Write fact
    try:
        db.write_fact(
            subject_id="room:test_room",
            relation="has_temperature",
            value="22.50C",
            timestamp="2025-08-23 14:00:00",
            episode_id="test_episode_001",
            step_id=0,
        )
        rows = db.run(
            "MATCH (:Entity {entity_id:'room:test_room'})-[:HAS_FACT]->(f:Fact {relation:'has_temperature'}) "
            "RETURN f.value AS v"
        )
        assert rows and rows[0]["v"] == "22.50C", f"Got: {rows}"
        ok("1.4  write_fact() stored and retrieved correctly")
    except Exception as e:
        fail("1.4  write_fact() failed", e)

    # 1.5 Write fact with validity window
    try:
        db.write_fact(
            subject_id="room:test_room",
            relation="scheduled_action",
            value="workflow_abc123",
            timestamp="2025-08-23 14:00:00",
            episode_id="test_episode_001",
            step_id=1,
            valid_from="2025-08-23 14:05:00",
            valid_until="2025-08-23 14:30:00",
        )
        rows = db.run(
            "MATCH (:Entity {entity_id:'room:test_room'})-[:HAS_FACT]->(f:Fact {relation:'scheduled_action'}) "
            "RETURN f.valid_from AS vf, f.valid_until AS vu"
        )
        assert rows[0]["vf"] == "2025-08-23 14:05:00", f"Got valid_from: {rows[0]['vf']}"
        assert rows[0]["vu"] == "2025-08-23 14:30:00", f"Got valid_until: {rows[0]['vu']}"
        ok("1.5  write_fact() with validity window stored correctly")
    except Exception as e:
        fail("1.5  write_fact() with validity window failed", e)

    # 1.6 get_latest_fact
    try:
        # Write a second temperature fact (newer)
        db.write_fact(
            subject_id="room:test_room",
            relation="has_temperature",
            value="23.10C",
            timestamp="2025-08-23 14:05:00",
            episode_id="test_episode_001",
            step_id=2,
        )
        fact = db.get_latest_fact("room:test_room", "has_temperature")
        assert fact is not None, "Returned None"
        assert fact["value"] == "23.10C", f"Expected 23.10C, got {fact['value']}"
        ok(f"1.6  get_latest_fact() returned most recent value → {fact['value']}")
    except Exception as e:
        fail("1.6  get_latest_fact() failed", e)

    # 1.7 get_recent_facts
    try:
        rows = db.get_recent_facts(
            "room:test_room", "has_temperature",
            since_timestamp="2025-08-23 00:00:00",
            limit=10,
        )
        assert len(rows) >= 2, f"Expected ≥2 facts, got {len(rows)}"
        ok(f"1.7  get_recent_facts() returned {len(rows)} facts")
    except Exception as e:
        fail("1.7  get_recent_facts() failed", e)

    # 1.8 get_entity_snapshot
    try:
        rows = db.get_entity_snapshot("room:test_room")
        relations = {r["relation"] for r in rows}
        assert "has_temperature" in relations, f"Missing has_temperature in {relations}"
        ok(f"1.8  get_entity_snapshot() returned relations: {relations}")
    except Exception as e:
        fail("1.8  get_entity_snapshot() failed", e)

    # 1.9 get_recently_changed
    try:
        rows = db.get_recently_changed("2025-08-23 00:00:00", limit=20)
        assert len(rows) > 0, "No recent changes returned"
        ok(f"1.9  get_recently_changed() returned {len(rows)} rows")
    except Exception as e:
        fail("1.9  get_recently_changed() failed", e)

    # 1.10 write_episode
    try:
        db.write_episode(
            episode_id="test_episode_001",
            task_query="Is the air purifier on?",
            query_type="qt1",
            case="feasible",
            user_location="living_room",
            base_time="2025-08-23 13:57:53",
        )
        rows = db.run("MATCH (ep:Episode {episode_id:'test_episode_001'}) RETURN ep.query_type AS qt")
        assert rows and rows[0]["qt"] == "qt1", f"Got: {rows}"
        ok("1.10 write_episode() stored episode node correctly")
    except Exception as e:
        fail("1.10 write_episode() failed", e)

    # 1.11 clear_episode
    try:
        db.clear_episode("test_episode_001")
        rows = db.run("MATCH (f:Fact {episode_id:'test_episode_001'}) RETURN count(f) AS c")
        assert rows[0]["c"] == 0, f"Facts still remain: {rows[0]['c']}"
        ok("1.11 clear_episode() deleted all facts for episode")
    except Exception as e:
        fail("1.11 clear_episode() failed", e)

    # cleanup test entity
    db.run("MATCH (e:Entity {entity_id:'room:test_room'}) DETACH DELETE e")
    db.close()
    print()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — EpisodeIngestor (bootstrap from real QT1 file)
# ══════════════════════════════════════════════════════════════════════════════

def test_ingestor_qt1():
    section("BLOCK 2 — EpisodeIngestor (QT1 bootstrap)")

    if not QT1_EPISODE.exists():
        fail(f"2.0  Episode file not found: {QT1_EPISODE}")
        return

    from tkg_agent.graph.neo4j_client import Neo4jClient
    from tkg_agent.ingestor.episode_ingestor import EpisodeIngestor

    db = Neo4jClient()
    db.create_constraints()
    ingestor = EpisodeIngestor(db)
    episode = load_episode(QT1_EPISODE)

    # 2.1 bootstrap returns episode_id
    try:
        episode_id = ingestor.bootstrap(episode)
        assert isinstance(episode_id, str) and len(episode_id) > 0
        ok(f"2.1  bootstrap() returned episode_id: {episode_id}")
    except Exception as e:
        fail("2.1  bootstrap() raised", e)
        db.close()
        return

    # 2.2 Episode node written
    try:
        rows = db.run(f"MATCH (ep:Episode {{episode_id:'{episode_id}'}}) RETURN ep.query_type AS qt")
        assert rows and rows[0]["qt"] == "qt1"
        ok(f"2.2  Episode node found with query_type=qt1")
    except Exception as e:
        fail("2.2  Episode node missing", e)

    # 2.3 Rooms created as Entity nodes
    try:
        rows = db.run("MATCH (e:Entity {entity_type:'room'}) RETURN e.entity_id AS eid")
        room_ids = [r["eid"] for r in rows]
        assert len(room_ids) >= 1, f"No room entities found"
        ok(f"2.3  Room entities created: {room_ids}")
    except Exception as e:
        fail("2.3  Room entities missing", e)

    # 2.4 Room environment facts exist
    try:
        rows = db.run(
            "MATCH (e:Entity {entity_type:'room'})-[:HAS_FACT]->(f:Fact) "
            "WHERE f.relation IN ['has_temperature','has_humidity','has_illuminance','has_pm10'] "
            "RETURN count(f) AS c"
        )
        count = rows[0]["c"]
        assert count > 0, "No environment facts found"
        ok(f"2.4  Room environment facts written: {count} facts")
    except Exception as e:
        fail("2.4  Room environment facts missing", e)

    # 2.5 Device entities created
    try:
        rows = db.run("MATCH (e:Entity {entity_type:'device'}) RETURN count(e) AS c")
        count = rows[0]["c"]
        assert count > 0, "No device entities found"
        ok(f"2.5  Device entities created: {count} devices")
    except Exception as e:
        fail("2.5  Device entities missing", e)

    # 2.6 Check bathroom_air_purifier_1 specifically (the QT1 target device)
    try:
        rows = db.run(
            "MATCH (e:Entity {entity_id:'device:bathroom_air_purifier_1'})-[:HAS_FACT]->(f:Fact) "
            "RETURN f.relation AS rel, f.value AS val"
        )
        facts = {r["rel"]: r["val"] for r in rows}
        print(f"         bathroom_air_purifier_1 facts: {facts}")
        assert "is_on" in facts, f"Missing is_on. Got: {list(facts.keys())}"
        assert "located_in" in facts
        ok(f"2.6  Target device facts: is_on={facts.get('is_on')}, located_in={facts.get('located_in')}")
    except Exception as e:
        fail("2.6  Target device facts missing", e)

    # 2.7 User location written
    try:
        fact = db.get_latest_fact("user:resident_1", "located_in")
        assert fact is not None
        ok(f"2.7  User location: {fact['value']}")
    except Exception as e:
        fail("2.7  User location fact missing", e)

    # 2.8 ingest_observation for get_room_devices
    try:
        fake_obs = {
            "status": {"code": 200, "message": "OK"},
            "data": {
                "bathroom_air_purifier_1": {"device_type": "air_purifier"},
                "bathroom_humidifier_1":   {"device_type": "humidifier"},
            },
            "error": None,
        }
        written = ingestor.ingest_observation(
            tool_name="get_room_devices",
            tool_result=fake_obs,
            timestamp="2025-08-23 14:01:00",
            episode_id=episode_id,
            step_id=1,
            tool_params={"room_id": "bathroom"},
        )
        assert written > 0, f"Expected >0 facts written, got {written}"
        ok(f"2.8  ingest_observation(get_room_devices) wrote {written} facts")
    except Exception as e:
        fail("2.8  ingest_observation(get_room_devices) failed", e)

    # 2.9 ingest_observation for get_current_time
    try:
        fake_time_obs = {
            "status": {"code": 200},
            "data": {"now": "2025-08-23 14:02:00"},
            "error": None,
        }
        written = ingestor.ingest_observation(
            tool_name="get_current_time",
            tool_result=fake_time_obs,
            timestamp="2025-08-23 14:02:00",
            episode_id=episode_id,
            step_id=2,
        )
        assert written == 1
        ok(f"2.9  ingest_observation(get_current_time) stored clock fact")
    except Exception as e:
        fail("2.9  ingest_observation(get_current_time) failed", e)

    db.close()
    print()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — EpisodeIngestor (QT4 bootstrap — scheduled actions)
# ══════════════════════════════════════════════════════════════════════════════

def test_ingestor_qt4():
    section("BLOCK 3 — EpisodeIngestor (QT4 bootstrap + schedule_workflow)")

    if not QT4_EPISODE.exists():
        fail(f"3.0  Episode file not found: {QT4_EPISODE}")
        return

    from tkg_agent.graph.neo4j_client import Neo4jClient
    from tkg_agent.ingestor.episode_ingestor import EpisodeIngestor

    db = Neo4jClient()
    ingestor = EpisodeIngestor(db)
    episode = load_episode(QT4_EPISODE)

    # 3.1 bootstrap QT4
    try:
        episode_id = ingestor.bootstrap(episode)
        ok(f"3.1  QT4 bootstrap succeeded: {episode_id}")
    except Exception as e:
        fail("3.1  QT4 bootstrap failed", e)
        db.close()
        return

    # 3.2 ingest schedule_workflow observation
    try:
        fake_schedule_obs = {
            "status": {"code": 200},
            "data": {"workflow_id": "wf-test-abc123"},
            "error": None,
        }
        written = ingestor.ingest_observation(
            tool_name="schedule_workflow",
            tool_result=fake_schedule_obs,
            timestamp="2025-08-23 14:00:00",
            episode_id=episode_id,
            step_id=3,
            tool_params={
                "start_time": "2025-08-23 14:05:00",
                "tool_call": [
                    {
                        "tool": "write_attribute",
                        "args": {
                            "device_id": "utility_room_dimmable_light_1",
                            "endpoint_id": 1,
                            "cluster_id": "LevelControl",
                            "attribute_id": "CurrentLevel",
                            "value": 70,
                        },
                    }
                ],
            },
        )
        assert written >= 1, f"Expected ≥1 facts, got {written}"
        ok(f"3.2  ingest_observation(schedule_workflow) wrote {written} scheduled_action fact(s)")
    except Exception as e:
        fail("3.2  ingest_observation(schedule_workflow) failed", e)

    # 3.3 Check that valid_from is stored on the scheduled fact
    try:
        rows = db.run(
            "MATCH (:Entity {entity_id:'device:utility_room_dimmable_light_1'})"
            "-[:HAS_FACT]->(f:Fact {relation:'scheduled_action'}) "
            "RETURN f.valid_from AS vf, f.value AS wf_id"
        )
        assert rows, "No scheduled_action fact found"
        assert rows[0]["vf"] == "2025-08-23 14:05:00", f"Got valid_from: {rows[0]['vf']}"
        ok(f"3.3  scheduled_action valid_from stored: {rows[0]['vf']} | workflow={rows[0]['wf_id']}")
    except Exception as e:
        fail("3.3  scheduled_action valid_from check failed", e)

    db.close()
    print()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — TKGRetriever
# ══════════════════════════════════════════════════════════════════════════════

def test_retriever():
    section("BLOCK 4 — TKGRetriever")

    from tkg_agent.graph.neo4j_client import Neo4jClient
    from tkg_agent.ingestor.episode_ingestor import EpisodeIngestor
    from tkg_agent.retrieval.retriever import TKGRetriever

    db = Neo4jClient()
    ingestor = EpisodeIngestor(db)
    retriever = TKGRetriever(db, window_minutes=60)

    # Bootstrap QT1 episode so retriever has data to work with
    episode = load_episode(QT1_EPISODE)
    episode_id = ingestor.bootstrap(episode)

    # 4.1 get_device_state
    try:
        result = retriever.get_device_state("bathroom_air_purifier_1")
        print(f"\n         {result}\n")
        assert "[TKG]" in result
        assert "is_on" in result
        ok("4.1  get_device_state() returned formatted block with is_on")
    except Exception as e:
        fail("4.1  get_device_state() failed", e)

    # 4.2 get_room_state
    try:
        result = retriever.get_room_state("bathroom")
        print(f"\n         {result}\n")
        assert "[TKG]" in result
        assert "has_temperature" in result or "has_humidity" in result
        ok("4.2  get_room_state() returned formatted block with env vars")
    except Exception as e:
        fail("4.2  get_room_state() failed", e)

    # 4.3 get_latest_value
    try:
        val = retriever.get_latest_value("device:bathroom_air_purifier_1", "is_on")
        assert val is not None, "Returned None"
        ok(f"4.3  get_latest_value(is_on) → {val}")
    except Exception as e:
        fail("4.3  get_latest_value() failed", e)

    # 4.4 is_device_on
    try:
        on = retriever.is_device_on("bathroom_air_purifier_1")
        assert isinstance(on, bool) or on is None
        ok(f"4.4  is_device_on(bathroom_air_purifier_1) → {on}")
    except Exception as e:
        fail("4.4  is_device_on() failed", e)

    # 4.5 get_room_temperature
    try:
        temp = retriever.get_room_temperature("bathroom")
        ok(f"4.5  get_room_temperature(bathroom) → {temp}")
    except Exception as e:
        fail("4.5  get_room_temperature() failed", e)

    # 4.6 find_devices_in_room
    try:
        devices = retriever.find_devices_in_room("bathroom")
        assert len(devices) > 0, "No devices returned"
        ok(f"4.6  find_devices_in_room(bathroom) → {devices}")
    except Exception as e:
        fail("4.6  find_devices_in_room() failed", e)

    # 4.7 get_recent_changes
    try:
        result = retriever.get_recent_changes(since_timestamp="2025-08-23 00:00:00")
        print(f"\n         {result[:300]}...\n")
        assert "[TKG]" in result
        ok("4.7  get_recent_changes() returned formatted block")
    except Exception as e:
        fail("4.7  get_recent_changes() failed", e)

    # 4.8 build_grounding_block
    try:
        block = retriever.build_grounding_block(
            room_ids=["bathroom"],
            device_ids=["bathroom_air_purifier_1"],
            current_time="2025-08-23 14:00:00",
        )
        print(f"\n{block[:500]}\n")
        assert "TKG MEMORY" in block
        assert "END TKG MEMORY" in block
        ok("4.8  build_grounding_block() returned full memory block")
    except Exception as e:
        fail("4.8  build_grounding_block() failed", e)

    # 4.9 get_current_time_from_graph (requires clock fact)
    try:
        # plant a clock fact first
        ingestor.ingest_observation(
            tool_name="get_current_time",
            tool_result={"status": {"code": 200}, "data": {"now": "2025-08-23 14:10:00"}, "error": None},
            timestamp="2025-08-23 14:10:00",
            episode_id=episode_id,
            step_id=99,
        )
        t = retriever.get_current_time_from_graph()
        assert t == "2025-08-23 14:10:00", f"Got: {t}"
        ok(f"4.9  get_current_time_from_graph() → {t}")
    except Exception as e:
        fail("4.9  get_current_time_from_graph() failed", e)

    db.close()
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  TKG Agent — Manual Debug Test Suite")
    print("█"*60)
    print("\nRun blocks one at a time or all together.")
    print("Fix each FAIL before proceeding to the next block.\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block", type=int, default=0,
        help="Run only this block (1=Neo4j, 2=Ingestor QT1, 3=Ingestor QT4, 4=Retriever). 0=all"
    )
    args = parser.parse_args()

    blocks = {
        1: test_neo4j_client,
        2: test_ingestor_qt1,
        3: test_ingestor_qt4,
        4: test_retriever,
    }

    if args.block == 0:
        for fn in blocks.values():
            fn()
    else:
        blocks[args.block]()

    print("\n" + "─"*60)
    print("  Done. Fix any ❌ FAIL before moving to Phase 4.")
    print("─"*60 + "\n")