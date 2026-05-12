"""
TKG Retriever
=============
Provides the agent with structured answers to three classes of queries:
  1. Current state  — "what is device X's state right now?"
  2. Recent history — "what changed in the last N minutes?"
  3. Entity snapshot— "give me everything known about room:bathroom"

The retriever formats results as compact text blocks that can be
injected directly into the agent's system prompt or thought prefix.
"""
from __future__ import annotations

from datetime import datetime, timedelta, UTC
from typing import Optional

from tkg_agent.graph.neo4j_client import Neo4jClient


class TKGRetriever:
    def __init__(self, db: Neo4jClient, window_minutes: int = 30):
        self.db = db
        self.window_minutes = window_minutes

    # ── 1. Current state ─────────────────────────────────────────────────────

    def get_device_state(self, device_id: str) -> str:
        """
        Returns a formatted summary of a device's current known state.
        Example output:
            [TKG] device:bathroom_air_purifier_1
              is_on           → True  (t=2025-08-23 13:57:53)
              fan_mode        → 2     (t=2025-08-23 13:57:53)
              located_in      → room:bathroom
        """
        entity_id = f"device:{device_id}" if not device_id.startswith("device:") else device_id
        rows = self.db.get_entity_snapshot(entity_id)

        if not rows:
            return f"[TKG] No facts found for {entity_id}"

        lines = [f"[TKG] {entity_id}"]
        for row in rows:
            lines.append(f"  {row['relation']:<30} → {row['value']}  (t={row['timestamp']})")
        return "\n".join(lines)

    def get_room_state(self, room_id: str) -> str:
        """Returns a formatted summary of a room's current known environment."""
        entity_id = f"room:{room_id}" if not room_id.startswith("room:") else room_id
        rows = self.db.get_entity_snapshot(entity_id)

        if not rows:
            return f"[TKG] No facts found for {entity_id}"

        lines = [f"[TKG] {entity_id}"]
        for row in rows:
            lines.append(f"  {row['relation']:<30} → {row['value']}  (t={row['timestamp']})")
        return "\n".join(lines)

    def get_latest_value(self, entity_id: str, relation: str) -> Optional[str]:
        """
        Returns just the latest value string for a specific entity+relation.
        Returns None if not found.
        """
        fact = self.db.get_latest_fact(entity_id, relation)
        return fact["value"] if fact else None

    # ── 2. Recent history ─────────────────────────────────────────────────────

    def get_recent_changes(
        self,
        since_timestamp: Optional[str] = None,
        limit: int = 15,
    ) -> str:
        """
        Returns a formatted list of all fact changes since a given time.
        If since_timestamp is None, uses now - window_minutes.
        """
        if since_timestamp is None:
            since_dt = datetime.now(UTC) - timedelta(minutes=self.window_minutes)
            since_timestamp = since_dt.strftime("%Y-%m-%d %H:%M:%S")

        rows = self.db.get_recently_changed(since_timestamp, limit=limit)

        if not rows:
            return f"[TKG] No changes since {since_timestamp}"

        lines = [f"[TKG] Recent changes since {since_timestamp}"]
        for row in rows:
            lines.append(
                f"  {row['entity_id']:<40} {row['relation']:<25} → {row['value']}  (t={row['timestamp']})"
            )
        return "\n".join(lines)

    def get_device_history(
        self,
        device_id: str,
        relation: str,
        since_timestamp: Optional[str] = None,
    ) -> str:
        """
        Returns the history of a specific device attribute over time.
        Useful for: "has the AC been on recently?"
        """
        entity_id = f"device:{device_id}" if not device_id.startswith("device:") else device_id

        if since_timestamp is None:
            since_dt = datetime.now(UTC)- timedelta(minutes=self.window_minutes)
            since_timestamp = since_dt.strftime("%Y-%m-%d %H:%M:%S")

        rows = self.db.get_recent_facts(entity_id, relation, since_timestamp)

        if not rows:
            return f"[TKG] No history for {entity_id}.{relation} since {since_timestamp}"

        lines = [f"[TKG] History: {entity_id}.{relation}"]
        for row in rows:
            lines.append(f"  {row['timestamp']}  → {row['value']}")
        return "\n".join(lines)

    # ── 3. Pre-action grounding block ─────────────────────────────────────────

    def build_grounding_block(
        self,
        room_ids: Optional[list[str]] = None,
        device_ids: Optional[list[str]] = None,
        current_time: Optional[str] = None,
    ) -> str:
        """
        Build a compact memory block to inject before the agent decides its
        next action. Combines room states + device states + recent changes.

        This is the main entry point called by TKGAgent before every LLM call.
        """
        sections: list[str] = ["=== TKG MEMORY (retrieved before this action) ==="]

        # Room states
        if room_ids:
            for room_id in room_ids:
                sections.append(self.get_room_state(room_id))

        # Device states
        if device_ids:
            for device_id in device_ids:
                sections.append(self.get_device_state(device_id))

        # Recent changes
        since = current_time  # use sim time if provided, else fallback to wall clock
        recent = self.get_recent_changes(since_timestamp=None, limit=10)
        sections.append(recent)

        sections.append("=== END TKG MEMORY ===")
        return "\n\n".join(sections)

    # ── 4. Task-specific helpers ──────────────────────────────────────────────

    def is_device_on(self, device_id: str) -> Optional[bool]:
        """Returns True/False/None for is_on state of a device."""
        entity_id = f"device:{device_id}"
        val = self.get_latest_value(entity_id, "is_on")
        if val is None:
            return None
        return str(val).lower() in ("true", "1", "yes")

    def get_room_temperature(self, room_id: str) -> Optional[str]:
        entity_id = f"room:{room_id}"
        return self.get_latest_value(entity_id, "has_temperature")

    def get_current_time_from_graph(self) -> Optional[str]:
        """Return the last known simulator clock time stored in the graph."""
        return self.get_latest_value("system:clock", "current_time")

    def find_devices_in_room(self, room_id: str) -> list[str]:
        """Return all device entity_ids that are located_in a given room."""
        rows = self.db.run(
            """
            MATCH (d:Entity {entity_type: 'device'})-[:HAS_FACT {relation: 'located_in'}]->(f:Fact)
            WHERE f.value = $room_val
            WITH d.entity_id AS eid, f.timestamp AS ts
            ORDER BY ts DESC
            RETURN DISTINCT eid
            """,
            room_val=f"room:{room_id}",
        )
        return [r["eid"] for r in rows]