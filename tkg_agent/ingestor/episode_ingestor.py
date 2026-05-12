"""
Episode Ingestor
================
Converts a SimuHome episode JSON + live tool observations into
temporal triples that are written to Neo4j.

Two ingestion modes:
  1. bootstrap(episode)   — parse initial_home_config at episode start
  2. ingest_observation() — parse a tool response during the agent loop
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from tkg_agent.graph.neo4j_client import Neo4jClient


# ── Unit conversions (SimuHome raw → human readable) ──────────────────────────

def _temp_c(raw: float) -> str:
    return f"{raw / 100:.2f}C"

def _humidity_pct(raw: float) -> str:
    return f"{raw / 100:.2f}%"

def _illuminance_lux(raw: float) -> str:
    return f"{raw:.1f}lux"

def _pm10(raw: float) -> str:
    return f"{raw:.1f}ugm3"


class EpisodeIngestor:
    """
    Writes SimuHome data into the TKG as temporal triples.

    Every triple has the form:
        subject_id  : str  — e.g. "room:bathroom", "device:bathroom_ac_1"
        relation    : str  — e.g. "has_temperature", "is_on", "fan_mode"
        value       : Any  — the observed value
        timestamp   : str  — ISO-like simulator time string
        episode_id  : str
        step_id     : int
    """

    def __init__(self, db: Neo4jClient):
        self.db = db

    # ── Public: bootstrap from initial_home_config ────────────────────────────

    def bootstrap(self, episode: dict) -> str:
        """
        Parse initial_home_config and write all initial facts to the graph.
        Returns the episode_id (used for all subsequent writes in this run).
        """
        meta = episode["meta"]
        home = episode["initial_home_config"]
        base_time = home["base_time"]

        episode_id = f"{meta['query_type']}_{meta['case']}_seed{meta['seed']}"

        # Clear any previous facts for this episode (idempotent re-runs)
        self.db.clear_episode(episode_id)

        # Write episode node
        self.db.write_episode(
            episode_id=episode_id,
            query=episode["query"],
            query_type=meta["query_type"],
            case=meta["case"],
            user_location=episode["user_location"],
            base_time=base_time,
        )

        # Write user entity + location
        self.db.upsert_entity("user:resident_1", "user", "Resident")
        self.db.write_fact(
            subject_id="user:resident_1",
            relation="located_in",
            value=f"room:{episode['user_location']}",
            timestamp=base_time,
            episode_id=episode_id,
            step_id=0,
        )

        # Write rooms + devices + environment
        for room_id, room_data in home["rooms"].items():
            entity_id = f"room:{room_id}"
            self.db.upsert_entity(entity_id, "room", room_id.replace("_", " ").title())

            # Room environment state
            state = room_data.get("state", {})
            if state:
                self._write_room_state(entity_id, state, base_time, episode_id, step_id=0)

            # Devices in room
            for device in room_data.get("devices", []):
                self._write_device(device, room_id, base_time, episode_id, step_id=0)

        return episode_id

    # ── Public: ingest a live tool observation ────────────────────────────────

    def ingest_observation(
        self,
        tool_name: str,
        tool_result: dict,
        timestamp: str,
        episode_id: str,
        step_id: int,
        tool_params: Optional[dict] = None,
    ) -> int:
        """
        Parse a tool response from the live agent loop and write new facts.
        Returns the number of facts written.
        """
        tool_params = tool_params or {}
        data = tool_result.get("data", {})
        if not data:
            return 0

        written = 0

        if tool_name == "get_room_states":
            room_id = tool_params.get("room_id", "unknown")
            entity_id = f"room:{room_id}"
            self.db.upsert_entity(entity_id, "room", room_id)
            self._write_room_state(entity_id, data, timestamp, episode_id, step_id)
            written += 4  # temperature, humidity, illuminance, pm10

        elif tool_name == "get_room_devices":
            room_id = tool_params.get("room_id", "unknown")
            for device_id, device_info in data.items():
                entity_id = f"device:{device_id}"
                self.db.upsert_entity(entity_id, "device", device_id)
                self.db.write_fact(
                    subject_id=entity_id,
                    relation="located_in",
                    value=f"room:{room_id}",
                    timestamp=timestamp,
                    episode_id=episode_id,
                    step_id=step_id,
                )
                self.db.write_fact(
                    subject_id=entity_id,
                    relation="device_type",
                    value=device_info.get("device_type", "unknown"),
                    timestamp=timestamp,
                    episode_id=episode_id,
                    step_id=step_id,
                )
                written += 2

        elif tool_name == "get_device_structure":
            device_id = data.get("device_id", tool_params.get("device_id", "unknown"))
            entity_id = f"device:{device_id}"
            self.db.upsert_entity(entity_id, "device", device_id)
            # Extract key attributes from endpoints
            endpoints = data.get("endpoints", {})
            attrs = self._flatten_endpoints(endpoints)
            for relation, value in attrs.items():
                self.db.write_fact(
                    subject_id=entity_id,
                    relation=relation,
                    value=value,
                    timestamp=timestamp,
                    episode_id=episode_id,
                    step_id=step_id,
                )
                written += 1

        elif tool_name in ("execute_command", "write_attribute"):
            # Record the fact that a command was issued
            device_id = tool_params.get("device_id", "unknown")
            entity_id = f"device:{device_id}"
            self.db.upsert_entity(entity_id, "device", device_id)
            self.db.write_fact(
                subject_id=entity_id,
                relation=f"command_issued:{tool_name}",
                value=json.dumps(tool_params),
                timestamp=timestamp,
                episode_id=episode_id,
                step_id=step_id,
            )
            written += 1

        elif tool_name == "schedule_workflow":
            workflow_id = data.get("workflow_id", "unknown")
            # Record scheduled action as a fact on the device
            device_ids = self._extract_device_ids_from_workflow(tool_params)
            for device_id in device_ids:
                entity_id = f"device:{device_id}"
                self.db.upsert_entity(entity_id, "device", device_id)
                start_time = tool_params.get("start_time", timestamp)
                self.db.write_fact(
                    subject_id=entity_id,
                    relation="scheduled_action",
                    value=workflow_id,
                    timestamp=timestamp,
                    valid_from=start_time,
                    episode_id=episode_id,
                    step_id=step_id,
                )
                written += 1

        elif tool_name == "get_current_time":
            now_str = data.get("now", timestamp)
            # Store as a special "clock" entity
            self.db.upsert_entity("system:clock", "system", "Simulator Clock")
            self.db.write_fact(
                subject_id="system:clock",
                relation="current_time",
                value=now_str,
                timestamp=now_str,
                episode_id=episode_id,
                step_id=step_id,
            )
            written += 1

        return written

    # ── Private helpers ───────────────────────────────────────────────────────

    def _write_room_state(
        self,
        entity_id: str,
        state: dict,
        timestamp: str,
        episode_id: str,
        step_id: int,
    ) -> None:
        """Write the 4 environmental variables of a room."""
        mappings = {
            "temperature":  ("has_temperature",  _temp_c),
            "humidity":     ("has_humidity",      _humidity_pct),
            "illuminance":  ("has_illuminance",   _illuminance_lux),
            "pm10":         ("has_pm10",          _pm10),
        }
        for raw_key, (relation, convert) in mappings.items():
            if raw_key in state:
                self.db.write_fact(
                    subject_id=entity_id,
                    relation=relation,
                    value=convert(state[raw_key]),
                    timestamp=timestamp,
                    episode_id=episode_id,
                    step_id=step_id,
                )

    def _write_device(
        self,
        device: dict,
        room_id: str,
        timestamp: str,
        episode_id: str,
        step_id: int,
    ) -> None:
        """Write a device entity and its key attributes."""
        device_id = device["device_id"]
        device_type = device["device_type"]
        entity_id = f"device:{device_id}"

        self.db.upsert_entity(entity_id, "device", device_id)

        # located_in
        self.db.write_fact(
            subject_id=entity_id,
            relation="located_in",
            value=f"room:{room_id}",
            timestamp=timestamp,
            episode_id=episode_id,
            step_id=step_id,
        )

        # device_type
        self.db.write_fact(
            subject_id=entity_id,
            relation="device_type",
            value=device_type,
            timestamp=timestamp,
            episode_id=episode_id,
            step_id=step_id,
        )

        # Key attributes we care about
        attrs = device.get("attributes", {})
        key_attr_map = {
            "1.OnOff.OnOff":                    "is_on",
            "1.LevelControl.CurrentLevel":      "brightness_pct",
            "1.FanControl.FanMode":             "fan_mode",
            "1.FanControl.PercentCurrent":      "fan_speed_pct",
            "1.Thermostat.OccupiedCoolingSetpoint": "cooling_setpoint",
            "1.Thermostat.OccupiedHeatingSetpoint": "heating_setpoint",
            "1.OperationalState.OperationalState": "operational_state",
            "1.OperationalState.CountdownTime": "countdown_sec",
        }
        for attr_key, relation in key_attr_map.items():
            if attr_key in attrs:
                self.db.write_fact(
                    subject_id=entity_id,
                    relation=relation,
                    value=attrs[attr_key],
                    timestamp=timestamp,
                    episode_id=episode_id,
                    step_id=step_id,
                )

    def _flatten_endpoints(self, endpoints: dict) -> dict[str, Any]:
        """Extract readable key attributes from get_device_structure response."""
        result = {}
        for ep_id, ep_data in endpoints.items():
            clusters = ep_data.get("clusters", {})
            for cluster_name, cluster_data in clusters.items():
                attributes = cluster_data.get("attributes", {})
                for attr_name, attr_info in attributes.items():
                    if isinstance(attr_info, dict) and "value" in attr_info:
                        relation = f"{cluster_name}.{attr_name}".lower().replace(" ", "_")
                        result[relation] = attr_info["value"]
        return result

    def _extract_device_ids_from_workflow(self, tool_params: dict) -> list[str]:
        """Pull device_ids from a schedule_workflow tool_call list."""
        device_ids = []
        tool_calls = tool_params.get("tool_call", [])
        for tc in tool_calls:
            args = tc.get("args", {})
            if "device_id" in args:
                device_ids.append(args["device_id"])
        return device_ids