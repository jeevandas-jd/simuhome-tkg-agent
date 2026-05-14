"""
TKG-Enhanced ReAct Agent
========================
Extends the base ReAct loop with a mandatory retrieval step before
every LLM call. The agent:

  1. Bootstraps the TKG from the episode's initial_home_config
  2. Before each LLM call retrieves relevant facts from Neo4j
  3. Updates system message[0] with grounding block (not a user turn)
  4. After each tool observation writes new facts into the TKG
  5. Logs graph hits, misses, and facts written per episode
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from dotenv import load_dotenv

from tkg_agent.agent.base_agent import (
    AgentResult,
    AgentStep,
    BaseReActAgent,
    SYSTEM_PROMPT,
    USER_PROMPT,
    _render_tool_table,
    call_simulator_tool,
)
from tkg_agent.graph.neo4j_client import Neo4jClient
from tkg_agent.ingestor.episode_ingestor import EpisodeIngestor
from tkg_agent.retrieval.retriever import TKGRetriever

load_dotenv()

ROOM_KEYWORDS = [
    "bathroom", "bedroom", "kitchen", "living_room",
    "utility_room", "study_room", "garage", "hallway",
]


def _extract_rooms_from_text(text: str) -> list[str]:
    text_lower = text.lower().replace(" ", "_")
    return [r for r in ROOM_KEYWORDS if r in text_lower]


def _extract_devices_from_steps(steps: list[AgentStep]) -> list[str]:
    """Pull device_ids from get_room_devices observations. Strip any prefixes."""
    devices: set[str] = set()
    for step in steps:
        if step.action != "get_room_devices":
            continue
        obs = step.observation
        if isinstance(obs, dict):
            data = obs.get("data", {})
            if isinstance(data, dict):
                for key in data:
                    # keys are plain device_ids like "bathroom_air_purifier_1"
                    if isinstance(data[key], dict) and "device_type" in data[key]:
                        devices.add(key)
    return list(devices)


class TKGReActAgent(BaseReActAgent):

    def __init__(self, llm: Any, db: Neo4jClient, max_steps: int = 20, window_min: int = 30):
        super().__init__(llm, max_steps)
        self.db = db
        self.ingestor = EpisodeIngestor(db)
        self.retriever = TKGRetriever(db, window_minutes=window_min)
        self._graph_hits: int = 0
        self._graph_misses: int = 0
        self._facts_written: int = 0
        self._retrieval_log: list[str] = []

    def run(
        self,
        user_query: str,
        user_location: Optional[str] = None,
        current_time: Optional[str] = None,
        episode: Optional[dict] = None,
    ) -> AgentResult:

        self._graph_hits = 0
        self._graph_misses = 0
        self._facts_written = 0
        self._retrieval_log = []
        self._events = []

        episode_id = "live_run"
        if episode:
            episode_id = self.ingestor.bootstrap(episode)
            self._log("tkg_bootstrap", f"episode={episode_id}")

        task_rooms = _extract_rooms_from_text(user_query)
        if user_location:
            loc = user_location.replace("room:", "")
            if loc not in task_rooms:
                task_rooms.append(loc)

        base_system = SYSTEM_PROMPT.format(tools=_render_tool_table())

        messages = [
            {"role": "system", "content": base_system},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    user_query=user_query,
                    user_location=user_location or "unknown",
                    current_time=current_time or "unknown",
                ),
            },
        ]

        steps: list[AgentStep] = []
        raw_responses: list[str] = []
        final_answer = ""
        consecutive_failures = 0

        for step_idx in range(1, self.max_steps + 1):

            # 1. Retrieve and update system message
            seen_devices = _extract_devices_from_steps(steps)
            grounding = self._retrieve(task_rooms, seen_devices, current_time)

            if grounding:
                tkg_section = (
                    "\n\n[TKG GRAPH MEMORY — pre-verified facts from home state]\n"
                    "CRITICAL: If the fact needed to answer the user query is present below,\n"
                    "respond with action=\"finish\" immediately. Do NOT call a tool\n"
                    "to re-verify something the graph already confirms.\n\n"
                    + grounding
                    + "\n[END TKG GRAPH MEMORY]"
                )
                messages[0] = {"role": "system", "content": base_system + tkg_section}

            # 2. Trim history to keep context small (system + user_task + last 6 turns)
            if len(messages) > 10:
                messages = messages[:2] + messages[-8:]

            # 2. LLM call
            try:
                text = self.llm.generate(messages, response_format={"type": "json_schema"})
            except Exception as e:
                raise RuntimeError(f"LLM call failed at step {step_idx}: {e}") from e

            raw_responses.append(text)
            messages.append({"role": "assistant", "content": text})

            action, action_input, thought = self._parse(text)
            self._log("thought", thought or "")

            if action is None:
                consecutive_failures += 1
                messages.append({"role": "user", "content": 'observation: {"error": "Invalid JSON"}'})
                if consecutive_failures >= 3:
                    raise RuntimeError("3 consecutive parse failures")
                continue

            consecutive_failures = 0

            # 3. Finish
            if action.lower() == "finish":
                final_answer = action_input.get("answer", "")
                steps.append(AgentStep(step_idx, thought, "finish", action_input, None))
                self._log("finish", final_answer)
                break

            # 4. Tool call
            self._log("action", action)
            observation = call_simulator_tool(action, action_input)
            obs_str = json.dumps(observation, ensure_ascii=False)
            self._log("observation", obs_str[:300])
            messages.append({"role": "user", "content": f"observation: {obs_str}"})
            steps.append(AgentStep(step_idx, thought, action, action_input, observation))

            # 5. Ingest into TKG
            sim_time = self._get_sim_time(observation, current_time, step_idx)
            written = self.ingestor.ingest_observation(
                tool_name=action,
                tool_result=observation,
                timestamp=sim_time,
                episode_id=episode_id,
                step_id=step_idx,
                tool_params=action_input,
            )
            self._facts_written += written

            if action == "get_current_time":
                data = observation.get("data", {})
                if isinstance(data, dict) and "now" in data:
                    current_time = data["now"]

        if not final_answer:
            final_answer = "Max steps reached without finishing."

        self._log("tkg_metrics", json.dumps({
            "graph_hits": self._graph_hits,
            "graph_misses": self._graph_misses,
            "facts_written": self._facts_written,
            "steps": len(steps),
        }))

        return AgentResult(
            steps=steps,
            final_answer=final_answer,
            events=list(self._events),
            raw_responses=raw_responses,
        )

    def _retrieve(self, room_ids: list[str], device_ids: list[str], current_time: Optional[str]) -> str:
        try:
            sections: list[str] = []
            has_data = False

            for room_id in (room_ids or []):
                snippet = self.retriever.get_room_state(room_id)
                if "No facts found" not in snippet:
                    sections.append(snippet)
                    has_data = True

            for device_id in (device_ids or []):
                snippet = self.retriever.get_device_state(device_id)
                if "No facts found" not in snippet:
                    sections.append(snippet)
                    has_data = True

            # Use sim-time anchor so bootstrap facts are always in window
            since = current_time or "2000-01-01 00:00:00"
            # Only surface is_on, device_type, has_temperature — skip noisy attributes
            KEY_RELATIONS = {"is_on", "device_type", "located_in", "has_temperature",
                             "has_humidity", "has_illuminance", "scheduled_action", "countdown_sec"}
            recent_rows = self.db.get_recently_changed(since, limit=30)
            key_rows = [r for r in recent_rows if r['relation'] in KEY_RELATIONS][:8]
            if key_rows:
                lines = ["[TKG] Relevant facts:"]
                for row in key_rows:
                    lines.append(f"  {row['entity_id']:<35} {row['relation']:<20} -> {row['value']}")
                sections.append("\n".join(lines))
                has_data = True

            if has_data:
                self._graph_hits += 1
                self._retrieval_log.append(f"HIT rooms={room_ids} devices={device_ids}")
                return "\n\n".join(sections)
            else:
                self._graph_misses += 1
                return ""

        except Exception as e:
            self._log("tkg_retrieval_error", str(e))
            self._graph_misses += 1
            return ""

    def _get_sim_time(self, observation: dict, fallback: Optional[str], step_idx: int) -> str:
        data = observation.get("data", {})
        if isinstance(data, dict) and "now" in data:
            return data["now"]
        if fallback:
            return fallback
        graph_time = self.retriever.get_current_time_from_graph()
        if graph_time:
            return graph_time
        return f"step_{step_idx:03d}"

    def get_tkg_metrics(self) -> dict:
        return {
            "graph_hits":    self._graph_hits,
            "graph_misses":  self._graph_misses,
            "facts_written": self._facts_written,
            "retrieval_log": self._retrieval_log,
        }