"""
TKG-Enhanced ReAct Agent
========================
Extends the base ReAct loop with a mandatory retrieval step before
every LLM call. The agent:

  1. Bootstraps the TKG from the episode's initial_home_config
  2. Before each LLM call → retrieves relevant facts from Neo4j
  3. Injects the grounding block into the message as a system reminder
  4. After each tool observation → writes new facts into the TKG
  5. Logs how often graph memory influenced the decision

This is the agent we compare against BaseReActAgent on SimuHome episodes.
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field
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

# ── TKG memory injection prompt ───────────────────────────────────────────────

TKG_REMINDER = """
[TKG MEMORY — verified facts from graph, use these BEFORE calling tools]
{grounding_block}
[END TKG MEMORY — proceed with your action]
"""

# ── Entities to extract from a task query ────────────────────────────────────

ROOM_KEYWORDS = [
    "bathroom", "bedroom", "kitchen", "living_room",
    "utility_room", "study_room", "garage", "hallway",
]

RELATION_KEYWORDS = {
    "temperature": "has_temperature",
    "humidity":    "has_humidity",
    "illuminance": "has_illuminance",
    "light":       "has_illuminance",
    "pm10":        "has_pm10",
    "on":          "is_on",
    "off":         "is_on",
}


def _extract_rooms_from_text(text: str) -> list[str]:
    text_lower = text.lower().replace(" ", "_")
    return [r for r in ROOM_KEYWORDS if r in text_lower]


def _extract_devices_from_steps(steps: list[AgentStep]) -> list[str]:
    """Pull device_ids seen in previous tool observations."""
    devices: set[str] = set()
    for step in steps:
        obs = step.observation
        if isinstance(obs, dict):
            data = obs.get("data", {})
            if isinstance(data, dict):
                for key in data:
                    # get_room_devices returns {device_id: {device_type:...}}
                    if "_" in key and not key.startswith("room"):
                        devices.add(key)
    return list(devices)


# ── TKG Agent ─────────────────────────────────────────────────────────────────

class TKGReActAgent(BaseReActAgent):
    """
    ReAct agent with Temporal Knowledge Graph memory.

    Extra constructor args:
        db          : Neo4jClient (injected so it can be shared / tested)
        window_min  : how many minutes of history to surface (default 30)
    """

    def __init__(
        self,
        llm: Any,
        db: Neo4jClient,
        max_steps: int = 20,
        window_min: int = 30,
    ):
        super().__init__(llm, max_steps)
        self.db = db
        self.ingestor = EpisodeIngestor(db)
        self.retriever = TKGRetriever(db, window_minutes=window_min)

        # Metrics
        self._graph_hits: int = 0          # retrievals that returned real data
        self._graph_misses: int = 0        # retrievals that returned nothing
        self._facts_written: int = 0       # total facts ingested this episode
        self._retrieval_log: list[str] = []

    # ── Public: run with TKG ─────────────────────────────────────────────────

    def run(
        self,
        user_query: str,
        user_location: Optional[str] = None,
        current_time: Optional[str] = None,
        episode: Optional[dict] = None,       # full episode JSON for bootstrap
    ) -> AgentResult:

        # Reset metrics
        self._graph_hits = 0
        self._graph_misses = 0
        self._facts_written = 0
        self._retrieval_log = []
        self._events = []

        # ── Bootstrap TKG from episode initial state ──────────────────────
        episode_id = "live_run"
        if episode:
            episode_id = self.ingestor.bootstrap(episode)
            self._log("tkg_bootstrap", f"Bootstrapped episode {episode_id}")
        # Pre-extract rooms mentioned in the task
        task_rooms = _extract_rooms_from_text(user_query)
        if user_location:
            loc = user_location.replace("room:", "")
            if loc not in task_rooms:
                task_rooms.append(loc)

        # ── Build base messages ───────────────────────────────────────────
        system_msg = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(tools=_render_tool_table()),
        }
        user_msg = {
            "role": "user",
            "content": USER_PROMPT.format(
                user_query=user_query,
                user_location=user_location or "unknown",
                current_time=current_time or "unknown",
            ),
        }
        initial_grounding = self._retrieve(
         room_ids=task_rooms,
        device_ids=[],
        current_time=current_time,
)

        if initial_grounding:
            messages = [
                system_msg,
                {
                    "role": "system",
                    "content": TKG_REMINDER.format(
                        grounding_block=initial_grounding
                    ),
                },
                user_msg,
            ]
        else:
            messages = [system_msg, user_msg]
        

        steps: list[AgentStep] = []
        raw_responses: list[str] = []
        final_answer = ""
        consecutive_failures = 0


        for step_idx in range(1, self.max_steps + 1):

            # ── RETRIEVAL STEP (before every LLM call) ────────────────────
            seen_devices = _extract_devices_from_steps(steps)
            grounding = self._retrieve(task_rooms, seen_devices, current_time)

            if grounding:
                # Inject as a user turn so it's in the conversation history
                messages.append({
                    "role": "user",
                    "content": TKG_REMINDER.format(grounding_block=grounding),
                })

            # ── LLM call ──────────────────────────────────────────────────
            try:
                text = self.llm.generate(
                    messages,
                    response_format={"type": "json_schema"},
                )
            except Exception as e:
                raise RuntimeError(f"LLM call failed at step {step_idx}: {e}") from e

            raw_responses.append(text)
            # Replace the grounding injection with actual assistant reply
            # so we don't double-count it in history
            messages.append({"role": "assistant", "content": text})

            action, action_input, thought = self._parse(text)
            self._log("thought", thought)

            if action is None:
                consecutive_failures += 1
                obs = json.dumps({"error": "Invalid JSON output — try again"})
                self._log("parse_error", text[:200])
                messages.append({"role": "user", "content": f"observation: {obs}"})
                if consecutive_failures >= 3:
                    raise RuntimeError("3 consecutive parse failures — aborting")
                continue

            consecutive_failures = 0

            # ── finish ────────────────────────────────────────────────────
            if action.lower() == "finish":
                final_answer = action_input.get("answer", "")
                steps.append(AgentStep(step_idx, thought, "finish", action_input, None))
                self._log("finish", final_answer)
                break

            # ── tool call ─────────────────────────────────────────────────
            self._log("action", action)
            self._log("action_input", str(action_input))

            observation = call_simulator_tool(action, action_input)
            obs_str = json.dumps(observation, ensure_ascii=False)
            self._log("observation", obs_str[:300])

            messages.append({"role": "user", "content": f"observation: {obs_str}"})
            steps.append(AgentStep(step_idx, thought, action, action_input, observation))

            # ── INGEST observation back into TKG ──────────────────────────
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
            if written:
                self._log("tkg_ingest", f"step={step_idx} tool={action} facts={written}")

            # Update current_time if we just fetched it
            if action == "get_current_time":
                data = observation.get("data", {})
                if isinstance(data, dict) and "now" in data:
                    current_time = data["now"]

        if not final_answer:
            final_answer = "Max steps reached without finishing."

        # Attach TKG metrics to events
        self._log("tkg_metrics", json.dumps({
            "graph_hits":    self._graph_hits,
            "graph_misses":  self._graph_misses,
            "facts_written": self._facts_written,
            "steps":         len(steps),
        }))

        return AgentResult(
            steps=steps,
            final_answer=final_answer,
            events=list(self._events),
            raw_responses=raw_responses,
        )

    # ── Private: retrieval ────────────────────────────────────────────────────

    def _retrieve(
        self,
        room_ids: list[str],
        device_ids: list[str],
        current_time: Optional[str],
    ) -> str:
        """
        Build a grounding block from the TKG.
        Returns empty string if nothing useful is in the graph yet.
        """
        try:
            block = self.retriever.build_grounding_block(
                room_ids=room_ids or None,
                device_ids=device_ids or None,
                current_time=current_time,
            )
            # Consider it a hit if the block has real data beyond the headers
            is_meaningful = (
                "→" in block and
                "No facts found" not in block and
                "No changes since" not in block
            )
            if is_meaningful:
                self._graph_hits += 1
                self._retrieval_log.append(f"HIT rooms={room_ids} devices={device_ids}")
                return block
            else:
                self._graph_misses += 1
                return ""
        except Exception as e:
            self._log("tkg_retrieval_error", str(e))
            self._graph_misses += 1
            return ""

    # ── Private: sim time helper ──────────────────────────────────────────────

    def _get_sim_time(
        self,
        observation: dict,
        fallback: Optional[str],
        step_idx: int,
    ) -> str:
        """
        Extract simulator timestamp from an observation, or build a fallback.
        SimuHome observations don't always include a timestamp, so we use
        the last known current_time or a step-indexed placeholder.
        """
        # Some tools return {"data": {"now": "..."}}
        data = observation.get("data", {})
        if isinstance(data, dict) and "now" in data:
            return data["now"]

        # Use the last known sim time
        if fallback:
            return fallback

        # Use graph clock if available
        graph_time = self.retriever.get_current_time_from_graph()
        if graph_time:
            return graph_time

        return f"step_{step_idx:03d}"

    # ── Public: metrics ───────────────────────────────────────────────────────

    def get_tkg_metrics(self) -> dict:
        return {
            "graph_hits":    self._graph_hits,
            "graph_misses":  self._graph_misses,
            "facts_written": self._facts_written,
            "retrieval_log": self._retrieval_log,
        }