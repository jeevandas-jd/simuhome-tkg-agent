"""
TKG Agent
=========
Temporal Knowledge Graph enhanced smart-home agent for SimuHome.

Core loop:
    task
      ↓
    retrieve memory from Neo4j
      ↓
    inject grounding block
      ↓
    LLM reasoning
      ↓
    tool action
      ↓
    ingest observation into graph
      ↓
    repeat

This version is intentionally minimal:
- Groq-powered
- retrieval-before-action
- simulator-compatible JSON output
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq

from tkg_agent.graph.neo4j_client import Neo4jClient
from tkg_agent.ingestor.episode_ingestor import EpisodeIngestor
from tkg_agent.retrieval.retriever import TKGRetriever

load_dotenv()


SYSTEM_PROMPT = """
You are a smart-home reasoning agent operating inside SimuHome.

You MUST:
- reason carefully about current device and room states
- use the provided TKG memory grounding
- avoid hallucinating device states
- check state before acting
- prefer verification before control

You respond ONLY in valid JSON format:

{
  "thought": "...",
  "action": "...",
  "action_input": {...}
}

Available actions:
- get_rooms
- get_room_states
- get_room_devices
- get_device_structure
- execute_command
- write_attribute
- schedule_workflow
- get_current_time
- finish

Rules:
- Never output markdown
- Never output explanations outside JSON
- Always think before acting
"""


class TKGAgent:
    """
    Temporal Knowledge Graph enhanced agent.
    """

    def __init__(
        self,
        db: Neo4jClient,
        model: Optional[str] = None,
    ):
        self.db = db

        self.retriever = TKGRetriever(db)
        self.ingestor = EpisodeIngestor(db)

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "llama3-70b-8192"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Prompt building
    # ──────────────────────────────────────────────────────────────────────

    def build_prompt(
        self,
        task: str,
        room_ids: Optional[list[str]] = None,
        device_ids: Optional[list[str]] = None,
        current_time: Optional[str] = None,
        scratchpad: str = "",
    ) -> str:
        """
        Build grounded prompt with retrieved memory.
        """

        grounding = self.retriever.build_grounding_block(
            room_ids=room_ids,
            device_ids=device_ids,
            current_time=current_time,
        )

        return f"""
TASK:
{task}

{grounding}

PREVIOUS REASONING:
{scratchpad}

Decide the next best action.
Return ONLY valid JSON.
"""

    # ──────────────────────────────────────────────────────────────────────
    # LLM inference
    # ──────────────────────────────────────────────────────────────────────

    def think(
        self,
        task: str,
        room_ids: Optional[list[str]] = None,
        device_ids: Optional[list[str]] = None,
        current_time: Optional[str] = None,
        scratchpad: str = "",
    ) -> dict:
        """
        Run one grounded reasoning step.
        """

        prompt = self.build_prompt(
            task=task,
            room_ids=room_ids,
            device_ids=device_ids,
            current_time=current_time,
            scratchpad=scratchpad,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        text = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(text)
            return parsed

        except Exception:
            return {
                "thought": "Model returned invalid JSON.",
                "action": "finish",
                "action_input": {
                    "success": False,
                    "reason": text,
                },
            }

    # ──────────────────────────────────────────────────────────────────────
    # Observation ingestion
    # ──────────────────────────────────────────────────────────────────────

    def ingest_tool_observation(
        self,
        tool_name: str,
        tool_result: dict,
        timestamp: str,
        episode_id: str,
        step_id: int,
        tool_params: Optional[dict] = None,
    ) -> int:
        """
        Ingest simulator observation into graph memory.
        """

        return self.ingestor.ingest_observation(
            tool_name=tool_name,
            tool_result=tool_result,
            timestamp=timestamp,
            episode_id=episode_id,
            step_id=step_id,
            tool_params=tool_params,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Helper: infer relevant entities from task
    # ──────────────────────────────────────────────────────────────────────

    def infer_relevant_rooms(self, task: str) -> list[str]:
        """
        Very simple heuristic entity extraction.
        Replace later with proper parsing.
        """

        known_rooms = [
            "bedroom",
            "bathroom",
            "living_room",
            "kitchen",
            "utility_room",
        ]

        task_lower = task.lower()

        found = []

        for room in known_rooms:
            if room.replace("_", " ") in task_lower:
                found.append(room)

        return found

    # ──────────────────────────────────────────────────────────────────────
    # Main reasoning step
    # ──────────────────────────────────────────────────────────────────────

    def step(
        self,
        task: str,
        current_time: Optional[str] = None,
        scratchpad: str = "",
    ) -> dict:
        """
        One retrieval-grounded reasoning step.
        """

        room_ids = self.infer_relevant_rooms(task)

        device_ids = []

        for room_id in room_ids:
            device_ids.extend(
                self.retriever.find_devices_in_room(room_id)
            )

        result = self.think(
            task=task,
            room_ids=room_ids,
            device_ids=device_ids,
            current_time=current_time,
            scratchpad=scratchpad,
        )

        return result


# ──────────────────────────────────────────────────────────────────────────
# Minimal test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    db = Neo4jClient()

    agent = TKGAgent(db)

    task = """
    The bedroom temperature is high.
    Turn on the AC if needed.
    """

    result = agent.step(task)

    print("\n=== TKG Agent Output ===\n")
    print(json.dumps(result, indent=2))

    db.close()