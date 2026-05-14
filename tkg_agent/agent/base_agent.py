"""
Base ReAct Agent (Baseline)
===========================
A plain ReAct loop powered by Groq.
No TKG memory — this is the baseline we compare against.

Loop:
  1. Build system prompt with tool table
  2. Send user query
  3. Parse {thought, action, action_input} JSON
  4. Call SimuHome tool via HTTP
  5. Append observation, repeat until "finish"

Compatible with SimuHome's ReActStrategyAdapter.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

SIMULATOR_BASE = os.getenv("SIMULATOR_BASE", "http://localhost:8000")

# ── SimuHome tool registry ────────────────────────────────────────────────────

TOOLS = {
    "get_rooms": {
        "desc": "List all rooms in the home.",
        "params": {},
    },
    "get_room_devices": {
        "desc": "List devices in a room.",
        "params": {"room_id": "string"},
    },
    "get_room_states": {
        "desc": "Get environmental state of a room (temp, humidity, illuminance, pm10).",
        "params": {"room_id": "string"},
    },
    "get_device_structure": {
        "desc": "Get full cluster/attribute structure of a device.",
        "params": {"device_id": "string"},
    },
    "get_cluster_doc": {
        "desc": "Get documentation for a cluster.",
        "params": {"cluster_id": "string"},
    },
    "get_current_time": {
        "desc": "Get the current simulator time.",
        "params": {},
    },
    "get_environment_control_rules": {
        "desc": "Get control rules for bringing a room to a target environmental state.",
        "params": {"room_id": "string", "target_state": "object"},
    },
    "write_attribute": {
        "desc": "Write a value to a device attribute.",
        "params": {"device_id": "string", "endpoint_id": "int", "cluster_id": "string", "attribute_id": "string", "value": "any"},
    },
    "execute_command": {
        "desc": "Execute a command on a device cluster.",
        "params": {"device_id": "string", "endpoint_id": "int", "cluster_id": "string", "command_id": "string", "command_fields": "object"},
    },
    "schedule_workflow": {
        "desc": "Schedule a list of tool calls to run at a future simulator time.",
        "params": {"start_time": "string (YYYY-MM-DD HH:MM:SS)", "tool_call": "list of {tool, args}"},
    },
    "get_workflow_status": {
        "desc": "Check the status of a scheduled workflow.",
        "params": {"workflow_id": "string"},
    },
    "finish": {
        "desc": "End the episode with a final answer.",
        "params": {"answer": "string"},
    },
}


def _render_tool_table() -> str:
    lines = ["| Tool | Description | Parameters |",
             "|------|-------------|------------|"]
    for name, info in TOOLS.items():
        params = ", ".join(f"{k}: {v}" for k, v in info["params"].items()) or "none"
        lines.append(f"| {name} | {info['desc']} | {params} |")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a Smart Home Assistant operating under the ReAct framework with the Matter protocol.

[HOW TO RESPOND]
Generate ONLY this JSON on every turn, then stop:
{{"thought": "<reasoning>", "action": "<tool_name>", "action_input": "<JSON-formatted string>"}}

- One thought = one action. Never plan multiple actions at once.
- Wait for the observation before your next step.
- End with action="finish" and action_input={{"answer": "<your answer>"}}.

[AVAILABLE TOOLS]
{tools}

[CRITICAL TOOL RULES]
- room_id must be plain string: "bathroom" NOT "room:bathroom"
- device_id must be plain string: "bathroom_ac_1" NOT "device:bathroom_ac_1"
- Always call get_room_devices before using any device_id.
- For any timed/scheduled action, ALWAYS call get_current_time FIRST.
- OnOff.OnOff attribute is READ-ONLY — use execute_command with command_id="On" or "Off".
- Brightness: use execute_command with cluster_id="LevelControl" command_id="MoveToLevelWithOnOff" command_fields={{"level": <0-254>, "transitionTime": 0}}.

[SCHEDULE WORKFLOW FORMAT]
action_input must be:
{{"start_time": "YYYY-MM-DD HH:MM:SS", "steps": [{{"tool": "execute_command", "args": {{"device_id": "...", "endpoint_id": 1, "cluster_id": "OnOff", "command_id": "On", "command_fields": {{}}}}}}]}}
- "steps" key required (NOT "tool_call")
- start_time must be a FUTURE simulator time (use get_current_time + offset)

[UNIT CONVERSIONS]
- Temperature raw ÷ 100 = °C (2950 = 29.50°C)
- Humidity raw ÷ 100 = % (4502 = 45.02%)
- Illuminance = direct lux | PM10 = direct µg/m³
- Brightness level: percent × 2.54 rounded (70% = 178, 60% = 153)
"""

USER_PROMPT = """[TASK]
User Query: {user_query}
Current user location: {user_location}
Current time: {current_time}

Generate ONE JSON response. End with 'finish' when fully resolved."""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    step: int
    thought: Optional[str]
    action: str
    action_input: dict
    observation: Any

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class AgentResult:
    steps: list[AgentStep]
    final_answer: str
    events: list[dict] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)


# ── HTTP tool runner ──────────────────────────────────────────────────────────

def _strip_prefix(val: str, prefix: str) -> str:
    """Remove a prefix like 'room:' from IDs the LLM accidentally includes."""
    return val[len(prefix):] if val.startswith(prefix) else val


def call_simulator_tool(tool_name: str, params: dict) -> dict:
    """
    Call the SimuHome simulator REST API.
    All routes confirmed from GET /api/openapi.json.

    Key facts:
      - room_id must NOT have 'room:' prefix
      - schedule_workflow body: {start_time, steps: [{tool, args}]}
        'steps' key (NOT 'tool_call')
      - execute_command body: {endpoint_id, cluster_id, command_id, command_fields}
      - write_attribute body: {endpoint_id, cluster_id, attribute_id, value}
      - start_time must be a future simulator time
    """
    B = SIMULATOR_BASE + "/api"

    # Strip accidental prefixes the LLM may add
    room_id     = _strip_prefix(params.get("room_id", ""),    "room:")
    device_id   = _strip_prefix(params.get("device_id", ""),  "device:")
    workflow_id = params.get("workflow_id", "")
    state       = params.get("state", "")

    # Fix schedule_workflow: rename tool_call -> steps if LLM used old key
    def _fix_schedule_body(p: dict) -> dict:
        body = {k: v for k, v in p.items()}
        if "tool_call" in body and "steps" not in body:
            body["steps"] = body.pop("tool_call")
        # Ensure steps items use correct keys
        steps = body.get("steps", [])
        fixed = []
        for s in steps:
            step = dict(s)
            # LLM sometimes uses 'args', API expects 'args' — both fine
            # Strip room:/device: prefixes inside step args too
            if "args" in step:
                a = dict(step["args"])
                if "room_id" in a:
                    a["room_id"] = _strip_prefix(a["room_id"], "room:")
                if "device_id" in a:
                    a["device_id"] = _strip_prefix(a["device_id"], "device:")
                step["args"] = a
            fixed.append(step)
        body["steps"] = fixed
        return body

    # (method, url, body_dict)
    routes: dict[str, tuple] = {
        "get_rooms":          ("GET",  f"{B}/rooms",                                   None),
        "get_room_devices":   ("GET",  f"{B}/rooms/{room_id}/devices",                 None),
        "get_room_states":    ("GET",  f"{B}/rooms/{room_id}/states",                  None),
        "get_device_structure":("GET", f"{B}/devices/{device_id}/structure",           None),
        "get_cluster_doc":    ("GET",  f"{B}/devices/{device_id}/attributes",          None),
        "get_current_time":   ("GET",  f"{B}/time",                                    None),
        "get_workflow_status":("GET",  f"{B}/schedule/workflow/{workflow_id}/status",  None),
        "get_environment_control_rules": (
            "GET", f"{B}/environment/control_rules/{state}", None
        ),
        "write_attribute": (
            "POST", f"{B}/devices/{device_id}/attributes/write",
            {k: v for k, v in params.items() if k not in {"device_id", "room_id"}},
        ),
        "execute_command": (
            "POST", f"{B}/devices/{device_id}/commands",
            {k: v for k, v in params.items() if k not in {"device_id", "room_id"}},
        ),
        "schedule_workflow": (
            "POST", f"{B}/schedule/workflow",
            _fix_schedule_body(params),
        ),
    }

    if tool_name not in routes:
        return {"status": {"code": 400}, "error": f"Unknown tool: {tool_name}", "data": None}

    method, url, body = routes[tool_name]
    try:
        if method == "GET":
            resp = requests.get(url, timeout=15)
        else:
            resp = requests.post(url, json=body or {}, timeout=15)
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "status": {"code": 503},
            "error": "Simulator not reachable — run: uv run simuhome server-start",
            "data": None,
        }
    except Exception as e:
        return {"status": {"code": 500}, "error": str(e), "data": None}


# ── Base ReAct Agent ──────────────────────────────────────────────────────────

class BaseReActAgent:
    """
    Baseline ReAct agent — no TKG memory.
    Reads prompt context only.
    """

    def __init__(self, llm: Any, max_steps: int = 20):
        self.llm = llm
        self.max_steps = max_steps
        self._events: list[dict] = []

    def _log(self, event: str, payload: Any) -> None:
        self._events.append({"event": event, "payload": str(payload)})

    def _parse(self, text: str) -> tuple[Optional[str], dict, Optional[str]]:
        """Parse {thought, action, action_input} from LLM output."""
        import re
        text = text.strip()
        # Strip markdown fences
        if "```" in text:
            text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                return None, {}, None
            obj = json.loads(text[start:end])
            thought = obj.get("thought")
            action = obj.get("action")
            raw_input = obj.get("action_input", "{}")
            if isinstance(raw_input, str):
                try:
                    action_input = json.loads(raw_input) if raw_input.strip() else {}
                except json.JSONDecodeError:
                    action_input = {}
            elif isinstance(raw_input, dict):
                action_input = raw_input
            else:
                action_input = {}
            return action, action_input, thought
        except Exception:
            return None, {}, None

    def run(
        self,
        user_query: str,
        user_location: Optional[str] = None,
        current_time: Optional[str] = None,
    ) -> AgentResult:

        self._events = []
        system_msg = {"role": "system", "content": SYSTEM_PROMPT.format(tools=_render_tool_table())}
        user_msg = {
            "role": "user",
            "content": USER_PROMPT.format(
                user_query=user_query,
                user_location=user_location or "unknown",
                current_time=current_time or "unknown",
            ),
        }
        messages = [system_msg, user_msg]

        steps: list[AgentStep] = []
        raw_responses: list[str] = []
        final_answer = ""
        consecutive_failures = 0

        for step_idx in range(1, self.max_steps + 1):

            # ── LLM call ──────────────────────────────────────────────────
            try:
                text = self.llm.generate(messages, response_format={"type": "json_schema"})
            except Exception as e:
                raise RuntimeError(f"LLM call failed at step {step_idx}: {e}") from e

            raw_responses.append(text)
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
            self._log("action_input", action_input)

            observation = call_simulator_tool(action, action_input)
            obs_str = json.dumps(observation, ensure_ascii=False)
            self._log("observation", obs_str[:300])

            messages.append({"role": "user", "content": f"observation: {obs_str}"})
            steps.append(AgentStep(step_idx, thought, action, action_input, observation))

        if not final_answer:
            final_answer = "Max steps reached without finishing."

        return AgentResult(
            steps=steps,
            final_answer=final_answer,
            events=list(self._events),
            raw_responses=raw_responses,
        )