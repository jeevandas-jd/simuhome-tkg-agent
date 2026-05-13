from __future__ import annotations

import cmd
import json
from pathlib import Path

from tkg_agent.graph.neo4j_client import Neo4jClient
from tkg_agent.ingestor.episode_ingestor import EpisodeIngestor
from tkg_agent.retrieval.retriever import TKGRetriever

from tkg_agent.agent.base_agent import (
    call_simulator_tool,
)

# Optional
try:
    from tkg_agent.agent.tkg_react_agent import TKGReActAgent
except Exception:
    TKGReActAgent = None


class TKGDebugCLI(cmd.Cmd):

    intro = """
========================================
      TKG MANUAL DEBUG CLI
========================================
Type 'help' to see commands.
"""

    prompt = "tkg> "

    def __init__(self):
        super().__init__()

        self.db = Neo4jClient()
        self.ingestor = EpisodeIngestor(self.db)
        self.retriever = TKGRetriever(self.db)

        self.current_episode = None
        self.step_id = 0

    # =========================================================
    # EPISODE
    # =========================================================

    def do_load(self, arg):
        """
        load <episode_json_path>
        """

        path = Path(arg.strip())

        if not path.exists():
            print("Episode file not found")
            return

        with open(path) as f:
            episode = json.load(f)

        self.current_episode = episode

        ep_id = self.ingestor.bootstrap(episode)

        print(f"\nLoaded episode: {ep_id}")

    def do_episode(self, arg):

        if not self.current_episode:
            print("No episode loaded")
            return

        print(json.dumps({
            "query": self.current_episode.get("query"),
            "type": self.current_episode.get("query_type"),
            "case": self.current_episode.get("case"),
            "user_location": self.current_episode.get("user_location"),
        }, indent=2))

    # =========================================================
    # TOOL CALLS
    # =========================================================

    def do_call(self, arg):
        """
        call get_room_states room_id=bedroom
        """

        try:

            parts = arg.strip().split()

            tool_name = parts[0]

            tool_args = {}

            for item in parts[1:]:

                if "=" not in item:
                    continue

                k, v = item.split("=", 1)

                tool_args[k] = v

            print("\n=== TOOL CALL ===")
            print(tool_name)
            print(tool_args)

            obs = call_simulator_tool(tool_name, tool_args)

            print("\n=== OBSERVATION ===")
            print(json.dumps(obs, indent=2))

            self.step_id += 1

            self.ingestor.ingest_observation(
                tool_name=tool_name,
                tool_result=obs,
                timestamp=f"step_{self.step_id:03d}",
                episode_id="debug_episode",
                step_id=self.step_id,
                tool_params=tool_args,
            )

            print("\n[Observation ingested into TKG]")

        except Exception as e:
            print(f"ERROR: {e}")

    # =========================================================
    # GROUNDING
    # =========================================================

    def do_grounding(self, arg):
        """
        grounding
        grounding bathroom
        """

        room_ids = None

        arg = arg.strip()

        if arg:
            room_ids = [arg]

        block = self.retriever.build_grounding_block(
            room_ids=room_ids,
            device_ids=None,
            current_time=None,
        )

        print("\n=== GROUNDING ===\n")
        print(block)
        print("\n=================\n")

    # =========================================================
    # ROOM
    # =========================================================

    def do_room(self, arg):
        """
        room bedroom
        """

        room_id = arg.strip()

        if not room_id:
            print("Usage: room <room_id>")
            return

        print(self.retriever.get_room_state(room_id))

    # =========================================================
    # DEVICE
    # =========================================================

    def do_device(self, arg):
        """
        device bathroom_air_purifier_1
        """

        device_id = arg.strip()

        if not device_id:
            print("Usage: device <device_id>")
            return

        if not device_id.startswith("device:"):
            device_id = f"device:{device_id}"

        print(self.retriever.get_device_state(device_id))

    # =========================================================
    # RECENT
    # =========================================================

    def do_recent(self, arg):

        print(
            self.retriever.get_recent_changes(
                since_timestamp=None,
                limit=20,
            )
        )

    # =========================================================
    # GRAPH STATS
    # =========================================================

    def do_graph(self, arg):

        rows = self.db.run(
            """
            MATCH (n)
            RETURN labels(n)[0] AS label, count(*) AS count
            """
        )

        print("\n=== GRAPH ===")

        for r in rows:
            print(f"{r['label']}: {r['count']}")

    # =========================================================
    # CLEAR
    # =========================================================

    def do_clear(self, arg):

        self.db.run(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )

        print("Graph cleared")

    # =========================================================
    # AGENT
    # =========================================================

    def do_agent(self, arg):
        """
        agent Turn on the AC if needed
        """

        if TKGReActAgent is None:
            print("TKG agent unavailable")
            return

        print("\n[Agent integration pending]\n")

    # =========================================================
    # EXIT
    # =========================================================

    def do_exit(self, arg):
        return True

    def do_quit(self, arg):
        return True

    def do_EOF(self, arg):
        return True

    # =========================================================
    # CLEANUP
    # =========================================================

    def postloop(self):

        self.db.close()

        print("\nNeo4j connection closed")


def main():

    cli = TKGDebugCLI()

    cli.cmdloop()


if __name__ == "__main__":
    main()