# tkg_agent/cli/graph_debug_cli.py

from __future__ import annotations

import cmd
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from tkg_agent.graph.neo4j_client import Neo4jClient
from tkg_agent.ingestor.episode_ingestor import EpisodeIngestor
from tkg_agent.retrieval.retriever import TKGRetriever


class GraphDebugCLI(cmd.Cmd):

    intro = """
=================================================
        TKG GRAPH DEBUG CLI
=================================================

This CLI DOES NOT use:
- LLMs
- Agents
- SimuHome live execution

It only tests:
- episode ingestion
- Neo4j graph state
- retrieval layer
- grounding generation

Type 'help' for commands.
"""

    prompt = "graph> "

    def __init__(self):

        super().__init__()

        self.db = Neo4jClient()
        self.ingestor = EpisodeIngestor(self.db)
        self.retriever = TKGRetriever(self.db)

        self.current_episode = None

    # =========================================================
    # LOAD EPISODE
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

        print("\n=== INGESTING EPISODE ===")

        episode_id = self.ingestor.bootstrap(episode)

        print(f"\nEpisode bootstrapped: {episode_id}")

    # =========================================================
    # SHOW EPISODE
    # =========================================================

    def do_episode(self, arg):

        if not self.current_episode:
            print("No episode loaded")
            return

        print("\n=== EPISODE ===\n")

        print(json.dumps({
            "query": self.current_episode.get("query"),
            "query_type": self.current_episode.get("query_type"),
            "case": self.current_episode.get("case"),
            "user_location": self.current_episode.get("user_location"),
        }, indent=2))

    # =========================================================
    # SHOW GRAPH STATS
    # =========================================================

    def do_stats(self, arg):

        rows = self.db.run(
            """
            MATCH (n)
            RETURN labels(n)[0] AS label, count(*) AS count
            """
        )

        print("\n=== GRAPH STATS ===\n")

        for r in rows:
            print(f"{r['label']:20} {r['count']}")

    # =========================================================
    # SHOW ALL ENTITIES
    # =========================================================

    def do_entities(self, arg):

        rows = self.db.run(
            """
            MATCH (e:Entity)
            RETURN e.entity_id AS entity_id,
                   e.entity_type AS entity_type
            ORDER BY entity_type
            """
        )

        print("\n=== ENTITIES ===\n")

        for r in rows:
            print(f"{r['entity_type']:12} {r['entity_id']}")

    # =========================================================
    # SHOW ROOM FACTS
    # =========================================================

    def do_room(self, arg):
        """
        room bedroom
        """

        room_id = arg.strip()

        if not room_id:
            print("Usage: room <room_id>")
            return

        if not room_id.startswith("room:"):
            room_id = f"room:{room_id}"

        print("\n=== ROOM STATE ===\n")

        print(
            self.retriever.get_room_state(room_id)
        )

    # =========================================================
    # SHOW DEVICE FACTS
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

        print("\n=== DEVICE STATE ===\n")

        print(
            self.retriever.get_device_state(device_id)
        )

    # =========================================================
    # SHOW RECENT FACTS
    # =========================================================

    def do_recent(self, arg):

        print("\n=== RECENT FACTS ===\n")

        print(
            self.retriever.get_recent_changes(
                since_timestamp=None,
                limit=20,
            )
        )

    # =========================================================
    # SHOW GROUNDING
    # =========================================================

    def do_grounding(self, arg):
        """
        grounding
        grounding bedroom
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

        print("\n=== GROUNDING BLOCK ===\n")

        print(block)

        print("\n=======================\n")

    # =========================================================
    # SHOW RAW FACTS
    # =========================================================

    def do_facts(self, arg):

        rows = self.db.run(
            """
            MATCH (e:Entity)-[:HAS_FACT]->(f:Fact)

            RETURN e.entity_id AS entity,
                   f.relation AS relation,
                   f.value AS value,
                   f.timestamp AS timestamp

            ORDER BY f.timestamp DESC
            LIMIT 100
            """
        )

        print("\n=== RAW FACTS ===\n")

        for r in rows:

            print(
                f"{r['entity']:40}"
                f"{r['relation']:25}"
                f"{str(r['value']):20}"
                f"{r['timestamp']}"
            )

    # =========================================================
    # VISUALIZE GRAPH
    # =========================================================

    def do_visualize(self, arg):
        """
        visualize
        visualize bedroom
        """

        try:

            query_filter = arg.strip()

            if query_filter:

                rows = self.db.run(
                    """
                    MATCH (e:Entity)-[:HAS_FACT]->(f:Fact)

                    WHERE e.entity_id CONTAINS $q
                       OR f.value CONTAINS $q

                    RETURN e.entity_id AS source,
                           f.relation AS relation,
                           f.value AS target

                    LIMIT 100
                    """,
                    q=query_filter,
                )

            else:

                rows = self.db.run(
                    """
                    MATCH (e:Entity)-[:HAS_FACT]->(f:Fact)

                    RETURN e.entity_id AS source,
                           f.relation AS relation,
                           f.value AS target

                    LIMIT 200
                    """
                )

            if not rows:
                print("No graph data found")
                return

            G = nx.MultiDiGraph()

            for r in rows:

                source = r["source"]
                relation = r["relation"]
                target = str(r["target"])

                G.add_node(source)
                G.add_node(target)

                G.add_edge(
                    source,
                    target,
                    label=relation,
                )

            plt.figure(figsize=(16, 12))

            pos = nx.spring_layout(
                G,
                seed=42,
                k=1.5,
            )

            nx.draw(
                G,
                pos,
                with_labels=True,
                node_size=2500,
                font_size=8,
                arrows=True,
            )

            edge_labels = {
                (u, v): d["label"]
                for u, v, d in G.edges(data=True)
            }

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=7,
            )

            plt.title("Temporal Knowledge Graph")

            plt.show()

        except Exception as e:
            print(f"Visualization error: {e}")

    # =========================================================
    # CLEAR GRAPH
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

    cli = GraphDebugCLI()

    cli.cmdloop()


if __name__ == "__main__":
    main()
