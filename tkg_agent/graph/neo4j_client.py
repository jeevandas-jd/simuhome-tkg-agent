"""
Neo4j client for the Temporal Knowledge Graph.
Handles connection, schema constraints, and all write/read operations.
"""
from __future__ import annotations

import os
from datetime import datetime,UTC
from typing import Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


# ── Connection ────────────────────────────────────────────────────────────────

class Neo4jClient:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "http://localhost:7474")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def run(self, cypher: str, **params) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]

    # ── Schema setup ──────────────────────────────────────────────────────────

    def create_constraints(self):
        """Create uniqueness constraints and indexes on first run."""
        constraints = [
            # Unique entity nodes
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            # Index on Fact timestamps for fast range queries
            "CREATE INDEX fact_timestamp IF NOT EXISTS FOR (f:Fact) ON (f.timestamp)",
            "CREATE INDEX fact_valid_from IF NOT EXISTS FOR (f:Fact) ON (f.valid_from)",
            "CREATE INDEX fact_valid_until IF NOT EXISTS FOR (f:Fact) ON (f.valid_until)",
            # Index on episode id
            "CREATE INDEX episode_id IF NOT EXISTS FOR (e:Episode) ON (e.episode_id)",
        ]
        for c in constraints:
            try:
                self.run(c)
            except Exception:
                pass  # constraint may already exist

    # ── Entity upsert ─────────────────────────────────────────────────────────

    def upsert_entity(self, entity_id: str, entity_type: str, display_name: str = "") -> None:
        """
        Create or update an Entity node.
        entity_type: 'room' | 'device' | 'user' | 'environment'
        """
        self.run(
            """
            MERGE (e:Entity {entity_id: $entity_id})
            SET e.entity_type = $entity_type,
                e.display_name = $display_name,
                e.updated_at   = $now
            """,
            entity_id=entity_id,
            entity_type=entity_type,
            display_name=display_name or entity_id,
            now=datetime.utcnow().isoformat(),
        )

    # ── Temporal fact write ───────────────────────────────────────────────────

    def write_fact(
        self,
        subject_id: str,
        relation: str,
        value: Any,
        timestamp: str,
        episode_id: str,
        step_id: int,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
    ) -> None:
        """
        Write a temporal fact triple:
            (subject) -[relation]-> Fact(value, timestamp, validity window)

        Schema:
            (:Entity {entity_id}) -[:HAS_FACT {relation}]-> (:Fact {value, timestamp, ...})
        """
        self.run(
            """
            MATCH (e:Entity {entity_id: $subject_id})
            CREATE (f:Fact {
                relation:    $relation,
                value:       $value,
                timestamp:   $timestamp,
                valid_from:  $valid_from,
                valid_until: $valid_until,
                episode_id:  $episode_id,
                step_id:     $step_id,
                created_at:  $now
            })
            CREATE (e)-[:HAS_FACT {relation: $relation}]->(f)
            """,
            subject_id=subject_id,
            relation=relation,
            value=str(value),
            timestamp=timestamp,
            valid_from=valid_from or timestamp,
            valid_until=valid_until or "",
            episode_id=episode_id,
            step_id=step_id,
            now=datetime.utcnow().isoformat(),
        )

    # ── Episode node ──────────────────────────────────────────────────────────

    def write_episode(
        self,
        episode_id: str,
        task_query: str,
        query_type: str,
        case: str,  
        user_location: str,
        base_time: str,
    ) -> None:

        self.run(
            """
            MERGE (ep:Episode {episode_id: $episode_id})
            SET ep.task_query    = $task_text,
                ep.query_type    = $query_type,
                ep.case          = $case,
                ep.user_location = $user_location,
                ep.base_time     = $base_time,
                ep.created_at    = $now
            """,
            episode_id=episode_id,
            task_text=task_query,
            query_type=query_type,
            case=case,
            user_location=user_location,
            base_time=base_time,
            now=datetime.now(UTC).isoformat(),
        )
    # ── Retrieval: current state ──────────────────────────────────────────────

    def get_latest_fact(self, entity_id: str, relation: str) -> Optional[dict]:
        """
        Return the most recent fact for a given entity + relation.
        Used for: "what is the current state of device X?"
        """
        rows = self.run(
            """
            MATCH (e:Entity {entity_id: $entity_id})-[:HAS_FACT {relation: $relation}]->(f:Fact)
            RETURN f.value AS value, f.timestamp AS timestamp,
                   f.valid_from AS valid_from, f.valid_until AS valid_until,
                   f.episode_id AS episode_id, f.step_id AS step_id
            ORDER BY f.timestamp DESC
            LIMIT 1
            """,
            entity_id=entity_id,
            relation=relation,
        )
        return rows[0] if rows else None

    # ── Retrieval: recent history ─────────────────────────────────────────────

    def get_recent_facts(
        self,
        entity_id: str,
        relation: str,
        since_timestamp: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Return all facts for entity+relation after a given timestamp.
        Used for: "what changed in the last N minutes?"
        """
        rows = self.run(
            """
            MATCH (e:Entity {entity_id: $entity_id})-[:HAS_FACT {relation: $relation}]->(f:Fact)
            WHERE f.timestamp >= $since_timestamp
            RETURN f.value AS value, f.timestamp AS timestamp,
                   f.valid_from AS valid_from, f.valid_until AS valid_until,
                   f.episode_id AS episode_id, f.step_id AS step_id
            ORDER BY f.timestamp DESC
            LIMIT $limit
            """,
            entity_id=entity_id,
            relation=relation,
            since_timestamp=since_timestamp,
            limit=limit,
        )
        return rows

    # ── Retrieval: all facts for an entity ───────────────────────────────────

    def get_entity_snapshot(self, entity_id: str) -> list[dict]:
        """
        Return the latest fact for every relation of an entity.
        Used for: "give me the full current state of room:bedroom"
        """
        rows = self.run(
            """
            MATCH (e:Entity {entity_id: $entity_id})-[:HAS_FACT]->(f:Fact)
            WITH f.relation AS relation, f, e
            ORDER BY f.timestamp DESC
            WITH relation, collect(f)[0] AS latest
            RETURN relation,
                   latest.value      AS value,
                   latest.timestamp  AS timestamp,
                   latest.valid_from AS valid_from,
                   latest.episode_id AS episode_id
            """,
            entity_id=entity_id,
        )
        return rows

    # ── Retrieval: recently changed entities ─────────────────────────────────

    def get_recently_changed(self, since_timestamp: str, limit: int = 20) -> list[dict]:
        """
        Return all fact changes across all entities after a timestamp.
        Used for: "what happened recently in the home?"
        """
        rows = self.run(
            """
            MATCH (e:Entity)-[:HAS_FACT]->(f:Fact)
            WHERE f.timestamp >= $since_timestamp
            RETURN e.entity_id AS entity_id, e.entity_type AS entity_type,
                   f.relation  AS relation,  f.value       AS value,
                   f.timestamp AS timestamp, f.episode_id  AS episode_id
            ORDER BY f.timestamp DESC
            LIMIT $limit
            """,
            since_timestamp=since_timestamp,
            limit=limit,
        )
        return rows

    # ── Utility ───────────────────────────────────────────────────────────────

    def clear_episode(self, episode_id: str) -> None:
        """Remove all facts written during a specific episode (for reruns)."""
        self.run(
            """
            MATCH (f:Fact {episode_id: $episode_id})
            DETACH DELETE f
            """,
            episode_id=episode_id,
        )

    def health_check(self) -> bool:
        try:
            self.run("RETURN 1 AS ok")
            return True
        except Exception:
            return False
