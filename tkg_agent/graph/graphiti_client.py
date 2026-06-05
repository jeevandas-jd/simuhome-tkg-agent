"""
Graphiti client for Temporal Knowledge Graph benchmarking.

Backend:
- Graphiti
- Neo4j storage
- Gemini LLM
"""

from __future__ import annotations

import os
from datetime import datetime, UTC
from typing import Optional, Any

from dotenv import load_dotenv

from graphiti_core import Graphiti

from graphiti_core.llm_client.gemini_client import (
    GeminiClient,
    LLMConfig,
)

from graphiti_core.embedder.gemini import (
    GeminiEmbedder,
    GeminiEmbedderConfig,
)

from graphiti_core.cross_encoder.gemini_reranker_client import (
    GeminiRerankerClient,
)

load_dotenv()


class GraphitiClient:

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
    ):

        self.uri = uri or os.getenv(
            "GRAPHITI_NEO4J_URI",
            "bolt://localhost:7687"
        )

        self.user = user or os.getenv(
            "GRAPHITI_NEO4J_USER",
            "neo4j"
        )

        self.password = password or os.getenv(
            "GRAPHITI_NEO4J_PASSWORD",
            "neo4j"
        )

        self.api_key = api_key or os.getenv(
            "GOOGLE_API_KEY"
        )

        self.client_model = "gemini-2.5-flash"

        self.embed_model = "embedding-001"

        self.reranker_model = (
            "gemini-2.5-flash-lite"
        )

        self.graphiti = Graphiti(

            uri=self.uri,

            user=self.user,

            password=self.password,

            llm_client=GeminiClient(
                config=LLMConfig(
                    api_key=self.api_key,
                    model=self.client_model,
                )
            ),

            embedder=GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=self.api_key,
                    model=self.embed_model,
                )
            ),

            cross_encoder=GeminiRerankerClient(
                config=LLMConfig(
                    api_key=self.api_key,
                    model=self.reranker_model,
                )
            ),
        )

    # -------------------------------------------------
    # lifecycle
    # -------------------------------------------------

    async def connect(self):

        await self.graphiti.build_indices()

    async def close(self):

        await self.graphiti.close()

    async def __aenter__(self):

        await self.connect()

        return self

    async def __aexit__(self, *args):

        await self.close()

    # -------------------------------------------------
    # health
    # -------------------------------------------------

    async def health_check(self):

        try:

            await self.graphiti.search(
                query="health"
            )

            return True

        except Exception:

            return False

    # -------------------------------------------------
    # write
    # -------------------------------------------------

    async def write_episode(
        self,
        episode_id: str,
        task_query: str,
        query_type: str = "",
        case: str = "",
        user_location: str = "",
        base_time: Optional[str] = None,
    ):

        text = f"""
Episode: {episode_id}

Query:
{task_query}

Query Type:
{query_type}

Case:
{case}

Location:
{user_location}

Time:
{base_time}
"""

        await self.graphiti.add_episode(

            name=episode_id,

            episode_body=text,

            source="text",

            reference_time=(
                datetime.fromisoformat(base_time)
                if base_time
                else datetime.now(UTC)
            ),
        )

    # -------------------------------------------------
    # write fact
    # -------------------------------------------------

    async def write_fact(
        self,
        subject_id: str,
        relation: str,
        value: Any,
        timestamp: Optional[str] = None,
        episode_id: str = "",
        step_id: int = 0,
    ):

        body = f"""
entity: {subject_id}
relation: {relation}
value: {value}
timestamp: {timestamp}
episode: {episode_id}
step: {step_id}
"""

        await self.graphiti.add_episode(

            name=f"{subject_id}_{relation}",

            episode_body=body,

            source="text",

            reference_time=(
                datetime.fromisoformat(timestamp)
                if timestamp
                else datetime.now(UTC)
            ),
        )

    # -------------------------------------------------
    # retrieval
    # -------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
    ):

        results = await self.graphiti.search(

            query=query,

            num_results=limit,
        )

        return results

    async def get_latest_fact(
        self,
        entity_id: str,
        relation: str,
    ):

        results = await self.search(
            f"""
latest {relation}
for {entity_id}
""",
            limit=1,
        )

        if not results:
            return None

        return results[0]

    async def get_recent_facts(
        self,
        entity_id: str,
        relation: str,
        since_timestamp: str,
        limit: int = 10,
    ):

        return await self.search(
            f"""
facts for {entity_id}
relation {relation}
after {since_timestamp}
""",
            limit,
        )

    async def get_entity_snapshot(
        self,
        entity_id: str,
    ):

        return await self.search(
            f"""
current state
of {entity_id}
"""
        )

    async def get_recently_changed(
        self,
        since_timestamp: str,
    ):

        return await self.search(
            f"""
what changed
after {since_timestamp}
"""
        )

    # -------------------------------------------------
    # delete
    # -------------------------------------------------

    async def clear_episode(
        self,
        episode_id: str,
    ):

        await self.graphiti.driver.execute_query(
            """
            MATCH (n)
            WHERE n.group_id=$episode
            DETACH DELETE n
            """,
            episode=episode_id,
        )