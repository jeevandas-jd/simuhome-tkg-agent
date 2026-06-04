#using google gemini

import os
from dotenv import load_dotenv
load_dotenv()
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient



class GraphitiClient:
    def __init__(self):
        self.uri =os.getenv("NEO4J_URI", "http://localhost:7474")
        self.user =os.getenv("NEO4J_USER", "neo4j")
        self.password =os.getenv("NEO4J_PASSWORD", "neo4j")
        self._api_key=os.getenv("GOOGLE_API_KEY")
        self.cliet_model="gemini-2.5-flash"
        self.embedder_model="embedding-001"
        self.encoder_model="gemini-2.5-flash-lite"

        self.llm_client=GeminiClient(config=LLMConfig(
            api_key=self._api_key,
            model=self.cliet_model))
        self.embedder=GeminiEmbedder(config=GeminiEmbedderConfig(
            api_key=self._api_key,
            model=self.embedder_model))
        self.cross_encoder=GeminiRerankerClient(config=LLMConfig(
            api_key=self._api_key,
            model=self.encoder_model))
        self.graphiti=Graphiti(
            uri=self.uri,
            user=self.user,
            password=self.password,
            llm_client=self.llm_client,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder
        )
        
