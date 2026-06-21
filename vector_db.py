from qdrant_client import QdrantClient # this is the client for interacting with Qdrant, a vector search engine.
from qdrant_client.models import VectorParams, Distance, PointStruct # these are models for defining vector parameters, distance metrics, and point structures in Qdrant.
import os

class QdrantStorage:
    def __init__(self, url=None, collection="docs", dim=3072):
        url = url or os.environ.get("QDRANT_URL")
        api_key = api_key or os.environ.get("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )


    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points)


    def search(self, query_vector, top_k=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )

        contexts = []
        sources = set()

        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)

        return {"Contexts" : contexts, "Sources": list(sources)}