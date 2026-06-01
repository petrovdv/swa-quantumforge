from typing import Any, Dict, List

import chromadb
import numpy as np

COLLECTION_NAME = "text_files_collection"


class VectorStore:

    def __init__(self, persist_dir="/app/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def search(
            self,
            query_embedding: List[float] | np.ndarray,
            k: int = 3
    ) -> Dict[str, Any]:
        if not query_embedding:
            raise ValueError("query_embedding не может быть пустым")

        emb = np.atleast_2d(query_embedding)

        results = self.collection.query(
            query_embeddings=emb.tolist(),
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )

        if not results["documents"] or not results["documents"][0]:
            return {"documents": [], "distances": [], "metadatas": [], "ids": []}

        return {
            "documents": results["documents"][0],
            "distances": results["distances"][0],
            "metadatas": results["metadatas"][0],
            "ids": results["ids"][0]
        }
