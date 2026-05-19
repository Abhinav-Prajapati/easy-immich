import numpy as np
import chromadb
from chromadb.config import Settings
from src.config import CHROMA_PATH, CHROMA_COLLECTION

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,

            metadata={"hnsw:space": "cosine"},
        )
        print(f"Vector store ready at: {CHROMA_PATH}")
        print(f"  → Collection '{CHROMA_COLLECTION}' has {self.count()} embeddings")

    def upsert(
        self,
        asset_id: str,
        embedding: np.ndarray,
        asset_meta: dict,
    ) -> None:
        """
        Insert or update a single embedding.
        Safe to call multiple times — ChromaDB will overwrite on same ID.
        """

        exif = asset_meta.get("exifInfo") or {}
        local_dt = asset_meta.get("localDateTime", "")

        year, month = 0, 0
        if local_dt:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(local_dt.replace("Z", "+00:00"))
                year, month = dt.year, dt.month
            except ValueError:
                pass

        metadata = {
            "fileName": asset_meta.get("originalFileName", ""),
            "localDateTime": local_dt,
            "year": year,
            "month": month,
            "width": exif.get("exifImageWidth") or 0,
            "height": exif.get("exifImageHeight") or 0,
            "make": exif.get("make") or "",
            "model": exif.get("model") or "",
        }

        self.collection.upsert(
            ids=[asset_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
        )

    def upsert_batch(
        self,
        asset_ids: list[str],
        embeddings: list[np.ndarray],
        assets_meta: list[dict],
    ) -> None:
        """Batch upsert — much faster than looping upsert() for large sets."""
        from datetime import datetime

        ids, embs, metas = [], [], []

        for asset_id, embedding, asset_meta in zip(asset_ids, embeddings, assets_meta):
            exif = asset_meta.get("exifInfo") or {}
            local_dt = asset_meta.get("localDateTime", "")

            year, month = 0, 0
            if local_dt:
                try:
                    dt = datetime.fromisoformat(local_dt.replace("Z", "+00:00"))
                    year, month = dt.year, dt.month
                except ValueError:
                    pass

            ids.append(asset_id)
            embs.append(embedding.tolist())
            metas.append({
                "fileName": asset_meta.get("originalFileName", ""),
                "localDateTime": local_dt,
                "year": year,
                "month": month,
                "width": exif.get("exifImageWidth") or 0,
                "height": exif.get("exifImageHeight") or 0,
                "make": exif.get("make") or "",
                "model": exif.get("model") or "",
            })

        self.collection.upsert(ids=ids, embeddings=embs, metadatas=metas)

    def already_embedded(self, asset_ids: list[str]) -> set[str]:
        """
        Return the subset of asset_ids that are already in the store.
        Use this to skip re-embedding assets on repeated runs.
        """
        if not asset_ids:
            return set()
        result = self.collection.get(ids=asset_ids, include=[])
        return set(result["ids"])

    def query_similar(
        self,
        embedding: np.ndarray,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Find the n most similar images to a given embedding.
        Optionally filter by metadata fields, e.g. where={"month": 3}
        """
        kwargs = {
            "query_embeddings": [embedding.tolist()],
            "n_results": n_results,
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        return [
            {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                **results["metadatas"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def count(self) -> int:
        return self.collection.count()

    def get_all_ids(self) -> list[str]:
        return self.collection.get(include=[])["ids"]
