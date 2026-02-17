from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


# ------------------------------
# Hard-coded corpus
# ------------------------------
CORPUS = [
    {
        "id": "a",
        "text": "The Old Man and the Sea is a short novel written by the American author Ernest Hemingway.",
    },
    {
        "id": "b",
        "text": "Hemingway also wrote A Farewell to Arms and For Whom the Bell Tolls.",
    },
    {
        "id": "c",
        "text": "Moby-Dick is a novel by Herman Melville.",
    },
]


# ------------------------------
# Minimal TF-IDF Retriever
# ------------------------------
class TfidfLocalRetriever:
    def __init__(self):
        self._vectorizer = None
        self._X = None
        self._docs = None

    def fit(self, docs: List[Dict[str, Any]]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._docs = docs
        texts = [d["text"] for d in docs]
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        self._X = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self._vectorizer is None:
            raise RuntimeError("Retriever not fitted")

        qv = self._vectorizer.transform([query])
        scores = cosine_similarity(qv, self._X).ravel()
        idxs = scores.argsort()[::-1][:top_k]

        results = []
        for i in idxs:
            d = self._docs[int(i)]
            results.append(
                {
                    "id": d.get("id", str(i)),
                    "text": d["text"],
                    "score": float(scores[int(i)]),
                }
            )
        return results


# ------------------------------
# FastAPI schema
# ------------------------------
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    results: List[Dict[str, Any]]


# ------------------------------
# App
# ------------------------------
app = FastAPI(title="Minimal TF-IDF Retrieval Server")

retriever = TfidfLocalRetriever()


@app.on_event("startup")
def startup():
    retriever.fit(CORPUS)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    try:
        results = retriever.search(req.query, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "query": req.query,
        "top_k": req.top_k,
        "results": results,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------
# Direct run
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
