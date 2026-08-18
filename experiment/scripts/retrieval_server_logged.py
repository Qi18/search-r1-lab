from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from search_r1.search.retrieval_server import Config, get_retriever


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-log", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--faiss-gpu", action="store_true")
    args = parser.parse_args()

    config = Config(
        retrieval_method="e5",
        retrieval_topk=args.topk,
        index_path=args.index,
        corpus_path=args.corpus,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    retriever = get_retriever(config)
    log_path = Path(args.request_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "documents": len(retriever.corpus), "topk": args.topk}

    @app.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict:
        topk = request.topk or args.topk
        results, scores = retriever.batch_search(request.queries, topk, True)
        event = {
            "at": datetime.now(timezone.utc).isoformat(),
            "query_count": len(request.queries),
            "queries": request.queries,
            "topk": topk,
            "titles": [[doc.get("title", "") for doc in docs] for docs in results],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if request.return_scores:
            payload = [
                [{"document": doc, "score": float(score)} for doc, score in zip(docs, row_scores)]
                for docs, row_scores in zip(results, scores)
            ]
        else:
            payload = results
        return {"result": payload}

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
