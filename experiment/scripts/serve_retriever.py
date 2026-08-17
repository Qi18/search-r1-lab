from __future__ import annotations

import argparse
import itertools
import json
import threading
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from search_r1_lab.retrieval import DenseRetriever


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", default="intfloat/e5-small-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--request-log")
    args = parser.parse_args()

    retriever = DenseRetriever(
        corpus_path=args.corpus,
        index_path=args.index,
        model_name=args.model,
        device=args.device,
    )
    app = FastAPI()
    request_counter = itertools.count()
    request_lock = threading.Lock()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "documents": len(retriever.corpus)}

    @app.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict:
        topk = request.topk or args.topk
        batches = []
        for query in request.queries:
            results = retriever.search(query, topk)
            batches.append(
                [
                    {
                        "document": {"id": row["id"], "contents": row["contents"]},
                        **({"score": row["score"]} if request.return_scores else {}),
                    }
                    for row in results
                ]
            )
        if args.request_log:
            record = {
                "request_id": next(request_counter),
                "queries": request.queries,
                "topk": topk,
                "result_ids": [
                    [row["document"]["id"] for row in batch] for batch in batches
                ],
            }
            with request_lock:
                with Path(args.request_log).open("a") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"result": batches}

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
