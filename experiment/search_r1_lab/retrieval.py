from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .io import read_jsonl


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class E5Encoder:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Iterable[str], *, query: bool, batch_size: int = 32) -> np.ndarray:
        prefix = "query: " if query else "passage: "
        prepared = [prefix + text for text in texts]
        batches: list[np.ndarray] = []
        for start in range(0, len(prepared), batch_size):
            encoded = self.tokenizer(
                prepared[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded, return_dict=True)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).bool()
            pooled = hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            batches.append(pooled.float().cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32)


class DenseRetriever:
    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        model_name: str,
        device: str,
    ) -> None:
        self.corpus = read_jsonl(corpus_path)
        self.index = faiss.read_index(index_path)
        self.encoder = E5Encoder(model_name, device)

    @staticmethod
    def build(
        corpus_path: str,
        index_path: str,
        metadata_path: str,
        model_name: str,
        device: str,
    ) -> None:
        corpus = read_jsonl(corpus_path)
        encoder = E5Encoder(model_name, device)
        embeddings = encoder.encode((row["contents"] for row in corpus), query=False)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, index_path)
        metadata = {
            "corpus": str(Path(corpus_path).resolve()),
            "corpus_sha256": _sha256(corpus_path),
            "documents": len(corpus),
            "dimensions": int(embeddings.shape[1]),
            "retriever_model": model_name,
        }
        Path(metadata_path).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def search(self, query: str, topk: int) -> list[dict]:
        embedding = self.encoder.encode([query], query=True)
        scores, indexes = self.index.search(embedding, topk)
        results = []
        for score, index in zip(scores[0].tolist(), indexes[0].tolist()):
            if index < 0:
                continue
            results.append({"score": float(score), **self.corpus[index]})
        return results


def format_results(results: list[dict]) -> str:
    blocks = []
    for position, result in enumerate(results, start=1):
        blocks.append(f"Doc {position} (score={result['score']:.4f})\n{result['contents']}")
    return "\n\n".join(blocks)
