"""Chroma vector store with local embeddings, BM25 Hybrid Search (RRF), and Re-ranking."""
import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from .ingest import chunk_text


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def transform_query(query: str) -> str:
    """Transform conversational or verbose questions into high-signal search terms (HyDE / Query Expansion)."""
    filler_patterns = [
        r"^can you (please )?(tell|show|explain|find|summarize) (me )?",
        r"^what (is|are|was|were) (the )?",
        r"^how (do|does|can) (i|we|one) ",
        r"^where (is|are|can i find) (the )?",
        r"^i want to (know|find|learn|see) (about )?",
        r"^please (search for|find|retrieve) ",
    ]
    cleaned = query.strip()
    for pat in filler_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip("?.! ")
    return cleaned if cleaned else query


class BM25Index:
    """In-memory BM25Okapi implementation for exact keyword and terminology retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.records: list[dict] = []

    def index(self, docs: Sequence[str], metadatas: Sequence[Any]):
        self.records = [
            {
                "text": doc,
                "source": meta.get("source", "unknown") if meta else "unknown",
                "page": meta.get("page", 1) if meta else 1,
            }
            for doc, meta in zip(docs, metadatas)
        ]
        self.corpus = [tokenize(doc) for doc in docs]
        self.doc_len = [len(c) for c in self.corpus]
        n_docs = len(self.corpus)
        self.avgdl = sum(self.doc_len) / n_docs if n_docs > 0 else 0.0
        self.doc_freqs = [Counter(c) for c in self.corpus]

        df = Counter()
        for c in self.corpus:
            df.update(set(c))

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.corpus:
            return []
        tokens = tokenize(query)
        scores = []
        for idx, (freqs, doc_l) in enumerate(zip(self.doc_freqs, self.doc_len)):
            score = 0.0
            for t in tokens:
                if t in freqs:
                    count = freqs[t]
                    num = count * (self.k1 + 1)
                    denom = count + self.k1 * (1 - self.b + self.b * (doc_l / (self.avgdl or 1)))
                    score += self.idf.get(t, 0.0) * (num / denom)
            if score > 0:
                rec = self.records[idx].copy()
                rec["bm25_score"] = round(score, 3)
                scores.append(rec)
        scores.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scores[:top_k]


def rerank_passages(query: str, candidates: list[dict], top_k: int = 4) -> list[dict]:
    """Re-rank candidate passages by scoring keyword coverage, exact phrase match, and header prominence."""
    query_terms = set(tokenize(query))
    if not query_terms or not candidates:
        return candidates[:top_k]

    scored = []
    for cand in candidates:
        text = str(cand.get("text", "")).lower()
        cand_terms = set(tokenize(text))

        # 1. Base semantic/RRF score (0 to 1)
        try:
            sem_score = float(cand.get("score", 0.5))
        except (ValueError, TypeError):
            sem_score = 0.5

        # 2. Keyword coverage (fraction of query terms present in passage)
        overlap = len(query_terms.intersection(cand_terms))
        coverage = overlap / len(query_terms) if query_terms else 0.0

        # 3. Exact phrase match bonus
        phrase_bonus = 0.2 if query.lower() in text else 0.0

        # 4. Heading / Title bonus (if query terms appear in first 100 chars)
        early_match = 0.1 if any(t in text[:100] for t in query_terms) else 0.0

        rerank_score = round(0.5 * sem_score + 0.3 * coverage + phrase_bonus + early_match, 3)
        cand_copy = cand.copy()
        cand_copy["rerank_score"] = rerank_score
        # Display rerank score as primary similarity metric
        cand_copy["score"] = rerank_score
        scored.append(cand_copy)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


class VectorStore:
    def __init__(self, path: str = "chroma_db", collection: str = "docs"):
        client = chromadb.PersistentClient(path=path)
        ef: Any = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.col = client.get_or_create_collection(
            collection, embedding_function=ef, metadata={"hnsw:space": "cosine"}
        )
        self.bm25 = BM25Index()
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        try:
            data = self.col.get(include=["documents", "metadatas"])
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            self.bm25.index(docs, metas)
        except Exception:
            pass

    def add_document(self, source: str, pages: list[tuple[int, str]]) -> int:
        ids, docs, metas = [], [], []
        for page, text in pages:
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(hashlib.sha1(f"{source}|{page}|{i}".encode()).hexdigest())
                docs.append(chunk)
                metas.append({"source": source, "page": page})
        if ids:
            self.col.upsert(ids=ids, documents=docs, metadatas=metas)
            self._rebuild_bm25()
        return len(ids)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Hybrid search combining Chroma dense vector search, BM25 via RRF, and Cross-Relevance Re-ranking."""
        n = self.col.count()
        if n == 0:
            return []

        # 1. Query Transformation & Expansion (HyDE-lite)
        transformed = transform_query(query)
        queries = [query]
        if transformed.lower() != query.lower():
            queries.append(transformed)

        # 2. Dense vector search
        candidate_k = min(max(k * 2, 8), n)
        res = self.col.query(query_texts=queries, n_results=candidate_k)
        all_docs = res.get("documents") or []
        all_metas = res.get("metadatas") or []
        all_dists = res.get("distances") or []

        dense_hits = []
        for q_docs, q_metas, q_dists in zip(all_docs, all_metas, all_dists):
            for doc, meta, dist in zip(q_docs, q_metas, q_dists):
                dense_hits.append(
                    {
                        "text": str(doc),
                        "source": meta.get("source", "unknown") if meta else "unknown",
                        "page": meta.get("page", 1) if meta else 1,
                        "score": round(1 - float(dist), 3),
                    }
                )

        # 3. Sparse BM25 search
        bm25_hits = self.bm25.search(query, top_k=candidate_k)
        if transformed.lower() != query.lower():
            bm25_hits.extend(self.bm25.search(transformed, top_k=candidate_k))

        # 4. Reciprocal Rank Fusion (RRF, constant = 60)
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            key = f"{hit['source']}|{hit['page']}|{hit['text'][:100]}"
            doc_map[key] = hit
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60.0 + rank))

        for rank, hit in enumerate(bm25_hits, start=1):
            key = f"{hit['source']}|{hit['page']}|{hit['text'][:100]}"
            if key not in doc_map:
                hit_copy = hit.copy()
                hit_copy["score"] = hit_copy.get("bm25_score", 0.5)
                doc_map[key] = hit_copy
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60.0 + rank))

        # Sort candidate pool by fused score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        candidates = [doc_map[k] for k in sorted_keys[:candidate_k]]

        # 5. Re-Ranking Stage: Pick the best k chunks based on relevance
        return rerank_passages(query, candidates, top_k=k)

    def get_document_chunks(self, source: str) -> list[dict]:
        """Retrieve all individual chunks for a document with page numbers."""
        try:
            data = self.col.get(where={"source": source}, include=["documents", "metadatas"])
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            chunks = []
            for doc, meta in zip(docs, metas):
                chunks.append(
                    {
                        "text": str(doc),
                        "page": meta.get("page", 1) if meta else 1,
                        "source": source,
                    }
                )
            chunks.sort(key=lambda x: x["page"])
            return chunks
        except Exception:
            return []

    def get_document_text(self, source: str) -> str:
        """Retrieve all text chunks for a specific document to enable summarization."""
        try:
            data = self.col.get(where={"source": source}, include=["documents", "metadatas"])
            docs = data.get("documents") or []
            return "\n\n".join(docs)
        except Exception:
            return ""

    def sources(self) -> list[str]:
        data = self.col.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        found = {str(m["source"]) for m in metas if m and "source" in m}
        return sorted(found)

    def delete_source(self, source: str) -> None:
        self.col.delete(where={"source": source})
        self._rebuild_bm25()

    def count(self) -> int:
        return self.col.count()
