"""面向繁简中文招股书段落的 BM25 词法检索。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import jieba
from opencc import OpenCC
from rank_bm25 import BM25Okapi

from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.observability.trace import retrieval_outputs, trace_step


jieba.setLogLevel(logging.WARNING)
_TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")
_CJK_SEQUENCE = re.compile(r"[\u3400-\u9fff]+")
_LATIN_OR_NUMBER = re.compile(r"[a-zA-Z]+(?:\.[a-zA-Z]+)*|\d+(?:[.,]\d+)*%?")


def tokenize_text(text: str) -> list[str]:
    """组合 jieba 词、中文二元字组和数字英文词，增强繁体召回。"""
    normalized = "".join(_TRADITIONAL_TO_SIMPLIFIED.convert(text.lower()).split())
    if not normalized:
        return []

    tokens = [token.strip() for token in jieba.lcut(normalized, HMM=False) if token.strip()]
    tokens.extend(_LATIN_OR_NUMBER.findall(normalized))
    for sequence in _CJK_SEQUENCE.findall(normalized):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


@dataclass(frozen=True)
class SearchHit:
    """一条带 BM25 分数和排名的检索命中。"""

    chunk: DocumentChunk
    score: float
    rank: int


class BM25Retriever:
    """在 DocumentChunk 列表上建立轻量内存 BM25 索引。"""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self._chunks = list(chunks)
        tokenized = [tokenize_text(chunk.text) or ["__empty__"] for chunk in chunks]
        self._index = BM25Okapi(tokenized)

    @trace_step("bm25.search", run_type="retriever", inputs=("query", "top_k"), process_outputs=retrieval_outputs)
    def search(self, query: str, top_k: int = 20) -> list[SearchHit]:
        """返回按 BM25 分数降序排列的候选段落。"""
        if top_k < 1:
            return []
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )[: min(top_k, len(self._chunks))]
        return [
            SearchHit(
                chunk=self._chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indexes, start=1)
        ]


__all__ = ["BM25Retriever", "SearchHit", "tokenize_text"]
