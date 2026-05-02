"""
retriever.py — BM25-based corpus retrieval over local support documents.

Loads all .md files from data/{hackerrank,claude,visa}/ at startup,
parses frontmatter for metadata (title, breadcrumbs, source_url),
and builds per-domain BM25 indexes. Retrieval is fully deterministic
(no embeddings, no network calls).

Enhancements over V1:
  - Frontmatter parsing for rich chunk metadata
  - Heading-aware chunking (split on ## first)
  - Product area derived from breadcrumbs / directory path
  - Chunk metadata stored for downstream use (product_area, title, source_url)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

try:
    import frontmatter as fm
except ImportError:
    fm = None  # graceful fallback — will use raw text parsing

from constants import DOMAINS, MAX_CHUNK_TOKENS, CHUNK_OVERLAP, DIR_TO_PRODUCT_AREA


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """A single retrieved chunk with metadata."""
    text: str
    source: str           # relative path to source file
    domain: str           # hackerrank | claude | visa
    score: float = 0.0
    product_area: str = "general_support"
    title: str = ""
    source_url: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lower-cased."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _strip_html(raw: str) -> str:
    """Strip HTML tags; keep content."""
    clean = re.sub(r"<[^>]+>", " ", raw)
    clean = re.sub(r"&[a-z]+;", " ", clean)
    return clean


def _infer_product_area(path: Path, data_dir: Path, breadcrumbs: list[str]) -> str:
    """
    Derive product_area from breadcrumbs or directory path.

    Priority:
      1. First breadcrumb mapped via DIR_TO_PRODUCT_AREA
      2. Subdirectory name mapped via DIR_TO_PRODUCT_AREA
      3. Fallback to 'general_support'
    """
    # Try breadcrumbs first
    if breadcrumbs:
        bc_lower = breadcrumbs[0].lower().strip()
        for key, area in DIR_TO_PRODUCT_AREA.items():
            if key == bc_lower or bc_lower.startswith(key):
                return area

    # Try directory name
    try:
        rel = path.relative_to(data_dir)
        parts = rel.parts  # e.g. ('hackerrank', 'screen', 'subdir', 'file.md')
        for part in parts[1:]:  # skip the domain directory
            if part in DIR_TO_PRODUCT_AREA:
                return DIR_TO_PRODUCT_AREA[part]
            # Try with common transformations
            slug = part.lower().replace(" ", "-")
            if slug in DIR_TO_PRODUCT_AREA:
                return DIR_TO_PRODUCT_AREA[slug]
    except (ValueError, IndexError):
        pass

    return "general_support"


def _parse_file(path: Path, data_dir: Path, domain: str) -> dict | None:
    """
    Parse a markdown file, extracting frontmatter metadata and body.

    Returns dict with keys: domain, product_area, title, source_url,
    body, source (relative path).
    """
    try:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    if len(raw_text.strip()) < 20:
        return None

    source = str(path.relative_to(data_dir))
    title = path.stem
    source_url = ""
    breadcrumbs: list[str] = []
    body = raw_text

    # Try frontmatter parsing
    if fm is not None:
        try:
            doc = fm.loads(raw_text)
            title = doc.get("title", title)
            source_url = doc.get("source_url", "")
            bc = doc.get("breadcrumbs", [])
            if isinstance(bc, str):
                breadcrumbs = [b.strip() for b in bc.split(">")]
            elif isinstance(bc, list):
                breadcrumbs = [str(b).strip() for b in bc]
            body = doc.content.strip()
        except Exception:
            # Fallback: manual frontmatter stripping
            if raw_text.startswith("---"):
                parts = raw_text.split("---", 2)
                if len(parts) >= 3:
                    body = parts[2].strip()
    else:
        # No frontmatter library — manual strip
        if raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()

    # Strip HTML if present
    if "<" in body and ">" in body:
        body = _strip_html(body)

    product_area = _infer_product_area(path, data_dir, breadcrumbs)

    return {
        "domain": domain,
        "product_area": product_area,
        "title": title,
        "source_url": source_url,
        "body": body,
        "source": source,
    }


def _chunk_by_headings(text: str, source: str, domain: str,
                       product_area: str, title: str,
                       source_url: str) -> list[dict]:
    """
    Split text by ## headings first (semantic boundaries).
    If a section exceeds MAX_CHUNK_TOKENS words, further split
    with word-level sliding window.
    """
    max_words = MAX_CHUNK_TOKENS
    overlap_words = CHUNK_OVERLAP

    # Split on ## headings
    sections = re.split(r"\n(?=##\s)", text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section or len(section) < 20:
            continue

        words = section.split()
        if len(words) <= max_words:
            chunks.append(_make_chunk(
                section, source, domain, product_area, title, source_url
            ))
        else:
            # Sliding window
            i = 0
            while i < len(words):
                end = min(i + max_words, len(words))
                chunk_text = " ".join(words[i:end])
                chunks.append(_make_chunk(
                    chunk_text, source, domain, product_area, title, source_url
                ))
                if end == len(words):
                    break
                i += max_words - overlap_words

    return chunks


def _make_chunk(text: str, source: str, domain: str,
                product_area: str, title: str, source_url: str) -> dict:
    return {
        "text": text,
        "source": source,
        "domain": domain,
        "product_area": product_area,
        "title": title,
        "source_url": source_url,
    }


# ---------------------------------------------------------------------------
# CorpusIndex — one BM25 index per domain + global
# ---------------------------------------------------------------------------

class CorpusIndex:
    """
    Holds BM25 indexes for each support domain.

    Usage:
        idx = CorpusIndex.load()
        results = idx.query("reset my password", domain="hackerrank", top_k=5)
    """

    def __init__(self, chunks_by_domain: dict[str, list[dict]],
                 use_enriched: bool = False):
        self._chunks: dict[str, list[dict]] = chunks_by_domain
        self._indexes: dict[str, BM25Okapi] = {}

        # When enriched, use enriched_text for BM25 indexing
        def _get_index_text(c: dict) -> str:
            if use_enriched and "enriched_text" in c:
                return c["enriched_text"]
            return c["text"]

        for domain, chunks in chunks_by_domain.items():
            if chunks:
                tokenized = [_tokenize(_get_index_text(c)) for c in chunks]
                self._indexes[domain] = BM25Okapi(tokenized)

        # Global index across all domains
        all_chunks = []
        for domain in DOMAINS:
            all_chunks.extend(chunks_by_domain.get(domain, []))
        self._all_chunks = all_chunks
        if all_chunks:
            self._global_index = BM25Okapi(
                [_tokenize(_get_index_text(c)) for c in all_chunks]
            )
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._embeddings: dict[str, np.ndarray] = {}
            print("[retriever] Building semantic embeddings... this may take a moment.")
            for domain, chunks in chunks_by_domain.items():
                if chunks:
                    texts = [_get_index_text(c) for c in chunks]
                    self._embeddings[domain] = self.model.encode(texts, show_progress_bar=False)
            
            if self._all_chunks:
                texts = [_get_index_text(c) for c in self._all_chunks]
                self._global_embeddings = self.model.encode(texts, show_progress_bar=False)
            else:
                self._global_embeddings = None
        except Exception as e:
            print(f"[retriever] Semantic search unavailable: {e}")
            self.model = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, data_dir: Optional[Path] = None) -> "CorpusIndex":
        """
        Load the corpus index. Tries enriched index first, falls back to raw.

        Priority:
          1. data/index/contextual_chunks.json (if exists and complete)
          2. Hybrid: enriched for domains that have chunks, raw for empty domains
          3. Raw markdown files with frontmatter parsing
        """
        from constants import DATA_DIR as default_data_dir
        if data_dir is None:
            data_dir = default_data_dir

        enriched_path = data_dir / "index" / "contextual_chunks.json"
        if enriched_path.exists():
            idx = cls._load_enriched(enriched_path, data_dir)
            # Backfill any domains that are empty in the enriched index
            # (enrichment is still in progress)
            empty_domains = [d for d in DOMAINS if not idx._chunks.get(d)]
            if empty_domains:
                print(f"[retriever] Enrichment incomplete — backfilling "
                      f"{empty_domains} from raw markdown")
                raw_idx = cls._load_raw(data_dir)
                for d in empty_domains:
                    idx._chunks[d] = raw_idx._chunks.get(d, [])
                # Rebuild indexes for backfilled domains
                for d in empty_domains:
                    if idx._chunks[d]:
                        tokenized = [_tokenize(c["text"]) for c in idx._chunks[d]]
                        idx._indexes[d] = BM25Okapi(tokenized)
                        print(f"[retriever] {d}: backfilled {len(idx._chunks[d])} raw chunks")
                # Rebuild global index
                all_chunks = []
                for d in DOMAINS:
                    all_chunks.extend(idx._chunks.get(d, []))
                idx._all_chunks = all_chunks
                if all_chunks:
                    def _get_idx_text(c):
                        if c.get("enriched_text"):
                            return c["enriched_text"]
                        return c["text"]
                    idx._global_index = BM25Okapi(
                        [_tokenize(_get_idx_text(c)) for c in all_chunks]
                    )
            return idx

        return cls._load_raw(data_dir)


    @classmethod
    def _load_enriched(cls, path: Path, data_dir: Path) -> "CorpusIndex":
        """Load from pre-built contextual chunks JSON."""
        import json
        print(f"[retriever] Loading enriched corpus from {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks_by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
        for item in data.get("chunks", []):
            domain = item.get("domain", "")
            if domain in chunks_by_domain:
                # Re-derive product_area from source path using current taxonomy
                source = item.get("source", "")
                stored_area = item.get("product_area", "general_support")
                if source:
                    source_path = data_dir / source
                    fresh_area = _infer_product_area(source_path, data_dir, [])
                    area = fresh_area if fresh_area != "general_support" else stored_area
                else:
                    area = stored_area

                chunk = {
                    "text": item.get("text", ""),
                    "enriched_text": item.get("enriched_text", item.get("text", "")),
                    "source": source,
                    "domain": domain,
                    "product_area": area,
                    "title": item.get("title", ""),
                    "source_url": item.get("source_url", ""),
                }
                chunks_by_domain[domain].append(chunk)

        for domain in DOMAINS:
            count = len(chunks_by_domain[domain])
            enriched = sum(1 for c in chunks_by_domain[domain]
                          if c.get("enriched_text") != c.get("text"))
            print(f"[retriever] {domain}: {count} chunks ({enriched} enriched)")

        return cls(chunks_by_domain, use_enriched=True)

    @classmethod
    def _load_raw(cls, data_dir: Path) -> "CorpusIndex":
        """Load from raw markdown files with frontmatter parsing."""
        chunks_by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}

        for domain in DOMAINS:
            domain_dir = data_dir / domain
            if not domain_dir.exists():
                print(f"[retriever] WARNING: {domain_dir} not found -- skipping.")
                continue

            md_files = list(domain_dir.rglob("*.md"))

            total_chunks = 0
            for fpath in md_files:
                doc = _parse_file(fpath, data_dir, domain)
                if not doc:
                    continue

                chunks = _chunk_by_headings(
                    text=doc["body"],
                    source=doc["source"],
                    domain=domain,
                    product_area=doc["product_area"],
                    title=doc["title"],
                    source_url=doc["source_url"],
                )
                chunks_by_domain[domain].extend(chunks)
                total_chunks += len(chunks)

            print(f"[retriever] {domain}: {len(md_files)} files -> {total_chunks} chunks")

        return cls(chunks_by_domain)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
    ) -> list[ChunkResult]:
        """
        Return up to `top_k` most relevant chunks using Hybrid Search (BM25 + Semantic).
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        if domain and domain in self._indexes:
            return self._search_single(query, query_tokens, domain, top_k)

        # Global search
        if self._global_index is None:
            return []

        bm25_scores = self._global_index.get_scores(query_tokens)
        
        # Hybrid scoring
        if hasattr(self, 'model') and self.model is not None and self._global_embeddings is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            query_emb = self.model.encode([query], show_progress_bar=False)
            semantic_scores = cosine_similarity(query_emb, self._global_embeddings)[0]
            
            # Normalize BM25
            if max(bm25_scores) > 0:
                bm25_scores = [s / max(bm25_scores) for s in bm25_scores]
            
            # Combine
            final_scores = [0.4 * b + 0.6 * s for b, s in zip(bm25_scores, semantic_scores)]
        else:
            final_scores = bm25_scores

        ranked = sorted(
            enumerate(final_scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            ChunkResult(
                text=self._all_chunks[i]["text"],
                source=self._all_chunks[i]["source"],
                domain=self._all_chunks[i]["domain"],
                score=float(s),
                product_area=self._all_chunks[i].get("product_area", "general_support"),
                title=self._all_chunks[i].get("title", ""),
                source_url=self._all_chunks[i].get("source_url", ""),
            )
            for i, s in ranked
        ]

    def _search_single(
        self, query: str, query_tokens: list[str], domain: str, top_k: int
    ) -> list[ChunkResult]:
        idx = self._indexes[domain]
        chunks = self._chunks[domain]
        bm25_scores = idx.get_scores(query_tokens)
        
        # Hybrid scoring
        if hasattr(self, 'model') and self.model is not None and domain in self._embeddings:
            from sklearn.metrics.pairwise import cosine_similarity
            query_emb = self.model.encode([query], show_progress_bar=False)
            semantic_scores = cosine_similarity(query_emb, self._embeddings[domain])[0]
            
            if max(bm25_scores) > 0:
                bm25_scores = [s / max(bm25_scores) for s in bm25_scores]
                
            final_scores = [0.4 * b + 0.6 * s for b, s in zip(bm25_scores, semantic_scores)]
        else:
            final_scores = bm25_scores

        ranked = sorted(
            enumerate(final_scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            ChunkResult(
                text=chunks[i]["text"],
                source=chunks[i]["source"],
                domain=domain,
                score=float(s),
                product_area=chunks[i].get("product_area", "general_support"),
                title=chunks[i].get("title", ""),
                source_url=chunks[i].get("source_url", ""),
            )
            for i, s in ranked
        ]

    def format_context(self, results: list[ChunkResult], max_chars: int = 6000) -> str:
        """Concatenate retrieved chunks into a context block for the LLM."""
        lines = []
        total = 0
        for r in results:
            header = f"[Source: {r.source} | Area: {r.product_area}]\n"
            body = r.text + "\n\n"
            entry = header + body
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)
        return "".join(lines).strip()
