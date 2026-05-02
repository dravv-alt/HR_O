"""
build_contextual_corpus.py — Offline corpus enrichment via Contextual BM25.

Inspired by Anthropic's "Contextual Retrieval" technique:
  For each chunk, prepend a short LLM-generated "situational context"
  sentence that describes what the chunk is about and when it would be
  relevant. This dramatically improves BM25 matching when the user's
  query uses different language than the article.

Usage:
    python code/build_contextual_corpus.py [--provider groq|openrouter]

This script runs OFFLINE (build-time), not at query-time.
Output: data/index/contextual_chunks.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from constants import DATA_DIR, DOMAINS, LLM_PROVIDER, LLM_TEMPERATURE

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

from retriever import CorpusIndex, _parse_file, _chunk_by_headings


# ---------------------------------------------------------------------------
# Enrichment prompt
# ---------------------------------------------------------------------------

CONTEXT_PROMPT = """\
You are an indexing assistant. Given a document chunk from a support knowledge base, generate a single concise sentence (20-40 words) that describes:
1. What specific topic or task this chunk addresses
2. What kind of user question would lead to needing this information

This context sentence will be prepended to the chunk to improve search matching.

CHUNK:
{chunk_text}

SOURCE: {source}
PRODUCT AREA: {product_area}

Respond with ONLY the context sentence, no quotes, no explanation."""


# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------

def create_client(provider: str):
    """Create the appropriate LLM client based on provider."""
    if provider == "groq":
        keys = []
        for k, v in os.environ.items():
            if k.startswith("GROQ_API_KEY") and v.strip():
                keys.append(v.strip())
        if not keys:
            raise EnvironmentError("No GROQ_API_KEY found")
        client = openai.OpenAI(
            api_key=keys[0],
            base_url="https://api.groq.com/openai/v1",
        )
        return {"provider": "groq", "client": client, "keys": keys, "idx": 0}
    elif provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise EnvironmentError("OPENROUTER_API_KEY not set")
        client = openai.OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )
        return {"provider": "openrouter", "client": client}
    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        return {"provider": "anthropic", "client": anthropic.Anthropic(api_key=key)}
    else:
        raise ValueError(f"Unknown provider: {provider}")


MODEL_MAP = {
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "openrouter": "anthropic/claude-haiku-3.5",
    "anthropic": "claude-3-5-haiku-20241022",
}


def generate_context(state: dict, chunk_text: str,
                     source: str, product_area: str) -> str:
    """Generate a context sentence for a single chunk."""
    provider = state["provider"]
    client = state["client"]
    prompt = CONTEXT_PROMPT.format(
        chunk_text=chunk_text[:800],  # limit input size
        source=source,
        product_area=product_area,
    )

    attempts = 0
    max_attempts = len(state.get("keys", [])) if provider == "groq" else 1

    while attempts < max_attempts:
        try:
            if provider == "anthropic":
                resp = client.messages.create(
                    model=MODEL_MAP[provider],
                    max_tokens=80,
                    temperature=LLM_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
            else:
                resp = client.chat.completions.create(
                    model=MODEL_MAP[provider],
                    max_tokens=80,
                    temperature=LLM_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = "429" in err_msg or "rate limit" in err_msg
            if provider == "groq" and is_rate_limit and attempts < len(state["keys"]) - 1:
                state["idx"] = (state["idx"] + 1) % len(state["keys"])
                print(f"    [RATE LIMIT] Rotating to Groq key {state['idx']+1}/{len(state['keys'])}")
                state["client"] = openai.OpenAI(
                    api_key=state["keys"][state["idx"]],
                    base_url="https://api.groq.com/openai/v1",
                )
                client = state["client"]
                attempts += 1
                import time
                time.sleep(1.0)
                continue
            else:
                print(f"    [WARN] LLM error: {str(e)[:80]}")
                return ""
    return ""


# ---------------------------------------------------------------------------
# Main build process
# ---------------------------------------------------------------------------

def build(provider: str, data_dir: Path, output_path: Path,
          resume: bool = True) -> None:
    """Build the enriched contextual corpus."""

    print("=" * 60)
    print("  Contextual BM25 Corpus Builder")
    print(f"  Provider: {provider}")
    print("=" * 60)

    # Load existing progress if resuming
    existing: dict[str, dict] = {}
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("chunks", []):
                key = f"{item['source']}::{item['text'][:50]}"
                existing[key] = item
        print(f"[resume] Loaded {len(existing)} previously enriched chunks")

    # Setup client
    prov_state = create_client(provider)
    prov = prov_state["provider"]

    # Collect all chunks
    print("\n[1/3] Loading raw corpus chunks...")
    all_chunks = []
    for domain in DOMAINS:
        domain_dir = data_dir / domain
        if not domain_dir.exists():
            continue
        for fpath in domain_dir.rglob("*.md"):
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
            all_chunks.extend(chunks)

    print(f"      {len(all_chunks)} total chunks across {len(DOMAINS)} domains")

    # Enrich each chunk
    print(f"\n[2/3] Enriching chunks via {provider} ({MODEL_MAP[provider]})...")
    enriched = []
    skipped  = 0
    errors   = 0

    # Wall-clock rate limiter: stay under 28 req/min (safety margin below 30)
    BATCH_SIZE  = 28          # requests per window
    WINDOW_SECS = 62.0        # seconds per window (slightly over 60 for safety)
    batch_start = time.monotonic()
    batch_count = 0

    for i, chunk in enumerate(all_chunks):
        # Check if already enriched (resume support)
        key = f"{chunk['source']}::{chunk['text'][:50]}"
        if key in existing:
            enriched.append(existing[key])
            skipped += 1
            continue

        # Rate limiting — enforce BATCH_SIZE requests per WINDOW_SECS
        batch_count += 1
        if batch_count >= BATCH_SIZE:
            elapsed = time.monotonic() - batch_start
            if elapsed < WINDOW_SECS:
                wait = WINDOW_SECS - elapsed
                print(f"      [rate-limit] sleeping {wait:.1f}s to stay under {BATCH_SIZE} req/min...")
                time.sleep(wait)
            batch_start = time.monotonic()
            batch_count = 0

        # Generate context
        context = generate_context(
            prov_state,
            chunk["text"], chunk["source"], chunk["product_area"]
        )

        if context:
            enriched_chunk = {
                **chunk,
                "context": context,
                "enriched_text": f"{context} | {chunk['text']}",
            }
        else:
            enriched_chunk = {
                **chunk,
                "context": "",
                "enriched_text": chunk["text"],
            }
            errors += 1

        enriched.append(enriched_chunk)

        # Progress
        if (i + 1) % 50 == 0 or i == len(all_chunks) - 1:
            pct = (i + 1) / len(all_chunks) * 100
            print(f"      [{i+1:>5}/{len(all_chunks)}] {pct:.0f}%  "
                  f"(skipped={skipped}, errors={errors})")

            # Checkpoint save every 100 chunks (more frequent = less lost work)
            if (i + 1) % 100 == 0:
                _save(output_path, enriched)

    # Final save
    _save(output_path, enriched)

    # Stats
    with_context = sum(1 for e in enriched if e.get("context"))
    print(f"\n[3/3] Done!")
    print(f"      Total: {len(enriched)} chunks")
    print(f"      Enriched: {with_context}")
    print(f"      Skipped (resumed): {skipped}")
    print(f"      Errors: {errors}")
    print(f"      Saved to: {output_path}")


def _save(output_path: Path, chunks: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "count": len(chunks)}, f, indent=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build contextual BM25 enriched corpus."
    )
    parser.add_argument(
        "--provider", default=LLM_PROVIDER,
        choices=["groq", "openrouter", "anthropic"],
        help="LLM provider for enrichment (default: auto from env)"
    )
    parser.add_argument(
        "--data", type=Path, default=DATA_DIR,
        help="Path to data/ directory"
    )
    parser.add_argument(
        "--output", type=Path,
        default=DATA_DIR / "index" / "contextual_chunks.json",
        help="Output path for enriched corpus"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, don't resume from previous progress"
    )
    args = parser.parse_args()
    build(args.provider, args.data, args.output, resume=not args.no_resume)
