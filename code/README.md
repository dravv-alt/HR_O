# HackerRank Orchestrate — Support Triage Agent

## Architecture

```
CSV Input
    |
    v
[Classifier] ──> Hard-risk? ──> ESCALATE immediately
    |                   Invalid/junk? ──> REPLY (out-of-scope)
    v
[BM25 Retriever] ──> Searches per-domain + global index
    |                  Uses enriched contextual chunks if available
    v
[LLM Triage Agent] ──> Generates grounded response from corpus
    |                    Multi-provider: Anthropic / OpenRouter / Groq
    |                    Auto-failover if primary provider fails
    v
[Citation Verifier] ──> Token overlap check against raw corpus
    |                    <30% overlap = hard override to escalated
    v
CSV Output (response, product_area, status, request_type, justification)
```

## Key Design Decisions

1. **BM25 over Embeddings**: Support language is keyword-specific; BM25 is
   deterministic, fast, and requires no vector DB. Improved with optional
   contextual enrichment (Anthropic's Contextual Retrieval technique).

2. **Tiered Safety**: Hard-risk (self-harm, data breach) → immediate escalation.
   Soft-risk (refund, password) → check corpus first, let LLM decide.

3. **Citation Verification**: Two-tier grounding check:
   - Fuzzy sliding window (10-word) for verbatim matches
   - Token overlap ratio fallback (<30% → hallucination guard → escalate)
   Every justification includes `[cite:verified|partial|inferred|FAILED]`.

4. **Multi-Provider Failover**: If primary LLM fails (rate limit, credits),
   automatically tries next provider (Groq → OpenRouter → Anthropic).

5. **Product Area from Corpus**: Derived from file paths/breadcrumbs at
   index-build time, not hallucinated by the LLM.

## Files

| File | Purpose |
|---|---|
| `main.py` | CLI entry point, orchestration loop, `--verbose` tracing |
| `agent.py` | LLM triage with citation verification & provider failover |
| `retriever.py` | BM25 index with frontmatter parsing & heading-aware chunking |
| `classifier.py` | Rule-based pre-classification (risk, domain, request type) |
| `constants.py` | All config, thresholds, keywords, templates — single source of truth |
| `build_contextual_corpus.py` | Offline corpus enrichment (Contextual BM25) |
| `requirements.txt` | Dependencies |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key (pick one)
echo "GROQ_API_KEY=gsk_..." > ../.env

# 3. Run (default input/output paths)
python main.py

# 4. Run with verbose tracing
python main.py --verbose --dry-run

# 5. (Optional) Build enriched index for better retrieval
python build_contextual_corpus.py --provider groq
```

## CLI Flags

| Flag | Description |
|---|---|
| `--input PATH` | Input CSV (default: `support_tickets/support_tickets.csv`) |
| `--output PATH` | Output CSV (default: `support_tickets/output.csv`) |
| `--data PATH` | Corpus directory (default: `data/`) |
| `--dry-run` | Process only first 10 tickets |
| `--verbose` / `-v` | Show per-ticket processing trace |

## Verbose Output

With `--verbose`, each ticket shows:
```
[CLASSIFY] domain=hackerrank, risk=soft, type_hint=product_issue
[CLASSIFY] risk_flags: ['delete my account']
[RETRIEVE] 6 chunks found (best_score=67.91)
  #1: score=67.91 area=community src=hackerrank\hackerrank_community\...
  #2: score=50.28 area=community src=...
[SOFT-RISK] corpus_covers_flags=False
[TRIAGE] path=LLM, status=replied, type=product_issue, area=community
[JUSTIFY] The user signed up with Google... [cite:verified]
[RESPONSE] To delete your account, contact HackerRank support at...
```

## Sample Accuracy

| Metric | Score |
|---|---|
| Status accuracy | **100%** (10/10) |
| Request type accuracy | **100%** (10/10) |
