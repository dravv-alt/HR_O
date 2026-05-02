"""
classifier.py — Rule-based pre-classification for support tickets.

Runs BEFORE the LLM call to:
  1. Detect invalid / junk / injection tickets → replied, invalid
  2. Detect hard-risk signals → escalated immediately
  3. Detect soft-risk signals → flag for corpus coverage check
  4. Infer domain from company field or ticket text
  5. Guess initial request_type from signal words

Keeping this layer heuristic makes the pipeline fast and deterministic,
and lets us catch obvious safety cases even when the LLM call fails.

Key fix: invalid tickets get status="replied" (not "escalated") per sample CSV.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from constants import (
    HARD_RISK_KEYWORDS,
    SOFT_RISK_KEYWORDS,
    HARD_RISK_PATTERNS,
    SOFT_RISK_PATTERNS,
    DOMAIN_SIGNALS,
    INJECTION_PATTERNS,
    BUG_SIGNALS,
    FEATURE_SIGNALS,
)


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Structured output from the classifier."""
    # Invalid detection
    is_invalid: bool = False

    # Escalation
    force_escalate: bool = False
    escalate_reason: str = ""

    # Risk
    risk_level: str = "none"         # "hard" | "soft" | "none"
    risk_flags: list[str] = field(default_factory=list)

    # Domain
    domain: Optional[str] = None

    # Request type hint
    initial_request_type: str = "product_issue"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_entropy(text: str) -> float:
    """Shannon entropy in bits per character. Low = repetitive/gibberish."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(issue: str, subject: str, company: str) -> ClassificationResult:
    """
    Run all pre-classification checks in order and return a structured result.

    Order of checks:
      1. Invalid / junk / injection detection
      2. Hard-risk keyword detection
      3. Domain inference
      4. Soft-risk keyword detection
      5. Request type guess
    """
    result = ClassificationResult()
    combined = f"{subject} {issue}".lower()
    stripped = f"{issue} {subject}".strip()

    # ── Step 1: Invalid / junk / injection detection ──────────────────
    words = re.findall(r"[a-z]{3,}", stripped.lower())
    entropy = _compute_entropy(stripped)

    # Check injection patterns
    is_injection = any(p.search(combined) for p in INJECTION_PATTERNS)

    # Check for junk
    is_junk = (
        len(stripped) < 8
        or len(words) < 2
        or entropy < 1.8
        or re.match(r"^(.)\1{10,}$", stripped) is not None
    )

    if is_injection or is_junk:
        result.is_invalid = True
        result.initial_request_type = "invalid"
        # Invalid tickets get replied (not escalated) per sample CSV
        return result

    # Check for pure gratitude / thank-you messages with no actionable request
    _GRATITUDE_PATTERNS = [
        r"^(thank(s| you)|thx|cheers|appreciate|great job|well done)",
        r"^(thanks?|ty) (for|so much|a lot)",
        r"thank you for helping",
    ]
    # Exclusion keywords that indicate a real support request, not just gratitude
    # Use word boundaries so "helping" doesn't match "help"
    _GRATITUDE_EXCLUDERS = [
        "but", "however", "still", "issue", "problem", "error",
        "how do", "how can", "how to", "why",
        "can't", "cannot", "doesn't", "not working",
    ]
    is_gratitude = (
        len(words) < 8
        and any(re.search(p, combined) for p in _GRATITUDE_PATTERNS)
        and not any(kw in combined for kw in _GRATITUDE_EXCLUDERS)
    )
    if is_gratitude:
        result.is_invalid = True
        result.initial_request_type = "invalid"
        return result

    # ── Step 2: Hard-risk keyword detection ───────────────────────────
    for pattern, kw in zip(HARD_RISK_PATTERNS, HARD_RISK_KEYWORDS):
        if pattern.search(combined):
            result.force_escalate = True
            result.risk_level = "hard"
            result.risk_flags.append(kw)
            result.escalate_reason = f"Hard-risk keyword detected: '{kw}'"
            break  # one is enough to escalate

    # ── Step 3: Domain inference ──────────────────────────────────────
    result.domain = detect_domain(issue, subject, company)

    # ── Step 4: Soft-risk detection ───────────────────────────────────
    if result.risk_level != "hard":
        for pattern, kw in zip(SOFT_RISK_PATTERNS, SOFT_RISK_KEYWORDS):
            if pattern.search(combined):
                result.risk_level = "soft"
                result.risk_flags.append(kw)

    # ── Step 5: Request type guess ────────────────────────────────────
    if not result.is_invalid:
        if any(s in combined for s in BUG_SIGNALS):
            result.initial_request_type = "bug"
        elif any(s in combined for s in FEATURE_SIGNALS):
            result.initial_request_type = "feature_request"
        else:
            result.initial_request_type = "product_issue"

    return result


def detect_domain(issue: str, subject: str, company: str) -> Optional[str]:
    """
    Return normalised domain name ('hackerrank', 'claude', 'visa') or None.

    Priority:
      1. Explicit `company` field (if not 'None' or blank).
      2. Keyword match in issue + subject text.
      3. None (agent will search all domains).
    """
    company_clean = (company or "").strip().lower()
    if company_clean and company_clean not in ("none", "n/a", ""):
        if "hackerrank" in company_clean:
            return "hackerrank"
        if "claude" in company_clean:
            return "claude"
        if "visa" in company_clean:
            return "visa"
        return None

    # Keyword inference
    combined = f"{issue} {subject}".lower()
    scores: dict[str, int] = {d: 0 for d in DOMAIN_SIGNALS}
    for domain, keywords in DOMAIN_SIGNALS.items():
        for kw in keywords:
            if kw in combined:
                scores[domain] += 1

    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else None


def build_retrieval_query(issue: str, subject: str) -> str:
    """Construct the best retrieval query from the ticket fields."""
    subject_words = subject.strip().split()
    if len(subject_words) >= 4:
        return f"{subject} {issue[:300]}"
    return issue[:400]
