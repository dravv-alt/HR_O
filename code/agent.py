"""
agent.py — Claude-powered support triage agent with citation verification.

Takes a support ticket + retrieved corpus chunks, calls the Anthropic API
with a strict structured-output prompt, verifies citations against the
raw corpus text, and returns a TriageResult.

Enhancements over V1:
  - Citation requirement in system prompt
  - Citation verification post-processing (fuzzy sliding window)
  - Invalid tickets get status="replied" (not "escalated") per sample CSV
  - Imports config from constants.py (no hardcoded values)
  - Better error handling with structured fallbacks
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import anthropic

try:
    import openai
except ImportError:
    openai = None  # Only needed for OpenRouter/Groq

from constants import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    VALID_STATUS,
    VALID_REQUEST_TYPES,
    ESCALATION_HARD_TEMPLATE,
    INVALID_REPLY_TEMPLATE,
    ESCALATION_LOW_CONFIDENCE,
    DIR_TO_PRODUCT_AREA,
    INVALID_PRODUCT_AREA,
)

ALLOWED_PRODUCT_AREAS = set(DIR_TO_PRODUCT_AREA.values()) | {INVALID_PRODUCT_AREA}

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    status: str          # replied | escalated
    product_area: str    # support category
    response: str        # user-facing answer
    justification: str   # concise routing rationale
    request_type: str    # product_issue | feature_request | bug | invalid


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a support triage specialist for {ecosystem}. Your responses appear directly
in a customer-facing support system. Ground your answers in the support article
provided below.

═══════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════

RULE 1 — PREFER REPLIED
Your DEFAULT status is "replied". If the support article contains ANY information
that is relevant or helpful to the user's question — even partial guidance,
related context, or a pointer in the right direction — set status to "replied"
and provide a helpful answer. You do NOT need a perfect answer to reply.

RULE 2 — ESCALATE ONLY WHEN NECESSARY
Set status to "escalated" ONLY when:
  (a) The article contains absolutely NO information related to the ticket, OR
  (b) The ticket requires account-specific action (e.g., modifying a specific
      user's data, processing a specific refund) that cannot be resolved via
      documentation alone.
Do NOT escalate just because the article only partially covers the topic.

RULE 3 — CITATION
Your "citation" field should contain a relevant sentence from the article.
If you cannot find one, write "NO_GROUNDING_FOUND" and set status to "escalated".

RULE 4 — OUTPUT FORMAT
Respond with ONLY a valid JSON object. No markdown. No backticks. No preamble.
No trailing text after the closing brace.

═══════════════════════════════════════════════════════
OUTPUT SCHEMA (all fields required)
═══════════════════════════════════════════════════════

{{
  "status": "replied" or "escalated",
    "product_area": "Choose ONE value from this list: {allowed_product_areas}. If unsure, use 'general_support'.",
  "response": "User-facing message. Max 120 words. Polite, direct, grounded.",
  "justification": "Internal routing reason. Max 60 words. Reference the article.",
  "request_type": "product_issue" or "feature_request" or "bug" or "invalid",
  "citation": "Exact verbatim sentence(s) from the article. If escalated because
               no grounding exists, write: NO_GROUNDING_FOUND"
}}

═══════════════════════════════════════════════════════
CLASSIFICATION GUIDE
═══════════════════════════════════════════════════════

product_issue  : User needs help with existing functionality or has a how-to question
feature_request: User is asking for new functionality that does not currently exist
bug            : User is reporting that existing functionality is broken or erroring out
invalid        : Off-topic, spam, or nonsense

DO NOT classify as "invalid" just because the question is hard or corpus coverage
is thin. Invalid means the request has no business purpose.

RULE 5 — OFF-TOPIC TICKETS
If the ticket is clearly off-topic or spam (e.g. asking about movies like Iron Man, or random chatter), set:
- request_type to "invalid"
- status to "replied"
- product_area to "general_support"
- citation to "NO_GROUNDING_FOUND"
(Do NOT escalate off-topic tickets).

═══════════════════════════════════════════════════════
SUPPORT ARTICLE
═══════════════════════════════════════════════════════

{article_content}

═══════════════════════════════════════════════════════
TICKET
═══════════════════════════════════════════════════════

Subject: {subject}

{issue}
"""


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

def _fuzzy_citation_check(
    citation: str,
    raw_contents: list[str],
    window: int = 5,
) -> bool:
    """
    Sliding window check: if any `window`-word window from citation
    appears in any raw_content, return True.

    This handles minor punctuation/whitespace normalisation by the LLM
    without loosening the gate significantly.
    """
    if not citation or not raw_contents:
        return False

    citation_lower = citation.lower().strip()

    # Direct substring check first (fast path)
    for rc in raw_contents:
        if citation_lower in rc.lower():
            return True

    # Sliding window fallback
    words = citation_lower.split()
    if len(words) <= window:
        return False  # already tried direct match above

    for i in range(len(words) - window + 1):
        window_str = " ".join(words[i:i + window])
        for rc in raw_contents:
            if window_str in rc.lower():
                return True

    return False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TriageAgent:
    """Wraps the LLM client (Anthropic/OpenRouter/Groq) and exposes a single `triage()` method."""

    def __init__(self, api_key: Optional[str] = None):
        self._provider = LLM_PROVIDER
        self._groq_keys: list[str] = []
        self._current_groq_key_idx: int = 0

        if self._provider == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Export it or put it in a .env file."
                )
            self._anthropic = anthropic.Anthropic(api_key=key)
            self._openai = None

        elif self._provider == "openrouter":
            key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise EnvironmentError(
                    "OPENROUTER_API_KEY is not set. "
                    "Export it or put it in a .env file."
                )
            if openai is None:
                raise ImportError("pip install openai  — required for OpenRouter")
            self._openai = openai.OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
            )
            self._anthropic = None

        elif self._provider == "groq":
            if api_key:
                self._groq_keys.append(api_key)
            else:
                # Load all keys starting with GROQ_API_KEY (e.g., GROQ_API_KEY, GROQ_API_KEY_2)
                for k, v in os.environ.items():
                    if k.startswith("GROQ_API_KEY") and v.strip():
                        self._groq_keys.append(v.strip())
            
            if not self._groq_keys:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. "
                    "Export it or put it in a .env file."
                )
            if openai is None:
                raise ImportError("pip install openai  — required for Groq")
            
            self._openai = openai.OpenAI(
                api_key=self._groq_keys[0],
                base_url="https://api.groq.com/openai/v1",
            )
            self._anthropic = None

        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self._provider}")

        print(f"[agent] Using provider={self._provider} model={LLM_MODEL} (keys: {max(1, len(self._groq_keys))})")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def triage(
        self,
        issue: str,
        subject: str,
        company: str,
        domain: Optional[str],
        corpus_context: str,
        corpus_chunks: list = None,
        force_escalate: bool = False,
        escalate_reason: str = "",
        force_invalid: bool = False,
        initial_request_type: str = "product_issue",
    ) -> TriageResult:
        """
        Run full triage for one ticket row.

        Parameters
        ----------
        issue             : ticket body
        subject           : ticket subject line
        company           : declared company (may be 'None')
        domain            : inferred domain or None
        corpus_context    : pre-formatted retrieved corpus excerpts
        corpus_chunks     : raw ChunkResult objects for citation verification
        force_escalate    : set by classifier for hard-risk
        escalate_reason   : human-readable reason from classifier
        force_invalid     : set by classifier for junk tickets
        initial_request_type : rule-based guess from classifier
        """

        # Fast-path: junk / injection / out-of-scope
        # Per sample CSV: invalid tickets get status="replied"
        if force_invalid:
            return TriageResult(
                status="replied",
                product_area="general_support",
                response=INVALID_REPLY_TEMPLATE,
                justification="Ticket identified as invalid/junk/injection — no actionable content.",
                request_type="invalid",
            )

        # Fast-path: pre-classified hard-risk
        if force_escalate:
            return TriageResult(
                status="escalated",
                product_area="general_support",
                response=ESCALATION_HARD_TEMPLATE,
                justification=f"Pre-classified for escalation: {escalate_reason}",
                request_type=initial_request_type,
            )

        # No corpus context
        if not corpus_context.strip():
            if domain is None:
                # No domain + no corpus = likely off-topic → reply as invalid
                return TriageResult(
                    status="replied",
                    product_area="general_support",
                    response=INVALID_REPLY_TEMPLATE,
                    justification="No domain identified and no corpus match — "
                                 "ticket appears off-topic.",
                    request_type="invalid",
                )
            return TriageResult(
                status="escalated",
                product_area="general_support",
                response=ESCALATION_LOW_CONFIDENCE,
                justification="No corpus articles matched this query.",
                request_type=initial_request_type,
            )

        # Build formatted system prompt with corpus + ticket data baked in
        ecosystem = {"hackerrank": "HackerRank", "claude": "Claude (by Anthropic)", "visa": "Visa"}.get(
            domain or "", "a multi-domain support desk"
        )
        formatted_prompt = SYSTEM_PROMPT.format(
            ecosystem=ecosystem,
            article_content=corpus_context,
            subject=subject or "(no subject)",
            issue=issue,
            allowed_product_areas=", ".join(sorted(ALLOWED_PRODUCT_AREAS)),
        )

        user_msg = "Analyse the ticket above and produce the JSON response."

        raw = self._call_llm(user_msg, formatted_prompt)
        result = self._parse(raw, domain, corpus_chunks)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, user_msg: str, system_prompt: str) -> str:
        """Call the LLM with automatic failover across providers."""
        # Try primary provider first
        result = self._try_call(user_msg, system_prompt)
        if result is not None:
            return result

        # Primary failed — try failover providers
        fallback_order = ["groq", "openrouter", "anthropic"]
        for fb_provider in fallback_order:
            if fb_provider == self._provider:
                continue
            fb_client = self._create_fallback_client(fb_provider)
            if fb_client is None:
                continue
            from rich.console import Console
            Console().print(f"[bold cyan]i [FAILOVER][/bold cyan] Trying {fb_provider}...")
            result = self._try_call_with(user_msg, fb_provider, fb_client, system_prompt)
            if result is not None:
                return result

        # All providers failed
        print("  [LLM ERROR] All providers failed")
        return json.dumps({
            "status": "escalated",
            "product_area": "general_support",
            "response": ESCALATION_LOW_CONFIDENCE,
            "justification": "All LLM providers failed.",
            "request_type": "product_issue",
            "citation": "",
        })

    def _try_call(self, user_msg: str, system_prompt: str) -> Optional[str]:
        """Try calling the primary provider. Returns None on failure."""
        retries = 3
        for attempt in range(retries):
            try:
                if self._anthropic:
                    response = self._anthropic.messages.create(
                        model=LLM_MODEL,
                        max_tokens=LLM_MAX_TOKENS,
                        temperature=LLM_TEMPERATURE,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                    return response.content[0].text.strip()
                else:
                    response = self._openai.chat.completions.create(
                        model=LLM_MODEL,
                        max_tokens=LLM_MAX_TOKENS,
                        temperature=LLM_TEMPERATURE,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                    )
                    return response.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(x in error_str for x in
                    ["rate limit", "429", "too many requests", "rate_limit"])
                is_auth = any(x in error_str for x in
                    ["401", "403", "unauthorized", "invalid api key", "authentication"])

                if is_auth:
                    # Don't retry auth errors — tell the user immediately
                    raise RuntimeError(
                        f"\n{'='*50}\n"
                        f"⛔ API KEY ERROR — EXECUTION PAUSED\n"
                        f"{'='*50}\n"
                        f"Your {self._provider} API key is no longer valid.\n"
                        f"Error: {e}\n\n"
                        f"Please update .env with a fresh key:\n"
                        f"  ANTHROPIC_API_KEY=your_anthropic_key_here  (console.anthropic.com)\n"
                        f"  GROQ_API_KEY=your_groq_key_here          (console.groq.com)\n\n"
                        f"After updating, re-run: python code/main.py\n"
                        f"The partial output.csv has been saved up to this point.\n"
                        f"{'='*50}"
                    )

                if is_rate_limit:
                    import time
                    # Rotate to next Groq key if available
                    if self._provider == "groq" and len(self._groq_keys) > 1:
                        self._current_groq_key_idx = (self._current_groq_key_idx + 1) % len(self._groq_keys)
                        next_key = self._groq_keys[self._current_groq_key_idx]
                        self._openai = openai.OpenAI(
                            api_key=next_key,
                            base_url="https://api.groq.com/openai/v1",
                        )
                        wait = 20  # Wait before retrying with rotated key
                        from rich.console import Console
                        Console().print(f"\n⏳ Rate limited. Error: {error_str}. Rotated to key {self._current_groq_key_idx + 1}/{len(self._groq_keys)}, waiting {wait}s (attempt {attempt+1}/{retries})...")
                    else:
                        wait = 30 * (attempt + 1)  # 30s, 60s, 90s
                        from rich.console import Console
                        Console().print(f"\n⏳ Rate limited. Error: {error_str}. Waiting {wait}s (attempt {attempt+1}/{retries})...")
                    time.sleep(wait)
                    continue

                if attempt < retries - 1:
                    import time
                    wait = 5 * (2 ** attempt)  # 5s, 10s
                    time.sleep(wait)
                    continue

                from rich.console import Console
                Console().print(f"[bold red]✖ [LLM ERROR] {self._provider}:[/bold red] {error_str[:150]}")
                import traceback
                with open("error.log", "a") as f:
                    f.write(f"Provider {self._provider} error:\n")
                    traceback.print_exc(file=f)
                return None
        return None

    def _try_call_with(self, user_msg: str, provider: str,
                       client, system_prompt: str) -> Optional[str]:
        """Try calling a specific fallback provider."""
        from constants import _MODEL_DEFAULTS
        model = _MODEL_DEFAULTS.get(provider, LLM_MODEL)
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return response.content[0].text.strip()
            else:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                )
                if provider == "groq":
                    import time
                    time.sleep(2.5)  # Respect 30 RPM limit
                return response.choices[0].message.content.strip()
        except Exception as e:
            from rich.console import Console
            Console().print(f"[bold red]✖ [FAILOVER ERROR] {provider}:[/bold red] {str(e)[:150]}")
            return None

    @staticmethod
    def _create_fallback_client(provider: str):
        """Try to create a fallback client for a provider. Returns None if no key."""
        if provider == "groq":
            key = os.environ.get("GROQ_API_KEY")
            if key and openai:
                return openai.OpenAI(
                    api_key=key,
                    base_url="https://api.groq.com/openai/v1",
                )
        elif provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            if key and openai:
                return openai.OpenAI(
                    api_key=key,
                    base_url="https://openrouter.ai/api/v1",
                )
        elif provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if key:
                return anthropic.Anthropic(api_key=key)
        return None

    def _parse(
        self,
        raw: str,
        domain: Optional[str],
        corpus_chunks: list = None,
    ) -> TriageResult:
        """Parse JSON from LLM, verify citation, with fallback for malformed output."""

        # Strip possible markdown fences
        clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Attempt to extract JSON substring
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    return self._fallback_escalate("LLM output could not be parsed.")
            else:
                return self._fallback_escalate("LLM returned no JSON.")

        # Validate and sanitise
        status = data.get("status", "escalated")
        if status not in VALID_STATUS:
            status = "escalated"

        request_type = data.get("request_type", "product_issue")
        if request_type not in VALID_REQUEST_TYPES:
            request_type = "product_issue"

        product_area = str(data.get("product_area", domain or "general_support"))
        product_area = product_area.strip().lower().replace(" ", "_").replace("-", "_")
        if product_area not in ALLOWED_PRODUCT_AREAS:
            product_area = "general_support"
        response_text = str(data.get("response", ""))
        justification = str(data.get("justification", ""))

        # ── Citation verification ──────────────────────────────────────
        citation = str(data.get("citation", "")).strip()
        cite_verified = "no_citation"

        if status == "replied" and corpus_chunks:
            raw_contents = [c.text for c in corpus_chunks]

            # Build corpus token set once
            corpus_tokens = set()
            for rc in raw_contents:
                corpus_tokens.update(rc.lower().split())

            citation_ok = False
            if citation and citation.upper() != "NO_GROUNDING_FOUND":
                # Try exact/sliding-window match
                citation_ok = _fuzzy_citation_check(citation, raw_contents)

                # Fallback: token overlap check for paraphrased citations
                if not citation_ok:
                    cite_tokens = set(citation.lower().split())
                    overlap = len(cite_tokens & corpus_tokens)
                    ratio = overlap / max(len(cite_tokens), 1)
                    if ratio >= 0.35:
                        citation_ok = True

            # Check response grounding regardless of citation
            resp_tokens = set(response_text.lower().split())
            resp_overlap = len(resp_tokens & corpus_tokens)
            resp_ratio = resp_overlap / max(len(resp_tokens), 1)

            if citation_ok:
                cite_verified = "verified"
            elif resp_ratio >= 0.15:
                # Citation failed but response is grounded in corpus — allow
                cite_verified = "response_grounded"
            else:
                # Both citation AND response have low corpus overlap
                # → likely hallucinated; force escalation
                status = "escalated"
                cite_verified = "FAILED"
                response_text = (
                    "Your request has been forwarded to our support team. "
                    "A specialist will follow up with you directly."
                )
                justification = (
                    "Citation and response verification both failed -- "
                    "escalated for safety."
                )

        justification_final = justification.strip()
        if cite_verified not in ("no_citation",):
            # Append citation status for traceability
            justification_final += f" [cite:{cite_verified}]"

        return TriageResult(
            status=status,
            product_area=product_area,
            response=response_text,
            justification=justification_final,
            request_type=request_type,
        )

    @staticmethod
    def _fallback_escalate(reason: str) -> TriageResult:
        return TriageResult(
            status="escalated",
            product_area="general_support",
            response=(
                "We've received your request and are routing it to a specialist "
                "for review. You'll hear back shortly."
            ),
            justification=f"Escalated due to processing error: {reason}",
            request_type="product_issue",
        )
