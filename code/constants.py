"""
constants.py — Central configuration for the support triage agent.

All thresholds, keywords, paths, and configuration values live here.
Never hardcode tunable values elsewhere.
"""

from pathlib import Path
import os
import re

def _build_patterns(keywords: list[str]) -> list[re.Pattern]:
    """
    Compile each keyword as a whole-word regex pattern.
    Handles keywords with trailing spaces (e.g. "sue ") by stripping them.
    Handles multi-word phrases with internal spaces correctly.
    """
    patterns = []
    for kw in keywords:
        kw_clean = kw.strip()
        # For single words, use \b word boundaries
        # For multi-word phrases, anchor the whole phrase
        if " " in kw_clean:
            pattern = re.compile(re.escape(kw_clean), re.IGNORECASE)
        else:
            pattern = re.compile(r'\b' + re.escape(kw_clean) + r'\b', re.IGNORECASE)
        patterns.append(pattern)
    return patterns

# === Paths ===
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_INPUT = REPO_ROOT / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT = REPO_ROOT / "support_tickets" / "output.csv"

# === Retrieval ===
TOP_K = int(os.environ.get("TOP_K", "6"))
MAX_CHUNK_TOKENS = 400
CHUNK_OVERLAP = 40
MIN_ABS_SCORE: float = 1.0   # placeholder — calibrate via sample CSV
MIN_GAP: float = 0.3          # placeholder — calibrate via sample CSV

# === Reproducibility ===
SEED: int = 42

# === LLM ===
# Provider: "anthropic" | "openrouter" | "groq" — auto-detected from env vars
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "auto")

# Auto-detect provider from available API keys
if LLM_PROVIDER == "auto":
    if os.environ.get("ANTHROPIC_API_KEY"):
        LLM_PROVIDER = "anthropic"
    elif os.environ.get("OPENROUTER_API_KEY"):
        LLM_PROVIDER = "openrouter"
    elif os.environ.get("GROQ_API_KEY"):
        LLM_PROVIDER = "groq"
    else:
        LLM_PROVIDER = "anthropic"  # default, will fail with clear error

# Provider-specific model defaults
_MODEL_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openrouter": "anthropic/claude-sonnet-4.5",
    "groq": "llama-3.1-8b-instant",
}
LLM_MODEL: str = os.environ.get("LLM_MODEL_NAME", _MODEL_DEFAULTS.get(LLM_PROVIDER, "claude-sonnet-4-20250514"))
LLM_MAX_TOKENS: int = 1024
LLM_TEMPERATURE: float = 0.0

# === Domains ===
DOMAINS: list[str] = ["hackerrank", "claude", "visa"]

# === Output schema ===
# Must match the template in support_tickets/output.csv exactly
OUTPUT_FIELDS: list[str] = [
    "issue", "subject", "company", "response", "product_area",
    "status", "request_type", "justification",
]
VALID_STATUS: set[str] = {"replied", "escalated"}
VALID_REQUEST_TYPES: set[str] = {"product_issue", "feature_request", "bug", "invalid"}

# ═══════════════════════════════════════════════════════════════════
# Risk keyword registries
# ═══════════════════════════════════════════════════════════════════

# Hard-risk: immediate escalation, no LLM call needed
# These are safety-critical or require specialist human intervention
HARD_RISK_KEYWORDS: list[str] = [
    # Safety
    "self-harm", "suicide", "emergency services",
    # Security reports
    "data breach", "security breach", "security vulnerability",
    "account compromised", "fraud", "hacked",
    # Legal
    "lawsuit", "sue", "attorney", "court order", "subpoena",
    "legal action", "regulatory complaint", "gdpr complaint",
    # Assessment integrity (HackerRank-specific)
    "impersonation", "proxy candidate", "cheating on test",
    # System Status
    "site is down",
]
HARD_RISK_PATTERNS: list[re.Pattern] = _build_patterns(HARD_RISK_KEYWORDS)

# Soft-risk: check corpus coverage before deciding
# If the corpus can answer it, let it through; otherwise escalate
SOFT_RISK_KEYWORDS: list[str] = [
    "refund", "overcharged", "double charged", "billing error",
    "wrong amount", "delete my account", "close my account",
    "lost access", "locked out", "forgot password", "reset password",
    "two-factor", "2fa", "cannot log in", "can't log in",
    "account suspended", "account banned", "account deleted",
]
SOFT_RISK_PATTERNS: list[re.Pattern] = _build_patterns(SOFT_RISK_KEYWORDS)

# ═══════════════════════════════════════════════════════════════════
# Domain signal words — for inferring domain when company is None
# ═══════════════════════════════════════════════════════════════════

DOMAIN_SIGNALS: dict[str, list[str]] = {
    "hackerrank": [
        "hackerrank", "coding test", "coding challenge", "coding assessment",
        "test case", "proctoring", "plagiarism", "candidate", "recruiter",
        "screen", "code editor", "compile", "submission", "hire", "sprint",
        "assessment", "interview kit", "certif", "code pair",
        "hackerrank.com", "work sample", "library",
    ],
    "claude": [
        "claude", "anthropic", "claude.ai", "claude pro", "claude team",
        "claude enterprise", "conversation", "artifact", "system prompt",
        "context window", "token limit", "usage limit", "message limit",
        "claude subscription", "claude billing", "projects feature",
        "claude code", "claude desktop", "claude api", "bedrock",
    ],
    "visa": [
        "visa", "visa card", "credit card", "debit card", "transaction",
        "payment", "dispute", "chargeback", "merchant", "atm", "pin",
        "visa checkout", "visa direct", "prepaid card", "visa gift",
        "contactless", "chip card", "international transaction",
        "foreign transaction", "traveller",
    ],
}

# ═══════════════════════════════════════════════════════════════════
# Injection / junk patterns
# ═══════════════════════════════════════════════════════════════════

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(previous|all)\s+instructions", re.I),
    re.compile(r"(jailbreak|DAN\s+mode|act\s+as\s+(an?\s+)?AI\s+without\s+restrictions)", re.I),
    re.compile(r"(forget|disregard)\s+(your|the)\s+(rules|instructions|system prompt)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"(affiche|montre|display)\s+.{0,30}(règles? internes?|internal rules|system prompt|logic)", re.I),
]

# ═══════════════════════════════════════════════════════════════════
# Request-type signal words
# ═══════════════════════════════════════════════════════════════════

BUG_SIGNALS: list[str] = [
    "error", "broken", "not working", "bug", "failed", "crash",
    "loading error", "site down", "doesn't work", "stopped working",
    "failing", "down", "is down",
]

FEATURE_SIGNALS: list[str] = [
    "can you add", "i wish", "suggest", "feature request",
    "would be nice", "please implement", "is it possible to add",
    "would love if", "should support",
]

# ═══════════════════════════════════════════════════════════════════
# Escalation response templates
# ═══════════════════════════════════════════════════════════════════

ESCALATION_HARD_TEMPLATE: str = (
    "Thank you for reaching out. Your request involves a sensitive matter "
    "that requires attention from a specialist. A member of our support team "
    "will follow up with you shortly. Please do not share any sensitive "
    "details (passwords, card numbers) over this channel."
)

INVALID_REPLY_TEMPLATE: str = (
    "Thank you for reaching out. Your message does not appear to be a valid "
    "support request for our services. If you have a specific question about "
    "HackerRank, Claude, or Visa, please resubmit with more detail."
)

ESCALATION_LOW_CONFIDENCE: str = (
    "We were unable to find documentation that directly addresses your question. "
    "Your request has been forwarded to our support team for a more thorough review."
)

# ═══════════════════════════════════════════════════════════════════
# Product area mapping from directory/breadcrumb to canonical label
# ═══════════════════════════════════════════════════════════════════

# Ground-truth taxonomy (from sample_support_tickets.csv):
#   community, conversation_management, general_support,
#   privacy, screen, travel_support

# Maps subdirectory names to canonical product area strings.
# Every value MUST match the ground-truth taxonomy exactly.
DIR_TO_PRODUCT_AREA: dict[str, str] = {
    # ── HackerRank ──────────────────────────────────────────────────────────
    "screen":                             "screen",
    "engage":                             "engage",
    "interviews":                         "interviews",
    "integrations":                       "integrations",
    "library":                            "library",
    "settings":                           "settings",
    "general-help":                       "general_help",
    "general_help":                       "general_help",
    "hackerrank_community":               "community",
    "chakra":                             "chakra",
    "uncategorized":                      "general_support",

    # ── Claude ──────────────────────────────────────────────────────────────
    "claude-api-and-console":             "api_and_console",
    "claude-code":                        "claude_code",
    "claude-desktop":                     "claude_desktop",
    "claude-for-education":               "claude_for_education",
    "claude-for-government":              "claude_for_government",
    "claude-for-nonprofits":              "claude_for_nonprofits",
    "claude-in-chrome":                   "claude_in_chrome",
    "claude-mobile-apps":                 "claude_mobile_apps",
    "amazon-bedrock":                     "amazon_bedrock",
    "team-and-enterprise-plans":          "team_and_enterprise",
    "billing-and-subscription":           "billing_and_subscription",
    "pro-and-max-plans":                  "pro_and_max_plans",
    "safeguards":                         "privacy",
    "privacy":                            "privacy",
    "conversation-management":            "conversation_management",
    "conversation_management":            "conversation_management",

    # ── Visa ────────────────────────────────────────────────────────────────
    "consumer":                           "consumer_support",
    "merchant":                           "merchant_support",
    "small-business":                     "small_business",
    "travel-support":                     "travel_support",
    "travellers-cheques":                 "travel_support",
    "general":                            "general_support",

    # ── Fallback ─────────────────────────────────────────────────────────────
    "general_support":                    "general_support",
    "support":                            "general_support",
}

# For invalid tickets, use this area (matches GT for off-topic tickets)
INVALID_PRODUCT_AREA: str = "general_support"

