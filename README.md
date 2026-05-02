# HackerRank Orchestrate — Support Triage Agent

> **Hackathon submission for HackerRank Orchestrate (May 2026)**  
> Multi-domain AI support triage agent. Deterministic pre-classification + BM25 retrieval + LLM response generation.

---

## What This Is

A command-line pipeline that reads a CSV of customer support tickets and produces a fully-populated output CSV. Each ticket is classified, routed through a retrieval system, and answered (or escalated) by an LLM.

**Three supported domains:** HackerRank · Claude (Anthropic) · Visa

---

## What Has Actually Been Built

This section is intentionally honest about exactly what exists, what works, and what the limitations are.

### ✅ Built and Working

#### 1. Rule-Based Pre-Classifier (`code/classifier.py`)
- Detects **junk / injection attacks** using Shannon entropy + regex patterns before any LLM call
- Detects **hard-risk tickets** (legal threats, safety, fraud, account compromise) via a keyword registry → instant escalation, zero LLM cost
- Detects **soft-risk tickets** (refund, account locked, billing errors) → flags for corpus coverage check before deciding
- Infers **domain** (`hackerrank`, `claude`, `visa`) from the `company` field first, then via keyword scoring across the issue text
- Guesses initial **request type** (`bug`, `feature_request`, `product_issue`) from signal words
- All logic is deterministic — no LLM involved, no API key needed

#### 2. BM25 Corpus Retrieval (`code/retriever.py`)
- Loads all `.md` files from `data/{hackerrank,claude,visa}/` at startup
- Parses YAML frontmatter (title, breadcrumbs, source URL) using `python-frontmatter`
- Chunks documents by `##` heading boundaries first, then by a sliding word window (400 words, 40-word overlap)
- Builds one **BM25Okapi index per domain** + one **global index** across all domains
- Retrieval is fully deterministic — no embeddings, no network calls, no GPU required
- Derives `product_area` from breadcrumbs or directory path using a mapping table
- **Corpus size:** 774 files → 3,878 chunks (438 HR + 322 Claude + 14 Visa files)
- Optionally loads a pre-built **enriched corpus** (`data/index/contextual_chunks.json`) with LLM-generated context prefixes

#### 3. Contextual Corpus Builder (`code/build_contextual_corpus.py`)  
- Offline enrichment script (run separately, not at query time)
- For each BM25 chunk, calls an LLM to generate a 20–40 word "situational context" sentence
- Prepends that sentence to the chunk before BM25 indexing (Anthropic's "Contextual Retrieval" technique)
- Supports resume from checkpoint — won't re-process already enriched chunks
- Saves output to `data/index/contextual_chunks.json`
- **Limitation:** This script has not been run against the full corpus yet — the pipeline currently falls back to raw BM25

#### 4. LLM Triage Agent (`code/agent.py`)
- Wraps three LLM providers behind one interface: **Anthropic** (claude-sonnet-4), **Groq** (llama-4-scout, free tier), **OpenRouter**
- Provider is auto-detected from env vars (`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`) or set via `LLM_PROVIDER` in `.env`
- Constructs a structured system prompt that instructs the LLM to:
  - Ground all answers in corpus excerpts (no hallucination)
  - Produce a strict JSON schema: `status`, `product_area`, `response`, `justification`, `request_type`, `citation`
  - Only escalate when corpus cannot answer or the issue requires account-specific action
- **Citation verification:** After LLM response, performs a fuzzy sliding-window check to confirm the citation field appears in the raw corpus text. If the citation fails verification and the ticket was "replied", the LLM response is still used (the citation is checked but the current implementation does not downgrade the confidence score as a result)
- **Fast path:** Hard-risk and invalid tickets bypass the LLM entirely — they use template responses from `constants.py`
- **JSON parsing:** Falls back gracefully if the LLM returns malformed JSON

#### 5. Main Pipeline (`code/main.py`)
- End-to-end orchestration: load corpus → load tickets → classify → retrieve → triage → write CSV
- **Rich terminal UI** with panels, progress bars, ruled section headers, and a final summary table
- Two display modes:
  - **Default:** Progress bar during processing, then compact per-ticket panels showing input + output
  - `--verbose`: Full pipeline trace per ticket (CLASSIFY → RETRIEVE → TRIAGE → JUSTIFY → RESPONSE)
- `--dry-run` flag processes the first 10 tickets only
- Graceful error handling — if one ticket crashes, it gets a safe escalation response and processing continues

#### 6. Configuration (`code/constants.py`)
- Single file for all tunable values: paths, thresholds, keyword lists, domain signals, escalation templates
- Multi-provider model defaults
- `DIR_TO_PRODUCT_AREA` mapping from corpus directory names to canonical `product_area` labels

---

### ⚠️ Honest Limitations

| Area | Status | Limitation |
|------|--------|------------|
| **Retrieval** | ✅ | Now uses **Hybrid Search** (BM25 + Semantic via `sentence-transformers` `all-MiniLM-L6-v2`) |
| **Visa corpus** | ⚠️ | Only 14 source files → 62 chunks. Coverage is inherently thin unless more docs are added |
| **Citation verification** | ✅ | Fuzzy check runs and **actively overrides** `replied → escalated` when citation overlap < 30% |
| **Contextual enrichment** | ✅ | Completed. Pipeline auto-loads `contextual_chunks.json` for Contextual Retrieval |
| **product_area accuracy** | ✅ | Fixed. LLM's classification now correctly overrides the top-chunk heuristic |
| **Evaluation harness** | ✅ | `evaluate.py` exists, outputs per-field accuracy table + JSON report |
| **No streaming** | ❌ | LLM responses are blocking — each ticket waits for a full API round-trip |
| **Rate limiting** | ✅ | Safe for Groq free tier when using `llama-3.3-70b-versatile` |

---

## Project Structure

```
hackerrank-orchestrate-may26/
├── code/
│   ├── main.py                    # Entry point + CLI + rich terminal UI
│   ├── classifier.py              # Rule-based pre-classifier (no LLM)
│   ├── retriever.py               # BM25 corpus indexer and query engine
│   ├── agent.py                   # LLM triage agent (Anthropic/Groq/OpenRouter)
│   ├── constants.py               # Central config — all tunable values
│   ├── build_contextual_corpus.py # Offline corpus enrichment script
│   ├── evaluate.py                # Accuracy evaluation vs ground-truth CSV
│   └── requirements.txt
├── data/
│   ├── hackerrank/                # 438 .md support docs
│   ├── claude/                    # 322 .md support docs
│   ├── visa/                      # 14 .md support docs
│   └── index/                     # (optional) contextual_chunks.json
├── support_tickets/
│   ├── support_tickets.csv        # Input tickets
│   ├── sample_support_tickets.csv # Reference sample
│   ├── output.csv                 # Generated output
│   └── sample_output.csv          # Reference output
├── .env                           # API keys (not committed)
├── .env.example                   # Template
└── AGENTS.md                      # Agent architecture notes
```

---

## How to Run

### Prerequisites

```powershell
# Python 3.10+ required
# Activate the virtual environment
.venv\Scripts\activate

# Install dependencies (first time only)
pip install -r code/requirements.txt
pip install rich
```

### Environment

Copy `.env.example` to `.env` and fill in your key:

```env
# Use any ONE provider:
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here

# OR:
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your_key_here
```

### Run the Pipeline

```powershell
# Full run (all 29 tickets) — default compact UI
.venv\Scripts\python.exe code\main.py

# Dry run — first 10 tickets only
.venv\Scripts\python.exe code\main.py --dry-run

# Verbose — full pipeline trace per ticket (best for demos)
.venv\Scripts\python.exe code\main.py --dry-run --verbose

# Custom paths
.venv\Scripts\python.exe code\main.py \
    --input  support_tickets/support_tickets.csv \
    --output support_tickets/output.csv \
    --data   data/
```

### (Optional) Build Enriched Corpus

```powershell
# Run once offline — takes ~10 min on Groq free tier
.venv\Scripts\python.exe code\build_contextual_corpus.py --provider groq
```

---

## Output Schema

The output CSV matches the submission format exactly:

| Field | Values |
|-------|--------|
| `issue` | Original ticket body |
| `subject` | Original ticket subject |
| `company` | Original company field |
| `status` | `replied` or `escalated` |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` |
| `product_area` | Lowercase area label (e.g. `screen`, `pro_and_max_plans`, `general_support`) |
| `response` | User-facing reply (max 150 words) |
| `justification` | Internal routing rationale (max 60 words) |

---

## Pipeline Decision Flow

```
Ticket
  │
  ├── [classifier.py] ─── Junk / injection? ──────────────────► replied / invalid (no LLM)
  │
  ├── [classifier.py] ─── Hard-risk keyword? ─────────────────► escalated (no LLM, template)
  │
  ├── [classifier.py] ─── Detect domain + request type hint
  │
  ├── [retriever.py]  ─── BM25 query (domain-scoped or global) → top-K chunks
  │
  ├── [classifier.py] ─── Soft-risk + corpus doesn't cover it? ► escalated (no LLM)
  │
  └── [agent.py]      ─── LLM call with corpus context ────────► replied or escalated + JSON
```

---

## Dependencies

```
anthropic          # Anthropic SDK (also used for its types even when on Groq)
openai             # OpenAI-compatible SDK — used for Groq and OpenRouter endpoints
rank-bm25          # BM25Okapi retrieval
python-dotenv      # .env loading
python-frontmatter # YAML frontmatter parsing from .md files
rich               # Terminal UI (panels, tables, progress bars)
```

---

## Measured Accuracy on Sample CSV

Run against `sample_support_tickets.csv` (10 tickets, ground truth provided by the challenge):

| Metric | Score | Notes |
|--------|-------|-------|
| **Status** (replied/escalated) | **10/10 — 100%** | All routing decisions correct |
| **Request Type** | **10/10 — 100%** | product_issue / bug / invalid all correct |
| **Product Area** | **2/10 — 20%** | Known issue: BM25 top-chunk area overrides LLM label |
| **Overall** (status + type) | **100%** | Core triage decisions are reliable |

Product area accuracy is the primary improvement target — expected to improve after contextual enrichment completes, as enriched chunks carry explicit domain context.

To run evaluation yourself:
```powershell
.venv\Scripts\python.exe code\evaluate.py
# against a custom output:
.venv\Scripts\python.exe code\evaluate.py --predicted support_tickets/output.csv
```

---

## What Was Not Built

To be completely transparent about scope:

- **No authentication** — the pipeline is a local script, not a service

---

## Final Output Dump

```csv
issue,subject,company,response,product_area,status,request_type,justification
I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system.,Test Active in the system,HackerRank,We cannot find information on how long tests stay active in the system. Please contact support for further assistance.,integrations,escalated,product_issue,No information found in the article regarding test duration.
site is down & none of the pages are accessible,,None,Sorry to hear that the site is down and none of the pages are accessible. We'll look into this issue.,library,escalated,product_issue,The article does not provide a self-service resolution path for site downtime or accessibility issues.
"I'm noticing that you all have many default versions of roles. (e.g. front end developer for react, angular, vue.js, etc.) What do you consider best practice 
for when to create a new test versus create a variant of the test? What are the advantages and disadvantages of using variants?","When should I create a variant versus
 have a different test?",HackerRank,The article does not provide guidance on when to create a new test versus a variant. Please contact support for further assistance.,general_help,escalated,product_issue,The article does not address best practices for creating new tests versus variants.
"Hi there

We have sent a candidate a Hackerrank assessment already, but we have been informed that they require extra time. As the assessment is 105 minutes, they need an extra 50% extra time added (so around 53 minutes) in addition.

Please can you provide step-by-step instruction on how we are to reinvite them.

Do we find their profile in 'Candidates' , put in Add time 53 minutes and then click 'reinvite' and send the email? The email to the candidate says 105 minutes as the duration though still so not sure how we can check?

Thanks","How to reinvite candidate to Hackerrank 
assessment and add extra time",HackerRank,The article does not provide a step-by-step guide on how to reinvite a candidate and add extra time. The process of reinviting a candidate is not clearly outlined.,screen,escalated,product_issue,"The article provides information on adding extra time, but not on reinviting candidates."
"i signed up using google login on hackerrank community , so i do not have a separate hackerrank password. please delete my account",,HackerRank,"To delete your account, please contact HackerRank support at help@hackerrank.com for assistance.",community,escalated,product_issue,Account deletion for Google sign-in requires support assistance
"One of my claude conversations has some private info, i forgot to make a temporary chat, is there anything
 else that can be done? like delete etc?",,Claude,"We cannot assist with deleting private information from Claude conversations. For concerns about data privacy, please refer to our support resources.",privacy,escalated,product_issue,The article does not provide a self-service resolution path for deleting private information from conversations.
What is the name of the actor in Iron Man?,"Urgent, please help",None,The support article does not provide information about actors in Iron Man. Your question is outside the scope of the provided article.,settings,escalated,invalid,The article only discusses company settings and report settings in HackerRank.
I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?,,Visa,"To report your stolen Traveller's Cheques, have your information ready, including cheque serial numbers, purchase location and date, and the issuer of the cheques.",general_support,replied,product_issue,Traveller's Cheques refund process [cite:verified]
Where can I report a lost or stolen Visa card from India?,Card stolen,Visa,"To report your Lost or Stolen Visa Card, please visit our Lost or Stolen card page or call +1 303 967 1090 from anywhere in the world.",general_support,replied,product_issue,The article provides a self-service resolution path for reporting a lost or stolen card. [cite:verified]
Thank you for helping me,,None,"We're here to help, but we need more information about your issue. Please provide details so we can better assist you.",general_support,escalated,invalid,No specific issue mentioned in the ticket
```

## Sample Output

```csv
issue,subject,company,response,product_area,status,request_type,justification
I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system.,Test Active in the system,HackerRank,We cannot find information on how long tests stay active in the system. Please contact support for further assistance.,integrations,escalated,product_issue,No relevant information found in the article.
site is down & none of the pages are accessible,,None,"Sorry to hear that the site is down. We're here to help with accessibility questions, but site downtime is outside our scope.",library,escalated,product_issue,The article does not address site downtime.
"I'm noticing that you all have many default versions of roles. (e.g. front end developer for react, angular, vue.js, etc.) What do you consider best practice 
for when to create a new test versus create a variant of the test? What are the advantages and disadvantages of using variants?","When should I create a variant versus
 have a different test?",HackerRank,We cannot provide guidance on when to create a new test versus a variant. Please refer to our documentation for more information.,general_help,escalated,product_issue,The article does not provide clear guidance on best practices for creating new tests versus variants.
"Hi there

We have sent a candidate a Hackerrank assessment already, but we have been informed that they require extra time. As the assessment is 105 minutes, they need an extra 50% extra time added (so around 53 minutes) in addition.

Please can you provide step-by-step instruction on how we are to reinvite them.

Do we find their profile in 'Candidates' , put in Add time 53 minutes and then click 'reinvite' and send the email? The email to the candidate says 105 minutes as the duration though still so not sure how we can check?

Thanks","How to reinvite candidate to Hackerrank 
assessment and add extra time",HackerRank,"To add extra time, follow the steps in the article. However, re-inviting a candidate requires further clarification.",screen,escalated,product_issue,The article provides steps for adding extra time but does not explicitly cover re-inviting a candidate.
"i signed up using google login on hackerrank community , so i do not have a separate hackerrank password. please delete my account",,HackerRank,"To delete your account, please contact HackerRank support at help@hackerrank.com for assistance.",account_settings,escalated,product_issue,Account deletion for Google login requires support assistance
"One of my claude conversations has some private info, i forgot to make a temporary chat, is there anything
 else that can be done? like delete etc?",,Claude,"We cannot assist with deleting private information from conversations. For concerns about private information, please reach out to our support team.",conversation_management,escalated,product_issue,The article does not provide a self-service resolution path for deleting private information from conversations.
What is the name of the actor in Iron Man?,"Urgent, please help",None,We cannot assist with your question as it is outside our support scope.,settings,escalated,invalid,The support article does not provide information about actors in movies.
I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?,,Visa,"To request a refund for your stolen traveller's cheques, have your information ready. The issuer will ask for cheque serial numbers, purchase location and date, and when the cheques were lost or stolen.",general_support,replied,product_issue,Traveller's cheque refund process [cite:verified]
Where can I report a lost or stolen Visa card from India?,Card stolen,Visa,"To report your Lost or Stolen Visa Card, please visit our Lost or Stolen card page or call +1 303 967 1090 from anywhere in the world.",general_support,replied,product_issue,The article provides a self-service resolution path for reporting a lost or stolen card. [cite:verified]
```
