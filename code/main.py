"""
main.py — Entry point for the HackerRank Orchestrate support triage agent.
# Windows-safe: forces UTF-8 stdout to avoid cp1252 UnicodeEncodeError.

Usage
-----
    # From the repo root:
    python code/main.py

    # Custom paths:
    python code/main.py \\
        --input  support_tickets/support_tickets.csv \\
        --output support_tickets/output.csv \\
        --data   data/

    # Dry run (first 10 rows only):
    python code/main.py --dry-run

    # Verbose mode (shows pipeline trace per ticket):
    python code/main.py --verbose

Environment
-----------
    GROQ_API_KEY        (required) — your Groq API key
    LLM_MODEL_NAME      (optional) — model string override
    TOP_K               (optional) — number of corpus chunks to retrieve (default: 6)

The script writes progress to stdout and produces a fully populated output.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

# Allow running from either repo root or code/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # type: ignore[import-untyped]

# Load .env if present (never required — API key can be set in shell env)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── Windows UTF-8 fix ────────────────────────────────────────────────────────
import io
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

# ── Rich imports ─────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.columns import Columns
from rich.padding import Padding
from rich.align import Align
from rich.console import Group

console = Console(highlight=False, force_terminal=True)

from constants import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DATA_DIR,
    TOP_K,
    OUTPUT_FIELDS,
    SEED,
    MIN_ABS_SCORE,
    MIN_GAP,
    LLM_PROVIDER,
    LLM_MODEL,
)
from retriever import CorpusIndex
from classifier import classify, build_retrieval_query
from agent import TriageAgent


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def read_tickets(path: Path) -> list[dict]:
    tickets = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clean = {}
            for k, v in row.items():
                key = k.strip().lower().replace(" ", "_")
                clean[key] = (v or "").strip()
            tickets.append(clean)
    return tickets


def write_output(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------

STATUS_STYLE = {
    "replied":   "bold green",
    "escalated": "bold yellow",
    "error":     "bold red",
}

RISK_STYLE = {
    "none": "dim green",
    "soft": "yellow",
    "hard": "bold red",
}

PATH_STYLE = {
    "LLM":  "cyan",
    "FAST": "magenta",
}

DOMAIN_LABEL = {
    "hackerrank": "[HR]",
    "claude":     "[CL]",
    "visa":       "[VI]",
    None:         "[--]",
}


def _status_badge(status: str) -> Text:
    label = f" {status.upper()} "
    style = STATUS_STYLE.get(status, "white")
    return Text(label, style=f"on {style.split()[-1]} bold white" if "green" in style
                else f"bold {style.split()[-1]}")


def print_header(dry_run: bool, total_tickets: int) -> None:
    console.print()
    console.print(Panel.fit(
        Align.center(
            "[bold white]HackerRank Orchestrate[/bold white]\n"
            "[dim]Support Triage Agent  ·  Multi-Domain AI Pipeline[/dim]"
        ),
        border_style="bright_blue",
        padding=(1, 4),
    ))

    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(justify="right", style="dim")
    info_table.add_column()
    info_table.add_row("Provider",  f"[cyan]{LLM_PROVIDER}[/cyan]")
    info_table.add_row("Model",     f"[cyan]{LLM_MODEL}[/cyan]")
    info_table.add_row("Top-K",     f"[cyan]{TOP_K}[/cyan]")
    info_table.add_row("Tickets",   f"[cyan]{total_tickets}{'  [yellow](DRY RUN — first 10)[/yellow]' if dry_run else ''}[/cyan]")

    console.print(Padding(Align.center(info_table), (1, 0)))
    console.print()


def print_corpus_stats(domain_stats: dict[str, tuple[int, int]]) -> None:
    """domain_stats: {domain: (file_count, chunk_count)}"""
    table = Table(
        title="Corpus Index",
        box=box.ROUNDED,
        border_style="bright_blue",
        title_style="bold bright_blue",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Domain",  style="bold white", min_width=12)
    table.add_column("Files",   justify="right", style="cyan")
    table.add_column("Chunks",  justify="right", style="green")

    total_files  = 0
    total_chunks = 0
    for domain, (files, chunks) in domain_stats.items():
        label = DOMAIN_LABEL.get(domain, "[--]")
        table.add_row(f"{label} {domain}", str(files), f"{chunks:,}")
        total_files  += files
        total_chunks += chunks

    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_files}[/bold]", f"[bold]{total_chunks:,}[/bold]")
    console.print(Align.center(table))
    console.print()


def print_ticket_panel(
    i: int,
    total: int,
    subject: str,
    issue: str,
    company: str,
    cls,
    corpus_chunks: list,
    result,
    elapsed: float,
    verbose: bool,
) -> None:
    domain_tag   = DOMAIN_LABEL.get(cls.domain, "[--]")
    domain_label = f"{domain_tag} {cls.domain or 'unknown'}"

    status_color = "green" if result.status == "replied" else "yellow"
    path = "FAST" if (cls.is_invalid or cls.force_escalate or not corpus_chunks) else "LLM"
    path_color = PATH_STYLE.get(path, "white")

    title_text = (subject or issue)[:70] or "(no subject)"

    # ── Header row ────────────────────────────────────────────────────
    header = Table.grid(padding=(0, 1), expand=True)
    header.add_column(ratio=6)
    header.add_column(ratio=4, justify="right")
    header.add_row(
        f"[bold white]{i:>2}/{total}[/bold white]  [bold]{title_text}[/bold]",
        f"[{status_color}]●[/{status_color}] [{status_color}]{result.status.upper()}[/{status_color}]"
        f"  [{path_color}]{path}[/{path_color}]  [dim]{elapsed:.1f}s[/dim]",
    )

    # ── INPUT block (always shown) ─────────────────────────────────────
    issue_preview = issue.replace("\n", " ").strip()
    if len(issue_preview) > 200:
        issue_preview = issue_preview[:200] + "…"
    company_display = company if company and company.lower() not in ("none", "n/a", "") else "(not specified)"
    input_lines = (
        f"[dim]Subject :[/dim] [white]{subject or '(none)'}[/white]\n"
        f"[dim]Company :[/dim] [white]{company_display}[/white]\n"
        f"[dim]Issue   :[/dim] {issue_preview}"
    )
    divider = Text("─" * 77, style="dim")

    if not verbose:
        meta = (
            f"[dim]Domain:[/dim] {domain_label}   "
            f"[dim]Risk:[/dim] [{RISK_STYLE.get(cls.risk_level,'white')}]{cls.risk_level}[/]   "
            f"[dim]Type:[/dim] [cyan]{result.request_type}[/cyan]   "
            f"[dim]Area:[/dim] [cyan]{result.product_area}[/cyan]\n"
            f"[dim]Response:[/dim] {result.response[:160].replace(chr(10),' ')}{'…' if len(result.response)>160 else ''}"
        )
        console.print(Panel(
            Group(header, divider, Text.from_markup(input_lines), divider, Text.from_markup(meta)),
            border_style=status_color,
            padding=(0, 1),
        ))
        return

    # ── Verbose: full breakdown ────────────────────────────────────────
    content_lines: list[str] = []

    # Classification
    risk_str = f"[{RISK_STYLE.get(cls.risk_level,'white')}]{cls.risk_level}[/]"
    flags_str = (
        f"  [dim]flags:[/dim] [red]{', '.join(cls.risk_flags)}[/red]"
        if cls.risk_flags else ""
    )
    escalate_note = (
        f"\n  [red]!! {cls.escalate_reason}[/red]" if cls.force_escalate else ""
    )
    content_lines.append(
        f"[bold dim]CLASSIFY[/bold dim]  "
        f"domain=[bold]{domain_label}[/bold]  "
        f"risk={risk_str}{flags_str}  "
        f"type=[cyan]{cls.initial_request_type}[/cyan]  "
        f"invalid=[yellow]{cls.is_invalid}[/yellow]"
        f"{escalate_note}"
    )

    # Retrieval
    if corpus_chunks:
        top = corpus_chunks[0]
        content_lines.append(
            f"[bold dim]RETRIEVE[/bold dim]  "
            f"[green]{len(corpus_chunks)} chunks[/green]  "
            f"best_score=[green]{top.score:.2f}[/green]  "
            f"area=[cyan]{top.product_area}[/cyan]"
        )
        for j, c in enumerate(corpus_chunks[:3], 1):
            content_lines.append(
                f"  [dim]#{j}[/dim] score=[green]{c.score:.2f}[/green]"
                f"  [{RISK_STYLE.get('none','white')}]{c.product_area}[/]"
                f"  [dim]{c.source[:65]}[/dim]"
            )
    elif cls.is_invalid:
        content_lines.append("[bold dim]RETRIEVE[/bold dim]  [dim]skipped -- invalid ticket[/dim]")
    elif cls.force_escalate:
        content_lines.append("[bold dim]RETRIEVE[/bold dim]  [dim]skipped -- hard-risk escalation[/dim]")
    else:
        content_lines.append("[bold dim]RETRIEVE[/bold dim]  [yellow]no chunks found[/yellow]")

    # LLM / FAST triage
    content_lines.append(
        f"[bold dim]TRIAGE  [/bold dim]  "
        f"path=[{path_color}]{path}[/{path_color}]  "
        f"status=[{status_color}]{result.status}[/{status_color}]  "
        f"type=[cyan]{result.request_type}[/cyan]  "
        f"area=[cyan]{result.product_area}[/cyan]"
    )

    # Justification
    just_short = result.justification[:160].replace("\n", " ")
    if len(result.justification) > 160:
        just_short += "…"
    content_lines.append(f"[bold dim]JUSTIFY [/bold dim]  [dim]{just_short}[/dim]")

    # Response (full visible)
    resp_short = result.response[:200].replace("\n", " ")
    if len(result.response) > 200:
        resp_short += "…"
    content_lines.append(f"[bold dim]RESPONSE[/bold dim]  [italic]{resp_short}[/italic]")

    body_text = Text.from_markup("\n".join(content_lines))
    console.print(Panel(
        Group(header, divider, Text.from_markup(input_lines), divider, body_text),
        border_style=status_color,
        padding=(0, 1),
    ))


def print_summary(results: list[dict], output_path: Path, total_elapsed: float) -> None:
    replied   = [r for r in results if r["status"] == "replied"]
    escalated = [r for r in results if r["status"] == "escalated"]
    errors    = [r for r in results if r["status"] not in ("replied", "escalated")]

    # ── Stats panel ───────────────────────────────────────────────────
    stats = Table.grid(padding=(0, 3))
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_row(
        f"[bold green]{len(replied)}[/bold green]\n[dim]Replied[/dim]",
        f"[bold yellow]{len(escalated)}[/bold yellow]\n[dim]Escalated[/dim]",
        f"[bold red]{len(errors)}[/bold red]\n[dim]Errors[/dim]",
        f"[bold cyan]{total_elapsed:.1f}s[/bold cyan]\n[dim]Total Time[/dim]",
    )
    console.print()
    console.print(Panel(
        Align.center(stats),
        title="[bold bright_blue]Run Complete[/bold bright_blue]",
        border_style="bright_blue",
        padding=(1, 4),
    ))

    # ── Results table ─────────────────────────────────────────────────
    table = Table(
        title="Ticket Results",
        box=box.ROUNDED,
        border_style="bright_blue",
        title_style="bold bright_blue",
        show_lines=True,
        expand=True,
    )
    table.add_column("#",            style="dim", width=4, justify="right")
    table.add_column("Subject",      min_width=22, max_width=38)
    table.add_column("Domain",       min_width=12)
    table.add_column("Status",       min_width=10, justify="center")
    table.add_column("Type",         min_width=14)
    table.add_column("Product Area", min_width=16)
    table.add_column("Time",         width=6, justify="right")

    for i, r in enumerate(results, 1):
        status = r.get("status", "?")
        sc = "green" if status == "replied" else ("yellow" if status == "escalated" else "red")
        company = r.get("company", "")
        domain = (company or "").lower()
        if "hackerrank" in domain:
            dom_display = "[HR] hackerrank"
        elif "claude" in domain:
            dom_display = "[CL] claude"
        elif "visa" in domain:
            dom_display = "[VI] visa"
        else:
            dom_display = "[--] unknown"

        subject = (r.get("subject") or r.get("issue") or "")[:35]
        table.add_row(
            str(i),
            subject,
            dom_display,
            f"[{sc}]{status}[/{sc}]",
            r.get("request_type", "—"),
            r.get("product_area", "—"),
            f"[dim]{r.get('_elapsed', '—')}[/dim]",
        )

    console.print()
    console.print(table)
    console.print()
    console.print(f"[dim]Output written to:[/dim] [bold cyan]{output_path}[/bold cyan]")
    console.print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: Path, output_path: Path, data_dir: Path,
        dry_run: bool = False, verbose: bool = False) -> None:

    random.seed(SEED)
    t_start = time.monotonic()

    # ── 1. Load corpus ─────────────────────────────────────────────────
    console.print(Rule("[bold bright_blue]  Step 1 · Loading Corpus  [/bold bright_blue]", style="bright_blue"))
    console.print()

    # Monkey-patch retriever prints → captured for our table
    domain_stats: dict[str, tuple[int, int]] = {}
    _orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else print  # type: ignore

    import builtins
    _real_print = builtins.print
    captured: list[str] = []

    def _capturing_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if msg.startswith("[retriever]"):
            captured.append(msg)
        else:
            _real_print(*args, **kwargs)

    builtins.print = _capturing_print  # type: ignore
    index = CorpusIndex.load(data_dir)
    builtins.print = _real_print  # type: ignore

    # Parse captured stats
    for line in captured:
        # e.g. "[retriever] hackerrank: 438 files -> 1793 chunks"
        import re
        m = re.search(r"\[retriever\] (\w+): (\d+) files -> (\d+) chunks", line)
        if m:
            domain_stats[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        # enriched format
        m2 = re.search(r"\[retriever\] (\w+): (\d+) chunks", line)
        if m2 and m2.group(1) not in domain_stats:
            domain_stats[m2.group(1)] = (0, int(m2.group(2)))

    print_corpus_stats(domain_stats)

    # ── 2. Load tickets ────────────────────────────────────────────────
    console.print(Rule("[bold bright_blue]  Step 2 · Loading Tickets  [/bold bright_blue]", style="bright_blue"))
    console.print()
    tickets = read_tickets(input_path)
    if dry_run:
        tickets = tickets[:10]

    # ── 3. Initialise agent ────────────────────────────────────────────
    console.print(Rule("[bold bright_blue]  Step 3 · Initialising Agent  [/bold bright_blue]", style="bright_blue"))
    console.print()
    agent = TriageAgent()

    print_header(dry_run, len(tickets))

    # ── 4. Process tickets ─────────────────────────────────────────────
    console.print(Rule("[bold bright_blue]  Step 4 · Processing Tickets  [/bold bright_blue]", style="bright_blue"))
    console.print()

    results: list[dict] = []

    if verbose:
        # Verbose: print each ticket panel as we go (no progress bar — panels are the progress)
        for i, ticket in enumerate(tickets, 1):
            issue   = ticket.get("issue", "")
            subject = ticket.get("subject", "")
            company = ticket.get("company", "None")
            t0 = time.monotonic()

            try:
                row, cls, corpus_chunks, result_obj = _process_ticket_verbose(
                    issue, subject, company, index, agent
                )
            except Exception as e:
                elapsed = time.monotonic() - t0
                console.print(f"  [red]ERROR on ticket {i}: {e}[/red]")
                row = {
                    "status": "escalated",
                    "product_area": "general_support",
                    "response": (
                        "We've received your request and are routing it to a "
                        "specialist for review. You'll hear back shortly."
                    ),
                    "justification": f"Unhandled error: {str(e)[:100]}",
                    "request_type": "product_issue",
                    "_elapsed": f"{elapsed:.1f}s",
                }
                row["issue"]   = issue
                row["subject"] = subject
                row["company"] = company
                results.append(row)
                continue

            elapsed = time.monotonic() - t0
            row["issue"]    = issue
            row["subject"]  = subject
            row["company"]  = company
            row["_elapsed"] = f"{elapsed:.1f}s"

            print_ticket_panel(
                i, len(tickets),
                subject, issue, company,
                cls, corpus_chunks, result_obj,
                elapsed, verbose=True,
            )
            results.append(row)
            write_output(output_path, results)

    else:
        # Non-verbose: rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=36),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Triaging tickets", total=len(tickets))

            for i, ticket in enumerate(tickets, 1):
                issue   = ticket.get("issue", "")
                subject = ticket.get("subject", "")
                company = ticket.get("company", "None")
                t0 = time.monotonic()

                try:
                    row, cls, corpus_chunks, result_obj = _process_ticket_verbose(
                        issue, subject, company, index, agent
                    )
                except Exception as e:
                    elapsed = time.monotonic() - t0
                    row = {
                        "status": "escalated",
                        "product_area": "general_support",
                        "response": (
                            "We've received your request and are routing it to a "
                            "specialist for review. You'll hear back shortly."
                        ),
                        "justification": f"Unhandled error: {str(e)[:100]}",
                        "request_type": "product_issue",
                    }

                elapsed = time.monotonic() - t0
                row["issue"]    = issue
                row["subject"]  = subject
                row["company"]  = company
                row["_elapsed"] = f"{elapsed:.1f}s"
                results.append(row)

                status_color = "green" if row["status"] == "replied" else "yellow"
                short = (subject or issue)[:40]
                progress.update(
                    task,
                    advance=1,
                    description=f"[{status_color}]{row['status']:>9}[/{status_color}]  {short:<40}",
                )
                write_output(output_path, results)

        # After progress: print compact panel per ticket
        console.print()
        for i, (ticket, row) in enumerate(zip(tickets, results), 1):
            # reconstruct cls + result for display — cheap re-classify
            issue   = ticket.get("issue", "")
            subject = ticket.get("subject", "")
            company = ticket.get("company", "None")
            cls_light = classify(issue, subject, company)
            chunks_light: list = []  # not re-running retrieval for display

            class _FakeResult:
                status       = row["status"]
                request_type = row["request_type"]
                product_area = row["product_area"]
                response     = row["response"]
                justification= row["justification"]

            print_ticket_panel(
                i, len(tickets),
                subject, issue, company,
                cls_light, chunks_light, _FakeResult(),
                float(row["_elapsed"].replace("s", "")),
                verbose=False,
            )

    # ── 5. Write output ────────────────────────────────────────────────
    write_output(output_path, results)
    total_elapsed = time.monotonic() - t_start
    print_summary(results, output_path, total_elapsed)


def _process_ticket_verbose(
    issue: str, subject: str, company: str,
    index: CorpusIndex, agent: TriageAgent,
):
    """Process a single ticket and return (row_dict, cls, corpus_chunks, result_obj)."""

    # Step 1: Pre-classification
    cls = classify(issue, subject, company)

    # Step 2: Retrieval
    corpus_context = ""
    corpus_chunks  = []

    if not cls.is_invalid and not cls.force_escalate:
        query         = build_retrieval_query(issue, subject)
        corpus_chunks = index.query(query, domain=cls.domain, top_k=TOP_K)
        corpus_context = index.format_context(corpus_chunks)

    # Step 3: LLM triage
    result = agent.triage(
        issue=issue,
        subject=subject,
        company=company,
        domain=cls.domain,
        corpus_context=corpus_context,
        corpus_chunks=corpus_chunks,
        force_escalate=cls.force_escalate,
        escalate_reason=cls.escalate_reason,
        force_invalid=cls.is_invalid,
        initial_request_type=cls.initial_request_type,
    )

    row = {
        "status":        result.status,
        "product_area":  result.product_area if getattr(result, "product_area", None) and result.product_area != "general_support" else (corpus_chunks[0].product_area if corpus_chunks else "general_support"),
        "response":      result.response,
        "justification": result.justification,
        "request_type":  result.request_type,
    }
    return row, cls, corpus_chunks, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-domain support triage agent — HackerRank Orchestrate."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Path to input CSV (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Path to write output CSV (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--data", type=Path, default=DATA_DIR,
        help=f"Path to data/ corpus directory (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only the first 10 rows for quick testing."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed pipeline trace panel per ticket."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    missing = [p for p in [args.input, args.data] if not p.exists()]
    if missing:
        for m in missing:
            console.print(f"[bold red]ERROR:[/bold red] path not found — {m}")
        sys.exit(1)

    run(args.input, args.output, args.data,
        dry_run=args.dry_run, verbose=args.verbose)
