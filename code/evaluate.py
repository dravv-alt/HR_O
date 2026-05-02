"""
evaluate.py — Accuracy evaluation for the support triage agent.

Compares model output CSV against a ground-truth CSV row-by-row and reports:
  - Status accuracy       (replied / escalated)
  - Request type accuracy (product_issue / bug / feature_request / invalid)
  - Product area accuracy (exact match)
  - Overall accuracy

Usage
-----
    # Compare our sample_output.csv against sample_support_tickets.csv ground truth:
    python code/evaluate.py

    # Custom paths:
    python code/evaluate.py \\
        --predicted support_tickets/output.csv \\
        --ground-truth support_tickets/sample_support_tickets.csv

Output
------
    Rich terminal table + JSON report saved to support_tickets/eval_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich import box

console = Console(highlight=False, force_terminal=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent.parent
GT_PATH     = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"
PRED_PATH   = REPO_ROOT / "support_tickets" / "sample_output.csv"
REPORT_PATH = REPO_ROOT / "support_tickets" / "eval_report.json"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clean = {k.strip().lower().replace(" ", "_"): (v or "").strip()
                     for k, v in row.items()}
            rows.append(clean)
    return rows


def _normalise(val: str) -> str:
    return val.strip().lower().replace(" ", "_").replace("-", "_")


def _trunc(text: str, max_chars: int = 55) -> str:
    """Truncate at a word boundary; append ellipsis if truncated."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    # Walk back from max_chars to the nearest space
    cut = text.rfind(" ", 0, max_chars)
    if cut == -1:
        cut = max_chars  # no space found — hard cut
    return text[:cut] + "…"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(gt_rows: list[dict], pred_rows: list[dict]) -> dict:
    """
    Match predicted rows to ground-truth rows by `issue` text (first 80 chars).
    Returns a results dict with per-row breakdown and summary metrics.
    """
    # Build lookup by issue prefix
    gt_by_issue: dict[str, dict] = {}
    for row in gt_rows:
        key = _normalise(row.get("issue", ""))[:80]
        gt_by_issue[key] = row

    results = []
    status_correct   = 0
    reqtype_correct  = 0
    area_correct     = 0
    matched          = 0
    unmatched        = 0

    for pred in pred_rows:
        issue_key = _normalise(pred.get("issue", ""))[:80]
        gt = gt_by_issue.get(issue_key)

        if gt is None:
            unmatched += 1
            results.append({
                "issue": _trunc(pred.get("issue", "")),
                "matched": False,
                "pred_status":   pred.get("status", ""),
                "gt_status":     "?",
                "status_ok":     None,
                "pred_reqtype":  pred.get("request_type", ""),
                "gt_reqtype":    "?",
                "reqtype_ok":    None,
                "pred_area":     pred.get("product_area", ""),
                "gt_area":       "?",
                "area_ok":       None,
            })
            continue

        matched += 1
        pred_status  = _normalise(pred.get("status", ""))
        gt_status    = _normalise(gt.get("status", ""))
        s_ok = pred_status == gt_status
        if s_ok:
            status_correct += 1

        pred_rt = _normalise(pred.get("request_type", ""))
        gt_rt   = _normalise(gt.get("request_type", ""))
        rt_ok = pred_rt == gt_rt
        if rt_ok:
            reqtype_correct += 1

        pred_area = _normalise(pred.get("product_area", ""))
        gt_area   = _normalise(gt.get("product_area", ""))
        area_ok = pred_area == gt_area
        if area_ok:
            area_correct += 1

        results.append({
            "issue":         _trunc(pred.get("issue", "")),
            "_issue_full":   pred.get("issue", "").replace("\n", " ").strip(),
            "matched":       True,
            "pred_status":   pred_status,
            "gt_status":     gt_status,
            "status_ok":     s_ok,
            "pred_reqtype":  pred_rt,
            "gt_reqtype":    gt_rt,
            "reqtype_ok":    rt_ok,
            "pred_area":     pred_area,
            "gt_area":       gt_area,
            "area_ok":       area_ok,
        })

    n = matched
    summary = {
        "matched_rows":       matched,
        "unmatched_rows":     unmatched,
        "total_gt":           len(gt_rows),
        "total_pred":         len(pred_rows),
        "status_correct":     status_correct,
        "status_accuracy":    round(status_correct / n, 3) if n else 0.0,
        "reqtype_correct":    reqtype_correct,
        "reqtype_accuracy":   round(reqtype_correct / n, 3) if n else 0.0,
        "area_correct":       area_correct,
        "area_accuracy":      round(area_correct / n, 3) if n else 0.0,
        "overall_accuracy":   round(
            (status_correct + reqtype_correct) / (2 * n), 3
        ) if n else 0.0,
    }
    return {"summary": summary, "rows": results}


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------

def display(report: dict) -> None:
    s = report["summary"]
    rows = report["rows"]

    # ── Summary panel ──────────────────────────────────────────────────
    stats = Table.grid(padding=(0, 3))
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")

    def _pct(val: float) -> str:
        pct = val * 100
        color = "green" if pct >= 80 else ("yellow" if pct >= 60 else "red")
        return f"[bold {color}]{pct:.0f}%[/bold {color}]"

    stats.add_row(
        f"{_pct(s['status_accuracy'])}\n[dim]Status[/dim]",
        f"{_pct(s['reqtype_accuracy'])}\n[dim]Request Type[/dim]",
        f"{_pct(s['area_accuracy'])}\n[dim]Product Area[/dim]",
        f"{_pct(s['overall_accuracy'])}\n[dim]Overall (Status+Type)[/dim]",
    )

    console.print()
    console.print(Panel(
        Align.center(stats),
        title="[bold bright_blue]Evaluation Results[/bold bright_blue]",
        subtitle=f"[dim]{s['matched_rows']}/{s['total_gt']} rows matched"
                 f"  |  {s['unmatched_rows']} unmatched[/dim]",
        border_style="bright_blue",
        padding=(1, 4),
    ))

    # ── Per-row breakdown ──────────────────────────────────────────────
    table = Table(
        title="Per-Ticket Breakdown",
        box=box.ROUNDED,
        border_style="bright_blue",
        title_style="bold bright_blue",
        show_lines=True,
        expand=True,
    )
    table.add_column("#",       width=3, justify="right", style="dim")
    table.add_column("Issue",   min_width=28, max_width=55)
    table.add_column("Status",  min_width=10, justify="center")
    table.add_column("Req Type",min_width=14, justify="center")
    table.add_column("Area",    min_width=18, justify="center")

    for i, r in enumerate(rows, 1):
        if not r["matched"]:
            table.add_row(str(i), r["issue"], "[dim]?[/dim]",
                          "[dim]?[/dim]", "[dim]unmatched[/dim]")
            continue

        def _cell(ok, pred, gt):
            if ok:
                return f"[green]{pred}[/green]"
            return f"[red]{pred}[/red]\n[dim]gt: {gt}[/dim]"

        table.add_row(
            str(i),
            r["issue"],
            _cell(r["status_ok"],  r["pred_status"],  r["gt_status"]),
            _cell(r["reqtype_ok"], r["pred_reqtype"],  r["gt_reqtype"]),
            _cell(r["area_ok"],    r["pred_area"],     r["gt_area"]),
        )

    console.print()
    console.print(table)

    # ── Mismatch drilldown: full issue text for wrong rows ───────────────────
    mismatches = [
        r for r in rows
        if r["matched"] and not all([
            r["status_ok"], r["reqtype_ok"], r["area_ok"]
        ])
    ]

    if mismatches and not getattr(display, "_no_drilldown", False):
        console.print()
        console.rule("[bold yellow]Mismatch Drilldown - Full Issue Text[/bold yellow]", style="yellow")
        for r in mismatches:
            wrong_fields = []
            if not r["status_ok"]:
                wrong_fields.append(f"status: [red]{r['pred_status']}[/red] != [green]{r['gt_status']}[/green]")
            if not r["reqtype_ok"]:
                wrong_fields.append(f"req_type: [red]{r['pred_reqtype']}[/red] != [green]{r['gt_reqtype']}[/green]")
            if not r["area_ok"]:
                wrong_fields.append(f"area: [red]{r['pred_area']}[/red] != [green]{r['gt_area']}[/green]")

            console.print()
            console.print(f"[bold white]{r['_issue_full']}[/bold white]")
            # Print each field diff on its own line to avoid markup-in-join crash
            for field in wrong_fields:
                console.print(f"  {field}")
        console.print()

    # ── Raw numbers ────────────────────────────────────────────────────
    console.print()
    console.print(
        f"  Status:   [bold]{s['status_correct']}/{s['matched_rows']}[/bold] correct  "
        f"({s['status_accuracy']*100:.0f}%)\n"
        f"  ReqType:  [bold]{s['reqtype_correct']}/{s['matched_rows']}[/bold] correct  "
        f"({s['reqtype_accuracy']*100:.0f}%)\n"
        f"  Area:     [bold]{s['area_correct']}/{s['matched_rows']}[/bold] correct  "
        f"({s['area_accuracy']*100:.0f}%)\n"
        f"  Overall:  [bold green]{s['overall_accuracy']*100:.0f}%[/bold green]  "
        f"(status + request_type combined)"
    )
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate triage agent output against ground truth CSV."
    )
    parser.add_argument(
        "--predicted", type=Path, default=PRED_PATH,
        help=f"Path to model output CSV (default: {PRED_PATH})"
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=GT_PATH,
        help=f"Path to ground-truth CSV (default: {GT_PATH})"
    )
    parser.add_argument(
        "--report", type=Path, default=REPORT_PATH,
        help=f"Path to write JSON report (default: {REPORT_PATH})"
    )
    parser.add_argument(
        "--no-drilldown", action="store_true",
        help="Suppress full-issue mismatch drilldown"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for p in [args.predicted, args.ground_truth]:
        if not p.exists():
            console.print(f"[bold red]ERROR:[/bold red] File not found: {p}")
            sys.exit(1)

    console.print(f"\n[dim]Ground truth:[/dim] [cyan]{args.ground_truth}[/cyan]")
    console.print(f"[dim]Predicted:   [/dim] [cyan]{args.predicted}[/cyan]")

    gt_rows   = _load(args.ground_truth)
    pred_rows = _load(args.predicted)

    report = evaluate(gt_rows, pred_rows)
    display._no_drilldown = args.no_drilldown  # type: ignore[attr-defined]
    display(report)

    # Save JSON report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    console.print(f"[dim]Report saved to:[/dim] [cyan]{args.report}[/cyan]\n")
