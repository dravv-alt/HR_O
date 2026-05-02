"""Quick comparison of sample output vs expected."""
import csv, sys
sys.path.insert(0, "code")
from main import read_tickets
from pathlib import Path

sample = read_tickets(Path("support_tickets/sample_support_tickets.csv"))
produced = read_tickets(Path("support_tickets/sample_output.csv"))

print(f"Sample rows: {len(sample)}, Produced rows: {len(produced)}")
print(f"Output columns: {list(produced[0].keys())}")
print()

for i, (s, p) in enumerate(zip(sample, produced)):
    exp_st = s.get("status", "?").lower()
    got_st = p.get("status", "?").lower()
    exp_rt = s.get("request_type", "?")
    got_rt = p.get("request_type", "?")
    match = "OK" if exp_st == got_st else "MISS"
    subj = s.get("subject", s.get("issue", ""))[:40]
    print(f"  Row {i+1:>2}: status={match:>4} | exp={exp_st:>9} got={got_st:>9} | rt_exp={exp_rt:>15} rt_got={got_rt:>15} | {subj}")

status_correct = sum(
    1 for s, p in zip(sample, produced)
    if s.get("status", "").lower() == p.get("status", "").lower()
)
rt_correct = sum(
    1 for s, p in zip(sample, produced)
    if s.get("request_type", "") == p.get("request_type", "")
)
print(f"\nStatus accuracy:       {status_correct}/{len(sample)} = {status_correct/len(sample)*100:.0f}%")
print(f"Request type accuracy: {rt_correct}/{len(sample)} = {rt_correct/len(sample)*100:.0f}%")
