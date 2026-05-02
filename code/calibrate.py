import csv
from retriever import CorpusIndex

def run_calibration():
    # load sample CSV
    with open("support_tickets/sample_support_tickets.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    idx = CorpusIndex()
    idx.load("data", domains=["hackerrank", "claude", "visa"])
    
    replied_scores = []
    escalated_scores = []
    
    for row in rows[:10]:
        q = (row.get("Subject", "") + " " + row.get("Issue", "")).strip()
        chunks = idx.query(q, domain=None, top_k=1)
        if chunks:
            score = chunks[0].score
            if row.get("Status", "").lower() == "replied":
                replied_scores.append(score)
            else:
                escalated_scores.append(score)
        else:
            if row.get("Status", "").lower() == "replied":
                replied_scores.append(0)
            else:
                escalated_scores.append(0)
                
    print("Replied BM25 Scores:", replied_scores)
    print("Escalated BM25 Scores:", escalated_scores)

if __name__ == "__main__":
    run_calibration()
