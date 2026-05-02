import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path("code").resolve()))
from retriever import _infer_product_area

p = Path("data/index/contextual_chunks.json")
data = json.loads(p.read_text(encoding="utf-8"))
data_dir = Path("data")

changed = 0
for chunk in data["chunks"]:
    path = Path(chunk["source"])
    new_area = _infer_product_area(path, data_dir, [])
    if chunk.get("product_area") != new_area:
        chunk["product_area"] = new_area
        changed += 1

p.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"Updated {changed} chunks.")
