import json
import os

path = "data/graphdb/graph_store.json"
print("Loading graph store...")
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)

rels = d.get("relations", {})
triplets = d.get("triplets", [])

print(f"Original relations: {len(rels)}")
print(f"Original triplets: {len(triplets)}")

new_triplets = []
missing_count = 0

for t in triplets:
    # t is [subj, rel, obj]
    key = f"{t[0]}_{t[1]}_{t[2]}"
    if key in rels:
        new_triplets.append(t)
    else:
        missing_count += 1

print(f"Missing relations found: {missing_count}")
d["triplets"] = new_triplets

with open(path, "w", encoding="utf-8") as f:
    json.dump(d, f)

print("Graph store repaired successfully!")
