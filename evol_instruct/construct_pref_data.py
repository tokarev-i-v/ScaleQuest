import json
from datasets import Dataset


with open("output/gen_data-v2-4o-mini.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
# dataset.map(lambda x: {"chosen": dataset})
dataset = dataset.filter(lambda x: x["generation"].startswith("Step 1 #Methods List#:"))
dataset = dataset.map(
    lambda x: {"chosen": x["rewritten"], "rejected": x["query"], "query": x["rewritten"]}
)
dataset.to_json("/nvme1/dyy/QueryPreference/evol_instruct/output/dpo_data_v1/train.jsonl")