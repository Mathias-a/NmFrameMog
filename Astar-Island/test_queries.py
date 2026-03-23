import json
with open("benchmark/previous_run_data.json") as f:
    d = json.load(f)
    print("Keys of a query:", d["queries"][0].keys())
