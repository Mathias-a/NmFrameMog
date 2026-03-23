import json, sys
from pathlib import Path
sys.path.insert(0, str(Path('benchmark/src').resolve()))
from astar_twin.solver.adapters.prod import _resolve_token
import httpx

token = _resolve_token()
client = httpx.Client(headers={"Authorization": f"Bearer {token}"})

# 1. Get rounds
rounds_resp = client.get("https://api.ainm.no/astar-island/my-rounds")
rounds = rounds_resp.json()
completed_rounds = [r for r in rounds if r['status'] == 'completed']

# Get the most recent one that we also have the fixture for
fixture_dirs = [d.name for d in Path("data/rounds").iterdir() if d.is_dir()]
matching_rounds = [r for r in completed_rounds if r['id'] in fixture_dirs]

if not matching_rounds:
    print("No matching completed rounds.")
    sys.exit(1)

matching_rounds.sort(key=lambda x: x['round_number'], reverse=True)
rid = matching_rounds[0]['id']
rnum = matching_rounds[0]['round_number']
print(f"Extracting for round {rnum} ({rid})")

queries = client.get(f"https://api.ainm.no/astar-island/my-queries/{rid}").json()
print(f"Found {len(queries)} queries.")

analysis_data = []
for seed in range(matching_rounds[0].get('seeds_count', 5)):
    resp = client.get(f"https://api.ainm.no/astar-island/analysis/{rid}/{seed}")
    if resp.status_code == 200:
        analysis_data.append(resp.json())
    else:
        print(f"Failed to get analysis for seed {seed}")

data = {
    "round_id": rid,
    "round_number": rnum,
    "queries": queries,
    "analysis": analysis_data
}
with open("previous_run_data.json", "w") as f:
    json.dump(data, f)
print("Saved to previous_run_data.json")
