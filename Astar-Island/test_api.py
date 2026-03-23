import sys, os
from pathlib import Path
sys.path.insert(0, str(Path('benchmark/src').resolve()))
from astar_twin.solver.adapters.prod import _resolve_token
import httpx

token = _resolve_token()
client = httpx.Client(headers={"Authorization": f"Bearer {token}"})
r = client.get("https://api.ainm.no/astar-island/my-rounds")
rounds = r.json()

completed_round = next(r for r in rounds if r['status'] == 'completed')
rid = completed_round['id']
print(f"Checking round {rid} ({completed_round['round_number']})")
pred = client.get(f"https://api.ainm.no/astar-island/my-predictions/{rid}")
data = pred.json()
print("My predictions is list?", isinstance(data, list))
if isinstance(data, list) and len(data) > 0:
    print("Keys in prediction:", data[0].keys())

print("\nLet's check GET /astar-island/analysis/{rid}/0")
analysis = client.get(f"https://api.ainm.no/astar-island/analysis/{rid}/0")
adata = analysis.json()
if isinstance(adata, dict):
    print("Keys in analysis:", adata.keys())
else:
    print("Analysis is not a dict:", type(adata))

print("\nLet's see if there is an endpoint for my-queries")
queries = client.get(f"https://api.ainm.no/astar-island/my-queries/{rid}")
print("my-queries endpoint:", queries.status_code)
