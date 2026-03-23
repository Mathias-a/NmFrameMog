import sys, os
from pathlib import Path
sys.path.insert(0, str(Path('benchmark/src').resolve()))
from astar_twin.solver.adapters.prod import _resolve_token
import httpx

token = _resolve_token()
client = httpx.Client(headers={"Authorization": f"Bearer {token}"})
r = client.get("https://api.ainm.no/astar-island/my-rounds")
rid = next(r for r in r.json() if r['status'] == 'completed')['id']

queries = client.get(f"https://api.ainm.no/astar-island/my-observations/{rid}")
print("my-observations endpoint:", queries.status_code)
