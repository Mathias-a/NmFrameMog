import os
import sys
import httpx
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def _load_token() -> str:
    token = os.environ.get("ACCESS_TOKEN", "").strip()
    if token:
        return token
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("access_token="):
                token = line.split("=", 1)[1].strip()
                if token:
                    return token
    raise RuntimeError("No API token found.")

token = _load_token()
with httpx.Client(timeout=30.0) as client:
    resp = client.get("https://api.ainm.no/astar-island/rounds")
    rounds = resp.json()
    completed_rounds = [r for r in rounds if r.get("status") == "completed"]
    if not completed_rounds:
        print("No completed rounds")
        sys.exit(0)
    
    # Sort by round_number descending
    completed_rounds.sort(key=lambda x: x.get("round_number", 0), reverse=True)
    latest = completed_rounds[0]
    print(f"Latest completed round: {latest['round_number']} (ID: {latest['id']}, Status: {latest['status']})")
    
    # Fetch analysis for seed 0
    resp = client.get(f"https://api.ainm.no/astar-island/analysis/{latest['id']}/0", headers={"Authorization": f"Bearer {token}"})
    print(f"Analysis status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print("Keys in analysis response:", data.keys())
        if "score" in data:
            print("Score:", data["score"])
        if "kl_divergence" in data:
            print("KL:", data["kl_divergence"])
