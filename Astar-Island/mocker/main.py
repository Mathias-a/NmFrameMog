# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "pydantic",
# ]
# ///

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, conlist

app = FastAPI(title="Astar Island Mock API")

MOCK_ROUND_ID = "mock-round-0000-1111-2222-333333333333"
MAP_WIDTH = 40
MAP_HEIGHT = 40
SEEDS_COUNT = 5

budget_db = {
    "queries_used": 0,
    "queries_max": 50,
}

class CheckportPayload(BaseModel):
    round_id: str
    seed_index: int
    viewport_x: int = 0
    viewport_y: int = 0
    viewport_w: int = 15
    viewport_h: int = 15

class SubmitPayload(BaseModel):
    round_id: str
    seed_index: int
    prediction: List[List[List[float]]]

def generate_full_grid(seed_index: int) -> List[List[int]]:
    # Simple deterministic grid based on seed_index
    grid = []
    for y in range(MAP_HEIGHT):
        row = []
        for x in range(MAP_WIDTH):
            val = 10 if (x + y + seed_index) % 5 == 0 else 11  # mixing ocean and plains
            row.append(val)
        grid.append(row)
    return grid

def extract_viewport(grid: List[List[int]], x: int, y: int, w: int, h: int):
    clamped_x = max(0, min(x, MAP_WIDTH - 1))
    clamped_y = max(0, min(y, MAP_HEIGHT - 1))
    clamped_w = min(w, MAP_WIDTH - clamped_x)
    clamped_h = min(h, MAP_HEIGHT - clamped_y)
    
    viewport_grid = []
    for row_y in range(clamped_y, clamped_y + clamped_h):
        viewport_grid.append(grid[row_y][clamped_x:clamped_x + clamped_w])
    
    return {
        "grid": viewport_grid,
        "x": clamped_x,
        "y": clamped_y,
        "w": clamped_w,
        "h": clamped_h,
    }


@app.get("/astar-island/rounds")
def get_rounds():
    return [
        {
            "id": MOCK_ROUND_ID,
            "round_number": 1,
            "event_date": "2026-03-20",
            "status": "active",
            "map_width": MAP_WIDTH,
            "map_height": MAP_HEIGHT,
            "width": MAP_WIDTH,
            "height": MAP_HEIGHT,
            "prediction_window_minutes": 165,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "closes_at": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "round_weight": 1,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    ]

@app.get("/astar-island/rounds/{round_id}")
def get_round_detail(round_id: str):
    if round_id != MOCK_ROUND_ID:
        raise HTTPException(status_code=404, detail="Round not found")
        
    initial_states = []
    for i in range(SEEDS_COUNT):
        grid = generate_full_grid(i)
        initial_states.append({
            "grid": grid,
            "settlements": [
                {"x": 10 + i, "y": 10 + i, "has_port": True, "alive": True}
            ]
        })
        
    return {
        "id": MOCK_ROUND_ID,
        "round_number": 1,
        "status": "active",
        "map_width": MAP_WIDTH,
        "map_height": MAP_HEIGHT,
        "width": MAP_WIDTH,
        "height": MAP_HEIGHT,
        "seeds_count": SEEDS_COUNT,
        "initial_states": initial_states
    }

@app.get("/astar-island/budget")
def get_budget():
    return {
        "round_id": MOCK_ROUND_ID,
        "queries_used": budget_db["queries_used"],
        "queries_max": budget_db["queries_max"],
        "active": True
    }

@app.post("/astar-island/simulate")
def simulate(payload: CheckportPayload):
    if payload.round_id != MOCK_ROUND_ID:
        raise HTTPException(status_code=400, detail="Round not active")
        
    # Increment query usage
    budget_db["queries_used"] += 1
    
    grid = generate_full_grid(payload.seed_index)
    view = extract_viewport(grid, payload.viewport_x, payload.viewport_y, payload.viewport_w, payload.viewport_h)
    
    return {
        "grid": view["grid"],
        "settlements": [
            {
                "x": view["x"] + 2,
                "y": view["y"] + 2,
                "population": 2.5,
                "food": 0.5,
                "wealth": 0.8,
                "defense": 0.4,
                "has_port": True,
                "alive": True,
                "owner_id": 1
            }
        ] if 0 <= view["x"] + 2 < MAP_WIDTH and 0 <= view["y"] + 2 < MAP_HEIGHT else [],
        "viewport": {"x": view["x"], "y": view["y"], "w": view["w"], "h": view["h"]},
        "width": MAP_WIDTH,
        "height": MAP_HEIGHT,
        "queries_used": budget_db["queries_used"],
        "queries_max": budget_db["queries_max"]
    }

@app.post("/astar-island/submit")
def submit(payload: SubmitPayload):
    if payload.round_id != MOCK_ROUND_ID:
        raise HTTPException(status_code=400, detail="Round not active")
        
    return {
        "status": "accepted",
        "round_id": MOCK_ROUND_ID,
        "seed_index": payload.seed_index
    }

@app.get("/astar-island/analysis/{round_id}/{seed_index}")
def get_analysis(round_id: str, seed_index: int):
    # Just return a dummy tensor
    dummy_prob = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dummy_ground_truth = []
    for y in range(MAP_HEIGHT):
        row = []
        for x in range(MAP_WIDTH):
            row.append(dummy_prob)
        dummy_ground_truth.append(row)
        
    return {
        "prediction": dummy_ground_truth,
        "ground_truth": dummy_ground_truth,
        "score": 85.0,
        "width": MAP_WIDTH,
        "height": MAP_HEIGHT,
        "initial_grid": generate_full_grid(seed_index)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
