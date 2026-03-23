#!/bin/bash
set -e

echo "Starting Mock API Server..."
uv run mocker/main.py > mocker.log 2>&1 &
MOCK_PID=$!

# Wait for server to be ready
sleep 4

echo "Refreshing dataset against mock API to get a frozen round state..."
export AINM_ACCESS_TOKEN="dummy_token"
export AINM_BASE_URL="http://127.0.0.1:8000/astar-island"

# Note: Using python -m directly assuming the environment has the modules, or just PYTHONPATH=.
export PYTHONPATH="idk_1:idk_2:idk_3:."

# Use uv run isolated from workspace but with necessary dependencies
pycmd="uv run --no-project --with httpx --with pydantic --with numpy --with scipy python"

$pycmd -m idk_2.astar_island_dr_plan_1.cli refresh-dataset \
    --round-id mock-round-0000-1111-2222-333333333333 \
    --dataset-version mock_v1 \
    --base-url "http://127.0.0.1:8000/astar-island" \
    --cache-dir .artifacts/astar-island

ROUND_FILE=".artifacts/astar-island/datasets/mock_v1/rounds/mock-round-0000-1111-2222-333333333333.json"

echo ""
echo "======================================"
echo "Benchmarking idk_1"
echo "======================================"
# idk_1 processes per-seed, we'll benchmark one seed prediction
time $pycmd -m idk_1.cli solve \
    --round-id mock-round-0000-1111-2222-333333333333 \
    --seed-index 0 \
    --base-url "http://127.0.0.1:8000/astar-island" \
    --token dummy \
    --output .artifacts/astar-island/debug/idk1-out.json

echo ""
echo "======================================"
echo "Benchmarking idk_2"
echo "======================================"
time $pycmd -m idk_2.astar_island_dr_plan_1.cli solve-round \
    --round-detail-file "$ROUND_FILE" \
    --base-url "http://127.0.0.1:8000/astar-island" \
    --execute-live-queries \
    --cache-dir .artifacts/astar-island

echo ""
echo "======================================"
echo "Benchmarking idk_3"
echo "======================================"
time $pycmd -m idk_3.astar_island_dt_mc.solver.solve_round \
    --round-detail-file "$ROUND_FILE" \
    --cache-dir .artifacts/astar-island

echo ""
echo "Killing mock server..."
kill $MOCK_PID
Wait $MOCK_PID 2>/dev/null || true
echo "Benchmarking complete."
