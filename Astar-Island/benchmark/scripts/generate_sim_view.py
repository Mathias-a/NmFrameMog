import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

from astar_twin.engine.simulator import Simulator
from astar_twin.params.simulation_params import SimulationParams
from astar_twin.data.loaders import load_fixture
from numpy.random import default_rng
from astar_twin.contracts.types import SIM_YEARS, TerrainCode

def color_for_code(code: int, is_ruin: bool = False) -> str:
    if is_ruin:
        return "#777777"
    if code == TerrainCode.OCEAN:
        return "#1a5276"
    if code == TerrainCode.PLAINS:
        return "#7dce13"
    if code == TerrainCode.EMPTY:
        return "#d4e157"
    if code == TerrainCode.SETTLEMENT:
        return "#e74c3c"
    if code == TerrainCode.PORT:
        return "#9b59b6"
    if code == TerrainCode.RUIN:
        return "#777777"
    if code == TerrainCode.FOREST:
        return "#1e8449"
    if code == TerrainCode.MOUNTAIN:
        return "#5d6d7e"
    return "#000000"

def generate_html(map_idx=1, sim_seed=1337):
    fixture_path = Path("data/rounds/b0f9d1bf-4b71-4e6e-816c-19c718d29056")
    fixture = load_fixture(fixture_path)
    
    # We use default params
    params = SimulationParams()
    sim = Simulator(params)
    rng = default_rng(sim_seed)
    
    frames = []
    
    # Init state
    state = sim.init_state(fixture.initial_states[map_idx], rng)
    
    from astar_twin.phases import apply_growth, apply_conflict, apply_trade, apply_winter, apply_environment
    from astar_twin.engine.simulator import _sync_settlement_cells
    
    war_registry = {}
    prev_severity = 0.0
    
    # Pre-calculate frames
    for year in range(SIM_YEARS + 1):
        if year > 0:
            state = apply_growth(state, params, rng)
            state = apply_conflict(state, params, rng, war_registry, year)
            state = apply_trade(state, params, rng, war_registry, year)
            state, prev_severity = apply_winter(state, params, rng, prev_severity)
            state = apply_environment(state, params, rng)
            state = _sync_settlement_cells(state)
            state.year = year
            
        grid_data = []
        for y in range(state.grid.height):
            row = []
            for x in range(state.grid.width):
                code = state.grid.get(y, x)
                row.append(color_for_code(code))
            grid_data.append(row)
            
        settlement_data = []
        for s in state.settlements:
            settlement_data.append({
                "y": s.y,
                "x": s.x,
                "alive": s.alive,
                "owner": s.owner_id,
                "pop": round(s.population, 1),
                "food": round(s.food, 1),
                "wealth": round(s.wealth, 1)
            })
            
        frames.append({
            "year": year,
            "grid": grid_data,
            "settlements": settlement_data
        })
        
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Astar Island Simulator View - Seed {map_idx}</title>
        <style>
            body {{ font-family: monospace; display: flex; flex-direction: column; align-items: center; background: #222; color: #fff; }}
            #grid {{ display: grid; grid-template-columns: repeat({fixture.map_width}, 15px); gap: 1px; background: #000; padding: 2px; }}
            .cell {{ width: 15px; height: 15px; display: flex; align-items: center; justify-content: center; font-size: 8px; font-weight: bold; color: white; }}
            .controls {{ margin: 20px 0; display: flex; gap: 10px; align-items: center; }}
            button {{ padding: 5px 15px; cursor: pointer; }}
            #info {{ margin-top: 20px; text-align: left; width: 600px; }}
        </style>
    </head>
    <body>
        <h2>Seed: {map_idx} | Year: <span id="year-disp">0</span></h2>
        
        <div class="controls">
            <button onclick="prev()">Prev Year</button>
            <input type="range" id="slider" min="0" max="{SIM_YEARS}" value="0" oninput="setYear(this.value)" style="width: 300px;">
            <button onclick="next()">Next Year</button>
            <button onclick="togglePlay()" id="play-btn">Play</button>
        </div>
        
        <div id="grid"></div>
        
        <div id="info">
            <h3>Settlement Stats (Hover)</h3>
            <pre id="cell-info">Hover over a settlement...</pre>
        </div>
        
        <script>
            const frames = {json.dumps(frames)};
            let currentYear = 0;
            let playing = false;
            let interval = null;
            
            function render() {{
                const frame = frames[currentYear];
                document.getElementById("year-disp").innerText = currentYear;
                document.getElementById("slider").value = currentYear;
                
                const gridEl = document.getElementById("grid");
                gridEl.innerHTML = '';
                
                // Map settlements for quick lookup
                const sMap = {{}};
                for (const s of frame.settlements) {{
                    sMap[s.y + ',' + s.x] = s;
                }}
                
                for (let y = 0; y < frame.grid.length; y++) {{
                    for (let x = 0; x < frame.grid[y].length; x++) {{
                        const cell = document.createElement("div");
                        cell.className = "cell";
                        cell.style.backgroundColor = frame.grid[y][x];
                        
                        const key = y + ',' + x;
                        if (sMap[key]) {{
                            const s = sMap[key];
                            cell.innerText = s.alive ? s.owner : "X";
                            if (!s.alive) cell.style.opacity = "0.5";
                            
                            cell.onmouseover = () => {{
                                document.getElementById("cell-info").innerText = 
                                    `Pos: ${{x}},${{y}}\\nOwner: ${{s.owner}}\\nAlive: ${{s.alive}}\\nPop: ${{s.pop}}\\nFood: ${{s.food}}\\nWealth: ${{s.wealth}}`;
                            }};
                        }}
                        
                        gridEl.appendChild(cell);
                    }}
                }}
            }}
            
            function setYear(y) {{
                currentYear = parseInt(y);
                render();
            }}
            
            function prev() {{
                if (currentYear > 0) setYear(currentYear - 1);
            }}
            
            function next() {{
                if (currentYear < frames.length - 1) setYear(currentYear + 1);
                else togglePlay(false); // Stop at end
            }}
            
            function togglePlay(force) {{
                if (force !== undefined) playing = !force; // inverted logic for toggle
                
                playing = !playing;
                document.getElementById("play-btn").innerText = playing ? "Pause" : "Play";
                
                if (playing) {{
                    if (currentYear >= frames.length - 1) setYear(0);
                    interval = setInterval(next, 500);
                }} else {{
                    clearInterval(interval);
                }}
            }}
            
            // Initial render
            render();
        </script>
    </body>
    </html>
    """
    
    with open("sim_view.html", "w") as f:
        f.write(html)
        
    print(f"Generated {Path('sim_view.html').resolve()}")

if __name__ == "__main__":
    generate_html(map_idx=1, sim_seed=1337)
