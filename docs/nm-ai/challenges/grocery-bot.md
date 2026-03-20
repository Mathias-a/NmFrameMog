# Grocery Bot Challenge

Sources:

- `challenge://game/overview`
- `challenge://game/mechanics`
- `challenge://game/endpoint`
- `challenge://game/scoring`
- `challenge://game/examples`

## Live MCP summary

### Overview

- "The Grocery Bot is one of four tasks in NM i AI 2026. Build a bot that controls agents via WebSocket to navigate a grocery store, pick up items, and deliver orders."
- "Task type: Real-time game (WebSocket)"
- "Platform: [app.ainm.no](https://app.ainm.no)"
- "Pick a map from the 21 available maps on the Challenge page."
- "Get a WebSocket URL — click Play to get a game token."
- "Connect your bot to the WebSocket URL."
- "Receive game state each round as JSON."

### Mechanics

- "All bots start at bottom-right of the store (inside border)."
- "Each round, your bot receives the full game state via WebSocket."
- "You respond with actions for each bot."
- "The game runs for 300 rounds maximum."
- The docs also expose a **5 Difficulty Levels** table with grid size, bot count, aisles, item types, maps, rounds, and time limit.

### Live protocol details now visible

- Connection format:

```text
wss://game.ainm.no/ws?token=<jwt_token>
```

- Every `game_state` now exposes a much richer structure than the original local snapshot captured:
  - `action_status`
  - `grid.width`, `grid.height`, `grid.walls`
  - `bots[]` with `id`, `position`, and `inventory`
  - `items[]` with `id`, `type`, and `position`
  - `orders[]` with `items_required`, `items_delivered`, `complete`, and `status`
  - `drop_off`, `score`, `active_order_index`, `total_orders`
- `action_status` has concrete live values:
  - `ok`
  - `timeout`
  - `error`
- The docs now explicitly describe a **2 second response deadline** for each round.
- The client may send an optional `round` field in its action payload to detect desync.
- Supported actions are now explicit:
  - `move_up`
  - `move_down`
  - `move_left`
  - `move_right`
  - `pick_up` with `item_id`
  - `drop_off`
  - `wait`
- Pickup and movement rules are fully documented in the live page, including silent failures for invalid moves and collision handling in bot-ID order.

### Endpoint / protocol

- The endpoint docs are titled **WebSocket Protocol Specification**.
- Request/response flow excerpt:

```json
Server -> Client: {"type": "game_state", ...}
Client -> Server: {"actions": [...]}
Server -> Client: {"type": "game_state", ...}
```

- Example state excerpt:

```json
{
  "type": "game_state",
  "round": 42,
  "max_rounds": 300
}
```

- Terminal message excerpt:

```json
{
  "type": "game_over",
  "score": 47,
  "rounds_used": 200
}
```

- "When the game ends, the server sends a `game_over` message instead of another `game_state`. This is the final message — the WebSocket closes after this."

### Scoring

- The full scoring formula is now visible in the live resource:

```text
score = items_delivered × 1 + orders_completed × 5
```

- Leaderboard scoring is also explicit: it is the **sum of your best scores across all 21 maps**.
- Live limits now visible in scoring/examples pages:
  - 60 second cooldown between games
  - 40 games per hour per team
  - 300 games per day per team
- The live scoring page also exposes:
  - daily item/order rotation at midnight UTC
  - infinite-order behavior until round/time cap
  - score examples for common scenarios
  - game end conditions: max rounds, wall-clock timeout, disconnect

### Examples and operational notes

- The examples page includes a minimal Python bot using `websockets`.
- Example control loop excerpt:

```python
if data["type"] == "game_state":
    actions = decide_actions(data)
    await ws.send(json.dumps({"actions": actions}))
```

- Example termination handling:

```python
if data["type"] == "game_over":
    print(f"Game over! Score: {data['score']}, Rounds: {data['rounds_used']}")
    break
```

- Common pitfall excerpt: not handling `game_over` causes connection errors.
- The examples page also exposes a **Rate Limits** section with `60 second cooldown between games`.

### Additional live end-of-game details

- The `game_over` payload now includes:
  - `score`
  - `rounds_used`
  - `items_delivered`
  - `orders_completed`
- The live endpoint docs also distinguish:
  - round cap: 300 normally, 500 for Nightmare
  - wall-clock timeout: 120 seconds normally, 300 for Nightmare

## What matters for implementation

- This is a low-latency state-action loop rather than a batch API.
- A robust local simulator or replay harness will matter more than fancy prompting.
- The first production-hardening task is protocol correctness: parse the full `game_state`, honor the 2-second deadline, emit valid `actions`, and close cleanly on `game_over`.
