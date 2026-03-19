# Grocery Bot Challenge

Sources:

- `challenge://game/overview`
- `challenge://game/mechanics`
- `challenge://game/endpoint`
- `challenge://game/scoring`
- `challenge://game/examples`

## MCP excerpts

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

- The docs expose **Grocery Bot Scoring** with a **Score Formula** section.
- The excerpt we could retrieve was the heading `Per game:`; capture a fresh snapshot from the MCP later if a full formula export becomes available.

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

## What matters for implementation

- This is a low-latency state-action loop rather than a batch API.
- A robust local simulator or replay harness will matter more than fancy prompting.
- The first production-hardening task is protocol correctness: parse `game_state`, emit valid `actions`, and close cleanly on `game_over`.
