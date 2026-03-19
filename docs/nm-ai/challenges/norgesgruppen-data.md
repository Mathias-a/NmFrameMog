# NorgesGruppen Data

Sources:

- `challenge://norgesgruppen-data/overview`
- `challenge://norgesgruppen-data/submission`
- `challenge://norgesgruppen-data/scoring`
- `challenge://norgesgruppen-data/examples`

## MCP excerpts

### Overview

- "NorgesGruppen Data: Object Detection"
- "Detect grocery products on store shelves. Upload your model code as a `.zip` file — it runs in a sandboxed Docker container on our servers."
- "Upload your `.zip` at the submission page on the competition website."
- The overview also includes an **MCP Setup** section for connecting the docs server to an AI coding tool.

### Submission format

- The submission docs are titled **Submission Format**.
- Zip requirement excerpt: "Your `.zip` must contain `run.py` at the root. You may include model weights and Python helper files."
- The page also includes a **Sandbox Environment** section: "Your code runs in a Docker container with these constraints:"

### Scoring

- The scoring docs expose **mAP@0.5**.
- "Both components use mAP@0.5 (Mean Average Precision at IoU threshold 0.5)."
- The scoring page includes **Hybrid Scoring** and states: "Your final score combines detection and classification."

### Examples and common errors

- The examples page includes a **Random Baseline** section.
- It also includes a **Common Errors** table with `Error` and `Fix` columns.

## What matters for implementation

- This task is closer to reproducible ML packaging than interactive agent serving.
- Winning depends on a deterministic offline evaluation pipeline, a clean `run.py` entrypoint, and strict sandbox compatibility.
- The repo should treat this as a model packaging + scoring loop, not a chat-agent problem.
