# NorgesGruppen Data

Sources:

- `challenge://norgesgruppen-data/overview`
- `challenge://norgesgruppen-data/submission`
- `challenge://norgesgruppen-data/scoring`
- `challenge://norgesgruppen-data/examples`

## Live MCP summary

### Overview

- "NorgesGruppen Data: Object Detection"
- "Detect grocery products on store shelves. Upload your model code as a `.zip` file — it runs in a sandboxed Docker container on our servers."
- "Upload your `.zip` at the submission page on the competition website."
- The overview also includes an **MCP Setup** section for connecting the docs server to an AI coding tool.
- The live overview page now also exposes concrete dataset details:
  - COCO dataset size and annotation scale
  - `357` product categories (`0–356`)
  - product reference images with multi-angle photos
  - explicit COCO annotation fields such as `bbox`, `product_code`, and `corrected`
- The live overview also confirms the sandbox uses an **NVIDIA L4 GPU (24 GB VRAM)** with no network access.

### Submission format

- The submission docs are titled **Submission Format**.
- Zip requirement excerpt: "Your `.zip` must contain `run.py` at the root. You may include model weights and Python helper files."
- The page also includes a **Sandbox Environment** section: "Your code runs in a Docker container with these constraints:"
- The full live submission page now exposes exact limits:
  - max zip size: `420 MB`
  - max files: `1000`
  - max Python files: `10`
  - max weight files: `3`
  - max total weight size: `420 MB`
  - explicit allowed file types
- `run.py` contract is fully visible:

```bash
python run.py --input /data/images --output /output/predictions.json
```

- The live page also fully documents:
  - output JSON schema (`image_id`, `category_id`, `bbox`, `score`)
  - Python version, CPU, memory, GPU, CUDA, timeout, and network isolation
  - preinstalled package versions
  - blocked imports and security restrictions
  - ONNX/model-version compatibility guidance
  - zip-packaging pitfalls on macOS/Windows

### Scoring

- The scoring docs expose **mAP@0.5**.
- "Both components use mAP@0.5 (Mean Average Precision at IoU threshold 0.5)."
- The scoring page includes **Hybrid Scoring** and states: "Your final score combines detection and classification."
- The live scoring page now also exposes:
  - explicit formula `0.7 × detection_mAP + 0.3 × classification_mAP`
  - detection-only submissions can score up to `0.70`
  - submission limits:
    - `2` in-flight per team
    - `3` submissions per day
    - `2` infrastructure-failure freebies per day

### Examples and common errors

- The examples page includes a **Random Baseline** section.
- It also includes a **Common Errors** table with `Error` and `Fix` columns.
- The live examples page now also exposes:
  - a YOLOv8 example with GPU auto-detection
  - ONNX export and inference examples
  - concrete fixes for zip-root errors, disallowed file types, timeout, OOM, segfault, and version mismatch failures

## What matters for implementation

- This task is closer to reproducible ML packaging than interactive agent serving.
- Winning depends on a deterministic offline evaluation pipeline, a clean `run.py` entrypoint, and strict sandbox compatibility.
- The repo should treat this as a model packaging + scoring loop, not a chat-agent problem.
- The live docs make environment matching a first-class requirement: package versions, allowed file types, GPU assumptions, and blocked imports are all part of the contract.
