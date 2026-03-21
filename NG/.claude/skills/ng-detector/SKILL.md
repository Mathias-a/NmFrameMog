# ng-detector

Use this skill for NorgesGruppen detector code only.

## Focus

- keep the mainline path detector-first
- place detector code under `src/ng_data/detector/` and shared flow under `src/ng_data/pipeline/`
- optimize for a submission-safe baseline before optional classifier or retrieval work

## Guardrails

- do not imply retrieval is required
- do not introduce extra architecture branches unless a task explicitly asks for them
- keep outputs compatible with COCO-style prediction JSON and the future zip-root `run.py` contract

## Done when

- detector changes are isolated and typed
- tests target the detector-first workflow
- packaging assumptions stay compatible with the offline sandbox
