# NorgesGruppen Data Agents

Use this repo path for NorgesGruppen work:

- training and local tooling live under `src/ng_data/`
- task checks live under `tests/unit/`
- lightweight helper entrypoints live under `scripts/`

Operator rules for this challenge:

- keep the delivery path detector-first; retrieval is optional later, not the default
- target the offline sandbox contract from `docs/submission.md`
- package submissions with `run.py` at the zip root
- assume no network and avoid blocked imports such as `os`, `subprocess`, and `yaml`
- keep the primary cloud path simple: GCE + GCS first, Vertex AI only if later tasks require it

Preferred workflow:

1. validate project structure with `python -m src.ng_data.cli.doctor`
2. work inside the relevant `src/ng_data/` package
3. verify with the structure tests before expanding tooling
