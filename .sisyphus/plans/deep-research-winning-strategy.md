# Deep Research README + Winning Strategy Consolidation

## TL;DR
> **Summary**: Synthesize the three `deep_research/` folders into source-grounded documentation: one README per folder plus a cross-folder winning-strategy index that maximizes expected NM i AI score under equally weighted normalized task scoring.
> **Deliverables**:
> - `deep_research/astar-island/README.md`
> - `deep_research/NG/README.md`
> - `deep_research/tripx/README.md`
> - `deep_research/README.md`
> **Effort**: Short
> **Parallel**: YES - 2 waves
> **Critical Path**: Task 1 → Tasks 2/3/4 → Tasks 5/6

## Context
### Original Request
- Review the different deep research plans in the repository.
- Summarize each in a README.
- Create a concrete implementation TODO list for each research method and potentially a combined one.
- Optimize for the strongest possible NM i AI competition outcome.

### Interview Summary
- TODOs must be organized by research folder: `astar-island`, `NG`, and `tripx`.
- Summary outputs should be per-folder READMEs.
- Combined strategy should optimize for **max score only**.

### Metis Review (gaps addressed)
- `tripx` is mixed/duplicated and must be disclosed as such, not treated as a clean third independent method.
- `NG` contains two distinct strategy tracks (forecasting and object detection) and the README must preserve that split.
- Unknowns must stay labeled as unknown; no invented scoring specs, datasets, or APIs.
- The combined document needs an explicit canonical-source map and score-maximization rationale.

## Work Objectives
### Core Objective
Create source-grounded READMEs that summarize each deep-research folder and embed concrete implementation TODOs, then publish a cross-folder winning strategy that prioritizes normalized total competition score rather than equal narrative coverage.

### Deliverables
- `deep_research/astar-island/README.md` with summary, source files, unknowns, and implementation TODOs.
- `deep_research/NG/README.md` with separate forecasting and object-detection sections, scoring notes, and implementation TODOs.
- `deep_research/tripx/README.md` that clearly marks duplicated/mixed provenance and states how to use or ignore the folder.
- `deep_research/README.md` as the combined strategy/index containing canonical-source mapping, score model, cross-folder comparison, and a prioritized winning TODO list.

### Definition of Done (verifiable conditions with commands)
- `test -f deep_research/README.md && test -f deep_research/astar-island/README.md && test -f deep_research/NG/README.md && test -f deep_research/tripx/README.md`
- `python - <<'PY'
from pathlib import Path
files = [
  Path('deep_research/README.md'),
  Path('deep_research/astar-island/README.md'),
  Path('deep_research/NG/README.md'),
  Path('deep_research/tripx/README.md'),
]
required = {
  'deep_research/README.md': ['# ', '## Canonical Sources', '## Winning Strategy', '## Combined Implementation TODOs'],
  'deep_research/astar-island/README.md': ['# ', '## Source Files', '## Summary', '## Implementation TODOs', '## Unknowns'],
  'deep_research/NG/README.md': ['# ', '## Source Files', '## Forecasting Strategy', '## Object Detection Strategy', '## Implementation TODOs'],
  'deep_research/tripx/README.md': ['# ', '## Source Files', '## Mixed Provenance', '## Implementation TODOs', '## Recommended Usage'],
}
for f in files:
    text = f.read_text()
    for needle in required[str(f)]:
        assert needle in text, (f, needle)
PY`
- `python - <<'PY'
from pathlib import Path
text = Path('deep_research/README.md').read_text()
assert 'normalized' in text.lower()
assert '33.33' in text or 'equally weighted' in text.lower()
assert 'tripx' in text.lower() and 'duplicate' in text.lower()
PY`

### Must Have
- Every README is explicitly tied to concrete source files in its folder.
- Every README includes a concrete implementation TODO list using action verbs.
- `NG/README.md` keeps forecasting and object detection separate.
- `tripx/README.md` explains duplication/mixed content rather than pretending the folder is independent.
- `deep_research/README.md` explains why the chosen combined strategy maximizes normalized overall score.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Must NOT invent datasets, submission schemas, APIs, or exact scoring metrics where the source docs are silent.
- Must NOT count duplicated `tripx` content as independent evidence.
- Must NOT rewrite or implement source code, training pipelines, or benchmark scripts.
- Must NOT collapse NG forecasting and NG object detection into one vague blended method.
- Must NOT optimize the combined write-up for balance or maintainability over score; score is the explicit target.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after + command-level assertions on file existence, required headings, source citations, and score-rationale keywords.
- QA policy: Every task includes agent-executed checks for headings, citations, and folder-specific constraints.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: canonical-source map + README contract, astar-island README, NG README, tripx README

Wave 2: combined TODO matrix + final winning-strategy synthesis

### Dependency Matrix (full, all tasks)
| Task | Depends On | Unlocks |
|---|---|---|
| 1 | None | 2, 3, 4, 5, 6 |
| 2 | 1 | 5, 6 |
| 3 | 1 | 5, 6 |
| 4 | 1 | 5, 6 |
| 5 | 1, 2, 3, 4 | 6 |
| 6 | 1, 2, 3, 4, 5 | Final Verification |

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 4 tasks → writing / unspecified-low
- Wave 2 → 2 tasks → writing / deep

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Build canonical source map and README contract

  **What to do**: Inspect `deep_research/astar-island/plan_1`, `deep_research/astar-island/plan_2`, `deep_research/NG/deep-research-1`, `deep_research/NG/deep-research-1.bak`, `deep_research/NG/plan_2`, `deep_research/tripx/plan_1`, and `deep_research/tripx/plan_2`. Create a canonical-source matrix inside `deep_research/README.md` draft notes that classifies each file as canonical, duplicate, backup, or mixed. Declare the exact README section contract all folder READMEs must follow. Mark `tripx/plan_1` as duplicate retail-detection content and `tripx/plan_2` as duplicate A-Star-Island-style probabilistic content unless direct file comparison reveals otherwise.
  **Must NOT do**: Do not summarize from memory. Do not cite `.bak` as independent evidence. Do not invent any missing dataset, API, or scoring details.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: documentation schema and source-grounded synthesis.
  - Skills: `[]` — no extra skill required.
  - Omitted: `['git-master']` — no git operation needed for execution.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 3, 4, 5, 6 | Blocked By: none

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `deep_research/NG/deep-research-1:312-320` — normalized 0–100 per-task scoring and null-submission warning.
  - Pattern: `deep_research/NG/deep-research-1:373-406` — public repository and operational submission constraints.
  - Pattern: `deep_research/NG/plan_2:1-8` — object-detection workflow, mAP@0.5, SAHI, TensorRT, WBF.
  - Pattern: `deep_research/tripx/plan_1:1-8` — duplicate of NG shelf-detection plan; must be treated as duplicate unless a diff shows otherwise.
  - Pattern: `deep_research/astar-island/plan_2:1` — probabilistic ABM framing and final spatial probability distribution objective.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `deep_research/README.md` contains a `## Canonical Sources` table listing all seven plan files and one classification per file.
  - [ ] The table marks `NG/deep-research-1.bak` as backup/reference only, not canonical.
  - [ ] The README contract lists required headings for every folder README.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Canonical map exists and classifies every source
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/README.md').read_text()
for needle in [
    '## Canonical Sources',
    'deep_research/astar-island/plan_1',
    'deep_research/astar-island/plan_2',
    'deep_research/NG/deep-research-1',
    'deep_research/NG/deep-research-1.bak',
    'deep_research/NG/plan_2',
    'deep_research/tripx/plan_1',
    'deep_research/tripx/plan_2',
]:
    assert needle in text, needle
PY
    Expected: command exits 0 and the source map covers all known research files.
    Evidence: .sisyphus/evidence/task-1-canonical-source-map.txt

  Scenario: Backup file is not treated as canonical
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/README.md').read_text().lower()
assert 'deep-research-1.bak' in text
assert 'backup' in text
PY
    Expected: command exits 0 and the backup classification is explicit.
    Evidence: .sisyphus/evidence/task-1-canonical-source-map-error.txt
  ```

  **Commit**: YES | Message: `docs(deep-research): add canonical source map and README contract` | Files: `deep_research/README.md`

- [ ] 2. Write `astar-island/README.md` from source plans only

  **What to do**: Create `deep_research/astar-island/README.md` with these exact sections: `## Source Files`, `## Summary`, `## Core Research Method`, `## Evaluation Notes`, `## Implementation TODOs`, and `## Unknowns`. Summarize `plan_1` and `plan_2` as a probabilistic forecasting approach for a partially observable agent-based model. State clearly that explicit API/spec/scoring details remain unavailable in the folder. Produce a concrete TODO list focused on acquiring official specs, creating a reproducible inference harness, defining local evaluation proxies, and preparing a submission pipeline.
  **Must NOT do**: Do not claim an exact official metric unless the source states it. Do not reframe this as deterministic pathfinding only. Do not mention code artifacts that do not exist.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: source-grounded summary writing.
  - Skills: `[]` — no extra skill required.
  - Omitted: `['playwright']` — no browser or UI work required.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 6 | Blocked By: 1

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `deep_research/astar-island/plan_1:1` — championship framing, 69-hour window, three independent challenges, and partial launch-opacity warning.
  - Pattern: `deep_research/astar-island/plan_2:1` — black-box simulator, fifty-year horizon, final spatial probability distribution objective.
  - Pattern: `deep_research/tripx/plan_2:1` — duplicate/supporting wording for the same probabilistic framing; use only as secondary corroboration.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `deep_research/astar-island/README.md` exists with all required headings.
  - [ ] The README explicitly contains `Unknown` or `not specified in source` language for missing API/scoring specifics.
  - [ ] `## Implementation TODOs` contains at least 5 atomic action items starting with verbs.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Astar README has required sections and uncertainty markers
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/astar-island/README.md').read_text()
for needle in ['## Source Files', '## Summary', '## Core Research Method', '## Evaluation Notes', '## Implementation TODOs', '## Unknowns']:
    assert needle in text, needle
assert 'not specified in source' in text.lower() or 'unknown' in text.lower()
PY
    Expected: command exits 0 and the README preserves source uncertainty.
    Evidence: .sisyphus/evidence/task-2-astar-readme.txt

  Scenario: Astar TODOs are concrete and actionable
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
lines = Path('deep_research/astar-island/README.md').read_text().splitlines()
todo_lines = [ln.strip() for ln in lines if ln.strip().startswith('- [ ]')]
assert len(todo_lines) >= 5, todo_lines
verbs = ('Acquire', 'Confirm', 'Create', 'Build', 'Define', 'Implement', 'Package', 'Document')
assert all(any(t.startswith(f'- [ ] {v}') for v in verbs) for t in todo_lines[:5])
PY
    Expected: command exits 0 and at least five checkbox TODOs begin with action verbs.
    Evidence: .sisyphus/evidence/task-2-astar-readme-error.txt
  ```

  **Commit**: YES | Message: `docs(astar-island): add research summary and implementation todos` | Files: `deep_research/astar-island/README.md`

- [ ] 3. Write `NG/README.md` with separated strategy tracks

  **What to do**: Create `deep_research/NG/README.md` with sections `## Source Files`, `## Forecasting Strategy`, `## Object Detection Strategy`, `## Competition Constraints`, `## Implementation TODOs`, and `## Unknowns`. Summarize `deep-research-1` as the forecasting plan and `plan_2` as the object-detection plan. Include the normalized/equally weighted competition framing and the public-repo constraint from `deep-research-1`. In `## Implementation TODOs`, maintain two labeled sublists: forecasting and object detection.
  **Must NOT do**: Do not merge the two strategy tracks into one blended paragraph. Do not use `.bak` as a primary source. Do not omit mAP@0.5, contiguous CV, or public-repo constraints.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: structured technical synthesis with multiple subtracks.
  - Skills: `[]` — no extra skill required.
  - Omitted: `['dev-browser']` — no web interaction needed.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 6 | Blocked By: 1

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `deep_research/NG/deep-research-1:6-31` — 69-hour competition framing and predictive-performance objective.
  - Pattern: `deep_research/NG/deep-research-1:111-118` — stockout/censorship mitigation recommendation.
  - Pattern: `deep_research/NG/deep-research-1:132-177` — lag, rolling, promotion, and categorical feature factory.
  - Pattern: `deep_research/NG/deep-research-1:312-320` — normalized 0–100 score, equal weighting, null submission penalty.
  - Pattern: `deep_research/NG/deep-research-1:373-406` — public repo and submission readiness.
  - Pattern: `deep_research/NG/plan_2:1-8` — object-detection metric and end-to-end workflow.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `deep_research/NG/README.md` exists with all required headings.
  - [ ] The README contains both `mAP@0.5` and `contiguous`/`chronological` validation language.
  - [ ] The implementation TODO section contains at least 4 forecasting TODOs and 4 detection TODOs.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: NG README preserves both strategy tracks and key constraints
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/NG/README.md').read_text()
for needle in ['## Source Files', '## Forecasting Strategy', '## Object Detection Strategy', '## Competition Constraints', '## Implementation TODOs', '## Unknowns', 'mAP@0.5']:
    assert needle in text, needle
assert 'contiguous' in text.lower() or 'chronological' in text.lower()
assert 'public repo' in text.lower() or 'public repository' in text.lower()
PY
    Expected: command exits 0 and both NG tracks plus operational constraints are documented.
    Evidence: .sisyphus/evidence/task-3-ng-readme.txt

  Scenario: NG TODOs are split by forecasting and detection
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/NG/README.md').read_text()
assert 'Forecasting TODOs' in text or '### Forecasting TODOs' in text
assert 'Detection TODOs' in text or '### Detection TODOs' in text
todo_lines = [ln for ln in text.splitlines() if ln.strip().startswith('- [ ]')]
assert len(todo_lines) >= 8, len(todo_lines)
PY
    Expected: command exits 0 and the TODO list preserves two separate NG workstreams.
    Evidence: .sisyphus/evidence/task-3-ng-readme-error.txt
  ```

  **Commit**: YES | Message: `docs(NG): add strategy summary and implementation todos` | Files: `deep_research/NG/README.md`

- [ ] 4. Write `tripx/README.md` as a mixed-provenance folder guide

  **What to do**: Create `deep_research/tripx/README.md` with sections `## Source Files`, `## Mixed Provenance`, `## Summary`, `## Implementation TODOs`, and `## Recommended Usage`. Explain that `plan_1` mirrors the NG shelf-detection strategy and `plan_2` mirrors the A-Star-Island probabilistic strategy unless the executor finds concrete differences during comparison. The TODOs must focus on folder rationalization: compare against canonical sources, decide whether `tripx` remains an archive/index/integration area, and prevent duplicate strategy counting in later planning.
  **Must NOT do**: Do not present `tripx` as an independent third challenge method. Do not produce score-weight recommendations based on duplicate evidence. Do not hide the duplication issue.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: documentation and provenance clarification.
  - Skills: `[]` — no extra skill required.
  - Omitted: `['oracle']` — no architecture consultation needed at execution time.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 6 | Blocked By: 1

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `deep_research/tripx/plan_1:1-8` — duplicated NG retail-detection content.
  - Pattern: `deep_research/tripx/plan_2:1` — duplicated A-Star-Island probabilistic content.
  - Pattern: `deep_research/NG/plan_2:1-8` — canonical comparison target for `tripx/plan_1`.
  - Pattern: `deep_research/astar-island/plan_2:1` — canonical comparison target for `tripx/plan_2`.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `deep_research/tripx/README.md` exists with all required headings.
  - [ ] The README explicitly contains `duplicate`, `mixed`, or `mirrors` language.
  - [ ] The TODOs include a concrete canonicalization decision task.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Tripx README explicitly documents mixed provenance
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/tripx/README.md').read_text().lower()
for needle in ['## source files', '## mixed provenance', '## summary', '## implementation todos', '## recommended usage']:
    assert needle in text, needle
assert 'duplicate' in text or 'mirrors' in text or 'mixed' in text
PY
    Expected: command exits 0 and the README makes duplicate provenance explicit.
    Evidence: .sisyphus/evidence/task-4-tripx-readme.txt

  Scenario: Tripx TODOs include canonicalization action
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/tripx/README.md').read_text().lower()
todo_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith('- [ ]')]
assert any('canonical' in ln or 'compare' in ln or 'archive' in ln or 'index' in ln for ln in todo_lines), todo_lines
PY
    Expected: command exits 0 and at least one TODO addresses folder rationalization.
    Evidence: .sisyphus/evidence/task-4-tripx-readme-error.txt
  ```

  **Commit**: YES | Message: `docs(tripx): document mixed provenance and implementation todos` | Files: `deep_research/tripx/README.md`

- [ ] 5. Derive a comparable implementation TODO matrix across all folders

  **What to do**: After the three folder READMEs exist, add a `## Combined Implementation TODOs` section to `deep_research/README.md` that normalizes the research methods into a common planning matrix. Group TODOs by folder (`astar-island`, `NG`, `tripx`) and add one final subsection `### Shared score-maximizing priorities`. Shared priorities must emphasize: always submit something for every competition task, prioritize NG because it has the clearest metrics and operational guidance, reuse duplicate `tripx` insights only once, and explicitly label unknown-spec work for A-Star-Island as risk-reduction work.
  **Must NOT do**: Do not treat duplicated `tripx` items as extra votes. Do not collapse folder-specific TODOs into generic filler. Do not produce implementation steps unsupported by the research notes.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: cross-document consolidation and prioritization.
  - Skills: `[]` — no extra skill required.
  - Omitted: `['refactor']` — no code transformations involved.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 6 | Blocked By: 1, 2, 3, 4

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `deep_research/astar-island/README.md` — finalized folder summary and TODO list from Task 2.
  - Pattern: `deep_research/NG/README.md` — finalized folder summary and TODO list from Task 3.
  - Pattern: `deep_research/tripx/README.md` — finalized mixed-provenance explanation and TODO list from Task 4.
  - Pattern: `deep_research/NG/deep-research-1:312-320` — equal-weight normalized scoring means all competition tasks matter.
  - Pattern: `deep_research/NG/deep-research-1:320` — null submission penalty informs “always ship a baseline” priority.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `deep_research/README.md` contains `## Combined Implementation TODOs`.
  - [ ] The section contains four subsections: `Astar-Island`, `NG`, `Tripx`, and `Shared score-maximizing priorities`.
  - [ ] At least one shared priority explicitly mentions avoiding a zero / null submission on any task.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Combined TODO matrix exists with required structure
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/README.md').read_text()
for needle in ['## Combined Implementation TODOs', '### Astar-Island', '### NG', '### Tripx', '### Shared score-maximizing priorities']:
    assert needle in text, needle
PY
    Expected: command exits 0 and the combined TODO matrix is structurally complete.
    Evidence: .sisyphus/evidence/task-5-combined-todo-matrix.txt

  Scenario: Shared priorities reflect equal-weight score logic
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/README.md').read_text().lower()
assert 'null submission' in text or 'zero' in text
assert 'equally weighted' in text or '33.33' in text or 'normalized' in text
PY
    Expected: command exits 0 and the prioritization matches the documented competition scoring model.
    Evidence: .sisyphus/evidence/task-5-combined-todo-matrix-error.txt
  ```

  **Commit**: YES | Message: `plan(deep-research): add combined implementation todo matrix` | Files: `deep_research/README.md`

- [ ] 6. Write the combined winning strategy in `deep_research/README.md`

  **What to do**: Finalize `deep_research/README.md` as the combined index and winning-strategy document. Required sections: `## Canonical Sources`, `## Folder Summaries`, `## Winning Strategy`, `## Combined Implementation TODOs`, and `## Open Risks`. In `## Winning Strategy`, explicitly explain the max-score logic: because competition tasks are normalized and equally weighted, the team should first ensure baseline coverage on all tasks, then push the clearest high-leverage task(s) hardest, especially NG where the source material includes explicit metrics (`mAP@0.5`), contiguous validation guidance, and operational deadlines. Use A-Star-Island work as a risk-managed probabilistic track and treat `tripx` as a supporting/duplicate source rather than a separate pillar.
  **Must NOT do**: Do not present the strategy as official competition guidance. Do not say any path is proven if it is only proposed in the research notes. Do not recommend ignoring a task completely.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: competitive prioritization across heterogeneous evidence sources.
  - Skills: `[]` — no extra skill required.
  - Omitted: `['momus']` — high-accuracy review is a later user option, not part of base execution.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: Final Verification | Blocked By: 1, 2, 3, 4, 5

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `deep_research/NG/deep-research-1:312-320` — normalized equal weighting and null-submission penalty.
  - Pattern: `deep_research/NG/deep-research-1:373-406` — public repository and endgame operational pressure.
  - Pattern: `deep_research/NG/plan_2:1-8` — explicit detection metric and high-leverage optimization levers.
  - Pattern: `deep_research/astar-island/plan_1:1` — limited official specification context; emphasize uncertainty.
  - Pattern: `deep_research/astar-island/plan_2:1` — probabilistic forecasting frame.
  - Pattern: `deep_research/tripx/README.md` — mixed provenance and canonical-usage rules from Task 4.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `deep_research/README.md` contains all required top-level sections.
  - [ ] `## Winning Strategy` explicitly mentions normalized/equal weighting and a no-zero-submission policy.
  - [ ] `## Open Risks` includes at least three source-grounded risks.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Combined strategy includes required sections and score rationale
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/README.md').read_text()
for needle in ['## Canonical Sources', '## Folder Summaries', '## Winning Strategy', '## Combined Implementation TODOs', '## Open Risks']:
    assert needle in text, needle
lower = text.lower()
assert 'normalized' in lower or '33.33' in text or 'equally weighted' in lower
assert 'null submission' in lower or 'zero submission' in lower or 'avoid a zero' in lower
PY
    Expected: command exits 0 and the winning strategy is explicitly tied to the competition score model.
    Evidence: .sisyphus/evidence/task-6-winning-strategy.txt

  Scenario: Open risks remain source-grounded rather than speculative
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('deep_research/README.md').read_text().lower()
risk_markers = ['missing api', 'not specified', 'duplicate', 'public repository', 'mixed provenance', 'unknown']
assert sum(marker in text for marker in risk_markers) >= 3
PY
    Expected: command exits 0 and the risk section reflects documented uncertainties and constraints.
    Evidence: .sisyphus/evidence/task-6-winning-strategy-error.txt
  ```

  **Commit**: YES | Message: `synth(deep-research): add combined winning strategy` | Files: `deep_research/README.md`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- `docs(deep-research): create canonical source map and README contract`
- `docs(astar-island): add source-grounded summary and implementation todos`
- `docs(NG): add forecasting/object-detection summary and implementation todos`
- `docs(tripx): document mixed provenance and implementation todos`
- `synth(deep-research): add combined winning strategy and prioritized todo matrix`

## Success Criteria
- All four markdown files exist with required sections.
- Each folder README contains only source-grounded claims and explicitly flags unknowns.
- `tripx` is documented without double-counting duplicated strategy content.
- The combined strategy explains score-maximizing prioritization using the repo’s normalized equal-weight competition framing.
