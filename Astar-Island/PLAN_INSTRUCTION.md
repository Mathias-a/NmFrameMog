# Instruction To Give The Agent

Read `PARALLEL_AVENUE.md` first, then read the local mission/context files it references.

Your task is to create a **deeper execution plan for this avenue only**. Do not implement code yet.

Write your output to `WORKTREE_PLAN.md` in this `Astar-Island` folder.

Your plan must include:

1. A short restatement of the mission and why this avenue matters.
2. A map of the current implementation with exact file references.
3. The main hypotheses worth testing.
4. A proposed change list grouped into 2-5 coherent experiments.
5. A benchmark protocol:
   - what to run,
   - on which fixtures,
   - what metrics to compare,
   - what counts as a win.
6. Risks, invariants, and things that must not be changed.
7. An ordered first-pass task list that an implementer can execute.

Hard requirements:

- Use evidence from the actual code and docs, not guesses.
- Call out interactions with the local digital twin and benchmark harness.
- Preserve score safety and determinism.
- Keep the simulator stable unless the avenue explicitly allows otherwise.
- Favor experiments that can be compared offline before using any real API budget.

The plan should be detailed enough that another agent can implement it without needing a second discovery pass.
