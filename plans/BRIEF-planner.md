# Phase 2 planner brief (identical for all five planners; only your plan number k differs)

You are working on adding NVIDIA cuOpt support to linopy at `/home/azureuser/linopy-gpu`.
Read `/home/azureuser/linopy-gpu/dev-scripts/cuopt/PROJECT_LEAD.md` (the charter) and `/home/azureuser/linopy-gpu/AGENTS.md` first, and obey both.
**Use absolute paths for everything under `dev-scripts/` — it is untracked, so those paths do not exist relative to a worktree checkout.** Templates: `/home/azureuser/linopy-gpu/dev-scripts/cuopt/templates/`. Logs: `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/`.
Treat §4 of the charter as verified fact — do not re-derive it; contradict it only with reproducible evidence.
Your model of correctness is the **current state of `master`**, not PR #516.
Do not escalate to the human — report to the project lead agent.
Never claim a test passes without pasting its output.
Scope: **Phase 2 — write ONE independent implementation plan.** Do nothing outside it.

Also read, in this order:
1. The "Rules for later agents" block at the top of `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/30-decisions.md`.
2. `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/10-exploration.md` — the Phase 1 empirical findings. This is your single source of empirical truth about cuOpt; do not re-run experiments.
3. The charter's §5 integration map and §5.3 requirement IDs, §6.4 work-package rules, §6.4a commit discipline, §4.5 QP sequencing, §4.6 installability constraints.
4. The relevant linopy source (read-only): `linopy/solvers.py` (esp. `class cuPDLPx` at ~:4156, `class Highs` at ~:1544, `Solver` base, `SolverFeature`, `SolverName`, `_SOLVER_PROBE_ORDER`), `linopy/io.py`, `linopy/model.py`, `linopy/constants.py` (`TerminationCondition`), `linopy/solver_capabilities.py`, `test/conftest.py`, `test/test_optimization.py`, `test/test_solvers.py`, `pyproject.toml`, and `gh pr diff 516 --repo PyPSA/linopy` for shape (not content).

> Tolerances and thresholds are **numbers**, proposed in the verification plan (`logs/15-verification.md`) and **frozen at Gate 3**. If any check misses its threshold, the responsible agent **reports the miss verbatim — with the actual numbers — and stops on that item.** It may not widen a tolerance, re-run until a favourable result appears, downgrade the check to a weaker form, skip the test, or explain the discrepancy away. Changing a threshold means returning to Gate 3.

> **Do not read `/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/`.** Do not list it, do not open another plan, do not ask another
> agent what it proposed. Your plan must be arrived at independently — four other agents are
> writing theirs in parallel and the *disagreement between them* is a deliberate signal the
> selector depends on. A plan contaminated by a sibling destroys that signal while looking
> like agreement.
(The one permitted exception to "do not read plans/": THIS brief file, which the lead placed there and which contains no plan content. You may re-read this brief; open nothing else in that directory.)

## Your deliverable

Write `/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/plan-<k>.md` — your k is given in your dispatch message. Write ONLY that file; touch no other file under `plans/`, no tracked file, no log file (planners do not write logs). Do not run code; this is a reading-and-design task (the empirics are all in 10-exploration.md).

The plan MUST contain, as clearly headed sections:

1. **Entry-point decision** — which cuOpt entry point(s) linopy calls and why, taking a position on the Phase 1 Q0 recommendation (agree or diverge, with reasons). Include the `io_api` surface you will declare (`accepted_io_apis`).
2. **Full file-by-file change set** — every file touched, what changes, anchored to current master line numbers where useful. Cover every row of the charter §5 map.
3. **Feature-flag set** with justification for each `SolverFeature` (and each one deliberately NOT declared).
4. **Status-mapping table** — every (problem_category, status) pair → `TerminationCondition`, consistent with exploration item 1.
5. **Design decisions on the hard findings** — you must take an explicit, justified position on each of these Phase 1 findings:
   a. The presolve+maximize negated-duals cuOpt bug and mitigation M2 (always-minimize).
   b. The Concurrent-method repeat-solve segfault (default method choice, or other mitigation).
   c. The `is_available()` fork: import probe advertises cuOpt on GPU-less machines (cuPDLPx/I5 precedent) vs §4.6 constraint 3 (honest degradation). Say exactly what `is_available()` and `model.solve("cuopt")` do on a GPU-less machine.
   d. QP: the 0.5*M.Q convention, MIQP guard, non-PSD handling.
   e. Reduced costs: linopy has no reduced-cost surface (grep confirmed) — what, if anything, do you do?
   f. Solver options: bool→int coercion, unknown-name errors, `log_file`/`log_to_console` handling.
   g. Interruptibility: adopt the HiGHS worker-thread pattern or not.
   h. Warm start: expose or defer, given the three-setting requirement and the CUDA-poisoning bug in set_initial_primal_solution.
   i. The pyproject extra: name, contents, marker (Phase 1 item 13 has a verified candidate).
6. **Requirement-ID map** — a table mapping EACH of C-R1…C-R17 (charter §5.3: the §5 table rows, numbered top to bottom) to the concrete work that satisfies it. If your plan implies new requirements, propose C-R18+ entries.
7. **Work-package breakdown** — per charter §6.4 rules: solvers.py core first and alone; then disjoint parallel packages; QP last as its own package; packaging as a first-class package; no two packages sharing a file concurrently. For each WP: files, requirement IDs, concrete acceptance criteria, and which of the ~6 commits (§6.4a) it lands in.
8. **Test plan** — where each test lives, the V0/V1/V2 tier it belongs to (§6.3a), the six-cell sign matrix coverage, the presolve-solvable max-model V1 case (exploration item 2 says the matrix alone misses the negated-duals bug), GPU gating via conftest, and proposed NUMERIC tolerances with rationale (note exploration item 7: default 1e-4 PDLP tolerance moved an objective by 5.5e-3 relative — your tolerances must be defensible).
9. **Docs plan** — every surface in charter §6.6b's table, with what each will say.
10. **Explicit non-goals** — so scope creep is detectable later.
11. **Risks** — what you think is most likely to go wrong with YOUR plan specifically.

Keep the plan concrete enough that a coding agent can implement a work package from it without re-deriving decisions, and keep the total diff proportionate (charter §9 yardstick: PR #516 was +771/−62 over 15 files; ~1.5× is the watermark).

In your final report back to the lead: your plan number, its path, a 10-line summary of your key positions on items 5a–5i, and **the model you are running on**.

Do not summarise away failures — a red test reported honestly is acceptable work, a hidden one is not.
