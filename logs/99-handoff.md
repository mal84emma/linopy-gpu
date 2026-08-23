# cuOpt support — handoff

*Run completed: 2026-08-23 · Branch: `feat/cuopt-solver` @ e7bef1d (post-run rewrites: extra renamed `cuopt`→`gpu` on user instruction, was a8fb35b) · Lead: Claude Fable 5 (project-lead session)*
*Harness record (§6.7): https://github.com/mal84emma/linopy-gpu/tree/harness/cuopt — the
charter, plans, review rounds and logs behind this run. Not part of the PR.*

## 1. Unreviewed autonomous decisions

| ID | Decision taken | Alternative rejected | Where it shows in the diff | Cost to reverse |
|---|---|---|---|---|
| D9 | WP-A+B landed as one slice = commit 1 verbatim; WP-C's packaging commit cherry-picked+amended by WP-E into commit 3 | Separate WP-A and WP-B commits; WP-C as its own commit | Commit arc shape (7 commits, §7.3 plan) | High — history rewrite |
| D11 (A2) | FINAL.md corrected post-freeze: `keep_files=True` **raises** `NotImplementedError` (measured), not "empty file remains" | Keep the frozen-but-wrong claim | `doc/gpu-acceleration.rst` limitation bullet; `test_solution_file_is_refused` | Low — docs/test wording |
| D12 | B17 fix (persistent worker thread) landed by amending commit 1 + rebasing the arc | New "fix" commit breaking the planned arc | `_cuopt_solve_queue`/`_run_cuopt_with_keyboard_interrupt` in commit 1 | High — history rewrite |
| D13 | B19 fix (disposal routed to worker) same amend-and-rebase mechanics | Separate fix commit | `cuOpt.close()` + `_dispose_on_cuopt_worker` in commit 1 | High — history rewrite |
| D15 | Round-5 test tightenings (V1-8 N=20, `milp_time_limit`) amended into commit 2 | Fifth "strengthen tests" commit outside the planned arc | `test/test_cuopt.py` in commit 2 | High — history rewrite |
| D16 (ESC-2) | B21 (upstream heap corruption): V2-7 left **red with cause**; Gate 4 GPU read as met-in-correctness/failed-in-stability; project proceeded | (a) Re-scope V2-7 post-measurement (exculpatory bar-editing); (b) stop the project | No diff — verification record only (`40-blockers.md` B21, KI11, `90-escalations.md` ESC-2) | None — a record, reversible by maintainer decision |
| D17 (ESC-3) | V3-5 bulk cap (≤1157 added lines) missed at +2071: accepted with rationale | (a) Delete frozen-mandated test coverage; (b) sixth Gate-3 round to raise the cap | Whole-branch size, concentrated in `test/test_cuopt.py` (1228 lines) | None — a record; trimming tests is §2-below territory |

## 2. Where this run is weakest

| # | Area | Why I am not confident | What would settle it |
|---|---|---|---|
| 1 | **B21 — cuOpt 26.08.00 destabilises the host process** (KI11) | Upstream, root-caused (glibc malloc-metadata corruption from cuOpt's bundled libomp; gdb evidence; linopy exonerated twice), ~2/3 of full `--run-gpu` runs abort. Every *completed* run is baseline-identical, but a user's long-lived process (dask, notebook, server) is exposed. Shipping ahead of the upstream fix is a product judgment this run could not make. | Maintainer decision: ship-with-warning vs wait for the NVIDIA fix (tracked at NVIDIA/cuopt#1768 + our evidence comment). |
| 2 | **Test bulk vs repo norms** (ESC-3) | `test/test_cuopt.py` (1228 lines) is nearly the size of the whole shared `test_optimization.py`. Every test maps to a frozen row or a measured defect, but the density of in-test fingerprints and subprocess harnesses is this harness's culture, not linopy's. | A maintainer read; a deliberate slimming pass (~20–30% is ceremony) if they want repo-norm density — knowing it deletes regression guards for real measured bugs. |
| 3 | **Single-machine evidence** | Everything GPU was measured on one T4 (CC 7.5, 16 GB, one driver version). The `method=0` crash boundary (~1300 vars), the B17 abort threshold (5–13 solves), tolerances like `CUOPT_PRIMAL_ATOL_DEGENERATE=1e-4` — all could shift on other hardware/driver combos. | A second GPU environment running `pytest test/test_cuopt.py --run-gpu`. |
| 4 | **Worker-thread lifetime edge cases** | The persistent daemon worker + at-fork cache-clear covers solve, interrupt, GC-disposal and fork-after-solve, each tested. But exotic embeddings (multiple interpreters, `spawn`-vs-`fork` mixes, interpreter shutdown ordering) are reasoned about, not all tested. | Adversarial review of `_cuopt_solve_queue` / `_dispose_on_cuopt_worker` / `cuOpt.close()` (solvers.py ~:260–330, :4970–4995). |
| 5 | **Frozen-plan naming divergences** (B20, F6/F6b) | Four status-map tests exist under different names than the frozen table's node ids, and WP-D substituted V1-7's semi-continuous recipe (thresholds met, discriminating answers reproduced — but it was an unauthorized substitution). A conformance re-run copying ids from the frozen file will hit `not found`. | Read `40-blockers.md` B20 + the Phase-5 report's F6/F6b before re-running any frozen row. |
| 6 | **QP on degenerate fixtures** | `quadratic_model`'s x-block agrees with HiGHS only to 5.6e-05 (both solvers are ~1e-4-accurate on that singular QP); the `y` block is underdetermined and deliberately never compared. Correct per analysis, but the loosest numeric corner of the branch. | The Nondeterminism table in `logs/15-verification.md` documents it; a reviewer sanity-check of that argument. |

## 3. What shipped

| Requirement ID | Delivered | Where |
|---|---|---|
| C-R1–C-R16 (core solver, LP+MILP, direct API, probe, status maps, duals/signs, options, interrupt) | Yes | `linopy/solvers.py` cuOpt class + worker helpers (commit 1) |
| C-R6/C-R7 (`to_cuopt` bridges) | Yes | `linopy/io.py`, `linopy/model.py` (commit 1); test in `test/test_io.py` (commit 7) |
| C-R17/C-R23 + QP rows of C-R19/C-R21 (QP, MIQP guard) | Yes | commit 5 |
| C-R18–C-R29 test halves | Yes | `test/test_cuopt.py`, `test/test_cuopt_interrupt.py`, shared-suite edits (commits 2, 5, 7) |
| C-R25 (packaging extra) | Yes | `pyproject.toml` (commit 3) |
| Docs (§9, all ten surfaces) | Yes | commits 3, 4, 5, 6, 7 |
| C-R30 (limitation rationales, A1/A1.1 — human-directed) | Yes | commit 6 (verified structurally, V3-6) |

## 4. What did not ship

None. All 30 requirement IDs delivered, QP included.

## 5. Verification summary

- Frozen verification plan: `logs/15-verification.md`, sha256 `feef773d8fface670eda923095ee216de32a8fedf50b0561caaebe9687a3f820` (mode 444; second copy of record in `logs/50-review.md`)
- Checks executed / passed: 49 rows — **47 PASS**, **1 red-with-cause** (V2-7, upstream B21, D16/ESC-2), **1 missed-and-accepted** (V3-5 bulk, +2071 vs ≤1157, D17/ESC-3) — detail in `logs/60-conformance.md` and the Phase-5 report
- Checks that missed thresholds, actual numbers: V2-7 (5 consecutive abort-free full GPU runs required; measured ~8/12 runs abort from upstream corruption; 3 completed runs at final content all baseline-identical); V3-5 (added lines 2071 vs cap 1157)
- Baseline comparison: CPU suite failure set byte-identical to master baseline (74 ids, symdiff 0); every completed GPU run likewise; docs build 19 = 19 warnings, warning-set identical

## 6. Escalations

| ID | Question | Trigger | How it was resolved | Still open? |
|---|---|---|---|---|
| ESC-1 | GPU-less machines: does `available_solvers` include cuOpt? | E1 (planner split) | Option A (probe in `is_available`) — charter §4.6 pre-decided it | No |
| ESC-2 | B21 makes V2-7/Gate-4-GPU unmeetable as written | E2/E5 blend | D16: row stays red with cause; correctness verified on completed runs; proceed | No (maintainer weighs ship-vs-wait) |
| ESC-3 | V3-5 bulk cap missed verbatim | E5-adjacent | D17: accepted with rationale; no coverage deleted, no cap edited | No |

## 7. Known issues not fixed

All in `logs/45-known-issues.md`: KI1 (46F/28E pre-existing baseline failures, test/remote), KI2 (cuPDLPx not installed in dev env), KI3 (28 pre-existing mypy errors, Xpress), KI4 (probe cost/fail-closed scenarios), KI5 (api.rst lists no other `to_*` solver bridges), KI6/KI7 (docs quirks), KI8 (+2 addenda — upstream OpenMP leak, #1768), KI9/KI10 (test tolerance provenance), **KI11 (cuOpt 26.08.00 heap corruption — the load-bearing one, see §2 row 1)**, KI12 (`Model.copy()` silently drops quadratic objectives — pre-existing master bug found by this run, worth its own upstream-linopy issue).

## 8. State of the tree

- Branch pushed: **yes** (`origin/feat/cuopt-solver` @ e7bef1d; origin only, never upstream) · Commits: **7** · Every author a real Claude model, author matching its Co-Authored-By trailer (I11): **yes** (all 7 `Claude Opus 5`) · Fable-authored commits: **none on this branch** (the harness branch's single commit is `Claude Fable 5` — the lead)
- Merge commits present: **none** — confirmed (checked at every merge and at Gate 6)
- `git log --stat` shows zero files under `dev-scripts/`: **confirmed** (+ both §6.6b leak checks empty against upstream/master, re-run at push time)
- `PR_BODY.md` drafted with the `<!-- HUMAN: write your intent here -->` placeholder: **confirmed** (`dev-scripts/cuopt/PR_BODY.md`)
- **PR not opened** (charter §2.1): **confirmed**

## 9. What the human should do next

1. Review §1 (the seven unreviewed decisions) and §2 row 1 — the B21 ship-vs-wait call is the one real product decision in this run.
2. Read `PR_BODY.md`, write your own intent over the placeholder (AGENTS.md rule 2), trim anything you disagree with.
3. Open the PR from `mal84emma/linopy-gpu:feat/cuopt-solver` → `PyPSA/linopy:master` when ready. If you want to link the harness branch from the PR, the note alert is yours to write (AGENTS.md rule 1).
4. ~~Consider filing the KI12 finding (`Model.copy()` drops quadratic objectives) as its own linopy issue~~ **Done post-run on your instruction: https://github.com/PyPSA/linopy/issues/903** (see KI12 in `45-known-issues.md` for the filing record).
5. Optional: a test-slimming pass on `test/test_cuopt.py` (§2 row 2) if you prefer repo-norm density over this run's regression-guard density.
