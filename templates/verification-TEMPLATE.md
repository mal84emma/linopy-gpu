# cuOpt support — verification plan

*Written by: {{agent id}} · Phase 3 · **FROZEN at Gate 3** · sha256 recorded in `logs/00-lead-journal.md` and `logs/50-review.md`*

<!-- HOW TO USE (delete this comment block when filling in):
     Copy to /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md at Phase 3. This file is BOTH the executable
     checklist Phase 5 runs AND the frozen record of the agreed thresholds. After Gate 3 it is
     edited ONLY by returning to that gate.
     Drift is detected by CONTENT HASH, not by git: dev-scripts/ is gitignored, so this file is
     untracked and `git diff` on it is always empty. At Gate 3 the lead runs
     `sha256sum` on it, records the digest in TWO logs, and `chmod 444`s it; the Phase 5
     reviewer recomputes and compares against both.
     EVERY threshold is a NUMBER, never an adjective. "duals correct" is not a threshold;
     "rtol=1e-6 vs HiGHS" is. Write "None" rather than deleting a section. -->

## Checks

<!-- Tier: V0 = structural/lint (mechanical), V1 = differential vs a LIVE oracle,
     V2 = end-to-end on linopy's own models. Oracle for V1 is the fully-qualified solver
     called live in the same process on the same model — never a copied-in expected value. -->

| Check ID | Tier | Proves (req ID) | Oracle | Threshold (a number) | Rationale |
|---|---|---|---|---|---|
| C-V1 | V1 | C-R… | `model.solve("highs")` on the same model, same process | `rtol=…` | … |

## Sign-convention matrix

<!-- The highest-risk area. One row per combination; all must be covered. -->

| Sense | Constraint | Dual sign expected | Reduced-cost sign expected | Test (path::name) |
|---|---|---|---|---|
| min | `<=` | … | … | … |
| min | `>=` | … | … | … |
| min | `==` | … | … | … |
| max | `<=` | … | … | … |
| max | `>=` | … | … | … |
| max | `==` | … | … | … |

## Status mapping

| cuOpt termination status | `TerminationCondition` | How it is forced in the test | Test (path::name) |
|---|---|---|---|

## Nondeterminism

<!-- Any cuOpt result that is not bit-reproducible run to run (PDLP tolerances, concurrent
     solver racing dual simplex vs barrier vs PDLP — the winner can vary). Name it and state
     the statistical criterion its comparison moves to. "None" if none — absence must be explicit. -->
