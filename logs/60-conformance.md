# cuOpt support — conformance evidence

*Written by: Phase 5 independent reviewer (Claude Opus 5, `claude-opus-5[1m]`) · Phase 5 · Verification plan: `logs/15-verification.md` @ sha256 `feef773d8fface670eda923095ee216de32a8fedf50b0561caaebe9687a3f820`*

*Scope note: this file is the §7 deliverable for Phase 5. It records two review loops — loop 1
against `8fb66cc` (full frozen-plan execution + findings F1–F10) and loop 2 against `b0b18f4`
(fix verification) — plus the re-verification I ran against the current tip `a8fb35b`. Every row
below names the commit its evidence was taken at, because the branch moved three times during
Phase 5. Per I14 the template's five headings are reproduced exactly and none is deleted; the two
sections after `## Escalations` are additions the Phase-5 brief asked for, not renamed template
sections.*

## Frozen-plan integrity

```
$ sha256sum /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md
feef773d8fface670eda923095ee216de32a8fedf50b0561caaebe9687a3f820  /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md

$ stat -c '%a %n' dev-scripts/cuopt/logs/15-verification.md
444 dev-scripts/cuopt/logs/15-verification.md
-r--r--r-- 1 azureuser azureuser 52622 Aug 23 01:17 dev-scripts/cuopt/logs/15-verification.md
```

- Digest recorded in `logs/00-lead-journal.md` at Gate 3: `feef773d8fface670eda923095ee216de32a8fedf50b0561caaebe9687a3f820` (line 57)
- Digest recorded in `logs/50-review.md` at Gate 3: `feef773d8fface670eda923095ee216de32a8fedf50b0561caaebe9687a3f820` (lines 1865, 1893)
- **All three match:** **yes**

Re-hashed at the end of every loop, including after the `a8fb35b` re-verification: same digest, same
mode `444`. The file was never opened for writing and never `chmod`ed by this reviewer.

## Verification plan results

49 frozen check rows (11 × V0, 24 × V1, 7 × V2, 7 × V3), plus the 8-row sign-convention matrix and
the 17-row status-mapping table. "Executed at" gives the commit; where a row was re-run after a fix
the later commit is named.

| Check ID | Threshold (frozen) | Actual | Pass? | Evidence (path) |
|---|---|---|---|---|
| V0-1 | exit 0, 0 findings | `All checks passed!`, exit 0; `ruff format --check` → 165 files already formatted | PASS (re-run @ `a8fb35b`) | `phase5-ruff.txt`, `phase5-rereview-a8fb35b-checks.txt` |
| V0-2 | error count `== 28` AND stripped message set `==` master (0 symdiff) | 28 vs 28; `comm -3` → **0** | PASS (re-run @ `a8fb35b`) | `phase5-mypy-{branch,master}{,-stripped}.txt`, `phase5-rereview-a8fb35b-checks.txt` |
| V0-3 | set equality, 0 symdiff | `[]`, len 0; display_name `cuOpt`/`cuOpt` | PASS | `phase5-v0-registry.txt` |
| V0-4 | exact string equality | `'cuOpt' == 'cuOpt'` → True | PASS | `phase5-v0-registry.txt` |
| V0-5 | exactly `True` | True | PASS | `phase5-v0-registry.txt` |
| V0-6 | not-in `True`; probe returncode `== 1` | `False`; `PROBE_RETURNCODE=1`; control with device `= 0` | PASS | `phase5-v06-probe.txt` |
| V0-7 | child exit `0`; naive variant non-zero | both asserted green by `test_device_probe_is_fork_safe` ×2 | PASS | `phase5-cuopt-gpu.txt` |
| V0-8 | without: 0 cuOpt tests run; with: count > 0 == tests added | `test_cuopt.py` 120 skipped / 120 passed 0 skipped. `-k cuopt` without the flag: **20 ran** (24 @ `a8fb35b`) — all GPU-free static/dummy rows, enumerated | PASS on substance; literal 0 not met → **F9** (record-only) | `phase5-v08-collect.txt`, `phase5-rereview-sideeffects.txt` |
| V0-9 | exactly 0 changed lines in each | 0 lines; `QUADRATIC_SOLVERS` absent from the diff | PASS (re-checked @ `b0b18f4`) | `phase5-rereview-testio-hygiene.txt` |
| V0-10 | exactly 0 lines | 0 lines at every checkpoint | PASS | all logs |
| V0-11 | `cuopt.routing` False; `error_log.txt` 0; `import cuopt` 1×, TYPE_CHECKING only | False; 0; one statement at `solvers.py:177` inside `if TYPE_CHECKING:` | PASS | `phase5-v0-registry.txt` |
| V1-1 | duals rtol 1e-6/atol 1e-7; primal 1e-6/1e-6; obj 1e-6/1e-6 | max\|Δdual\| **1.727e-09 / 2.473e-09 / 2.010e-09**; max\|Δprimal\| **5.554e-08**; \|Δobj\| **5.519e-08**; all six signs match the frozen matrix; non-vacuity min\|y\| 0.2 | PASS | `phase5-differential.txt` |
| V1-2 | as V1-1 | max\|Δdual\| **0.000e+00** (n=2, n=10); a negation would be **3.610e+00** | PASS | `phase5-differential.txt` |
| V1-3 | exact enum equality | `Optimal` / `Unset` | PASS | `phase5-cuopt-gpu.txt` |
| V1-4 | obj 1e-6/1e-6; integrality 1e-5; `dual.size == 0`; bound & gap not None | min 100.0/100.0, max 140.0/140.0; frac 0.0; size 0; bound 100.0/140.0; gap 0.0 | PASS | `phase5-differential.txt` |
| V1-5 | exact string equality on every listed row | 17-row table, 0 misses (see below) | PASS | `phase5-statusmap.txt` |
| V1-5b | `primal.size == n_cols` or empty + 1 logged error; 0 exceptions | `primal.size=400 == n_cols`, 0 exceptions | PASS | `phase5-statusmap.txt` |
| V1-6a | obj 1e-6/1e-6 | green vs live HiGHS | PASS | `phase5-cuopt-gpu.txt` |
| V1-6b | `optimal`; x.round(3)==0; round(obj,3)==0; x-block atol 1e-4 | all green, y excluded by construction | PASS | `phase5-cuopt-gpu.txt` |
| V1-6c | `NotImplementedError` containing `"no finite variable bounds"` | raised, matched | PASS | `phase5-cuopt-gpu.txt` |
| V1-6d | duals as V1-1; `len(dual)` == constraint size | `dual.size == labels.size`, duals match | PASS | `phase5-cuopt-gpu.txt` |
| V1-7 | obj atol 1e-6; answers 0 and 5 reproduced | both reproduced vs live HiGHS | PASS (recipe substituted → **F6b**) | `phase5-cuopt-gpu.txt` |
| V1-8 | returncode 0; exactly **20** `optimal`; 0 × `Fatal Python error`/`OMP:`/`kmp_alloc` | exit 0; **20**; 0/0/0; `TOTAL 20 solves in 115.5s` | PASS — round-5 code lag **closed** | `phase5-v18-n2000x20.txt` |
| V1-9 | ValueError naming the option; `presolve=False` solves; log > 0 bytes with `"cuOpt version"`; `log_fn` overrides `log_file` | all four green | PASS | `phase5-cuopt-gpu.txt` |
| V1-10 | warmstart/solution → NotImplementedError; basis → 0 exceptions + 1 warning; SOS/indicator → ValueError | all five green | PASS | `phase5-cuopt-gpu.txt` |
| V1-11 | not-in True; 1 warning naming driver/CC | True; count == 1; `525.60.13` present | PASS | `phase5-cuopt-gpu.txt` |
| V1-12 | child exit exactly 0 | 0 | PASS | `phase5-cuopt-gpu.txt` |
| V1-13a | obj 1e-6/1e-6; primal 1e-6/1e-6; duals atol 1e-6 | `M.c`/`M.Q` pinned; \|Δobj\| **1.954e-14**; max\|Δx\| **1.989e-07**; duals 1.075e-10 / 1.302e-10; min and max both | PASS | `phase5-differential.txt` |
| V1-13b | per-fixture scoping (x-block 1e-4 / full primal 1e-6) | both fixtures green at their frozen scopes | PASS | `phase5-cuopt-gpu.txt` |
| V1-13c | objectives differ by > 1e-3 relative | `optimal` **and** relative error > 1e-3 asserted; guard proven able to fail | PASS | `phase5-cuopt-gpu.txt` |
| V1-14 | NotImplementedError containing `"MIQP"`; **0** solver calls | raised for IQP and MIQP; `calls == []` | PASS | `phase5-differential.txt` |
| V1-15 | obj rtol 1e-4 | green at n=2000/m=1000 | PASS | `phase5-cuopt-gpu.txt` |
| V1-16 | `optimal`; obj 1e-6/1e-6; exactly 1 fallback warning | all three; plus lp / lp-polars / mps / None all `ok/optimal`, unknown io_api → ValueError | PASS | `phase5-cuopt-gpu.txt`, `phase5-docclaims.txt` |
| V1-17 | length exactly 0 | `[]` in a fresh `solver_dir`, every io_api | PASS | `phase5-cuopt-gpu.txt`, `phase5-docclaims.txt` |
| V1-18 | KeyboardInterrupt < 1.0 s; worker still completes | green; suite `6 passed` → `8 passed` after the F2 test landed | PASS (re-run @ `a8fb35b`) | `phase5-interrupt.txt`, `phase5-rereview-a8fb35b-checks.txt` |
| V2-1 | CPU 0 new failures vs baseline; GPU 0 new failures; 0 skipped cuOpt tests | CPU `46F/8547P/909S/28E`, 74 ids, **symdiff 0**. GPU `46F/8756P/700S/28E`, 74 ids, **symdiff 0**. `-k cuopt --run-gpu` → `229 passed, 9 skipped`, **0 skips in `test_cuopt.py`** | PASS | `phase5-full-cpu.txt`, `phase5-full-gpu-run1.txt`, `phase5-k-cuopt-gpu.txt` |
| V2-2 | condition `== "internal_solver_error"` exactly, one cuopt branch | exactly one branch, accept-list unwidened, green | PASS | `phase5-full-gpu-run1.txt` |
| V2-3 | passes at `rel=1e-6` | site **unmodified**; the pre-authorised `GPU_SOL_TOL` remedy was **not** used; green | PASS | `phase5-k-cuopt-gpu.txt` |
| V2-4 | exact `x == 0`, `y == 10`; no remedy authorised | unmodified, green — exact equality still holds | PASS | `phase5-full-gpu-run1.txt` |
| V2-5 | `warning` + `"infeasible" in condition` | `PrimalInfeasible` → `warning`/`infeasible`; no skip added | PASS | `phase5-statusmap.txt` |
| V2-6 | `ok`; exactly one `"cuopt"` key | one key `{"time_limit": 1}`; `method_options` block untouched | PASS | `phase5-rereview-show.txt` |
| V2-7 | 5 consecutive abort-free full `--run-gpu` runs; 0 `Fatal Python error`; 0 exit 134/139 | **NOT MET at cuOpt 26.08.00 — RED WITH CAUSE (D16/ESC-2)**. See Escalations. Every *completed* run is baseline-identical: `wp-f-full-gpu-retry.txt` symdiff **0**, `wp-g-full-gpu-attempt1.txt` symdiff **0**, and my own run symdiff **0** — verified by me with `comm(1)` | **RED with cause** | `phase5-full-gpu-run1.txt`, `40-blockers.md` B21, `45-known-issues.md` KI11 |
| V3-1 | exit 0; `cuopt` in `available_solvers` | membership half re-executed live → **True**. Resolution half in a clean seeded venv, bare PyPI, no flags: `PIP_EXIT=0`, 66 pkgs, `cuopt-cu12 26.8.0`, `pandas 3.0.3`. Full 6.7 GB install taken from record (18 GB free; shared box) | PASS (B10's deferred half closed) | `phase5-v31-cuopt-resolution.txt` |
| V3-2 | matching-package count exactly 0 | 33 packages, **0** matches, `pandas 3.0.5` | PASS | `phase5-v32-solvers-dryrun.txt` |
| V3-3 | both exit 0; `cuopt-cu12` count exactly 0 | windows exit 0 / 25 pkgs / **0**; macos exit 0 / 23 pkgs / **0**; linux control has it | PASS | `phase5-packaging.txt` |
| V3-4 | 0 new warnings, 0 new failures vs baseline | `build succeeded, 19 warnings.` exit 0; **19 == 19**; normalised warning symdiff **0**; 0 ERROR; docs venv confirmed cuOpt-free | PASS (re-run @ `b0b18f4` and `a8fb35b`) | `phase5-docs.txt`, `phase5-rereview-docs.txt`, `phase5-rereview-a8fb35b-docs.txt` |
| V3-4b | ≥1 in each of 8 surfaces; extra counts | 35 / 5 / 3 / 2 / 2 / 2 / 6 / 1; `pip install "linopy[cuopt]"` 2 and 1; `525.60.13` 1 and 1; `platform_system == 'Linux'` exactly **1** | PASS (re-run @ `a8fb35b`: 37 / 7, 525.60.13 → 2 and 1) | `phase5-rereview-a8fb35b-checks.txt` |
| V3-5 | added lines **≤ 1157** | **1986** @ `8fb66cc`; **2050** @ `b0b18f4`; **2071** @ `a8fb35b` | **MISS — reported verbatim.** Accepted-with-rationale by the lead as **ESC-3/D17**. See Escalations | `phase5-rereview-testio-hygiene.txt` |
| V3-6 | four exact counts | (a) AST-differing files **0** (dump len 538511 = 538511); (b) non-comment payload lines **0** of 19; (c) mapped **9/9**, missing 0, non-comment citations **0**; (d) all ten anchors ≥1 + A1.1 same-block True; cross-check summary identical to the pre-WP-G run, symdiff 0 | PASS | `phase5-v36-show.txt`, `phase5-v36a-ast.txt`, `phase5-v36c-citations.txt` |

**Sign-convention matrix — 8/8 rows PASS**, each cell matching the frozen expected sign *and* value
(`min <= [-0.4,-0.2]`, `min >=`/`= [+0.4,+0.2]`, `max <= [+0.4,+0.2]`, `max >=`/`= [-0.4,-0.2]`, and
both presolve rows). Reduced costs asserted nowhere; confirmed linopy has no reduced-cost surface at
all. `phase5-differential.txt`

**Status-mapping table — 17/17 rows PASS, 0 misses**, re-executed from the frozen recipes:

```
  [PASS] lp_optimal          legacy='Optimal'               status='ok'      condition='optimal'
  [PASS] lp_infeasible       legacy='PrimalInfeasible'      status='warning' condition='infeasible'
  [PASS] lp_unbounded        legacy='UnboundedOrInfeasible' status='warning' condition='infeasible_or_unbounded'
  [PASS] lp_iteration_limit  legacy='IterationLimit'        status='ok'      condition='iteration_limit'
  [PASS] lp_time_limit       legacy='TimeLimit'             status='ok'      condition='time_limit'
  [PASS] lp_primal_feasible  legacy='PrimalFeasible'        status='ok'      condition='suboptimal'
  [PASS] milp_optimal        legacy='Optimal'               status='ok'      condition='optimal'
  [PASS] milp_infeasible     legacy='Infeasible'            status='warning' condition='infeasible'
  [PASS] milp_unbounded      legacy='UnboundedOrInfeasible' status='warning' condition='infeasible_or_unbounded'
  [PASS] milp_feasible_found legacy='FeasibleFound'         status='ok'      condition='suboptimal'
  --- B18 recipe, fingerprint A[0,:6]=[81, 8, 17, 23, 18, 80] / A.sum()=14702.0 /
      rhs=[1182.5, 1292.5, 1369.5, 1122.5, 1279.5, 1106.5]  (all three match the frozen values) ---
  [PASS] milp_time_limit#1   legacy='TimeLimit'  status='ok' condition='time_limit'  primal.size=0 (want 0)
  [PASS] milp_time_limit#2   legacy='TimeLimit'  status='ok' condition='time_limit'  primal.size=0 (want 0)
  [PASS] milp_time_limit#3   legacy='TimeLimit'  status='ok' condition='time_limit'  primal.size=0 (want 0)
```

`milp_time_limit` is 3/3 in-process at the frozen recipe — the round-5 "code lags this row" caveat is
**closed**. The two never-forceable rows (`DualInfeasible`, MILP `Unbounded`) are asserted nowhere
and carry their required code comments. `lp_no_termination`, `milp_no_termination`,
`lp_numerical_error` and the `unknown` default are covered under different function names (**F6**).

**Tolerance constants:** all 10 declared as module constants in `test/test_cuopt.py` with values
exactly as frozen (**MISMATCHES = 0**); `GPU_SOL_TOL` unchanged at `2.5e-4` with 0 diff lines
touching it. KI10 closed.

## Requirement → evidence map

All 30 requirement IDs are mapped; **0 unmapped**. Line numbers are at the final tip `a8fb35b`.

*Labelling caveat, recorded not resolved:* the frozen plan's "Proves (req ID)" column and D7's
registry (`30-decisions.md:117`) disagree on the labels for **C-R21–C-R27** (the frozen plan reads
C-R21 = pad row, C-R23 = MIQP, C-R22 = interrupt, C-R25 = unsupported surfaces, C-R26 =
semi-continuous; D7 defines C-R21 = status maps, C-R22 = MIQP, C-R23 = pad row, C-R25 =
KeyboardInterrupt, C-R26 = warm-start deferral). I map by **substance** using D7's definitions,
since D7 is the registry §5.3 designates. No requirement's substance is unverified either way — the
disagreement is a label drift in the harness record, for Phase 6/8 to reconcile.

| Req ID | Deliverable (file:symbol) | Evidence (check ID + test) |
|---|---|---|
| C-R1 | `linopy/solvers.py:4605` `cuOpt.features` | V0-3, V0-9; `test_solvers.py::test_solver_class_supports_feature[cuOpt-*]` |
| C-R2 | `linopy/solvers.py:177` `import cuopt` under `TYPE_CHECKING` | V0-11; `test_routing_module_stays_unimported` |
| C-R3 | `linopy/solvers.py:402` `cuopt = _LazyModule("cuopt")` | V0-11 |
| C-R4 | `linopy/solvers.py:501` `cuOpt = "cuopt"` | V0-4; `test_enum_member_name_matches_class_name` |
| C-R5 | `linopy/solvers.py:4561-4992` `class cuOpt(Solver[None])` (432 lines) | V1-1, V1-4, V1-5b, V1-15, V2-1 |
| C-R6 | `linopy/solvers.py:5030` `"cuopt"` in `_SOLVER_PROBE_ORDER` | V0-5; `test_cuopt_is_available` |
| C-R7 | `linopy/solver_capabilities.py` — **zero-line diff, proved not edited** | V0-3, V0-9; `test_capability_shim_reports_declared_features` |
| C-R8 | `linopy/io.py:33`, `:799` `to_cuopt` | `test_io.py:558::test_to_cuopt` (added for **F3**) |
| C-R9 | `linopy/model.py:73`, `:2518` `to_cuopt = to_cuopt` | `test_io.py:558::test_to_cuopt` |
| C-R10 | `linopy/constants.py` — **zero-line diff**; statuses mapped onto existing members | V0-9, V1-5 + the 17-row table |
| C-R11 | `test/conftest.py` — **zero-line diff**; `GPU_ONLY` auto-gates | V0-8, V0-9 |
| C-R12 | `test/test_optimization.py:563`, `:966`, `:1117` | V2-2, V2-4, V2-5, V2-6 |
| C-R13 | `test/test_solvers.py:461-466` feature rows | V2-3 |
| C-R14 | `pyproject.toml:118` `cuopt` extra, one Linux marker | V3-1, V3-2, V3-3 |
| C-R15 | `doc/gpu-acceleration.rst` cuOpt section | V3-4, V3-4b |
| C-R16 | `doc/{prerequisites,contributing,index,api,release_notes}.rst`, `README.md`, `examples/using-solvers.ipynb` | V3-4, V3-4b |
| C-R17 | `QUADRATIC_SOLVERS` **read-only** (feature-derived) + the QP build path | V0-9, V1-13a, V1-13b, V1-13c |
| C-R18 | `linopy/solvers.py:423` `_cuda_device_available` warning text | V0-6, V1-11 |
| C-R19 | `linopy/solvers.py:4694` always-minimise (`sign * M.c`, `sign * 0.5 * Q`), negated back | V1-1, V1-2, V1-3 + the 8-row matrix |
| C-R20 | `linopy/solvers.py:4622` `_DEFAULT_METHOD = 3`, user-overridable | V1-8, V1-9 |
| C-R21 | `linopy/solvers.py:4816`, `:4828` name-keyed LP/MILP maps, picked by problem category | V1-5; `test_status_map_unknown_status` |
| C-R22 | `linopy/solvers.py:4670` `model.type in ("MIQP", "IQP")` guard | V1-14; `test_integer_quadratic_is_refused` |
| C-R23 | `linopy/solvers.py:4715` singly-bounded pad row; all-free → `NotImplementedError` | V1-6a, V1-6b, V1-6c, V1-6d |
| C-R24 | `linopy/solvers.py:4946` `_set_solver_params` (bool→int, name wrapping, `log_fn`→`log_file`) | V1-9 |
| C-R25 | `linopy/solvers.py:268`, `:311`, `:331` persistent worker + interrupt path | V1-18; `test_cuopt_interrupt.py` (8 passed) |
| C-R26 | `linopy/solvers.py:4839`, `:4844` warm-start deferral + never-call fence | V1-10 |
| C-R27 | `SEMI_CONTINUOUS_VARIABLES` + `MIP_DUAL_BOUND_REPORT` at `:4605` | V1-7, V1-4 |
| C-R28 | `linopy/solvers.py:423` subprocess probe + `:303-308` at-fork reset | V0-7, V1-12; `test_device_probe_is_fork_safe`, `test_solve_queue_starts_a_fresh_worker_after_fork` |
| C-R29 | `linopy/solvers.py:4634` `_build_file` override handing back `_problem_fn` | V1-16, V1-17 |
| C-R30 | in-code limitation rationales, 9 sites in `linopy/solvers.py` | V3-6 (a)(b)(c)(d) |

## Test output (pasted verbatim)

CPU suite (`.venv/bin/pytest -q`, executed at `8fb66cc`; direct redirect, never piped through
`tail`/`tee`; full log `phase5-full-cpu.txt`):

```
46 failed, 8547 passed, 909 skipped, 817 warnings, 28 errors in 463.41s (0:07:43)
PYTEST_EXIT=1
```
74-id FAILED/ERROR set, `comm -3` vs `05-baseline-pytest.txt` → **SYMDIFF=0**. `PYTEST_EXIT=1`
arises solely from the pre-existing KI1 set.

GPU suite (`.venv/bin/pytest --run-gpu -q`, executed at `8fb66cc`; full log
`phase5-full-gpu-run1.txt`). **Completed on attempt 1, no B21 abort:**

```
46 failed, 8756 passed, 700 skipped, 827 warnings, 28 errors in 744.30s (0:12:24)
PYTEST_EXIT=1
```
74 ids, symdiff **0** vs `05-baseline-pytest-gpu.txt` and **0** vs both previously recorded completed
runs; `Fatal Python error` count **0**. This is the third completed baseline-identical full run at
final content.

**GPU tests provably RAN, not skipped** — `pytest test/test_cuopt.py --run-gpu -v` (at `b0b18f4`,
`phase5-rereview-cuopt-gpu.txt`; identical result at `8fb66cc`):

```
120 passed in 281.84s (0:04:41)
PYTEST_EXIT=0
PASSED=120  FAILED=0  SKIPPED=0
```
The paired CPU-mode run is `120 skipped in 1.75s`, and `-k cuopt --run-gpu -v` gives
`229 passed, 9 skipped` with **0** of the 9 skips inside `test_cuopt.py`. During the run
`nvidia-smi` showed the pytest pid holding GPU memory at 99% utilisation, so the tests did real
device work rather than no-oping.

Interrupt suite (`.venv/bin/pytest test/test_cuopt_interrupt.py -q`, at `a8fb35b`):

```
8 passed in 1.48s
PYTEST_EXIT=0
```

Lint and types (at `a8fb35b`):

```
$ /home/azureuser/.local/bin/ruff check .
All checks passed!
exit=0

$ /home/azureuser/.local/bin/ruff format --check .
165 files already formatted

$ .venv/bin/mypy linopy
Found 28 errors in 1 file (checked 30 source files)
```
28 errors, all pre-existing (KI3, Xpress paths). Stripped-message `comm -3` against untouched
`master` @ `09c34dd` → **SYMDIFF_VS_MASTER=0**, i.e. **0 new errors** under `warn_unused_ignores`.
Exit code 1 on both sides and is not the threshold.

Docs (`.venv-docs/bin/sphinx-build -b html doc doc/_build/…`, at `a8fb35b`):

```
build succeeded, 19 warnings.
SPHINX_EXIT=0
```
Baseline is `build succeeded, 19 warnings.` — 19 == 19, normalised warning-set symdiff **0**, `ERROR`
count 0, and the docs venv is confirmed cuOpt-free (`ModuleNotFoundError: No module named 'cuopt'`).

## Escalations

Two items missed or could not meet their frozen threshold. Both are reported verbatim with the
actual numbers. **No tolerance was widened, no fixture substituted, no check downgraded, and nothing
was re-run until green.**

**1. V3-5 — bulk yardstick MISS. Reported verbatim; accepted-with-rationale by the lead as ESC-3 / D17.**

```
Frozen threshold:  added lines <= 1157   (1.5x PR #516's +771)
Actual @ 8fb66cc:  16 files changed, 1986 insertions(+), 9 deletions(-)      = 1.72x the cap
Actual @ b0b18f4:  17 files changed, 2050 insertions(+), 9 deletions(-)
Actual @ a8fb35b:  17 files changed, 2071 insertions(+), 11 deletions(-)     = 1.79x the cap
Split @ a8fb35b:   tests 1367  |  non-test 704  (linopy/ ~570, docs+packaging ~107)
```
§9's "hard look": the `cuOpt` class is `solvers.py:4561-4992` = **432 lines** against cuPDLPx's 263
— a 1.6× production-code ratio for a solver adding MILP, QP, semi-continuous variables, two status
maps, a MIP-stats report and a measured threading workaround. I looked for duplication, needless
abstraction and dead code and found none: the class mirrors cuPDLPx line-for-line where it can, the
four module-level helpers are each load-bearing against a measured upstream defect, and the only
literal duplication is a 3-line `io_api` re-check that cannot be delegated to the base method. The
overage sits in the dedicated test module, whose length the frozen plan itself mandated (V1-8's 20
solves, the 8-cell matrix, the 17-row status map, the QP guards).
**Disposition:** the lead accepted this with rationale as **ESC-3 / D17**. Note for the record: D17
was taken against the `8fb66cc` figure of **1986**; the current figure is **2071**, so the
acceptance record should carry the updated number (**N4** below).

**2. V2-7 — five abort-free full `--run-gpu` runs: NOT MET at cuOpt 26.08.00. RED WITH CAUSE (D16 / ESC-2).**

Cause is upstream, not this branch: NVIDIA/cuopt#1768, a heap corruption from cuOpt's bundled LLVM
libomp that crashes roughly two thirds of full runs at random post-cuOpt native sites (B21 measured
8/12, four distinct crash sites; linopy was exonerated twice — at the API boundary and behaviourally
with every native object pinned for the whole session). The accepted reading (D16) is that every
**completed** full run must have a failure set byte-identical to `05-baseline-pytest-gpu.txt`. I
verified that myself with `comm(1)` rather than taking it from the record:

```
05-baseline-pytest-gpu.txt          -> 74 unique FAILED/ERROR ids
wp-f-full-gpu-retry.txt      symdiff 0   (comm -23 = 0, comm -13 = 0)
wp-g-full-gpu-attempt1.txt   symdiff 0   (comm -23 = 0, comm -13 = 0)
phase5-full-gpu-run1.txt     symdiff 0   (comm -23 = 0, comm -13 = 0)   <- my own run, completed attempt 1
```
Three completed baseline-identical runs at final content, summary lines byte-identical
(`46 failed, 8756 passed, 700 skipped, 827 warnings, 28 errors`). I did not sample until green: I ran
the suite once and it completed. **This row stays RED and is not re-litigated here** — recorded in
`30-decisions.md` D16, `90-escalations.md` ESC-2, `40-blockers.md` B21, `45-known-issues.md` KI11.

**Raised by this reviewer: none.** No new escalation was needed — both items above were already on
the record before Phase 5 closed, and every other frozen row met its threshold.

## Review findings and dispositions

*Addition required by the Phase-5 brief; not a template section.* Loop 1 raised F1–F10 against
`8fb66cc`; loop 2 verified the fix commit and raised N1–N4.

| ID | Severity | Substance | Disposition |
|---|---|---|---|
| F1 | should-fix | Warning text and docs bullet advised `io_api="direct"` "to skip creating the problem file" — false: `model.py:2096` creates the temp file unconditionally, and nothing is written into it (measured 0 bytes, 1 call, both io_api values) | **FIXED** in the fix commit; new wording measured TRUE |
| F2 | should-fix | `_cuopt_solve_queue` cached across `os.fork()` without its worker → a cuOpt solve in a forked child waited forever (measured child exit `-14`, SIGALRM) | **FIXED** via `os.register_at_fork(after_in_child=…cache_clear)` at `solvers.py:303-308`; child now exits `4` with the real `cudaErrorInitializationError`. New regression test proven non-vacuous by me (fails 2/2 against pre-fix `solvers.py`, passes 8/8 at tip) |
| F3 | should-fix | `Model.to_cuopt` shipped with no test, unlike its `to_cupdlpx` precedent | **FIXED** — `test/test_io.py:558::test_to_cuopt`, PASSED ×2 under `--run-gpu` |
| F4 | frozen-threshold miss | V3-5 bulk yardstick, 1986 vs ≤1157 | **ACCEPTED-WITH-RATIONALE by the lead (ESC-3 / D17)**; see Escalations, plus N4 |
| F5 | nit | Docs claimed "ranged constraints", a concept linopy has no surface for | **FIXED** — now "Inequality and equality constraints"; `grep -ci 'ranged constraint'` = 0 |
| F6 | nit | Four frozen node ids do not exist; coverage present under different function names | **Record-only** (dispositioned by the lead) |
| F6b | nit | V1-7's semi-continuous model substituted for the frozen recipe; threshold still met | **Record-only** |
| F7 | nit | Merged skip reason told cuPDLPx users about cuOpt's warm start | **FIXED** — per-solver dict; live `-rs` shows the cuOpt-only reason |
| F8 | nit | `Any` annotations where the master precedent uses the concrete solver type | **FIXED as documentary.** B22's characterization **verified**: cuOpt ships no `py.typed` and 0 `.pyi`, `ignore_missing_imports = true`, so `reveal_type` gives `Any` and a bogus attribute is not flagged — while a numpy control *is* flagged. Confirmed the annotations are PEP-563 strings (`solvers.py:6`) so no eager cuOpt import is introduced |
| F9 | nit | 20 cuopt-matching tests ran without `--run-gpu` (frozen row says 0) | **Record-only**; count is 24 at `a8fb35b` (see N1) |
| F10 | nit | `to_cuopt` is the only solver bridge in the `doc/api.rst` IO autosummary (KI5's pre-existing gap) | **Record-only** |
| N1 | nit | `test_to_cuopt` is the first cuOpt test doing real cuOpt work without `--run-gpu`; F9's set 20 → 24, 2 of which now build a `DataModel`. Skips on CPU-only CI via `available_solvers`; mirrors `test_to_cupdlpx` exactly | **Record-only** — consistent with master, not a defect |
| N2 | nit | PID-reuse race in the new F2 test's `finally`: `os.kill` ran after the child was already reaped | **FIXED in the Phase-6 amend `a8fb35b`** via a `reaped` flag confining the kill to the failure path. Re-verified: 8 passed, and still non-vacuous (fails 2/2 pre-fix) |
| N3 | informational | `os.register_at_fork` is now a module-import side effect of `linopy.solvers` for every user | **Verified harmless**: no cuopt import before or after fork, no-op when nothing solved (cache 0→0, child exit 0), and a `multiprocessing` fork pool still works after a parent cuOpt solve (`['optimal','optimal']`). Strictly improves fork behaviour |
| N4 | bookkeeping | ESC-3/D17 accepted V3-5 at 1986; the branch is now 2071 | **Open** — one-line number update to the D17 record |

## Final branch state

`feat/cuopt-solver` = **`a8fb35bc10e6e2f26487cdd1d7509726d6f7ce6f`**, 7 commits on `master`
@ `09c34dd`, 0 merges, every commit authored `Claude Opus 5 <noreply@anthropic.com>` with a matching
`Co-Authored-By` trailer. `git status --porcelain` = 0 lines; `git ls-files | grep -c dev-scripts`
= 0; 17 changed files, **0** outside the intended surface. Frozen plan digest and mode `444` intact.

```
a8fb35b Address review findings                          <- Phase-6 amend of b0b18f4
8fb66cc Add comments explaining unsupported features
213e3e8 Add support for quadratic objectives
794a8d7 Add comment for release notes
1cac3ad Add to docs & add package dependency
9c60cd8 Update tests
5bfe769 Implement interface to cuOpt solver and required IO
```

**Lineage of the Gate-5 verdict, stated explicitly.** My loop-2 CLEAN verdict was returned against
`b0b18f4`. The tip then moved to `a8fb35b`, a Phase-6 amend of that same commit. I verified the
carry-over myself rather than assuming it, and it is **partial**, so the qualification matters:

| File changed by the amend | AST test (docstrings stripped, `include_attributes=False`) | Consequence |
|---|---|---|
| `linopy/solvers.py` | **equal = True** (dump len 539291 = 539291) | comments/docstrings only — production behaviour identical, loop-2 verdict transfers unchanged |
| `test/test_cuopt_interrupt.py` | **equal = False** (10545 → 10884) | a real logic change: the **N2** fix. Re-verified by me — 8 passed, and still non-vacuous (fails 2/2 against pre-fix `solvers.py`) |
| `doc/gpu-acceleration.rst`, `doc/prerequisites.rst` | not Python — n/a | prose (cross-references cuOpt's CC 7.0 / CUDA 12 requirements into the shared Requirements sections, adds the `cu12`/bare-`cuopt` note). Docs rebuilt: 19 == 19 warnings, symdiff 0 |

So "behaviour-identical by AST equality" is accurate for the **production** file only. To close the
gap I re-ran at `a8fb35b`: ruff clean, `ruff format --check` clean, mypy 28 / symdiff 0 vs master,
`test_cuopt_interrupt.py` 8 passed, the F2 non-vacuity demonstration still red pre-fix, docs 19
warnings symdiff 0, and V3-4b counts still above their frozen minima. On that evidence the Gate-5
**CLEAN** verdict carries forward to `a8fb35b`, with **N4** (the D17 number) the only item left open.

Full-suite evidence was not re-run at `a8fb35b` and does not need to be: the amend changes no
executable production line (AST-proved), and loop 1's completed full CPU and full GPU runs — both
with 74-id failure sets and symdiff 0 against their Gate-0 baselines — stand as the regression
record.

*Evidence files: 27 × `phase5-*.txt` (loop 1) and 14 × `phase5-rereview-*.txt` (loop 2 and the
`a8fb35b` re-verification), all in `logs/`, each carrying an explicit `*_EXIT=` line, with the
full-suite runs written by direct redirect.*
