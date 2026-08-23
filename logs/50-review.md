# cuOpt support — plan review, round 1

*Reviewer: Phase 3 selector-reviewer (Opus 5, `claude-opus-5[1m]`) · 2026-08-22 · Reviewing: `/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/plan-1.md` … `plan-5.md` (all read in full from disk)*

Basis: charter §§4, 5, 5.3, 6.3/6.3a, 6.4/6.4a, 6.6b, 9; `AGENTS.md`; the "Rules for later agents"
block and **D6** in `logs/30-decisions.md`; `logs/10-exploration.md` in full; `logs/90-escalations.md`
(ESC-1); `logs/45-known-issues.md`; and read-only inspection of current `master` (`09c34dd`):
`linopy/solvers.py`, `linopy/model.py`, `linopy/io.py`, `linopy/matrices.py`, `linopy/objective.py`,
`linopy/constants.py`, `linopy/common.py`, `test/conftest.py`, `test/test_optimization.py`,
`test/test_solvers.py`, `test/test_available_solvers.py`, `test/test_semi_continuous.py`,
`doc/api.rst`, `pyproject.toml`, and the installed `cuopt` 26.08.00 package.

Where the five plans disagreed with each other on a checkable fact, I checked the fact rather than
the plans. Ten such checks were run read-only (no repo file was modified; scratch scripts live in
this session's scratchpad, not in the tree). Their pasted output is in
**"What all five missed"** below and is the evidence for findings B1–B9.

## Verdict

**REVISE**

Round 1 is the selection round, so there is no `FINAL.md` to approve yet. The verdict is:
**base plan = `plan-1.md`**, with the grafts, rejections and positions below. Sixteen findings must
be resolved in `FINAL.md`; nine of them (B1–B9) are things **all five plans got wrong or left
unverified**, and three (B1, B3, B5) would each have produced a red shared test or a silent
regression if built as written.

## Blocking findings

| # | Finding | Where (file:line) | Why it blocks | Suggested fix |
|---|---|---|---|---|
| **B1** | **The zero-nnz "free row" pad does not work for QP.** A padded row with both bounds infinite (or ±1e30) makes cuOpt return `NumericalError`/`obj=nan` on a box-only QP; a padded row carrying *any* finite bound returns `Optimal`. Verified 11 ways (evidence W1/W2 below). | plan-1 §2.1.5, plan-2 §2.1, plan-5 `_pad_free_row`; `test/test_optimization.py:927` | `test_quadratic_model_wo_constraint` removes `con0` and then asserts `condition == "optimal"` for every solver in `feasible_quadratic_solvers`. With a doubly-infinite pad row cuOpt returns `internal_solver_error` → **red test** the moment `QUADRATIC_OBJECTIVE` is declared. | Pad with **one redundant, singly-bounded row**: pick a column `j` with a finite `M.lb[j]` or `M.ub[j]` and append the row `1·x_j ∈ [M.lb[j], M.ub[j]]`. Verified `Optimal`, `obj=7.6e-15`, `x≈2.6e-08` → `round(3)==0` passes the existing assertions. If no variable has a finite bound, that residual case must be handled explicitly (state which: pad free for LP — verified `Optimal` — and raise a clear `NotImplementedError` for QP, or verify a third shape) and tested. |
| **B2** | **Raising `ValueError` on a zero-nnz constraint matrix breaks a shared test.** | plan-3 §2.1.6, plan-4 §2.1 (`_build_solver_model`) | Same test as B1: `feasible_quadratic_solvers` contains cuopt, so the `pytest.raises(ValueError)` branch is *not* taken and the `ValueError` fails the test. `M.A is None` is reachable from an ordinary model — verified (`A: None` for `quadratic_model` minus `con0`). | Adopt the padded-row approach of B1. Do **not** copy `cuPDLPx._build_solver_model`'s `raise ValueError("Model has no constraints…")` (`solvers.py:4256`): cuPDLPx is never exercised on this machine (KI2), so its behaviour here is untested precedent. |
| **B3** | **A bare `model.solve("cuopt")` (no `io_api`) must succeed, or existing shared tests go red.** `test/test_solvers.py:46` `test_solver_instance_attached_after_solve` and `:89`/`:286` call `simple_model.solve(solver)` with no `io_api`, parametrized over `licensed_solvers`, and assert `status.is_ok`. | plan-2 §1 / §2.1 (`_run_file` raises), plan-4 §1 (`_run_file` raises) | With `io_api=None`, `Solver._build` routes to `_build_file`; a `_run_file` that raises `NotImplementedError` turns those tests red under `--run-gpu`. The wart is invisible today only because cuPDLPx is not installed (KI2). It is also the DoD line "`model.solve("cuopt")` solves LP and MILP models correctly". | Override **`_build_file`** to build through the direct API (plan-3 §2.1.9 / plan-5 §1.3). Verified mechanically: `_build_direct` sets `self.io_api = "direct"`, and `Solver.solve` dispatches on `if self.io_api == "direct" or self.solver_model is not None` (`solvers.py:858-861`) → `_run_direct`. Leave `_run_file` unoverridden (base message is correct for the deprecated file entry point). |
| **B4** | **The `_build_file` interception leaks a temp problem file.** `Model.solve` calls `get_problem_file()` *before* building, and `NamedTemporaryFile(..., delete=False)` (`model.py:1886-1893`) creates the file on disk. `Model.solve` then does `if io_api != "direct": problem_fn = solver._problem_fn` (`model.py:2135`) and only unlinks what it gets back. | plan-3 §2.1.9, plan-5 §1.3 (neither sets `_problem_fn`) | An empty `linopy-problem-*.lp` accumulates in `solver_dir` on **every** bare solve. Silent, and exactly the kind of thing Phase 5's `git status`/stray-file check will not catch (it is outside the repo). | Either set `self._problem_fn = Path(build_kwargs["problem_fn"])` so linopy's own `finally` unlinks it, or unlink it in the override and leave `_problem_fn = None`. Pick one, and add a verification row: after a bare `model.solve("cuopt")`, `list(Path(m.solver_dir).glob("linopy-problem-*"))` is empty. |
| **B5** | **The proposed CUDA device probe poisons `fork()`.** Calling `cuda.bindings.runtime.cudaGetDeviceCount()` in a process makes a *subsequently forked* child fail every cuOpt solve with `InputValidationError: A CPU-memory problem requires remote execution…`. Verified (W4): without the probe the forked child solves (`Optimal 2.0`, exit 0); with it the child fails (exit status 768). | plan-1 §2.1.8 `_cuda_device_available`, plan-2 §5c, plan-4 §5c | `is_available()` runs on first access to `linopy.available_solvers` — i.e. during ordinary `import linopy; linopy.available_solvers`. Initialising CUDA there breaks cuOpt **and any other CUDA library** in every later-forked worker (dask/multiprocessing default to `fork` on Linux). That is a user-visible regression far worse than the problem D6 asks us to fix, and it is invisible in single-process tests. plan-5 §5c predicted this hazard for a `ctypes cuInit(0)` probe; it is now measured for the runtime API too. | Make the probe **out-of-process and cached**: `subprocess.run([sys.executable, "-c", "from cuda.bindings import runtime; e,c = runtime.cudaGetDeviceCount(); raise SystemExit(0 if (int(e)==0 and c) else 1)"])`, `functools.cache`d. Measured cost **0.28 s once**, correct in both directions (`returncode 0` with a GPU, `1` under `CUDA_VISIBLE_DEVICES=''`), and **fork-safe** (child solved, exit 0) — W5. A CUDA-free heuristic (`CUDA_VISIBLE_DEVICES` + `/dev/nvidia[0-9]*`, also verified fork-safe) is an acceptable alternative if the lead prefers no subprocess in library code; merely *importing* `cuda.bindings.runtime` is fork-safe, calling into it is not. |
| **B6** | **plan-4's probe raises on the GPU-less branch, and fails open.** `cudaGetDeviceCount()` returns `(cudaErrorNoDevice, None)` — `count` is `None`, not `0` (W3). plan-4's `return not err and count > 0` sits **outside** its `try`, so `None > 0` raises `TypeError` uncaught from `is_available()`. Its `except: return True` also advertises cuOpt when the probe cannot run, which contradicts D6. | plan-4 §5c | The one branch §4.6 c3 exists for is the branch that crashes. | Use `return not err and bool(count)` (None-safe) inside the guarded region, and **fail closed** with a `logger.warning` that names the driver/CC requirement. Fail-closed is safe here because the positive branch is pinned by a test (see B16) that goes red if the probe ever breaks on a working GPU. |
| **B7** | **D6 override.** plan-3 §5c and plan-5 §5c keep `is_available()` as a pure import probe and put the GPU check in `_license_probe`, so `available_solvers` still lists cuopt on a GPU-less host. | plan-3 §5c, plan-5 §5c, §6 (C-R20) | D6 (`logs/30-decisions.md`) is a lead ruling, binding on this phase, resolving ESC-1 in favour of charter §4.6 c3: on GPU-less machines `is_available()` is false and `available_solvers` omits cuopt. | `FINAL.md` takes plan-1/2/4's position with the mechanics of B5/B6. Keep plan-3/5's *reasoning* only as the recorded downside (the `_AvailableSolvers` docstring at `solvers.py:4464-4470` now has a solver-specific exception; the "not installed" message is inaccurate for an installed-but-GPU-less host). Note `test_available_solvers.py:39` stays green either way — it compares `is_available()` to membership, so both sides move together. |
| **B8** | **An empty MIP dual must not be scattered.** `_solution_from_labels(np.array([]), clabels, n_cons)` raises `IndexError: boolean index did not match indexed array` (verified, W6) — and `safe_get_solution` re-raises for an `ok` status (`solvers.py:1190-1191`), so the solve dies. | plan-2 §2.1 `_solve`, plan-3 §2.1.6 step 7, plan-4 §4 (flagged as AC-6, not resolved) | Every MILP solve would raise. `get_dual_solution()` genuinely raises for a MIP solution (verified: `AttributeError: Attribute get_dual_solution is not supported for milp solution`), so the empty-dual path is the normal MILP path, not an edge case. | Return the **unscattered** empty array for MIP (plan-1 §2.1.6; plan-5 guards with `if dual.size`). `Model.assign_result` guards with `if len(result.solution.dual):` (`model.py:2206`), so an empty dual is accepted and duals are simply not assigned. |
| **B9** | **Solver-option value validation happens at `Solve()`, not at `set_parameter()` — so plan-3's bool retry is dead code, and a blanket `bool → int` coercion is both safe and sufficient.** Verified (W7/W8): `set_parameter("presolve", False)` **succeeds**; `Solve()` then raises `ValueError: Parameter presolve value False is not an integer`. `log_to_console=0`, `crossover=1`, `infeasibility_detection=0` are all accepted at solve time. | plan-3 §5f item 2, plan-1 §2.1.7 (`_BOOL_PARAMETERS` allowlist), plan-2 §5f (`_coerce_option` via `get_parameter`) | plan-3's `except ValueError: set_parameter(key, int(value))` can never fire, so `presolve=False` still explodes — a silent no-op mitigation. plan-1's hand-maintained allowlist and plan-2's `get_parameter` type-sniffing are unnecessary complexity for a case where the simple rule is verified correct. | Coerce **every** Python `bool` option value with `int(v)` before `set_parameter`. Bool-typed parameters accept `0`/`1` (verified), int-typed ones require it. Keep the wrapped `ValueError` for an unknown *name* (that one **is** eager at `set_parameter` — exploration §7) and pass value errors through unwrapped (they already name parameter and value). |
| **B10** | **`test_quadratic_model_unbounded` will be red for cuopt, and its disposition must be frozen at Gate 3.** cuOpt returns `NumericalError` (`obj=nan`) on that fixture (verified, W1 case `e`), which every plan maps to `internal_solver_error`; the test accepts only `["unbounded","unknown","infeasible_or_unbounded"]` (`test_optimization.py:955-965`). | all five plans' §4 status maps; plan-1 risk 5, plan-2 §5d.4 and plan-5 risk 3 flagged it as unverified | A predicted red shared test with no pre-agreed disposition invites exactly the improvisation I12/§3.6 forbids (widening the accept-list, or quietly re-mapping `NumericalError → unknown` to make it pass). | Freeze the disposition in `logs/15-verification.md` **now**, with the measured status pasted: keep the honest mapping (`NumericalError → internal_solver_error`) and add a cuopt-specific branch in that one test citing the measured status — the same shape as the existing cuPDLPx/mindopt/gurobi branches in that file. Re-mapping `NumericalError → unknown` is rejected: `safe_get_solution`'s `unknown` branch parses the solution anyway and flips the status to `ok` (`solvers.py:1192-1198`), which would report a `nan`-objective solve as successful. |
| **B11** | **plan-3 refuses `MIP_DUAL_BOUND_REPORT` on the grounds that no dual-bound accessor was verified. It is verified now.** On the exact `milp_model` fixture: `milp_stats = {'mip_gap': 0.0, 'solution_bound': 100.0, …}` (W9). | plan-3 §3.2 | Charter §5 row 1 lists it as a candidate feature and 4/5 plans declare it; declining it on a factual premise that is false would drop a real capability. | Declare it (plan-1 §3 / plan-2 §3 / plan-4 §3 / plan-5 §3), negate `solution_bound` for `max` under M2, and keep the contingency: if it ever comes back `None`, report verbatim and drop the flag — never weaken `test_optimization.py:834`. |
| **B12** | **plan-5 declines `SEMI_CONTINUOUS_VARIABLES`, which makes linopy emit a false claim.** Without the flag, `Solver._validate_model` raises *"Solver cuopt does not support semi-continuous variables"* (`solvers.py:731-737`) for a model cuOpt demonstrably solves (exploration §10). | plan-5 §3, non-goal N2 | Exploration §10 is explicit that asserting impossibility here would be false; the DoD line is "unsupported features raise `NotImplementedError` with actionable messages", not "supported features raise". | Declare it (4/5 plans do), map `'B' → 'I'` and pass `'S'` through, and cover it with one differential test. Note `test/test_semi_continuous.py` has **no** solver parametrization (hard-coded gurobi/highs) and no GPU gating — verified — so declaring the flag brings **zero** automatic coverage; the test must live in the GPU-gated cuOpt module (plan-1 §8 is right about this). |
| **B13** | **`doc/api.rst` — the `to_cuopt` row is unsettled across plans (3 against, 1 for, 1 silent).** | plan-1 §9, plan-3 §9, plan-4 §9 (omit + justify); plan-5 §9 (add) | Charter §6.6b names `to_cuopt` explicitly. Gate 6 allows "satisfied **or** its absence justified", so either is legal — but leaving it to a later agent guarantees a Gate-6 argument. | Settle it now: **add both** `solvers.cuOpt` to the Implementations autosummary **and** `model.Model.to_cuopt` to the Model IO autosummary (`doc/api.rst:114`). The precedent objection is weaker than the plans thought: `to_netcdf` is bound the same way (`to_netcdf = to_netcdf`, `model.py:2507`) and *is* listed at `api.rst:115`, so `Model.to_cuopt` is proven safe for the build. The missing siblings (`to_gurobipy`, `to_highspy`, `to_cupdlpx`) are a pre-existing doc gap → `logs/45-known-issues.md`, not fixed here (I13/I4). |
| **B14** | **Shared assertions tighter than cuOpt's default tolerances are not in any test plan as an early check, and their disposition is not frozen.** `test_solvers.py:101-103` asserts `pytest.approx(3.3)` / `approx(-0.1)` / `approx(1.7)` (rel `1e-6`) parametrized over `licensed_solvers`; `test_optimization.py:717-718` asserts **exact** equality (`solution.x == 0`, `solution.y == 10`). | plan-5 §8 / risk 2 and 4 (only plan that saw them); plan-1 §8, plan-2 §8, plan-3 §8.5, plan-4 §8 | If these are discovered in Phase 5, the only quick fixes available are the ones I12 forbids. (Good news: I measured both and they currently pass — `test_model_with_inf` returns exactly `x=0`, `y=10`, W10 — but that is a measurement, not a guarantee, and `test_solvers.py`'s `-0.1` is 1e-7-tight.) | Put both in `logs/15-verification.md` as **explicit named rows run in WP-A/WP-D**, with the disposition pre-decided *before* the freeze: a miss is reported verbatim, and the only permitted remedy is the repo's own `GPU_SOL_TOL = 2.5e-4` parameterisation (the move #516's own commit 6 made in `test_optimization.py`) applied to *that* assertion — decided at Gate 3, never by a fix agent. |
| **B15** | **plan-3's new `method_options` entry (`{"method": 1, …}`) is likely red.** `test_solver_method_options` asserts `np.isclose(objective, 3.3)` (default `rtol=1e-5`); exploration §2c measured PDLP drifting the objective of a 2-variable model to `2.80013079` vs `2.8` — `4.7e-5` relative, ~5× outside that. | plan-3 §2.7 | It would introduce a red test to gain coverage that the cuOpt-specific suite already provides. | Do not add a `method=1` row there. If a non-default-method row is wanted, use `{"method": 2}` (DualSimplex, measured exact on small models) — or omit the entry entirely (the dict is keyed by `if solver in method_options`, so absence is legal, unlike `time_limit_option`, which is indexed unconditionally at `:568` and **must** gain a `"cuopt"` key or every parametrized cell raises `KeyError`). |
| **B16** | **No plan pins the *positive* branch of the availability probe as a test.** With a fail-closed probe (B6), any future breakage of the probe silently removes cuOpt from `available_solvers`, which reads exactly like "cuOpt isn't installed" and skips every GPU test — a green suite hiding a dead feature (charter §9, "a harness check that cannot fire"). | plan-1 risk 1 names the hazard; no plan makes it a check | Fail-closed is only defensible with the positive branch pinned. | Add a V0 row: on this machine `"cuopt" in linopy.available_solvers` must be `True` (already implied by `test_available_solvers.py:39` only if the probe agrees with itself — that test cannot catch a uniformly-false probe). Plus the negative branch under `CUDA_VISIBLE_DEVICES=''` in a subprocess, plus the **fork-safety** row from B5. All three are checks that can fire. |

## Non-blocking observations

1. **`_license_probe` should not solve on the GPU.** plan-1 §2.1.2, plan-3 §5c and plan-5 §2.8 all
   run a real 1×1 `Solve` in the probe. `test_optimization.py` and `test_solvers.py` parametrize on
   `licensed_solvers` at import time, so a plain CPU-only `pytest` run would create a CUDA context
   and spend ~0.3 s before collecting anything (plan-1 risk 8 and plan-3 risk 3 both flag it
   honestly). Under D6 the device check already lives in `is_available()`, so the solve is
   redundant. Prefer plan-2 §2.1's cheap constructor probe (`cuopt.DataModel();
   cuopt.SolverSettings()`) — it is not vacuous, because `_has_module` uses `importlib.util.find_spec`
   and therefore never executes `cuopt/__init__.py`; the constructor is the first thing that proves
   the compiled extension actually loads. Constructing allocates no device memory (exploration §9).
2. **`accepted_io_apis`: do not declare it.** Verified dead code — `grep` finds exactly the base
   declaration (`solvers.py:466`) and one test double (`test_persistent_solver_orchestrator.py:24`);
   **no shipped solver class declares it and no production code reads it**. plan-1 §1.3 and
   plan-4 §0b are right; plans 2/3/5 declare `frozenset({"direct"})`. A lone declaration is a §6.6a
   stylistic outlier and mildly misleading (a reader assumes it is enforced). Record the reason so
   Phase 6 sees a decision.
3. **Status-map keying: any of the four shapes proposed is safe, but the enum-keyed *single* dict
   would not be.** plan-4 §0a is correct and I reproduced it: `LPTerminationStatus` and
   `MILPTerminationStatus` are `IntEnum`s, `LP.PrimalInfeasible == MILP.Infeasible` is `True`, they
   hash equal, and one dict silently keeps a single key. Two dicts keyed on enum members (plan-1,
   plan-5) and name-keyed maps (plan-2, plan-3, plan-4) are all fine —
   `get_termination_reason()` returns `termination_status.name` (verified,
   `solution/solution.py:285-289`). Name keys avoid importing the `.pyx` module
   `cuopt.linear_programming.solver.solver_wrapper`; the enum-keyed form is more type-safe.
   `ProblemCategory` has three members (`LP=0, MIP=1, IP=2`) — verified — so the LP test must be
   `category == ProblemCategory.LP`, never `!= MIP`.
4. **`_LazyModule("cuopt.linear_programming")` (plan-2 §1, plan-3 §2.1.3) resolves.**
   `cuopt.linear_programming` re-exports `DataModel`, `Solve`, `SolverSettings`, `SolverMethod`,
   `PDLPSolverMode`, `Solution`, `Read`, `ParseMps`, `BatchSolve` plus the submodules — verified by
   `dir()`. Either handle works; the charter's literal `_LazyModule("cuopt")` (C-R3) keeps call
   sites self-documenting (`cuopt.linear_programming.DataModel`), the subpackage handle makes
   accidentally widening the import to `cuopt.routing` structurally harder. Cosmetic; log whichever.
5. **No `# type: ignore` on the `_LazyModule` line.** plan-5 §2.1 is right: `warn_unused_ignores =
   true` (`pyproject.toml:155`) and cuopt ships no `py.typed`, so a redundant ignore fails mypy. The
   `cupdlpx` line carries none.
6. **`set_objective_offset` is genuinely unnecessary.** Verified: `Objective` raises
   `ValueError("Constant values in objective function not supported.")` (`objective.py:193`). All
   five plans agree; recorded because it removes M2's offset-negation worry entirely.
7. **`M.vtypes` is `<U1`** (built with `np.full(n, "B")`, `matrices.py:91`) and `M.sense` is
   `dtype=object` — both fine for `np.where`/`np.equal` comparisons, and `<U1` is what
   `set_variable_types` wants. `M.A` is a `csr_array | None`.
8. **`_SOLVER_PROBE_ORDER` position is a real (if small) user-visible choice** and the plans split:
   before `cupdlpx` (plan-2, plan-4 — cuOpt is the capability superset) vs after (plan-1, plan-3,
   plan-5 — do not change any existing user's default). Both are defensible; note the mechanical
   consequence either way: omitting the entry entirely makes `test_available_solvers.py:39`
   (`is_available() == (value in available_solvers)`) fail, which is the cheap guard.
9. **plan-4's insight on `NoTermination` is worth carrying into `FINAL.md` verbatim**: it must map to
   `internal_solver_error`, not `unknown`, because `safe_get_solution`'s `unknown` branch parses the
   solution and *flips the status to `ok`* (`solvers.py:1192-1198`) — which on cuOpt's malformed-input
   catch-all means reporting an empty/`nan` solution as a success.
10. **Bulk.** All five estimate +510…+770 over 12–14 files against #516's +771/−62 over 15 — all
    inside the §9 yardstick. plan-1's +560 is the leanest credible estimate; the grafts below add
    perhaps +40 (subprocess probe, pad-row helper, `_build_file` override) and remove ~10 (no
    `_license_probe` solve, no bool allowlist).
11. **QP on linopy's own fixtures now has data** (W1): `test_quadratic_model` → `Optimal`,
    `obj=3.2e-08`, `x≈5.7e-05` → `round(3)==0` passes; `test_quadratic_model_cross_terms` →
    `Optimal 77.50000039` → `round(3)==77.5` passes. Only `wo_constraint` (B1) and `unbounded`
    (B10) need work. plan-1 risk 5 was right to name these as WP-F's biggest unknown.
12. **The `0.5·M.Q` convention is confirmed a third time** from linopy's own matrices: for
    `m.add_objective(x*x)`, `M.Q = [[2,0],[0,0]]` (the Hessian, doubled diagonal), so
    `Q_cuopt = 0.5·M.Q` gives `Q + Qᵀ = H`. All five plans have this right; keep the
    "fails with the naive encoding" demonstration (plan-2 WP-E, plan-3 WP-F AC-2, plan-5 WP6 AC-2)
    as a real acceptance item — a check that cannot fail is not a check.
13. **plan-2's and plan-5's `to_cuopt` docstring note** ("a `max` model comes back as the equivalent
    minimisation") is a small, high-value detail the other plans omit. `to_cuopt` is public API and
    M2 makes its output surprising; document it.

## Checklist

- [x] Every requirement ID in scope is covered — all five plans map C-R1…C-R17 with no unmapped ID;
      plan-1, plan-2, plan-4 and plan-5 additionally propose C-R18+ items. **Note for `FINAL.md`:**
      no `C-R18+` has been recorded in `logs/30-decisions.md` yet (it holds D1–D6 only), so the five
      plans' conflicting numberings are all still free. `FINAL.md` must fix one numbering — see the
      instruction block.
- [x] No requirement silently dropped or reinterpreted — two are reinterpreted and both are called
      out: plan-3 drops `MIP_DUAL_BOUND_REPORT` from C-R1 (B11) and plan-5 drops
      `SEMI_CONTINUOUS_VARIABLES` (B12). The four charter-declared zero-diff rows (C-R7, C-R10,
      C-R11, C-R17) are treated as proof obligations by all five, which is the charter's own reading.
- [ ] Frozen verification plan unchanged — **not applicable in round 1**:
      `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md` does not exist yet and
      no Gate 3 digest has been recorded. This row becomes live at the first review of `FINAL.md`.
- [x] No threshold widened, no check downgraded to a weaker tier — nothing is frozen yet, so
      nothing could be widened. Reviewed instead as *proposals*: plan-1 §8 and plan-3 §8.5 are the
      best-argued tolerance tables (every number tied to a measurement); B10, B14 and B15 are the
      three places where a plan would have had to widen or improvise later, and they are fixed
      *before* the freeze, which is the legitimate path.
- [x] Tests actually ran (GPU tests not silently skipped) — reviewed as a plan property: all five
      require the paired `-k cuopt` runs with and without `--run-gpu`, with pasted `PASSED` lines
      (plan-3 §8.4 and plan-5 WP4 are the most explicit). Carried into the instruction block.
- [x] Diff is surgical — no drive-by reformatting or refactoring — all five keep
      `solver_capabilities.py`, `constants.py` and `conftest.py` at zero diff and refuse to edit
      `QUADRATIC_SOLVERS`. Two drive-by temptations are correctly refused by all five: the shared
      "not installed" message (`model.py:2077`) and cuPDLPx's own guards. I13 records instead.
- [x] Matches repo idiom (current `master`, not PR #516) — verified against the real
      `class cuPDLPx` (`solvers.py:4155-4416`) and `class Highs`. All five follow the `Highs`
      `log_file` idiom rather than cuPDLPx's warn-and-ignore, all five refuse to copy cuPDLPx's
      `if MAXIMIZE: dual = -dual` (`:4385-4386`), and all five prefer `Highs`'s
      `.get(..., unknown)` default over cuPDLPx's raw-code passthrough (`:4374-4376`). All three are
      correct readings of `master`.

## Selection — base plan and why

**Base plan: `plan-1.md`.**

It is the only plan that is simultaneously (a) compliant with the binding D6 ruling without needing
an override, (b) correct on every feature-flag call that later turned out to be checkable, and
(c) free of the two mechanical defects that would have crashed a MILP or a bare solve. Concretely,
against the charter's requirements, the exploration evidence and repo idiom:

* **D6 / §4.6 c3** — plan-1 §5c puts the device probe in `is_available()` and spells out the exact
  GPU-less behaviour surface by surface. plan-3 and plan-5 chose the position D6 overrides (B7);
  plan-4's implementation of the same position crashes on the branch that matters (B6).
* **Feature flags** — plan-1 §3 is the only feature table where every declared and every *denied*
  flag survives checking: `MIP_DUAL_BOUND_REPORT` (verified populatable, B11),
  `SEMI_CONTINUOUS_VARIABLES` (verified supported, B12), no `accepted_io_apis` (verified dead code),
  no new `SolverFeature` member, `QUADRATIC_OBJECTIVE` deferred to the QP package so the LP/MILP
  core is never green-by-omission.
* **Mechanical correctness** — plan-1 is the only plan that both keeps the MIP dual unscattered
  (B8 — plan-5 also guards it) *and* keeps a working bare `model.solve("cuopt")` (B3 — plans 3 and 5
  also do, better). Its `_solve` also validates the primal length before parsing, which matters
  because `NoTermination` returns an **empty** primal array without raising.
* **Verification quality** — plan-1 §8 is the strongest test plan in the set: 15 numbered V1 rows,
  every one differential against live in-process HiGHS, the presolve-solvable `max` case *and* a
  `get_solved_by() == Unset` coverage guard (V1-3, with a docstring saying that a red there means
  re-establish coverage, never delete M2), the repeat-solve regression run in a **subprocess** so a
  SIGSEGV is a failed test rather than a dead session, the GPU-less degradation checked in a
  subprocess, and a tolerance table where each number cites the measurement it came from
  (`atol=1e-7` from Barrier's measured 1.73e-09/2.47e-09; `rtol=1e-4` for the medium random LP from
  the measured 2.5e-5 inter-method spread, chosen up front rather than after a failure).
* **Honesty about its own soft spots** — plan-1 risk 1 names the device-probe API as "the single
  most likely place this plan is wrong" (it was: B5/B6), risk 5 names the QP fixtures (B10 and B1
  both landed there), risk 8 names its own `_license_probe` cost. A plan that predicts where it will
  break is worth more than one that reads cleanly.
* **Repo idiom** — §2.1's anchor table is the most complete (imports, `_LazyModule` with the
  never-import-`cuopt.routing` comment, enum, class placement after `cuPDLPx`, probe order), and it
  is explicit about the three anchors that must stay zero-diff.

plan-4 is a close second and is the best *analytical* document of the five (its §0a IntEnum
verification and §0b `accepted_io_apis` finding are the two most valuable independent checks any
planner did, and both hold). It loses the base slot on three implementation calls — the fail-open
probe that raises (B6), the zero-nnz `ValueError` (B2) and the `_run_file` refusal (B3) — each of
which is a red test or a silent contradiction of D6. plan-2 is third: excellent priced arguments and
the only plan that spotted why zero-nnz padding is needed at all, but two mechanical bugs (B8, and
its `_run_file` refusal). plan-3 and plan-5 contribute the best fallback shape (`_build_file`) but
both take the D6-overridden side, and each drops a verified capability (B11, B12).

## Grafts — take these into `FINAL.md`

| # | Graft | From | Why |
|---|---|---|---|
| **G1** | Override **`_build_file`** to build directly, with a one-shot `logger.warning` naming `io_api="direct"`, and do **not** override `_run_file`. | **plan-3 §2.1.9**, plan-5 §1.3 | Verified against `Solver.solve`'s dispatch (`solvers.py:858`): it works, and unlike plan-1's `_run_file` fallback it never writes-then-discards a problem file (0.23 s / 7.7 MB at 10⁵ variables, exploration Q0). Must carry B4's no-leak fix. |
| **G2** | The two verified factual claims and the reasoning around them: `IntEnum` members of the two status enums collide as dict keys; `accepted_io_apis` is unread by production code. Plus "`NoTermination` must not map to `unknown` because `safe_get_solution` would flip the status to `ok`". | **plan-4 §0a, §0b, §4** | Independently reproduced (Non-blocking 3, 2, 9). These are the sharpest pieces of analysis in the plan set and they belong in `FINAL.md`'s comments, because both are things a later reader would "simplify" back. |
| **G3** | Blanket `bool → int` coercion of option values (replacing plan-1's `_BOOL_PARAMETERS` allowlist), plus the wrapped unknown-*name* `ValueError` and untouched value errors. | **plan-2 §5f** (idea), corrected by W7/W8 | plan-2 was right that the coercion must exist and right that the allowlist would rot; the measurement shows the simplest form is also the correct one (B9). |
| **G4** | WP-A acceptance criteria **front-loaded with empirical checks before any test file is written** (sign matrix by hand, presolve-`max` case, repeat solves, device probe both directions, pad shape), and the discipline of demonstrating once that a guard *fails* when its constant is removed (`0.5·M.Q` → `M.Q` must change the answer). | **plan-2 §7 (WP-A AC 1-10, WP-E), plan-3 WP-F AC-2, plan-5 WP1 AC-5/WP6 AC-2** | Charter §9's "make it fail on purpose once" applied to the two silent-wrong-answer sites (Q convention, method default). It also front-loads B1's pad shape into WP-A instead of discovering it in WP-F. |
| **G5** | The named risk register for **shared assertions tighter than cuOpt's tolerances** — `test_solvers.py:101-103` `approx(rel=1e-6)` and `test_optimization.py:717-718` exact equality — with the disposition pre-decided at Gate 3. | **plan-5 §8 "Pre-declared contingency", risks 2 and 4** | The only plan that read the shared suite closely enough to find these. Fold into `logs/15-verification.md` per B14 (as a pre-freeze decision, not an ad-hoc permission). |
| **G6** | `to_cuopt`'s docstring stating that a `max` model is returned in minimisation form. | **plan-4 §2.2, plan-5 §2.2** | Public API whose output M2 makes surprising. Two lines. |
| **G7** | The `_license_probe` that constructs `DataModel()` + `SolverSettings()` and does **not** solve. | **plan-2 §2.1** | Non-blocking 1: proves the extension loads (which `find_spec` never does) without a GPU solve at collection time. |
| **G8** | The pre-decided contingency shape for `method=3` on MILP: if Barrier misbehaves for MILP, apply the default only to non-integral models and report the numbers — never fall back to `method=0`, never invent a third option. | **plan-3 §5b "Named contingency", plan-2 §5b, plan-5 §5b** | Removes the one place a coding agent would otherwise have to improvise under I12. (My probes show Barrier solving both MILP fixtures exactly — W9/W10 — so this is insurance, not an expected path.) |
| **G9** | Every V1 test parametrized on a `solver` argument (or an explicit `pytest.mark.gpu` + `LINOPY_RUN_GPU_TESTS` module gate) so `conftest.py` stays a zero-diff row, plus the paired skipped/ran evidence. | **plan-2 §2.6/§8, plan-3 §8.1, plan-1 §8** | Verified: `pytest_collection_modifyitems` only sees tests whose callspec has a `solver` param (`conftest.py:68-79`); `LINOPY_RUN_GPU_TESTS` is set at `:47-50` for exactly the other case. |
| **G10** | Commit 3's docs must describe **LP and MILP only**, with the QP paragraph landing in the QP commit, so every intermediate commit is truthful. | **plan-4 §7 note, plan-3 WP-F** | Cheap, and it removes the "docs promise an unimplemented feature" review finding at Gate 6. |

## Rejections — do not carry these into `FINAL.md`

| # | Rejected | From | Why |
|---|---|---|---|
| **R1** | `is_available()` as a pure import probe with the GPU check in `_license_probe`. | plan-3 §5c, plan-5 §5c | Overridden by **D6** (B7). Keep the argument only as the recorded downside. |
| **R2** | Any in-process CUDA call inside `is_available()` (`cudaGetDeviceCount`, `cuInit`, a tiny solve). | plan-1 §2.1.8, plan-2 §5c, plan-4 §5c | Verified to break `fork()`ed children for cuOpt *and* any other CUDA user in the process (B5, W4). |
| **R3** | Fail-**open** device probe (`except: return True`). | plan-4 §5c | Contradicts D6 whenever the probe cannot run, and pairs with a `count > 0` that raises on the GPU-less branch (B6). Fail closed + pin the positive branch with a test (B16). |
| **R4** | `raise ValueError` for a zero-nonzero constraint matrix. | plan-3 §2.1.6, plan-4 §2.1 | Fails `test_quadratic_model_wo_constraint` (B2). |
| **R5** | Doubly-infinite pad row. | plan-1 §2.1.5, plan-2 §2.1, plan-5 `_pad_free_row` | Verified to produce `NumericalError` for QP (B1). |
| **R6** | `_run_file` raising `NotImplementedError` as the answer to a bare `model.solve("cuopt")`. | plan-2 §1, plan-4 §1 | Fails at least two shared tests in `test_solvers.py` (B3). |
| **R7** | Bool→int coercion by retrying after a `set_parameter` `ValueError`. | plan-3 §5f | Dead code: `set_parameter` does not validate values (B9, W7). |
| **R8** | `get_parameter`-based type sniffing for the coercion. | plan-2 §5f | Works (verified: `presolve` default is `int(-1)`, `log_to_console` default is `bool(True)`), but unnecessary now that a blanket coercion is verified safe. Extra API surface for no gain. |
| **R9** | Declining `MIP_DUAL_BOUND_REPORT`; declining `SEMI_CONTINUOUS_VARIABLES`. | plan-3 §3.2; plan-5 §3 | Both premises are false (B11, B12). |
| **R10** | Declaring `accepted_io_apis`. | plans 2, 3, 5 | Dead code and a lone stylistic outlier (Non-blocking 2). |
| **R11** | Adding a `"cuopt": {"method": 1, …}` row to `test_solver_method_options`. | plan-3 §2.7 | Predicted red against `np.isclose`'s default `rtol=1e-5` (B15). |
| **R12** | Declaring `READ_MODEL_FROM_FILE` / wiring `cuopt.linear_programming.io` in this PR. | (none proposed it — all five diverged from Q0's secondary recommendation) | **Confirmed as the right call, unanimously and for good reasons.** Recorded here so the 5/5 agreement is visible as a decision (§3.5: 5/5 → obvious, do not escalate), with plan-3 §1.3's strongest argument preserved: the file path *inverts* the Q convention (`.lp` yields triangular, `.mps` symmetric), so it would need a second, opposite implementation of the most dangerous transform in the change. Record as an evidence-backed follow-up. |
| **R13** | A `logger.warning` for `solution_fn` instead of `NotImplementedError`. | plan-4 §5h | Cosmetic split (plan-1/2/3/5 raise; cuPDLPx raises at `:4368-4371`). Follow cuPDLPx: raise. Not worth a review round — decided here so nobody re-litigates it. |

## What all five missed

Charter §9's planner-groupthink risk. Ten checks, each run because the plans disagreed on a fact or
asserted something unverified. All output pasted; all read-only.

**W1 — cuOpt's answers on linopy's own QP fixtures** (this settles B1, B2 and B10; every plan
treated these as an unknown or assumed padding would fix them):

```
(a) quadratic_model WITH con0     -> ('Optimal', 3.23384535631667e-08, x[:4]=[5.687e-05 …])   # round(3)==0 -> test passes
(b) box-only QP, full PD Q, free pad row -> ('NumericalError', nan, [0. 0. 0. 0.])
(c) box-only LP, free pad row            -> ('Optimal', 0.0, [0. 0. 0. 0.])
(e) quadratic_model_unbounded            -> ('NumericalError', nan, [0. 0. 0. 0.])
(f) quadratic_model_cross_terms          -> ('Optimal', 77.50000038766152)                     # round(3)==77.5 -> passes
```

**W2 — the pad row must not be doubly infinite** (`quadratic_model` minus `con0`, i.e. the exact
fixture of `test_quadratic_model_wo_constraint`, 20 variables, `Q_cuopt = 0.5·M.Q`):

```
pad bounds (-inf, +inf)      -> ('NumericalError', nan)
pad bounds (-1e30, 1e30)     -> ('NumericalError', nan)
pad bounds (lb0, +inf)       -> ('Optimal', 7.589716418676626e-15, x[:3]=[2.64e-08 2.77e-08 2.77e-08])
pad bounds (lb0, ub0)        -> ('Optimal', 7.589716418676626e-15, …)
pad row = sum(x) >= 0        -> ('Optimal', 2.2703098195813027e-08, …)
```

and, on a strictly positive-definite box-only QP (10 vars, `H = 2I`), isolating the row bounds as
the cause rather than the singular Hessian:

```
free   -> NumericalError nan          ge0 -> Optimal 1.47e-15      box -> Optimal 1.69e-14
sumge0 -> Optimal 3.44e-09         sumbox -> Optimal 2.65e-08      eq0 -> Optimal 3.32e-08
```

**W3 — the device-probe return convention** (named as risk #1 by plans 1, 2 and 4; unverified in all
three, and `count` is `None`, not `0`):

```
normal:      (<cudaError_t.cudaSuccess: 0>, 1)
CUDA_VISIBLE_DEVICES='':  (<cudaError_t.cudaErrorNoDevice: 100>, None)
CUDA_VISIBLE_DEVICES='-1': (<cudaError_t.cudaErrorNoDevice: 100>, None)
```

**W4 — the in-process probe poisons `fork()`** (the most consequential thing in this review; no plan
tested it, plan-5 §5c predicted it for a different API):

```
--- no probe ---   child solve: Optimal 2.0        child exit status: 0
--- probe ---      parent probe: (<cudaError_t.cudaSuccess: 0>, 1)
                   child FAILED: InputValidationError A CPU-memory problem requires remote execution.
                                 Set CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT to solve on a remote GPU server.
                   child exit status: 768
```

Fork-safe alternatives, measured in the same harness: importing `cuda.bindings.runtime` without
calling it (`child solve: Optimal 2.0`, exit 0); a `/dev/nvidia[0-9]*` + `CUDA_VISIBLE_DEVICES`
heuristic (exit 0); `driver.cuDeviceGetCount()` without `cuInit` (returns
`CUDA_ERROR_NOT_INITIALIZED`, exit 0 — fork-safe but uninformative).

**W5 — the out-of-process probe: correct, cheap and fork-safe:**

```
probe -> True in 0.280s
probe -> True in 0.274s
with CUDA_VISIBLE_DEVICES='' -> returncode 1
  child exit status: 0          # cuOpt still solves in a forked child after the probe
```

**W6 — an empty MIP dual cannot be scattered** (B8):

```
values_to_lookup_array(np.array([]), np.array([0,1,2]), size=3)
IndexError: boolean index did not match indexed array along axis 0; size of axis is 0 but size of
corresponding boolean axis is 3
```

**W7 — `set_parameter` does not validate values; `Solve()` does** (B9; contradicts the natural
reading of exploration §6's closing note, which every plan built an option strategy on):

```
set presolve=False OK          set presolve=0 OK           set log_to_console=False OK
set log_to_console=0 OK        set crossover=False OK      set crossover=0 OK
get_parameter defaults: presolve=-1 (int)  log_to_console=True (bool)  crossover=False (bool)
                        method=0 (int)     time_limit=inf (float)      log_file='' (str)
```

**W8 — at `Solve()` time, ints are accepted everywhere bools are, and only ints are accepted by
int-typed parameters** (so a blanket `bool → int` is the right rule):

```
presolve_False   -> ValueError: Parameter presolve value False is not an integer
presolve_0       -> Optimal obj=2.0
ltc_int0         -> Optimal obj=2.0      crossover_int1 -> Optimal obj=2.0
infdet_int0      -> Optimal obj=2.0      method_bool    -> ValueError: Parameter method value True is not an integer
```

**W9 — `MIP_DUAL_BOUND_REPORT` is honestly declarable** (B11), measured on the exact `milp_model`
fixture whose test asserts it, under `method=3`:

```
status: Optimal cat: 1 obj: 100.0
x: [0.5 …]   y: [9. …]      assert (y==9)|(x==0.5): True
milp_stats: {'mip_gap': 0.0, 'solution_bound': 100.0, 'presolve_time': 0.0876, 'max_int_violation': 0.0,
             'num_nodes': 0, 'num_simplex_iterations': 0}
get_dual_solution -> AttributeError Attribute get_dual_solution is not supported for milp solution
```

**W10 — the exact-equality MILP assertion currently holds** (B14's first half), on `model_with_inf`
under `method=3`:

```
status: Optimal cat: 1 obj: 100.0
x block: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]     y block: [10. 10. …]
EXACT x==0 all: True   EXACT y==10 all: True
```

**Two further gaps that are not empirical**, both missed by all five:

* **Nobody noticed that a bare `model.solve(...)` is exercised by `test_solvers.py`** (B3). Every
  plan reasoned about the bare call only against the DoD sentence, and two of them concluded it was
  acceptable to refuse it. The shared suite settles it.
* **Nobody costed the temp-file leak** the `_build_file` interception creates (B4), because nobody
  traced `Model.solve`'s `problem_fn` bookkeeping (`model.py:2098`, `:2135`, `:2145-2147`) through the
  override.

**Sequencing (§6.4) — all five respect the rules**, with one thing to fix: plan-1 WP-C and plan-4
WP-C run packaging concurrently with the `solvers.py` core (correct, disjoint file, and §6.4 rule 5
explicitly wants it early), and every plan puts QP last and alone. plan-1's WP-F touches
`solvers.py`, `test_cuopt.py`, `test_solvers.py` and `gpu-acceleration.rst` — all after their
owners merge, so no concurrent sharing. The one gap: **no plan carries the expected base commit SHA
into its slice briefs** (§6.4's lineage check). `FINAL.md` should say, per work package, that the
brief must carry the base SHA and that the agent stops on `WRONG LINEAGE`.

## Positions on the five disagreement points

1. **bool→int option coercion** — **coerce every `bool` with `int(v)`, unconditionally.** Evidence
   W7/W8: int-typed parameters reject bools *at `Solve()`*; bool-typed parameters accept `0`/`1`.
   Therefore no allowlist (plan-1), no `get_parameter` sniffing (plan-2), and definitely no retry
   (plan-3 — dead code). Keep the wrapped `ValueError` for an unknown parameter *name* (that error
   **is** eager at `set_parameter`) and pass value errors through unchanged; note the value error
   surfaces from inside `Solve()`, i.e. from the interrupt helper's worker thread, so the helper must
   re-raise it (all five do).
2. **`MIP_DUAL_BOUND_REPORT`** — **declare it.** W9 shows `solution_bound = 100.0` and
   `mip_gap = 0.0` on the very fixture whose assertion (`test_optimization.py:834`) the flag
   activates. Negate `solution_bound` for `max` under M2. Keep the drop-the-flag-and-report
   contingency (plan-1 §3, plan-4 §3) — it is now insurance rather than a live risk.
3. **`_build_file` / `_run_file` fallback shape** — **override `_build_file` to build directly;
   leave `_run_file` alone.** B3 makes a working bare solve mandatory, and B4 makes the no-leak fix
   part of the same change. plan-1's `_run_file` fallback also works but pays a full file write it
   then discards (0.23 s / 7.7 MB at 10⁵ variables) and leaves the model exposed to
   `read_io_api_from_problem_file` bookkeeping for no benefit. A user who explicitly passes
   `io_api="lp"` gets the direct path plus one warning — state that in the docs (it is the ergonomic
   wart plan-3 risk 4 names, and it is better than an exception).
4. **`doc/api.rst` `to_cuopt` row** — **add it**, alongside `solvers.cuOpt`. The "no `to_*` helper is
   listed" objection is real but weaker than the plans assumed: `Model.to_netcdf` is bound by the
   identical `to_netcdf = to_netcdf` assignment (`model.py:2507`) and is listed at `api.rst:115`, so
   the autosummary entry is proven to build. Charter §6.6b names `to_cuopt` explicitly; two lines
   close the row instead of arguing it at Gate 6. Record the missing siblings in
   `logs/45-known-issues.md` (I13).
5. **Probe mechanics under D6** — **out-of-process, cached, fail-closed, both branches pinned by
   tests.** D6 fixes the observable behaviour (`available_solvers` omits cuopt without a GPU) and
   leaves the mechanics to `FINAL.md`; W4 rules out every in-process CUDA call, and W5 shows the
   subprocess form is correct in both directions, costs 0.28 s once, and is fork-safe. Required
   verification rows: (i) positive branch on this T4 (`"cuopt" in available_solvers`), (ii) negative
   branch under `CUDA_VISIBLE_DEVICES=''` in a subprocess, (iii) **fork safety** — probe, `fork`,
   solve in the child, child exits 0. Row (iii) is a check that can fire: the naive in-process probe
   fails it, so run it once against the naive form to prove the check works (charter §9). If the lead
   rejects `subprocess` in library code, the fallback is the CUDA-free heuristic
   (`CUDA_VISIBLE_DEVICES` + `/dev/nvidia[0-9]*`, verified fork-safe) — weaker (it cannot see driver
   version or compute capability) but sufficient for §4.6 c3's two scenarios, and the honest
   error-message translation at solve time (plan-3 §2.1.6 step 5, plan-5 `_run_cuopt_solve`) should
   be grafted in **either** case, because a machine with a device node but too old a driver still
   needs to fail with something better than "Set CUOPT_REMOTE_HOST".

## Instruction block for the revision planner

Write `/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/FINAL.md` and
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md`. Do not edit this file.

1. **Base**: `plan-1.md`. Carry its structure, feature table, status maps, C-R map, work-package
   split, docs plan, non-goals and tolerance table forward as the spine.
2. **Apply G1–G10** (grafts) and **R1–R13** (rejections) above, and resolve **B1–B16**. Every
   blocking finding must be visibly addressed in `FINAL.md` — a sentence saying what changed, not a
   silent edit.
3. **Requirement IDs**: no `C-R18+` has been recorded anywhere yet, so `FINAL.md` assigns them for
   the first time. Adopt **plan-1's C-R18…C-R27 verbatim** (honest GPU degradation, M2,
   repeat-solve mitigation, zero-nnz handling, Ctrl-C, MIQP guard, option handling, refusals,
   semi-continuous, MIP report) and add:
   * **C-R28** — the availability probe must not initialise CUDA in the caller's process
     (fork-safety), with the three verification rows of position 5.
   * **C-R29** — a bare `model.solve("cuopt")` (no `io_api`) solves, warns once, and leaves no
     `linopy-problem-*` file behind.
   Fold B1's corrected pad shape into C-R21. Every ID must appear in the `FINAL.md` map and in
   `logs/15-verification.md`.
4. **Freeze these numbers** in `logs/15-verification.md`, each with the measurement it rests on
   (plan-1 §8's table is the starting point; plan-3 §8.5's interlock note — "these tolerances are
   only valid because the default method is Barrier; a late method change means re-deriving them at
   Gate 3" — must be carried verbatim). Add rows for: the corrected pad shape (B1), the
   `test_quadratic_model_unbounded` disposition (B10), the two tight shared assertions (B14), the
   three probe branches incl. fork safety (B5/B16), the no-leak check (B4), and the MILP
   dual-shape check (B8).
5. **State the six-cell V1 matrix and the presolve-solvable `max` case as separate rows**, both
   against live in-process HiGHS, plus the `get_solved_by() == Unset` coverage guard with plan-1
   V1-3's docstring rule (a red there means re-establish coverage; it never means remove M2).
6. **Slice briefs**: each work package's brief must carry the expected base commit SHA and the
   §6.4 lineage check, and must end with the §3.6 threshold paragraph verbatim.
7. Keep the diff proportionate: plan-1's +560/−5 over 13 files plus the grafts is ~+600, well inside
   the §9 yardstick. If `FINAL.md` grows past ~+900, say why.

Round 2 will check `FINAL.md` against this list, against §5's C-R coverage, and against the four
Gate-3 conditions in §6.3 (every ID mapped; every threshold a number with a rationale; all six
sign-matrix cells; explicit non-goals) — plus the frozen-plan sha256, once it exists.

---

# cuOpt support — plan review, round 2

*Reviewer: Phase 3 selector-reviewer (Opus 5, `claude-opus-5[1m]`) · 2026-08-22 · Reviewing:
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/FINAL.md` (1621 lines) and
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md` (193 lines), both read in
full from disk*

Basis for this round: `FINAL.md` and `logs/15-verification.md` in full; charter §§3.6, 5, 5.1, 5.3,
6.3, 6.3a, 6.4, 6.4a, invariants I4/I9/I11/I12/I13; `templates/verification-TEMPLATE.md` and
`templates/review-TEMPLATE.md`; round 1 of this file; and read-only inspection of `master`
(`09c34dd`): `linopy/solvers.py`, `linopy/model.py`, `linopy/objective.py`,
`test/test_optimization.py`, `test/test_solvers.py`, `test/test_semi_continuous.py`. Every numeric
claim below was re-measured in this session; the output is pasted.

## Verdict

**REVISE** — one blocking finding (**B19**). Everything else checked out.

`FINAL.md` resolves B1–B16 substantively, not narratively: I re-derived nine of the sixteen
independently and every disposition holds. B17 and B18 are real defects, correctly diagnosed and
correctly fixed, and the B3 line-number correction is right — mine was wrong (see NB-3). All 29
`C-R` IDs map to concrete work and all 29 appear in the verification plan. The four §6.3 Gate-3
conditions are confirmed on the record below.

The single blocker is a **frozen verification row that is provably red on a non-defect**: V1-13b (and,
latently, V1-6b) asserts primal agreement with HiGHS on the `quadratic_model` fixture, whose ten `y`
variables are not determined by the problem. Measured discrepancy **145.98 absolute against a frozen
`atol` of `1e-4`** — 1.46e+06×. Under I12/§3.6 a WP-F agent hitting that must report and stop, and the
realistic failure mode is that it widens the tolerance instead — the exact hazard §3.6 exists to
prevent. It is a two-row fix, and it is worth one more iteration rather than freezing it.

**No sha256 is recorded this round.** `15-verification.md` must change to resolve B19, so a digest
recorded now would be stale. Its current digest is
`33066438f1522cdff72793f295a7474f39f1729387e22dec2a270288dad6ba49` (informational only, **not** a
freeze). The Gate-3 digest goes into this file in the round that returns APPROVE.

## Blocking findings

| # | Finding | Where (file:line) | Why it blocks | Suggested fix |
|---|---|---|---|---|
| **B19** | **V1-13b asserts a primal quantity the problem does not determine, and is red by 1.46e+06× its own frozen tolerance.** The `quadratic_model` fixture (`test/test_optimization.py:261-271`) has 10 `x` and 10 `y` variables; the objective is `x*x` only. Measured: `M.c` is **all zeros**, `M.Q` diagonal is `[0., 2.]` (zero for every `y`), `lb=0`, `ub=+inf`. So each `y_i` has zero cost, zero curvature and no upper bound: with `con0` the optimal face is `{x=0} × {y_i ≥ 10}`, without it `{x=0} × [0,∞)^10` — **unbounded in `y` either way**. Measured on this exact fixture, `method=3`: cuOpt returns `y = 155.97919941`, HiGHS returns `y = 9.9999995`. Both are optimal (objectives `3.23e-08` and `2.50e-12`). `\|Δy\| = 145.98` against the frozen `CUOPT_PRIMAL_ATOL_DEGENERATE = 1e-4`. | `logs/15-verification.md` V1-13b (line 77) — "primal `rtol=1e-6, atol=CUOPT_PRIMAL_ATOL_DEGENERATE=1e-4`" on `quadratic_model`; same defect latent in V1-6b (line 67), which asserts primal vs HiGHS on the `con0`-removed form (measured `y = 0` on both sides **today**, so it passes by luck on the same underdetermined quantity); and `FINAL.md` §8.3 V1-13, which claims "the **solution** comparison is what catches the `0.5·M.Q` trap" for these fixtures | A frozen row that goes red with nothing broken is worse than no row: under I12/§3.6 the WP-F agent must report and stop, so this manufactures a hard stop in the last work package — and the pressure it creates is to widen `atol`, which is precisely the failure §3.6 names. It also contradicts the plan's own Nondeterminism section, which is supposed to enumerate every quantity that is "never asserted on". The Q-convention guard does not need this row: V1-13a (3-variable, positive-definite `Q`, unique optimum) and V1-13c (deliberate-failure demo) carry it, as the plan itself says two sentences later. | (1) **V1-13b**: compare the objective (`rtol=1e-6, atol=1e-6`) and **the `x` variables only** (`atol=1e-4`) for `quadratic_model`; keep the full primal comparison for `quadratic_model_cross_terms`, which is non-degenerate (`y` carries a `+1` objective coefficient; unique optimum `x=1.5`, `y=8.5`). (2) **V1-6b**: assert `condition == "optimal"`, `x.round(3)==0`, `round(obj,3)==0` and the `x`-only differential; do **not** compare `y`. (3) Add one row to the **Nondeterminism** table: *"the 10 `y` components of `quadratic_model` / `quadratic_model_unbounded` — zero cost, zero curvature, `ub=+inf` ⇒ unbounded optimal face; measured cuOpt `155.979` vs HiGHS `9.9999995`; never asserted against the oracle."* (4) **`FINAL.md` §8.3 V1-13**: drop the claim that the `quadratic_model` *solution* comparison catches the `0.5·M.Q` trap — it does not (the optimum is `x=0` for **any** positive `Q` scale); attribute that to V1-13a and V1-13c. |

Evidence, measured this session on the installed stack (numpy 2.4.6, cuopt 26.08.00, HiGHS 1.15.1),
through `model.matrices` on the real fixture:

```
$ .venv/bin/python r2_probe.py matrices
with_con= True  c= [0.] Qdiag= [0. 2.]
   lb= [0.] ub= [inf] A is None: False
with_con= False c= [0.] Qdiag= [0. 2.]
   lb= [0.] ub= [inf] A is None: True

$ .venv/bin/python r2_probe.py highs
with_con= True  ok optimal obj= 2.4999994899963244e-12
   x= [4.99999949e-07 ...]  y= [9.9999995 9.9999995 9.9999995 9.9999995]
with_con= False ok optimal obj= 0.0
   x= [0. 0. 0. 0.]         y= [0. 0. 0. 0.]

$ .venv/bin/python r2_probe.py cuopt_withcon        # 0.5*M.Q, method=3
term= Optimal obj= 3.23384535631667e-08
x= [5.68669092e-05 ... ]
y= [155.97919941 155.97919941 155.97919941 155.97919941 ... ]
cat= 0 LP type(reason)= <class 'str'>

$ .venv/bin/python r2_probe.py cuopt_free_y         # con0 removed, singly-bounded pad row
term= Optimal obj= 7.589716418676626e-15
x= [2.64331215e-08 2.76706955e-08 ... ]
y= [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
max|y|= 0.0
```

**`|Δy| = 155.97919941 − 9.9999995 = 145.97919991`, against `atol = 1e-4`.**

Two things this does **not** break, stated so nobody over-corrects:

* **No shared test goes red.** `test_quadratic_model` (`test_optimization.py:868-882`) asserts
  `solution.y.round(3) >= 10` — an **inequality** — precisely because `y` is underdetermined;
  `155.979 >= 10` passes, as do `x.round(3)==0` (`5.687e-05`) and `round(objective,3)==0`
  (`3.23e-08`). `test_quadratic_model_wo_constraint` (`:927-944`) asserts only `x` and the
  objective. V2-1 is unaffected.
* **The tolerance *tier* is not the problem and does not need re-deriving.**
  `CUOPT_PRIMAL_ATOL_DEGENERATE = 1e-4` is defensible as written (NB-16). Only the **scope** of the
  two rows that use it is wrong.

## Non-blocking observations

**Re-checks the revision planner asked for.** All five hold; three needed correcting in my
direction, not the planner's.

1. **B17 confirmed, and it is worse than `-W error` suggests.** The claim does not depend on
   `-W error` at all — both forms raise a plain `ValueError` on numpy 2.4.6:

   ```
   $ .venv/bin/python -c "import numpy as np; print(len(np.array([1.0,2.0]) or ()))"
   ValueError The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
   $ .venv/bin/python -c "import numpy as np; print(len(np.array([]) or ()))"
   ValueError The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.
   $ .venv/bin/python -c "import numpy as np; print(len(np.array([5.0]) or ()))"
   1
   ```

   Note the third line: a **one-element** array is truthy, so `len(arr or ())` returns `1` and looks
   correct — which is how the idiom survives a casual read. `_vlabels`/`_clabels` are typed
   `np.ndarray | None` (`solvers.py:451-452`), so `int(arr.size)` with an explicit `None` guard is
   the right replacement, and `len(None or ())` is `0`, matching the guard's intent. **Accepted.**
2. **B18 confirmed.** The base `_build_file` guard is at `solvers.py:765-768` and reads
   `if io_api is not None and io_api not in FILE_IO_APIS: raise ValueError(f"Keyword argument
   \`io_api\` has to be one of {IO_APIS} or None")`, with `FILE_IO_APIS = ["lp", "lp-polars", "mps"]`
   (`:185`) and `IO_APIS = FILE_IO_APIS + ["direct"]` (`:186`). The override restates it faithfully
   (base binds `io_api = self.io_api` first; the override reads `self.io_api` — equivalent). The
   `test_non_supported_solver_io` reasoning is right: `Model.solve` validates before `from_name`, so
   the guard is a low-level-API contract preservation, not a red test. **Accepted.**
3. **The B3 line-number correction is right and my round-1 numbers were wrong.** Verified:
   `test/test_solvers.py:48` (`test_solver_instance_attached_after_solve`) and `:209`
   (`test_solver_close_releases_state`) are the bare `simple_model.solve(solver)` sites, both
   `@pytest.mark.parametrize("solver", sorted(set(solvers.licensed_solvers)))` and both asserting
   success. `:81` passes `io_api="direct"`; `:201`/`:231` are gurobi-specific. Round 1 cited `:46`
   and `:286` and said "three tests" — **that was wrong on both counts; there are exactly two bare
   sites.** The finding stands on `:48` and `:209`, which is what `FINAL.md` §1.3, §2.1.4 and V1-16
   now say.
4. **Name-keyed status maps confirmed, plus one thing the plan did not state.**
   `sol.get_termination_reason()` returns a **`str`** (`type(reason)= <class 'str'>`, pasted above)
   and `get_problem_category().name` is `"LP"` / `"MIP"`, so the two-map-by-category design is
   mechanically sound. New supporting measurement: `set_variable_types` is called
   **unconditionally** in §2.1.6, so I checked that an all-`'C'` array does not flip an LP into a
   MIP — it does not:

   ```
   mode= none   cat= LP  term= Optimal obj= -2.7999999944487737  dual= [-0.4 -0.2]
   mode= allC   cat= LP  term= Optimal obj= -2.7999999944487737  dual= [-0.4 -0.2]
   mode= allC_S cat= MIP term= FeasibleFound obj= -2.8
                dual RAISED AttributeError Attribute get_dual_solution is not supported for milp solution
                milp_stats= {'mip_gap': inf, 'solution_bound': -inf, ...}
   ```

   Two consequences worth adding to `FINAL.md` §3/§4 as prose (neither changes the design):
   (a) a **semi-continuous** model is `cat=MIP`, so `is_mip` is `True`, duals are correctly refused
   and the MILP map applies — a user should not expect duals from an otherwise-LP model that
   contains one `'S'` variable; (b) `get_milp_stats()` can legitimately return `mip_gap=inf` and
   `solution_bound=-inf`. V1-4's `not None` threshold is still satisfied, and `∓inf` is the
   *correct* bound when none was proven (`+inf` after M2's negation for `max`), so this is honest —
   but the row should say so, or a Phase-5 agent will read `-inf` as a bug.
5. **The split tolerance tier is defensible under I12/§3.6 — it is not a loophole.** The test the
   charter cares about is whether the number was chosen **independently** or reverse-engineered from
   a miss. `CUOPT_PRIMAL_ATOL_DEGENERATE = 1e-4` is cuOpt's own documented `absolute_primal_tolerance`
   default, stated up front, applied only to two named fixtures, with the reason (singular Hessian,
   components at a bound) in the row. That is a number with a rationale. Two caveats to record:
   * the margin is thin. The row says "covers the measured `5.687e-05` with 1.8× margin", but the
     quantity being bounded is the **difference from the oracle**: HiGHS returns
     `x = 4.99999949e-07` on this fixture, so `|Δx| = 5.637e-05` — still inside `1e-4`, margin
     1.77×. HiGHS itself reports `P-D objective error 9.999e-05` here, i.e. both solvers are only
     ~`1e-4`-accurate on this singular QP. The number is right; the arithmetic in the rationale
     should cite `|Δx|`, not cuOpt's raw `x`.
   * the frozen consequence must stay as written: a miss at `1e-4` is **report-and-stop**, not a
     second widening. `15-verification.md`'s interlock paragraph already says this.

**Other observations from this round.**

6. **Semi-continuous support verified end-to-end through the real translation path**, which the
   exploration had only done with raw `DataModel` calls. `M.lb` carries the *"on"* lower bound
   (not `0`), which is exactly cuOpt's `'S'` semantics, so the passthrough is correct with no extra
   code:

   ```
   vtypes= ['S' 'C']   lb= [5. 0.]   ub= [ 10. 100.]   c= [1. 0.]   model.type= MILP
   ```

   and on V1-7's own shapes (`x` semi-continuous in `[1,10]`, `max x`), built from `model.matrices`
   and mapped exactly as §2.1.6 specifies:

   ```
   C=0.5  cat= MIP  term= Optimal  obj= 0.0   primal= [0.]   milp_stats={'mip_gap': 0.0, 'solution_bound': 0.0}
          HiGHS: ok optimal obj= 0.0   x= 0.0
   C=5.0  cat= MIP  term= Optimal  obj= -5.0  primal= [5.]   milp_stats={'mip_gap': 0.0, 'solution_bound': -5.0}
          HiGHS: ok optimal obj= -5.0  x= 5.0
   ```

   C-R26 and V1-7 are sound. Note also `model.type == "MILP"` for a semi-continuous model, which is
   why §2.1.5's `model.type in ("MIQP","IQP")` guard also catches quadratic + semi-continuous —
   `FINAL.md` §2.1.5 states this; it is now measured.
7. **`cuda.bindings` cannot be missing when cuOpt is installed**, which narrows §11 risk 4.
   `cuda-bindings 12.9.7` is a hard transitive requirement of `cuopt-cu12` via three of its direct
   `==26.8.*` dependencies (`cudf-cu12`, `rmm-cu12`, `pylibraft-cu12`, each `cuda-bindings<13.0,>=12.9.2`).
   So `_CUDA_PROBE_SNIPPET`'s import cannot fail for a missing module; the remaining fail-closed
   scenarios are the two §11 risk 4 already names (no usable `sys.executable`, `subprocess`
   forbidden).
8. **Every import the plan relies on already exists**, so §2.1's "no new imports" claim holds:
   `contextlib` (`:8`), `functools` (`:10`), `subprocess as sub` (`:16`), `sys` (`:17`),
   `threading` (`:18`), `Callable` (`:22`). Only `csr_array` and `vstack` are added to the existing
   `from scipy.sparse import tril, triu` (`:34`), and only the pad row needs them.
9. **A pre-existing latent defect found while confirming B3, for `logs/45-known-issues.md` (I13).**
   `Model.get_problem_file(io_api=None)` returns a **`.lp`** path (`model.py:1878-1893`:
   `suffix = ".mps" if io_api == "mps" else ".lp"`), and `cuPDLPx._run_file` accepts **only**
   `.netcdf`, raising `NotImplementedError` otherwise (`solvers.py:4210-4214`). So a bare
   `model.solve("cupdlpx")` raises today, which means `test_solvers.py:48` and `:209` are red for
   cuPDLPx on any machine where it is installed. `FINAL.md` §2.1.4 says "this wart is invisible today
   only because cuPDLPx is not installed on this machine" — correct, and per I13 it should be
   **recorded** in `logs/45-known-issues.md` (not fixed). It is not currently on the plan's
   known-issues list. This is also the strongest independent argument for G1.
10. **§5.3's ID registry has no home.** The charter says new IDs are "recorded in
    `logs/30-decisions.md`", and `FINAL.md` §6 correctly notes it is the first place `C-R18+` has
    ever been written down. But §13's list of D-entries for the lead does **not** include the
    `C-R18`–`C-R29` registry. One line, and it is what makes "IDs are never renumbered or reused"
    enforceable by a later agent that never reads `FINAL.md`. Lead action, not a plan edit.
11. **`15-verification.md`'s header already reads "FROZEN at Gate 3"** while the file is not frozen.
    That is the template's own text and harmless, but it means the file's self-description is
    currently false — worth a glance when the digest is actually taken.
12. **WP-C's parallelism is authorised but its acceptance criterion is not fully reachable in
    parallel.** §6.4 rule 1 says the `solvers.py` core goes "first, alone, and merges before
    anything else starts"; rule 5 explicitly overrides that for packaging ("parallelises freely — run
    it early"). So WP-C ∥ WP-A is correct. But WP-C acceptance item 2 (`available_solvers` **contains**
    `cuopt`, = V3-1) cannot pass until WP-A has merged. `FINAL.md`'s Runs column already says
    "re-verified after A merges" — keep that sentence in the brief verbatim, because the first run
    will legitimately print `[]` (exactly as Q13 did) and an agent without that sentence will read it
    as a packaging failure.
13. **V3-4 proves the docs *build*, not the docs *content*.** Its threshold is "0 new warnings and 0
    new failures", which cannot detect a missing §9 row. Content presence lives only in WP-E's
    acceptance criteria ("every §9 row present; the mechanical PR-#516 gap check run"). That is
    concrete work for C-R15/C-R16, so Gate-3 check 1 passes — but Phase 5 will execute
    `15-verification.md`, not `FINAL.md` §7.2, so consider a V3-4b row with a mechanical count (e.g.
    occurrences of `cuopt` in each of the eight surfaces `≥ 1`).
14. **`_set_solver_params` mutates `self.solver_options` in place** (`log_file` written into the
    user's options dict), so a stale `log_file` survives into a second solve that passes no `log_fn`.
    `Highs._set_solver_params` does exactly the same (`:1774-1784`), so this is repo idiom, not a
    defect — recorded only so a Phase-5 reviewer does not raise it as new.
15. **The `_build_file` warning fires on the default happy path** — every bare `model.solve("cuopt")`
    logs it. Acceptable by precedent: `cuPDLPx._run_file` warns unconditionally on the same path
    (`solvers.py:4205-4207`, "cuPDLPx doesn't currently support file IO. Building model from file
    using linopy."). V1-16's "exactly `1` fallback warning" is therefore the right threshold; no
    change wanted.
16. **`test_quadratic_model_cross_terms` is a genuinely non-degenerate QP** and is the better fixture
    for a full primal differential: `min -2x + y + x²` s.t. `x + y ≥ 10` has the unique optimum
    `x=1.5`, `y=8.5`, `obj=77.5` (the row's measured `77.50000039` confirms it). Worth saying in
    V1-13b's rationale so the asymmetry between the two fixtures reads as deliberate.
17. **`assign_result` tolerates the empty `Solution`**, so V1-5b's "`0` exceptions raised" is sound
    even on the `Model.solve` path: `model.py:2195-2196` returns early on
    `result.solution is None or len(result.solution.primal) == 0`, and `:2206`
    (`if len(result.solution.dual):`) is the guard B8 relies on. Both verified by reading.
18. **`values_to_lookup_array` with *empty* labels is safe**, which is what §2.1.7's residual path
    needs (round 1's W6 measured only the *non-empty*-labels case):

    ```
    size 0  -> ndarray (0,)
    size 10 -> ndarray (10,) [nan nan nan]
    ```
19. **`Model.sense` is `self.objective.sense`** (`model.py:390-394`), so §2.1.6's
    `model.objective.sense == "max"` and `_solve`'s `sense == "max"` (from `self.sense`) are the same
    predicate — M2's `sign` cannot disagree between build and parse. And `M.c` is sense-agnostic:
    `cuPDLPx._build_solver_model` passes `M.c` unmodified and sets `ModelSense` separately
    (`:4270-4281`), confirming `sign * M.c` is the whole of the transform.
20. **Diff size.** `FINAL.md`'s `+605 / −5` over 13 files is 0.78× PR #516 and V3-5 caps it at
    `≤ 1157`. Round 1 asked for "say why" past ~+900; it did not go there. Nothing to raise.

## Checklist

- [x] **Every requirement ID in scope is covered** — all 29. `C-R1`–`C-R17` match charter §5's rows
      top-to-bottom exactly (I counted the table: 17 rows, `solvers.py`×6, `solver_capabilities.py`,
      `io.py`, `model.py`, `constants.py`, `conftest.py`, `test_optimization.py`, `test_solvers.py`,
      `pyproject.toml`, `gpu-acceleration.rst`, the docs-set row, and the QP `QUADRATIC_SOLVERS`
      row). `FINAL.md` §6 maps all 29 to work and a WP; every one of the 29 also appears in
      `logs/15-verification.md`'s "Proves (req ID)" column — I checked each ID individually, none is
      missing.
- [x] **No requirement silently dropped or reinterpreted** — the two that *could* have been are not:
      C-R7 and C-R11 are discharged as **zero-diff proof obligations** (V0-3/V0-4 and V0-8), which is
      what charter §5 explicitly asks for, and C-R17's `QUADRATIC_SOLVERS` row stays a read-only
      anchor with V0-9 pinning `0` changed lines. C-R12's `test_infeasible_model` *non*-skip is
      declared as a prediction with a named verification row (V2-5) rather than quietly assumed.
- [ ] **Frozen verification plan unchanged** — **not applicable, and deliberately not recorded.**
      Gate 3 has not been passed, so no digest exists yet in `logs/00-lead-journal.md` or in this
      file. `15-verification.md` must change to resolve B19; its digest today is
      `33066438f1522cdff72793f295a7474f39f1729387e22dec2a270288dad6ba49`, recorded here as
      *informational only* so the round-3 delta is visible. The Gate-3 digest is recorded in this
      file in the round that returns APPROVE.
- [x] **No threshold widened, no check downgraded to a weaker tier** — nothing in `FINAL.md` or
      `15-verification.md` relaxes an existing linopy threshold. `GPU_SOL_TOL = 2.5e-4` and
      `CPU_SOL_TOL` are inherited unchanged (§8.4, non-goal 12), the single pre-authorised widening
      (§8.5 item 2, V2-3) is scoped to three named assertions with a second widening explicitly
      refused, and `test_model_with_inf`'s exact equality is frozen with *"Authorised remedy if it
      misses: **none**"* (V2-4). The new degenerate tolerance tier is an additional, independently
      derived number, not a widening of an existing one (NB-5). B19 is the opposite failure — a row
      too *tight* for what it asserts — and the fix I ask for narrows the row's scope rather than
      loosening its number.
- [x] **Tests actually ran (GPU tests not silently skipped)** — as a *plan* property this is the
      thing round 1 pressed hardest on, and it is now structural: V0-5 pins the **positive** probe
      branch (with the reason `test_available_solvers.py:39` cannot catch a uniformly-false probe),
      V0-8 requires `--collect-only` **both** directions, V2-1 requires the paired
      `pytest` / `pytest --run-gpu` runs plus a `-k cuopt --run-gpu -v` listing showing `0` skipped
      cuOpt tests, and V0-7/V1-12 and V1-13c each require a **deliberate failure demonstrated once**.
      Every numeric claim in *this* review has its command output pasted above.
- [x] **Diff is surgical — no drive-by reformatting or refactoring** — five files are declared
      `±0` (`solver_capabilities.py`, `constants.py`, `conftest.py`, the `SolverFeature` enum, the
      `QUADRATIC_SOLVERS` block) with V0-9 asserting `0` changed lines in each; no new
      `SolverFeature` member; `accepted_io_apis` left undeclared; `CLAUDE.md` and
      `doc/contributing.rst`'s `uv sync` line untouched (I4/I9); and non-goal 13 refuses the three
      pre-existing defects noticed on the way. NB-9 adds a fourth for `45-known-issues.md`.
- [x] **Matches repo idiom (current `master`, not PR #516)** — anchors and idioms were re-read
      against `09c34dd`, not #516: `Solver.solve`'s dispatch (`:858`), the base `_build_file` guard
      (`:765-768`), `FILE_IO_APIS`/`IO_APIS` (`:185-186`), `Highs`'s in-`_solve` `CONDITION_MAP` and
      `contextlib.suppress` report block (`:1825-1892`), `cuPDLPx`'s override set and ClassVar order,
      `Model.solve`'s build-kwargs split and `finally` unlink (`model.py:2113-2151`),
      `assign_result`'s two guards (`:2195`, `:2206`), and `Model.sense` (`:390`). The one deliberate
      divergence from `cuPDLPx` (`is_available` probing the device) is D6, recorded as such with the
      downside kept. The commit-message style (§7.3: plain imperative, sentence case, no conventional
      prefix) contradicts recent `master` but is exactly what charter §6.4a directs and explicitly
      overrides I5 for — so it is correct as written, and §6.4a's instruction to log that in
      `logs/30-decisions.md` is a lead action.

## The four §6.3 Gate-3 conditions — confirmed on the record

Applied verbatim. Three pass outright; the second passes as *"every threshold is a number with a
rationale"* while B19 shows that one of those numbers is attached to the wrong quantity — which is a
scope defect, not a missing number.

1. **"Every `C-R*` ID maps to concrete work; no ID is unmapped or silently reinterpreted."**
   **CONFIRMED.** 29 IDs, 29 mappings in `FINAL.md` §6, 29 appearances in `logs/15-verification.md`.
   `C-R1`–`C-R17` correspond one-to-one to charter §5's 17 rows in order; `C-R18`–`C-R27` are
   plan-1's, adopted verbatim; `C-R28` (fork-safe probe) and `C-R29` (bare solve, no leak) are the
   two round 1 added, and both landed with real code *and* verification rows (V0-7/V1-12 and
   V1-16/V1-17). No ID is discharged by narration: the four that could only be "verified, not
   built" (C-R7, C-R11, C-R17's list, and the `SolverFeature` enum) each carry a zero-diff proof
   row. See NB-10 for the one bookkeeping gap (the registry is not in §13's D-entry list).
2. **"Every threshold in `logs/15-verification.md` is a number with a stated rationale — not a
   placeholder, not an adjective."** **CONFIRMED, with B19 against one row's scope.** I read all 44
   rows (V0-1…V0-11, V1-1…V1-18 incl. V1-5b/6a-d/13a-c, V2-1…V2-6, V3-1…V3-5). Every Threshold cell
   is a number, an exact-equality criterion, or an occurrence count, and every Rationale cell is
   non-empty and cites either an exploration item or a round-1 measurement — no adjectives, no
   "TBD", no "as appropriate". The eleven named tolerance constants are declared in one table with
   values. The interlock paragraph makes the whole table conditional on `method=3` and states that
   §5b's contingency forces a full re-derivation at Gate 3 — that dependency being written down is
   the strongest thing in the document. **B19 is not a missing number**: `1e-4` is a number with a
   rationale (NB-5); it is applied to a quantity the problem does not determine.
3. **"The V1 sign-convention matrix has all six cells filled."** **CONFIRMED, and exceeded.** All
   six `min`/`max` × `<=`/`>=`/`==` cells are filled with the *expected dual values*
   (`[∓0.4, ∓0.2]`), not just a sign word, on one shared model chosen so the primal optimum is
   unique and non-degenerate and both rows bind — each cell naming its test id. The matrix
   additionally carries the **presolve-solvable `max`** row that the exploration proved the six cells
   alone would miss (`max|Δdual| = 3.61` without M2) **and** a `get_solved_by() == SolverMethod.Unset`
   coverage-precondition row with a frozen docstring rule that a red there means re-establish
   coverage, never remove M2. Reduced costs are marked `n/a` in every cell with the reason
   (`−c − Aᵀy` upstream, recorded in `45-known-issues.md`), so the "not asserted" is a decision, not
   an omission.
4. **"The plan's non-goals are explicit, so scope creep is detectable later."** **CONFIRMED.**
   `FINAL.md` §10, sixteen numbered non-goals, each naming what is refused rather than gesturing at
   scope: file `io_api`s (with the four unmeasured contract points that justify it and a follow-up
   pointer), solution/basis files, warm start "in any form", reduced costs *including any new linopy
   surface for them*, quadratic constraints and MIQP, SOS/indicator, persistent updates, callbacks
   and `problem.Problem`/`routing`, remote cuOpt "including as a fallback on GPU-less hosts", `cu13`,
   tuning cuOpt's numerics other than `method`, touching `CPU_SOL_TOL`/`GPU_SOL_TOL` except the one
   pre-authorised contingency, fixing pre-existing defects (with three named refusals), benchmarks,
   the three zero-diff files, and rewriting cuOpt's error messages. Each is checkable against a diff.

## §6.4 / §6.4a sequencing — checked

* Rule 1 (`solvers.py` core first and alone, merged before anything else starts): **honoured** —
  WP-A owns `linopy/solvers.py` exclusively and runs "first, alone".
* Rule 2 (then parallelise the disjoint packages): **honoured** — B, C, D, E are pairwise disjoint;
  the file-collision paragraph enumerates the three overlaps and resolves each.
* Rule 3 (same file ⇒ sequential): **honoured** — A/F on `solvers.py`, D/F on `test_cuopt.py` and
  `test_solvers.py`, E/F on `gpu-acceleration.rst`; F runs after all of them.
* Rule 4 (QP last, alone, after Gate 4 green on LP/MILP): **honoured** — WP-F, "last, alone".
* Rule 5 (packaging first-class and early): **honoured** — WP-C is its own package with Q13's
  clean-venv acceptance test, running in parallel with A. See NB-12 on its acceptance ordering.
* §7.1 lineage discipline: **honoured, and better than the charter's floor** — every brief carries
  the base SHA, the `merge-base --is-ancestor` check is quoted, a **missing** SHA is also a stop, and
  the brief must end with §3.6 verbatim plus the "a red test reported honestly" sentence.
* §6.4a commit shape: **honoured** — five substantive commits mirroring #516's arc, a reserved sixth
  with the instruction *"do not manufacture a sixth"*, no merge commits, rebase + `--ff-only`, and
  per-agent `Co-Authored-By` authorship (I11). The "no conventional prefix" style is §6.4a's explicit
  override of I5, not a mistake (see the last checklist row).

## Round-1 findings — disposition audit

| Round-1 item | Verdict this round |
|---|---|
| B1, B2 (pad row / no `ValueError`) | **Resolved.** Singly-bounded pad from a variable's own bounds; `NotImplementedError` only for the all-free residual case, with the two feasible-set-changing alternatives rejected by name. V1-6a–d. |
| B3, B4 (bare solve, temp-file leak) | **Resolved**, and the line numbers corrected against me (NB-3). `_problem_fn` handback verified against `model.py:2140-2151`. |
| B5, B6, B7 (probe) | **Resolved.** Out-of-process, `functools.cache`d, `bool(count)`, fail-closed, D6's position kept with the downside recorded. Fork safety is a check proven to fire (V0-7). |
| B8 (empty MIP dual) | **Resolved**, and now doubly evidenced (NB-18 covers the empty-labels half round 1 had not measured). |
| B9 (option coercion) | **Resolved.** Blanket `bool → int`; allowlist, sniffing and retry all rejected with the measurement that kills each. |
| B10 (`test_quadratic_model_unbounded`) | **Resolved.** Disposition frozen in three places (§5j, §8.5 item 1, V2-2) with re-mapping and accept-list widening refused by name and the `safe_get_solution` reason given. |
| B11, B12 (feature flags) | **Resolved.** B12's "zero automatic coverage" consequence is drawn through to where the test lands (§8.1, V1-7) — and V1-7 is now measured end-to-end (NB-6). |
| B13 (`api.rst`) | **Resolved.** Both rows, with `to_netcdf` as the proof the autosummary entry builds, and the missing siblings recorded rather than fixed. |
| B14 (tight shared assertions) | **Resolved.** Both are named rows with pre-decided remedies, one of which is explicitly *nothing*. |
| B15, B16 | **Resolved.** No `method_options` row; `time_limit_option["cuopt"]` correctly identified as mandatory; three probe rows all able to fire. |
| G1–G10, R1–R13, NB 1–13 | **All applied / honoured / carried**, each with a section reference. I spot-checked G1, G2, G3, G5, G7, G10, R4, R5, R10, R11 against the text — all present and substantive, not just cited. |

## Instruction block for round 3

Small and mechanical. Nothing else in either document needs to move.

1. **`logs/15-verification.md` V1-13b** — scope the primal comparison to the `x` variables for
   `quadratic_model`; keep the full primal comparison for `quadratic_model_cross_terms` and say why
   the two differ (NB-16). Cite `|Δx|`, not cuOpt's raw `x`, in the tolerance rationale (NB-5).
2. **`logs/15-verification.md` V1-6b** — same restriction; do not compare `y`.
3. **`logs/15-verification.md` Nondeterminism table** — add the underdetermined `y` face of
   `quadratic_model` / `quadratic_model_unbounded`, with the measured `155.979` vs `9.9999995`, as a
   quantity never asserted against the oracle.
4. **`FINAL.md` §8.3 V1-13** — stop attributing the `0.5·M.Q` guard to the `quadratic_model`
   *solution* comparison; V1-13a and V1-13c carry it.
5. **Optional but cheap, and each already argued above:** the two prose additions in NB-4
   (semi-continuous ⇒ `cat=MIP` ⇒ no duals; `milp_stats` may be `±inf` and that is correct),
   NB-9's `45-known-issues.md` entry for cuPDLPx's bare-solve `NotImplementedError`, and NB-13's
   V3-4b docs-content count.
6. **Lead actions, not plan edits:** record the `C-R18`–`C-R29` registry as a D-entry (NB-10, charter
   §5.3), and §6.4a's commit-style override of I5.

Round 3 will re-read both documents in full and check only these items plus anything they disturb.
If items 1–4 land as described, the verdict is **APPROVE**, and that round records the
`sha256sum` of `logs/15-verification.md` in this file for the Gate-3 freeze.

---

# cuOpt support — plan review, round 3

*Reviewer: Phase 3 selector-reviewer (Opus 5, `claude-opus-5[1m]`) · 2026-08-22 · Reviewing:
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/FINAL.md` (1660 lines) and
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md` (197 lines), both re-read in
full from disk*

Basis: both documents in full; charter §§3.6, 5, 5.3, 6.3, 6.3a, 6.4, 6.4a, 9 and invariants
I4/I12/I13; rounds 1 and 2 of this file. Every claim of resolution below was checked against the
file, not against the planner's summary, and the one new threshold the revision introduced was
re-measured on the live stack before approving it.

## Verdict

**APPROVE.** Zero blocking findings stand.

**B19 is resolved correctly, and resolved by narrowing scope rather than by loosening a number** —
which is the distinction §3.6 turns on. Both rows now assert only quantities the problem determines;
the excluded quantity is documented in three places with its measured magnitude; and the tolerance
constant that was under discussion (`CUOPT_PRIMAL_ATOL_DEGENERATE = 1e-4`) is byte-for-byte
unchanged. Net across the revision the plan is **stricter**, not weaker: it *adds* an objective
comparison to V1-6b, *tightens* `quadratic_model_cross_terms` from `1e-4` to `1e-6`, and adds a whole
new mechanical row (V3-4b). I verified the one tightening by measurement rather than accepting it —
it passes with 8.2× and 27× margin (NB-1). The four §6.3 Gate-3 conditions are re-confirmed on the
record below. All four optional NB items from round 2 were taken, and all four landed as real
content rather than as a citation.

**Gate-3 freeze — digest recorded here as §6.3a requires.**

```
sha256sum /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md
31e7c45cd6a8fba832ff0aace1c56844d39fc45bdd019ee8079566536d504716
```

* **sha256:** `31e7c45cd6a8fba832ff0aace1c56844d39fc45bdd019ee8079566536d504716`
* **Recorded at:** `2026-08-22T20:34:50Z` (UTC)
* **File:** `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md`, 197 lines,
  38 650 bytes
* Superseded, for the record: the round-2 pre-B19 digest was
  `33066438f1522cdff72793f295a7474f39f1729387e22dec2a270288dad6ba49` — **never a freeze**, recorded
  in round 2 only so this delta would be visible. The digest above is the Gate-3 one.
* **Remaining lead actions to close Gate 3:** record this same digest and timestamp as a dated entry
  in `logs/00-lead-journal.md`, then `chmod 444` the file. Mode is currently `-rw-r--r--`. The
  Phase-5 reviewer recomputes and compares against **both** copies; a mismatch, or two recorded
  digests that disagree, is a blocking finding.

## Blocking findings

*None.* B19 is discharged; the round-2 blocker is the last one this review raised, and no new
blocking finding was found in round 3.

| # | Finding | Where (file:line) | Why it blocks | Suggested fix |
|---|---|---|---|---|
| — | *(none)* | — | — | — |

### B19 — verified resolved, claim by claim

Every one of the planner's six claims was checked against the file.

| Claim | Verified? | Evidence in the file |
|---|---|---|
| V1-13b scoped to objective + `x` block only | **Yes** | `15-verification.md:79` Threshold column now reads *"**`quadratic_model`:** objective `rtol=1e-6, atol=1e-6`; **`x` variables only** at `rtol=1e-6, atol=CUOPT_PRIMAL_ATOL_DEGENERATE=1e-4`. The 10 `y` variables are **not** compared against the oracle"*. |
| Rationale cites `\|Δx\| = 5.637e-05`, margin 1.77× | **Yes, and correctly** | `:79`: *"the bounded quantity is the **difference from the oracle**: cuOpt `5.68669e-05` vs HiGHS `4.99999949e-07` → `\|Δx\| = 5.637e-05`, inside `1e-4` with 1.77× margin"*, plus the note that HiGHS itself reports `P-D objective error 9.999e-05` here. That is exactly the correction NB-5 asked for: the bound is on the difference, not on cuOpt's raw `x`. |
| `quadratic_model_cross_terms` keeps the full primal comparison; `y` there carries `+1` cost, unique optimum | **Yes — and the *number* changed, which the summary did not mention** | `:79` gives cross-terms *"**full** primal (`x` and `y`) at `rtol=1e-6, atol=1e-6`"*. Round 2's row applied `atol=1e-4` to **both** fixtures, so this is a **tightening** from `1e-4` to `1e-6`. That is a new frozen threshold with no measurement stated, so I measured it rather than take it on trust — see NB-1. It passes. |
| V1-6b same restriction, with the pass-by-coincidence note | **Yes** | `:69`: *"**`x` variables only** compared against HiGHS at `atol=…1e-4`. **The 10 `y` variables are NOT compared against the oracle**"*, and *"`y` is excluded by construction, not by luck: … Both solvers happen to return `y = 0` **today** … so a `y` comparison would pass by coincidence on an underdetermined quantity and could go red on any solver update with nothing broken."* The row also **gained** an explicit `objective rtol=1e-6, atol=1e-6` comparison it did not have in round 2 (NB-3). |
| New Nondeterminism row for the free-`y` face | **Yes, and it is the right shape** | `:192`, last row of the table. It is honest about the category — *"Not solver nondeterminism but **model underdetermination**, which has the same consequence for a differential oracle"* — carries the measured `155.97919941` vs `9.9999995`, states that any version change may move it arbitrarily far, and closes with the rule that matters for later agents: *"A future row that compares `y` against HiGHS is a defect in the row, not in the solver."* |
| §8.3 split into 13a/13b/13c with the `0.5·M.Q` guard on 13a/13c, and `quadratic_model` stated as unable to carry it | **Yes, with the correct reason** | `FINAL.md:1439`: *"**V1-13b — fixture coverage, not convention coverage** … `quadratic_model` **cannot** catch the `0.5·M.Q` trap: its optimum is `x = 0` for *any* positive scaling of `Q`, so both encodings give the same `x`."* That is the right argument — stronger than the one I gave, which only said the row was mis-attributed. |
| WP-F acceptance mirrors the scope | **Yes** | `FINAL.md:1354-1357`: *"**scoped exactly as V1-13a/b/c in `logs/15-verification.md`** — in particular `quadratic_model`'s 10 `y` variables are underdetermined by the problem and are **never** compared against the oracle"*. §8.3's V1-6 row (`:1432`) carries the same scope. Both were needed: an agent reading `FINAL.md` alone can no longer rebuild the `y` comparison. |

## Non-blocking observations

1. **The one new threshold in this revision is `quadratic_model_cross_terms`' full primal at
   `atol=1e-6`, tightened from `1e-4`. Its row states why the fixture is *comparable* but not why
   `1e-6` is the right *number*, so I measured it. It passes with margin**, on the real fixture
   through `model.matrices`, `0.5·M.Q`, `method=3`:

   ```
   cuOpt term= Optimal obj= 77.50000038766152
   cuOpt x= [1.50000004 ...]      cuOpt y= [8.49999999 ...]
   HiGHS ok optimal obj= 77.50000000000122
   HiGHS x= [1.50000035 ...]      HiGHS y= [8.49999965 ...]
   max|dx|= 3.051790284924749e-07   max|dy|= 3.439451763398438e-07
   allclose(atol=1e-06, rtol=1e-06): x=True y=True
   tol budget at atol=1e-6,rtol=1e-6 for x=1.5: 2.500000349999965e-06
   tol budget at atol=1e-6,rtol=1e-6 for y=8.5: 9.499999650000035e-06
   ```

   Margins **8.2×** on `x` and **27×** on `y`. `1e-6` is also `CUOPT_PRIMAL_ATOL`, the plan's standard
   non-degenerate primal tolerance, whose derivation is given at V1-1 and V1-13a — so the number has
   a rationale by reference and now has a measurement behind it as well. **Not blocking**, and the
   fix if the planner wants one is a single clause in `:79` citing these two figures. Recording them
   here puts them inside the frozen record either way. Note the direction: the fixture whose
   Hessian is *singular in `y`* is nonetheless the accurate one, because `y` is pinned by a binding
   constraint plus a `+1` cost — which is the whole point of the asymmetry between the two rows.
2. **V3-4b is not a check that cannot fire.** I measured the baseline it will run against — every one
   of the eight surfaces contains cuOpt **zero** times today, and `platform_system == 'Linux'`
   appears **zero** times in `pyproject.toml`:

   ```
   doc/gpu-acceleration.rst : 0      doc/index.rst   : 0     doc/release_notes.rst        : 0
   doc/prerequisites.rst    : 0      README.md       : 0     examples/using-solvers.ipynb : 0
   doc/contributing.rst     : 0      doc/api.rst     : 0
   platform_system == 'Linux' in pyproject.toml : 0
   (for contrast: cupdlpx appears 14× in gpu-acceleration.rst, 1× in contributing.rst)
   ```

   So every `≥ 1` is a real obligation and the `exactly 1` on the marker is exact. The row's own
   boundary is drawn correctly too — it takes the countable half and leaves the qualitative
   "actually explains cuOpt" judgement with the Gate-6 reviewer against §6.6b.
3. **V1-6b silently gained a check**, which the planner's summary did not mention and which is worth
   recording as a strengthening: round 2's row had no objective-vs-oracle threshold (only
   `round(objective,3) == 0`); round 3 adds `objective rtol=1e-6, atol=1e-6` against live HiGHS.
   Measured safe by a very wide margin — cuOpt `7.5897e-15` vs HiGHS `0.0`.
4. **Excluding `y` from the oracle comparison does not remove `y` from coverage.** `test_quadratic_model`'s
   `y.round(3) >= 10` still runs against cuOpt under `--run-gpu` via V2-1, and the measured `155.979`
   satisfies it. So `y` is still exercised end-to-end; what was removed is a *point* comparison
   against a second solver, which is the only form the fixture cannot support. Worth knowing before
   Phase 5 reads the Nondeterminism row as a coverage hole.
5. **Line-number nit, mine to correct since I introduced the citation.** Both files cite
   `test_quadratic_model:876` for the `y.round(3) >= 10` assertion; in the file `:876` is
   `assert (quadratic_model.solution.x.round(3) == 0).all()` and `:877` is the `y` inequality
   (round 2 of this review said `:877`). Cosmetic, affects nothing, and **not** worth reopening
   Gate 3 for — noted so a Phase-5 agent that follows the pointer is not confused.
6. **The four optional round-2 items all landed as content, not as citations.** Spot-checked each:
   the semi-continuous ⇒ `cat=MIP` consequence is in §3's `SEMI_CONTINUOUS_VARIABLES` row
   (`FINAL.md:822`) *with* the measured contrast (all-`'C'` leaves `cat=LP`; one `'S'` gives
   `cat=MIP FeasibleFound` and `get_dual_solution()` then raises) and the `model.type == "MILP"`
   consequence for the MIQP guard; the `±inf` correctness note is in **both** §2.1.7's report
   paragraph (`:509-513`, with *"must not be filtered, clamped or replaced with `None`"*) and V1-4's
   rationale; the cuPDLPx bare-solve defect is non-goal 13 (`:1533-1541`) as an **entry to add when
   WP-A lands**, framed record-don't-fix with the two file anchors and the note that it is the
   strongest independent argument for the `_build_file` graft; and NB-7's dependency finding
   rewrote §11 risk 4 to say those two scenarios *"are now the **only** fail-closed scenarios"*
   (`:1569-1575`).
7. **NB-11 and NB-12 also landed.** `15-verification.md:5` is a new revision-state line that fixes
   the header's premature "FROZEN" self-description — *"**Not yet frozen:** the lead records the
   sha256 in both logs and applies `chmod 444` at Gate 3"* — and carries the superseded round-2
   digest. `FINAL.md:1334-1340` turns NB-12 into an **"Ordering caveat that MUST be carried into
   WP-C's brief verbatim"**, ending with the sentence that matters: *"Without this sentence an agent
   will read `[]` as a packaging failure and start debugging the extra."*
8. **NB-10 landed as §13 item 0** (`FINAL.md:1644-1649`), a D-entry listing all twelve new IDs and
   their topics, with the reason: it makes "IDs are never renumbered or reused" enforceable by an
   agent that never reads `FINAL.md`. It also correctly folds in §6.4a's commit-style override of I5
   as *"confirm rather than duplicate"* against the existing D2.
9. **Nothing regressed.** §12's index still carries all nineteen findings
   (`B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19`) plus the
   grafts/rejections/non-blocking carry-forward lines; §10 still has exactly 16 non-goals; the
   §6 requirement map and `15-verification.md` still each mention all 29 IDs
   (`C-R1`…`C-R29`, checked mechanically); the tolerance-constants table is unchanged in all eleven
   rows; the sign-convention matrix, the status-mapping table and the interlock paragraph are
   byte-identical to round 2.

## Checklist

- [x] **Every requirement ID in scope is covered** — verified mechanically this round, not by
      re-reading: `grep -o "C-R[0-9]\+" | sort -u -V` yields `C-R1 … C-R29` with no gaps in
      **both** `FINAL.md` and `logs/15-verification.md`. `C-R1`–`C-R17` still correspond
      one-to-one to charter §5's 17 rows in order; `C-R15`/`C-R16` gained a second, mechanical row
      (V3-4b) this round, which is a strengthening of the weakest mapping in the set (round-2 NB-13).
- [x] **No requirement silently dropped or reinterpreted** — the B19 fix removes an *assertion*, not
      a *requirement*: C-R17 and C-R21 keep their rows, keep their oracle, and keep a
      primal-versus-HiGHS comparison; only the `y` block is excluded, and the exclusion is stated in
      the row, in the Nondeterminism table, in §8.3, in §7.2's WP-F criterion and in §12 — five
      places, none of them silent. C-R19's Q coverage moved to V1-13a/V1-13c, which is where it
      always belonged.
- [x] **Frozen verification plan** — this is the round that freezes it. `sha256` recorded above with
      its UTC timestamp; `git diff` cannot check this file (`dev-scripts/` is gitignored, so it is
      untracked and always shows empty), which is exactly why the digest exists. The remaining half
      of §6.3a — the same digest in `logs/00-lead-journal.md` plus `chmod 444` — is a lead action,
      flagged in the Verdict.
- [x] **No threshold widened, no check downgraded to a weaker tier** — checked **row by row** against
      round 2, and this is the load-bearing row of this review, so the reasoning is explicit:
      * the eleven tolerance **constants** are unchanged, `CUOPT_PRIMAL_ATOL_DEGENERATE = 1e-4`
        included;
      * V0-1…V0-11, V1-1…V1-5b, V1-6a, V1-6c, V1-6d, V1-7…V1-13a, V1-13c…V1-18, V2-1…V2-6,
        V3-1…V3-4, V3-5, the sign matrix, the status map and the interlock are **unchanged**;
      * **V1-4** gained the `±inf` clarification but its threshold is still `is not None` — and
        `inf is not None` was already true under the round-2 wording, so nothing was loosened; the
        note prevents a *misreading*, it does not admit a value the threshold previously excluded;
      * **V1-6b** and **V1-13b** narrowed *scope*, kept `1e-4`, and each **added** an objective
        comparison / a tightened primal comparison. `quadratic_model_cross_terms` went `1e-4 → 1e-6`,
        i.e. 100× **stricter**, measured safe at 8.2×/27× margin (NB-1);
      * **V3-4b** is purely additive and non-vacuous (NB-2).
      On whether the `y` narrowing is itself a "downgrade": I12/§3.6 forbids an agent weakening a
      check **so its own work passes**. This narrowing (a) followed no failing run — the row had never
      been executed; (b) rests on the fixture's algebra, not on a result; (c) was ordered by the
      reviewer **at Gate 3**, which is the charter's own legitimate path for changing a threshold
      ("Changing a threshold means returning to Gate 3"); and (d) removes coverage of a quantity the
      problem does not determine, i.e. no real coverage. A check that would go red at random is not
      a weaker check, it is not a check.
- [x] **Tests actually ran (GPU tests not silently skipped)** — unchanged from round 2 and still the
      strongest part of the plan: V0-5 pins the positive probe branch, V0-6/V1-11 the negative,
      V0-7/V1-12 fork safety **with the naive variant run once to prove the check fires**, V0-8 the
      `--collect-only` count in both directions, V2-1 the paired `pytest` / `pytest --run-gpu` runs
      plus a `-k cuopt --run-gpu -v` listing showing `0` skipped cuOpt tests, and V1-13c the
      deliberate Q-convention failure. Every numeric claim in *this* review has its command output
      pasted.
- [x] **Diff is surgical — no drive-by reformatting or refactoring** — unchanged: five `±0` files with
      V0-9 asserting `0` changed lines in each, no new `SolverFeature` member, `accepted_io_apis`
      undeclared, `CLAUDE.md` and `doc/contributing.rst`'s `uv sync` line untouched (I4/I9). Non-goal
      13 now names **seven** refused pre-existing defects, up from three — including NB-9's cuPDLPx
      bare-solve wart, correctly queued as record-don't-fix rather than fixed in passing (I13).
      Estimated volume is still `+605 / −5` over 13 files, 0.78× PR #516, capped by V3-5 at `≤ 1157`.
- [x] **Matches repo idiom (current `master`, not PR #516)** — nothing in this round's edits touches
      the code plan; the anchors re-verified in round 2 against `09c34dd` still hold. The one
      deliberate divergence (`is_available` probing the device) remains D6 with the downside
      recorded, and §7.3's commit style remains §6.4a's explicit override of I5, now scheduled for
      confirmation as a D-entry.

## The four §6.3 Gate-3 conditions — re-confirmed on the record

1. **"Every `C-R*` ID maps to concrete work; no ID is unmapped or silently reinterpreted."**
   **CONFIRMED.** 29 IDs; verified mechanically in both files this round (`C-R1 … C-R29`, no gaps,
   no duplicates, none renumbered). `FINAL.md` §6 gives each a concrete deliverable and a work
   package; every one also carries at least one row in `logs/15-verification.md`. The four that can
   only be *verified* rather than built (C-R7, C-R11, the `SolverFeature` enum under C-R1, and
   C-R17's `QUADRATIC_SOLVERS` list) each have a zero-diff proof row (V0-3/V0-4, V0-8, V0-9). The
   weakest mapping in round 2 — C-R15/C-R16 resting on a docs *build* — now has a mechanical content
   row (V3-4b) whose baseline I measured at `0` on all eight surfaces, so it can fire. §13 item 0
   closes §5.3's bookkeeping requirement.
2. **"Every threshold in `logs/15-verification.md` is a number with a stated rationale — not a
   placeholder, not an adjective."** **CONFIRMED.** All **45** rows read (V0-1…V0-11, V1-1…V1-18
   including V1-5b/6a-d/13a-c, V2-1…V2-6, V3-1…V3-4, V3-4b, V3-5). Every Threshold cell is a number,
   an exact-equality criterion, or an occurrence count; every Rationale cell is non-empty and cites
   an exploration item or a named measurement. Eleven tolerance constants are declared once with
   values and referenced by name. The `method=3` interlock makes the whole table conditional on the
   default method and states that §5b's contingency forces a full re-derivation at Gate 3 — the
   single most valuable sentence in the document. Round 2's one gap in this condition (a number
   attached to an undetermined quantity) is closed; round 3's only new number,
   `quadratic_model_cross_terms` at `1e-6`, has a rationale by reference to `CUOPT_PRIMAL_ATOL` and
   is now measured to pass with 8.2×/27× margin (NB-1).
3. **"The V1 sign-convention matrix has all six cells filled."** **CONFIRMED**, byte-identical to
   round 2 and still exceeding the requirement: all six `min`/`max` × `<=`/`>=`/`==` cells carry the
   *expected dual values* (`[∓0.4, ∓0.2]`) on one shared model with a unique non-degenerate dual and
   both rows binding, each naming its test id; **plus** the presolve-solvable `max` row the
   exploration proved the six cells alone would miss (`max|Δdual| = 3.61` without M2), **plus** the
   `get_solved_by() == SolverMethod.Unset` coverage-precondition row with the frozen docstring rule
   that a red there means re-establish coverage and never means remove M2. Reduced costs are `n/a`
   in every cell with the upstream reason, so the omission is a decision.
4. **"The plan's non-goals are explicit, so scope creep is detectable later."** **CONFIRMED.**
   §10, sixteen numbered non-goals, each naming what is refused: file `io_api`s (with the four
   unmeasured contract points behind the refusal), solution/basis files, warm start in any form,
   reduced costs including any new linopy surface, quadratic constraints and MIQP, SOS/indicator,
   persistent updates, callbacks and `problem.Problem`/`routing`, remote cuOpt "including as a
   fallback on GPU-less hosts", `cu13`, tuning cuOpt's numerics other than `method`, touching
   `CPU_SOL_TOL`/`GPU_SOL_TOL` except the one pre-authorised contingency, fixing pre-existing
   defects (now **seven** named refusals), benchmarks, the three zero-diff files, and rewriting
   cuOpt's error messages. Each is checkable against a diff.

## Gate 3 — what the lead still has to do

The plan side is approved and the digest is recorded here. To close Gate 3 per §6.3a:

1. Record `31e7c45cd6a8fba832ff0aace1c56844d39fc45bdd019ee8079566536d504716` and the timestamp
   `2026-08-22T20:34:50Z` as a dated entry in `logs/00-lead-journal.md` — a **new** entry, never by
   overwriting.
2. `chmod 444 /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md` (currently
   `-rw-r--r--`).
3. Record the D-entries in `FINAL.md` §13 — **item 0 first**, the `C-R18`–`C-R29` registry, since
   later agents depend on it and it is the one entry whose absence is not self-correcting.
4. Carry into every slice brief: the expected base SHA plus §6.4's lineage check, the §3.6 threshold
   paragraph **verbatim**, the "a red test reported honestly is acceptable work, a hidden one is
   not" sentence, and — for WP-C only — §7.2's ordering-caveat paragraph verbatim.

Any later edit to `logs/15-verification.md` means returning to Gate 3: `chmod 644`, revise,
re-approve, re-hash, and record the new digest as a **new** dated entry in both logs.

*Selector-reviewer role closed for Phase 3 at round 3 of 5 (charter §6.3 cap). Three rounds, one
selection and two review iterations; nineteen blocking findings raised and nineteen resolved.*

---

# cuOpt support — plan review, round 4 (scoped Gate-3 return, V0-2 only)

*Reviewer: Phase 3 selector-reviewer (Opus 5, `claude-opus-5[1m]`) · 2026-08-22 · Reviewing:
the **V0-2 row only** of `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md`
(now 199 lines, 40 994 bytes, mode `-rw-r--r--`). The remainder of the file carries the round-3
approval and was **not** re-reviewed — it was proved unchanged instead (see §"No other row changed").*

Scope, per the lead's scoped-return brief: (1) is the V0-2 edit a scope fix rather than a widening;
(2) independent verification of the 28-error baseline; (3) proof that nothing else moved; (4) verdict
and a fresh digest. Nothing outside the V0-2 change was reviewed.

## Verdict

**APPROVE.** Zero blocking findings.

The V0-2 edit is a **scope fix, and the quality bar is strengthened, not widened.** The old threshold
was not a stricter bar — it was an **unreachable** one, and an unreachable check discriminates exactly
as poorly as one that always passes: it can never distinguish a clean slice from a dirty one, because
it is red either way. The new threshold is the first version of this row that can actually **fail on a
real defect and pass on a clean slice**. I verified the baseline myself on a fresh detached worktree of
untouched `master` @ `09c34dd` (`Found 28 errors in 1 file`, exit `1`, all inside `Xpress`), verified
that V0-2's own comparison passes as specified on the merged slice (`0` symmetric difference, and
identical as multisets too), and proved byte-for-byte that no other row, heading, threshold or
character in the file changed.

Two non-blocking corrections are recorded below — one in the row's rationale (a line number quoted
from the wrong tree), one in the *argument* offered for the row (message-set equality is **not**
strictly stronger than a count; the row is right precisely because it requires **both**). Neither
touches a threshold. A third correction is **mine**: my round-2 and round-3 reviews miscounted the
number of check rows, and the lead's brief inherited that miscount. Reported rather than smoothed
over, per §3.6.

**New digest for the lead to record and freeze:**

```
sha256sum /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md
a6e93a73f9c51b9ffcea9705f39173f07fae7ba93c72783f10b88ea9773cbca7
```

* **sha256:** `a6e93a73f9c51b9ffcea9705f39173f07fae7ba93c72783f10b88ea9773cbca7`
* **Computed at:** `2026-08-22T21:22:11Z` (UTC) · 199 lines · 40 994 bytes · mode `-rw-r--r--`
* **Supersedes:** `31e7c45cd6a8fba832ff0aace1c56844d39fc45bdd019ee8079566536d504716` (round 3,
  `2026-08-22T20:34:50Z`), which remains the digest of record for rounds 3→4 and must be kept in
  `logs/00-lead-journal.md` as a **superseded** entry, not overwritten (§6.3a: "never by overwriting
  the old one").
* **Lead actions to re-close Gate 3:** record the digest above with its timestamp as a **new** dated
  entry in `logs/00-lead-journal.md`, then `chmod 444`. If the lead chooses to take NB-1's one-token
  correction, do it **before** hashing — it is free at that moment and costs a full Gate-3 return
  afterwards.

## Blocking findings

*None.*

| # | Finding | Where (file:line) | Why it blocks | Suggested fix |
|---|---|---|---|---|
| — | *(none)* | — | — | — |

## Scope check — is the new V0-2 a fix or a widening?

**A fix.** The reasoning, in the order it has to hold:

1. **The old threshold asserted a false precondition, not a strict standard.** `exit code 0, 0 errors`
   is unsatisfiable on *untouched `master`* — so it was unsatisfiable on every branch cut from it,
   including a perfect one. Under I12/§3.6 the only honest response to an unreachable threshold is
   "report and stop", forever, on every work package. That is not a high bar; it is a broken
   instrument. Replacing it is exactly what a Gate-3 return is for, and the lead took the correct
   route (return to Gate 3) rather than letting the slice widen it in place — which is the behaviour
   §3.6 exists to force. **The process here is the thing that worked.**
2. **The specific concern the old rationale existed to protect is preserved verbatim, and I checked
   that it still bites.** The row still says: `warn_unused_ignores = true`, so a redundant
   `# type: ignore` on the `_LazyModule` line (FINAL.md §2.1.2) is itself an error. Under the new
   threshold such an ignore produces a **new message** (`unused "type: ignore" comment`) that is not
   in `master`'s set → symmetric difference ≠ 0 → **miss**, *and* pushes the count to 29 → **miss**.
   Both halves fire. The mechanism changed; the coverage did not.
3. **The new threshold detects strictly more than "no new errors in the diff".** It is scoped to the
   whole `linopy` package, not to touched files, so a new error introduced *anywhere* — including in
   a file the slice never opened, e.g. via a changed import or a widened `_LazyModule` handle — is a
   miss. That is a stronger obligation than a diff-scoped lint.
4. **The two exclusions are correct and carry no signal.**
   * **Exit code**: verified `mypy exit=1` with 28 errors on untouched `master`. mypy exits `1`
     while any error remains, so `exit code 0` is not an independent check — it is a restatement of
     `0 errors`, i.e. of the unreachable bar. Excluding it removes a redundancy, not a check.
   * **`checked N source files`**: verified to differ for a reason with nothing to do with type
     safety — `29` in the fresh worktree, `30` in the developed checkout, and the difference is
     exactly the setuptools-scm–generated `linopy/version.py`, which exists in the checkout and not
     in a fresh worktree:

     ```
     $ ls dev-scripts/cuopt/worktrees/verify-r4/linopy/version.py
     ls: cannot access '.../verify-r4/linopy/version.py': No such file or directory
     $ ls /home/azureuser/linopy-gpu/linopy/version.py
     /home/azureuser/linopy-gpu/linopy/version.py
     ```

     Pinning that count would have made the row fail or pass on whether an untracked generated file
     happened to exist — a check that fires on the wrong variable. Excluding it is right.
5. **It says what a miss is and forbids the escape.** The row ends: *"Any NEW error anywhere, and any
   increase above 28, is a **miss** — report verbatim with the actual count and the new message(s),
   and stop on this item"*, plus *"If KI3 is ever fixed upstream, the baseline must be **re-measured**
   on the work package's base commit, not assumed — a changed baseline is a return to Gate 3, never a
   silent edit."* That second sentence is the one that keeps `28` from decaying into a number nobody
   re-derives. It is the right closing clause and I would have asked for it if it were missing.
6. **The 28 are legitimately baseline, not swept under a rug.** They are pre-existing on untouched
   `master`, recorded as **KI3** in `logs/45-known-issues.md` and **B11** in `logs/40-blockers.md`,
   and under **I13** ("record, don't fix") they must *not* be repaired in this PR — fixing 28
   `Xpress` stub mismatches would be the drive-by change I4 forbids and would swamp the diff. Keeping
   them in the baseline is the only disposition consistent with both invariants.

## Independent verification of the baseline (§6.3a two-independent-copies spirit)

Measured myself, on a **fresh detached worktree** of untouched `master` @ `09c34dd` — created for
this check and removed afterwards, no branch left behind:

```
$ git worktree add --detach dev-scripts/cuopt/worktrees/verify-r4 09c34dd
Preparing worktree (detached HEAD 09c34dd)
HEAD is now at 09c34dd fix: _as_index() calls to_index() for DataArrays (#897)
$ git rev-parse --short HEAD
09c34dd
$ git status --porcelain          # (empty)
$ /home/azureuser/linopy-gpu/.venv/bin/mypy linopy
linopy/solvers.py:2727: error: Module has no attribute "maximize"  [attr-defined]
linopy/solvers.py:2729: error: Module has no attribute "minimize"  [attr-defined]
linopy/solvers.py:2752: error: Argument 1 to "enter_context" of "_BaseExitStack" has incompatible type "problem"; expected "AbstractContextManager[Never, bool | None]"  [arg-type]
...
linopy/solvers.py:2906: error: "problem" has no attribute "chgobjsense"; maybe "chgObjSense"?  [attr-defined]
...
linopy/solvers.py:3077: error: "problem" has no attribute "getDual"; maybe "getDuals"?  [attr-defined]
Found 28 errors in 1 file (checked 29 source files)
mypy exit=1
```

Accounting, so the number in the threshold is unambiguous:

```
lines with ': error:' = 28        <- this is the 28
lines with ': note:'  = 1         <- a var-annotated hint at :2975, NOT an error line
total output lines    = 30        <- 28 errors + 1 note + 1 summary line
min/max error line numbers: 2727 / 3077
class Xpress at linopy/solvers.py:2613 ; next class Knitro at :3111
```

**Confirmed, matching the row exactly:** `28`; exit `1`; all 28 error lines within `2727-3077`, i.e.
strictly inside the `Xpress` class (`2613 < 2727` and `3077 < 3111`); `checked 29 source files` in a
fresh worktree. The `Xpress`-only localisation is not incidental — it is what makes the baseline
*safe* to freeze: the 28 live in a class this PR never touches, so a slice cannot accidentally inherit
credit for changing one.

**I also ran V0-2's own comparison, as specified,** against the merged slice `a2db13f` — because a
threshold nobody has executed is a threshold nobody has tested:

```
$ .venv/bin/mypy linopy            # on feat/cuopt-solver @ a2db13f
Found 28 errors in 1 file (checked 30 source files)
error lines = 28
$ # V0-2's check: strip line numbers, compare the error-message sets
master unique msgs: 24 / branch unique msgs: 24
symmetric difference:
(empty - identical)
multiset diff (with duplicates):
(empty - identical as multisets)
```

So the row **passes as written** on the slice it was rewritten for, on both the set comparison it
demands and the stricter multiset comparison it does not. The slice's "zero new errors" claim is
independently confirmed. `git status --porcelain` stayed empty throughout (`.mypy_cache` is
gitignored at `.gitignore:4`).

## No other row changed — proved, not spot-checked

Rather than re-read the file, I proved byte-identity. I reconstructed the round-3 revision from the
current file by reverting **only** the two claimed edits, and hashed it:

```
current file:  40994 bytes  sha256 a6e93a73f9c51b9ffcea9705f39173f07fae7ba93c72783f10b88ea9773cbca7
  - delete the 2-line "Revision state (round 4)" note (file lines 5-6)
  - restore the round-3 V0-2 row verbatim (file line 52)
rebuilt file:  38650 bytes  sha256 31e7c45cd6a8fba832ff0aace1c56844d39fc45bdd019ee8079566536d504716
MATCHES ROUND-3 APPROVED DIGEST: True
```

The rebuilt file is **byte-for-byte** the artefact I approved at round 3. That is conclusive: not a
heading, not a threshold, not a tolerance constant, not a whitespace character outside those two
edits was touched — no spot-check could establish that, and a hash collision is the only alternative
explanation. `diff` against the verified reconstruction shows exactly two hunks and nothing else:

```
4a5,6
> *Revision state (round 4, 2026-08-22): scoped Gate-3 return — **V0-2 only**. …
>
52c54
< | V0-2 | V0 | C-R1–C-R6, C-R8, C-R9 | `.venv/bin/mypy` | exit code `0`, `0` errors | …
---
> | V0-2 | V0 | C-R1–C-R6, C-R8, C-R9 | `.venv/bin/mypy linopy`, compared against the sa…
```

Belt-and-braces structural counts on the current file, all matching the reconstruction:

| Property | Round-3 reconstruction | Round-4 current | Same? |
|---|---|---|---|
| `## ` headings | 4 (`Checks`, `Sign-convention matrix`, `Status mapping`, `Nondeterminism`) | 4, byte-identical | yes |
| check rows | 47 | 47 | yes |
| unique `C-R` IDs | `C-R1`…`C-R29` (29) | `C-R1`…`C-R29` (29) | yes |
| tolerance-constant rows | 11 | 11 | yes |
| sign-matrix rows | 8 (6 cells + presolve-`max` + coverage precondition) | 8 | yes |
| status-map rows | 17 | 17 | yes |
| Nondeterminism rows | 10 | 10 | yes |
| V0-2's `Proves` column | `C-R1–C-R6, C-R8, C-R9` | `C-R1–C-R6, C-R8, C-R9` | yes |

The template headings are intact and V0-2 still maps the same eight IDs, so no ID lost coverage in
the rescope.

## Non-blocking observations

1. **The row quotes a line number from the wrong tree.** V0-2's rationale says its measurement was
   taken *"on a clean worktree of untouched `master` @ `09c34dd`"* and then illustrates it with
   ``:2985 "problem" has no attribute "chgobjsense"; maybe "chgObjSense"?``. On untouched `master`
   that error is at **`:2906`**, and **`:2985` is not an error line on `master` at all**
   (`grep -c ':2985:'` → `0`). `:2985` is the **post-slice** line number: on `a2db13f` the same error
   *is* at `:2985`, because WP-AB inserted ≈79 lines above the `Xpress` class (the `_LazyModule`
   handle, the device probe and the interrupt helper all land before `:315`), shifting it
   `2906 → 2985`. Verbatim, both sides:

   ```
   master   @ 09c34dd : linopy/solvers.py:2906: error: "problem" has no attribute "chgobjsense"; maybe "chgObjSense"?  [attr-defined]
   branch   @ a2db13f : linopy/solvers.py:2985: error: "problem" has no attribute "chgobjsense"; maybe "chgObjSense"?  [attr-defined]
   ```

   **Why this is not blocking:** every load-bearing number in that sentence is correct for `master`
   and independently confirmed above — `28`, `2727-3077`, `class Xpress` at `:2613`, `Knitro` at
   `:3111`, exit `1`, `checked 29 source files`. And the threshold itself **strips line numbers by
   construction**, so no check depends on the token. What it is: one illustrative citation copied
   from the slice's report instead of from the master run. **Recommendation:** change `:2985` → `:2906`
   before the lead hashes — one token, free now, a full Gate-3 return later. If the lead prefers not
   to re-edit, this entry is the correction of record and Phase 5 reads it alongside the plan.
2. **"Message-set equality is strictly stronger than a count" is not right as stated — and the row is
   correct anyway, because it requires both.** The claim fails in one direction: a *duplicated*
   message raises the count while leaving the unique set unchanged, so set-equality alone would miss
   it. That is not hypothetical for this file — `master` has **28 error lines but only 24 unique
   messages**, i.e. four are duplicates (`Module has no attribute "maximize"` appears three times, at
   `:2727`, `:2904`, `:2906`), so a regression that multiplies an existing message is exactly the
   shape this file already exhibits. Conversely a *substitution* (one `Xpress` error disappears, one
   new cuOpt error appears) keeps the count at 28 and is caught only by the set. So neither test
   dominates the other; the row's `AND` is **load-bearing, not belt-and-braces**, and it should not
   be "simplified" later to whichever half looks sufficient. The threshold as written is right; only
   the justification offered for it over-claims. I verified both comparisons pass on `a2db13f`,
   including the multiset form.
3. **Implementation note for whoever executes V0-2 in Phase 5.** The `28` counts lines matching
   `: error:`. Raw output is **30** lines (28 errors + one `: note:` continuation for the
   `var-annotated` hint at `:2975` + the `Found …` summary). A naive `wc -l` or a `grep -c error`
   over the whole output will not yield 28, and the message-set comparison should either filter to
   `: error:` lines on both sides or include the `note:` line consistently on both. Stated because
   the row gives the number but not the extraction rule, and an executor who gets 30 will think it
   has found a regression.
4. **My own miscount, corrected — and it did not come from any change to the file.** My round-3
   review said *"All **45** rows read"* and round 2 said *"all 44 rows"*; the true count is **47**
   (V0: 11, V1: 24, V2: 6, V3: 6 — the V1 tier carries the sub-lettered `V1-5b`, `V1-6a`–`V1-6d`,
   `V1-13a`–`V1-13c`, which is where I dropped two). The lead's round-4 brief asked me to confirm
   "45 check rows still present", inheriting my error. To be unambiguous: the count is **unchanged**
   between round 3 and round 4 — I ran the same count over the digest-verified round-3
   reconstruction and got **47** there too. So nothing was added or removed; only my report of the
   number was wrong, in rounds 2 and 3, and this entry is the correction. Reporting it rather than
   quietly using the right number, per §3.6.
5. **The round-4 note is correctly worded** and does what NB-11 of round 2 asked of the round-3 note:
   it states the file is **unfrozen** (`chmod 644` by the lead), names what must happen before it is
   frozen again, and carries the superseded round-3 digest inside the file itself. Combined with the
   journal entries, the digest chain `3306…ba49` (never a freeze) → `31e7…ba49`… → `a6e9…bca7` is now
   reconstructible from the artefact alone.
6. **Housekeeping from my own verification.** The worktree I created was removed
   (`git worktree remove`), `--detach` left **no** branch behind, `git worktree list` and `git branch -v`
   are back to the four work-package entries, and `git status --porcelain` is empty. The only file
   this review wrote is `logs/50-review.md`.

## Checklist

- [x] **Every requirement ID in scope is covered** — V0-2's `Proves` column is unchanged
      (`C-R1–C-R6, C-R8, C-R9`), and the file still carries `C-R1`…`C-R29` with no gaps. The rescope
      changed *how* the row is measured, not *what* it proves, so no ID lost coverage.
- [x] **No requirement silently dropped or reinterpreted** — nothing silent about it: a scoped Gate-3
      return opened by the lead, a two-line in-file revision note, a rewritten rationale that states
      what changed and why, KI3/B11 cross-references, and this review. C-R2/C-R3's own protections are
      untouched — V0-11 (the `cuopt.routing` / `TYPE_CHECKING` row) is byte-identical.
- [x] **Frozen verification plan** — the file was legitimately unfrozen for this scoped return
      (`chmod 644`, currently `-rw-r--r--`) and the round-3 digest it supersedes is recorded both in
      this file (round 3) and inside the artefact's own note. New digest computed above; re-freezing
      is the lead's remaining action. I verified the digest is stable and that writing this review
      does not perturb it (`15-verification.md` unchanged before and after).
- [x] **No threshold widened, no check downgraded to a weaker tier** — this is the row of this review
      and the answer is no, with the argument in §"Scope check": the previous threshold was
      *unreachable*, so it discriminated nothing; the replacement is package-wide, catches any new
      message and any count increase, still bites on the `warn_unused_ignores` case the original
      rationale was written to protect, and excludes only two quantities I verified carry no signal
      (exit code, which is a restatement of `0 errors`; and `checked N source files`, which tracks an
      untracked generated file). Tier is unchanged (**V0**). Every other threshold in the file is
      byte-identical to the round-3 version, proved by hash rather than asserted.
- [x] **Tests actually ran (GPU tests not silently skipped)** — out of scope for this row, and
      unchanged: V0-5/V0-6/V0-7, V0-8 and V2-1 are byte-identical to what I approved at round 3. For
      *this* review's own claims: every number above is pasted command output, including the
      independent `master` baseline and the executed V0-2 comparison.
- [x] **Diff is surgical — no drive-by reformatting or refactoring** — demonstrated at the strongest
      available standard: reverting the two claimed edits reproduces the round-3 digest exactly, and
      `diff` shows precisely two hunks (`4a5,6`, `52c54`). No reflow, no renumbering, no heading
      churn. The 28 `Xpress` errors themselves stay unfixed, which is the surgical choice (I13/I4).
- [x] **Matches repo idiom (current `master`, not PR #516)** — the baseline is measured against
      current `master` @ `09c34dd`, not against #516 or any intermediate; the row names the commit
      explicitly and requires re-measurement on the *work package's base commit* if the baseline ever
      moves.

## The four §6.3 Gate-3 conditions — status after the scoped return

Re-confirmed only insofar as the V0-2 edit could affect them; the round-3 confirmations otherwise
stand on the byte-identity proof above.

1. **Every `C-R*` ID maps to concrete work; none unmapped or reinterpreted — STILL CONFIRMED.**
   `C-R1`…`C-R29`, 29 unique, verified mechanically on the current file. V0-2 keeps its eight IDs.
2. **Every threshold is a number with a stated rationale — STILL CONFIRMED, and V0-2 is now the one
   row where this is *newly* true.** Its old threshold was a number that could not be met; the new
   one is `28`, plus a set-equality with `0` symmetric difference, plus `0` new errors — each with
   the measurement and the commit it was taken on, and with re-measurement rules if the baseline
   moves. All 47 rows still carry numbers or exact-equality criteria.
3. **The V1 sign-convention matrix has all six cells filled — STILL CONFIRMED**, byte-identical
   (8 rows: six cells + presolve-`max` + coverage precondition).
4. **Non-goals explicit — STILL CONFIRMED**, in `FINAL.md` §10, untouched by this return (16 items).

*Scoped Gate-3 return approved. Selector-reviewer role remains open for further scoped returns;
this is round 4 of the §6.3 cap of 5, and it consumed one iteration on a single row.*

### Round-4 closure — one-token fix applied, file frozen (2026-08-22)

NB-1's required correction was applied and nothing else was. **Verified mechanically, not accepted on
report:** reverting the single 4-byte token `:2906` → `:2985` in the V0-2 row returns the file
**byte-for-byte** to `a6e93a73…`, the artefact I approved at round 4 — so that token is the *only*
difference. Length is unchanged (40 994 bytes, 199 lines), as a same-length swap requires. Continuing
the reversion (drop the round-4 note, restore the round-3 V0-2 row) reproduces `31e7c45c…`, so the
whole chain is intact:

```
current  sha256 cf07b743c172a7dd374be5115ab5fff91ee8d2235fbf4c905132fa69c53ac977   (40994 bytes)
  occurrences of ":2906" = 1   ":2985" = 0
  revert :2906 -> :2985        -> a6e93a73f9c51b9ffcea9705f39173f07fae7ba93c72783f10b88ea9773cbca7  MATCH (round 4)
  + revert note and V0-2 row   -> 31e7c45cd6a8fba832ff0aace1c56844d39fc45bdd019ee8079566536d504716  MATCH (round 3)
```

The underlying fact is confirmed against the source, not just the mypy output I pasted at round 4:

```
$ git show master:linopy/solvers.py  | grep -n chgobjsense
2906:                problem.chgobjsense(xpress.maximize)
$ git show a2db13f:linopy/solvers.py | grep -n chgobjsense
2985:                problem.chgobjsense(xpress.maximize)
```

`:2906` on untouched `master` @ `09c34dd` is correct, and `:2985` was indeed the post-slice number.
The row now cites the tree it says it measured.

**FINAL frozen digest — this review log's copy of record (§6.3a, second independent copy):**

* **sha256:** `cf07b743c172a7dd374be5115ab5fff91ee8d2235fbf4c905132fa69c53ac977`
* **File:** `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md` · 199 lines ·
  40 994 bytes · mode `-r--r--r--` (frozen, verified)
* **Confirmed at:** `2026-08-22T21:25Z` (UTC), by the reviewer, against the lead's recorded digest
* **Supersedes:** `a6e93a73…` (round 4 interim, never frozen) and `31e7c45c…` (round 3). Both remain
  above as superseded entries and must not be overwritten. `3306…ba49` (round 2) was never a freeze.

`cf07b743…` is the digest Phase 5 recomputes and compares against **both** this file and
`logs/00-lead-journal.md`. A mismatch, or two recorded digests that disagree, is a blocking finding.
Any further edit is a return to Gate 3: `chmod 644`, revise, re-approve, re-hash, and record the new
digest as a **new** dated entry in both logs.

*Round-4 scoped Gate-3 return closed. Verdict stands: **APPROVE**.*

---

# cuOpt support — plan review, round 5 (Gate-3 return, four edits)

*Reviewer: Phase 3 selector-reviewer (Opus 5, `claude-opus-5[1m]`) · 2026-08-23 · Reviewing: the four
round-5 edits to `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md` (203 lines,
51 633 bytes, mode `-rw-r--r--`). The rest of the file carries the round-3/4 approvals and was proved
unchanged rather than re-reviewed.*

Context read first, from disk: `logs/40-blockers.md` **B18** and **B19** (plus B17 and its addendum,
which B18 and V2-7 both depend on), and `logs/00-lead-journal.md` from `[2026-08-23T00:57Z]` onward.
Scope: the four edits only, per the lead's brief. Last round of the §6.3 cap.

## Verdict

**APPROVE.** Zero blocking findings.

All four edits are scope-fixes or strengthenings; **none is a widening**, and I checked that claim
threshold by threshold rather than accepting it. The one measurement I re-ran independently — B18's
replacement recipe — reproduced **all three pinned fingerprints exactly** and returned
`condition=time_limit` on the same commit the row cites, together with the precise failure *shape*
the row predicts. The no-other-change proof is conclusive: reverting exactly the six claimed hunks
returns the file **byte-for-byte** to `cf07b743…`, the artefact I froze at round 4.

The strongest thing in this round is a **negative result the planner chose to record rather than
bury**: its targeted GC-on-foreign-thread reproducer for B19 did **not** fire on the unfixed tip
(60/60 instrumented teardowns survived, mechanism verified to be exercised), so no reproducer row was
frozen — charter §9's can-fire rule applied against the planner's own work, at the cost of admitting
its reproducer failed. That is the behaviour this harness exists to produce.

One non-blocking asymmetry is recorded (NB-1): three of the four edits describe work not yet in code,
and only V1-8 says so. I verified all three lags are real. It is not blocking — a verification plan is
a list of obligations, not a record of passes, and I approved V1-13a–c and V3-4b at round 3 before
WP-F and WP-E existed without demanding such notes. Applying NB-1's one clause before the lead
re-hashes is free; afterwards it costs a Gate-3 return.

**New digest for the lead to record and freeze:**

```
sha256sum /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md
4708be847ed6a0887a29e476d622753fe6d0dc17bf36db6d99c69965f1c4faf5
```

* **sha256:** `4708be847ed6a0887a29e476d622753fe6d0dc17bf36db6d99c69965f1c4faf5`
* **Confirmed at:** `2026-08-23T01:13:38Z` (UTC) · 203 lines · 51 633 bytes · mode `-rw-r--r--`
* **Supersedes:** `cf07b743…` (round 4, frozen) via the interim `873b31d8…` (items 1–2 only, never
  reviewed, never frozen). Digest chain of record, all as **new** entries, none overwritten:
  `3306…ba49` (r2, never a freeze) → `31e7c45c…` (r3) → `a6e93a73…` (r4 interim) → `cf07b743…`
  (r4 frozen) → `873b31d8…` (r5 interim) → **`4708be84…`** (r5, to freeze).
* **Lead actions:** record the digest above with its timestamp as a new dated entry in
  `logs/00-lead-journal.md`, then `chmod 444`. If NB-1's clause is taken, apply it **before** hashing.

## Blocking findings

*None.*

| # | Finding | Where (file:line) | Why it blocks | Suggested fix |
|---|---|---|---|---|
| — | *(none)* | — | — | — |

## (a) Each edit: scope-fix or strengthening, never a widening

### Edit 1 — B18, MILP `TimeLimit` recipe substitution (`:153`, plus V1-5's rationale at `:70`)

**A scope-fix, and the substitution rule is tightened.** The threshold itself did not move: V1-5 still
demands *"exact string equality on `termination_condition` for every listed row"*. What changed is the
**forcing recipe**, which had been frozen on a measurement taken under cuOpt's own `method=0` while
linopy defaults to `method=3` — so the frozen row asked for a status the recipe provably does not
produce. Replacing a recipe that cannot produce its own asserted status is the same class of fix as
round 4's V0-2, and by the same legitimate route (a lead-opened Gate-3 return, not an in-place edit by
the slice that hit it).

* **The old recipe's miss is preserved inside the frozen artefact**, not only in a side log: the row
  states the subset-sum recipe *"is **struck**: it was measured under cuOpt's own `method=0`, and
  under linopy's `method=3` it terminates `FeasibleFound` → `ok suboptimal 2.23s` (B18, WP-D)"*.
  A later reader cannot lose the history by not opening `40-blockers.md`. This matters more than it
  looks: it is what stops the struck recipe being "restored" by someone who finds it in the
  exploration log.
* **WP-D's handling was correct and is worth recording here:** faced with a frozen recipe that missed,
  it reported verbatim, left `test_status_map[milp_time_limit]` **unwritten**, and adopted **no**
  substitute. That is §3.6/I12 executed exactly as written, and it is why this round is a Gate-3
  return rather than a silent recipe swap.
* **Tightenings the new row adds:** the assertion scope is now explicit (*"Assert the **condition**
  only"*, with the reason — cuOpt returns 0 primal values and linopy yields an empty `Solution`);
  the recipe is **self-contained and fingerprinted** (seed, first-and-only draw, `A[0,:6]`,
  `A.sum()`, `rhs`) so it cannot drift with numpy's generator state; a 4/4 `time_limit ∈ {0.5, 1, 2, 5}`
  sweep shows it is not balanced on a razor edge; and *"Any other `time_limit`, seed or shape is a
  **new** recipe requiring its own 3/3 measurement — never a silent substitution"* closes the door
  the old row left open.
* **Bonus strengthening:** the in-process 3/3 leg is also a B17 regression guard, because this exact
  recipe was a B17 crash trigger before `67675e4`.
* **V1-5's rationale was corrected to match** (`:70`): *"Each row is forced the way expl §1 forced it,
  **except MILP `TimeLimit`**, whose recipe was re-measured in round 5 … because expl §1's was taken
  under `method=0` (B18 …)"*. Verified accurate — without this the file would have asserted a
  provenance for that row that is no longer true.

### Edit 2 — V2-7, five-run `--run-gpu` abort-free stability (`:96`, new)

**Additive and numeric; a strengthening on the lead's own suggestion.** Every threshold is a count:
`Fatal Python error` occurrences exactly `0`; runs exiting `134`/`139` exactly `0`; a pytest summary
line in each log (`passed` count `≥ 1`) so a *truncated capture* cannot masquerade as a clean run; and
each run's FAILED/ERROR set at `0` symmetric difference from `logs/05-baseline-pytest-gpu.txt`.

* **It can fire, and I verified the cited evidence rather than trusting the citation.**
  `logs/phase4-gpu-stability-run3.txt` **line 39** is indeed the abort:

  ```
  line 39: ..............................................Fatal Python error: Aborted
  line 41: Thread 0x00007a8dcaffd640 (most recent call first):
  line 42:   File "/home/azureuser/linopy-gpu/linopy/solvers.py", line 282 in _worker
  ```

  The citation is exact for what the row counts (`Fatal Python error` occurrences). One clarification
  for whoever reads that dump next: lines 41–42 are the **first** thread faulthandler lists — the
  persistent cuOpt solve worker, idle — not the crashing thread; B19's attribution to a dask worker
  executing linopy graph tasks is unaffected, and the row does not claim otherwise.
* **Five rather than three is a strengthening with stated arithmetic**, and the arithmetic checks out:
  at the observed abort rate `2/4 = 0.5`, three clean runs pass a broken build with `0.5³ = 12.5%`,
  five with `0.5⁵ = 3.125%`. The row prices it (~46 min) and forbids reducing it (*"Fewer runs is a
  **weakening** and requires a return to Gate 3, not a judgement call at execution time"*) — which is
  the sentence that matters, because 46 minutes is exactly the kind of cost an execution-time agent
  talks itself out of.
* **Implementation-agnostic by construction** — it measures observable process outcomes only, so it
  survives either candidate B19 fix or a third. That is the right shape for a row frozen *before* the
  fix is chosen.
* **The `PYTEST_EXIT=1` carve-out is not a loophole:** it is scoped to *"solely from the pre-existing
  KI1 set"*, and the `0`-symmetric-difference clause is what makes "solely" mechanically decidable.
* **The recorded negative result is the right call.** 60/60 instrumented cycles (20 fresh-thread +
  40 reused-pool, `gc.disable()`d, cycle-reclaimed, with `Solver.close` instrumentation proving
  native teardown really happened on the foreign thread) and **0** aborts → no reproducer row frozen,
  per §9. The row also pre-commits the route if that changes: a reproducer that *does* abort on
  `9e1e96e` is *"a further Gate-3 return with its BEFORE/AFTER pasted, never a silent addition"*.

### Edit 3 — V3-6, C-R30 / WP-G verification (`:103`, new)

**Additive, and it closes a genuine coverage hole the planner found itself.** Before this round
`FINAL.md` carried `C-R30` (Amendment A1 / D10) and the verification plan did not — and since Phase 5
executes *this* file rather than `FINAL.md` §7.2, the nine limitation rationales, including the
**human-directed** A1.1 developer pointer, would have shipped unverified. Verified mechanically: both
files now carry the identical 30-ID set with `0` symmetric difference.

* **All four thresholds are exact counts** — `0` differing files under docstring-normalised AST
  equality; `0` non-comment/non-docstring payload lines; `9` mapped checklist items with `0`
  missing/deferred and `0` citations landing on non-comment lines; `≥ 1` for each of ten grep anchors;
  plus GPU-summary identity with the pre-WP-G run at `0` symmetric difference.
* **"Nine" is real and countable.** I checked `FINAL.md` §7.2's WP-G block: it is a numbered list
  `1.`–`9.`, and the ten anchors in threshold (d) map onto it cleanly (item 2 carries two:
  `pdlp_solver_mode` and `set_initial_primal_solution`). So threshold (c) is not an adjective
  wearing a number.
* **It is strictly stronger than `FINAL.md` §7.2's own acceptance criterion**, which asks for the diff
  to be *"shown to contain only comment/docstring lines"* — i.e. a human reading a diff. V3-6 keeps
  that (threshold b) and adds an AST proof (threshold a). The row is candid about why: *"'comments
  only' is exactly the claim a reviewer is least able to check by reading a large diff, and AST
  equality decides it mechanically."* Correct — and comments genuinely never reach the AST, so the
  test is sound.
* **The division of labour matches V3-4b's precedent I approved at round 3**: counts here, prose
  quality and §6.6a comment density with the Gate-6 reviewer. The row says so explicitly and
  disclaims asserting prose quality.
* **Good failure-attribution clause:** if either `--run-gpu` cross-check run aborts, *"that is a
  **V2-7** miss, not a V3-6 result"*. Without that, B19 flakiness would have been charged to a
  comments-only commit.

### Edit 4 — V1-8, 3 → 20 sequential LP solves (`:77`)

**A pure strengthening, and the reason it was needed is itself the interesting part.** The upstream
agent measured the fresh-thread OpenMP abort threshold on the LP path at **5–13 solves**
(NVIDIA/cuopt#1768), so the frozen `N=3` sat **below** the threshold: the row would have passed on
defective code while reading as "repeated LP solves are safe". A check that cannot reach the failure
it exists to detect is the same defect class as round 4's unreachable V0-2 threshold, seen from the
other side.

* `20 / 13 = 1.538` → the row's *"margin of 1.54×"* is correct.
* **It also adds thresholds** rather than only raising one: `0` occurrences of `Fatal Python error`,
  `OMP:` or `kmp_alloc` in the captured output, and *"a **fresh** `Model` built inside each
  iteration"* — which is what makes the loop exercise the per-iteration teardown path rather than
  re-solving one retained model.
* Re-measured on the cited commit: *"20/20 `optimal`, exit `0`, 0 OMP/abort lines, TOTAL 20 solves in
  134.6s"*, and the row names the cost honestly (~6.7 s/iteration, dominated by building the dense
  1000×2000 model) and forbids the obvious future weakening: *"shrinking N to save suite time is a
  weakening and needs a return to Gate 3."*
* **Its lag disclosure is accurate — I verified it.** `test/test_cuopt.py::test_repeated_solves_of_a_medium_model`
  does still assert the old N:

  ```
  test/test_cuopt.py:659:        for _ in range(3):
  test/test_cuopt.py:672:    assert result.stdout.count("optimal") == 3
  ```

## (b) Independent spot-verification — B18's replacement recipe

Rebuilt from the row's own construction text (not from the planner's script), on the same commit the
row cites, `feat/cuopt-solver` @ **`9e1e96e`**, working tree clean:

```
FINGERPRINT A[0,:6] = [81, 8, 17, 23, 18, 80]  expected [81, 8, 17, 23, 18, 80]
FINGERPRINT A.sum() = 14702.0                  expected 14702.0
FINGERPRINT rhs     = [1182.5, 1292.5, 1369.5, 1122.5, 1279.5, 1106.5]
            expected [1182.5, 1292.5, 1369.5, 1122.5, 1279.5, 1106.5]
cuOpt returned 0 primal values for 50 variables (TimeLimit).
RESULT status=ok condition=time_limit wall=3.32s
solution assigned: False
```

**All three fingerprints match exactly**, so the recipe is genuinely self-contained and seed-pinned —
my `default_rng(3)` first-and-only draw reproduced the planner's `A` bit for bit. `condition` is
`time_limit`, which is what V1-5 asserts. And the *shape* the row predicts is confirmed, not just the
status: exactly one length-mismatch error logged (`cuOpt returned 0 primal values for 50 variables
(TimeLimit)`), no solution assigned to the model — i.e. the empty-`Solution` path V1-5b pins on the LP
side, reached here on the MILP side. The row's claim that this recipe exercises that shape is
therefore load-bearing and true.

**One deviation, reported verbatim rather than smoothed over:** my wall time was **3.32 s** against
the row's fresh-process figures of **3.26 / 3.22 / 3.22 s**. It is outside that three-sample spread.
Wall time is **not** a threshold in this row (V1-5 asserts the condition only, and the row says so
explicitly), the model is solved under `time_limit=2.0` so ~1.2–1.3 s of the wall is build plus
transfer either way, and my run was a cold first solve in a fresh process. I record the number
because §3.6 asks for the actual figure, not because it is a miss.

## (c) Mechanical no-other-change proof

I did not re-read the file. I reconstructed the round-4 frozen artefact from the round-5 file, taking
the three edited rows' previous text **byte-exact** from my own digest-verified round-3 copy
(`31e7c45c…`, retained from round 4) rather than retyping them, so transcription cannot enter:

```
current bytes: 51633 sha256: 4708be847ed6a0887a29e476d622753fe6d0dc17bf36db6d99c69965f1c4faf5
  removed : round-5 note (2 lines), '| **V2-7** |', '| **V3-6** |'
  restored: '| V1-5 |', '| MILP `TimeLimit` (5) |', '| **V1-8** |'   (from the 31e7c45c… artefact)
rebuilt bytes: 40994 sha256: cf07b743c172a7dd374be5115ab5fff91ee8d2235fbf4c905132fa69c53ac977
MATCHES ROUND-4 FROZEN DIGEST cf07b743...: True
```

`diff` against that reconstruction gives exactly the six claimed hunks and nothing else — 10 payload
lines, which is the arithmetic of 2 added note lines + 3 changed rows (2 lines each) + 2 added rows:

```
4a5,6      round-5 revision-state note
68c70      V1-5 rationale
75c77      V1-8
93a96      V2-7 (new)
99a103     V3-6 (new)
149c153    MILP `TimeLimit` status row
```

| Property | Round-4 frozen (`cf07b743…`) | Round-5 current | Result |
|---|---|---|---|
| `## ` heading text | 4 headings | 4 headings | **byte-identical** (`diff` empty) |
| heading line numbers | 49, 101, 129, 161 | 51, 105, 133, 165 | shifted `+2/+4/+4/+4` — exactly the note (+2) then V2-7 and V3-6 (+1 each, both above line 101) |
| check rows | 47 | **49** | +2, the two new rows |
| unique `C-R` IDs, this file | 29 | **30** | +`C-R30` |
| `C-R` symmetric difference vs `FINAL.md` | — | **0** (30 each) | the 30-ID gap is closed |
| V3 tier ordering | V3-1…V3-5 | V3-1, V3-2, V3-3, V3-4, V3-4b, V3-5, **V3-6** | monotonic |
| tolerance-constant rows | 11 | 11 | unchanged |
| sign-matrix rows | 8 | 8 | unchanged |
| status-map rows | 17 | 17 | unchanged (one row's cells edited, no row added or removed) |
| Nondeterminism rows | 10 | 10 | unchanged |

The round-5 note also states *"Check-row count 47 → 49 (counted mechanically, including the lettered
sub-rows)"* — the planner adopted the corrected count from my round-4 NB-4 rather than propagating my
earlier 45. Worth recording: the count in the artefact is now right.

## (d) The plan is ahead of the branch — what the rows do and do not say

Confirmed, and all three lags are real. Verified on the current tip `9e1e96e`:

| Row | Work it requires | Present in code? | Does the row say so? |
|---|---|---|---|
| V1-8 | test at `N=20` | **No** — `range(3)` / `count("optimal") == 3` at `test_cuopt.py:659,672` | **Yes**, explicitly: *"Code lags this row … the conformance reviewer should read this row as **not yet implemented at its frozen strength** — not as a passing check."* |
| MILP `TimeLimit` (`:153`) | `test_status_map[milp_time_limit]` | **No** — `grep -c milp_time_limit test/test_cuopt.py` → `0` | **No** (see NB-1) |
| V3-6 | the WP-G commit | **No** — branch is `67675e4, 8a32152, eb3b37b, 9e1e96e`; no "unsupported features" commit exists | **No**, though "the WP-G commit … FINAL.md §7.3 commit 6" reads as forward-looking |
| V2-7 | a five-run sequence | **No** — 4 runs exist, 2 aborted | **Effectively yes**: the row records *"2 aborts in 4 full runs"* on this tip, so it cannot be read as passing |

## Non-blocking observations

1. **Lag-disclosure asymmetry — one clause would close it.** V1-8 carries an explicit "code lags this
   row / read this as not yet implemented" sentence; it is the **only** such sentence in the file
   (`grep` → 1 occurrence, line 77). The MILP `TimeLimit` row names
   `test/test_cuopt.py::test_status_map[milp_time_limit]` in its Test column, and that test **does not
   exist** — verified, `0` occurrences — which is precisely the fact B18 was raised to report.
   **Recommendation:** one clause in that row's Test cell or rationale, e.g. *"test not yet written
   (B18); the follow-up test agent adds it from this recipe."* V3-6 needs nothing (a row specifying a
   not-yet-written work package is normal — I approved V1-13a–c and V3-4b at round 3 before WP-F and
   WP-E existed), and V2-7 self-discloses. **Why non-blocking:** a verification plan is a list of
   obligations, not a register of passes; Phase 5 executing this row will find the test absent and
   report a miss, which is the system working. The asymmetry is worth removing only because having
   said it once for V1-8 invites the inference that silence elsewhere means "implemented".
2. **Implementation note for V3-6 threshold (a) — call `ast.dump()` without `include_attributes`.**
   The row says "compare `ast.dump(...)`" without pinning that argument. Default `ast.dump` omits
   `lineno`/`col_offset`, which is exactly what makes the check work: inserting comment lines shifts
   every subsequent node's line number, so an implementation passing `include_attributes=True` would
   report a mismatch for **every** comment added and the row would fail on correct work. Same species
   as round 4's note about counting `: error:` lines — the row gives the criterion, not the extraction
   rule.
3. **V2-7's false-pass figure is an estimate, not a bound**, and the row should not later be read as
   guaranteeing 3.1%. It assumes runs are independent Bernoulli trials at `p = 0.5`, and `p` is
   itself estimated from 4 observations (a 2/4 point estimate carries a wide interval, roughly
   0.15–0.85 at 95%). The *threshold* is unaffected — it is "five runs, zero aborts" — and stating
   the model explicitly is what Gate-3 check 2 asks for. Recorded so nobody later argues three runs
   are "97% as good".
4. **The B18 wall-time deviation** (mine 3.32 s vs the row's 3.26/3.22/3.22 s) is reported in §(b)
   above rather than here, because §3.6 asks for actual numbers. It is not a threshold.

## Checklist

- [x] **Every requirement ID in scope is covered** — and this round *increases* it: `C-R30` had no row
      before, and both files now carry `C-R1`…`C-R30` with `0` symmetric difference (verified
      mechanically, 30 IDs each). No ID renumbered; V3-6 states so explicitly.
- [x] **No requirement silently dropped or reinterpreted** — nothing silent: a lead-opened Gate-3
      return, an in-file round-5 revision note enumerating all four edits, the struck subset-sum
      recipe recorded *inside* the row with its measured miss, and B18/B19 cross-referenced. V1-5's
      provenance sentence was corrected rather than left stale, which is the opposite of silent.
- [x] **Frozen verification plan** — legitimately unfrozen for this return (`chmod 644`); the
      superseded `cf07b743…` is recorded in this file (round 4) and inside the artefact's own note;
      new digest computed above; re-freezing is the lead's remaining action. Verified the digest is
      stable and that writing this review does not perturb it.
- [x] **No threshold widened, no check downgraded to a weaker tier** — the load-bearing row, checked
      edit by edit: V1-5's threshold text is **unchanged** (a recipe moved, not a bar); V2-7 and V3-6
      are **additive** with all-numeric thresholds; V1-8 raises `3 → 20` **and adds** three new
      zero-count assertions plus the fresh-`Model` requirement. Tiers unchanged (V2-7 is V2, V3-6 is
      V0). Three rows explicitly forbid the future weakening they invite (fewer than five runs;
      shrinking V1-8's N; substituting the B18 recipe) — each naming Gate 3 as the only route. Every
      other threshold in the file is byte-identical to `cf07b743…`, proved by hash. The only
      *removal* is a forcing recipe that could not produce its own asserted status.
- [x] **Tests actually ran (GPU tests not silently skipped)** — strengthened this round rather than
      unchanged: V2-7 requires a completed pytest summary line in **each** of five logs specifically
      so a truncated capture cannot read as clean (the failure mode that already bit once, journal
      `00:45Z`), and forbids `tail`/`tee` piping. V1-8 now counts abort markers as well as `optimal`
      lines. For *this* review's own claims: every number is pasted command output, including the
      independent recipe run and the reconstruction hashes.
- [x] **Diff is surgical — no drive-by reformatting or refactoring** — at the strongest available
      standard: reverting the six claimed hunks reproduces `cf07b743…` byte-for-byte, `diff` shows
      exactly those six and 10 payload lines, and the four `## ` headings are byte-identical in text.
      No reflow, no renumbering, no ID churn.
- [x] **Matches repo idiom (current `master`, not PR #516)** — measurements are pinned to real commits
      on this branch (`9e1e96e`, B17 fix `67675e4`) and to upstream issue #1768, not to #516; my own
      spot-check ran on `9e1e96e`, the same commit the row cites.

## The four §6.3 Gate-3 conditions — status after this return

1. **Every `C-R*` ID maps to concrete work; none unmapped or reinterpreted — CONFIRMED, and improved.**
   `C-R1`…`C-R30`, 30 unique, `0` symmetric difference between `FINAL.md` and this file. Round 5 closed
   the one genuine gap (`C-R30`), which had been mapped in the plan but unverified by the plan.
2. **Every threshold is a number with a stated rationale — CONFIRMED across all 49 rows.** The two new
   rows are all counts (`0`, `0`, `≥ 1`, `0` symmetric difference; `0`, `0`, `9`/`0`/`0`, `≥ 1`), each
   with its measurement and its can-fire argument. V1-8's `20` is derived from a measured 5–13
   threshold with the margin stated. B18's recipe is fingerprinted so the number cannot drift.
3. **The V1 sign-convention matrix has all six cells filled — STILL CONFIRMED**, byte-identical
   (8 rows: six cells + presolve-`max` + coverage precondition).
4. **Non-goals explicit — STILL CONFIRMED**, `FINAL.md` §10, untouched by this return (16 items).

*Round-5 Gate-3 return approved; the §6.3 cap is not exhausted and no E5 escalation is needed. Three
of the four edits describe obligations the branch has yet to meet — that is the plan being correctly
ahead of the code after a Gate-3 return, and §(d) above is the record of which ones, so the
conformance reviewer cannot mistake them for passing checks.*

### Round-5 closure — both precision corrections applied, file frozen (2026-08-23)

NB-1 and NB-2 were applied and nothing else was. **Verified mechanically, not accepted on report:**
reverting exactly three in-line string edits returns the file **byte-for-byte** to `4708be84…`, the
artefact I approved at round 5 — so those three are the only differences. Line count is unchanged at
**203** (all three are in-line, no row added or removed) and the byte count rises `51 633 → 52 622`,
i.e. `+989` bytes of added prose and nothing else:

```
current  sha256 feef773d8fface670eda923095ee216de32a8fedf50b0561caaebe9687a3f820   (52622 bytes, 203 lines)
  revert (1) the "Post-approval precision pass" sentence in the round-5 revision note
  revert (2) V3-6's ast.dump pin  -> `compare `ast.dump(...)` — **exact string equality, `0` differing files**`
  revert (3) the MILP `TimeLimit` row's "Code lags this row" clause
         -> 4708be847ed6a0887a29e476d622753fe6d0dc17bf36db6d99c69965f1c4faf5   MATCH (round 5, approved)
```

Each replacement was asserted unique (`count == 1`) before being applied, so no second occurrence was
silently rewritten. Both corrections landed **stronger than I asked for**, which is worth recording:

* **NB-1 (MILP `TimeLimit` lag clause)** does not merely note the gap — it makes it *checkable* and
  attributes it correctly: *"`test_status_map[milp_time_limit]` does **not** exist yet (`milp_time_limit`
  occurrence count in `test/test_cuopt.py` is `0` at `9e1e96e`) — WP-D correctly declined to write it
  against the struck recipe (B18, report-and-stop)."* The occurrence count is the number I verified
  independently at round 5, and crediting WP-D's report-and-stop is right: leaving that test unwritten
  was correct conduct under §3.6, not an oversight, and the frozen file now says so.
* **NB-2 (V3-6 `ast.dump`)** pins the argument *and* the reason: *"at its **defaults —
  `include_attributes=False`** (explicitly: never `include_attributes=True`, which embeds
  `lineno`/`col_offset` and would make every added comment shift the offsets below it, false-failing
  the row on exactly the commits it exists to pass)."* That is the failure mode I flagged, written
  where the executor will read it.

Neither edit touches a threshold: (1) is a revision note, (2) pins an implementation detail of an
existing exact-equality criterion, (3) is a status disclosure. No number moved, no scope changed, no
row added or removed. My round-5 **APPROVE** therefore stands unchanged.

**FINAL frozen digest — this review log's copy of record (§6.3a, second independent copy):**

* **sha256:** `feef773d8fface670eda923095ee216de32a8fedf50b0561caaebe9687a3f820`
* **File:** `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md` · 203 lines ·
  52 622 bytes · mode `-r--r--r--` (frozen, verified)
* **Confirmed at:** `2026-08-23T01:17Z` (UTC), by the reviewer, against the lead's recorded digest
* **Supersedes:** `4708be84…` (round 5, approved but superseded by this precision pass),
  `cf07b743…` (round 4, frozen), `31e7c45c…` (round 3, frozen), and the never-frozen interims
  `a6e93a73…`, `873b31d8…`, `3306…ba49`. All are kept above as superseded entries and must not be
  overwritten. Full chain of record:
  `3306…ba49` (r2, never a freeze) → `31e7c45c…` (r3) → `a6e93a73…` (r4 interim) → `cf07b743…` (r4)
  → `873b31d8…` (r5 interim) → `4708be84…` (r5 approved) → **`feef773d…`** (r5 final, frozen).

`feef773d…` is the digest Phase 5 recomputes and compares against **both** this file and
`logs/00-lead-journal.md`. A mismatch, or two recorded digests that disagree, is a blocking finding.
Any further edit is a return to Gate 3 — and note the §6.3 iteration cap: round 5 was the last of
five, so a sixth substantive return needs the lead's E5 filing under §3.7, not another quiet
iteration. A precision pass inside an already-approved round, verified by hash as this one was, is
not a sixth round.

*Round-5 Gate-3 return closed. Verdict stands: **APPROVE**. Gate 3 is met: `FINAL.md` approved,
`logs/15-verification.md` frozen at `feef773d…`, and the four §6.3 conditions confirmed on the record
in rounds 3–5.*

---

## Phase 5 + Phase 6 review record (2026-08-23, appended by the lead for Gate-6 closure)

**Phase 5 (independent conformance + correctness), 2 loops of 3:** loop 1 executed all 49 frozen rows (48 PASS; V2-7 RED-with-cause per D16/ESC-2, verified via three completed baseline-identical full-GPU runs) and returned findings F1-F10; blocking F4/V3-5 (bulk +1986 vs ≤1157) accepted-with-rationale by the lead (ESC-3/D17, UNREVIEWED); F1/F2/F3/F5/F7/F8 fixed in commit 7 (b0b18f4); loop 2 re-verified every fix independently (incl. non-vacuity proof of the F2 regression test) → **GATE 5 CLEAN**. Evidence: phase5-*.txt (38 files).

**Phase 6 (style + docs conformance, different reviewer), 2 loops of 3:** loop 1 → CLEAN with S1/S2/S3/N1/N3 (+N2 declined by lead, D18); every §6.6b surface satisfied — none absent, so nothing to justify here beyond the ipynb decision (cuOpt added to the backend list + a GPU-solvers note; judged sufficient, logged); PR-#516 mechanical gap check fully justified (3 shared-file gaps all covered by master's replacement machinery or this branch's own edits); docs build 19=19 vs baseline; leak check EMPTY. Fixes landed by tip amend (a8fb35b); loop 2 proved zero behaviour change by AST byte-equality + zero-fact-loss token audit, accepted the N1 resolution (no in-repo precedent existed), re-ran the leak check EMPTY → **GATE 6 CLEAN** (optional nit N4 declined, ship-as-is per reviewer). Evidence: phase6-*.txt.

**Gate-6 §6.4a full-branch re-run (lead, at a8fb35b):** 7 commits, 0 non-Claude authors, 0 merges, 7/7 trailers, 0 author/trailer mismatches, porcelain clean.
