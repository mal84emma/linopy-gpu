# FINAL — NVIDIA cuOpt support in linopy

*Written by: Phase 3 **revision planner** (Opus 5, `claude-opus-5[1m]`) · round 1 · 2026-08-22 ·
against `master`/`feat/cuopt-solver` @ `09c34dd`*

**This document is self-contained.** A coding agent implementing a work package needs this file,
the charter (`/home/azureuser/linopy-gpu/dev-scripts/cuopt/PROJECT_LEAD.md`),
`/home/azureuser/linopy-gpu/AGENTS.md`, the frozen
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md`, and the "Rules for later
agents" block of `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/30-decisions.md`. It does **not**
need `plan-1..5.md` or `logs/50-review.md`.

Reading basis for this revision (all read in full from disk, none paraphrased): charter §§4, 5, 5.3,
6.3/6.3a, 6.4/6.4a, 6.6a/6.6b, 7, 8, 9; `AGENTS.md`; `logs/30-decisions.md` (rules block + D1–D6);
`logs/10-exploration.md`; `logs/50-review.md`; `plans/plan-1.md` in full plus the graft sections the
review names in `plan-2.md`, `plan-3.md`, `plan-4.md`, `plan-5.md`; and read-only inspection of
`linopy/solvers.py`, `linopy/model.py`, `linopy/io.py`, `linopy/common.py`, `linopy/constants.py`,
`linopy/objective.py`, `doc/api.rst`, `pyproject.toml`, `test/conftest.py`,
`test/test_optimization.py`, `test/test_solvers.py`, and the installed `cuopt` 26.08.00 package.

**Base plan:** `plan-1.md` (the selector-reviewer's choice), carried forward as the spine.
**Grafts G1–G10 applied. Rejections R1–R13 honoured. Blocking findings B1–B16 resolved.** Two
further defects found during this revision are resolved as **B17** and **B18** and flagged for the
selector to re-check (§12).

---

## 0. What changed relative to the base plan

Every change is traceable to a finding. Nothing was edited silently.

| Finding | Base plan said | `FINAL.md` says | Where |
|---|---|---|---|
| B1 / R5 | doubly-infinite pad row for zero-nnz `A` | **singly-bounded pad row** on a variable with a finite bound; explicit residual branch | §2.1.6, §5j |
| B2 / R4 | (base plan already padded, did not raise) | confirmed: **never** `raise ValueError` for zero-nnz | §5j |
| B3 / G1 / R6 | override `_run_file` to fall back | override **`_build_file`**; leave `_run_file` at the base | §2.1.4, §5k |
| B4 | — (leak not costed) | override sets `self._problem_fn` so linopy's own `finally` unlinks the temp file | §2.1.4, §5k |
| B5 / R2 | in-process `cudaGetDeviceCount()` | **out-of-process, cached** probe; in-process CUDA calls forbidden | §2.1.9, §5c |
| B6 / R3 | (base plan was already fail-closed) | confirmed fail-closed, `bool(count)` None-safe, warning names driver/CC | §2.1.9, §5c |
| B7 / R1 | GPU probe in `is_available()` | unchanged — D6 confirms it; plan-3/5's argument recorded as the accepted downside | §5c |
| B8 | empty MIP dual returned unscattered | unchanged, and now justified with the measured `IndexError` | §2.1.7, §5e |
| B9 / G3 / R7 / R8 | `_BOOL_PARAMETERS` allowlist | **blanket `bool → int`** coercion | §2.1.8, §5f |
| B10 | risk, unresolved | disposition **frozen**: honest mapping + one cuopt branch in `test_quadratic_model_unbounded` | §5j, §8, verification V2-2 |
| B11 | `MIP_DUAL_BOUND_REPORT` declared | unchanged; premise now measured (`solution_bound = 100.0`) | §3 |
| B12 | `SEMI_CONTINUOUS_VARIABLES` declared | unchanged; note that the flag brings **zero** automatic coverage | §3, §8 |
| B13 | `to_cuopt` deliberately **not** in `doc/api.rst` | **added**, alongside `solvers.cuOpt`; missing siblings → `45-known-issues.md` | §9 |
| B14 / G5 | not in the test plan | both tight shared assertions are **named verification rows** with a pre-decided remedy | §8, verification V2-3/V2-4 |
| B15 / R11 | (base plan added no `method_options` row) | confirmed: no `test_solver_method_options` entry | §2.7 |
| B16 | named as a risk only | three probe branches are **checks**, incl. fork safety, one proven to fire | §8, verification V0-5/V0-6/V0-7 |
| G2 | enum-keyed status dicts | **name-keyed** status dicts + `NoTermination ≠ unknown` comment; no `.pyx` import | §2.1.7, §4 |
| G4 | acceptance criteria after tests | WP-A acceptance criteria **front-loaded**, incl. two "make it fail on purpose" guards | §7 |
| G6 | plain `to_cuopt` docstring | docstring states the `max` → minimisation form | §2.2 |
| G7 | `_license_probe` solves a 1×1 LP on the GPU | **constructs only**, no solve | §2.1.3 |
| G8 | `method=3` unconditional | pre-decided contingency if Barrier misbehaves for MILP | §5b |
| G9 | (base plan already gated) | confirmed gating mechanics, with the paired skipped/ran evidence | §8 |
| G10 | docs describe LP/MILP/QP in one commit | commit 3's docs are **LP/MILP only**; QP paragraph lands in commit 5 | §7, §9 |
| R10 | `accepted_io_apis` not declared | unchanged, with the dead-code evidence recorded | §1.3 |
| R12 | file `io_api`s out of scope | unchanged; recorded as a 5/5 agreement and an evidence-backed follow-up | §10 |
| R13 | `solution_fn` → `NotImplementedError` | unchanged (cuPDLPx precedent) | §2.1.7 |
| **B17 (new)** | `len(self._vlabels or ())` | **raises `ValueError` on numpy 2.4.6** — replaced with `.size` | §2.1.7, §12 |
| **B18 (new)** | `_build_file` override drops the base io_api validation | guard re-stated in the override | §2.1.4, §12 |
| **B19 (round 2)** | V1-13b / V1-6b compared the **full** primal of `quadratic_model` against HiGHS, and §8.3 credited that comparison with catching the `0.5·M.Q` trap | both rows scoped to the **`x` block and the objective** — `y` is underdetermined by the fixture and is never compared against the oracle; the Q-convention guard is re-attributed to V1-13a and V1-13c | §8.3, §7.2 (WP-F), §12, and `logs/15-verification.md` V1-6b / V1-13b / Nondeterminism |

---

## 1. Entry-point decision

**Decision: `cuopt.linear_programming` — `DataModel` + `Solve` — as the single entry point.
`io_api = "direct"` is the only io_api cuOpt supports.** Logged as a D-entry at Gate 3.

### 1.1 Why (measured, from `logs/10-exploration.md` Q0)

* **Speed** — 0.012 s to build a 10⁵-variable model, versus 0.86–1.04 s for `problem.Problem`
  (~70×) and 0.83–0.95 s for the file path. `Problem`'s per-variable Python calls do not improve
  with scale (0.069–0.146 s already at 10⁴).
* **Expressiveness** — `DataModel` is the only candidate that can express ranged rows;
  `Problem._to_data_model` structurally cannot (`problem.py:1593-1594`, `:1524`).
* **Fit** — 13 lines of translation from `model.matrices` for LP+MILP, 16 with QP.
  `M.A.tocsr()` already produces exactly the dtypes cuOpt stores (float64 / int32 / int32,
  exploration item 6), so no conversion is needed.
* **No lost surface** — warm start, MIP callbacks, options and log control all live on
  `SolverSettings`, which every candidate shares. `Problem.Slack`/`DualValue`/`ReducedCost` are
  `populate_solution` copies of the same `Solution` arrays.
* **Stability** — public, documented, no experimental marker. `internals` is not a candidate.

Verified live: `import cuopt; cuopt.linear_programming.DataModel` resolves, so
`_LazyModule("cuopt")` plus `cuopt.linear_programming.<X>` call sites work without importing the
subpackage explicitly.

### 1.2 Divergence from Q0's *secondary* recommendation (file `io_api`s) — R12

Q0 additionally recommends declaring `accepted_io_apis ⊇ {"lp", "mps"}` via
`cuopt.linear_programming.io.Read`, "~3 lines". **No file io_api in this change.** All five Phase-2
planners independently reached the same conclusion (§3.5: 5/5 → obvious, do not escalate). The
reasons, kept because a later reader will be tempted to add it:

1. **Solution files.** `SolverFeature.READ_MODEL_FROM_FILE` puts `cuopt` into `file_io_solvers`
   (`test_optimization.py:45`) and therefore into `test_set_files` / `test_set_files_and_keep_files`
   (`:655`, `:676-678`), which pass a real `solution_fn` and then **assert the file exists**.
   cuOpt's `solution_file` parameter exists but was explicitly left unmeasured (exploration
   Unknowns, agent B).
2. **Explicit coordinate names.** `file_io_solvers` are exercised with
   `explicit_coordinate_names=True` across ~30 parametrized cells. Q0's fidelity measurement used
   default labels (`x0`, `c0`); whether cuOpt's parser accepts linopy's `name(coord)#label` shapes
   was never measured.
3. **The Q convention inverts on the file path.** Q0 measured `.lp` yielding a *triangular* Q
   (`Qvals=[1,1,1]`) and `.mps` a *symmetric* one (`[1,0.5,0.5,1]`). Shipping file IO would mean a
   second, opposite implementation of the single most dangerous transform in this change (§5d).
4. **The M2 sense flip on a file-read model** (§5a) would need `get_objective_coefficients()`,
   negate, re-assert `set_maximize(False)` — plausible, unmeasured.

Cost of declaring it anyway: ~6 extra parametrizations × ~60 GPU-gated tests (≈360 extra GPU solves
per `--run-gpu` run) on top of four unverified contract points. Wrong trade for a first landing;
clean follow-up PR once (1)–(3) are measured. Q0's round-trip fidelity result is the evidence that
follow-up starts from. **Recorded as an evidence-backed follow-up in `logs/45-known-issues.md`.**

### 1.3 The io surface actually declared

* `features` contains `DIRECT_API` and **not** `READ_MODEL_FROM_FILE` / `LP_FILE_NAMES`. In this
  repo that *is* how the io surface is declared: `Model.solve`, `Solver._build_file`,
  `get_available_solvers_with_feature` and the test matrix all read the features.
* **`accepted_io_apis` is deliberately left at the base default** (`frozenset()`). Verified dead
  code on current `master`: `grep -rn accepted_io_apis linopy/ test/` finds exactly one declaration
  (`solvers.py:466`) and one test double (`test/test_persistent_solver_orchestrator.py:24`); **no
  shipped solver class declares it and no production code reads it.** Declaring it on cuOpt alone
  would be dead code and a lone stylistic outlier (I4, §6.6a). The supported surface is documented
  in the class docstring and in `doc/gpu-acceleration.rst`.
* **A bare `model.solve("cuopt")` (no `io_api`) must work** — it is exercised by
  `test/test_solvers.py:48` (`test_solver_instance_attached_after_solve`) and `:209`
  (`test_solver_close_releases_state`), both parametrized over `licensed_solvers` and both
  asserting success. Achieved by overriding `_build_file` (§2.1.4, §5k), **not** by overriding
  `_run_file`.

---

## 2. Full file-by-file change set

Line anchors are current-`master` (`09c34dd`) positions; they are insert points unless stated.
Estimated volume in brackets.

### 2.1 `linopy/solvers.py`  [≈ +250 / −0]

| Anchor | Change |
|---|---|
| `:34` `from scipy.sparse import tril, triu` | → `from scipy.sparse import csr_array, tril, triu, vstack` (needed only for the zero-nnz pad, §2.1.6). |
| `:148` `SolverFeature` | **No edit.** Every feature cuOpt needs already exists (`INTEGER_VARIABLES`, `QUADRATIC_OBJECTIVE`, `DIRECT_API`, `SOLUTION_FILE_NOT_NEEDED`, `GPU_ACCELERATION`, `GPU_ONLY`, `SEMI_CONTINUOUS_VARIABLES`, `MIP_DUAL_BOUND_REPORT`). Adding a member would be a drive-by (I4). |
| `:175-181` `TYPE_CHECKING` | add `import cuopt` (isort order: before `cupdlpx`). |
| `:304` after `cupdlpx = _LazyModule("cupdlpx")` | add `cuopt = _LazyModule("cuopt")` **plus the two-line comment** in §2.1.2. **No `# type: ignore`** (see §2.1.2). |
| after `_run_highs_with_keyboard_interrupt` (`:189-252`) | new module-level `_run_cuopt_with_keyboard_interrupt(solve)` (§2.1.10). |
| near `_has_module` (`:307-315`) | new module-level `_CUDA_PROBE_SNIPPET` + `@functools.cache def _cuda_device_available()` (§2.1.9). |
| `:346-359` `SolverName` | add `cuOpt = "cuopt"` immediately after `cuPDLPx = "cupdlpx"`. Vendor spelling; the member name must equal the class name for the `solver_capabilities` shim's `getattr(solvers, SolverName(name).name)`. |
| after the `cuPDLPx` class (ends `:4416`), before `_solver_class_for` (`:4419`) | new `class cuOpt(Solver[None])` (§2.1.1–§2.1.10). |
| `:4426` `QUADRATIC_SOLVERS` | **No edit** — derived from the feature (charter §4.5). |
| `:4443-4457` `_SOLVER_PROBE_ORDER` | insert `"cuopt",` **after** `"cupdlpx",` and before `"pips",`. |

#### 2.1.1 Class header

```python
class cuOpt(Solver[None]):
    """
    Solver subclass for the NVIDIA cuOpt solver. cuOpt must be installed
    with working GPU support for usage. Install it with
    ``pip install "linopy[cuopt]"`` (Linux only, CUDA 12 driver >= 525.60.13,
    compute capability >= 7.0).

    The full list of solver options is documented at
    https://docs.nvidia.com/cuopt/ and can be listed at runtime with
    ``cuopt.linear_programming.solver_settings.solver_settings.get_solver_parameter_names()``.
    Option names are lower-case and snake_case.

    Some example options are:

    * method : 3 (Barrier) by default in linopy - 0 (Concurrent), 1 (PDLP) and
      2 (DualSimplex) are alternatives. linopy does not use cuOpt's own default
      of 0, which crashes the process on repeated solves (see Notes).
    * time_limit : inf by default.
    * log_to_console : True by default.
    * absolute_primal_tolerance, relative_primal_tolerance, ... : 1e-4 by default.

    Notes
    -----
    Duals are returned in linopy's (HiGHS's) sign convention without
    transformation. Maximisation problems are handed to cuOpt as the equivalent
    minimisation, because cuOpt 26.08 returns negated duals for maximised
    models that its presolve solves outright; the objective, the duals and the
    MIP bound are negated back here.

    cuOpt's own default method (0, Concurrent) segfaults on the second or third
    solve in one process for models above roughly 1300 variables, so linopy
    defaults to method 3 (Barrier) instead.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "cuOpt"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.DIRECT_API,
            SolverFeature.GPU_ACCELERATION,
            SolverFeature.GPU_ONLY,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.SEMI_CONTINUOUS_VARIABLES,
            SolverFeature.MIP_DUAL_BOUND_REPORT,
            # SolverFeature.QUADRATIC_OBJECTIVE is added by WP-F (QP)
        }
    )

    # cuOpt's own default (0 = Concurrent) segfaults on the 2nd-3rd solve in a
    # process for models above ~1300 variables. Barrier is GPU-resident, stable
    # across repeated solves, the most accurate of the single methods, and the
    # method cuOpt forces for quadratic objectives anyway.
    _DEFAULT_METHOD: ClassVar[int] = 3
```

`supports_persistent_update` and `supports_sign_update` stay at their `False` defaults —
`DataModel` has no in-place update API, and `Problem.reset_solved_values()` rebuilds anyway
(`problem.py:1650-1651`).

ClassVar and method ordering follow `cuPDLPx` exactly (§6.6a): `display_name`, `features`,
then `is_available`, `_license_probe`, `_build_file`, `_run_file` *(not overridden)*,
`_build_direct`, `_build_solver_model`, `_run_direct`, `_solve`, `_set_solver_params`.

#### 2.1.2 The lazy import

```python
cupdlpx = _LazyModule("cupdlpx")
# Only cuopt.linear_programming is ever touched: importing cuopt.routing
# installs a global sys.excepthook that writes error_log.txt into the cwd
# (routing/vehicle_routing_wrapper.pyx:144-160).
cuopt = _LazyModule("cuopt")
```

**No `# type: ignore`.** `warn_unused_ignores = true` (`pyproject.toml`, `[tool.mypy]`) and cuOpt
ships no `py.typed`, so with `ignore_missing_imports = true` the `import cuopt` in `TYPE_CHECKING`
resolves to an untyped module and the assignment is accepted — the same reason the `cupdlpx` line
carries no ignore, while `gurobipy`/`highspy` (which have stubs) do.
**Named contingency:** if `mypy` disagrees, add `# type: ignore[assignment]` matching the
`gurobipy` line at `:296` and **paste mypy's message** in the WP report. Do not silence mypy any
other way.

#### 2.1.3 `is_available` / `_license_probe`  (G7)

```python
    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("cuopt") and _cuda_device_available()

    @classmethod
    def _license_probe(cls) -> None:
        cuopt.linear_programming.DataModel()
        cuopt.linear_programming.SolverSettings()
```

The probe **does not solve.** `_has_module` uses `importlib.util.find_spec` and therefore never
executes `cuopt/__init__.py`, so constructing the two native objects is the first thing that proves
the compiled extension and its CUDA libraries actually load — it is not vacuous. Constructing a
`DataModel` allocates **no** device memory (exploration item 9: the CUDA context appears at the
first `Solve`), so this costs nothing at test-collection time. `test_optimization.py` and
`test_solvers.py` parametrize on `licensed_solvers` at import time; a solve there would create a
CUDA context in **every** pytest process, including CPU-only runs. Exceptions are caught by
`Solver.license_status`.

#### 2.1.4 `_build_file` — build directly, leak nothing  (G1, B3, B4, B18)

```python
    def _build_file(self, **build_kwargs: Any) -> None:
        if self.io_api is not None and self.io_api not in FILE_IO_APIS:
            raise ValueError(
                f"Keyword argument `io_api` has to be one of {IO_APIS} or None"
            )
        logger.warning(
            "cuOpt does not support file IO. Building the model through the "
            "direct API instead; pass io_api='direct' to skip creating the "
            "problem file."
        )
        # Model.solve creates the problem file before building and unlinks only
        # what it gets back through `_problem_fn` (model.py:2141-2142), so hand
        # the unused file back rather than leaving it in solver_dir.
        problem_fn = build_kwargs.pop("problem_fn", None)
        if problem_fn is not None:
            self._problem_fn = Path(problem_fn)
        build_kwargs.pop("slice_size", None)
        build_kwargs.pop("progress", None)
        self._build_direct(**build_kwargs)
```

Why `_build_file` and not `_run_file`:

* **It has to work at all.** `test_solvers.py:48` and `:209` call `simple_model.solve(solver)` with
  no `io_api`, parametrized over `licensed_solvers`, and assert success. `Model.solve` routes
  `io_api=None` to `_build_file`; a `_run_file` that raises `NotImplementedError` makes both red
  under `--run-gpu`. It is also the DoD line "`model.solve("cuopt")` solves LP and MILP models
  correctly". This wart is invisible today only because cuPDLPx is not installed on this machine.
* **It is cheaper than the base plan's `_run_file` fallback**, which writes a full LP file and then
  discards it — 0.23 s and 7.7 MB at 10⁵ variables (exploration Q0).
* **Dispatch is mechanical, not hopeful.** `_build_direct` sets `self.io_api = "direct"`, and
  `Solver.solve` dispatches on `if self.io_api == "direct" or self.solver_model is not None`
  (`solvers.py:858`) → `_run_direct`. Verified by reading both.
* **`_run_file` is left unoverridden.** It is unreachable on this path, and the base
  `NotImplementedError` (`solvers.py:979-983`) is the correct message if it ever is reached.
* **B18:** the io_api guard is re-stated because the base `_build_file` (`solvers.py:765-768`) is
  the only place `Solver.from_name(..., io_api="bogus")` is validated. `Model.solve` validates
  earlier (`model.py:2022-2024`), so `test_non_supported_solver_io` (`test_optimization.py:1146`)
  passes either way — but dropping the guard would silently weaken the low-level API contract.
* **B4:** with `_problem_fn` set, `Model.solve`'s `finally` block (`model.py:2145-2147`) unlinks the
  empty `linopy-problem-*.lp` under the default `keep_files=False`. Setting `_problem_fn` is chosen
  over unlinking inside the override because a user may pass `problem_fn=` explicitly, and deleting
  a path the user named would be destructive. ~~With `keep_files=True` an empty file remains~~
  **(A2, 2026-08-22, measured by WP-E on `a2db13f`): `keep_files=True` never gets that far — it
  RAISES `NotImplementedError`**, because `Model.solve` (`model.py:2098-2106`) suppresses the
  solution file only when the solver declares `SOLUTION_FILE_NOT_NEEDED` *and* `keep_files=False`;
  with `keep_files=True` it passes a `solution_fn`, which cuOpt refuses (C-R25). cuPDLPx has the
  byte-identical guard, so this is pre-existing linopy behaviour (KI7 in
  `logs/45-known-issues.md`, B16 in `logs/40-blockers.md`) — recorded, never fixed (I13). The docs
  state the measured behaviour: `keep_files=True` is unsupported for cuOpt and raises.

#### 2.1.5 `_build_direct`

Mirrors `cuPDLPx._build_direct` (`:4227-4244`) with the MIQP guard replacing the QP/MILP guard:

```python
    def _build_direct(self, **kwargs: Any) -> None:
        model = self.model
        assert model is not None
        if model.type in ("MIQP", "IQP"):
            msg = (
                "cuOpt does not support quadratic objectives together with "
                "integer variables. Use a solver that supports MIQP "
                "(gurobi, xpress, mosek)."
            )
            raise NotImplementedError(msg)
        if kwargs.get("explicit_coordinate_names"):
            warnings.warn(
                "cuOpt does not support named variables/constraints. "
                "The explicit_coordinate_names parameter is ignored.",
                UserWarning,
                stacklevel=2,
            )
        self.solver_model = self._build_solver_model(model)
        self.io_api = "direct"
        self.sense = model.sense
        self._cache_model_labels(model)
```

`set_names` is accepted and ignored (absorbed by `**kwargs`): cuOpt's
`set_variable_names`/`set_row_names` cost one Python list per entity and buy nothing, because the
solution is indexed positionally. Documented in `doc/gpu-acceleration.rst`.
The MIQP guard must be a *pre*-check: cuOpt does not raise on MIQP, it returns `NoTermination`,
`obj=0.0`, an **empty** solution array and `error_status=4`
(`"Presolve does not support optimization_problem with a quadratic objective"`), which is
indistinguishable from "the solver failed" (exploration item 12).
`model.type` is `"MIQP"` when quadratic + (integers ∪ binaries ∪ semi-continuous) + continuous, and
`"IQP"` when there are no continuous variables (`model.py:1666-1679`).

#### 2.1.6 `_build_solver_model` (staticmethod, so `to_cuopt` reuses it via `from_model`)

```python
    @staticmethod
    def _build_solver_model(model: Model) -> Any:
        """Build a cuopt DataModel that mirrors the linopy `model`."""
        M = model.matrices
        # cuOpt is always handed a minimisation: set_maximize(True) is the only
        # route onto a presolve path that returns negated duals (verified
        # against live HiGHS on 21 models). `_solve` negates the results back.
        # This is NOT a dual-convention fix -- cuOpt's duals already match
        # HiGHS, which is linopy's convention. Do not replace it with
        # cuPDLPx's `if MAXIMIZE: dual = -dual` (solvers.py:4385-4386).
        sign = -1.0 if model.objective.sense == "max" else 1.0

        lower = np.where(
            np.logical_or(np.equal(M.sense, ">"), np.equal(M.sense, "=")), M.b, -np.inf
        )
        upper = np.where(
            np.logical_or(np.equal(M.sense, "<"), np.equal(M.sense, "=")), M.b, np.inf
        )
        A = None if M.A is None else M.A.tocsr()
        if A is None or A.nnz == 0:
            # cuOpt returns NoTermination and an empty solution for a
            # constraint matrix without nonzeros. Repeat one variable's own
            # bounds as a redundant row: it leaves the feasible set untouched,
            # and unlike a row with two infinite bounds it does not make a
            # quadratic objective fail with NumericalError. The pad row is the
            # last one, so its dual is sliced off again in `_solve`.
            finite = np.flatnonzero(np.isfinite(M.lb) | np.isfinite(M.ub))
            if not finite.size:
                msg = (
                    "cuOpt cannot solve a model that has no constraints and no "
                    "finite variable bounds. Add a constraint or a bound, or "
                    "use a CPU solver (highs, gurobi)."
                )
                raise NotImplementedError(msg)
            j = int(finite[0])
            pad = csr_array(([1.0], ([0], [j])), shape=(1, len(M.c)))
            A = pad if A is None else vstack([A, pad], format="csr")
            lower = np.append(lower, M.lb[j])
            upper = np.append(upper, M.ub[j])

        dm = cuopt.linear_programming.DataModel()
        dm.set_csr_constraint_matrix(A.data, A.indices, A.indptr)
        dm.set_constraint_lower_bounds(lower)
        dm.set_constraint_upper_bounds(upper)
        dm.set_variable_lower_bounds(M.lb)
        dm.set_variable_upper_bounds(M.ub)
        dm.set_objective_coefficients(sign * M.c)
        # cuOpt knows 'C', 'I' and 'S'; every other character is silently taken
        # as continuous, so linopy's 'B' must be mapped explicitly. Binary
        # bounds are already 0/1 in M.lb/M.ub -- cuOpt does not clamp 'B'.
        dm.set_variable_types(np.where(M.vtypes == "B", "I", M.vtypes).astype("<U1"))
        return dm
```

WP-F adds three lines here for `Q` (§5d). Notes:

* `set_row_types` is deliberately never called: the bounds route covers `<`/`>`/`=` and wins if both
  are set (`data_model.py:468-470`), and there is no `'R'` (ranged) row type at all.
* `set_objective_offset` is never called: `Objective` raises
  `ValueError("Constant values in objective function not supported.")` (`objective.py:192-193`), so
  there is nothing to offset. This also removes M2's offset-negation worry entirely.
* `np.inf` / `-np.inf` are passed straight through — verified working, no sentinels
  (exploration item 5).
* `M.vtypes` is `<U1` and `M.sense` is `dtype=object`; both are fine for `np.where`/`np.equal`, and
  `<U1` is exactly what `set_variable_types` wants (a `list`/`str`/`bytes` raises).
* **A list, str or bytes must never be passed** to `set_variable_types`, and user input must never
  reach it: `'i'`, `'Z'`, `73` are all silently taken as continuous.

#### 2.1.7 `_run_direct` / `_solve`

`_run_direct` forwards to `_solve(self.solver_model, ..., io_api=self.io_api, sense=self.sense)`
exactly as `cuPDLPx._run_direct` does (`:4284-4301`), **including `log_fn`** — cuOpt's settings
object is created per solve, so `log_fn` must reach `_solve`.

`_solve` (numpydoc docstring modelled on `cuPDLPx._solve`, `:4313-4338`):

```python
        # cuOpt's LP and MILP status enums are distinct IntEnums whose members
        # compare and hash equal by value (LP.PrimalInfeasible == MILP.Infeasible
        # == 2), so one dict keyed on the enum members would silently merge them.
        # Key on the status name and pick the map from the problem category.
        LP_CONDITION_MAP: dict[str, TerminationCondition] = {...}    # §4
        MILP_CONDITION_MAP: dict[str, TerminationCondition] = {...}  # §4

        if warmstart_fn is not None:
            raise NotImplementedError("Warmstarting is not yet implemented for cuOpt.")
        if basis_fn is not None:
            logger.warning("Basis files are not supported by cuOpt. Ignoring.")
        if solution_fn is not None:
            raise NotImplementedError(
                "Solution file output is not yet implemented for cuOpt."
            )

        settings = cuopt.linear_programming.SolverSettings()
        self._set_solver_params(settings, log_fn)
        sol = _run_cuopt_with_keyboard_interrupt(
            lambda: cuopt.linear_programming.Solve(dm, settings)
        )

        is_mip = sol.get_problem_category().name != "LP"
        condition_map = MILP_CONDITION_MAP if is_mip else LP_CONDITION_MAP
        termination_condition = condition_map.get(
            sol.get_termination_reason(), TerminationCondition.unknown
        )
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = sol.get_termination_reason()
        if sol.get_error_status():  # ErrorStatus.Success == 0
            logger.error(f"cuOpt reported an error: {sol.get_error_message()}")

        sign = -1.0 if sense == "max" else 1.0
        n_cols = 0 if self._vlabels is None else int(self._vlabels.size)
        n_rows = 0 if self._clabels is None else int(self._clabels.size)

        def get_solver_solution() -> Solution:
            primal = np.asarray(sol.get_primal_solution(), dtype=float)
            if primal.size != n_cols:
                # cuOpt returns an empty primal for NoTermination and for some
                # limit terminations; scattering it would misalign every label.
                msg = (
                    f"cuOpt returned {primal.size} primal values for {n_cols} "
                    f"variables ({sol.get_termination_reason()})."
                )
                if primal.size:
                    raise ValueError(msg)
                logger.error(msg)
                return Solution()
            objective = sign * float(sol.get_primal_objective())
            if is_mip:
                # get_dual_solution() raises for a MIP solution, and an empty
                # array cannot be scattered (values_to_lookup_array would raise
                # IndexError), so return it unscattered. Model.assign_result
                # guards with `if len(result.solution.dual)` (model.py:2206).
                dual = np.array([], dtype=float)
            else:
                dual = sign * np.asarray(sol.get_dual_solution(), dtype=float)
                # slices off the padded row's dual, if any (§2.1.6)
                dual = _solution_from_labels(
                    dual[:n_rows], self._clabels, self._n_cons
                )
            primal = _solution_from_labels(primal, self._vlabels, self._n_vars)
            # Reduced costs are deliberately not read: linopy has no surface for
            # them, and cuOpt returns `-c - A'y` for maximised models with `<=`
            # rows (logs/45-known-issues.md).
            return Solution(primal, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)
```

**Report.** `runtime = float(sol.get_solve_time())`; for a MIP additionally
`mip_gap = stats["mip_gap"]`, `dual_bound = sign * stats["solution_bound"]` and
`simplex_iterations = int(stats["num_simplex_iterations"])` from `sol.get_milp_stats()`
(documented at `solution/solution.py:368-405`; it raises for an LP solution) — each read inside
`contextlib.suppress(Exception)`, exactly as `Highs._solve` does (`:1884-1892`). **`get_milp_stats()`
may legitimately return `mip_gap=inf` and `solution_bound=-inf`** (measured on a semi-continuous
model terminating `FeasibleFound`): that is the *correct* report when no bound was proven, and under
M2 a `max` model negates it to `+inf`. It is not a bug and must not be filtered, clamped or replaced
with `None` — `SolverReport` fields are `float | None`, and V1-4's `is not None` threshold is
satisfied by `±inf`. LP iteration counts
are **not** reported: `lp_stats["nb_iterations"]` is method-dependent and is neither
`simplex_iterations` nor `barrier_iterations`; mislabelling them would be worse than omitting them.
`lp_stats["dual_residual"]` is never surfaced (measured at `51.0` on a QP solved to optimality —
meaningless under QP).

Finish with `self.io_api = io_api` and `self._make_result(status, solution, solver_model=dm,
report=SolverReport(...))`, mirroring `cuPDLPx._solve`'s tail.

**B17 — why `self._vlabels.size`, not `len(self._vlabels or ())`.** Measured on the installed
numpy 2.4.6:

```
$ .venv/bin/python -W error -c "import numpy as np; print(len(np.array([1.0,2.0]) or ()))"
ValueError: The truth value of an array with more than one element is ambiguous.
$ .venv/bin/python -W error -c "import numpy as np; print(len(np.array([]) or ()))"
ValueError: The truth value of an empty array is ambiguous.
```

The base plan's `len(self._vlabels or ())` idiom raises on **every** solve. `int(arr.size)` with an
explicit `None` guard is the replacement. (`_solution_from_labels` itself is safe with empty labels:
`values_to_lookup_array(np.array([]), np.array([]), size=10)` returns a 10-element NaN array —
measured.)

#### 2.1.8 `_set_solver_params`  (G3, B9)

```python
    def _set_solver_params(self, settings: Any, log_fn: Path | None = None) -> None:
        if log_fn is not None:
            self.solver_options["log_file"] = path_to_string(log_fn)
            logger.info(f"Log file at {self.solver_options['log_file']}")
        options: dict[str, Any] = {"method": self._DEFAULT_METHOD, **self.solver_options}
        with contextlib.suppress(TypeError, ValueError):
            if int(options["method"]) == 0:
                logger.warning(
                    "cuOpt's Concurrent method (method=0) can crash the process on "
                    "repeated solves of models with more than about 1300 variables. "
                    "Consider method=3 (Barrier) or method=1 (PDLP)."
                )
        for k, v in options.items():
            # cuOpt types most parameters as int and rejects a Python bool for
            # them inside Solve(); the bool-typed ones accept 0/1.
            if isinstance(v, bool):
                v = int(v)
            try:
                settings.set_parameter(k, v)
            except ValueError as e:
                msg = f"cuOpt rejected the solver option {k}={v!r}: {e}"
                raise ValueError(msg) from e
```

* **Blanket `bool → int`.** Measured: `set_parameter` does **not** validate values —
  `set_parameter("presolve", False)` succeeds and `Solve()` then raises
  `ValueError: Parameter presolve value False is not an integer`. At `Solve()` time
  `log_to_console=0`, `crossover=1` and `infeasibility_detection=0` are all accepted, while
  `method=True` is rejected. So ints are accepted everywhere bools are, and only ints are accepted
  by int-typed parameters — the blanket rule is both the simplest and the correct one. This replaces
  the base plan's hand-maintained `_BOOL_PARAMETERS` allowlist (which would rot) and rejects
  plan-2's `get_parameter` type-sniffing (works, but extra API surface for no gain) and plan-3's
  `except ValueError: retry as int` (dead code — `set_parameter` never raises on a value).
* **Unknown *names*** do raise eagerly at `set_parameter`, with
  `ValueError: Invalid parameter. Please check documentation` — a message that does **not name the
  offending parameter**. Names are case-sensitive snake_case and users arrive with Gurobi/cuPDLPx
  habits (`TimeLimit`), so the exception is re-raised naming the option, `from e`.
* **Bad *values*** are raised late, from inside `Solve()`, with messages that already name parameter
  and value (`Parameter time_limit value -5.0 out of range`). They are **not** wrapped — they never
  pass through this `try`. They surface from the interrupt helper's worker thread, which re-raises
  them (§2.1.10).
* **`log_fn` wins over a user-supplied `log_file`**, matching `Highs._set_solver_params`
  (`:1774-1784`) — the one repo precedent, and cuOpt's parameter is spelled identically. No
  `dup2`/`redirect_stdout` machinery: `contextlib.redirect_stdout` captures 0 chars, `log_file`
  captures 1437 (banner + iteration log). `log_to_console` keeps cuOpt's default (`True`).
* Known `log_file` semantics, recorded in the docs: it **truncates** on a second solve, an
  unwritable path raises a clear `RuntimeError` from `Solve()`, and `log_file=''` writes nothing.

#### 2.1.9 Module-level device probe  (B5, B6, R2, R3)

```python
_CUDA_PROBE_SNIPPET = (
    "from cuda.bindings import runtime;"
    "e, c = runtime.cudaGetDeviceCount();"
    "raise SystemExit(0 if (int(e) == 0 and bool(c)) else 1)"
)


@functools.cache
def _cuda_device_available() -> bool:
    """
    True if a usable CUDA device is visible.

    ``import cuopt`` succeeds on hosts without a GPU by design -- its
    ``__init__`` defers every device-touching import so a remote solve can be
    configured -- so an import probe alone would advertise cuOpt on machines
    where every solve fails with a remote-execution error.

    The probe runs in a subprocess on purpose. Any in-process CUDA call
    initialises a CUDA context in the caller, and a context created before
    ``fork()`` makes every cuOpt solve in the child fail. ``is_available`` runs
    on first access to ``linopy.available_solvers``, so an in-process probe
    would break cuOpt -- and every other CUDA library -- in multiprocessing and
    dask workers.
    """
    try:
        completed = sub.run(
            [sys.executable, "-c", _CUDA_PROBE_SNIPPET],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, sub.SubprocessError):
        return False
    if completed.returncode:
        logger.warning(
            "cuOpt is installed but no usable CUDA device was found; cuOpt is "
            "not available. It requires an NVIDIA GPU of compute capability "
            "7.0 or higher and driver 525.60.13 or newer."
        )
        return False
    return True
```

Placement: next to `_has_module` (`:307-315`), where the other probes live. `subprocess as sub`,
`sys` and `functools` are already imported (`solvers.py:16-17`, `:10`) — no new imports.

Why this exact shape:

* **Measured fork poisoning.** With an in-process `cudaGetDeviceCount()` in the parent, a forked
  child fails every cuOpt solve with
  `InputValidationError: A CPU-memory problem requires remote execution. Set CUOPT_REMOTE_HOST…`
  (child exit status 768); without it the child solves (`Optimal 2.0`, exit 0). Merely *importing*
  `cuda.bindings.runtime` is fork-safe; calling into it is not.
* **Measured subprocess cost and correctness:** `True` in **0.280 s / 0.274 s**, `returncode 1`
  under `CUDA_VISIBLE_DEVICES=''`, and the forked child still solves (exit 0). `functools.cache`
  makes it once per process, and the `and` short-circuit means it only runs when `cuopt` is actually
  installed.
* **`bool(count)`, not `count > 0`.** `cudaGetDeviceCount()` returns
  `(cudaErrorNoDevice, None)` — `count` is `None`, not `0` — so `None > 0` would raise `TypeError`.
* **Fail closed.** A probe that cannot run reports "not available". Failing *open* would advertise
  cuOpt on exactly the machines §4.6 c3 exists for. Fail-closed is only defensible with the positive
  branch pinned by a test, so it is (verification V0-5).
* If the lead ever rejects `subprocess` in library code, the pre-decided fallback is the CUDA-free
  heuristic (`CUDA_VISIBLE_DEVICES` + `/dev/nvidia[0-9]*`, also measured fork-safe) — weaker (it
  cannot see driver version or compute capability) but sufficient for §4.6 c3's two scenarios. That
  substitution requires a new D-entry; a coding agent may not make it unilaterally.

#### 2.1.10 Interrupt helper

```python
def _run_cuopt_with_keyboard_interrupt(solve: Callable[[], Any]) -> Any:
    """
    Run a cuOpt solve while keeping Ctrl-C responsive.

    cuOpt exposes no cancel API and defers SIGINT for the entire duration of
    the C++ solve, so the call runs in a worker thread while the main thread
    waits. Note that the GPU work continues in the background until the process
    exits; ``time_limit`` is the only hard bound cuOpt offers.
    """
    finished = threading.Event()
    out: list[Any] = []
    run_error: BaseException | None = None

    def _target() -> None:
        nonlocal run_error
        try:
            out.append(solve())
        except BaseException as exc:  # noqa: BLE001
            run_error = exc
        finally:
            finished.set()

    threading.Thread(target=_target, name="linopy-cuopt-solve", daemon=True).start()
    while not finished.wait(0.1):
        pass
    if run_error is not None:
        raise run_error
    return out[0]
```

Measured: SIGINT during a bare `Solve()` is deferred **52.9 s** (raised inside `create_solution`,
after the solve already finished); in a worker thread the latency is **0.00 s**, and the process
exits cleanly (`exit=0`) even while the daemon thread is still inside cuOpt's C++ code. cuOpt
installs no SIGINT handler and has **no** in-process cancel API, so this is
`_run_highs_with_keyboard_interrupt` (`solvers.py:189-252`) with the `cancelSolve()` step removed —
hence a separate helper rather than a reuse. The helper takes a **zero-argument callable** so the
seam is testable without a GPU (`test/test_cuopt_interrupt.py`, CPU, ungated). Re-raising
`run_error` is what surfaces cuOpt's late value-validation `ValueError`s from `Solve()`.

### 2.2 `linopy/io.py`  [≈ +10]

* `:32-37` `TYPE_CHECKING`: add
  `from cuopt.linear_programming import DataModel as cuoptDataModel`
  (mirrors `from cupdlpx import Model as cupdlpxModel`).
* after `to_cupdlpx` (`:792-795`):

```python
def to_cuopt(m: Model) -> cuoptDataModel:
    """
    Build the cuopt DataModel for `m`.

    For a maximisation model the returned DataModel is the equivalent
    minimisation (negated objective coefficients, `maximize` unset); see
    :class:`linopy.solvers.cuOpt`.
    """
    solver = solvers.cuOpt.from_model(m, io_api="direct")
    return solver.solver_model
```

The docstring paragraph is required, not decorative (G6): `to_cuopt` is public API and returning a
minimising model for a `max` linopy model is surprising unless stated.

### 2.3 `linopy/model.py`  [≈ +2]

* `:70-80` `from linopy.io import (...)`: add `to_cuopt,` (alphabetical: after `shallowcopy`/
  `to_block_files`, before `to_cupdlpx`).
* `:2517`: add `to_cuopt = to_cuopt` immediately before `to_cupdlpx = to_cupdlpx`, following the
  file's blank-line-between-bindings style.

### 2.4 `linopy/solver_capabilities.py`  [±0 — verification only]

**No edit.** `SOLVER_REGISTRY` is a `_LazyRegistry` deriving `SolverInfo` from `SolverName` plus the
class's declared features; `SolverName.cuOpt.name == "cuOpt" == cuOpt.__name__` satisfies its
`getattr(solvers, SolverName(name).name)` resolution. Proved by `test_capability_shim_round_trips`
(`test_solvers.py:481`, already parametrized over every `SolverName` × every `SolverFeature`) plus
one explicit `SOLVER_REGISTRY["cuopt"]` assertion in `test_cuopt.py`. Editing the shim would be a
drive-by (I4) — charter §5 row 7 says so explicitly.

### 2.5 `linopy/constants.py`  [±0]

**No edit.** Every cuOpt status maps onto an existing `TerminationCondition` member (§4).

### 2.6 `test/conftest.py`  [±0 — verification only]

**No edit.** `pytest_collection_modifyitems` (`:53-79`) is generic on
`solver_supports(solver, SolverFeature.GPU_ONLY)` for any test whose callspec carries a `solver`
param; cuPDLPx is not named. Declaring `GPU_ONLY` auto-marks and auto-skips every such cuOpt cell.
`LINOPY_RUN_GPU_TESTS` is already exported at `:46-50` for the import-time case, which is what
`test_cuopt.py`'s module gate uses. Proof: `pytest --collect-only -q -k cuopt` with and without
`--run-gpu`, pasted in the WP-D report.

### 2.7 `test/test_optimization.py`  [≈ +6 / −0]

Three cuPDLPx-analogue touch points (charter cites `:562`, `:685`, `:1110`):

1. `:551-563` `time_limit_option`: add `"cuopt": {"time_limit": 1},`. **Mandatory** — the dict is
   indexed unconditionally at `:568` (`**time_limit_option[solver]`), so its absence is a `KeyError`
   in every parametrized cell.
2. `:682-689` `test_infeasible_model`: **no cuOpt skip.** The cuPDLPx skip exists because cuPDLPx
   *hangs* on some unbounded problems; cuOpt returns `PrimalInfeasible` on an infeasible model
   (exploration item 1), and both `infeasible` and `infeasible_or_unbounded` satisfy the test's
   `"infeasible" in condition` assertion with `SolverStatus.warning`. This is a **prediction**, so
   it is a named verification row (V2-5) with a pre-decided disposition, and the deliberate
   *non*-skip is stated in the WP report so Phase 6 reads it as a decision, not an omission.
3. `:1102-1125` `test_basis_and_warmstart`: extend the skip to cuOpt —
   `if solver in ("cupdlpx", "cuopt")` with a reason naming both and cuOpt's upstream constraint.
   Required: the second solve passes `warmstart_fn`, which raises `NotImplementedError` (§5h).

**No `method_options` entry** (R11): `test_solver_method_options` (`:574-583`) asserts
`np.isclose(objective, 3.3)` at numpy's default `rtol=1e-5`, and PDLP was measured drifting a
2-variable objective to `2.80013079` versus `2.8` — `4.7e-5` relative, ~5× outside that. The dict is
keyed by `if solver in method_options`, so absence is legal.

No other change: cuOpt joins `params` as `("cuopt", "direct", False)` through `direct_solvers`, and
joins the MIP / QP / GPU-tolerance groups through its features
(`feasible_mip_solvers`, `quadratic_solvers`, `gpu_solvers` → `GPU_SOL_TOL`).

### 2.8 `test/test_solvers.py`  [≈ +6]

Feature-matrix rows at `:448-465`, inserted after the `cuPDLPx` block:
`(solvers.cuOpt, SolverFeature.DIRECT_API, True)`, `(GPU_ONLY, True)`,
`(INTEGER_VARIABLES, True)`, `(SEMI_CONTINUOUS_VARIABLES, True)`, `(SOS_CONSTRAINTS, False)`, and
`(QUADRATIC_OBJECTIVE, True)` added by WP-F.

**Pre-authorised, contingent edit only** (B14/G5): if and only if
`test_from_name_set_names_false` (`:99-103`, `pytest.approx` default `rel=1e-6` on `3.3`, `-0.1`,
`1.7`) misses under `--run-gpu`, see §8 "Pre-decided contingencies" for the exact permitted edit.

### 2.9 New `test/test_cuopt.py`  [≈ +170] and `test/test_cuopt_interrupt.py`  [≈ +40]

§8.

### 2.10 `pyproject.toml`  [≈ +6]

New `cuopt` extra as a sibling of `solvers`, inserted immediately after the `solvers = [...]` block
(`:113`) — §5i.

### 2.11 Docs  [≈ +115]

§9: `doc/gpu-acceleration.rst`, `doc/prerequisites.rst`, `doc/contributing.rst`, `doc/index.rst`,
`doc/api.rst`, `doc/release_notes.rst`, `README.md`, `examples/using-solvers.ipynb` (markdown cell
only).

**Total ≈ +605 / −5 across 13 files** — 0.78× PR #516's +771/−62 over 15 files, comfortably inside
the §9 yardstick. (Base plan's +560, plus ≈ +45 for the subprocess probe, the `_build_file`
override, the pad-row branch and the `api.rst` rows, minus ≈ −10 for dropping the `_license_probe`
solve and the bool allowlist.)

---

## 3. Feature-flag set

| Feature | Declared? | Justification |
|---|---|---|
| `DIRECT_API` | **yes** | `DataModel` + `Solve` is the entry point (§1). |
| `GPU_ACCELERATION` | **yes** | Solves on the GPU (Barrier and PDLP are device-resident). |
| `GPU_ONLY` | **yes** | No CPU-only mode is reachable in-process; a GPU-less host cannot solve at all (exploration item 13). Also the switch that makes `conftest` auto-skip cuOpt without `--run-gpu` and that puts cuOpt into `gpu_solvers` → `GPU_SOL_TOL`. |
| `SOLUTION_FILE_NOT_NEEDED` | **yes** | The direct path returns arrays. Keeps `Model.solve` from allocating a solution file (`model.py:2097-2102`). |
| `INTEGER_VARIABLES` | **yes** | MILP verified (charter §4.3; exploration items 1 and 6: `set_variable_types(['I','I'])` → `cat=MIP Optimal`). |
| `SEMI_CONTINUOUS_VARIABLES` | **yes** | cuOpt genuinely supports `'S'` — measured obj `0.0` vs `5.0` on a discriminating model, on both entry points (exploration item 10) — and `M.vtypes` already emits `'S'`, so the code cost is zero. Without the flag, `Solver._validate_model` (`solvers.py:731-737`) raises *"Solver cuopt does not support semi-continuous variables"* for a model cuOpt demonstrably solves, i.e. linopy would state a falsehood. **The flag brings zero automatic coverage**: `test/test_semi_continuous.py` has no `solver` parametrization (hard-coded gurobi/highs) and no GPU gating, so the coverage must live in `test_cuopt.py` (V1-7). **Consequence a user must be told (documented under limitations):** a model containing even one `'S'` variable is `cat=MIP` for cuOpt — measured, an all-`'C'` array leaves an LP as `cat=LP`, but adding one `'S'` gives `cat=MIP FeasibleFound` and `get_dual_solution()` then raises. So `is_mip` is `True`, duals are correctly refused, and the MILP status map applies to what the user may think of as an LP. `model.type` is `"MILP"` for such a model, which is also why §2.1.5's `("MIQP","IQP")` guard catches quadratic + semi-continuous. |
| `MIP_DUAL_BOUND_REPORT` | **yes** | Measured on the exact `milp_model` fixture whose assertion the flag activates (`test_optimization.py:826-834`): `milp_stats = {'mip_gap': 0.0, 'solution_bound': 100.0, …}`. `solution_bound` is negated for `max` under M2. **Contingency:** if it ever comes back `None`, the flag is dropped and the miss reported verbatim — never by weakening `test_optimization.py:834`. |
| `QUADRATIC_OBJECTIVE` | **yes, in WP-F** | QP verified end-to-end against live HiGHS (exploration item 12). Declared in the QP commit so the LP/MILP core is never green-by-omission. Auto-registers cuOpt in `QUADRATIC_SOLVERS` (`:4426`) — no manual edit (charter §4.5). |
| `LP_FILE_NAMES` | **no** | No file io_api (§1.2); the flag only governs name writing into LP files. |
| `READ_MODEL_FROM_FILE` | **no** | §1.2. This is the flag that would admit cuOpt to `file_io_solvers`. |
| `IIS_COMPUTATION` | **no** | No IIS/infeasibility-diagnosis API on `DataModel` or `Solution`. `test_infeasible_model` then correctly expects `NotImplementedError` from `compute_infeasibilities()` (`:709-710`). |
| `SOS_CONSTRAINTS` | **no** | Genuinely absent: no `sos`/`ordered` attribute anywhere, cuOpt's own parser rejects an `SOS` section (`LP section 'SOS' is not supported`), and its test suite comments "SOS is explicitly out of scope". `_validate_model` then raises with the reformulation hint. |
| `INDICATOR_CONSTRAINTS` | **no** | Genuinely absent (`grep -rni indicator` over the installed package finds only an unrelated test list). |
| `SOLVER_ATTRIBUTE_ACCESS` | **no** | No per-entity attribute API. `test_solver_attribute_getter` (`:1152-1163`) then expects `NotImplementedError`, which is what linopy raises for every non-Gurobi solver. |
| `supports_persistent_update` | **no** (default) | `DataModel` has no in-place update; `_apply_locked` falls back to a rebuild. |
| `supports_sign_update` | **no** (default) | Follows from the above. |
| `accepted_io_apis` | **not declared** | Verified dead code (§1.3). |

No new `SolverFeature` member is added.

---

## 4. Status-mapping table

Two dicts **keyed on the status name string**, selected by problem category, defined inside `_solve`
(the `Highs` idiom — `CONDITION_MAP` sits inside `Highs._solve` at `:1825-1845`).
`Solution.get_termination_reason()` returns `termination_status.name`
(`solution/solution.py:285-289`), and `get_problem_category().name` is one of `"LP"`, `"MIP"`,
`"IP"` — so `is_mip = ... .name != "LP"` sends both `MIP` and `IP` to the MILP map.

**Why names and not enum members (G2).** `LPTerminationStatus` and `MILPTerminationStatus` are
distinct `IntEnum`s whose members compare **and hash** equal by value —
`LP.PrimalInfeasible == MILP.Infeasible == 2`, `LP.DualInfeasible == MILP.Unbounded == 3` — so a
single dict keyed on enum members silently keeps one key. Two enum-keyed dicts would also be
correct, but name keys additionally avoid importing
`cuopt.linear_programming.solver.solver_wrapper`, a `.pyx` module that `solver/__init__.py` does not
re-export. The comment in §2.1.7 exists so nobody "simplifies" this back.

Unmapped → `TerminationCondition.unknown` via `.get(..., unknown)` (the `Highs` default at
`:1855-1857`), **not** cuPDLPx's raw-code passthrough (`:4374-4376`).

| Category | cuOpt status (value, name) | `TerminationCondition` | `SolverStatus` | Basis |
|---|---|---|---|---|
| LP | `0` `NoTermination` | `internal_solver_error` | error | cuOpt's catch-all for malformed input, MIQP, zero-nnz matrices **and** the GPU-memory ceiling; empty primal, `obj=nan`. **Must not be `unknown`:** `safe_get_solution`'s `unknown` branch parses the solution anyway and flips the status to `ok` (`solvers.py:1191-1197`), which would report a `nan`-objective non-solve as a success. |
| LP | `1` `Optimal` | `optimal` | ok | verified |
| LP | `2` `PrimalInfeasible` | `infeasible` | warning | verified (`x>=2 & x<=1`) |
| LP | `3` `DualInfeasible` | `unbounded` | warning | mapping by definition; **status never observed** across 7 unbounded variants (all returned `11`). Recorded as an untested row in `logs/45-known-issues.md` and in a code comment; **no test asserts it.** |
| LP | `4` `IterationLimit` | `iteration_limit` | ok | verified (`iteration_limit=1`, 400×300 LP) |
| LP | `5` `TimeLimit` | `time_limit` | ok | verified. The primal may come back all-zero or empty; `time_limit` is `SolverStatus.ok` by linopy's own map (same as HiGHS's `kTimeLimit`), and the empty case is handled by the length pre-check in §2.1.7. |
| LP | `6` `NumericalError` | `internal_solver_error` | error | verified (non-PSD `Q`, and the `quadratic_model_unbounded` fixture); `obj=nan` |
| LP | `7` `PrimalFeasible` | `suboptimal` | ok | verified (`first_primal_feasible=True`) — usable but unproven point |
| LP | `11` `UnboundedOrInfeasible` | `infeasible_or_unbounded` | warning | verified; the status cuOpt *actually* returns for unbounded LPs |
| MIP/IP | `0` `NoTermination` | `internal_solver_error` | error | as LP `0` (MIQP lands here with `error_status=4`, and is guarded before the solve anyway) |
| MIP/IP | `1` `Optimal` | `optimal` | ok | verified |
| MIP/IP | `2` `Infeasible` | `infeasible` | warning | verified (`2x == 1`, `x` integer) |
| MIP/IP | `3` `Unbounded` | `unbounded` | warning | mapping by definition; **never observed** (see LP `3`) |
| MIP/IP | `5` `TimeLimit` | `time_limit` | ok | verified (subset-sum, market-split) |
| MIP/IP | `8` `FeasibleFound` | `suboptimal` | ok | verified (knapsack, `node_limit=1`) — feasible, optimality unproven; `suboptimal` is `ok` and the solution *is* usable |
| MIP/IP | `11` `UnboundedOrInfeasible` | `infeasible_or_unbounded` | warning | verified |
| both | anything else | `unknown` | unknown | defensive default for a future cuOpt release |

`ErrorStatus` is **not** mapped to a `TerminationCondition`. It is read alongside and logged
(`logger.error(sol.get_error_message())`) whenever it is not `Success` (`== 0`, so plain truthiness
suffices and no enum import is needed). `status.legacy_status` carries
`sol.get_termination_reason()`, mirroring `status.legacy_status = h.modelStatusToString(...)` in
`Highs`.

**Trap honoured:** a limit *setting* never implies a limit *status* — `iteration_limit=1` on a
knapsack returned `Optimal obj=-2020`. Nothing in `_solve` infers a condition from the options.

---

## 5. Positions on the hard findings

### 5a. Presolve + `maximize` negated duals → adopt M2 (always minimise)

**Position: adopt M2 unconditionally. `set_maximize(True)` is never called.** For a `max` model,
build with `-M.c` (and `-0.5·M.Q`), leave the sense at minimise, then negate the reported objective,
the duals and the MIP `solution_bound` on the linopy side. Duals otherwise pass through
**untransformed**.

Why:

* The bug is real, silent, and reachable from an ordinary linopy model through the natural
  translation: `max|Δdual| = 3.61` with status `Optimal` and a *correct* objective, at
  `n = 2, 4, 10, 40, 200` — all NEGATED. The correlation with `solved_by == SolverMethod.Unset`
  ("Optimal solution found during presolve") is 6/6 exact over a 144-cell sweep.
* **M1** (set `presolve` explicitly) fixes the equality cells but **fails on ranged rows** —
  rejected by measurement, not preference.
* `method=DualSimplex` "fixes" it by leaving the GPU. Rejected.
* **M2 is verified 21/21 against live HiGHS** (5/21 failing without it, 0/21 with it) and it avoids
  the buggy branch *by construction* — which matters because the root cause is an explicit
  exploration Unknown, so a narrower workaround could be defeated by a path nobody constructed.
* Cost: one `sign` scalar used in four places.

Two traps, both handled explicitly:

* **Do not copy cuPDLPx's `if ModelSense == MAXIMIZE: dual = -dual` (`:4385-4386`).** cuOpt's dual
  convention already matches HiGHS, which *is* linopy's convention (`Highs._solve` passes
  `row_dual` through unadjusted, `:1871-1879`). Under M2 the negation is against the sign *we*
  applied, not a convention fix — the code comment in §2.1.6 says exactly that, or the next reader
  will "simplify" it away.
* Everything derived from the objective carries the same sign: `get_primal_objective()` **and**
  `get_milp_stats()["solution_bound"]`. `mip_gap` is relative and is **not** negated.

`maybe_adjust_objective_sign` is still called (as cuPDLPx does) even though it is a no-op for
`io_api="direct"`; it keeps `_solve`'s shape identical to its siblings.

### 5b. Concurrent repeat-solve segfault → default `method=3` (Barrier), never override the user

**Position: linopy injects `method=3` as a default when the user has not set `method`; a user who
explicitly passes `method=0` gets a one-line warning and their choice.**

* Doing nothing is indefensible: `method=0` is a deterministic SIGSEGV on the 2nd–3rd solve above
  ~1300 variables (≥12 fresh processes; sweep `1200 → 8/8 ok`, `1300 → 6 solves then exit 139`,
  `1500 → 2 solves then exit 139`). linopy's own test models sit *below* the threshold, so the suite
  would stay green while a user's second `model.solve("cuopt")` kills the interpreter with no
  traceback.
* Barrier is stable on repeats (25/25 varied solves, device memory flat at 117 MiB) and
  GPU-resident, so `GPU_ACCELERATION` stays an honest claim — unlike `method=2` (DualSimplex),
  which leaves the GPU.
* Barrier is the **most accurate** of the four: dual agreement with HiGHS **1.73e-09 / 2.47e-09**
  versus PDLP's 2.60e-06 / 4.71e-05. This is what makes an `rtol=1e-6` differential test defensible
  instead of a tolerance-widening argument (I12). PDLP at default tolerances disagreed with HiGHS in
  every cell of the 144-cell sweep and moved a 2000×1000 objective by 5.52e-03 relative.
* cuOpt already forces Barrier for any quadratic objective regardless of `method` (all 8
  method × crossover combinations reported `solved_by=Barrier`), so LP, MILP relaxations and QP
  share one code path.
* PDLP carries its own hazards: a segfault with `time_limit=1e-6` + `save_best_primal_so_far`, and
  non-termination on a badly-scaled LP that Barrier solved in seconds.

**Named contingency (G8), so no agent has to improvise.** `method`'s interaction with the MILP
branch-and-bound path was not exercised in Phase 1 (though the selector's probes solved both MILP
fixtures exactly under `method=3`). If a MILP solve fails or `Solve` rejects `method=3` for a MILP,
the coding agent **reports the numbers verbatim** and applies exactly this fallback: pass the
default only when the model has no integrality, and document the MILP repeat-solve hazard in
`doc/gpu-acceleration.rst`'s limitations list. It never falls back to `method=0` and never invents a
third option.

Documented escape hatch, with the measured numbers, in `doc/gpu-acceleration.rst`: `method=1`
(PDLP) for very large sparse LPs (the only method exercised at 10⁶–2×10⁷ variables, peak 12.9 GiB on
the T4); `method=0` not recommended until the upstream crash is fixed. Regression test: V1-8, three
sequential solves of a 2000-variable LP **in a subprocess**, so a regression is a red test rather
than a dead pytest session.

### 5c. `is_available()` — GPU-gated, out-of-process, fail-closed  (D6, B5, B6, B7)

**Position: `is_available()` = `_has_module("cuopt") and _cuda_device_available()`, with
`_cuda_device_available()` implemented as a cached **subprocess** probe that fails closed (§2.1.9).**

This is a deliberate divergence from `cuPDLPx.is_available` (`:4187-4189`) and from
`_AvailableSolvers`' docstring ("Membership means the solver's Python package or binary is
importable", `:4463-4470`). **D6 in `logs/30-decisions.md` is a binding lead ruling that settles
it**, resolving ESC-1 in favour of charter §4.6 constraint 3.

Exact behaviour on a GPU-less machine with `cuopt-cu12` installed:

* `linopy.available_solvers` **omits** `"cuopt"`; so do `quadratic_solvers` and `licensed_solvers`.
* One `logger.warning` on the first probe naming the CC ≥ 7.0 and driver ≥ 525.60.13 requirements.
* `model.solve("cuopt")` raises `AssertionError: Solver cuopt not installed` from `Model.solve`'s
  existing membership assertion (`model.py:2078`) — the generic path every uninstalled solver takes.
  `Solver.from_name("cuopt", m)` raises the base-class `ImportError("Solver package for 'cuopt' is
  not installed…")`.
* **Never** a bare CUDA traceback, and never cuOpt's own
  `InputValidationError: A CPU-memory problem requires remote execution. Set CUOPT_REMOTE_HOST…`,
  which is exactly the message an import-only probe delivers (exploration item 13) and exactly what
  §4.6 c3 rules out.

**Accepted downside, recorded for the handoff (this is plan-3's and plan-5's argument, kept):**
`_AvailableSolvers`' docstring now has a solver-specific exception, and the *message* a GPU-less
user sees says "not installed" when the package is installed. Fixing the message properly means
touching the shared `Model.solve` assertion — out of scope (I4) and a behaviour change for every
solver. The `logger.warning` is the mitigation; the docs state it. Note
`test_available_solvers.py:39` stays green either way — it compares `is_available()` to membership,
so both sides move together, which is also why that test **cannot** catch a uniformly-false probe
(hence V0-5).

Additionally grafted regardless of probe mechanics: **honest error-message translation at solve
time.** If a machine has a device node but too old a driver, the probe may pass and `Solve()` may
still fail with cuOpt's remote-execution message. `_solve` therefore lets cuOpt's exception
propagate unchanged (it is actionable enough once the docs name the driver floor) but the docs
explicitly list that message and what it means. No message rewriting in `_solve` — that would mask
genuinely different failures.

### 5d. QP — `0.5 · M.Q`, full-symmetric CSR; MIQP rejected up front; non-PSD surfaces as an error

* **Convention:** `Q_cuopt = sign * 0.5 * model.matrices.Q`, kept **full symmetric**, converted to
  CSR. cuOpt minimises `cᵀx + xᵀQx` with `Q` internally symmetrised to `Q + Qᵀ`
  (`data_model/data_model.py:287-288`), and linopy's `M.Q` **is** the Hessian `H` of `½xᵀHx`
  (`expressions.py:2945-2970` docstring: "the quadratic term is 0.5 x^T Q x… the diagonal terms are
  doubled") — which is also why `Highs` passes `triu(M.Q)` to `passHessian` (`:1716-1720`).
  Confirmed three ways: closed-form single-variable sweep; a six-encoding cross-term test where only
  `0.5·H` and `strictly-upper(H) + ½diag(H)` match; and live against HiGHS through linopy's own
  matrices (`0.5*M.Q` → `-4.28571429`, matching HiGHS exactly, while the naive `M.Q` returns
  `-2.14285714` with status `Optimal` and **half** the solution vector). A fourth confirmation from
  linopy's own fixtures: for `m.add_objective(x*x)`, `M.Q = [[2,0],[0,0]]`.

  Three lines in `_build_solver_model`:

  ```python
        Q = M.Q
        if Q is not None:
            Qc = (sign * 0.5 * Q).tocsr()
            dm.set_quadratic_objective_matrix(Qc.data, Qc.indices, Qc.indptr)
  ```

  Full-symmetric is chosen because it is the form `M.Q` already has — no `triu`, no diagonal special
  case, which is the classic source of the factor-of-2 error.
* **`max` QP** takes the same `sign` as `c` (`Q → -Q`), verified 4/4 QP cells. QP never reaches the
  buggy presolve path (cuOpt forces Barrier), so M2 is not *required* here — applying it uniformly
  keeps one code path and was verified to do no harm.
* **MIQP: rejected before the build** (§2.1.5). The message names solvers that do support it.
* **Non-PSD `Q`: no special handling.** cuOpt detects it and returns `NumericalError` → mapped to
  `internal_solver_error` with `get_error_message()` logged. It does not crash and does not return a
  bogus answer, so a linopy-side eigenvalue check on every solve would be pure cost. Documented as a
  limitation. The asymmetric-but-`Q+Qᵀ`-PSD case is accepted by cuOpt and is unreachable from linopy
  anyway (`M.Q` is symmetric).
* **Quadratic *constraints*** (`add_quadratic_constraint`) are out of scope — linopy has no model
  surface for them, and cuOpt rejects equality quadratic constraints outright.
* **Sequencing** (charter §4.5): QP is WP-F, dispatched only after the LP/MILP core has merged and
  Gate 4's checks pass on it.
* **`method`/`crossover` are silently overridden by cuOpt for QP** (all 8 combinations →
  `solved_by=Barrier`). Documented, not fought.

### 5e. Reduced costs — do not surface them

**Position: `Solution.get_reduced_cost()` is never called.**

* linopy has nowhere to put them: `grep -rn "reduced_cost\|ReducedCost" linopy/` returns nothing and
  `linopy.constants.Solution` (`:273-287`) carries only `primal`, `dual`, `objective`.
* They are also *wrong* on the default path for maximised models with `<=` rows: cuOpt returns
  `−c − Aᵀy` instead of `c − Aᵀy` (`+0.39999985` where HiGHS says `-1.6`), and PDLP returns a flat
  `0` for `max`/`>=`. The exploration swept four values × four methods × three row senses and could
  **not** reduce this to one transformation; it explicitly declines to propose one.

Actions instead of an implementation: (i) the one-line comment in §2.1.7 next to the dual handling;
(ii) an entry in `logs/45-known-issues.md` recording the upstream defect so nobody later "fixes" it
by guessing a sign; (iii) one sentence in `doc/gpu-acceleration.rst` under limitations. Adding a
reduced-cost surface to linopy would be scope creep (E4 territory) and is an explicit non-goal.

### 5f. Solver options — blanket `bool → int`, wrap the unknown *name*, `log_fn` wins

Fully specified in §2.1.8. Summary of the positions taken: blanket coercion (not an allowlist, not
type-sniffing, not a retry); unknown-name `ValueError` wrapped with the option name; value errors
passed through unwrapped; `log_fn` beats a user `log_file`; `log_to_console` left at cuOpt's default;
**no default `time_limit`** (no other linopy solver imposes one, and a silent default would change
the meaning of "optimal"); `save_best_primal_so_far` never set by linopy but still reachable, with
the segfault hazard documented.

### 5g. Interruptibility — adopt the HiGHS worker-thread pattern, minus the cancel step

Fully specified in §2.1.10. Declining it would leave a solver aimed at multi-million-variable models
unresponsive to Ctrl-C for the full solve duration, in a file that already contains the fix pattern.
Cost ≈ 25 lines, and the CPU-testable seam is worth more than that on its own.

### 5h. Warm start — defer, and never touch the poisoning setter

**Position: `warmstart_fn` raises `NotImplementedError` (cuPDLPx precedent, `:4352-4355`); neither
`set_pdlp_warm_start_data` nor `set_initial_primal_solution` is ever called.**

* Usable PDLP warm start requires **three** non-default settings simultaneously — `method=1`,
  `pdlp_solver_mode=1 (Stable2)`, `presolve=0` — because the payload comes back in *presolved*
  coordinates (`primal=(1634,)` for a `n=2000` model at the default `presolve=-1`) and cannot be fed
  back into its own model otherwise (`Invalid PDLPWarmStart data…`), and because the default
  `Stable3` and the docstring-endorsed `Fast1` both return `NoTermination` with `obj=nan`. Wiring
  that up would mean silently forcing PDLP, disabling presolve, and changing the method the user
  asked for — three surprises to buy an 80× iteration win on a re-solve linopy cannot currently
  express (`warmstart_fn` is a *file* path; the payload is 17 numpy arrays, not a file format).
* `DataModel.set_initial_primal_solution` on an LP raises `cudaErrorIllegalAddress` and **poisons
  the CUDA context for the rest of the process** — reproduced with all-default settings and
  correctly sized C-contiguous float64 input. One warm-start attempt would take down an entire
  pytest session or user script. This is a hard "never call", enforced by a code comment at the only
  plausible call site and by an entry in `logs/45-known-issues.md`.
* `basis_fn` is warn-and-ignore (cuPDLPx precedent): cuOpt has no simplex basis to write, and the
  PDLP payload is algorithm state, not a basis.
* Test: the `test_basis_and_warmstart` skip (§2.7 item 3). Documented under limitations, with the
  three settings named so a motivated user can do it by hand.

### 5i. The pyproject extra — `cuopt`, one Linux-marked requirement

```toml
# NVIDIA cuOpt (GPU). Deliberately NOT part of `solvers`: it pulls ~34 packages
# and several GB of CUDA wheels. Linux-only wheels, no sdist, so the marker keeps
# macOS/Windows resolution unaffected. Requires a CUDA 12 driver (>= 525.60.13)
# and a GPU of compute capability >= 7.0.
cuopt = [
    "cuopt-cu12>=26.8; platform_system == 'Linux'",
]
```

* **Name `cuopt`**, not `gpu`: it matches the solver name, the `SolverName` value, and the file's
  existing lower-case single-word extras (`oetc`, `remote`, `docs`, `dev`, `solvers`, `benchmarks`).
  `gpu` would promise a category the extra cannot keep (one solver, Linux only, CUDA 12 only).
  Charter §5 writes `<gpu-extra>` as a placeholder, not a name. Logged as a D-entry.
* **Placement**: a sibling immediately after the `solvers = [...]` block (`pyproject.toml:113`),
  reachable from **no** other extra — there is no `all` extra in this project, and `dev`/`docs` do
  not reference it. Verified in a clean venv: `pip install 'linopy[solvers]'` resolves 34 packages
  with **zero** cuopt/cudf/nvidia/cuda entries.
* **Marker**: `platform_system == 'Linux'` — the positive twin of the file's established
  `"cplex; platform_system != 'Darwin'"` (`:105`). It must be an equality, because cuOpt has no
  Windows wheels either. No `python_version` marker: cuOpt ships cp311–cp314 and linopy already
  requires `>=3.11`. Verified: `uv pip compile --python-platform windows|macos --extra cuopt` both
  exit 0 with cuOpt absent (25 / 23 packages); `linux` resolves 66 including `cuopt-cu12==26.8.0`.
* **Lower bound, no ceiling**: `>=26.8` is the first release with cp311 wheels (§4.2). No upper
  bound, matching the file's style (`coptpy!=7.2.1` is the only exclusion anywhere). The `cu13`
  variant is **not** offered — the driver requirement differs (§4.1) — but the `cu12`/`cu13`
  distinction and the driver floor are stated next to every install command (§9).
* **Not touched**: `[tool.uv]` (no index entry needed — the whole tree is on PyPI — and workspace
  index config does not propagate to downstream installs anyway), and `benchmarks` (pandas
  `==3.0.5` collides with cuDF's `<3.0.4a0` ceiling; do not run benchmarks in a cuOpt venv).
* **I9's end date.** Once this lands, `uv sync --extra cuopt` becomes the *correct* command and is
  itself a test that the extra works. The WP-C report must say so explicitly, because every other
  agent is under standing instruction never to sync.

### 5j. Models with no constraint nonzeros  (B1, B2, R4, R5)

**Position: pad with one redundant, singly-bounded row derived from a variable's own bounds.**

cuOpt returns `NoTermination` with `obj=nan` and an **empty** primal whenever the constraint matrix
has **no nonzeros** — the trigger is `nnz == 0`, not `nrows == 0`, so a model with one genuine
all-zero row fails too. This is reachable from an ordinary linopy model: `M.A is None` for the
`quadratic_model` fixture once `con0` is removed, which is exactly what
`test_quadratic_model_wo_constraint` (`test_optimization.py:918-943`) does before asserting
`condition == "optimal"` for every solver in `feasible_quadratic_solvers`.

Measured, on that exact fixture (20 variables, `Q_cuopt = 0.5·M.Q`):

```
pad bounds (-inf, +inf)      -> ('NumericalError', nan)
pad bounds (-1e30, 1e30)     -> ('NumericalError', nan)
pad bounds (lb0, +inf)       -> ('Optimal', 7.589716418676626e-15, x[:3]=[2.64e-08 2.77e-08 2.77e-08])
pad bounds (lb0, ub0)        -> ('Optimal', 7.589716418676626e-15, ...)
```

and on a strictly positive-definite box-only QP (10 vars, `H = 2I`), isolating the row bounds rather
than the singular Hessian as the cause:

```
free -> NumericalError nan     ge0 -> Optimal 1.47e-15     box -> Optimal 1.69e-14
```

So:

* **A doubly-infinite pad row is rejected (R5).** It works for LP (`box-only LP, free pad row ->
  Optimal 0.0`) and fails for QP.
* **`raise ValueError` is rejected (R4).** `feasible_quadratic_solvers` contains cuopt, so the
  `pytest.raises(ValueError)` branch of that test is *not* taken and the `ValueError` fails it. Do
  not copy `cuPDLPx._build_solver_model`'s `raise ValueError("Model has no constraints…")`
  (`:4255-4256`): cuPDLPx is not installed on this machine, so its behaviour here is untested
  precedent.
* **The chosen shape**: pick the first index `j` with `np.isfinite(M.lb[j]) or np.isfinite(M.ub[j])`
  and append the row `1·x_j ∈ [M.lb[j], M.ub[j]]`. This is redundant by construction (it restates a
  bound cuOpt already enforces), and it is measured `Optimal` with `obj=7.59e-15` and `x≈2.6e-08` →
  `round(3) == 0`, which passes the existing assertions. `quadratic_model` has `lower = 0` for all
  20 variables, so `j = 0` and the pad row is `x_0 ≥ 0`.
* **Residual case — every variable free on both sides.** Reachable
  (`m.add_variables(name="x"); m.add_objective(x*x)`). No pad shape is verified for it, and the two
  candidates that would change the feasible set (`sum(x) >= 0`, a large finite box) are rejected
  outright: they can silently change the answer, which is precisely what §3.6 exists to prevent.
  `_build_solver_model` therefore raises `NotImplementedError` with an actionable message naming a
  CPU solver. Covered by V1-6c; the message text is asserted, not the solver behaviour.
* **The pad row is always the last row**, so `dual[:n_rows]` drops its dual (§2.1.7). V1-6 checks
  duals against live HiGHS both *with* and *without* constraints so a one-row misalignment fails
  loudly rather than silently.

**`test_quadratic_model_unbounded` — disposition frozen now (B10).** cuOpt returns `NumericalError`
(`obj=nan`) on that fixture (measured), which this plan maps to `internal_solver_error`; the test
accepts only `["unbounded", "unknown", "infeasible_or_unbounded"]` (`:963`). The **frozen**
disposition is: keep the honest mapping and add a cuopt-specific branch in that one test citing the
measured status — the same shape as the existing cuPDLPx/mindopt/gurobi branches in that file.
Re-mapping `NumericalError → unknown` is **rejected**: `safe_get_solution`'s `unknown` branch parses
the solution anyway and flips the status to `ok`, which would report a `nan`-objective solve as
successful. Widening the test's accept-list is **not** authorised. See verification row V2-2.

### 5k. A bare `model.solve("cuopt")`  (B3, B4, G1, R6)

Fully specified in §2.1.4. Position: `_build_file` is overridden to build directly, `_run_file` is
left at the base, the temp problem file is handed back through `_problem_fn` so linopy's own
`finally` removes it, and a user who explicitly passes `io_api="lp"` gets the direct path plus one
warning. That ergonomic wart is stated in the docs and is better than an exception. Rejected:
`_run_file` raising `NotImplementedError` (fails `test_solvers.py:48` and `:209`), and the base
plan's `_run_file` fallback (works, but writes and discards a file — 0.23 s / 7.7 MB at 10⁵
variables).

---

## 6. Requirement-ID map

Charter §5 rows numbered top to bottom as `C-R1`…`C-R17`. `C-R18`…`C-R27` are plan-1's new IDs,
adopted verbatim per the review's instruction; `C-R28` and `C-R29` are the two the review adds. **No
`C-R18+` had been recorded anywhere before this document, so this is the first and now binding
numbering.** IDs are never renumbered or reused (§5.3).

| ID | §5 row / topic | Concrete work that satisfies it | WP |
|---|---|---|---|
| C-R1 | `solvers.py:148` `SolverFeature` | `features` frozenset on `class cuOpt` (§3); **no enum edit** — every needed member exists | A (+F for `QUADRATIC_OBJECTIVE`) |
| C-R2 | `:176` `TYPE_CHECKING` | `import cuopt` added to the block | A |
| C-R3 | `:304` `_LazyModule` | `cuopt = _LazyModule("cuopt")` + the never-import-`cuopt.routing` comment; no `type: ignore` | A |
| C-R4 | `:346` `SolverName` | `cuOpt = "cuopt"` after `cuPDLPx` | A |
| C-R5 | `:4156` new solver class | `class cuOpt(Solver[None])`: `is_available`, `_license_probe`, `_build_file`, `_build_direct`, `_build_solver_model`, `_run_direct`, `_solve`, `_set_solver_params` (§2.1) | A (+F: Q) |
| C-R6 | `:4440` `_SOLVER_PROBE_ORDER` | `"cuopt"` after `"cupdlpx"` | A |
| C-R7 | `solver_capabilities.py` registry | **zero-diff proof**: `test_capability_shim_round_trips` (`test_solvers.py:481`, already parametrized over all `SolverName` × `SolverFeature`) + an explicit `SOLVER_REGISTRY["cuopt"].features` assertion in `test_cuopt.py`; pasted | D |
| C-R8 | `io.py:33`, `:792` | TYPE_CHECKING import + `to_cuopt(m)` with the M2 docstring | B |
| C-R9 | `model.py:73`, `:2517` | import + `to_cuopt = to_cuopt` binding | B |
| C-R10 | `constants.py:171` mapping | two name-keyed `CONDITION_MAP`s in `_solve`, selected by problem category (§4); **no `constants.py` edit** | A |
| C-R11 | `test/conftest.py:57-79` | **zero-diff proof**: `GPU_ONLY` drives the generic auto-skip; pasted `--collect-only` with and without `--run-gpu` | D |
| C-R12 | `test/test_optimization.py` branches | `time_limit_option["cuopt"]`; warmstart skip; documented *non*-skip for `test_infeasible_model`; the frozen `test_quadratic_model_unbounded` branch | D (+F) |
| C-R13 | `test/test_solvers.py` | 5 feature-matrix rows (+1 in F) | D (+F) |
| C-R14 | `pyproject.toml` extra | `cuopt` extra (§5i) + clean-venv acceptance test | C |
| C-R15 | `doc/gpu-acceleration.rst` | new "cuOpt" section: install, hardware/driver, usage, supported classes, limitations, method guidance (§9) | E (+F for the QP paragraph) |
| C-R16 | `prerequisites`/`contributing`/`index`/`api`/`release_notes`/`README` | §9 table, incl. `solvers.cuOpt` **and** `model.Model.to_cuopt` in `api.rst` | E |
| C-R17 | `:4426` `QUADRATIC_SOLVERS` (read-only anchor) | QP build path + MIQP guard + QP tests; flag declared, list untouched | F |
| **C-R18** | Honest degradation without a GPU (§4.6 c3, D6) | `_cuda_device_available()` + `is_available()` gate + warning; documented; subprocess test with `CUDA_VISIBLE_DEVICES=''` | A, D |
| **C-R19** | M2 always-minimise (dual-sign correctness) | `sign` in `_build_solver_model` / `_solve` for `c`, `Q`, objective, duals, `solution_bound` + the do-not-simplify comment | A (+F: Q) |
| **C-R20** | Repeat-solve crash mitigation | `_DEFAULT_METHOD = 3`, `method=0` warning, subprocess repeat-solve test, docs | A, D, E |
| **C-R21** | Models with no constraint nonzeros | singly-bounded pad row + dual slice + the all-free `NotImplementedError`; tests (LP and QP) | A, D (+F) |
| **C-R22** | Ctrl-C responsiveness | `_run_cuopt_with_keyboard_interrupt` + CPU unit test | A, D |
| **C-R23** | MIQP rejected honestly | `NotImplementedError` in `_build_direct`; test | F |
| **C-R24** | Option handling | blanket bool→int coercion, wrapped unknown-name `ValueError`, `log_fn` precedence; tests | A, D |
| **C-R25** | Warm start / basis / solution file refusals | `NotImplementedError` / warning + skip + docs | A, D, E |
| **C-R26** | Semi-continuous support | `'S'` passthrough, feature flag, differential test in `test_cuopt.py`, docs | A, D, E |
| **C-R27** | MIP report (gap + dual bound) | `get_milp_stats()` → `SolverReport`, sign-corrected; covered by `test_milp_model` and V1-4 | A, D |
| **C-R28** | The availability probe must not initialise CUDA in the caller's process | subprocess probe (§2.1.9) + three verification rows: positive branch, negative branch, **fork safety** | A, D |
| **C-R29** | A bare `model.solve("cuopt")` solves, warns once, and leaves no `linopy-problem-*` file | `_build_file` override + `_problem_fn` handback (§2.1.4); tests V1-16 and V1-17 | A, D |

**No §5 row is unmapped, and no ID is reinterpreted or silently narrowed.** Every ID also appears in
`logs/15-verification.md`.

---

## 7. Work-package breakdown

Rules honoured (charter §6.4): the `solvers.py` core goes **first and alone**; then disjoint
packages in parallel; QP **last and alone**; packaging is first-class and early; no two concurrent
packages share a file.

> **Amendment A1 (2026-08-22, human-lead directive, logged as D10 in `logs/30-decisions.md`; rows
> marked (A1) below added by the lead — plan mechanics only, no design change).** WP-G added:
> every unsupported/deferred-feature decision gets a concise in-code rationale at its enforcement
> site. New requirement ID **C-R30**. Commit 6 renumbered; the reserved review-fix commit becomes 7.
> **A1.1 (2026-08-22, human-lead directive):** WP-G item 1 extended — the direct-only-io comment
> also directs developers to cuOpt's own API for solving `.mps`/`.lp` files directly.

| WP | Files (exclusive) | IDs | Runs | Commit |
|---|---|---|---|---|
| **A — solvers.py core (LP + MILP)** | `linopy/solvers.py` | C-R1–C-R6, C-R10, C-R18–C-R22, C-R24–C-R29 | first, alone | 1 |
| **B — io + model binding** | `linopy/io.py`, `linopy/model.py` | C-R8, C-R9 | after A merges; parallel with C, D, E | 1 |
| **C — packaging** | `pyproject.toml` | C-R14 | parallel with A (disjoint file); re-verified after A merges | 3 |
| **D — tests** | `test/test_optimization.py`, `test/test_solvers.py`, new `test/test_cuopt.py`, new `test/test_cuopt_interrupt.py` | C-R7, C-R11–C-R13, and the test halves of C-R18–C-R29 | after A **and** B merge; parallel with E | 2 |
| **E — docs** | `doc/*.rst`, `README.md`, `examples/using-solvers.ipynb` | C-R15, C-R16 | after A merges; parallel with C, D | 3 + 4 |
| **F — QP** | `linopy/solvers.py`, `test/test_cuopt.py`, `test/test_solvers.py`, `doc/gpu-acceleration.rst` | C-R17, C-R23, plus the QP rows of C-R19 and C-R21 | last, alone (Gate 4 green on LP/MILP first) | 5 |
| **G — limitation-rationale comments (A1)** | `linopy/solvers.py`, `linopy/io.py` — comments/docstrings **only**, zero behaviour change | C-R30 | after F merges, alone | 6 |

File-collision check: A and F both own `solvers.py` → strictly sequential (F is last anyway).
D and F both own `test_cuopt.py` and `test_solvers.py`; E and F both own `gpu-acceleration.rst` → F
runs only after both merge. B, C, D and E are pairwise disjoint. (A1) G shares `solvers.py` with A
and F → strictly after F; it is an audit pass over merged code, so it collides with nothing live.

### 7.1 Worktree and lineage discipline — required in every slice brief

Every slice brief **must** carry the expected base commit SHA, and the coding agent's **first
action** is:

```bash
git rev-parse --short HEAD && git log --oneline -1
git merge-base --is-ancestor <BASE_SHA> HEAD && echo "LINEAGE OK" || echo "WRONG LINEAGE"
```

On `WRONG LINEAGE` — or a **missing** base SHA in the brief — the agent **stops, writes nothing, and
reports.** It must not adapt to the tree it finds or "fix" imports to match. Reporting a wrong base
is a *successful* outcome for this step, and item 0 of the slice report is where it goes.

* `feat/cuopt-solver` is at **`09c34dd`** at the time of writing, so WP-A's and WP-C's briefs carry
  `09c34dd`. Every later brief carries the then-current tip, obtained by
  `git rev-parse --short feat/cuopt-solver` at dispatch time.
* Worktrees are created **explicitly** from `feat/cuopt-solver`
  (`git worktree add /home/azureuser/linopy-gpu/dev-scripts/cuopt/worktrees/<wp-id> -b wp/<wp-id>
  feat/cuopt-solver`), never via `Agent(isolation: "worktree")`.
* Worktrees contain only *tracked* files, so `dev-scripts/` does not exist inside them. Every brief
  hands absolute paths for the charter, the templates, the plans and the logs.
* Every brief ends with the §3.6 threshold paragraph **verbatim**, and with *"Do not summarise away
  failures — a red test reported honestly is acceptable work, a hidden one is not."*

### 7.2 Acceptance criteria

Every criterion below is a **paste**, not an assertion. Front-loaded per G4: WP-A's empirical checks
run **before** any test file is written, so the pad shape, the sign matrix and the probe branches are
established rather than discovered.

**WP-A — `solvers.py` core.**

1. `/home/azureuser/.local/bin/ruff check .` and `.venv/bin/mypy` clean.
2. `python -c "import linopy; print(linopy.available_solvers)"` **lists** `cuopt`.
3. `linopy.solvers.cuOpt.supported_features()` printed and equal to §3's LP/MILP set.
4. **Sign matrix by hand, before writing tests**: all six `min`/`max` × `<=`/`>=`/`==` cells plus
   the presolve-solvable `max` case (square equality system, `n = 2` and `n = 10`), each compared
   live in-process against `model.solve("highs")`, duals included. Thresholds from
   `logs/15-verification.md`.
5. **Make a guard fail on purpose, once (charter §9):** with `sign` forced to `+1.0` for a `max`
   model, at least one of the presolve-`max` duals must disagree with HiGHS; paste both runs. A
   guard that cannot fail is not a guard.
6. **Zero-nnz pad shape**: `quadratic_model` minus `con0` built through
   `_build_solver_model`, and the all-free residual case raising `NotImplementedError`. Paste both.
7. A MILP solved; `result.report.dual_bound` and `.mip_gap` printed **non-`None`**.
8. `model.solve("cuopt")` **with no `io_api`** succeeds, emits the fallback warning **once**, and
   leaves `list(Path(m.solver_dir).glob("linopy-problem-*")) == []`.
9. Three sequential solves of a 2000-variable / 1000-row LP in **one** process: exit code 0.
10. Device probe, **both directions and fork safety**: `_cuda_device_available()` is `True` here; a
    subprocess with `CUDA_VISIBLE_DEVICES=''` shows `available_solvers` without `cuopt` plus the
    warning line; and probe → `os.fork()` → solve in the child exits 0.
11. Options: `model.solve("cuopt", TimeLimit=1)` raises the wrapped `ValueError` **naming
    `TimeLimit`**; `presolve=False` is accepted and solves; `log_fn` produces a non-empty file
    containing `cuOpt version`.
12. `test_infeasible_model`'s model solved through cuOpt: paste `status` and `condition`.

**WP-B — io + model.** `m.to_cuopt()` returns a `DataModel` whose
`get_objective_coefficients()` and `get_constraint_matrix_values()` match `m.matrices` (sign-flipped
for `max`); `ruff`/`mypy` clean.

**WP-C — packaging.** Q13 re-run pasted: clean venv, **no** extra index and **no** resolver flags,
`pip install '<repo>[cuopt]'` then `python -c "import linopy; print(linopy.available_solvers)"`
printing a list **containing** `cuopt`; `pip install 'linopy[solvers]' --dry-run --report` showing
no cuda/cuopt/cudf/nvidia package; `uv pip compile --python-platform windows|macos --extra cuopt`
both exit 0 without `cuopt-cu12`. Note this re-run needs a GPU-visible machine because of §5c.
The report states I9's end date explicitly.

**Ordering caveat that MUST be carried into WP-C's brief verbatim.** WP-C runs in parallel with
WP-A (§6.4 rule 5 explicitly wants packaging early), but the `available_solvers` **contains**
`cuopt` half of this criterion **cannot pass until WP-A has merged** — until then the branch has no
`cuOpt` class and no `"cuopt"` entry in `_SOLVER_PROBE_ORDER`, so the clean-venv probe legitimately
prints `[]`, exactly as Q13 did. That is a **pass** for the packaging half (the install itself
succeeded and `import cuopt` reports `26.08.00`) and must be re-verified after A merges. Without
this sentence an agent will read `[]` as a packaging failure and start debugging the extra.

**WP-D — tests.** `.venv/bin/pytest -q` (no GPU flag) shows **no new failures** versus
`logs/05-baseline-pytest.txt` **and** cuOpt tests **skipped**; `.venv/bin/pytest --run-gpu -q` green
versus `logs/05-baseline-pytest-gpu.txt`, with `-k cuopt --run-gpu -v` pasted so it is *provable*
the GPU tests ran rather than skipped; `pytest --collect-only -q -k cuopt` pasted with and without
`--run-gpu`; the `SOLVER_REGISTRY["cuopt"]` assertion passing.

**WP-E — docs.** Built in `/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs` with
`PATH=/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin:$PATH` (D3) — **no new warning**
versus `logs/05-baseline-docs.txt`; every §9 row present; the mechanical PR-#516 gap check run and
every difference justified. The build must succeed in a venv with **no cuOpt installed**.

**WP-F — QP.** QP differential versus live HiGHS on `quadratic_model`,
`quadratic_model_cross_terms` and the 3-variable model from exploration item 12, **scoped exactly as
V1-13a/b/c in `logs/15-verification.md`** — in particular `quadratic_model`'s 10 `y` variables are
underdetermined by the problem and are **never** compared against the oracle (measured cuOpt
`y=155.979` vs HiGHS `y=9.9999995`, both optimal); the
`0.5·M.Q → M.Q` guard demonstrated to **change the answer** once (paste both); `pytest -k
"quadratic" --run-gpu` green including the frozen `test_quadratic_model_unbounded` branch;
`test_quadratic_model_wo_constraint` green; MIQP raising `NotImplementedError`;
`linopy.quadratic_solvers` containing `cuopt`; full `pytest --run-gpu` green.

**WP-G — limitation-rationale comments (A1, C-R30).** Audit-then-add: each decision below carries a
**1–3-line** rationale at its enforcement site — a comment, or a docstring "Limitations"/Notes line
where the surface is user-facing. Where WP-A/WP-F already wrote it, verify and leave it; never
duplicate. Comment density must match the surrounding file (charter §6.6a) — state the constraint
the code cannot show, nothing more. The list:

1. Direct-only io (`_build_file` override / class docstring): file `io_api`s excluded because
   solution-file output and linopy's explicit coordinate names are unverified against cuOpt's
   parser, and the file path inverts the Q convention (triangular `.lp` vs symmetric `.mps`).
   (A1.1, human-directed) The comment must also point developers at the workaround: to solve an
   existing `.mps`/`.lp` file directly with cuOpt, use cuOpt's own file reader in
   `cuopt.linear_programming` — one line, e.g. *"to solve .mps/.lp files directly, use cuOpt's
   own API (`cuopt.linear_programming`); linopy support is a planned follow-up."*
2. Warm start (`warmstart_fn` refusal site): usable PDLP warm start needs `method=1` +
   `pdlp_solver_mode=1` + `presolve=0` simultaneously (payload is in presolved coordinates);
   and the never-call fence on `set_initial_primal_solution` (crashes and poisons the CUDA
   context for the process — upstream bug).
3. Reduced costs: not populated — cuOpt's reduced-cost sign is wrong upstream for maximised
   models with `<=` rows.
4. SOS / indicator constraints: genuinely absent upstream (at the refusal sites).
5. MIQP/IQP guard: cuOpt silently mis-solves MIQP rather than erroring.
6. `method=3` (Barrier) default: default Concurrent segfaults on repeated solves ≳1300 vars.
7. MIP duals left empty: cuOpt refuses duals for MIPs.
8. Subprocess GPU probe: an in-process CUDA probe poisons later `os.fork()` children.
9. Blanket bool→int option coercion: cuOpt rejects Python bools for its int-typed parameters.

Acceptance (every one a paste): full `git diff` of the commit shown to contain **only**
comment/docstring lines; a file:line map of all nine items (pre-existing or added);
`/home/azureuser/.local/bin/ruff check .` clean; mypy showing zero **new** errors; and
`.venv/bin/pytest --run-gpu -q` identical to the pre-WP-G run (no behaviour change).

### 7.3 Commits (§6.4a shape — ~6, plain imperative, sentence case, no conventional prefix)

| # | Message | Contents |
|---|---|---|
| 1 | `Implement interface to cuOpt solver and required IO` | WP-A + WP-B (`solvers.py`, `io.py`, `model.py`) |
| 2 | `Update tests` | WP-D |
| 3 | `Add to docs & add package dependency` | WP-E docs + WP-C `pyproject.toml`. **LP and MILP only** (G10) — no QP claim yet, so every intermediate commit is truthful |
| 4 | `Add comment for release notes` | `doc/release_notes.rst` |
| 5 | `Add support for quadratic objectives` | WP-F, including the QP paragraph in `doc/gpu-acceleration.rst` and the release-notes amendment |
| 6 | `Add comments explaining unsupported features` | WP-G (A1) — comments/docstrings only |
| 7 | *(reserved)* | The targeted Phase 5/6 follow-up fix. #516's own arc ends with two such commits. If review finds nothing, **six substantive commits is the honest count — do not manufacture a seventh.** |

No merge commits. Each WP branch is rebased onto the tip of `feat/cuopt-solver` and fast-forwarded
in. Every commit is authored by the model that wrote it, with a matching `Co-Authored-By` trailer
(I11).

---

## 8. Test plan

Tiers per charter §6.3a. **Every V1 row calls HiGHS live, in-process, on the identical model — no
baked-in expected values anywhere.** The numeric thresholds live in
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md` and are frozen at Gate 3.

### 8.1 Locations and gating (G9)

* **`test/test_cuopt.py` (new)** — all cuOpt-specific V1 work. Module gate:

  ```python
  pytestmark = [
      pytest.mark.gpu,
      pytest.mark.skipif(
          not os.environ.get("LINOPY_RUN_GPU_TESTS"),
          reason="need --run-gpu option to run GPU tests",
      ),
      pytest.mark.skipif("cuopt" not in licensed_solvers, reason="cuOpt is not installed"),
      pytest.mark.skipif("highs" not in licensed_solvers, reason="HiGHS oracle not installed"),
  ]
  ```

  This is the intended use of the `LINOPY_RUN_GPU_TESTS` env var that `conftest.py:46-50` already
  sets: `pytest_collection_modifyitems` only sees tests whose callspec carries a `solver` param
  (`:68-79`), which these differential tests do not. Verified by reading both.
* **`test/test_cuopt_interrupt.py` (new, CPU, ungated)** — mirrors
  `test/test_highs_keyboard_interrupt.py`.
* **`test/test_optimization.py`, `test/test_solvers.py`** — the existing-file touch points (§2.7,
  §2.8) only.
* **Semi-continuous cuOpt coverage lives in `test_cuopt.py`**, not `test_semi_continuous.py`, which
  has no `solver` parametrization and no GPU gating (verified). Recorded so Phase 6 reads it as a
  decision.
* linopy's own assertions (`assert_linequal`, `assert_varequal`, …) are used wherever a comparison
  is between linopy objects (`AGENTS.md`); numeric solver comparisons use `np.isclose`/
  `pytest.approx` with the frozen thresholds.

### 8.2 V0 — structural (every WP)

`ruff check .`; `mypy`; `"cuopt" in available_solvers`; `SOLVER_REGISTRY["cuopt"].features` equals
the declared set; `SolverName.cuOpt.name == cuOpt.__name__`; `conftest.py`,
`solver_capabilities.py` and `constants.py` unchanged; `git status --porcelain` clean.

### 8.3 V1 — differential versus live HiGHS (`test_cuopt.py`)

| # | Test | What it proves |
|---|---|---|
| V1-1 | **Six-cell sign matrix**: `sense ∈ {min,max}` × `row ∈ {<=,>=,==}` on the exploration model (`A=[[1,2,1],[3,1,1]]`, `b=[4,6]`, `0<=x<=10`, `c` per cell so `x*=(1.6,1.2,0)` with both rows binding and a unique non-degenerate dual). Two deep copies of one model, one solved with cuopt and one with highs; compare `solution`, `dual` and `objective.value`. | dual and primal sign convention, all six cells |
| V1-2 | **Presolve-solvable `max` model** — 2-variable square equality system (`x+2y=4`, `3x+y=6`, `max x+y`) and an `n=10` variant. This is the case the six-cell matrix **misses**: without M2 the duals come back negated with status `Optimal` and a correct objective (`max|Δ| = 3.61`). | that M2 works on the branch that motivated it |
| V1-3 | **Presolve branch still reached**: build a raw `DataModel` for V1-2's model with `set_maximize(True)` and assert `sol.get_solved_by()` is `SolverMethod.Unset`. Asserts a *coverage precondition*, not a correctness claim. Its docstring says: **if this goes red, cuOpt changed its routing and V1-2's coverage must be re-established — it never means M2 should be removed.** | V1-2 keeps exercising the risky path |
| V1-4 | **MILP differential**: a `milp_model`-shaped model, objective versus HiGHS; integer variables integral within `mip_integrality_tolerance`; `model.dual` **empty** (cuOpt refuses duals for a MIP); `report.dual_bound`/`.mip_gap` non-`None` and the bound sign-correct for `max`. | integrality, the MIP dual guard (B8), C-R27 |
| V1-5 | **Status mapping**: infeasible LP → `infeasible`; unbounded LP → `infeasible_or_unbounded`; `time_limit=1e-6` on a 400×300 LP → `time_limit`; `iteration_limit=1` → `iteration_limit`; the malformed-input path (a zero-nnz model built through a raw `DataModel`, *without* the pad) → `internal_solver_error`. | the §4 rows that are reachable |
| V1-5b | **Empty primal under an `ok` status**: the `time_limit=1e-6` case must either return a full-length primal or log the length-mismatch error and yield an empty `Solution` — never raise and never scatter. | §2.1.7's length pre-check |
| V1-6 | **No-constraint models**: (a) an LP with no constraints; (b) `quadratic_model` with `con0` removed — objective and the **`x` block only** against the oracle, because `y` is underdetermined there too (see V1-13b); (c) the all-free residual case raising `NotImplementedError`. Also assert the pad row's dual is dropped — `len(model.dual)` per constraint set unchanged, and duals on a model *with* constraints still match HiGHS. | C-R21, and that the pad does not misalign duals |
| V1-7 | **Semi-continuous**: `x` semi-continuous in `[1,10]` with `x <= 0.5`, `max x` → 0; and `x <= 5` → 5 (the shapes the HiGHS tests use), differential against HiGHS. | C-R26 (`'S'` passthrough) — the only coverage the flag gets |
| V1-8 | **Repeat solves in a subprocess** (`sys.executable -c ...`): three sequential `model.solve("cuopt")` of a 2000-variable / 1000-row LP under linopy's defaults; assert `returncode == 0` and three `optimal` lines. A subprocess so a SIGSEGV regression is a *failed test*, not a dead pytest session. | C-R20 (n=2000 is above the measured 1300 boundary) |
| V1-9 | **Options**: unknown name → wrapped `ValueError` naming the parameter; `presolve=False` accepted (bool→int); `log_fn` produces a non-empty file containing `cuOpt version`; a user `log_file` is overridden by `log_fn`. | C-R24 |
| V1-10 | **Refusals**: `warmstart_fn` → `NotImplementedError`; `solution_fn` → `NotImplementedError`; `basis_fn` → warning only; SOS model → `ValueError` with the reformulation hint; indicator model → `ValueError`. | C-R25 and the honest-guard DoD line |
| V1-11 | **No-GPU degradation** in a subprocess with `CUDA_VISIBLE_DEVICES=''`: `"cuopt" not in linopy.available_solvers`. | C-R18 |
| V1-12 | **Fork safety**: probe, `os.fork()`, solve in the child; child exits 0. Run once against the *naive in-process* probe to prove the check can fire (charter §9). | C-R28 |
| V1-13 | **QP (WP-F)**, three rows with deliberately different scopes. **V1-13a — the Q-convention guard**: the 3-variable model from exploration item 12 (`M.c=[-3,-1,2]`, `M.Q=[[2,1,0],[1,4,0],[0,0,1]]`), positive-definite with a unique optimum — solution *and* objective versus live HiGHS, for `min` and `max`. The naive `M.Q` encoding returns `obj=-2.14285714` with status `Optimal` and **half** the solution vector, against `-4.28571429`. **V1-13c** re-runs the same model with `Q_cuopt = M.Q` to demonstrate once that the guard actually fires. **V1-13b — fixture coverage, not convention coverage**: `quadratic_model` (objective and the **`x` block only**) and `quadratic_model_cross_terms` (full primal — it is non-degenerate). `quadratic_model` **cannot** catch the `0.5·M.Q` trap: its optimum is `x = 0` for *any* positive scaling of `Q`, so both encodings give the same `x`. Its 10 `y` variables are underdetermined (zero cost, zero curvature, `ub=+inf`) and are never compared against the oracle — measured cuOpt `y=155.979` vs HiGHS `y=9.9999995`, both optimal. | C-R17, C-R19 for Q |
| V1-14 | **MIQP (WP-F)**: a quadratic objective plus one integer variable → `NotImplementedError` mentioning MIQP. | C-R23 |
| V1-15 | **Medium random LP** (n=2000, m=1000, fixed seed): objective versus live HiGHS under linopy's cuOpt defaults. | that the default method and tolerances are usable beyond toy sizes |
| V1-16 | **Bare solve**: `model.solve("cuopt")` with no `io_api` succeeds, `condition == "optimal"`, objective matches live HiGHS, and exactly one fallback warning is emitted. | C-R29, B3 |
| V1-17 | **No temp-file leak**: after V1-16, `list(Path(m.solver_dir).glob("linopy-problem-*")) == []`. | C-R29, B4 |
| V1-18 | **Interrupt helper** (`test_cuopt_interrupt.py`, CPU): a dummy blocking callable; `_thread.interrupt_main()` from a helper thread; assert `KeyboardInterrupt` reaches the main thread quickly and the worker still completes. | C-R22, and that the helper needs no GPU |

### 8.4 V2 — linopy's own suite

`pytest --run-gpu`: cuOpt runs as `("cuopt", "direct", False)` through the `params` matrix, the MIP
group (`feasible_mip_solvers`), the QP group (`quadratic_solvers`, after WP-F), and
`gpu_solvers` → `GPU_SOL_TOL`. Plain `pytest` must show cuOpt **skipped** and no new failures versus
`logs/05-baseline-pytest.txt`.

**No change to `CPU_SOL_TOL`/`GPU_SOL_TOL`.** The existing `GPU_SOL_TOL = 2.5e-4`
(`test_optimization.py:81`) is inherited as-is via `GPU_ONLY`.

### 8.5 Pre-decided contingencies (frozen at Gate 3 — a fix agent may not improvise here)

These exist so that a miss has exactly one authorised response, decided **before** the freeze.

1. **`test_quadratic_model_unbounded`** (B10): cuOpt returns `NumericalError` → `internal_solver_error`.
   *Authorised:* one cuopt-specific branch in that test, citing the measured status, in the shape of
   the existing cuPDLPx/mindopt branches. *Forbidden:* widening the accept-list; re-mapping
   `NumericalError` to `unknown`.
2. **`test_solvers.py:99-103`** `pytest.approx` default `rel=1e-6` on `3.3`, `-0.1`, `1.7` (B14).
   *Authorised, only if measured to miss:* add
   `GPU_SOL_TOL: float = 2.5e-4` / `CPU_SOL_TOL: float = 1e-6` module constants to
   `test_solvers.py` with a comment pointing at `test_optimization.py:80-81`, and parametrize
   **those three assertions only** with
   `tol = GPU_SOL_TOL if solver_supports(solver, SolverFeature.GPU_ONLY) else CPU_SOL_TOL`. This is
   the move #516's own commit 6 made in `test_optimization.py`. *Forbidden:* a bespoke number, a
   wider scope, or a second widening if `2.5e-4` also misses — that is a report-and-stop.
3. **`test_model_with_inf`** exact equality (`solution.x == 0`, `solution.y == 10`,
   `test_optimization.py:722-723`) (B14). Measured currently exact under `method=3`. *Authorised if
   it misses:* nothing. Report the numbers verbatim and stop on that item. Rounding integer
   solutions inside the solver class would be a silent data change and is **not** authorised.
4. **`test_infeasible_model`** (§2.7 item 2). *Authorised if cuOpt's condition does not contain
   `"infeasible"` or the status is not `"warning"`:* one cuopt branch in that test citing the
   measured status. *Forbidden:* changing the status map to make it pass.
5. **`method=3` for MILP** (§5b / G8): the per-class fallback described in §5b, and nothing else.
6. **`mypy` and the `_LazyModule` ignore comment** (§2.1.2): add `# type: ignore[assignment]` and
   paste mypy's message. Nothing else.

Any miss not on this list is reported verbatim, with actual numbers, and work stops on that item
(I12, §3.6).

---

## 9. Docs plan (charter §6.6b — every row)

| Surface | Content |
|---|---|
| `doc/gpu-acceleration.rst` | New `cuOpt` section, placed **before** the existing cuPDLPx section (cuOpt is the more capable solver). Contains: one-line description (NVIDIA cuOpt — concurrent GPU barrier/PDLP plus CPU dual simplex); **install** `pip install "linopy[cuopt]"` with the uv equivalent below it, **Linux-only**, **driver ≥ 525.60.13**, **compute capability ≥ 7.0** on the same lines, the `cu12` vs `cu13` caveat, and the warning that the bare `cuopt` package on PyPI is **a different project**; **usage** `m.solve("cuopt")` and `m.solve("cuopt", io_api="direct")`; **supported** LP, MILP, semi-continuous variables, ranged/equality/inequality rows, duals for LP, MIP gap and dual bound *(the QP line is added by commit 5, not commit 3 — G10)*; **unsupported** SOS, indicator constraints, MIQP, warm start, basis files, solution files, reduced costs, file `io_api`s — with the note that `io_api=None` or a file `io_api` transparently falls back to the direct API with one warning, and that `keep_files=True` is unsupported and raises *(A2 — corrected from "an empty problem file remains" against WP-E's measurement; see §2.1.4/B4 note and KI7)*; **notes** `method` defaults to 3 (Barrier) in linopy with the reason (upstream repeat-solve crash under `method=0`) and the guidance to try `method=1` (PDLP) for very large sparse LPs; `log_file`/`log_to_console` and that `log_file` truncates; Ctrl-C cancels the wait but the GPU work runs to completion, and `time_limit` is the only hard bound; default tolerances are 1e-4 so objectives can differ from a simplex solver in the 4th–5th significant digit; on a GPU-less host cuOpt is simply absent from `available_solvers`, and what cuOpt's `Set CUOPT_REMOTE_HOST` message means if a stale driver lets the probe pass. Update the existing "Verifying Installation" prose to mention cuOpt and the `pytest -k cupdlpx --run-gpu` example to `-k "cuopt or cupdlpx"`. |
| `doc/prerequisites.rst` | Under GPU-accelerated solvers: the cuOpt bullet with its link, and the one-command install `pip install "linopy[cuopt]"` **as the headline**, stating Linux-only and the driver/CC requirement on the same lines. `pypi.nvidia.com` is **not mentioned** — the whole tree is on PyPI, and naming an optional index invites `--index-strategy`, which §4.2 shows is never needed. |
| `doc/contributing.rst` | Extend the existing GPU-testing paragraph: name cuOpt alongside cuPDLPx, and one sentence that cuOpt-specific tests live in `test/test_cuopt.py` and need `--run-gpu`. **Do not** touch the `uv sync --extra dev --extra solvers` line (I9 / I4). |
| `doc/index.rst` | Add `cuOpt <https://docs.nvidia.com/cuopt/>`__ (GPU-accelerated) to the solver list. |
| `README.md` | Add `* [cuOpt](https://docs.nvidia.com/cuopt/)` to the solver list. |
| `doc/api.rst` | **Add both** (B13): `solvers.cuOpt` to the *Implementations* autosummary (before `solvers.cuPDLPx`, matching that list's case-sensitive alphabetical order at `:626-637`), **and** `model.Model.to_cuopt` to the *IO* autosummary (`:111-116`). The "no `to_*` bridge is listed" objection is real but weaker than it looks: `Model.to_netcdf` is bound by the identical `to_netcdf = to_netcdf` assignment (`model.py:2507`) and **is** listed at `:115`, so the autosummary entry is proven to build. Charter §6.6b names `to_cuopt` explicitly; two lines close the row instead of arguing it at Gate 6. The missing siblings (`to_gurobipy`, `to_highspy`, `to_cupdlpx`, `to_mosek`, `to_xpress`) are a **pre-existing doc gap** → `logs/45-known-issues.md`, not fixed here (I13, I4). |
| `doc/release_notes.rst` | One bullet under *Upcoming Version*: support for NVIDIA cuOpt (LP, MILP) via `io_api="direct"`, installable with `linopy[cuopt]` on Linux with a CUDA 12 GPU. Commit 5 amends it to name QP. |
| `pyproject.toml` | The extra carries the four-line comment in §5i — why it is separate, the marker, the driver/CC floor. That is where extras are explained in this file. |
| `examples/using-solvers.ipynb` | **Include** — one markdown sentence appending cuOpt to the existing backend list in the intro cell. **Markdown only, no code cell**: the notebooks execute during the docs build with no GPU, so a cuOpt code cell would break ReadTheDocs. Logged as a D-entry. |
| `CLAUDE.md` | **Not touched.** #516 edited it (+9/−1); this repo's `CLAUDE.md` now delegates to `AGENTS.md` and says nothing solver-specific, so an edit would be a drive-by. This is an expected difference in the §6.6b gap check. |

Mechanical gap check (§6.6b) — expected differences and their justifications:
`CLAUDE.md`, `linopy/solver_capabilities.py` and `test/conftest.py` appear in #516's file list and
**not** in ours (delegated file; zero-diff shim; generic gating). `test/test_cuopt.py` and
`test/test_cuopt_interrupt.py` are new files #516 has no analogue for.

---

## 10. Explicit non-goals

1. **File `io_api`s** (`lp`, `lp-polars`, `mps`) for cuOpt, and therefore `READ_MODEL_FROM_FILE` /
   `LP_FILE_NAMES` (§1.2). `io_api=None` or a file `io_api` builds directly with a warning; it does
   not read the file. Recorded as an evidence-backed follow-up.
2. **Solution-file output** and **basis files**.
3. **Warm start** in any form — PDLP payload or initial point (§5h).
4. **Reduced costs**, including any new linopy surface for them (§5e).
5. **Quadratic constraints** (`add_quadratic_constraint`) and **MIQP**.
6. **SOS / indicator constraints** for cuOpt (genuinely absent upstream; linopy raises).
7. **Persistent in-place updates** (`supports_persistent_update` stays `False`).
8. **MIP callbacks**, `BatchSolve`, `problem.Problem`, `cuopt.routing`, `cuopt.distance_engine`, and
   the gRPC/server client.
9. **Remote cuOpt** (`CUOPT_REMOTE_HOST`) — including as a fallback on GPU-less hosts.
10. **`cu13` wheels** and any attempt to auto-select `cu12`/`cu13`.
11. **Tuning cuOpt's numeric defaults** other than `method`: no default `time_limit`, no tightened
    tolerances, no `pdlp_solver_mode`, no `crossover`.
12. **Touching `CPU_SOL_TOL` / `GPU_SOL_TOL`** or any existing threshold, except the single
    pre-authorised contingency in §8.5 item 2.
13. **Fixing any pre-existing linopy defect** noticed on the way (I13 — record in
    `logs/45-known-issues.md`). Specifically refused: the shared "not installed" message
    (`model.py:2078`), cuPDLPx's own guards, the `doc/api.rst` `to_*` gap, cuOpt's upstream
    reduced-cost sign for maximised models with `<=` rows, the unobservable
    `DualInfeasible`/`Unbounded` status rows, the `int32` CSR index truncation, and — **entry to add
    when WP-A lands** — **a bare `model.solve("cupdlpx")` raises today.**
    `Model.get_problem_file(io_api=None)` returns a `.lp` path
    (`model.py:1885-1893`: `suffix = ".mps" if io_api == "mps" else ".lp"`) and
    `cuPDLPx._run_file` accepts **only** `.netcdf` (`solvers.py:4205-4214`), so
    `test_solvers.py:48` and `:209` are red for cuPDLPx on any machine where it is installed. This is
    the strongest independent argument for the `_build_file` graft (§2.1.4) — and per I13 it is
    **recorded, never fixed**: cuOpt does not inherit the wart, and repairing cuPDLPx's own path is
    out of scope and would pollute the diff (I4).
14. **Benchmarks** in the cuOpt venv (pandas pin collision, §4.4).
15. **Editing `linopy/solver_capabilities.py`, `linopy/constants.py`, `test/conftest.py`** — all
    three are satisfied by verification with a zero-line diff.
16. **Rewriting cuOpt's own error messages** in `_solve` (§5c) — masking them would hide genuinely
    different failures.

---

## 11. Risks

Ordered by probability × cost. Each names its mitigation and, where relevant, the verification row
that would catch it.

1. **`test_quadratic_model_unbounded` and the QP fixtures** (WP-F's largest surface).
   `quadratic_model` has a *singular* PSD Hessian and `quadratic_model_unbounded` is genuinely
   unbounded; both were measured, but only through raw `DataModel` calls, not through the finished
   class. Mitigated by the frozen disposition in §8.5 item 1 and by V1-13.
2. **`method=3` (Barrier) as the default.** Barrier's scaling to very large sparse LPs was never
   measured; the only 10⁶–10⁷-variable evidence is for PDLP. A user with a huge model may hit memory
   pressure or `NoTermination` where cuOpt's own default would have coped. Mitigated by the
   documented escape hatch, V1-15 (n=2000), and §5b's named contingency. Residual risk accepted
   because the alternative default is a deterministic SIGSEGV.
3. **The pad row (§5j / C-R21).** `vstack` on a `csr_array`, `indices`/`indptr` dtype preservation,
   and the `dual[:n_rows]` slice are one-line details that are easy to get subtly wrong; a mistake
   would misalign **every** dual by one row rather than fail loudly. Mitigated by V1-6, which
   compares duals against HiGHS both with and without constraints.
4. **The subprocess probe's cost and environments.** 0.28 s once per process is cheap here, but the
   probe fails closed in an embedded interpreter (no usable `sys.executable`) or a sandbox that
   forbids `subprocess`. Those two are now the **only** fail-closed scenarios: a missing
   `cuda.bindings` cannot happen when cuOpt is installed — `cuda-bindings 12.9.7` is a hard
   transitive requirement via three of `cuopt-cu12`'s direct `==26.8.*` dependencies (`cudf-cu12`,
   `rmm-cu12`, `pylibraft-cu12`, each `cuda-bindings<13.0,>=12.9.2`), so
   `_CUDA_PROBE_SNIPPET`'s import cannot fail for a missing module. cuOpt then silently disappears —
   which reads exactly like "not installed".
   Mitigated by V0-5 pinning the positive branch on this machine, and by documenting the behaviour.
   This is the single most likely place this plan is wrong.
5. **V1-3 is brittle by design.** A cuOpt point release that changes presolve routing turns it red
   with nothing actually broken. Its docstring says exactly what to do. A loud, informative red is
   better than silently losing coverage of the worst bug in this integration.
6. **Diverging from repo precedent on `is_available`.** A maintainer may reject a GPU probe in
   `is_available` on principle ("membership means importable"). D6 settles it for this run, but it is
   a real fork and it contradicts the code's own docstring. Surfaced in the handoff.
7. **Shared assertions tighter than cuOpt's default tolerances** (§8.5 items 2 and 3). Both were
   measured passing; that is a measurement, not a guarantee, and `test_solvers.py`'s `-0.1` is
   1e-7-tight at `rel=1e-6`.
8. **`explicit_coordinate_names` warning untested by V2.** cuOpt joins `params` only as
   `("cuopt", "direct", False)`, so the `True` variant never runs. Covered by an explicit V1-10-style
   assertion if WP-D has room; otherwise recorded as a coverage gap.
9. **Diff drift in WP-F.** QP touches `solvers.py` after Phase 5 has already reviewed it; the §6.4a
   authorship and no-merge-commit checks must re-run at Gate 6 (the charter requires this).
10. **`int32` CSR index overflow.** cuOpt casts `int64` indices to `int32` with no range check and no
    warning; a >2³¹-nonzero model would be silently truncated. Not reachable on a 16 GB T4.
    Recorded in `logs/45-known-issues.md` as a latent upstream risk.

---

## 12. Blocking-finding resolution index

Every finding from `logs/50-review.md`, with where it is resolved. **None is resolved silently.**

| # | Disposition | Where |
|---|---|---|
| B1 | **Accepted.** Singly-bounded pad row on a variable with a finite bound; residual all-free case raises `NotImplementedError` (the two feasible-set-changing alternatives are rejected by name). | §2.1.6, §5j, V1-6 |
| B2 | **Accepted.** Never `raise ValueError` for zero-nnz; cuPDLPx's line is explicitly not copied. | §5j |
| B3 | **Accepted.** `_build_file` overridden; `_run_file` left at the base. One precision note: of the three bare `simple_model.solve(solver)` call sites the review cites, `:48` and `:209` are bare (`test_solver_instance_attached_after_solve`, `test_solver_close_releases_state`); `:81` passes `io_api="direct"` and `:286` skips for non-`READ_MODEL_FROM_FILE` solvers. The finding stands on `:48` and `:209`. | §1.3, §2.1.4, §5k |
| B4 | **Accepted.** `self._problem_fn` is set from `build_kwargs["problem_fn"]` so linopy's own `finally` unlinks it — chosen over unlinking in the override, because a user-supplied `problem_fn=` must never be deleted by us. Verification row added. | §2.1.4, V1-17 |
| B5 | **Accepted.** Out-of-process, `functools.cache`d probe; every in-process CUDA call is forbidden, and fork safety is a check that is proven to fire. | §2.1.9, §5c, V1-12 |
| B6 | **Accepted.** `bool(count)` inside the guarded region, fail **closed**, warning naming driver and compute capability. | §2.1.9 |
| B7 | **Accepted.** D6's position kept; plan-3/5's reasoning retained as the recorded downside. | §5c |
| B8 | **Accepted.** Empty MIP dual returned **unscattered**, with the reason in a code comment. | §2.1.7, §5e |
| B9 | **Accepted.** Blanket `bool → int`; unknown-name `ValueError` wrapped; value errors untouched. Allowlist, sniffing and retry all dropped. | §2.1.8, §5f |
| B10 | **Accepted.** Disposition frozen with the measured status; re-mapping and accept-list widening both rejected by name. | §5j, §8.5 item 1, V2-2 |
| B11 | **Accepted.** `MIP_DUAL_BOUND_REPORT` declared, premise now measured, contingency retained as insurance. | §3, V1-4 |
| B12 | **Accepted.** `SEMI_CONTINUOUS_VARIABLES` declared, and the zero-automatic-coverage consequence is stated so the test lands in `test_cuopt.py`. | §3, §8.1, V1-7 |
| B13 | **Accepted.** Both `api.rst` rows added; missing siblings recorded as a known issue. | §9 |
| B14 | **Accepted.** Both assertions are named verification rows with the remedy pre-decided — and for `test_model_with_inf` the pre-decided remedy is explicitly *nothing*. | §8.5 items 2–3, V2-3, V2-4 |
| B15 | **Accepted.** No `test_solver_method_options` row. The related mechanical point is honoured: `time_limit_option` **must** gain a `"cuopt"` key because it is indexed unconditionally. | §2.7 |
| B16 | **Accepted.** Three probe rows, all able to fire; the fork-safety row is run once against the naive probe to prove it. | V0-5, V0-6, V0-7, V1-11, V1-12 |
| **B17 (new, found here)** | plan-1's `len(self._vlabels or ())` / `len(self._clabels or ())` **raises `ValueError` on numpy 2.4.6** — for a non-empty array *and* for an empty one. Replaced with `int(arr.size)` plus a `None` guard. Pasted evidence in §2.1.7. **Flagged for the selector to re-check.** | §2.1.7 |
| **B18 (new, found here)** | The `_build_file` override would drop the base method's `io_api` validation (`solvers.py:765-768`), the only guard on `Solver.from_name(..., io_api="bogus")`. Guard re-stated in the override. `test_non_supported_solver_io` passes either way because `Model.solve` validates first (`model.py:2022-2024`) — so this is a contract preservation, not a red test. **Flagged for the selector to re-check.** | §2.1.4 |

| **B19 (round 2, blocking)** | **Accepted in full, and independently confirmed before acting.** The `quadratic_model` fixture (`test_optimization.py:261-271`) has objective `x*x` only, so every `y_i` has zero cost, zero curvature, `lb=0` and `ub=+inf` — the optimal face is unbounded in `y`. Measured by the reviewer at `method=3`: cuOpt `y = 155.97919941` vs HiGHS `y = 9.9999995`, **both optimal**, `\|Δy\| = 145.98` against a frozen `atol = 1e-4` (1.46e+06×). My own confirmation from the shared suite rather than a re-run: `test_quadratic_model:876` asserts `y.round(3) >= 10` — an **inequality** — and `test_quadratic_model_wo_constraint` (`:927-944`) asserts no `y` at all, so linopy's own tests already treat `y` as underdetermined. A frozen row that goes red on a non-defect is worse than no row: under I12/§3.6 the WP-F agent must report and stop, and the pressure it creates is to widen the `atol` — precisely the failure §3.6 exists to prevent. **Four edits made:** V1-13b scoped to the `x` block plus the objective (with `quadratic_model_cross_terms` keeping its full primal comparison, and the asymmetry explained); V1-6b likewise; one Nondeterminism row for the underdetermined `y` face with the measured numbers; and §8.3's V1-13 entry rewritten so the `0.5·M.Q` guard is attributed to V1-13a (positive-definite, unique optimum) and V1-13c (deliberate-failure demo) — `quadratic_model` **cannot** carry it, because its optimum is `x = 0` for any positive scaling of `Q`. Two consistency edits followed: §7.2's WP-F acceptance criterion and §8.3's V1-6 row now state the same scope, so an agent reading `FINAL.md` alone cannot rebuild the `y` comparison. **The tolerance tier itself is unchanged** — `CUOPT_PRIMAL_ATOL_DEGENERATE = 1e-4` was not the defect and is not widened; the fix narrows a row's *scope*. Per the reviewer's re-check note, its rationale now cites `\|Δx\| = 5.637e-05` (cuOpt `5.68669e-05` vs HiGHS `4.99999949e-07`, margin 1.77×) instead of cuOpt's raw `x`. | §8.3, §7.2, §0 |

**Grafts G1–G10:** all applied — G1 §2.1.4; G2 §2.1.7 and §4; G3 §2.1.8; G4 §7.2; G5 §8.5;
G6 §2.2; G7 §2.1.3; G8 §5b; G9 §8.1; G10 §7.3 and §9.
**Rejections R1–R13:** all honoured — R1 §5c; R2/R3 §2.1.9; R4/R5 §5j; R6 §5k; R7/R8 §2.1.8;
R9 §3; R10 §1.3; R11 §2.7; R12 §1.2 and §10; R13 §2.1.7.

**Non-blocking observations 1–13** are all carried: 1 → §2.1.3 (G7); 2 → §1.3; 3 → §4; 4 → §2.1.2
(charter's literal `_LazyModule("cuopt")` chosen, with the routing comment as the guard against
widening); 5 → §2.1.2 (no ignore comment); 6 → §2.1.6 (no `set_objective_offset`); 7 → §2.1.6 (dtype
notes); 8 → probe order **after** `cupdlpx`, so no existing user's default solver changes — logged as
a D-entry; 9 → §4 (`NoTermination ≠ unknown`); 10 → §2.11 bulk estimate; 11 → §8.5 item 1 and V1-13;
12 → §5d and WP-F acceptance criterion; 13 → §2.2 (G6).

---

## 13. Decisions to record in `logs/30-decisions.md` at Gate 3

The lead records these as D-entries (this document is the evidence; the planner does not write that
log):

0. **The `C-R18`–`C-R29` registry itself** — charter §5.3 says new IDs are "recorded in
   `logs/30-decisions.md`", and `FINAL.md` §6 is the first place they have ever been written down.
   One D-entry listing all twelve IDs and their one-line topics is what makes "IDs are never
   renumbered or reused" enforceable by a later agent that never reads this file. Also record
   §6.4a's commit-style override of I5 (D2 already covers it; confirm rather than duplicate).
1. Entry point: `DataModel` + `Solve`, `io_api="direct"` only; no file `io_api` in this PR (§1).
2. Probe order: `"cuopt"` **after** `"cupdlpx"` — preserves every existing user's default solver.
3. Probe mechanics under D6: out-of-process, cached, fail-closed (§2.1.9).
4. Status maps keyed on the status **name**, two dicts by problem category (§4).
5. Blanket `bool → int` option coercion (§2.1.8).
6. `_build_file` override, `_run_file` untouched, temp file handed back via `_problem_fn` (§2.1.4).
7. Zero-nnz pad shape, and `NotImplementedError` for the all-free residual case (§5j).
8. `test_quadratic_model_unbounded` disposition (§8.5 item 1).
9. Extra named `cuopt`, single Linux-marked requirement (§5i).
10. `doc/api.rst` gains both `solvers.cuOpt` and `model.Model.to_cuopt` (§9).
11. `examples/using-solvers.ipynb` gains a markdown-only mention, no code cell (§9).
12. `accepted_io_apis` deliberately not declared (§1.3).
