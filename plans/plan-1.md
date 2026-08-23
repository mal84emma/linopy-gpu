# plan-1 — NVIDIA cuOpt support in linopy

*Planner k=1 · Phase 2 · written against `master` @ `09c34dd` + charter §5 map + `logs/10-exploration.md`
(Phase 1 empirical truth) · model: Opus 5 (`claude-opus-5[1m]`)*

Reading basis: charter §§4, 5, 6.3a, 6.4, 6.4a, 6.6b, 9; `AGENTS.md`; `logs/30-decisions.md` rules block;
`logs/10-exploration.md` in full; read-only source review of `linopy/solvers.py`, `linopy/io.py`,
`linopy/model.py`, `linopy/matrices.py`, `linopy/constants.py`, `linopy/objective.py`,
`linopy/solver_capabilities.py`, `test/conftest.py`, `test/test_optimization.py`, `test/test_solvers.py`,
`test/test_semi_continuous.py`, `test/test_available_solvers.py`, `test/test_highs_keyboard_interrupt.py`,
`pyproject.toml`, `doc/*.rst`, `README.md`, the installed `cuopt` 26.08.00 package sources, and
`gh pr view 516 --json files` (shape only: +771/−62 over 15 files).

No code was run (per brief).

---

## 1. Entry-point decision

**Decision: `data_model.DataModel` + `solver.Solve` (via `cuopt.linear_programming`) as the single
entry point. `io_api = "direct"` is the only io_api cuOpt supports; `accepted_io_apis` is *not*
declared (see below).**

### 1.1 Agreement with Phase 1 Q0's primary recommendation

Adopted without reservation. The evidence in Q0 is one-sided:

* **Speed**: 0.012 s vs 0.86 s (`problem.Problem`) vs 0.83–0.95 s (file path) to build a 10⁵-variable
  model. `Problem` is ~70× slower and per-variable Python calls do not improve with scale.
* **Expressiveness**: `DataModel` is the only candidate that can express ranged rows
  (`problem.py:1524`, `:1593-1594` show `Problem` structurally cannot). linopy's `M.sense`/`M.b` pair
  never *emits* a ranged row today, so this is not load-bearing for us — but it means `Problem` is a
  dead end for any future ranged support, at 70× the cost.
* **Fit**: 13–16 lines of translation from `model.matrices`, and `M.A.tocsr()` already produces
  exactly the dtypes cuOpt stores (float64/int32/int32, exploration item 6).
* **No lost surface**: everything §4.3 listed as `Problem`-only (warm start, MIP callbacks, options,
  log control) actually lives on `SolverSettings`, which all candidates share; `Slack`/`DualValue`/
  `ReducedCost` are `populate_solution` copies of the same `Solution` arrays.
* **Stability**: public, documented, no experimental marker. `internals` is not a candidate.

### 1.2 Divergence from Phase 1 Q0's *secondary* recommendation (file `io_api`s)

Q0 recommends additionally declaring `accepted_io_apis ⊇ {"lp", "mps"}` via
`cuopt.linear_programming.io.Read`, calling it "~3 lines". **I diverge: no file io_api in this
change.** The 3-line estimate is the cuOpt half of the contract; the linopy half is not 3 lines and
rests on three *unverified* behaviours:

1. **Solution files.** Declaring `SolverFeature.READ_MODEL_FROM_FILE` puts `cuopt` into
   `file_io_solvers` (`test_optimization.py:45`) and therefore into `test_set_files` and
   `test_set_files_and_keep_files` (`:635`, `:656`), which pass a real `solution_fn` and then
   **assert the file exists** for `io_api != "direct"`. cuOpt's `solution_file` parameter exists but
   was explicitly left untested (exploration Unknowns, agent B: "UNKNOWN — whether `log_file` and
   `solution_file` interact"). The alternative — writing the solution file ourselves in Python —
   is something no other linopy solver does (every one delegates to the vendor writer), i.e. a
   Phase 6 style outlier.
2. **Explicit coordinate names.** `file_io_solvers` are exercised with
   `explicit_coordinate_names=True` across ~30 parametrized tests. Q0's fidelity measurement used
   default labels (`x0`, `c0`). Whether cuOpt's LP/MPS parser accepts linopy's explicit-coordinate
   names (`name(coord)#label` shapes, which `_names_to_labels` at `solvers.py:81` is built for) was
   never measured.
3. **The M2 sense flip on a file-read model.** The mitigation below (§5a) requires handing cuOpt a
   *minimisation*. On a `DataModel` produced by `io.Read` that means reading back
   `get_objective_coefficients()`, negating, and re-asserting `set_maximize(False)` — plausible,
   unmeasured.

Cost of declaring it anyway: ~6 extra parametrizations × ~60 GPU-gated tests (≈360 extra GPU solves
per `--run-gpu` run) on top of three unverified contract points, against the §9 bulk yardstick. That
is the wrong trade for a first landing. It is a clean, cheap follow-up PR once those three points
are measured — and Q0's finding that cuOpt's reader round-trips linopy's own `.lp`/`.mps` with full
fidelity (names, order, integrality, sense, Q) is exactly the evidence that follow-up needs.

**What would change my mind:** pasted evidence for all three points above. Two of the three are
one spike script each.

### 1.3 The `io_api` surface actually declared

* `features` contains `DIRECT_API` and **not** `READ_MODEL_FROM_FILE` / `LP_FILE_NAMES`. In this
  repo that *is* how the io surface is declared: `Model.solve`, `_build_file`,
  `get_available_solvers_with_feature`, and the test matrix all read the features, and
  `SolverFeature.READ_MODEL_FROM_FILE` is the only switch that admits a solver to the file path.
* **`accepted_io_apis` is deliberately left at the base-class default** (`frozenset()`).
  Justification: grep shows **no concrete `Solver` subclass declares it** and **no production code
  reads it** (`linopy/solvers.py:466` is the only definition; the sole other hit is a test double at
  `test/test_persistent_solver_orchestrator.py:24`). Declaring it on cuOpt alone would be dead code
  and a lone stylistic outlier — I4 (no drive-by additions) and §6.6a ("no methods the siblings do
  not have without a reason"). The supported surface is documented in the class docstring and in
  `doc/gpu-acceleration.rst`.
* **`_run_file` does not dead-end.** cuPDLPx's `_run_file` (`solvers.py:4195-4225`) reads the file
  only when it is a `.netcdf`, so `model.solve("cupdlpx")` *without* `io_api="direct"` raises
  `NotImplementedError` today. cuOpt keeps the same shape but drops the netcdf restriction: with a
  Model attached (always true on the `Model.solve` path) it warns once and builds through the direct
  API. This satisfies the DoD line "`model.solve("cuopt")` solves LP and MILP models correctly"
  literally, costs ~10 lines, and is transparent (a `logger.warning` naming `io_api="direct"` as the
  way to skip the wasted file write).

---

## 2. Full file-by-file change set

Line anchors are current-`master` (`09c34dd`) positions. Insert points, not replacements, unless
stated. Estimated volume in brackets.

### 2.1 `linopy/solvers.py`  [≈ +235 / −0]

| Anchor | Change |
|---|---|
| `:34` `from scipy.sparse import tril, triu` | → `from scipy.sparse import csr_array, tril, triu, vstack` (needed only for the empty-`A` pad, §5d/§2.1.6). |
| `:148` `SolverFeature` | **No edit.** Every feature cuOpt needs already exists. C-R1 is satisfied by the class's `features` frozenset. Adding an enum member would be a drive-by (I4). |
| `:175-181` `TYPE_CHECKING` | add `import cuopt` (isort order: before `cupdlpx`). |
| `:304` after `cupdlpx = _LazyModule("cupdlpx")` | add `cuopt = _LazyModule("cuopt")` **plus a two-line comment**: only `cuopt.linear_programming` is ever touched — importing `cuopt.routing` installs a global `sys.excepthook` that writes `error_log.txt` into the cwd (`routing/vehicle_routing_wrapper.pyx:144-160`). |
| after `_run_highs_with_keyboard_interrupt` (`:189-252`) | new module-level helper `_run_cuopt_with_keyboard_interrupt(solve)` (§5g). |
| near `_has_module` (`:307`) | new module-level `@functools.cache def _cuda_device_available()` (§5c). |
| `:346` `SolverName` | add `cuOpt = "cuopt"` immediately after `cuPDLPx = "cupdlpx"`. Vendor spelling; member name must equal the class name for the `solver_capabilities` shim. |
| after the `cuPDLPx` class (ends `:4416`), before `_solver_class_for` (`:4419`) | new `class cuOpt(Solver[None])` (§2.1.1–§2.1.8). |
| `:4426` `QUADRATIC_SOLVERS` | **No edit** (derived from the feature). |
| `:4443` `_SOLVER_PROBE_ORDER` | insert `"cuopt",` after `"cupdlpx",` (before `"pips"`). |

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

    Some example options are:

    * method : 3 (Barrier) by default in linopy - 0 (Concurrent), 1 (PDLP) and
      2 (DualSimplex) are alternatives. linopy does not use cuOpt's own default
      of 0, which crashes the process on repeated solves (see notes below).
    * time_limit : inf by default.
    * log_to_console : True by default.
    * absolute_primal_tolerance / relative_primal_tolerance / ... : 1e-4 by default.

    Notes
    -----
    Duals are returned in linopy's (HiGHS's) sign convention without
    transformation. Maximisation problems are handed to cuOpt as the equivalent
    minimisation because cuOpt 26.08 returns negated duals for maximised models
    that its presolve solves outright; the objective, the duals and the MIP
    bound are negated back here.

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
            # SolverFeature.QUADRATIC_OBJECTIVE added by WP-F (QP)
        }
    )

    # cuOpt's own default (0 = Concurrent) segfaults on the 2nd-3rd solve in a
    # process for models above ~1300 variables; Barrier is GPU-resident, stable
    # across repeated solves, the most accurate of the single methods, and the
    # method cuOpt forces for quadratic objectives anyway.
    _DEFAULT_METHOD: ClassVar[int] = 3
    # cuOpt types most parameters as int; only these accept a Python bool.
    _BOOL_PARAMETERS: ClassVar[frozenset[str]] = frozenset({"log_to_console", "crossover"})
```

`supports_persistent_update` / `supports_sign_update` stay at their `False` defaults —
`DataModel` has no update API (`Problem.updateConstraint` exists but rebuilds, `problem.py:1650`).

#### 2.1.2 `is_available` / `_license_probe`

```python
    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("cuopt") and _cuda_device_available()

    @classmethod
    def _license_probe(cls) -> None:
        dm = cuopt.linear_programming.DataModel()
        dm.set_csr_constraint_matrix(np.array([1.0]), np.array([0], dtype=np.int32),
                                     np.array([0, 1], dtype=np.int32))
        dm.set_constraint_lower_bounds(np.array([2.0]))
        dm.set_constraint_upper_bounds(np.array([np.inf]))
        dm.set_variable_lower_bounds(np.array([0.0]))
        dm.set_variable_upper_bounds(np.array([np.inf]))
        dm.set_objective_coefficients(np.array([1.0]))
        settings = cuopt.linear_programming.SolverSettings()
        settings.set_parameter("log_to_console", False)
        settings.set_parameter("method", cls._DEFAULT_METHOD)
        cuopt.linear_programming.Solve(dm, settings)
```

Rationale: constructing a `DataModel` allocates **no** device memory (exploration item 9 — the CUDA
context appears at the first `Solve`), so a cuPDLPx-style construction-only probe would prove
nothing. The 1-var/1-row model above is the exact shape verified working in exploration item 5
(`1 var 1 row: min x, x>=2 -> Optimal obj=2.0`); cost ≈ 0.3 s once (first-solve CUDA init), and it
makes `licensed_solvers` mean "the GPU actually works". Exceptions are caught by
`Solver.license_status`.

#### 2.1.3 `_run_file` — transparent fallback

```python
    def _run_file(self, solution_fn=None, log_fn=None, warmstart_fn=None,
                  basis_fn=None, env=None, **kw) -> Result:
        if self.model is None:
            msg = ("cuOpt cannot read problem files. Rebuild the model with "
                   "io_api='direct'.")
            raise NotImplementedError(msg)
        logger.warning(
            "cuOpt does not support file IO. Building the model via the direct "
            "API instead; pass io_api='direct' to skip writing the problem file."
        )
        self._build_direct()
        return self._run_direct(solution_fn=solution_fn, log_fn=log_fn,
                                warmstart_fn=warmstart_fn, basis_fn=basis_fn, env=env)
```

#### 2.1.4 `_build_direct`

Mirrors cuPDLPx (`:4227-4244`) with the MIQP guard replacing the QP/MILP guard:

```python
    def _build_direct(self, **kwargs: Any) -> None:
        model = self.model
        assert model is not None
        if model.type in ("MIQP", "IQP"):
            msg = ("cuOpt does not support quadratic objectives together with "
                   "integer variables. Use a solver that supports MIQP "
                   "(gurobi, xpress, mosek).")
            raise NotImplementedError(msg)
        if kwargs.get("explicit_coordinate_names"):
            warnings.warn(
                "cuOpt does not support named variables/constraints. "
                "The explicit_coordinate_names parameter is ignored.",
                UserWarning, stacklevel=2,
            )
        self.solver_model = self._build_solver_model(model)
        self.io_api = "direct"
        self.sense = model.sense
        self._cache_model_labels(model)
```

`set_names` is accepted and ignored (absorbed by `**kwargs`): cuOpt's
`set_variable_names`/`set_row_names` cost one Python list per entity and buy nothing, because the
solution is indexed positionally. Documented in `doc/gpu-acceleration.rst`.

#### 2.1.5 `_build_solver_model` (staticmethod, so `to_cuopt` can reuse it via `from_model`)

```python
    @staticmethod
    def _build_solver_model(model: Model) -> cuopt.linear_programming.DataModel:
        """Build a cuopt DataModel that mirrors the linopy `model`."""
        M = model.matrices
        # cuOpt is always handed a minimisation: set_maximize(True) is the only
        # route onto a presolve path that returns negated duals (verified
        # against HiGHS on 21 models). `_solve` negates the results back.
        sign = -1.0 if model.objective.sense == "max" else 1.0

        lower = np.where(
            np.logical_or(np.equal(M.sense, ">"), np.equal(M.sense, "=")), M.b, -np.inf
        )
        upper = np.where(
            np.logical_or(np.equal(M.sense, "<"), np.equal(M.sense, "=")), M.b, np.inf
        )
        A = M.A.tocsr() if M.A is not None else None
        if A is None or A.nnz == 0:
            # cuOpt returns NoTermination and an empty solution for a constraint
            # matrix without nonzeros. One free row leaves the feasible set
            # untouched; its dual is dropped again in `_solve`.
            free_row = csr_array(([1.0], ([0], [0])), shape=(1, len(M.c)))
            A = free_row if A is None else vstack([A, free_row], format="csr")
            lower = np.append(lower, -np.inf)
            upper = np.append(upper, np.inf)

        dm = cuopt.linear_programming.DataModel()
        dm.set_csr_constraint_matrix(A.data, A.indices, A.indptr)
        dm.set_constraint_lower_bounds(lower)
        dm.set_constraint_upper_bounds(upper)
        dm.set_variable_lower_bounds(M.lb)
        dm.set_variable_upper_bounds(M.ub)
        dm.set_objective_coefficients(sign * M.c)
        # cuOpt knows 'C', 'I' and 'S'; every other character is silently taken
        # as continuous, so linopy's 'B' must be mapped explicitly. Binary
        # bounds are already 0/1 in M.lb/M.ub.
        dm.set_variable_types(np.where(M.vtypes == "B", "I", M.vtypes).astype("<U1"))
        return dm
```

QP adds three lines here in WP-F (§5d). `set_row_types` is deliberately never called (the bounds
route covers `<`/`>`/`=` and wins if both are set, `data_model.py:468-470`).
`set_objective_offset` is never called: linopy rejects constants in the objective
(`objective.py:191-192`), so there is nothing to offset.

#### 2.1.6 `_run_direct` / `_solve`

`_run_direct` forwards to `_solve(self.solver_model, ..., io_api=self.io_api, sense=self.sense)`,
exactly as cuPDLPx does (`:4284-4301`), **including `log_fn`** (cuOpt's settings object is created
per solve, so `log_fn` must reach `_solve`).

`_solve` shape (numpydoc docstring modelled on cuPDLPx's `:4313-4338`):

```python
        statuses = cuopt.linear_programming.solver.solver_wrapper
        LP_CONDITION_MAP: dict[Any, TerminationCondition] = {...}     # §4
        MILP_CONDITION_MAP: dict[Any, TerminationCondition] = {...}   # §4

        if warmstart_fn is not None:
            raise NotImplementedError("Warmstarting is not yet implemented for cuOpt.")
        if basis_fn is not None:
            logger.warning("Basis files are not supported by cuOpt. Ignoring.")
        if solution_fn is not None:
            raise NotImplementedError("Solution file output is not yet implemented for cuOpt.")

        settings = cuopt.linear_programming.SolverSettings()
        self._set_solver_params(settings, log_fn)
        sol = _run_cuopt_with_keyboard_interrupt(
            lambda: cuopt.linear_programming.Solve(dm, settings)
        )

        is_mip = sol.get_problem_category() != statuses.ProblemCategory.LP
        condition_map = MILP_CONDITION_MAP if is_mip else LP_CONDITION_MAP
        termination_condition = condition_map.get(
            sol.get_termination_status(), TerminationCondition.unknown
        )
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = sol.get_termination_reason()
        if sol.get_error_status() != statuses.ErrorStatus.Success:
            logger.error(f"cuOpt reported an error: {sol.get_error_message()}")

        sign = -1.0 if sense == "max" else 1.0

        def get_solver_solution() -> Solution:
            objective = sign * float(sol.get_primal_objective())
            primal = np.asarray(sol.get_primal_solution(), dtype=float)
            if primal.size != len(self._vlabels or ()):
                raise ValueError(
                    f"cuOpt returned {primal.size} primal values for "
                    f"{len(self._vlabels or ())} variables "
                    f"({sol.get_termination_reason()})."
                )
            primal = _solution_from_labels(primal, self._vlabels, self._n_vars)
            if is_mip:
                # cuOpt raises on duals for a MIP solution; linopy accepts an
                # empty dual array (model.py:2206).
                dual = np.array([], dtype=float)
            else:
                dual = sign * np.asarray(sol.get_dual_solution(), dtype=float)
                # drops the padded free row, if any (§2.1.5)
                dual = _solution_from_labels(
                    dual[: len(self._clabels or ())], self._clabels, self._n_cons
                )
            return Solution(primal, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)
        # report: runtime always; mip_gap + dual_bound (sign-corrected) for MIP
```

`SolverReport`: `runtime=float(sol.get_solve_time())`; for MIP,
`mip_gap=stats["mip_gap"]`, `dual_bound=sign * stats["solution_bound"]`,
`simplex_iterations=int(stats["num_simplex_iterations"])` from `sol.get_milp_stats()`
(documented at `solution/solution.py:368-405`), each inside `contextlib.suppress(Exception)` as
`Highs._solve` does (`:1887-1892`). LP iteration counts are **not** reported: cuOpt's
`lp_stats["nb_iterations"]` is method-dependent (PDLP iterations are neither
`simplex_iterations` nor `barrier_iterations`), and mislabelling them would be worse than omitting
them. `lp_stats["dual_residual"]` is never surfaced (meaningless under QP — exploration item 12).

#### 2.1.7 `_set_solver_params`

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
            if isinstance(v, bool) and k not in self._BOOL_PARAMETERS:
                v = int(v)
            try:
                settings.set_parameter(k, v)
            except ValueError as e:
                msg = f"cuOpt rejected the solver option {k}={v!r}: {e}"
                raise ValueError(msg) from e
```

`log_fn` wins over a user-supplied `log_file`, matching `Highs._set_solver_params`
(`:1779-1784`) — the one repo precedent, and cuOpt's parameter is spelled identically.

#### 2.1.8 Module-level helpers

```python
def _run_cuopt_with_keyboard_interrupt(solve: Callable[[], Any]) -> Any:
    """
    Run a cuOpt solve while keeping Ctrl-C responsive.

    cuOpt exposes no cancel API and defers SIGINT for the entire duration of
    the C++ solve, so the call runs in a worker thread while the main thread
    waits. Note that the GPU work continues in the background until the
    process exits; ``time_limit`` is the only hard bound cuOpt offers.
    """
    finished = threading.Event()
    out: list[Any] = []
    run_error: BaseException | None = None

    def _target() -> None:
        nonlocal run_error
        try:
            out.append(solve())
        except BaseException as exc:  # pragma: no cover
            run_error = exc
        finally:
            finished.set()

    threading.Thread(target=_target, name="linopy-cuopt-solve", daemon=True).start()
    while not finished.wait(0.1):
        pass
    if run_error is not None:
        raise run_error
    return out[0]


@functools.cache
def _cuda_device_available() -> bool:
    """
    True if a usable CUDA device is visible.

    ``import cuopt`` succeeds on hosts without a GPU by design (its ``__init__``
    defers every device-touching import so a remote solve can be configured),
    so an import probe alone would advertise cuOpt on machines where every
    solve fails with a remote-execution error.
    """
    try:
        from cuda.bindings import runtime

        err, count = runtime.cudaGetDeviceCount()
    except Exception:
        return False
    if err or not count:
        logger.warning(
            "cuOpt is installed but no CUDA device is visible; cuOpt is not "
            "available. It requires an NVIDIA GPU of compute capability 7.0 or "
            "higher and driver 525.60.13 or newer."
        )
        return False
    return True
```

`cuda-bindings` is part of cuOpt's own dependency tree (`cuda-bindings-12.9.7` in the Q13
clean-venv install list), so it is present whenever `cuopt` is. The coding agent **must paste** the
verified return convention of `cudaGetDeviceCount()`; the `except Exception` fallback keeps the
probe fail-safe either way. Placement: next to `_has_module` (`:307`), where the other probes live.

### 2.2 `linopy/io.py`  [≈ +8]

* `:33` `TYPE_CHECKING`: add `from cuopt.linear_programming import DataModel as cuoptDataModel`
  (mirrors `from cupdlpx import Model as cupdlpxModel`).
* after `to_cupdlpx` (`:792-795`):

```python
def to_cuopt(m: Model) -> cuoptDataModel:
    """Build the cuopt DataModel for `m`."""
    solver = solvers.cuOpt.from_model(m, io_api="direct")
    return solver.solver_model
```

### 2.3 `linopy/model.py`  [≈ +2]

* `:73` import list: add `to_cuopt,` (alphabetical, before `to_cupdlpx`).
* `:2517`: add `to_cuopt = to_cuopt` above `to_cupdlpx = to_cupdlpx`.

### 2.4 `linopy/solver_capabilities.py`  [±0 — verification only]

No edit. `SOLVER_REGISTRY` is a `_LazyRegistry` deriving `SolverInfo` from `SolverName` + the
class's features; `SolverName.cuOpt.name == "cuOpt" == cuOpt.__name__` satisfies its
`getattr(solvers, ...)` resolution. Proved by `test_capability_shim_round_trips`
(`test_solvers.py:482`, already parametrized over every `SolverName`) plus one explicit registry
assertion in `test_cuopt.py`. Editing the shim would be a drive-by (I4) — see charter §5 row 7.

### 2.5 `linopy/constants.py`  [±0]

No edit. Every cuOpt status maps onto an existing `TerminationCondition` (§4).

### 2.6 `test/conftest.py`  [±0 — verification only]

No edit. The auto-skip at `:53-79` is generic on `SolverFeature.GPU_ONLY`, so declaring `GPU_ONLY`
auto-skips and auto-marks every `params`-parametrized cuOpt test. Proof: `pytest --collect-only -q -k cuopt`
before/after `--run-gpu`, pasted in the WP report.

### 2.7 `test/test_optimization.py`  [≈ +6 / −0]

Three cuPDLPx-analogue touch points (charter cites `:562`, `:685`, `:1110`):

1. `:562` `time_limit_option`: add `"cuopt": {"time_limit": 1},`.
2. `:685` `test_infeasible_model`: **no cuOpt skip needed** — the cuPDLPx skip exists because
   cuPDLPx hangs on unbounded problems; cuOpt returns `PrimalInfeasible` on this model
   (exploration item 1). Deliberately unchanged; noted in the WP report so Phase 6 sees it was a
   decision, not an omission.
3. `:1110` `test_basis_and_warmstart`: extend the skip to cuOpt
   (`if solver in ("cupdlpx", "cuopt")`, reason naming both).

No other change: cuOpt joins `params` as `("cuopt", "direct", False)` through `direct_solvers`, and
joins the MIP / QP / GPU-tolerance groups through its features.

### 2.8 `test/test_solvers.py`  [≈ +6]

Feature-matrix rows at `:449-466`: `(solvers.cuOpt, DIRECT_API, True)`,
`(GPU_ONLY, True)`, `(INTEGER_VARIABLES, True)`, `(SEMI_CONTINUOUS_VARIABLES, True)`,
`(SOS_CONSTRAINTS, False)`, and `(QUADRATIC_OBJECTIVE, True)` (added by WP-F).

### 2.9 New `test/test_cuopt.py`  [≈ +150] and `test/test_cuopt_interrupt.py`  [≈ +40]

§8.

### 2.10 `pyproject.toml`  [≈ +6]

New extra as a sibling of `solvers`, inserted after the `solvers = [...]` block (`:113`) — §5i.

### 2.11 Docs  [≈ +110]

§9: `doc/gpu-acceleration.rst`, `doc/prerequisites.rst`, `doc/contributing.rst`, `doc/index.rst`,
`doc/api.rst`, `doc/release_notes.rst`, `README.md`, `examples/using-solvers.ipynb` (markdown cell
only).

**Total ≈ +560 / −5 across 13 files** — 0.73× PR #516's +771, comfortably inside the §9 yardstick.

---

## 3. Feature-flag set

| Feature | Declared? | Justification |
|---|---|---|
| `DIRECT_API` | **yes** | `DataModel` + `Solve` is the entry point (§1). |
| `GPU_ACCELERATION` | **yes** | Solves on the GPU (Barrier/PDLP are device-resident). |
| `GPU_ONLY` | **yes** | No CPU-only mode reachable in-process; a GPU-less host cannot solve at all (exploration item 13). Also the switch that makes `conftest` auto-skip cuOpt without `--run-gpu` and that puts cuOpt into `gpu_solvers` → `GPU_SOL_TOL`. |
| `SOLUTION_FILE_NOT_NEEDED` | **yes** | The direct path returns arrays; no solution file is required. Keeps `Model.solve` from allocating one (`model.py:2098`). |
| `INTEGER_VARIABLES` | **yes** | MILP verified (§4.3, exploration items 1/6: `set_variable_types(['I','I'])` → `cat=MIP Optimal`). |
| `SEMI_CONTINUOUS_VARIABLES` | **yes** | cuOpt genuinely supports `'S'` (exploration item 10: obj 0.0 vs 5.0 on a discriminating model) and `M.vtypes` already emits `'S'`, so the code cost is zero. Exploration is explicit that copying cuPDLPx's semi-continuous `NotImplementedError` would be a *false* claim. Backed by two new tests (§8). |
| `MIP_DUAL_BOUND_REPORT` | **yes** | `Solution.get_milp_stats()["solution_bound"]` is the documented best bound (`solution/solution.py:395-399`), negated back for `max`. `test_milp_model` (`:829-834`) asserts `report.dual_bound is not None` for anyone declaring it — that assertion **is** the acceptance criterion; if the pasted evidence shows `None`, the flag is dropped and the miss reported (§3.6), not worked around. |
| `QUADRATIC_OBJECTIVE` | **yes, in WP-F** | QP verified end-to-end against HiGHS (exploration item 12). Declared in the QP commit so the LP/MILP core is never green-by-omission. Auto-registers cuOpt in `QUADRATIC_SOLVERS` (`:4426`) — no manual edit (charter §4.5). |
| `LP_FILE_NAMES` | **no** | No file io_api (§1.2); the flag only governs name writing into LP files. |
| `READ_MODEL_FROM_FILE` | **no** | §1.2. This is the flag that would admit cuOpt to `file_io_solvers`. |
| `IIS_COMPUTATION` | **no** | No IIS/infeasibility-diagnosis API on `DataModel`/`Solution`. `test_infeasible_model` then correctly expects `NotImplementedError` from `compute_infeasibilities()`. |
| `SOS_CONSTRAINTS` | **no** | Genuinely absent; cuOpt's own parser rejects an `SOS` section and its test suite comments "SOS is explicitly out of scope" (exploration item 10). `_validate_model` then raises with the reformulation hint. |
| `INDICATOR_CONSTRAINTS` | **no** | Genuinely absent (`grep -rni indicator` finds nothing but an unrelated test list). |
| `SOLVER_ATTRIBUTE_ACCESS` | **no** | No per-entity attribute API. `test_solver_attribute_getter` (`:1153`) then expects `NotImplementedError`, which is what linopy raises for every non-Gurobi solver. |
| `supports_persistent_update` | **no** (default) | `DataModel` has no in-place update; `Problem.reset_solved_values()` rebuilds anyway (`problem.py:1650`). `_apply_locked` falls back to a rebuild. |
| `supports_sign_update` | **no** (default) | Follows from the above. |

---

## 4. Status-mapping table

Keyed on `(problem_category, termination status)`, never on the bare int — the same integer means
different things for LP and MIP (`solution/solution.py:224-237`; exploration item 1). Two dicts,
selected by `sol.get_problem_category() != ProblemCategory.LP` (so `MIP` **and** `IP` take the MILP
map — `ProblemCategory` has three members, `solver_wrapper.pyx:108-111`). Unmapped → `unknown`
(the `Highs._solve` default at `:1855`, not cuPDLPx's pass-the-raw-code).

| Category | cuOpt status (value, name) | `TerminationCondition` | `SolverStatus` | Basis |
|---|---|---|---|---|
| LP | `0` `NoTermination` | `internal_solver_error` | error | cuOpt's catch-all for malformed input, MIQP, zero-nnz matrices **and** the memory ceiling; empty primal, `obj=nan`. `error` + a logged `get_error_message()` is the honest report (exploration items 1, 5, 6, 9). |
| LP | `1` `Optimal` | `optimal` | ok | verified |
| LP | `2` `PrimalInfeasible` | `infeasible` | warning | verified (`x>=2 & x<=1`) |
| LP | `3` `DualInfeasible` | `unbounded` | warning | mapping by definition; **status never observed** in 7 unbounded variants (all returned `11`). Recorded as an untested row in `logs/45-known-issues.md`; no test asserts it. |
| LP | `4` `IterationLimit` | `iteration_limit` | ok | verified (`iteration_limit=1`) |
| LP | `5` `TimeLimit` | `time_limit` | ok | verified. Note the primal may be all-zero — linopy treats `time_limit` as `ok`, as it does for every other solver. |
| LP | `6` `NumericalError` | `internal_solver_error` | error | verified (non-PSD `Q`); `obj=nan` |
| LP | `7` `PrimalFeasible` | `suboptimal` | ok | verified (`first_primal_feasible=True`) — a usable but unproven point |
| LP | `11` `UnboundedOrInfeasible` | `infeasible_or_unbounded` | warning | verified; the *actual* status cuOpt returns for unbounded LPs |
| MIP/IP | `0` `NoTermination` | `internal_solver_error` | error | as LP `0` (MIQP lands here with `error_status=4`) |
| MIP/IP | `1` `Optimal` | `optimal` | ok | verified |
| MIP/IP | `2` `Infeasible` | `infeasible` | warning | verified (`2x==1`, x integer) |
| MIP/IP | `3` `Unbounded` | `unbounded` | warning | mapping by definition; **never observed** (see LP `3`) |
| MIP/IP | `5` `TimeLimit` | `time_limit` | ok | verified (subset-sum, market-split) |
| MIP/IP | `8` `FeasibleFound` | `suboptimal` | ok | verified (knapsack, `node_limit=1`) — feasible, optimality unproven |
| MIP/IP | `11` `UnboundedOrInfeasible` | `infeasible_or_unbounded` | warning | verified |
| both | anything else | `unknown` | unknown | defensive default |

`ErrorStatus` is **not** mapped to a `TerminationCondition`; it is read alongside and logged
(`logger.error(get_error_message())`) whenever it is not `Success`. `status.legacy_status` carries
`sol.get_termination_reason()` (the enum member name), mirroring
`status.legacy_status = h.modelStatusToString(...)` in `Highs`.

Note the trap recorded from exploration item 1 and honoured here: a limit *setting* never implies a
limit *status* (`iteration_limit=1` on a knapsack returned `Optimal`), so nothing in `_solve`
infers a condition from the options.

---

## 5. Design decisions on the hard findings

### 5a. Presolve + `maximize` negated duals → adopt M2 (always minimise)

**Position: adopt M2 unconditionally. `set_maximize(True)` is never called.** For a `max` model,
build with `-M.c` (and `-0.5·M.Q`), leave the sense at minimise, then negate the reported objective,
the duals and the MIP `solution_bound` on the linopy side. Duals otherwise pass through
**untransformed**.

Why:

* The bug is real, silent, and reachable from an ordinary linopy model through the natural
  translation: `max|Δdual| = 3.61` with status `Optimal` and a *correct* objective
  (`a_02f_linopy_reach.py`, n = 2, 4, 10, 40, 200 all NEGATED).
* M1 (`presolve` explicitly set) fixes the equality cells but **fails on ranged rows** — rejected
  by measurement, not preference.
* `method=DualSimplex` "fixes" it by leaving the GPU. Rejected.
* M2 is verified **21/21 against live HiGHS** (`a_02i_m2_full.py`: 5/21 failing without it, 0/21
  with it) and it avoids the buggy branch *by construction*, which matters because the root cause is
  an explicit exploration Unknown — a narrower workaround could be defeated by a path nobody
  constructed.
* Cost: 4 lines (one `sign` scalar used in four places).

Two traps this decision creates, both handled explicitly:

* **Do not copy cuPDLPx's `if ModelSense == MAXIMIZE: dual = -dual` (`:4385-4386`).** cuOpt's dual
  convention already matches HiGHS, which *is* linopy's convention (`Highs._solve` passes
  `row_dual` through unadjusted, `:1871-1879`). Under M2 the negation is against the *sign we
  applied ourselves*, not a convention fix — the code comment must say exactly that, or the next
  reader will "simplify" it away.
* Everything derived from the objective must carry the same sign: `get_primal_objective()` **and**
  `get_milp_stats()["solution_bound"]`. `mip_gap` is relative and is not negated.

`maybe_adjust_objective_sign` is still called (as cuPDLPx does) even though it is a no-op for
`io_api="direct"`; it keeps the `_solve` shape identical to its siblings.

### 5b. Concurrent repeat-solve segfault → default `method=3` (Barrier), never override the user

**Position: linopy injects `method=3` (Barrier) as a default when the user has not set `method`;
a user who explicitly passes `method=0` gets a one-line warning and their choice.**

Why Barrier rather than cuOpt's Concurrent default or PDLP:

* Doing nothing is indefensible: `method=0` is deterministic SIGSEGV on the 2nd–3rd solve above
  ~1300 variables (12+ fresh processes, size sweep `1200 → 8 solves ok`, `1300 → exit 139`). linopy's
  own test models sit *below* the threshold, so the suite would stay green while a user's second
  `model.solve("cuopt")` kills the interpreter with no traceback.
* Barrier is stable on repeats (25/25 varied solves, device memory flat at 117 MiB) and GPU-resident,
  so `GPU_ACCELERATION` remains an honest claim — unlike `method=2` (DualSimplex), which the
  exploration notes throws the GPU away.
* Barrier is the **most accurate** of the four: dual agreement with HiGHS 1.73e-09 / 2.47e-09 versus
  PDLP's 2.60e-06 / 4.71e-05 (§2c). This is what makes a `rtol=1e-6` differential test defensible
  instead of a tolerance-widening argument (I12). PDLP at default tolerances disagrees with HiGHS in
  **every** cell of the 144-cell sweep ("PDLP DIFFERENT 17+17") and moved a 2000×1000 objective by
  5.52e-03 relative (§7) — making it the default would import that pressure into every test.
* cuOpt already forces Barrier for any quadratic objective regardless of `method` (exploration item
  12: all 8 method×crossover combinations report `solved_by=Barrier`), so LP, MILP relaxations and QP
  share one code path.
* PDLP also carries its own hazards: a segfault with `time_limit=1e-6` + `save_best_primal_so_far`,
  and non-termination on a badly-scaled LP that Barrier solved in seconds (exploration Surprise 3).

Documented escape hatch, with the measured numbers, in `doc/gpu-acceleration.rst`: `method=1` (PDLP)
for very large sparse LPs (the only method exercised at 10⁶–2×10⁷ variables, peak 12.9 GiB on the
T4), `method=0` not recommended until the upstream crash is fixed. Regression test: §8, V1-8
(3 sequential solves of a 2000-variable LP **in a subprocess**, so a regression is a red test rather
than a dead pytest session).

### 5c. `is_available()` — GPU-gated. §4.6 constraint 3 wins over the cuPDLPx precedent

**Position: `is_available()` = `_has_module("cuopt") and _cuda_device_available()`.** This is a
deliberate divergence from `cuPDLPx.is_available` (`:4188-4189`) and from `_AvailableSolvers`'
docstring ("Membership means the solver's Python package or binary is importable", `:4464-4470`).

Exact behaviour on a **GPU-less machine with `cuopt-cu12` installed**:

* `linopy.available_solvers` **omits** `"cuopt"`; `linopy.quadratic_solvers` and
  `linopy.licensed_solvers` likewise.
* One `logger.warning` on the first probe: *"cuOpt is installed but no CUDA device is visible; cuOpt
  is not available. It requires an NVIDIA GPU of compute capability 7.0 or higher and driver
  525.60.13 or newer."*
* `model.solve("cuopt")` raises `AssertionError: Solver cuopt not installed` from
  `Model.solve`'s existing membership assertion (`model.py:2079`) — the generic path every
  uninstalled solver takes. `Solver.from_name("cuopt", m)` raises the base-class
  `ImportError("Solver package for 'cuopt' is not installed…")` (`:475-480`).
* **Never** a bare CUDA traceback, and never cuOpt's own
  `InputValidationError: A CPU-memory problem requires remote execution. Set CUOPT_REMOTE_HOST…`,
  which is the message an import-only probe would deliver (exploration item 13) and which is exactly
  what §4.6 constraint 3 rules out.

Why I resolve the fork this way:

* §4.6 is recorded as **binding human-lead direction** ("a linopy user must be able to ask for cuOpt
  support and get it"; constraint 3: `available_solvers` "must simply omit cuOpt"). I5 is a
  tie-breaking heuristic for *style*; it does not outrank an explicit requirement.
* The precedent it breaks is weaker than it looks: for every other solver, "package importable" and
  "solver usable" coincide closely enough that the docstring's caveat is about *licenses*. cuOpt is
  the first solver whose package deliberately imports on hosts where it cannot run at all —
  `cuopt/__init__.py`'s own comment says so.
* The probe is cheap and side-effect-light (a device *count*, not a context), `functools.cache`d, and
  only ever runs when the `cuopt` package is actually installed. It does not import `cuopt`.
* Free bonus: it makes every non-parametrized cuOpt test (`skipif "cuopt" not in licensed_solvers`)
  self-gating on machines without a GPU, and keeps ReadTheDocs (no GPU, no cuOpt) unaffected.

Accepted downside, stated for the handoff: the *message* a GPU-less user sees says "not installed"
when the package is installed. Fixing that properly means touching the shared `__post_init__`
message — out of scope (I4) and a behaviour change for every solver. The `logger.warning` is the
mitigation; the docs say it explicitly. Logged as a D-entry / ESC candidate (E1 fork).

### 5d. QP — `0.5 · M.Q`, full-symmetric CSR; MIQP rejected up front; non-PSD surfaces as an error

* **Convention:** `Q_cuopt = 0.5 * model.matrices.Q`, kept **full symmetric**, converted to CSR.
  cuOpt minimises `cᵀx + xᵀQx` with `Q` internally symmetrised to `Q + Qᵀ`
  (`data_model/data_model.py:287-288`), and `linopy`'s `M.Q` *is* the Hessian `H` of `½xᵀHx`
  (`expressions.py:2945-2970` docstring; it is why `Highs` passes `triu(M.Q)` to `passHessian`,
  `:1716-1720`). Verified live against HiGHS: `0.5*M.Q` → `-4.28571429`, matching HiGHS exactly,
  while the naive `M.Q` returns `-2.14285714` with status `Optimal` and half the solution vector.
  Three lines in `_build_solver_model`:

  ```python
        Q = M.Q
        if Q is not None:
            Qc = (sign * 0.5 * Q).tocsr()
            dm.set_quadratic_objective_matrix(Qc.data, Qc.indices, Qc.indptr)
  ```

  Triangular encodings that also satisfy `Q + Qᵀ = H` exist and were verified; full-symmetric is
  chosen because it is the form `M.Q` already has (no `triu`, no diagonal special case — the
  classic source of the factor-of-2 error).
* **`max` QP** takes the same `sign` as `c` (`Q → -Q`), verified 4/4 QP cells in `a_02i_out.txt`.
  Note QP never reaches the buggy presolve path (cuOpt forces Barrier), so M2 is not *required*
  here — but applying it uniformly keeps one code path and does no harm (verified).
* **MIQP: rejected before the build.** `NotImplementedError` for `model.type in ("MIQP", "IQP")`.
  cuOpt does not raise on MIQP — it returns `NoTermination`, `obj=0.0`, an **empty** solution array
  and `error_status=4` ("Presolve does not support optimization_problem with a quadratic
  objective"). A pre-check is the only honest guard, because the post-check is indistinguishable
  from "solver failed". Message names solvers that do support MIQP.
* **Non-PSD Q:** no special handling. cuOpt detects it and returns `NumericalError` → mapped to
  `internal_solver_error` (`SolverStatus.error`) with `get_error_message()` logged. It does **not**
  crash and does not return a bogus answer, so a linopy-side PSD test (an eigenvalue computation on
  every solve) would be pure cost. Documented as a limitation. Note the asymmetric-but-`Q+Qᵀ`-PSD
  case is accepted by cuOpt and is unreachable from linopy anyway (`M.Q` is symmetric).
* **Quadratic *constraints*** (`add_quadratic_constraint`) are out of scope — linopy has no model
  surface for them.
* **Sequencing** (charter §4.5): QP is WP-F, dispatched only after the LP/MILP core has merged and
  Gate 4's checks pass on it.

### 5e. Reduced costs — do not surface them

**Position: `Solution.get_reduced_cost()` is never called.**

* linopy has nowhere to put them: `grep -rn "reduced_cost\|ReducedCost" linopy/` returns nothing and
  `linopy.constants.Solution` (`constants.py:279-286`) carries only `primal`, `dual`, `objective`.
* They are also *wrong* on the default path for maximised models with `<=` rows: cuOpt returns
  `−c − Aᵀy` instead of `c − Aᵀy` (`+0.39999985` where HiGHS says `-1.6`), and PDLP returns a flat
  `0` for `max`/`>=`. The exploration could not reduce this to one transformation and explicitly
  declines to propose one.

Actions instead of an implementation: (i) a one-line comment in `_solve` next to the dual handling
saying reduced costs are deliberately not read and why; (ii) an entry in
`logs/45-known-issues.md` recording the upstream defect so nobody later "fixes" it by guessing a
sign; (iii) one sentence in `doc/gpu-acceleration.rst` under limitations. Adding a reduced-cost
surface to linopy would be scope creep (E4 territory) and is an explicit non-goal (§10).

### 5f. Solver options — coerce bools to int, wrap the rejection, `log_fn` wins

* **`bool → int` coercion** for every parameter except the two known bool-typed ones
  (`log_to_console`, `crossover`): `set_parameter("presolve", False)` raises
  `ValueError: Parameter presolve value False is not an integer`, while `log_to_console=False` is
  accepted. A user writing `presolve=False` (or `crossover=False`, or `log_to_console=True`) must
  get what they meant. The allowlist is explicit rather than derived, because cuOpt exposes parameter
  *names* (117, 109 unique) but not their types; `get_parameter()`'s behaviour on a never-set
  parameter was not verified, so type-sniffing would rest on an unknown.
* **Unknown names** raise eagerly at `set_parameter` with
  `ValueError: Invalid parameter. Please check documentation` — a message that does **not name the
  offending parameter**. Since names are case-sensitive snake_case and users arrive with
  Gurobi/cuPDLPx habits (`TimeLimit`), the exception is re-raised as
  `ValueError("cuOpt rejected the solver option time_limit=…: …")` with `from e`. Bad *values* are
  rejected later, inside `Solve()`, with messages that already name parameter and value — those pass
  through unwrapped (they are raised from `Solve`, not `set_parameter`).
* **`log_file` / `log_to_console`:** cuOpt has a native `log_file` parameter spelled *identically*
  to HiGHS's, so `Highs._set_solver_params` (`:1779-1784`) transfers verbatim: assign
  `solver_options["log_file"] = path_to_string(log_fn)` before the loop, so **`log_fn` wins** over a
  user-supplied `log_file` (the precedence the exploration flagged as a plan decision). No
  `dup2`/`redirect_stdout` machinery is needed — `contextlib.redirect_stdout` captures 0 chars, but
  `log_file` captures everything (1437 chars incl. banner and iteration log). `log_to_console` keeps
  cuOpt's default (`True`), matching every other linopy solver's console behaviour; documented as the
  way to silence it. Known semantics recorded in the docs: `log_file` **truncates** on a second
  solve, and an unwritable path raises a clear `RuntimeError` from `Solve()`.
* **No default `time_limit`.** Exploration Surprise 3 suggests considering one; I decline —
  no other linopy solver imposes a time limit, and a silent default would change the meaning of
  "optimal". The non-termination hazard is a `method=1` property and is documented instead.
* `save_best_primal_so_far` is never set by linopy (segfaults in combination with a tiny
  `time_limit`); it remains reachable by a user who asks for it, and the hazard is documented.

### 5g. Interruptibility — adopt the HiGHS worker-thread pattern (minus the cancel step)

**Position: adopt it.** `_run_cuopt_with_keyboard_interrupt(solve)` (§2.1.8).

* Measured: SIGINT during a bare `Solve()` is deferred **52.9 s** (raised inside `create_solution`,
  after the solve already finished); in a worker thread the latency is **0.00 s**, and the process
  exits cleanly (`exit=0`) even while the daemon thread is still inside cuOpt's C++ code.
* cuOpt installs no SIGINT handler and exposes **no cancel API** in-process
  (`grep -rni "cancel|abort|interrupt"` hits only the gRPC client and the routing module), so the
  helper is the HiGHS pattern with the `cancelSolve()` step removed — hence a *separate* helper, not
  a reuse of `_run_highs_with_keyboard_interrupt` (which also toggles HiGHS-specific
  `HandleKeyboardInterrupt` flags).
* The helper takes a **zero-argument callable**, not `(dm, settings)`. That keeps the seam testable
  without a GPU — `test/test_cuopt_interrupt.py` mirrors the existing
  `test/test_highs_keyboard_interrupt.py` with a dummy callable and runs in ordinary CPU CI.
* Its docstring states plainly that the GPU work continues until the process exits and that
  `time_limit` is the only hard bound cuOpt offers. Same sentence in the docs.
* Cost ≈ 25 lines. Declining it would leave a solver aimed at multi-million-variable models
  unresponsive to Ctrl-C for the full solve duration, in a file that already contains the fix
  pattern.

### 5h. Warm start — defer, and never touch the poisoning setter

**Position: `warmstart_fn` raises `NotImplementedError` (cuPDLPx precedent, `:4352-4355`); neither
`set_pdlp_warm_start_data` nor `set_initial_primal_solution` is ever called.**

* Usable PDLP warm start requires **three** non-default settings simultaneously — `method=1`,
  `pdlp_solver_mode=1 (Stable2)`, `presolve=0` — because the payload comes back in *presolved*
  coordinates and cannot be fed back into its own model otherwise
  (`Invalid PDLPWarmStart data…`), and because the default `Stable3` (and the docstring-endorsed
  `Fast1`) return `NoTermination` with `obj=nan`. Wiring that up would mean silently forcing PDLP,
  disabling presolve, and changing the method the user asked for — three surprises to buy an 80×
  iteration win on a re-solve linopy cannot currently express (there is no warm-start object in
  linopy's API; `warmstart_fn` is a *file* path, and the payload is 17 numpy arrays, not a file
  format).
* `DataModel.set_initial_primal_solution` on an LP raises `cudaErrorIllegalAddress` and **poisons the
  CUDA context for the rest of the process** — reproduced with all-default settings, correctly sized
  C-contiguous float64 input. One warm-start attempt would take down an entire pytest session or user
  script. This is a hard "never call", enforced by a code comment at the only plausible call site and
  by an entry in `logs/45-known-issues.md`.
* `basis_fn` is warn-and-ignore (cuPDLPx precedent): cuOpt has no simplex basis to write, and the
  PDLP payload is algorithm state, not a basis.
* Test: `test_basis_and_warmstart` skip extended to cuOpt (§2.7), reason naming the upstream
  constraint. Documented under limitations, with the three settings named so a motivated user can do
  it by hand.

### 5i. The pyproject extra — `cuopt`, one Linux-marked requirement

Agree with Phase 1 item 13's verified candidate, verbatim, including its placement and comment:

```toml
# NVIDIA cuOpt (GPU). Deliberately NOT part of `solvers`: it pulls ~34 packages
# and several GB of CUDA wheels. Linux-only wheels, no sdist, so the marker keeps
# macOS/Windows resolution unaffected. Requires a CUDA 12 driver (>= 525.60.13)
# and a GPU of compute capability >= 7.0.
cuopt = [
    "cuopt-cu12>=26.8; platform_system == 'Linux'",
]
```

* **Name `cuopt`**, not `gpu`: it matches the solver name, the `SolverName` value and the file's
  existing lower-case single-word extras (`oetc`, `remote`, `docs`, `dev`, `solvers`, `benchmarks`);
  `gpu` would promise a category the extra cannot keep (one solver, Linux only, CUDA 12 only).
  Charter §5 writes `<gpu-extra>` as a placeholder, not a name.
* **Placement**: a sibling immediately after the `solvers = [...]` block (`pyproject.toml:113`),
  reachable from **no** other extra — there is no `all` extra in this project, and `dev`/`docs` do
  not reference it. Verified: `pip install 'linopy[solvers]'` resolves 34 packages with
  **zero** cuopt/cudf/nvidia/cuda entries.
* **Marker**: `platform_system == 'Linux'` — the positive twin of the file's established
  `"cplex; platform_system != 'Darwin'"` (`:105`); it must be an equality because cuOpt has no
  Windows wheels either. No `python_version` marker: cuOpt ships cp311–cp314 and linopy already
  requires `>=3.11`. Verified cross-platform: `uv pip compile --python-platform windows|macos
  --extra cuopt` both resolve with cuOpt absent; `linux` resolves 66 packages including
  `cuopt-cu12==26.8.0`.
* **Lower bound, no upper bound**: `>=26.8` is the first release with cp311 wheels (§4.2). No
  ceiling, matching the file's style for every other solver (`coptpy!=7.2.1` is the only exclusion
  anywhere). The `cu13` variant is **not** offered — the driver requirement differs (§4.1) — but the
  `cu12`/`cu13` distinction and the driver floor are stated next to every install command (§9).
* **Not touched**: `[tool.uv]` (no index entry needed; the whole tree is on PyPI, and workspace
  index config does not propagate to downstream installs anyway), and `benchmarks` (pandas
  `==3.0.5` collides with cuDF's `<3.0.4a0` ceiling — do not run benchmarks in a cuOpt venv).
* **I9's end date**: once this lands, `uv sync --extra cuopt` becomes the *correct* command and is
  itself a test that the extra works. The WP report must say so explicitly, because every other
  agent is under instruction never to sync.
* Acceptance test = Q13 re-run: clean venv, `pip install '<repo>[cuopt]'` with **no** extra index and
  **no** resolver flags, then `python -c "import linopy; print(linopy.available_solvers)"` must print
  a list **containing `cuopt`** (Q13 printed `[]` only because the branch had no cuOpt class yet).
  Note this re-run needs a GPU-visible machine because of §5c.

---

## 6. Requirement-ID map

Charter §5 rows numbered top to bottom. `C-R18`+ are new requirements this plan implies.

| ID | §5 row / topic | Concrete work that satisfies it | WP |
|---|---|---|---|
| C-R1 | `solvers.py:148` `SolverFeature` | `features` frozenset on `class cuOpt` (§3); **no enum edit** — every needed member exists | A (+F for `QUADRATIC_OBJECTIVE`) |
| C-R2 | `:176` `TYPE_CHECKING` | `import cuopt` added to the block | A |
| C-R3 | `:304` `_LazyModule` | `cuopt = _LazyModule("cuopt")` + the never-import-`cuopt.routing` comment | A |
| C-R4 | `:346` `SolverName` | `cuOpt = "cuopt"` after `cuPDLPx` | A |
| C-R5 | `:4156` new solver class | `class cuOpt(Solver[None])`: `is_available`, `_license_probe`, `_run_file`, `_build_direct`, `_build_solver_model`, `_run_direct`, `_solve`, `_set_solver_params` (§2.1) | A (+F: Q) |
| C-R6 | `:4440` `_SOLVER_PROBE_ORDER` | `"cuopt"` after `"cupdlpx"` | A |
| C-R7 | `solver_capabilities.py` registry | **zero-diff proof**: `test_capability_shim_round_trips` (already parametrized over all `SolverName`) + explicit `SOLVER_REGISTRY["cuopt"].features` assertion in `test_cuopt.py`; pasted | D |
| C-R8 | `io.py:33`, `:792` | TYPE_CHECKING import + `to_cuopt(m)` | B |
| C-R9 | `model.py:73`, `:2517` | import + `to_cuopt = to_cuopt` binding | B |
| C-R10 | `constants.py:171` mapping | two `CONDITION_MAP`s in `_solve` keyed on `(category, status)` (§4); **no `constants.py` edit** | A |
| C-R11 | `test/conftest.py:57-79` | **zero-diff proof**: `GPU_ONLY` drives the generic auto-skip; pasted `--collect-only` with and without `--run-gpu` | D |
| C-R12 | `test/test_optimization.py` branches | `time_limit_option["cuopt"]`; warmstart skip; documented *non*-skip for `test_infeasible_model` | D |
| C-R13 | `test/test_solvers.py` | 5–6 feature-matrix rows | D (+F) |
| C-R14 | `pyproject.toml` extra | `cuopt` extra (§5i) + clean-venv acceptance test | C |
| C-R15 | `doc/gpu-acceleration.rst` | new "cuOpt" section: install, hardware/driver, usage, supported classes, limitations, method guidance (§9) | E |
| C-R16 | `prerequisites`/`contributing`/`index`/`api`/`release_notes`/`README` | §9 table | E |
| C-R17 | `:4426` `QUADRATIC_SOLVERS` (read-only) | QP build path + MIQP guard + QP tests; flag declared, list untouched | F |
| **C-R18** | Honest degradation without a GPU (§4.6 c3) | `_cuda_device_available()` + `is_available()` gate + warning; documented; test with `CUDA_VISIBLE_DEVICES=''` in a subprocess | A, D |
| **C-R19** | M2 always-minimise (dual-sign correctness) | `sign` in `_build_solver_model` / `_solve` for `c`, `Q`, objective, duals, `solution_bound` + code comment | A (+F: Q) |
| **C-R20** | Repeat-solve crash mitigation | `_DEFAULT_METHOD = 3`, `method=0` warning, subprocess repeat-solve test, docs | A, D, E |
| **C-R21** | Models with no constraint nonzeros | padded free row + dual slice; test (LP and QP) | A, D (+F) |
| **C-R22** | Ctrl-C responsiveness | `_run_cuopt_with_keyboard_interrupt` + CPU unit test | A, D |
| **C-R23** | MIQP rejected honestly | `NotImplementedError` in `_build_direct`; test | F |
| **C-R24** | Option handling | bool→int coercion, wrapped `ValueError`, `log_fn` precedence; tests | A, D |
| **C-R25** | Warm start / basis / solution file refusals | `NotImplementedError` / warning + skip + docs | A, D, E |
| **C-R26** | Semi-continuous support | `'S'` passthrough, feature flag, 2 tests, docs | A, D, E |
| **C-R27** | MIP report (gap + dual bound) | `get_milp_stats()` → `SolverReport`, sign-corrected; covered by `test_milp_model` | A, D |

No §5 row is unmapped, and no ID is reinterpreted.

---

## 7. Work-package breakdown

Rules honoured: `solvers.py` core first and **alone**; then disjoint parallel packages; QP last and
alone; packaging first-class and early; no two concurrent packages share a file.

| WP | Files (exclusive) | IDs | Runs | Commit |
|---|---|---|---|---|
| **A — solvers.py core (LP + MILP)** | `linopy/solvers.py` | C-R1..C-R6, C-R10, C-R18..C-R22, C-R24..C-R27 | first, alone | 1 |
| **B — io + model binding** | `linopy/io.py`, `linopy/model.py` | C-R8, C-R9 | after A merges; parallel with C, D, E | 1 |
| **C — packaging** | `pyproject.toml` | C-R14 | parallel with A (disjoint file); re-verified after A merges | 3 |
| **D — tests** | `test/test_optimization.py`, `test/test_solvers.py`, new `test/test_cuopt.py`, new `test/test_cuopt_interrupt.py` | C-R7, C-R11..C-R13, and the test halves of C-R18..C-R27 | after A+B merge; parallel with E | 2 |
| **E — docs** | `doc/*.rst`, `README.md`, `examples/using-solvers.ipynb` | C-R15, C-R16 | after A merges; parallel with C, D | 3 + 4 |
| **F — QP** | `linopy/solvers.py`, `test/test_cuopt.py`, `test/test_solvers.py`, `doc/gpu-acceleration.rst` | C-R17, C-R23, plus QP rows of C-R19/C-R21 | last, alone (Gate 4 green on LP/MILP first) | 5 |

File-collision check: A and F both own `solvers.py` → strictly sequential (F is last anyway).
D and F both own `test_cuopt.py`/`test_solvers.py`, E and F both own `gpu-acceleration.rst` → F runs
after both merge. B, C, D, E are pairwise disjoint.

### Acceptance criteria

**WP-A** (paste all):
1. `ruff check .` and `mypy` clean.
2. `python -c "import linopy; print(linopy.available_solvers)"` lists `cuopt`.
3. `linopy.solvers.cuOpt.supported_features()` printed and equal to §3's LP/MILP set.
4. A `min` LP and a `max` LP solved through `Solver.from_name("cuopt", m, io_api="direct")`, with
   objective **and duals** compared live against `model.solve("highs")` in the same process
   (thresholds §8).
5. A MILP solved; `result.report.dual_bound` and `.mip_gap` printed non-`None`.
6. `model.solve("cuopt")` **with no `io_api`** succeeds and emits the fallback warning.
7. The `CUDA_VISIBLE_DEVICES=''` subprocess shows `available_solvers` without `cuopt` plus the
   warning line.
8. 3 sequential solves of a 2000-variable LP in one process: exit code 0.
9. `set_parameter` rejection path: `model.solve("cuopt", TimeLimit=1)` raises the wrapped
   `ValueError` naming `TimeLimit`; `presolve=False` is accepted.

**WP-B**: `m.to_cuopt()` returns a `DataModel` whose `get_objective_coefficients()` /
`get_constraint_matrix_values()` match `m.matrices` (sign-flipped for `max`); `ruff`/`mypy` clean.

**WP-C**: Q13 re-run pasted — clean venv, no extra index, no resolver flags, `import linopy;
available_solvers` **contains** `cuopt`; `pip install 'linopy[solvers]' --dry-run --report` shows no
cuda/cuopt package; `uv pip compile --python-platform windows|macos --extra cuopt` both exit 0
without `cuopt-cu12`.

**WP-D**: `pytest -q` (no GPU flag) shows **no** new failures vs `logs/05-baseline-pytest.txt` and
cuOpt tests **skipped**; `pytest --run-gpu -q` green with the cuOpt test count and the
`-k cuopt --run-gpu -v` listing pasted so it is provable the GPU tests *ran* rather than skipped;
`SOLVER_REGISTRY["cuopt"]` assertion passing.

**WP-E**: docs built in `/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs`
(with `PATH` prefixed per D3) — **no new warning** vs `logs/05-baseline-docs.txt`; every §9 row
present; the mechanical PR-#516 gap check run and every difference justified.

**WP-F**: QP differential vs HiGHS on `quadratic_model`, `quadratic_model_cross_terms` and the
3-variable `M.Q` model from exploration item 12 (thresholds §8); `test_quadratic_model_wo_constraint`
green (padding); MIQP raises `NotImplementedError`; `quadratic_solvers` contains `cuopt`;
`pytest --run-gpu` green overall.

### Commits (§6.4a shape, ~6, plain imperative, no prefix)

1. `Implement interface to cuOpt solver and required IO` — WP-A + WP-B (`solvers.py`, `io.py`,
   `model.py`).
2. `Update tests` — WP-D.
3. `Add to docs & add package dependency` — WP-E (docs) + WP-C (`pyproject.toml`).
4. `Add comment for release notes` — `doc/release_notes.rst`.
5. `Add support for quadratic objectives` — WP-F.
6. Reserved for the targeted Phase 5/6 follow-up fix (#516's own arc ends with two such commits). If
   review finds nothing, five substantive commits is the honest count — do not manufacture a sixth.

No merge commits; each WP branch rebased onto `feat/cuopt-solver` and fast-forwarded.

---

## 8. Test plan

Tier per charter §6.3a. Every V1 row calls HiGHS **live, in-process, on the identical model** — no
baked-in expected values anywhere.

### Locations

* `test/test_cuopt.py` **(new)** — all cuOpt-specific V1 work. Module gate:

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

  This is the intended use of the `LINOPY_RUN_GPU_TESTS` env var that `conftest.py:47-50` already
  sets and that nothing currently reads — the `conftest` auto-skip only covers tests parametrized on
  `solver`, which the differential tests are not.
* `test/test_cuopt_interrupt.py` **(new, CPU, ungated)** — mirrors
  `test/test_highs_keyboard_interrupt.py`.
* `test/test_optimization.py`, `test/test_solvers.py` — the three/six existing-file touch points
  (§2.7, §2.8).
* Semi-continuous cuOpt tests live in `test_cuopt.py` (single gating block) rather than in
  `test_semi_continuous.py`, which has no GPU gating mechanism. Recorded so Phase 6 sees the reason.

### V0 — structural (every WP)

`ruff check .`; `mypy`; `"cuopt" in available_solvers`; `SOLVER_REGISTRY["cuopt"].features` equals
the declared set; `SolverName.cuOpt.name == cuOpt.__name__`; `conftest.py` unchanged;
`git status --porcelain` clean.

### V1 — differential vs live HiGHS (`test_cuopt.py`)

| # | Test | What it proves | Threshold |
|---|---|---|---|
| V1-1 | **Six-cell sign matrix**: `sense ∈ {min,max}` × `row ∈ {<=,>=,==}` on the exploration model (`A=[[1,2,1],[3,1,1]]`, `b=[4,6]`, `0<=x<=10`, `c` per cell so `x*=(1.6,1.2,0)` with both rows binding and a unique dual). Two deep copies of one model, one solved with cuopt, one with highs; compare `solution`, `dual`, `objective.value`. | dual + primal sign convention, all six cells | duals & primal `rtol=1e-6, atol=1e-7`; objective `rtol=1e-6` |
| V1-2 | **Presolve-solvable `max` model** — 2-variable square equality system (`x+2y=4`, `3x+y=6`, `max x+y`) and an `n=10` variant. This is the case the six-cell matrix **misses**: without M2 the duals come back negated with status `Optimal` and a correct objective (`max|Δ| = 3.61`). | that M2 actually works on the branch that motivated it | as V1-1 |
| V1-3 | **Presolve branch still reached**: build a raw `DataModel` for the V1-2 model with `set_maximize(True)` and assert `sol.get_solved_by() == SolverMethod.Unset`. Asserts a *coverage precondition*, not a correctness claim; its docstring says that if it goes red, cuOpt changed routing and V1-2's coverage must be re-established (never that M2 should be removed). | V1-2 keeps exercising the risky path | exact enum equality |
| V1-4 | **MILP differential**: `milp_model`-shaped model, objective vs HiGHS; assert integer variables are integral within `mip_integrality_tolerance`; assert `model.dual` is empty (cuOpt refuses duals for MIP) and that no exception escaped. | integrality + the MIP dual guard | objective `rtol=1e-6`; integrality `atol=1e-5` |
| V1-5 | **Status mapping**: infeasible LP → `infeasible`; unbounded LP → `infeasible_or_unbounded`; `time_limit=1e-6` on a 400×300 LP → `time_limit`; `iteration_limit=1` → `iteration_limit`; malformed-input path (zero-nnz model *without* the pad, via a raw `DataModel`) → `internal_solver_error`. | §4 table rows that are reachable | exact string equality on `termination_condition` |
| V1-6 | **No-constraint models**: an LP with no constraints and `quadratic_model` with `con0` removed. | the padded free row (C-R21), and that the pad's dual is dropped (`len(model.dual)` per constraint set unchanged) | objective `rtol=1e-6` vs HiGHS |
| V1-7 | **Semi-continuous**: `x` semi-continuous in `[1,10]`, `x <= 0.5`, `max x` → 0; and `x <= 5` → 5 (the HiGHS tests' models). | `'S'` passthrough | `atol=1e-6` (as the HiGHS tests) |
| V1-8 | **Repeat solves** in a **subprocess** (`sys.executable -c ...`): 3 sequential `model.solve("cuopt")` of a 2000-variable / 1000-row LP under linopy's defaults; assert `returncode == 0` and 3 `optimal` lines. A subprocess so a SIGSEGV regression is a failed test, not a dead pytest session. | C-R20 (n=2000 is above the measured 1300 boundary) | exit code 0 |
| V1-9 | **Options**: unknown name → wrapped `ValueError` naming the parameter; `presolve=False` accepted (bool→int); `log_fn` produces a non-empty file containing `cuOpt version`; a user `log_file` is overridden by `log_fn`. | C-R24 | file size > 0 |
| V1-10 | **Refusals**: `warmstart_fn` → `NotImplementedError`; `solution_fn` → `NotImplementedError`; `basis_fn` → warning only; SOS model → `ValueError` (reformulation hint); indicator model → `ValueError`. | C-R25 and the honest-guard DoD line | exception types/messages |
| V1-11 | **No-GPU degradation** in a subprocess with `CUDA_VISIBLE_DEVICES=''`: `"cuopt" not in linopy.available_solvers`. | C-R18 | exact |
| V1-12 | **QP (WP-F)**: `quadratic_model`, `quadratic_model_cross_terms`, and the 3-variable model from exploration item 12 (`M.c=[-3,-1,2]`, `M.Q=[[2,1,0],[1,4,0],[0,0,1]]`) — solution *and* objective vs live HiGHS, for `min` and `max`. The solution comparison is what catches the `0.5·M.Q` trap: the naive encoding halves the solution vector. | C-R17, C-R19 for Q | primal & duals `rtol=1e-6, atol=1e-7`; objective `rtol=1e-6` |
| V1-13 | **MIQP (WP-F)**: quadratic objective + one integer variable → `NotImplementedError` mentioning MIQP. | C-R23 | exception type |
| V1-14 | **Medium random LP** (n=2000, m=1000, seed fixed): objective vs live HiGHS under linopy's cuOpt defaults. | that the default method/tolerances are usable beyond toy sizes | objective `rtol=1e-4` (see below) |
| V1-15 | **Interrupt helper** (`test_cuopt_interrupt.py`, CPU): a dummy callable that blocks; `_thread.interrupt_main()` from a helper thread; assert `KeyboardInterrupt` is raised in the main thread within ~0.2 s and the worker still completes. | C-R22, and that the helper needs no GPU | latency < 1 s |

### V2 — linopy's own suite

`pytest --run-gpu`: cuOpt runs as `("cuopt", "direct", False)` through ~60 `params` tests, the MIP
group (`feasible_mip_solvers`), the QP group (`quadratic_solvers`, after WP-F) and
`gpu_solvers`/`GPU_SOL_TOL`. Plain `pytest` must show cuOpt **skipped** and no new failures vs
`logs/05-baseline-pytest.txt`. **No change to `CPU_SOL_TOL`/`GPU_SOL_TOL`** — the existing
`GPU_SOL_TOL = 2.5e-4` is inherited as-is; if a V2 test needs more than that, the miss is reported
verbatim (§3.6), not accommodated.

### Proposed numeric thresholds and their rationale

Handed to Gate 3 as numbers, with the measurement each rests on:

| Threshold | Value | Rationale |
|---|---|---|
| V1 duals, small models | `rtol=1e-6`, `atol=1e-7` | Barrier (linopy's default method, §5b) agrees with HiGHS to **1.73e-09 / 2.47e-09** max-abs across the sweeps; Concurrent to 8.18e-08. `atol=1e-7` covers even the Concurrent figure with headroom; `rtol=1e-6` on a dual of 0.4 is 4e-7 absolute, ~200× the Barrier residual. |
| V1 primal, small models | `rtol=1e-6`, `atol=1e-7` | `x=[1.6, 1.2]` reproduced to 16 digits (`1.5999999999999999`). |
| V1 objective, small models | `rtol=1e-6` | `-2.80000000` vs HiGHS `-2.7999999999999994`; QP `-3.00000000` vs `-2.9999999999999964`. |
| QP non-binding duals | `atol=1e-7` | measured `1.07e-10` / `-1.30e-10` where HiGHS returns exact `0` — an `rtol` alone cannot express this. |
| MILP integrality | `atol=1e-5` | equals cuOpt's own `mip_integrality_tolerance` default; asserting tighter would test the solver's tolerance, not linopy. |
| V1-14 medium random LP objective | `rtol=1e-4` | the **measured** disagreement between cuOpt's four methods on one 5000×2500 LP at default tolerances is 2.5e-5 relative (`12506.8226 … 12506.8538`); 1e-4 is 4× that. Chosen *up front* from that number — not arrived at by loosening a failing 1e-6. The reason a tighter number is not proposed here is recorded: exploration item 7 measured a **5.52e-03** objective shift from the default 1e-4 tolerance under PDLP, so any tighter frozen threshold at this size would have to come with explicitly tightened solver tolerances, which would then be testing a non-default configuration. |
| V1-8 repeat solves | n=2000, 3 solves, exit 0 | above the measured `n≈1300` crash boundary (`1200 → 8/8 ok`, `1300 → exit 139`). |
| V2 | `GPU_SOL_TOL = 2.5e-4` (existing) | unchanged repo constant; cuOpt inherits it via `GPU_ONLY`. |

Explicitly **not** proposed: any threshold on `lp_stats["dual_residual"]` (meaningless under QP), on
reduced costs (not surfaced), or on `LPTerminationStatus.DualInfeasible` (never observed — untested
row, recorded as a known gap rather than asserted).

---

## 9. Docs plan (charter §6.6b, every row)

| Surface | Content |
|---|---|
| `doc/gpu-acceleration.rst` | New `cuOpt` section, placed **before** the existing cuPDLPx section (it is the more capable solver). Contains: one-line description (NVIDIA cuOpt, concurrent GPU barrier/PDLP + CPU dual simplex); **install** = `pip install "linopy[cuopt]"` (uv equivalent below it), Linux-only, driver ≥ 525.60.13, compute capability ≥ 7.0, `cu12` vs `cu13` caveat, and the warning that the bare `cuopt` package on PyPI is a **different project**; a usage example (`m.solve("cuopt", io_api="direct")`); **supported**: LP, MILP, QP, semi-continuous variables, ranged-free equality/inequality rows, duals for LP/QP, MIP gap and dual bound; **unsupported**: SOS, indicator constraints, MIQP, warm start, basis files, solution files, reduced costs, file `io_api`s (with the note that `io_api=None` transparently falls back to the direct API); **notes**: `method` defaults to 3 (Barrier) in linopy with the reason (upstream repeat-solve crash under `method=0`) and the guidance to try `method=1` (PDLP) for very large LPs; `log_file`/`log_to_console`; Ctrl-C cancels the wait but the GPU work runs to completion; tolerances default to 1e-4 so objectives can differ from a simplex solver in the 4th–5th significant digit. Update the existing "Verifying Installation" snippet's prose to mention cuOpt, and the `pytest -k cupdlpx --run-gpu` example to `-k "cuopt or cupdlpx"`. |
| `doc/prerequisites.rst` | Under "GPU-accelerated solvers": add the cuOpt bullet with its link, and the one-command install `uv pip install "linopy[cuopt]"` **as the headline**, stating Linux-only + driver/CC requirement on the same lines. `pypi.nvidia.com` is **not mentioned** (the whole tree is on PyPI; mentioning an optional index invites `--index-strategy`, which §4.2 shows is never needed). |
| `doc/contributing.rst` | Extend the existing GPU-testing paragraph: name cuOpt alongside cuPDLPx, and add one sentence that cuOpt-specific tests live in `test/test_cuopt.py` and require `--run-gpu`. |
| `doc/index.rst` | Add `- \`cuOpt <https://docs.nvidia.com/cuopt/>\`__ (GPU-accelerated)` to the solver list. |
| `README.md` | Add `* [cuOpt](https://docs.nvidia.com/cuopt/)` to the solver list. |
| `doc/api.rst` | Add `solvers.cuOpt` to the `Implementations` autosummary (before `solvers.cuPDLPx`). **`to_cuopt` is deliberately not added**: `api.rst` lists none of the `to_*` bridges (`to_cupdlpx`, `to_highspy`, `to_gurobipy` are all absent; only `Model.to_file`, `Model.to_netcdf`, `io.read_netcdf` appear). Adding one would be an inconsistent drive-by (I4) — recorded here so the §6.6b row reads as justified, not skipped. |
| `doc/release_notes.rst` | One bullet under *Upcoming Version* → *Other*: added support for NVIDIA cuOpt (LP, MILP, QP) via `io_api="direct"`, installable with `linopy[cuopt]` on Linux with a CUDA 12 GPU. |
| `pyproject.toml` | The extra carries the 4-line comment in §5i (why it is separate, the marker, the driver/CC floor) — that is where extras are explained in this file. |
| `examples/using-solvers.ipynb` | **Decision: yes, include** — one markdown sentence, appending "and NVIDIA cuOpt" to the existing backend list in the intro cell. Markdown only, **no code cell**: the notebooks are executed during the docs build with no GPU, so any cuOpt code cell would break RTD. Logged as a D-entry. |
| `CLAUDE.md` | **Not touched.** PR #516 edited it (+9/−1); this repo's `CLAUDE.md` now delegates to `AGENTS.md` and says nothing solver-specific, so an edit would be a drive-by. This is the expected difference in the §6.6b mechanical gap check. |

Mechanical gap check (§6.6b) expectations: `CLAUDE.md`, `linopy/solver_capabilities.py` and
`test/conftest.py` appear in #516's file list and **not** in ours — all three justified above
(delegated file; zero-diff shim; generic gating). `test/test_cuopt.py` and
`test/test_cuopt_interrupt.py` are new files #516 has no analogue for.

---

## 10. Explicit non-goals

1. **File `io_api`s** (`lp`, `lp-polars`, `mps`) for cuOpt, and therefore
   `READ_MODEL_FROM_FILE` / `LP_FILE_NAMES` (§1.2). `io_api=None` falls back to the direct API with
   a warning; it does not read the file.
2. **Solution-file output** and **basis files**.
3. **Warm start** in any form (PDLP payload or initial point) — §5h.
4. **Reduced costs**, including any new linopy surface for them — §5e.
5. **Quadratic constraints** (`add_quadratic_constraint`) and **MIQP**.
6. **SOS / indicator constraints** for cuOpt (genuinely absent upstream; linopy raises).
7. **Persistent in-place updates** (`supports_persistent_update` stays `False`).
8. **MIP callbacks**, `BatchSolve`, `problem.Problem`, `cuopt.routing`, `cuopt.distance_engine`, and
   the gRPC/server client.
9. **Remote cuOpt** (`CUOPT_REMOTE_HOST`) — including using it as a fallback on GPU-less hosts.
10. **`cu13` wheels** and any attempt to auto-select `cu12`/`cu13`.
11. **Tuning cuOpt's numeric defaults** other than `method`: no default `time_limit`, no tightened
    tolerances, no `pdlp_solver_mode`, no `crossover`.
12. **Touching `CPU_SOL_TOL` / `GPU_SOL_TOL`** or any existing threshold.
13. **Fixing any pre-existing linopy defect** noticed on the way (I13 — record in
    `logs/45-known-issues.md`).
14. **Benchmarks** in the cuOpt venv (pandas pin collision, §4.4).
15. **Editing `linopy/solver_capabilities.py`, `linopy/constants.py`, `test/conftest.py`** — all
   three are satisfied by verification with a zero-line diff.

---

## 11. Risks specific to this plan

Ordered by my estimate of probability × cost.

1. **`_cuda_device_available()`'s exact API (§5c).** I specify `cuda.bindings.runtime.cudaGetDeviceCount()`
   without having run it (planners do not run code). If the return convention differs, the
   `except Exception → False` fallback would make cuOpt **silently unavailable on the T4** — a
   false-negative that reads exactly like "cuOpt isn't installed" and would quietly skip every GPU
   test. Mitigation: WP-A acceptance criterion 2 (`available_solvers` must *list* cuopt) fails loudly
   if this happens, and criterion 7 checks the negative case. This is the single most likely place
   this plan is wrong.
2. **Diverging from repo precedent on `is_available` (§5c).** A maintainer may reject a GPU probe in
   `is_available` on principle ("membership means importable"). If so the fix is mechanical (move the
   probe into `_license_probe` and accept that `available_solvers` lists cuOpt on GPU-less hosts) but
   it then breaks §4.6 constraint 3 — i.e. it is a real fork, and I have chosen the side that
   contradicts the code's own docstring.
3. **`method=3` (Barrier) as the default (§5b).** Barrier's scaling to very large sparse LPs was
   never measured; the only 10⁶–10⁷-variable evidence is for PDLP. A user with a huge model may hit
   memory pressure or `NoTermination` where cuOpt's own default would have coped. Mitigation:
   documented escape hatch, and V1-14 exercises n=2000 at least. Residual risk accepted because the
   alternative default is a deterministic SIGSEGV.
4. **`MIP_DUAL_BOUND_REPORT` declared on documentation rather than a paste.** `get_milp_stats()`'s
   `solution_bound` is documented but was never printed in Phase 1. If it is absent or `None` on a
   MILP solved to optimality, `test_milp_model:834` goes red. Mitigation: it is an explicit WP-A
   acceptance paste, and the disposition is "drop the flag and report", never "loosen the test".
5. **QP on linopy's own fixtures.** `quadratic_model` has a *singular* PSD Hessian (only `x`
   appears quadratically) with `y` unbounded above, and `quadratic_model_unbounded` is genuinely
   unbounded. Phase 1 only exercised strictly-convex QPs. Barrier may return `NumericalError`
   (→ `internal_solver_error`) where the test expects `optimal`, or a condition outside
   `{unbounded, unknown, infeasible_or_unbounded}`. This is WP-F's biggest unknown and the most
   likely source of an honest red test.
6. **The padded free row (§5d/C-R21).** `vstack` on a `csr_array`, dtype preservation of
   `indices`/`indptr`, and the `dual[:len(clabels)]` slice are all one-line details that are easy to
   get subtly wrong; a mistake would misalign **every** dual by one row rather than fail loudly.
   Mitigation: V1-6 compares duals against HiGHS on a model *with* constraints and on one without.
7. **V1-3 (presolve branch still reached) is brittle by design.** A cuOpt point release that changes
   presolve routing turns it red with nothing actually broken. Mitigation: its docstring says
   exactly what to do (re-establish V1-2's coverage; do not delete M2). I judge a loud, informative
   red better than silently losing coverage of the worst bug in this integration.
8. **`_license_probe` solves on the GPU at test-collection time.** `test_optimization.py` imports
   `licensed_solvers`, so a plain `pytest` run performs one cuOpt GPU solve. Cheap (~0.3 s) and
   harmless here, but it means the "CPU, no GPU" run is not literally GPU-free on this machine.
   Flagged rather than hidden.
9. **Bool-parameter allowlist (§5f)** is hand-maintained (`log_to_console`, `crossover`). A third
   bool-typed parameter would get coerced to `int` and rejected by `Solve()` with a clear message —
   annoying, not silent.
10. **Diff drift in WP-F.** QP touches `solvers.py` after review has already passed over it; the
    §6.4a check must re-run at Gate 6 (it does).
11. **`explicit_coordinate_names` warning vs `pytest.warns`.** cuOpt joins `params` only as
    `("cuopt", "direct", False)`, so the `True` variant never runs and the warning path is
    untested by V2. Covered instead by an explicit V1 assertion if WP-D has room; otherwise recorded
    as a coverage gap.
