# Plan 5 — NVIDIA cuOpt support in linopy

*Author: planner 5 (Opus 5, `claude-opus-5[1m]`) · Phase 2 · 2026-08-22*
*Sources of truth: charter `PROJECT_LEAD.md` §4/§5/§6, `logs/10-exploration.md`, current `master`
(`09c34dd`). No sibling plan was read.*

Line references are to current `master` unless stated otherwise.

---

## 1. Entry-point decision

**Decision: `cuopt.linear_programming.DataModel` + `solver.Solve` is the sole entry point.
`accepted_io_apis = frozenset({"direct"})`. The `io` submodule is *not* wired up in this change.**

### 1.1 Agreement with Q0's primary recommendation

I agree with Phase 1's primary recommendation without reservation. The evidence is one-sided
(`10-exploration.md` Q0):

* 13–16 lines of translation from `model.matrices`, versus 21 for `problem.Problem` (which
  *cannot express ranged rows at all*, `problem.py:1524`, `:1593-1594`).
* 0.012 s vs 0.86 s to build a 10⁵-variable model — a **70×** gap. linopy's GPU users are exactly
  the people with 10⁵–10⁷ variables; a per-variable Python API would eat the GPU's advantage in
  model construction alone.
* Full coverage: LP, MILP, QP, duals, options, log control, warm start — everything that looked
  `Problem`-only lives on `SolverSettings`, which is shared.
* `DataModel` is the closest analogue of what `cuPDLPx._build_solver_model` already does
  (`solvers.py:4246-4282`), so the diff reads like the file it lands in (I5).

### 1.2 Divergence from Q0's *secondary* recommendation (file `io_api`s)

Q0 recommends additionally declaring `accepted_io_apis ⊇ {"lp", "mps"}` via
`cuopt.linear_programming.io.Read`, calling it "a real, cheap capability win". **I diverge.** Three
reasons, in order of weight:

1. **It multiplies the highest-risk surface in the whole change.** The file path hands cuOpt a
   model whose sense comes from the file, i.e. `set_maximize(True)` — precisely the branch that
   returns **negated duals** (exploration §2a). Mitigation M2 would have to be re-implemented
   *after* `io.Read`, by reading `get_objective_coefficients()` / the quadratic values back out of
   the freshly-built `DataModel` and negating them in place. That is a second, differently-shaped
   implementation of the single most dangerous transformation in this integration, needing its own
   six-cell verification. The capability is worth far less than that risk.
2. **It requires an unverified surface.** Declaring `READ_MODEL_FROM_FILE` puts `cuopt` into
   `file_io_solvers` (`test_optimization.py:44`), and `test_set_files_and_keep_files`
   (`test_optimization.py:664-678`) then asserts a **solution file exists** for every non-direct
   `io_api`. cuOpt's `solution_file` parameter exists but was **never exercised** — exploration
   lists it under Unknowns ("I only exercised `log_file`"). I will not declare a feature whose
   acceptance test depends on behaviour nobody measured (I7).
3. **Cost/benefit and bulk.** The same declaration multiplies cuOpt's slice of the parametrized
   suite by `len(io_apis) × len(explicit_coordinate_names)` — every one of those cells is GPU work
   that only ever runs behind `--run-gpu`, so the failures land on users rather than on us. Meanwhile
   the file path is 70× slower than the direct path it would shadow.

The `io` findings are not wasted: they are recorded as the natural follow-up (non-goal N4, §10) and
the docs state plainly that file `io_api`s are not supported.

### 1.3 What `io_api` values actually do

`accepted_io_apis` is advisory metadata today (`grep` finds it declared at `solvers.py:466` and used
nowhere else in `linopy/`), so the real behaviour is set by `_build`/`_build_file`
(`solvers.py:697-791`). Two consequences the plan must handle, because `Model.solve` **never
defaults `io_api` to `"direct"`** (`model.py:2021-2136`):

* `model.solve("cuopt")` with no `io_api` reaches `_build_file`. On `cuPDLPx` that path writes an LP
  file and then raises `NotImplementedError` from `_run_file` (`solvers.py:4195-4225`) — i.e. the
  bare call is broken today for cuPDLPx. The charter's Definition of Done names
  `model.solve("cuopt")` explicitly, so I will not reproduce that wart.
* Therefore `cuOpt` overrides `_build_file` to *not* write a file and to build directly instead,
  warning only when the user asked for a file API on purpose:

```python
    def _build_file(self, **build_kwargs: Any) -> None:
        # cuOpt has no file interface in linopy (see doc/gpu-acceleration.rst); build the
        # model through the direct API rather than writing a problem file nobody reads.
        if self.io_api is not None:
            logger.warning(
                f"cuOpt does not support io_api='{self.io_api}'. "
                "Building the model through the direct API instead."
            )
        self._build_direct(
            explicit_coordinate_names=build_kwargs.get("explicit_coordinate_names", False)
        )
```

`_build_direct` sets `self.io_api = "direct"`, so `Solver.solve` dispatches to `_run_direct`
(`solvers.py:858`), `Model.solve`'s `problem_fn` bookkeeping sees `solver._problem_fn is None`, and
nothing is left on disk. `_run_file` is **not** overridden — the base message ("File-based API not
implemented for cuopt", `solvers.py:979-983`) is correct for the deprecated
`solve_problem_from_file` entry point.

---

## 2. Full file-by-file change set

Anchors are current-`master` line numbers. Estimated line counts are for the whole feature
(WP1 + WP6), for the §9 bulk yardstick.

### 2.1 `linopy/solvers.py` (~+340)

| Anchor | Change |
|---|---|
| `:34` | `from scipy.sparse import csr_matrix, tril, triu, vstack` — `csr_matrix`/`vstack` are needed by the zero-nnz padding helper (§5, C-R21). |
| `:148-164` `SolverFeature` | **No edit.** Every flag cuOpt needs already exists. |
| `:176-179` `TYPE_CHECKING` | add `import cuopt` (alphabetically before `cupdlpx`). |
| `:304` | `cuopt = _LazyModule("cuopt")` immediately above the `cupdlpx` line, with a two-line comment: only `cuopt.linear_programming` is ever touched — **never** `cuopt.routing`, which installs a global `sys.excepthook` writing `error_log.txt` into the cwd (`routing/vehicle_routing_wrapper.pyx:144-160`). Mirror the `cupdlpx` line exactly: **no** `# type: ignore` (cuopt ships no `py.typed`, so mypy already sees `Any`; `warn_unused_ignores = true` would flag a redundant one). |
| `:346-359` `SolverName` | append `cuOpt = "cuopt"` after `cuPDLPx = "cupdlpx"`. Vendor spelling; the member name must equal the class name for `solver_capabilities` to resolve it. |
| before `:4419` | new module-level helper `_run_cuopt_solve(data_model, settings)` (~35 lines) — worker-thread Ctrl-C handling plus no-GPU error translation (§5g, §5c). Placed immediately above `class cuOpt` so the whole feature reads as one block; docstring cross-references `_run_highs_with_keyboard_interrupt` (`:189`). |
| before `:4419` | new module-level helper `_pad_free_row(A, lower, upper, n_vars)` (~10 lines) — §5, C-R21. |
| before `:4419` | **`class cuOpt(Solver[None])`** (~250 lines) — full sketch in §2.8. |
| `:4426` `QUADRATIC_SOLVERS` | **no edit** — derived from the feature (charter §5 last row). |
| `:4443-4457` `_SOLVER_PROBE_ORDER` | insert `"cuopt"` immediately **after** `"cupdlpx"`, before `"pips"`. Rationale: keeps the GPU solvers together, and changes the default-solver choice for *nobody* who has any other solver installed (I4 spirit). |

### 2.2 `linopy/io.py` (~+10)

* `:31-36` `TYPE_CHECKING`: add `from cuopt.linear_programming import DataModel as cuoptDataModel`
  (mirrors `from cupdlpx import Model as cupdlpxModel`).
* after `:792` `to_cupdlpx`:

```python
def to_cuopt(m: Model) -> cuoptDataModel:
    """
    Build the cuopt DataModel for `m`.

    A maximisation model is handed to cuOpt in equivalent minimisation form (negated
    objective, ``set_maximize(False)``) — see :class:`linopy.solvers.cuOpt`.
    """
    solver = solvers.cuOpt.from_model(m, io_api="direct")
    return solver.solver_model
```

The docstring note is load-bearing: without it, a user inspecting `model.to_cuopt()` on a `max`
model would think the sense was dropped.

### 2.3 `linopy/model.py` (+2)

* `:68-79` import block: add `to_cuopt` (alphabetical: before `to_file`).
* `:2517` area: `to_cuopt = to_cuopt`, directly above `to_cupdlpx = to_cupdlpx`.

### 2.4 `linopy/constants.py` (+0)

**Verification only, zero-line diff.** Every cuOpt status maps onto an existing
`TerminationCondition` member (§4). Adding a member would change a public enum for one solver — out
of scope. The proof is the status table in §4 plus the V1 status tests.

### 2.5 `linopy/solver_capabilities.py` (+0)

**Verification only, zero-line diff** (charter §5 is explicit: editing the shim is a drive-by
change). `SOLVER_REGISTRY` is a `_LazyRegistry` deriving `SolverInfo` from `SolverName` + the class
(`:78-100`). Proof obligations:

* `test_capability_shim_round_trips` (`test_solvers.py:478`) is parametrized over **every**
  `SolverName` value, so it covers `"cuopt"` the moment the enum member and class exist.
* `test_solver_registry_iter_and_index` (`:485`) indexes every name.
* WP4 adds one explicit assertion pinning the declared feature set (§8).

### 2.6 `test/conftest.py` (+0)

**Verification only, zero-line diff.** `pytest_collection_modifyitems` (`:53-79`) auto-skips and
auto-marks any test parametrized on a `solver` param whose solver declares `GPU_ONLY` — cuPDLPx is
not named, the mechanism is generic. Declaring `GPU_ONLY` therefore gates:

* every `params`-parametrized test in `test_optimization.py`,
* every `licensed_solvers`-parametrized test in `test_solvers.py` (same `"solver"` param name).

`test/test_cuopt.py` (new) is gated with the documented `LINOPY_RUN_GPU_TESTS` env var that
`conftest.py:47-50` sets — that variable exists precisely "so test modules can check if GPU tests
are enabled". Evidence for C-R11: paste `pytest -k cuopt -q` (all skipped) and
`pytest --run-gpu -k cuopt -q` (all run).

### 2.7 Test / packaging / docs files

Covered in §8 (`test/test_cuopt.py` new, `test/test_optimization.py` 3 branches,
`test/test_solvers.py` ~6 lines), §2.9 (`pyproject.toml`) and §9 (docs).

### 2.8 The `cuOpt` class — concrete sketch

Method order mirrors `cuPDLPx` (`display_name`, `features`, `accepted_io_apis`, `is_available`,
`_license_probe`, `_build_file`, `_build_direct`, `_build_solver_model`, `_run_direct`, `_solve`,
`_set_solver_params`). WP6 adds only the marked lines.

```python
class cuOpt(Solver[None]):
    """
    Solver subclass for the NVIDIA cuOpt solver. cuOpt must be installed with
    working GPU support for usage; install it with ``pip install 'linopy[cuopt]'``
    (Linux only, NVIDIA driver >= 525.60.13, compute capability >= 7.0).

    The full list of solver options is documented at
    https://docs.nvidia.com/cuopt/user-guide/latest/linear-programming/settings.html.

    Some example options are:

    * time_limit : inf by default.
    * method : 3 (barrier) by default in linopy — see below.
    * log_to_console : True by default.
    * mip_relative_gap : 1e-4 by default.

    Parameter names are snake_case and case-sensitive, and the integer-valued ones
    (``method``, ``presolve``, ...) require ``1``/``0`` rather than ``True``/``False``.

    linopy defaults ``method`` to ``3`` (barrier) instead of cuOpt's own ``0``
    (concurrent), which crashes the process with SIGSEGV on the second or third
    solve in one interpreter. Pass ``method=1`` for PDLP on very large models.

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
            SolverFeature.MIP_DUAL_BOUND_REPORT,
            SolverFeature.QUADRATIC_OBJECTIVE,   # WP6 only
        }
    )
    accepted_io_apis: ClassVar[frozenset[str]] = frozenset({"direct"})

    # cuOpt's own default method (0, concurrent) segfaults on repeat solves above
    # ~1300 variables; 3 (barrier) is stable and has the tightest dual agreement.
    _DEFAULT_METHOD: ClassVar[int] = 3

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("cuopt")

    @classmethod
    def _license_probe(cls) -> None:
        # ``import cuopt`` succeeds without a GPU by design, so the runtime probe has
        # to solve something. One variable, one row, no console output.
        dm = cuopt.linear_programming.DataModel()
        dm.set_csr_constraint_matrix(
            np.array([1.0]), np.array([0], dtype=np.int32), np.array([0, 1], dtype=np.int32)
        )
        dm.set_constraint_lower_bounds(np.array([0.0]))
        dm.set_constraint_upper_bounds(np.array([1.0]))
        dm.set_objective_coefficients(np.array([1.0]))
        dm.set_variable_lower_bounds(np.array([0.0]))
        dm.set_variable_upper_bounds(np.array([1.0]))
        settings = cuopt.linear_programming.SolverSettings()
        settings.set_parameter("log_to_console", False)
        cuopt.linear_programming.Solve(dm, settings)

    def _build_file(self, **build_kwargs: Any) -> None:   # see §1.3
        ...

    def _build_direct(self, **kwargs: Any) -> None:
        model = self.model
        assert model is not None
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

    @staticmethod
    def _build_solver_model(model: Model) -> cuopt.linear_programming.DataModel:
        """Build a cuopt DataModel that mirrors the linopy `model`."""
        if model.is_quadratic and model.type != "QP":            # WP6
            raise NotImplementedError(                            # WP6
                "cuOpt does not support quadratic objectives together with integer "
                "variables (MIQP). Use a solver that supports them (gurobi, cplex)."
            )
        M = model.matrices
        A = None if M.A is None else M.A.tocsr()
        lower = np.where(np.logical_or(M.sense == ">", M.sense == "="), M.b, -np.inf)
        upper = np.where(np.logical_or(M.sense == "<", M.sense == "="), M.b, np.inf)
        if A is None or A.nnz == 0:
            # cuOpt reports NoTermination with an empty solution for a matrix without
            # a single nonzero, so a box-only model needs a filler row to be solvable.
            A, lower, upper = _pad_free_row(A, lower, upper, len(M.vlabels))

        # Never call set_maximize(True): cuOpt returns *negated* duals for maximised
        # models that presolve closes on its own (verified against HiGHS on 21 cases).
        # Hand it the equivalent minimisation instead and undo the negation in _solve.
        # NB this is the opposite of cuPDLPx, which negates duals for max (`:4385`).
        sign = -1.0 if model.objective.sense == "max" else 1.0

        dm = cuopt.linear_programming.DataModel()
        dm.set_csr_constraint_matrix(A.data, A.indices, A.indptr)
        dm.set_constraint_lower_bounds(lower)
        dm.set_constraint_upper_bounds(upper)
        dm.set_objective_coefficients(sign * M.c)
        dm.set_variable_lower_bounds(M.lb)
        dm.set_variable_upper_bounds(M.ub)
        if len(model.binaries) + len(model.integers):
            # cuOpt knows 'C'/'I'/'S' only and treats every other character as
            # continuous *silently*; 'B' is mapped to 'I' (linopy already pins the
            # [0, 1] bounds in M.lb / M.ub).
            dm.set_variable_types(np.where(M.vtypes == "B", "I", M.vtypes))
        Q = M.Q                                                   # WP6
        if Q is not None:                                         # WP6
            # cuOpt minimises c'x + x'Qx with the effective Hessian Q + Q', while
            # M.Q *is* the Hessian, so Q_cuopt = 0.5 * M.Q. Passing M.Q straight
            # through returns `Optimal` with a silently halved solution.
            Qc = csr_matrix(0.5 * sign * Q)
            dm.set_quadratic_objective_matrix(Qc.data, Qc.indices, Qc.indptr)
        return dm

    def _run_direct(self, solution_fn=None, log_fn=None, warmstart_fn=None,
                    basis_fn=None, env=None, **kw) -> Result:
        return self._solve(
            self.solver_model,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=self.io_api,
            sense=self.sense,
        )

    def _solve(self, dm, solution_fn=None, log_fn=None, warmstart_fn=None,
               basis_fn=None, io_api=None, sense=None) -> Result:
        """Solve a linear problem from a cuopt DataModel object. <numpydoc as cuPDLPx>"""
        lp = cuopt.linear_programming.solver.solver_wrapper
        # The LP and MILP status enums are IntEnums that share integer values, so a
        # single dict keyed on status would collide — one map per problem category.
        LP_CONDITION_MAP: dict[Any, TerminationCondition] = { ... }      # §4
        MILP_CONDITION_MAP: dict[Any, TerminationCondition] = { ... }    # §4

        if warmstart_fn is not None:
            raise NotImplementedError("Warmstarting not yet implemented for cuOpt.")
        if basis_fn is not None:
            logger.warning("Basis files are not supported by cuOpt. Ignoring.")
        if solution_fn is not None:
            raise NotImplementedError(
                "Solution file output not yet implemented for cuOpt."
            )

        settings = cuopt.linear_programming.SolverSettings()
        self._set_solver_params(settings, log_fn)
        sol = _run_cuopt_solve(dm, settings)

        is_lp = sol.get_problem_category() == lp.ProblemCategory.LP
        condition_map = LP_CONDITION_MAP if is_lp else MILP_CONDITION_MAP
        termination_condition = condition_map.get(
            sol.get_termination_status(), TerminationCondition.unknown
        )
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = sol.get_termination_reason()
        message = sol.get_error_message()
        if message:
            status.legacy_status = f"{status.legacy_status}: {message}"

        def get_solver_solution() -> Solution:
            objective = sol.get_primal_objective()
            primal = np.asarray(sol.get_primal_solution(), dtype=float)
            if primal.size != len(self._vlabels):
                raise ValueError(
                    f"cuOpt returned {primal.size} primal values for "
                    f"{len(self._vlabels)} variables."
                )
            if is_lp:
                # Duals need no sign transformation: cuOpt's convention matches
                # HiGHS, hence linopy's. The negation below only undoes the
                # minimisation form built in _build_solver_model.
                dual = np.asarray(sol.get_dual_solution(), dtype=float)
                dual = dual[: len(self._clabels)]   # drop the filler row, if any
            else:
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)
            if sense == "max":
                objective = -objective
                dual = -dual
            primal = _solution_from_labels(primal, self._vlabels, self._n_vars)
            if dual.size:
                dual = _solution_from_labels(dual, self._clabels, self._n_cons)
            return Solution(primal, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        runtime: float | None = None
        mip_gap: float | None = None
        dual_bound: float | None = None
        with contextlib.suppress(Exception):
            runtime = float(sol.get_solve_time())
        if not is_lp:
            with contextlib.suppress(Exception):
                stats = sol.get_milp_stats()
                mip_gap = float(stats["mip_gap"])
                dual_bound = float(stats["solution_bound"])
                if sense == "max":
                    dual_bound = -dual_bound
        self.io_api = io_api
        return self._make_result(
            status, solution, solver_model=dm,
            report=SolverReport(runtime=runtime, mip_gap=mip_gap, dual_bound=dual_bound),
        )

    def _set_solver_params(self, settings, log_fn: Path | None = None) -> None:
        """
        Set solver options on a cuopt SolverSettings object.

        For the list of available options, see
        https://docs.nvidia.com/cuopt/user-guide/latest/linear-programming/settings.html
        """
        if log_fn is not None:
            self.solver_options["log_file"] = path_to_string(log_fn)
            logger.info(f"Log file at {self.solver_options['log_file']}")
        options = {"method": self._DEFAULT_METHOD, **self.solver_options}
        for k, v in options.items():
            try:
                settings.set_parameter(k, v)
            except ValueError as e:
                raise ValueError(f"Invalid solver option '{k}' for cuOpt: {e}") from e
```

Notes on the sketch, all deliberate:

* `_solve` builds the `SolverSettings` rather than `_build_direct`, because `log_fn` reaches the
  solver only on the run call for the direct path. `to_cuopt(m)` therefore returns a pure
  `DataModel`, exactly parallel to `to_cupdlpx`.
* `{"method": _DEFAULT_METHOD, **self.solver_options}` means a user-supplied `method` always wins,
  and `log_fn` wins over a user-supplied `log_file` — the same precedence
  `Highs._set_solver_params` (`:1774-1784`) establishes.
* `maybe_adjust_objective_sign` is a no-op for `io_api="direct"` (`:389-401`); it is kept because
  both `Highs` and `cuPDLPx` call it in the same position, and its absence would read as an
  oversight.
* **Reduced costs are never read.** A short comment says why (`get_reduced_cost()` returns
  `−c − Aᵀy` for maximised models with `<=` rows on the default path, and linopy has no
  reduced-cost surface anyway) so nobody adds it later.
* `set_initial_primal_solution` / `set_initial_dual_solution` are never called; the comment next to
  the warmstart `NotImplementedError` says the LP variant raises `cudaErrorIllegalAddress` and
  poisons the CUDA context for the rest of the process.
* `set_objective_offset` is never called: linopy rejects objective constants outright
  (`objective.py:192-193`), so there is no offset to pass.

The two module helpers:

```python
def _pad_free_row(A, lower, upper, n_vars):
    """
    Append one unconstrained row carrying a single coefficient.

    cuOpt cannot solve a model whose constraint matrix has no nonzero: it returns
    ``NoTermination`` with an empty solution and no error. Box-constrained models
    (no constraints at all, or only all-zero rows) are legitimate in linopy, so a
    filler row with infinite bounds is added and its dual dropped afterwards.
    """
    row = csr_matrix(([1.0], ([0], [0])), shape=(1, max(n_vars, 1)))
    A = row if A is None else vstack([A, row], format="csr")
    return A, np.append(lower, -np.inf), np.append(upper, np.inf)


def _run_cuopt_solve(data_model: Any, settings: Any) -> Any:
    """
    Run ``cuopt.linear_programming.Solve`` so Ctrl-C stays responsive.

    cuOpt has no cancel API and defers SIGINT for the entire remaining solve, so the
    solve runs in a worker thread and the main thread waits, exactly as
    :func:`_run_highs_with_keyboard_interrupt` does for HiGHS — minus the cancel step.
    The GPU work continues in the background until the process exits; ``time_limit``
    is the only bound cuOpt honours. A solve on a host without a usable GPU is
    re-raised with an actionable message.
    """
    finished = threading.Event()
    result: Any = None
    run_error: BaseException | None = None

    def _target() -> None:
        nonlocal result, run_error
        try:
            result = cuopt.linear_programming.Solve(data_model, settings)
        except BaseException as exc:  # pragma: no cover
            run_error = exc
        finally:
            finished.set()

    thread = threading.Thread(target=_target, name="linopy-cuopt-solve", daemon=True)
    thread.start()
    while not finished.wait(0.1):
        pass
    if run_error is not None:
        if "remote execution" in str(run_error):
            raise RuntimeError(
                "cuOpt could not find a usable GPU on this machine. cuOpt requires an "
                "NVIDIA GPU with compute capability >= 7.0 and driver >= 525.60.13. "
                f"Original error: {run_error}"
            ) from run_error
        raise run_error
    return result
```

The `while not finished.wait(0.1)` loop needs no `try/except KeyboardInterrupt`: with no cancel API
there is nothing to call, so the interrupt simply propagates (verified: latency 0.00 s, clean
interpreter exit while the daemon thread is still inside cuOpt's C++ code).

### 2.9 `pyproject.toml` (+5)

Inserted as a sibling of `solvers` immediately after `:113`, reachable from no other extra (this
project has no `all` extra) — exactly the block Phase 1 Q13 installed from a clean venv:

```toml
# NVIDIA cuOpt (GPU). Deliberately NOT part of `solvers`: it pulls ~34 packages
# and several GB of CUDA wheels. Linux-only wheels, no sdist, so the marker keeps
# macOS/Windows resolution unaffected. Requires a CUDA 12 driver (>= 525.60.13)
# and a GPU of compute capability >= 7.0.
cuopt = [
    "cuopt-cu12>=26.8; platform_system == 'Linux'",
]
```

---

## 3. Feature-flag set

### Declared

| Feature | Justification |
|---|---|
| `DIRECT_API` | The only supported path (§1). |
| `GPU_ACCELERATION` | cuOpt is GPU-accelerated (PDLP and the cuDSS-backed barrier). |
| `GPU_ONLY` | No CPU-only mode is exposed by linopy, and this is what gates the whole test surface through `conftest.py:53-79`. Not declaring it would silently run GPU work in CPU CI. |
| `SOLUTION_FILE_NOT_NEEDED` | The direct path returns arrays; `Model.solve` then leaves `solution_fn=None` (`model.py:2097-2103`). |
| `INTEGER_VARIABLES` | MILP verified (§4.3 of the charter, exploration §1: `cat=MIP Optimal`). |
| `MIP_DUAL_BOUND_REPORT` | `Solution.get_milp_stats()` returns `mip_gap` and `solution_bound` (`solution/solution.py:213-222, 368+`), which is exactly what `SolverReport` wants; `test_milp_model` (`test_optimization.py:828-834`) asserts `report.dual_bound is not None` for solvers declaring it. |
| `QUADRATIC_OBJECTIVE` (**WP6 only**) | QP verified against HiGHS, including duals in all six sign cells and the `0.5·M.Q` convention. Declared only in the QP work package so intermediate states stay green (§7). |

`supports_persistent_update = False`, `supports_sign_update = False`: `DataModel` exposes no
incremental update API (exploration Q0: even `Problem.reset_solved_values()` throws the model away
and rebuilds). Both are inherited from `Solver`, so no line is written.

### Deliberately NOT declared

| Feature | Why not |
|---|---|
| `LP_FILE_NAMES` | No file IO (§1.2). |
| `READ_MODEL_FROM_FILE` | No file IO (§1.2). Declaring it also drags cuOpt into `file_io_solvers` and its solution-file assertions, which rest on an unverified cuOpt parameter. |
| `SEMI_CONTINUOUS_VARIABLES` | cuOpt *does* support `'S'` (exploration §10), but linopy's semi-continuous semantics were never checked against cuOpt's `mip_semi_continuous_big_m` (default `1e10`) end to end, and there is no generic semi-continuous solver test to inherit. Declaring an unverified capability violates I7's spirit. **Critically, I do not copy cuPDLPx's `NotImplementedError` guard** (`solvers.py:4249-4253`) — that would assert cuOpt *cannot* do it, which is false. The base `_validate_model` (`:731-737`) already raises an accurate `ValueError` about *linopy's* interface, and the docs say plainly that cuOpt itself supports it while linopy's interface does not yet expose it. Non-goal N2. |
| `SOS_CONSTRAINTS` | Genuinely absent upstream ("SOS is explicitly out of scope" in cuOpt's own tests). Base `_validate_model` raises with the reformulation hint. |
| `INDICATOR_CONSTRAINTS` | Genuinely absent upstream (grep over the installed package). |
| `IIS_COMPUTATION` | No such API; `model.compute_infeasibilities()` (`model.py:2293`) then raises the generic message. |
| `SOLVER_ATTRIBUTE_ACCESS` | Gurobi-specific attribute passthrough (`variables.py:1157`); cuOpt has no equivalent. |

---

## 4. Status-mapping table

Two maps, selected on `sol.get_problem_category()`. **They must be separate dicts**:
`LPTerminationStatus` and `MILPTerminationStatus` are `IntEnum`s that share integer values
(`solver_wrapper.pyx:79-101`), so they collide as keys in one dict — `LPTerminationStatus.Optimal ==
MILPTerminationStatus.Optimal` is `True` and both hash to `1`. Unknown statuses fall back to
`TerminationCondition.unknown` via `.get(..., unknown)` as `Highs` does (`:1855-1857`); cuPDLPx's
habit of passing the raw code through (`:4374-4376`) is not copied.

### `ProblemCategory.LP` → `LPTerminationStatus`

| cuOpt status | `TerminationCondition` | Rationale / evidence |
|---|---|---|
| `NoTermination` (0) | `internal_solver_error` | cuOpt's catch-all for malformed input *and* for hitting the memory ceiling; `get_error_message()` is appended to `legacy_status`. Reachable (exploration §5, §6, §9). |
| `NumericalError` (6) | `internal_solver_error` | non-PSD `Q`; `obj=nan`. |
| `Optimal` (1) | `optimal` | |
| `PrimalInfeasible` (2) | `infeasible` | |
| `DualInfeasible` (3) | `unbounded` | Mapping by definition; **status never observed** in 7 attempts (exploration Unknowns). Documented as untested in a code comment. |
| `IterationLimit` (4) | `iteration_limit` | |
| `TimeLimit` (5) | `time_limit` | Note: on the LP path cuOpt returns a full-length **zero** primal vector with `obj=0`, not a partial solution. linopy classifies `time_limit` as `SolverStatus.ok` file-wide (`constants.py:211-217`), which is a linopy-wide convention, not ours to change (I13). WP1 adds a `logger.warning` on this branch saying the returned primal may be meaningless. |
| `PrimalFeasible` (7) | `suboptimal` | `first_primal_feasible=True`. |
| `UnboundedOrInfeasible` (11) | `infeasible_or_unbounded` | The status actually returned for every unbounded LP tried. |

### `ProblemCategory.MIP` / `ProblemCategory.IP` → `MILPTerminationStatus`

| cuOpt status | `TerminationCondition` | Rationale / evidence |
|---|---|---|
| `NoTermination` (0) | `internal_solver_error` | MIQP is rejected before the solve, so this is malformed input / OOM. |
| `Optimal` (1) | `optimal` | |
| `FeasibleFound` (8) | `suboptimal` | Incumbent without proof (time/node limit). `suboptimal` is `SolverStatus.ok`, which is right: there *is* a usable solution. |
| `Infeasible` (2) | `infeasible` | |
| `Unbounded` (3) | `unbounded` | **Never observed**; same comment as LP `DualInfeasible`. |
| `TimeLimit` (5) | `time_limit` | |
| `UnboundedOrInfeasible` (11) | `infeasible_or_unbounded` | |

`ErrorStatus` is **not** mapped to a `TerminationCondition` — it is read alongside, and
`get_error_message()` is folded into `Status.legacy_status` (`legacy_status` is typed
`tuple[str, str] | str`, so a string is fine).

---

## 5. Design decisions on the hard findings

### 5a. Presolve + maximize negated duals → **adopt M2 (always minimize)**

Adopt M2 exactly as validated (21/21 cases against live HiGHS): build with `-M.c` (and `-0.5·M.Q`),
never call `set_maximize(True)`, then negate `get_primal_objective()`, `get_dual_solution()` and
`get_milp_stats()["solution_bound"]` on the linopy side.

Why M2 and not the alternatives:

* **M1 (`presolve=0/1`) is a *verified incomplete* fix** — it repairs the equality cells and still
  returns negated duals on ranged rows. A mitigation that fixes the cases you tested and not the
  ones you did not is worse than none.
* Forcing `method=DualSimplex` fixes the sign by routing off the GPU. Absurd for a GPU solver.
* `dual_postsolve` is irrelevant (measured).
* M2 removes the bug **by construction**: the buggy branch is only reachable through
  `set_maximize(True)`, which linopy never calls. It also makes the root cause irrelevant — which
  matters, because the root cause is an explicit Unknown.

Costs, accepted: (i) `model.to_cuopt()` on a `max` model returns a minimisation `DataModel` —
documented in the `to_cuopt` docstring and in `doc/gpu-acceleration.rst`; (ii) two negations in
`_solve` that a careless reader could mistake for cuPDLPx's `if MAXIMIZE: dual = -dual`. The comment
in `_build_solver_model` states outright that cuOpt's convention is the **opposite** of cuPDLPx's
and that copying the sibling line would introduce a systematic sign error.

### 5b. Concurrent repeat-solve SIGSEGV → **default `method=3` (barrier)**, user-overridable

`method=0` (cuOpt's default) hard-crashes the interpreter on the 2nd–3rd solve above ~1300
variables, deterministically, in ≥12 fresh processes. linopy's own test models sit below the
threshold, so *doing nothing produces a green suite and a SIGSEGV in every real user script that
solves twice.* Options weighed:

| Option | Verdict |
|---|---|
| leave `method=0` | Rejected. A guaranteed crash beats every other failure mode in severity. |
| `method=2` (dual simplex) | Rejected. Measured device usage ≈ baseline: this is the CPU path. Declaring `GPU_ACCELERATION` and defaulting to a CPU simplex is a false claim in practice. |
| `method=1` (PDLP) | Rejected as the *default*. Two concrete strikes: a badly-scaled LP ran 13 minutes at 98 % GPU without terminating and had to be killed, whereas barrier/concurrent solved it in seconds; and PDLP's dual agreement with HiGHS is only 4.7e-05 absolute (vs 2.5e-09 for barrier), which would force a frozen dual tolerance ~10⁴ times looser — the exact "loose tolerance hides a sign bug" trap §3.6 warns about. Documented as the recommended override for very large models. |
| **`method=3` (barrier)** | **Chosen.** Stable over 25 varied sequential solves with flat device memory; GPU-backed (cuDSS is in the dependency tree); tightest dual agreement of all four methods (1.7e-09 / 2.5e-09), which buys a *meaningful* frozen tolerance; and QP is solved by barrier regardless of `method`, so LP and QP behave consistently. |

Implementation: `options = {"method": self._DEFAULT_METHOD, **self.solver_options}` — a user
`method` always wins, including `method=0`. Plus:

* a `logger.warning` when the resolved method is `0`, naming the crash;
* `doc/gpu-acceleration.rst` states the deviation from cuOpt's default, why, and the `method=1`
  escape hatch for very large models;
* a **repeat-solve regression test** at n=1500 (above the measured threshold) for LP and MILP
  (§8, V1-8). This is a check that demonstrably fires: WP1's report must include one run with
  `method=0` showing `exit=139`, per the charter's "make it fail on purpose once" rule.

Note this choice does **not** dodge 5a: the 144-cell sweep shows the negated-dual cells occur with
`solved_by=Unset` under `Barrier` too, because presolve runs before the method does. M2 remains
mandatory.

**Pre-authorised contingency (so no coding agent has to improvise):** if WP1's MILP acceptance runs
show barrier degrading MILP results or stability (never measured for MILP), the fallback is
`method=1` for `model.type` containing `I` and `3` otherwise — reported with the observed numbers
pasted. `method=0` is never a fallback.

### 5c. The `is_available()` fork → **import probe stays; the GPU probe lives in `_license_probe`**

Exact behaviour on a machine with `cuopt-cu12` installed and **no usable GPU**:

| Surface | Behaviour |
|---|---|
| `linopy.available_solvers` | **contains `"cuopt"`** — `is_available()` is `_has_module("cuopt")`, no import, no CUDA. |
| `linopy.licensed_solvers`, `check_solver_licenses()` | **omit `"cuopt"`**, with `LicenseStatus(ok=False, message="InputValidationError: A CPU-memory problem requires remote execution...")`, because `_license_probe` solves a 1×1 LP. |
| `model.solve("cuopt")` | raises `RuntimeError("cuOpt could not find a usable GPU on this machine. cuOpt requires an NVIDIA GPU with compute capability >= 7.0 and driver >= 525.60.13. Original error: ...")` — never a bare CUDA traceback and never the raw "Set CUOPT_REMOTE_HOST" message. |
| the parametrized test suite | skips cuOpt entirely: `test_optimization.py`/`test_solvers.py` parametrize on `licensed_solvers`, which the probe emptied. |

This is a **partial divergence from §4.6 constraint 3**, whose letter says `available_solvers` must
omit cuOpt. I take the divergence knowingly, and it should be filed as a decision entry (and, given
§3.7's "file more escalations, not fewer", an escalation marker) at Gate 3:

* `_AvailableSolvers`' own docstring defines the contract I am keeping: *"Membership means the
  solver's Python package or binary is importable — it does not mean a working license exists"*
  (`solvers.py:4460-4470`). `cuPDLPx.is_available` is `_has_module("cupdlpx")` (`:4188`). Redefining
  that contract for one solver is the drive-by change I4 forbids.
* The alternatives are worse. (a) Importing cuOpt inside `is_available` drags ~6 GB of CUDA modules
  into a `functools.cache`d probe that runs during *test collection* and on the `Model.solve`
  default-solver path. (b) A `ctypes` `cuInit(0)` probe avoids the import but initialises the CUDA
  driver in the calling process — which breaks CUDA in any child that is later `fork`ed, a
  notoriously hard-to-debug failure that would be linopy's fault, in a library whose users run dask.
* §4.6 constraint 3's *purpose* — "never a bare CUDA traceback", an actionable message, and no
  pretence that cuOpt works — is fully met, and it is met through the mechanism linopy already has
  for "installed but not usable".

### 5d. QP: `0.5·M.Q`, MIQP guard, non-PSD

* **Convention:** `Q_cuopt = csr_matrix(0.5 * M.Q)`, kept **full symmetric**. cuOpt minimises
  `c'x + x'Qx` with the effective Hessian `Q + Qᵀ`; `QuadraticExpression.to_matrix` documents that
  `M.Q` *is* the Hessian (`expressions.py:2945-2970`), which is also why HiGHS is handed
  `triu(M.Q)`. Verified live: `0.5*M.Q` reproduces HiGHS's `-4.28571429`, while passing `M.Q`
  returns `Optimal` with `-2.14285714` — a silently halved answer. Under `max`, `sign` negates `Q`
  alongside `c` (4/4 QP cells verified).
* **Do not use `triu`/`tril`.** Two encodings satisfy `Q + Qᵀ = H` and the triangular ones that
  *look* right (`triu(0.5*H)`) are wrong. Full symmetric `0.5*H` is the single least error-prone
  form, and the dedicated convention test (§8, V1-6) pins it against HiGHS.
* **MIQP:** rejected up front with `NotImplementedError`. cuOpt returns `NoTermination`, `obj=0.0`,
  an empty solution and `error_status=4` ("Presolve does not support optimization_problem with a
  quadratic objective"). The guard is `model.is_quadratic and model.type != "QP"` — this covers
  `MIQP` *and* `IQP` (pure-integer quadratic), which a naive `model.type == "MIQP"` check would miss.
  (cuPDLPx's guard at `:4230` has exactly that hole for `ILP`; recorded for
  `logs/45-known-issues.md`, not fixed — I13.)
* **Non-PSD `Q`:** no pre-check. cuOpt detects it and returns `NumericalError` with `obj=nan`, which
  maps to `internal_solver_error` (§4) — an honest, non-crashing report. Duplicating a PSD test in
  linopy would cost an eigenvalue computation on every solve to produce a worse message.
* **`method`/`crossover` are silently overridden to barrier for QP** by cuOpt. Documented as a
  known behaviour; no code.
* **`lp_stats["dual_residual"]` is meaningless under QP** (51.0 on a solved model) — never surfaced,
  which falls out of not populating iteration/residual fields at all.

### 5e. Reduced costs → **do nothing, deliberately**

`grep -rn "reduced_cost\|ReducedCost" linopy/` returns nothing and `constants.Solution` carries only
`primal`, `dual`, `objective` (`constants.py:274-286`). There is nowhere to put them. cuOpt's values
are additionally wrong for maximised models with `<=` rows on the default path (`−c − Aᵀy`), and
Phase 1 explicitly could not find a single corrective transformation. So: `get_reduced_cost()` is
never called, and a one-line comment records why, so that a future contributor does not "add the
missing reduced costs" by guessing a sign. No docs line (reduced costs are not a linopy concept).

### 5f. Solver options

* **bool→int coercion: none.** cuOpt's integer parameters reject Python bools strictly
  (`ValueError: Parameter presolve value False is not an integer`) while `log_to_console` accepts
  them. Whether the bool-typed parameters also accept ints was **not measured**, so a blanket
  `int(v)` coercion would be built on an assumption, and a hard-coded list of "the bool ones" would
  rot against 109 parameter names. cuOpt's own error already names the parameter and the value, so
  it is actionable as-is. The class docstring states the rule ("integer-valued options need 1/0, not
  True/False"), and so do the docs.
* **Unknown names:** cuOpt raises `ValueError("Invalid parameter. Please check documentation")`
  *without naming the offender* — the one message worth improving. `_set_solver_params` re-raises as
  `ValueError(f"Invalid solver option '{k}' for cuOpt: {e}")`. This matters because habitual
  Gurobi/cuPDLPx users type `TimeLimit`, and cuOpt's names are snake_case and case-sensitive.
* **Invalid values** are raised by cuOpt at `Solve()` with a message that already names the
  parameter and value — passed through untouched.
* **`log_fn`:** `self.solver_options["log_file"] = path_to_string(log_fn)`, verbatim the
  `Highs._set_solver_params` idiom (`:1779-1781`) — same option name, same precedence (`log_fn`
  wins over a user `log_file`, resolving Phase 1's open precedence question by repo precedent). No
  `dup2` gymnastics: cuOpt's native `log_file` captures the C++ output (1437 bytes, full banner).
  This makes cuOpt strictly better than cuPDLPx's `logger.warning("Log files are not supported")`.
* **`log_to_console`:** left at cuOpt's default (`true`). HiGHS also prints its log to the console
  under linopy, so this is consistent; users silence it with `log_to_console=False`.
* A `log_file` in an unwritable directory raises from `Solve()` with a clear OS message — surfaced
  unchanged.

### 5g. Interruptibility → **adopt the HiGHS worker-thread pattern**

Measured: SIGINT during `Solve()` is deferred **52.9 s**; with the worker-thread pattern the latency
is **0.00 s** and the interpreter exits cleanly (`exit=0`) even while the daemon thread is still
inside cuOpt's C++ code. There is no cancel API (the only `cancel` in the package is the
out-of-process gRPC client), so the pattern is HiGHS's minus the cancel step. ~35 lines for a solver
whose solves are the longest-running calls in linopy is clearly worth it, and the precedent is in
the same file. The docstring states plainly that the GPU work continues until the process exits and
that `time_limit` is the only real bound — no pretence of cancellation.

### 5h. Warm start → **defer, explicitly**

Usable PDLP warm start needs **three** simultaneous non-default settings (`method=1`,
`pdlp_solver_mode=1`, `presolve=0`); the payload comes back in *presolved* coordinates so it cannot
round-trip at all under the default `presolve=-1`; the default `pdlp_solver_mode` silently produces
`obj=nan`; and it is LP-only. Wiring that up would mean overriding the user's method and disabling
presolve behind their back. So: `warmstart_fn is not None` → `NotImplementedError("Warmstarting not
yet implemented for cuOpt.")`, matching cuPDLPx (`:4352-4355`); `basis_fn` → warn and ignore.

Non-negotiable adjunct: **`DataModel.set_initial_primal_solution` is never called on any code
path.** On an LP it raises `cudaErrorIllegalAddress` and poisons the CUDA context so every later
`Solve()` in the process fails — one warm-start attempt would take down an entire pytest session.
A comment next to the `NotImplementedError` records this, because the API looks inviting.

### 5i. The pyproject extra → `cuopt`

Name **`cuopt`**, contents `"cuopt-cu12>=26.8; platform_system == 'Linux'"`, as Phase 1 Q13 verified
end-to-end from a clean venv. I agree with the exploration's recommendation and against `gpu`:

* `cuopt` matches the solver name, the `SolverName` value and the file's existing lowercase
  single-word extras (`oetc`, `remote`, `docs`, `dev`, `solvers`, `benchmarks`).
* `gpu` promises a category it cannot deliver — it would install cuOpt specifically, Linux only,
  CUDA 12 only, while cuPDLPx (also a GPU solver) stays out. A future second GPU extra can be
  `cupdlpx`; a mis-scoped `gpu` extra can never be narrowed without breaking users.
* Marker: `platform_system == 'Linux'` must be an **equality**, not the file's `!= 'Darwin'` style
  (`:105`) — cuOpt has no Windows wheels either. No `python_version` marker: cuOpt ships cp311–cp314
  and linopy already requires ≥3.11.
* Reachable from **no** other extra: `pip install 'linopy[solvers]'` was verified to pull nothing
  CUDA-related, and Windows/macOS resolution with `--extra cuopt` succeeds with cuOpt absent.
* No `[tool.uv]` index entry: the whole tree is on PyPI, and index settings would not propagate to
  downstream users anyway (§4.6).

---

## 6. Requirement-ID map

§5-table rows numbered top to bottom as C-R1…C-R17.

| ID | §5 row | Work that satisfies it | WP |
|---|---|---|---|
| C-R1 | `SolverFeature` (`:148`) | the 6-flag `features` frozenset (+ `QUADRATIC_OBJECTIVE` in WP6); no enum edit needed. §3 | WP1, WP6 |
| C-R2 | `TYPE_CHECKING` (`:176`) | `import cuopt` | WP1 |
| C-R3 | `_LazyModule` (`:304`) | `cuopt = _LazyModule("cuopt")` + never-`routing` comment | WP1 |
| C-R4 | `SolverName` (`:346`) | `cuOpt = "cuopt"` | WP1 |
| C-R5 | `class cuPDLPx` (`:4156`) | `class cuOpt(Solver[None])` per §2.8, incl. options/log/warmstart/basis/solution-file handling | WP1 |
| C-R6 | `_SOLVER_PROBE_ORDER` (`:4440`) | `"cuopt"` after `"cupdlpx"` | WP1 |
| C-R7 | `solver_capabilities.SOLVER_REGISTRY` | **zero-line diff**; proven by `test_capability_shim_round_trips` + `test_solver_registry_iter_and_index` + the explicit feature-set assertion (V0-3) | WP4 |
| C-R8 | `io.py` `to_cuopt` | `to_cuopt(m)` + TYPE_CHECKING import (§2.2) | WP2 |
| C-R9 | `model.py` binding | import + `to_cuopt = to_cuopt` (§2.3) | WP2 |
| C-R10 | `constants.TerminationCondition` | **zero-line diff**; two status maps (§4) + V1-4 status tests | WP1, WP4 |
| C-R11 | `test/conftest.py` | **zero-line diff**; proven by the paired `-k cuopt` runs with and without `--run-gpu` (§2.6) | WP4 |
| C-R12 | `test_optimization.py` branches | `"cuopt": {"time_limit": 1}` in `test_solver_time_limit_options` (`:562`); cuOpt added to the `test_basis_and_warmstart` skip (`:1110`); `test_infeasible_model` (`:685`) needs **no** cuOpt branch (cuOpt reports `PrimalInfeasible`, it does not hang) | WP4 |
| C-R13 | `test_solvers.py` | 5 feature-matrix rows + 1 registry assertion (§8) | WP4 |
| C-R14 | `pyproject.toml` | the `cuopt` extra (§2.9) + clean-venv acceptance test | WP3 |
| C-R15 | `doc/gpu-acceleration.rst` | new cuOpt section (§9) | WP5 |
| C-R16 | the other doc surfaces + `README.md` | §9 table | WP5 |
| C-R17 | QP (`:4426` read-only) | `QUADRATIC_OBJECTIVE` flag + `0.5·sign·M.Q` build + MIQP guard + QP tests; `QUADRATIC_SOLVERS` untouched | WP6 |

### Proposed new requirements

| ID | Requirement | Work | WP |
|---|---|---|---|
| C-R18 | **M2 always-minimize**: no `set_maximize(True)`; negate `c`, `Q`, objective, duals, `solution_bound` | `_build_solver_model` `sign`, `_solve` negation, comments | WP1, WP6 |
| C-R19 | **Concurrent-segfault mitigation**: default `method=3`, warn on `method=0`, repeat-solve regression test | `_DEFAULT_METHOD`, `_set_solver_params`, V1-8 | WP1, WP4 |
| C-R20 | **Honest GPU degradation**: real `_license_probe`, actionable no-GPU `RuntimeError` | `_license_probe`, `_run_cuopt_solve` | WP1 |
| C-R21 | **Zero-nnz constraint matrix**: box-only models must solve, filler-row dual dropped | `_pad_free_row`, dual slice, V1-7 | WP1, WP4 |
| C-R22 | **Interruptibility**: Ctrl-C responsive within 0.1 s | `_run_cuopt_solve` | WP1 |
| C-R23 | **MIQP rejected up front** with `NotImplementedError` (covers `MIQP` *and* `IQP`) | guard in `_build_solver_model`, V0/V1 test | WP6 |
| C-R24 | **Dedicated V1 differential suite**: six-cell sign matrix **plus** presolve-solvable `max` models, HiGHS called live in-process | `test/test_cuopt.py` | WP4 |

---

## 7. Work-package breakdown

Ordering obeys §6.4: `solvers.py` core first and alone; then disjoint parallel packages; QP last as
its own package; packaging first-class; no two concurrent packages share a file.

### WP1 — `solvers-core` (serial, first, alone) → **commit 1**
*Files:* `linopy/solvers.py` only.
*Requirements:* C-R1 (minus QP), C-R2, C-R3, C-R4, C-R5, C-R6, C-R10 (maps), C-R18, C-R19, C-R20,
C-R21, C-R22.
*Acceptance criteria:*
1. `/home/azureuser/.local/bin/ruff check .` and `.venv/bin/mypy` clean (pasted).
2. `python -c "import linopy; print('cuopt' in linopy.available_solvers, 'cuopt' in linopy.licensed_solvers)"` → `True True` on this machine (pasted).
3. `linopy.solver_capabilities.SOLVER_REGISTRY["cuopt"]` resolves with the declared features (pasted).
4. A scratch script (not committed) proving, against live in-process HiGHS: the six sign cells; the
   presolve-solvable `max` equality model; a MILP; a box-only model. Numbers pasted.
5. Repeat-solve: 3 sequential `model.solve("cuopt")` calls on an n=1500 LP and an n=1500 MILP in one
   process, all `ok`. **Plus** the same script with `method=0` showing `exit=139`, proving the check
   can fire (charter §9 "a harness check that cannot fire").
6. `.venv/bin/pytest -q` and `.venv/bin/pytest --run-gpu -q` with **no new failures** vs
   `logs/05-baseline-pytest*.txt` (pasted). Note WP1 does not declare `QUADRATIC_OBJECTIVE`, so
   `test_quadratic_model*` takes its `pytest.raises(ValueError)` branch and stays green.
7. Ctrl-C latency demonstrated < 1 s on a long solve (pasted timing).

### WP2 — `io-model` (after WP1 merges; parallel with WP3/WP4/WP5) → **commit 2**
*Files:* `linopy/io.py`, `linopy/model.py`. *Requirements:* C-R8, C-R9.
*Acceptance:* ruff + mypy clean; `type(model.to_cuopt())` is a cuOpt `DataModel`; for a `max` model
`to_cuopt(m).get_sense()` shows minimisation and `get_objective_coefficients() == -M.c` (pasted —
this is the documented-surprise check); `pytest --run-gpu -q test/test_io.py` green.
*Must run after WP1* because it references `solvers.cuOpt` (mypy would fail otherwise).

### WP3 — `packaging` (after WP1 merges; parallel) → **commit 3**
*Files:* `pyproject.toml` only. *Requirements:* C-R14.
*Acceptance:*
1. Clean venv, no extra index, no resolver flags:
   `pip install '<worktree>[cuopt]'` → exit 0, `cuopt-cu12==26.8.*` present (pasted tail).
2. In that venv: `python -c "import linopy; print(linopy.available_solvers)"` → **contains
   `"cuopt"`** (this is the acceptance test Q13 could not yet run; pasted).
3. `pip install --dry-run --report` for `linopy[solvers]` → no `cuopt|cudf|nvidia|rmm|pylibraft`
   (pasted).
4. `uv pip compile --python-platform windows|macos --python-version 3.11 --extra cuopt` → exit 0,
   no `cuopt-cu12` (pasted).
5. The temp venv is deleted afterwards (disk).
Run early: a resolution failure here must surface while there is still time.

### WP4 — `tests` (after WP1 merges; parallel with WP2/WP3/WP5) → **commit 4**
*Files:* `test/test_cuopt.py` (new), `test/test_optimization.py`, `test/test_solvers.py`.
*Requirements:* C-R7, C-R11, C-R12, C-R13, C-R19 (test half), C-R21 (test half), C-R24.
*Acceptance:* every V0/V1 row of §8 present and green with pasted output; `pytest -k cuopt -q` shows
all cuOpt tests **skipped** and `pytest --run-gpu -k cuopt -q` shows them **run** (both pasted —
this is the "GPU tests skipped but reported green" guard); full `pytest -q` and
`pytest --run-gpu -q` with no new failures vs baseline.

### WP5 — `docs` (after WP1 merges; parallel) → **commit 5**
*Files:* `doc/gpu-acceleration.rst`, `doc/prerequisites.rst`, `doc/contributing.rst`,
`doc/index.rst`, `doc/api.rst`, `doc/release_notes.rst`, `README.md`,
`examples/using-solvers.ipynb` (prose only). *Requirements:* C-R15, C-R16.
*Acceptance:* every row of §9 present; docs build in the **cuOpt-free** venv
(`PATH=/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin:$PATH .venv-docs/bin/sphinx-build -b html doc doc/_build/check`)
with **no new warning** vs `logs/05-baseline-docs.txt` (diff pasted); no broken references.

### WP6 — `qp` (last; after WP4 merges) → **commit 6**
*Files:* `linopy/solvers.py`, `test/test_cuopt.py`. *Requirements:* C-R17, C-R23, C-R1 (QP flag),
C-R18 (Q negation).
Shares `solvers.py` with WP1 (merged) and `test_cuopt.py` with WP4 (merged) — so it runs strictly
after both, never concurrently.
*Acceptance:*
1. `QUADRATIC_SOLVERS` contains `"cuopt"` **without any edit** to `:4426` (pasted).
2. V1-6 (Q convention vs live HiGHS) green; deliberately re-run with `Q_cuopt = M.Q` to show the
   test fails (`-2.14` vs `-4.29`) — proof the check fires.
3. V1-5 (QP six-cell duals) green.
4. MIQP and IQP models both raise `NotImplementedError` (pasted).
5. `test_quadratic_model`, `test_quadratic_model_cross_terms`, `test_quadratic_model_wo_constraint`,
   `test_quadratic_model_unbounded` green under `--run-gpu` — the third of those is the box-only
   (C-R21) path and the fourth is the non-PSD-adjacent path.
6. Full `pytest -q` / `pytest --run-gpu -q` with no new failures vs baseline.

### Commit map (§6.4a: ~6 substantive commits, linear, one concern each)

| # | Message (plain imperative, no prefix — D2) | Files |
|---|---|---|
| 1 | `Implement interface to cuOpt solver` | `linopy/solvers.py` |
| 2 | `Add cuOpt model conversion function` | `linopy/io.py`, `linopy/model.py` |
| 3 | `Add cuopt extra for GPU solver installation` | `pyproject.toml` |
| 4 | `Update tests` | `test/*` |
| 5 | `Add to docs & release notes` | `doc/*`, `README.md`, `examples/using-solvers.ipynb` |
| 6 | `Add support for quadratic objectives with cuOpt` | `linopy/solvers.py`, `test/test_cuopt.py` |

---

## 8. Test plan

### Where tests live

* **`test/test_cuopt.py` (new, ~230 lines)** — all cuOpt-specific differential work. A new file
  keeps the six-cell matrix out of the heavily shared `test_optimization.py` (a merge-collision
  magnet) and makes the sign matrix readable as one unit.
* **`test/test_optimization.py`** — three surgical edits only (C-R12).
* **`test/test_solvers.py`** — feature-matrix rows (C-R13).

### GPU gating for the new file

No `conftest.py` change (C-R11). `test_cuopt.py` carries:

```python
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.environ.get("LINOPY_RUN_GPU_TESTS")
        or "cuopt" not in solvers.licensed_solvers,
        reason="cuOpt GPU tests need --run-gpu and an installed, GPU-backed cuOpt",
    ),
]
```

`LINOPY_RUN_GPU_TESTS` is set by `conftest.py:47-50` for exactly this purpose ("so test modules can
check if GPU tests are enabled"). `licensed_solvers` (not `available_solvers`) is the right guard:
it is the surface that reflects a *working* GPU (§5c). Tests in the shared files inherit the
automatic `GPU_ONLY` gating instead.

### Tier ladder (§6.3a)

| ID | Tier | Check | Threshold (frozen numbers) |
|---|---|---|---|
| V0-1 | V0 | `ruff check .`, `mypy` | zero findings |
| V0-2 | V0 | `"cuopt" in available_solvers`; `"cuopt" in licensed_solvers`; `_SOLVER_PROBE_ORDER` contains it | exact |
| V0-3 | V0 | `SOLVER_REGISTRY["cuopt"].features == cuOpt.features`; the 5 `test_solver_class_supports_feature` rows | exact |
| V0-4 | V0 | `QUADRATIC_SOLVERS` contains `"cuopt"` after WP6, with no registry edit | exact |
| V1-1 | V1 | **Six-cell sign matrix**: `min`/`max` × `<=`/`>=`/`==` on the 3-variable model from exploration §2 (`A=[[1,2,1],[3,1,1]]`, `b=[4,6]`, `0<=x<=10`), each solved with `model.solve("cuopt")` **and** `model.solve("highs")` live in the same process | dual `rtol=1e-6, atol=1e-7`; objective `rtol=1e-6, atol=1e-8`; primal `rtol=1e-5, atol=1e-6` |
| V1-2 | V1 | **Presolve-solvable `max` models** — the case the six cells miss: the 2-variable square equality system (`x+2y=4, 3x+y=6`, `max x+y`) and an n=10 variant, both of which returned `dual=[-0.4,-0.2]` against HiGHS's `[0.4,0.2]` before M2 | same as V1-1. Additionally the raw-API sibling assertion `get_solved_by() == SolverMethod.Unset`, so the test provably still exercises the presolve path |
| V1-3 | V1 | **MILP** vs live HiGHS on `milp_model`-shaped and binary models | objective `rtol=1e-4`; integrality `max|x - round(x)| <= 1e-5`; duals asserted **empty** |
| V1-4 | V1 | **Status mapping**: optimal, infeasible (`x>=2, x<=1`), unbounded, `time_limit=1e-6` on a large LP, `iteration_limit=1`, MIQP guard | exact `TerminationCondition` values from §4 |
| V1-5 | V1 | **QP six cells** (`min`/`max` × `<=`/`>=`/`==`) vs live HiGHS (WP6) | objective `rtol=1e-6, atol=1e-8`; dual `rtol=1e-6, atol=1e-7` |
| V1-6 | V1 | **Q convention**: the 3-variable cross-term model from exploration §12 vs live HiGHS (WP6) | objective `rtol=1e-6`; must also be demonstrated to fail with `M.Q` instead of `0.5*M.Q` |
| V1-7 | V1 | **Box-only models** (no constraints; and an all-zero row) for LP and QP vs live HiGHS | objective `rtol=1e-6, atol=1e-8`; `dual` length == `ncons` |
| V1-8 | V1 | **Repeat solves**: 3 sequential solves, n=1500 LP and n=1500/200-integer MILP, one process | all `ok`; pairwise objective `rtol=1e-6` (LP); process survives (the real assertion) |
| V1-9 | V1 | **PDLP path smoke** (`method=1`) on a 2000×1000 LP vs live HiGHS | objective `rtol=1e-4`, **no dual assertion** |
| V1-10 | V1 | **`log_fn`** writes a non-empty file containing `cuOpt version`; **unknown option** raises `ValueError` naming the option | exact |
| V2-1 | V2 | Full `pytest --run-gpu` on linopy's own models | no new failure vs `logs/05-baseline-pytest-gpu.txt` |
| V2-2 | V2 | Docs build in the cuOpt-free venv | no new warning vs `logs/05-baseline-docs.txt` |

### Tolerance rationale (numbers, per I12/§3.6)

All V1 differential models are **small (≤ 50 variables) and well-conditioned**, and run under the
class default `method=3` (barrier). Justification for each number, from measured data:

* **Duals `rtol=1e-6, atol=1e-7`.** Measured worst-case barrier disagreement with HiGHS is
  **2.47e-09** absolute across the sign sweeps — a 40× margin at `atol=1e-7`. The `atol` is
  *required*, not slack: non-binding duals come back as `~1e-10` rather than exact `0` (both LP and
  QP), so a pure `rtol` would compare `1e-10` against `0` and fail.
* **Objective `rtol=1e-6, atol=1e-8`.** On the small models cuOpt and HiGHS agreed to ~1e-16
  relative (`-2.80000000` vs `-2.7999999999999994`) and the QP convention check to 8 significant
  figures (`-4.28571429` vs `-4.285714285714258`). The loosest relevant datum is barrier-vs-simplex
  on a *5000-variable random* LP: `12506.8226` vs `12506.8355` = **1.03e-06** relative — which is
  exactly why V1 models are kept small and why `rtol` is not tightened further.
* **Primal `rtol=1e-5, atol=1e-6`.** Primal values can sit at degenerate vertices; measured
  `x=[1.6, 1.2]` was exact, and `2.5e-4` (the existing `GPU_SOL_TOL`) is 250× looser than needed.
* **MILP objective `rtol=1e-4`.** cuOpt's own default `mip_relative_gap` is `1e-4`; demanding better
  agreement than the solver's own optimality guarantee would be a test that fails for correct
  behaviour. **Integrality `1e-5`** is cuOpt's default `mip_integrality_tolerance`.
* **PDLP smoke `rtol=1e-4`.** Measured PDLP-vs-barrier disagreement at default tolerances on a
  5000-variable LP was **2.5e-06** relative (40× margin), while PDLP's *dual* error reaches
  **4.71e-05** absolute — hence objective-only. Note the exploration's 5.5e-03 relative objective
  move was between `tol=0.01` and `tol=1e-10`, i.e. a *tolerance-setting* effect, not the
  default-vs-default disagreement this row measures; it is the reason a PDLP-defaulted class would
  have needed a ~1e-2 threshold, and part of why `method=3` is the default (§5b).
* `GPU_SOL_TOL = 2.5e-4` / `CPU_SOL_TOL = 1e-5` in `test_optimization.py:80-81` are **unchanged**;
  cuOpt inherits `GPU_SOL_TOL` automatically via `gpu_solvers`.

### Pre-declared contingency in the shared test files

`test_solvers.py` has no GPU tolerance accommodation, and several of its assertions use
`pytest.approx` (default `rel=1e-6`) on values like `x=-0.1` — tighter than cuOpt's default `1e-4`
feasibility tolerances. cuPDLPx never exercised these (it is not installed). **If** any such
assertion goes red under `--run-gpu`, the only permitted response is: paste the failing output
verbatim, then relax *that* assertion to the existing repo constant `GPU_SOL_TOL` (2.5e-4) for GPU
solvers — the same move PR #516's commit 6 made in `test_optimization.py`, and never a bespoke
number. Any other red is reported and stops that item (I12).

### Edge cases explicitly covered (from exploration §5/§6)

Infinite bounds passed as `np.inf` (no sentinels); free variables; equality rows via
`lower == upper`; variables fixed by `lb == ub`; masked variables and masked constraints (already in
the shared suite); zero-nnz matrices (V1-7). Ranged rows are **unreachable** from linopy — every
`M.sense` entry is one of `<`, `>`, `=`, so our build always sets one bound to `±inf` or both to
`b`. That is worth stating because it is why the ranged manifestation of the dual bug cannot bite us
(and M2 covers it regardless).

---

## 9. Docs plan (every row of charter §6.6b)

| Surface | Content |
|---|---|
| `doc/gpu-acceleration.rst` | New `cuOpt` section under *Supported GPU Solvers*, after cuPDLPx. Contains: one-line intro (NVIDIA cuOpt, concurrent PDLP / barrier / dual-simplex, Apache-2.0); **install** `pip install 'linopy[cuopt]'` (and the uv equivalent) with **Linux-only**, **driver ≥ 525.60.13**, **compute capability ≥ 7.0** stated *next to* the command, plus the `cu12` vs `cu13` caveat (`cu13` wheels need a newer driver — the extra pins `cu12`) and the warning that the bare `cuopt` package on PyPI is **a different project**; **usage** `m.solve("cuopt")` and `m.solve("cuopt", time_limit=60)`; **supported** LP, MILP, QP (`io_api="direct"` only); **not supported** MIQP, SOS, indicator constraints, semi-continuous variables (noting cuOpt itself supports them, linopy's interface does not yet), warm start / basis files, file `io_api`s, solution files; **duals** are returned for LP/QP and not for MILP; **method default** — linopy defaults `method=3` (barrier) instead of cuOpt's `0` (concurrent), which crashes on repeat solves in one process, with `method=1` recommended for very large models; **options** are snake_case, case-sensitive, integer-valued ones take `1`/`0`; **logging** via `log_fn` / `log_to_console`; the `pytest test/test_cuopt.py --run-gpu` line; links to NVIDIA's docs. |
| `doc/prerequisites.rst` | Add to the *GPU-accelerated solvers* list: `cuOpt <https://docs.nvidia.com/cuopt>`__ — open source (Apache-2.0), GPU-accelerated LP/MILP/QP solver. Then the one-command install `uv pip install "linopy[cuopt]"` as the **headline**, with Linux-only + driver/CC on the adjacent line and a pointer to `gpu-acceleration`. `pypi.nvidia.com` is **not** mentioned (the tree is on PyPI; no `--index-strategy` anywhere). |
| `doc/contributing.rst` | Extend the existing GPU-testing note: "(e.g. cuPDLPx, cuOpt)" and one line — to run the cuOpt tests locally, install the extra (`uv pip install "linopy[cuopt]"`) and run `pytest --run-gpu`. No change to the `uv sync` line. |
| `doc/index.rst` | Add `- cuOpt <https://docs.nvidia.com/cuopt>`__ (GPU-accelerated) to the solver list, after cuPDLPx. |
| `README.md` | Add `* [cuOpt](https://docs.nvidia.com/cuopt)` to the solver list at `:153`. |
| `doc/api.rst` | Add `solvers.cuOpt` to the solver-class autosummary (`:637`, after `solvers.cuPDLPx`) **and** `model.Model.to_cuopt` to the `Model` autosummary (near `:114-115`). Note the file currently lists **no** `to_*` solver-conversion helper — not `to_gurobipy`, `to_highspy`, nor `to_cupdlpx` — so `to_cuopt` lands as an outlier. Charter §6.6b names `to_cuopt` explicitly, so it goes in; the missing siblings are a pre-existing documentation gap recorded in `logs/45-known-issues.md` and **not** fixed here (I13/I4). |
| `doc/release_notes.rst` | One bullet under *Upcoming Version* → *Other*: "Add support for the GPU-accelerated NVIDIA cuOpt solver (LP, MILP and QP) via ``io_api="direct"``, installable with ``pip install 'linopy[cuopt]'`` (Linux, CUDA 12). See :doc:`gpu-acceleration`." |
| `pyproject.toml` | The extra is explained where extras are explained — `doc/prerequisites.rst` (above). The TOML block itself carries the 4-line comment in §2.9. |
| Install instructions everywhere | One command first (`pip install 'linopy[cuopt]'`), Linux-only + driver/CC beside it, manual `cuopt-cu12` never the headline (and only mentioned in `gpu-acceleration.rst`, as a fallback), bare-`cuopt`-is-a-different-project warning present. |
| `examples/using-solvers.ipynb` | **Decision: yes, prose only.** The sentence at cell `:11` enumerates the backends; add "and NVIDIA cuOpt" next to cuPDLPx. No code cell — the docs build executes these notebooks in a **cuOpt-free, GPU-free** venv, so any executable cuOpt content would break ReadTheDocs. Log as a D-entry. |

**Gap check (§6.6b):** PR #516 also touched `CLAUDE.md` (+9/−1) with a GPU-testing note. That
content now lives in `AGENTS.md`, which is about contribution rules, not solvers, and its existing
guidance is already solver-generic. **Deliberate justified difference — do not touch
`CLAUDE.md`/`AGENTS.md`.** Also expect `test/conftest.py` and `linopy/solver_capabilities.py` to
appear in #516's list and not in ours: both are zero-line-diff by design (C-R7, C-R11), which is a
*better* outcome than #516's, not a gap.

---

## 10. Explicit non-goals

| # | Non-goal | Why |
|---|---|---|
| N1 | File `io_api`s (`lp`, `mps`, `lp-polars`) and `READ_MODEL_FROM_FILE` | §1.2. Proven feasible by Phase 1; deliberately deferred. |
| N2 | Semi-continuous variables | §3. Supported by cuOpt, unverified through linopy; the class must not *claim impossibility*, and does not. |
| N3 | Warm start / basis files / `set_initial_*_solution` | §5h. Three-setting requirement + CUDA-poisoning bug. |
| N4 | Solution-file output (`solution_fn`, cuOpt's `solution_file` parameter) | Unverified upstream surface; `SOLUTION_FILE_NOT_NEEDED` makes it unnecessary. |
| N5 | Reduced costs | §5e. No linopy surface, and cuOpt's values are wrong on the default `max`/`<=` path. |
| N6 | Persistent / in-place updates (`supports_persistent_update`) | `DataModel` has no update API. |
| N7 | MIQP and quadratic **constraints** | Unsupported upstream (`add_quadratic_constraint` exists but rejects equalities and has no linopy model surface). |
| N8 | MIP callbacks, batch solve (`BatchSolve`), `problem.Problem`, `cuopt.routing` | No linopy surface; `routing` must never be imported (global `sys.excepthook`). |
| N9 | Changing `is_available()` semantics, `TerminationCondition`, `SolverFeature`, `QUADRATIC_SOLVERS`, `solver_capabilities.py`, or `conftest.py` | I4 / charter §5: all zero-line-diff or out of scope. |
| N10 | Fixing pre-existing linopy defects noticed on the way (e.g. `cuPDLPx._build_direct`'s guard missing `ILP`/`MIQP`; `cuPDLPx._solve` passing a raw status code to `Status.from_termination_condition`) | I13 — record in `logs/45-known-issues.md`, never fix. |
| N11 | A default `time_limit`, or tightening cuOpt's default tolerances globally | Would silently change results for every user; the badly-scaled-LP hang is documented instead. |
| N12 | Benchmarks in this venv | §4.4 — pandas pin conflict. |

---

## 11. Risks specific to this plan

Ranked by expected damage × likelihood.

1. **`method=3` (barrier) is untested for MILP.** Every repeat-solve measurement for MILP used
   `method=2`. If barrier is a poor default for cuOpt's MILP engine (slower, or worse incumbents),
   `test_milp_model*` and `test_model_with_inf` could go red or slow. Mitigated by the pre-declared
   contingency in §5b (per-class default, never `method=0`) and by making MILP repeat-solve a WP1
   acceptance criterion rather than a discovery in Phase 5. This is also the point where I most
   expect a sibling plan to disagree — switching to `method=1` is a one-line change.
2. **Exact-equality assertions on MILP solutions in existing tests.**
   `test_model_with_inf` asserts `(solution.x == 0).all()` and `(solution.y == 10).all()` — exact,
   not `isclose`. If cuOpt returns `0.9999999997` for an integer variable (its
   `mip_integrality_tolerance` is `1e-5`), these fail. There is no honest fix on our side other than
   reporting it; the plan therefore surfaces it as a WP1 acceptance run rather than a Phase-5
   surprise. Rounding integer solutions inside the solver class would be a silent data change and is
   **not** authorised.
3. **`test_quadratic_model_unbounded` under barrier.** Objective `-x-y+x*x` with `Q` only
   semi-definite: cuOpt may return `NumericalError` (→ `internal_solver_error`, `status="error"`)
   where the test accepts only `unbounded`/`unknown`/`infeasible_or_unbounded`. If so it is a
   verbatim report, and the honest options are a cuOpt-specific branch in that test (precedent:
   cuPDLPx has three such branches) — decided by evidence in WP6, not now.
4. **`pytest.approx(rel=1e-6)` assertions in `test_solvers.py`** are tighter than cuOpt's default
   `1e-4` tolerances. Bounded by the pre-declared `GPU_SOL_TOL` contingency in §8; the risk is that
   it turns into a wider set of edits than expected, inflating the diff.
5. **The zero-nnz filler row is my most invasive invention.** It changes the matrix handed to the
   solver. If the filler row's dual is not the *last* element of `get_dual_solution()`, the
   truncation `dual[: len(self._clabels)]` silently misaligns duals for box-only models. V1-7 checks
   duals for the all-zero-row case explicitly, and the alternative (raise, and add a skip to
   `test_quadratic_model_wo_constraint`) is a clean fallback if it misbehaves.
6. **`cuopt.linear_programming.solver.solver_wrapper` is a deep attribute path** for the status
   enums, and it is a `.pyx` extension module. It is reachable today (its parent packages import it
   at load time), but it is more internal than `cupdlpx.PDLP.*`. If a future cuOpt release moves it,
   the status maps break at import of the first solve. WP1 must verify the path in one line and, if
   it is fragile, fall back to keying the maps on `get_termination_reason()` strings.
7. **`_license_probe` runs a real solve during test collection**, because
   `test_optimization.py`/`test_solvers.py` parametrize on `licensed_solvers`. That is ~0.3 s once,
   but on a busy or broken GPU it could slow or noisily fail collection for the *whole* suite. The
   probe is deliberately a 1-variable LP with `log_to_console=False`; if collection cost becomes a
   problem, the fallback is a `_license_probe` that only checks importability of
   `cuopt.linear_programming` — at the price of §5c's honest degradation.
8. **Divergence from §4.6 constraint 3's letter** (§5c). A reviewer holding the charter literally
   will read `available_solvers` still listing cuOpt on a GPU-less box as an unmet Definition-of-Done
   item. I believe the reasoning is right and I have made it explicit rather than quiet, but this is
   the plan's most likely *governance* dispute.
9. **Bulk.** Estimated ~+360 in `linopy/solvers.py`, ~+12 in `io.py`/`model.py`, ~+250 in tests,
   ~+120 in docs, +5 in `pyproject.toml` ≈ **+750/−0 over 13 files** — right at PR #516's +771/−62
   and well inside the 1.5× watermark. The two places that could push it over are the shared-test
   contingency (risk 4) and the QP package; both are called out so growth is visible.
10. **Ctrl-C leaves the GPU busy.** The worker-thread pattern returns control immediately, but the
    solve continues to completion in the background. A user who interrupts and immediately starts a
    second solve contends for the GPU. Documented in the helper docstring and the docs; not fixable
    without a cancel API cuOpt does not have.
