# plan-3 — cuOpt integration for linopy

*Author: Phase 2 planner k=3 (Opus 5, `claude-opus-5[1m]`) · 2026-08-22 · Independent; no sibling
plan was read.*

Basis: charter `PROJECT_LEAD.md` §§4–9, `logs/10-exploration.md` (single source of empirical
truth), `logs/30-decisions.md` rules block, and current `master` (`09c34dd`) as the model of
correctness. All line numbers below are current-`master` line numbers.

---

## 0. One-paragraph summary of the position

cuOpt is ingested through **`data_model.DataModel` + `solver.Solve` only** (`io_api="direct"`);
the verified-but-second file path (`cuopt.linear_programming.io.Read`) is a **documented
non-goal** for this run. Every `max` model is handed to cuOpt in **minimisation form**
(mitigation **M2**) so the presolve dual-sign defect is unreachable by construction. linopy
overrides cuOpt's default `method=0` (Concurrent) with **`method=3` (Barrier)** because
Concurrent segfaults the interpreter on repeat solves. `is_available()` stays a pure import
probe (repo precedent), while `_license_probe()` becomes a **real GPU probe** so
`licensed_solvers` and `check_solver_licenses()` are honest, and the no-GPU solve failure is
translated into an actionable message. QP ships last, with `Q_cuopt = 0.5 · model.matrices.Q`
kept full-symmetric, and MIQP/IQP rejected up front. Reduced costs and warm start are not
exposed. The extra is `linopy[cuopt]` = `cuopt-cu12>=26.8; platform_system == 'Linux'`.

---

## 1. Entry-point decision

### 1.1 Decision

| Aspect | Decision |
|---|---|
| Entry point | `cuopt.linear_programming.data_model.DataModel` + `cuopt.linear_programming.solver.Solve` |
| Settings | `cuopt.linear_programming.solver_settings.SolverSettings` |
| `accepted_io_apis` | `frozenset({"direct"})` |
| `SolverFeature.READ_MODEL_FROM_FILE` | **not** declared |
| `SolverFeature.LP_FILE_NAMES` | **not** declared |
| Non-direct `io_api` requests | served through the direct path after one `logger.warning` (see §2.1.9) |
| `problem.Problem` | never used |
| `cuopt.linear_programming.io` | not used in this run — recorded as an evidence-backed follow-up |
| `cuopt.routing` | never imported (installs a global `sys.excepthook` writing `error_log.txt` into cwd) |

### 1.2 Where I agree with the Phase 1 Q0 recommendation

`DataModel` as the **primary** entry point: agreed, and not marginally. Q0's measurements are
decisive and I do not re-derive them:

* 13–16 lines of translation from `model.matrices`, versus 21 for `Problem` (which
  additionally **cannot express ranged rows at all**).
* Build time at 10⁵ variables: **0.012–0.014 s** vs **0.858–1.042 s** for `Problem` (~70×) and
  **0.83–0.95 s** for the file path.
* `DataModel` is the only path with a matrix-shaped quadratic setter
  (`set_quadratic_objective_matrix`); `Problem` needs `n_terms` Python objects to express a
  matrix linopy already holds as `M.Q`.
* Everything that looked like a `Problem`-only advantage in charter §4.3 lives on
  `SolverSettings`, which all candidates share.

`problem.Problem` is rejected outright: 70× slower, structurally unable to express ranged rows,
already carries a `DeprecationWarning` on `getQcsr()`, and buys nothing linopy needs.

### 1.3 Where I diverge from Q0 — the `io` submodule as a secondary entry point

Q0 recommends declaring `accepted_io_apis ⊇ {"lp", "mps"}` and implementing `_run_file` via
`io.Read`. **I diverge.** The capability is real and Q0's fidelity evidence is convincing; my
objection is to shipping it *in this run*, for five reasons, in descending weight:

1. **It is a second, independent ingestion path that needs its own correctness proof.** Every
   guarantee the direct path establishes — the M2 sense transform, the `0.5·Q` convention, the
   `'B'→'I'` mapping, the vtype whitelist, the zero-nnz guard — has to be re-established against
   arrays that cuOpt's parser produced rather than arrays linopy built. Concretely: on the file
   path the reader sets the sense itself, so M2 becomes "read, then negate `c` **and** the
   quadratic CSR values in place, then flip `set_maximize(False)`"; and the Q convention
   *inverts* — exploration Q0 shows `.lp` yields triangular `Q=[1,1,1]` and `.mps` yields
   symmetric `[1,0.5,0.5,1]`, i.e. **the reader already applies cuOpt's factor**, so the
   `0.5 ×` scaling that is mandatory on the direct path is **wrong** on the file path. Two
   opposite rules for the same silently-wrong-objective trap (§9's named risk) is exactly the
   shape of bug this project exists to avoid.
2. **The parser surface linopy would exercise is only partly verified.** Exploration measured
   `lp` and `mps` at 750 variables with default names. Unverified: `lp-polars` output (a
   different writer in `FILE_IO_APIS`), `explicit_coordinate_names=True` (linopy's `x[a,b]`
   spellings), and MILP/QP name round-trips at scale. I can neutralise the third by not
   declaring `LP_FILE_NAMES` (linopy then warns and disables it automatically —
   `solvers.py:775-780`), but not the first.
3. **Test-matrix cost.** Declaring `READ_MODEL_FROM_FILE` puts `"cuopt"` into
   `file_io_solvers` (`test_optimization.py:44`), so `params` gains six combinations
   (`lp`/`lp-polars`/`mps` × `explicit_coordinate_names`) on top of `("cuopt","direct",False)`.
   Every one of the ~40 `params`-parametrized tests then runs seven times, doubled again by the
   `semantics` fixture, all on the GPU.
4. **Charter §6.4 rule 4 says "QP is last".** A file work package must touch `solvers.py` and
   would have to either precede QP (and then be extended by the QP package, so the QP
   convention lands in two places) or follow it (breaking rule 4). Dropping the file package
   removes the tension entirely.
5. **`accepted_io_apis` is inert.** `grep -rn "accepted_io_apis" linopy/ test/` matches only the
   declaration at `solvers.py:466` and a test double at
   `test_persistent_solver_orchestrator.py:24`. No production code consumes it, so "declaring
   `lp`/`mps`" buys documentation, not behaviour. The behaviour comes from
   `READ_MODEL_FROM_FILE` + `_run_file`, which is the expensive part.

**What I keep from Q0's finding.** The DoD line "`model.solve("cuopt")` solves LP and MILP
correctly" must hold for a bare call with no `io_api`, and today `model.solve("cupdlpx")` raises
`NotImplementedError` in exactly that situation (`solvers.py:4214`). I satisfy it with a
**6-line `_build_file` override** that warns once and builds directly (§2.1.9) — the same
warn-and-redirect idea cuPDLPx already uses in `_run_file`, moved one step earlier so no LP file
is written and discarded. cuOpt therefore ends up strictly better than cuPDLPx on the default
path (it works) at 6 lines instead of ~120 plus a doubled GPU matrix.

**Recorded follow-up (not this run).** `logs/30-decisions.md` should carry: *cuOpt can support
linopy's `lp`/`mps` `io_api`s via `cuopt.linear_programming.io.Read` (proven in Phase 1 Q0 with
full name/order/integrality/Q fidelity); deferred because it is a second ingestion path with an
inverted Q convention and an unverified `lp-polars`/named-variable surface.* That is a finding
handed forward, not a capability quietly dropped.

---

## 2. Full file-by-file change set

Estimated total: **+770 / −5 across 14 files** (yardstick: PR #516 was +771/−62 over 15;
watermark ~1.5× = +1150).

### 2.1 `linopy/solvers.py` — the bulk (~+335 lines, −0)

#### 2.1.1 `SolverFeature` (`:148-164`) — no edit
Every feature cuOpt needs already exists. **Zero-line diff**; requirement C-R1 is satisfied by
the `features` frozenset on the new class (§3), not by touching the enum.

#### 2.1.2 `TYPE_CHECKING` block (`:175-181`) — +1
```python
if TYPE_CHECKING:
    import cupdlpx
    import cuopt
    import gurobipy
```
Alphabetical inside the block, matching the existing order. `[tool.mypy] ignore_missing_imports
= true` (`pyproject.toml:152`) means the attribute chains below type as `Any`; no stubs needed.

#### 2.1.3 Lazy module handle (after `:304`) — +4
```python
cupdlpx = _LazyModule("cupdlpx")
# Only cuOpt's LP surface is ever touched: importing ``cuopt.routing`` installs a
# global ``sys.excepthook`` that writes ``error_log.txt`` into the working directory.
cuopt = _LazyModule("cuopt.linear_programming")
```
**Refinement of charter row C-R3** (`cuopt = _LazyModule("cuopt")`): the handle points at
`cuopt.linear_programming`, so call sites read `cuopt.data_model.DataModel()`,
`cuopt.solver.Solve(...)`, `cuopt.solver_settings.SolverSettings()` — one attribute hop, and the
comment documents why the top-level package is deliberately not the handle.
`_LazyModule.__getattr__` uses `importlib.import_module(self._name)`, which resolves a dotted
submodule path unambiguously.

> **WP-A acceptance step (mechanical, must be pasted):**
> `.venv/bin/python -c "from linopy.solvers import cuopt; print(cuopt.data_model.DataModel, cuopt.solver.Solve, cuopt.solver_settings.SolverSettings)"`
> If any attribute does not resolve (i.e. `cuopt/linear_programming/__init__.py` does not bind
> its submodules), fall back to `cuopt = _LazyModule("cuopt")` plus function-local
> `from cuopt.linear_programming import data_model, solver, solver_settings` inside
> `_build_solver_model` / `_solve`, and report the change.

#### 2.1.4 `SolverName` (`:346-359`) — +1
```python
    cuPDLPx = "cupdlpx"
    cuOpt = "cuopt"
```
Vendor spelling, appended (the enum's own convention). The member **name** must equal the class
name — `solver_capabilities._solver_class()` resolves via
`getattr(solvers, SolverName(name).name)` (`solver_capabilities.py:55`), and `Solver.solver_name`
resolves via `SolverName[self.__class__.__name__]` (`solvers.py:1205`). `cuOpt = "cuopt"` +
`class cuOpt` satisfies both.

#### 2.1.5 Keyboard-interrupt helper (new module-level function, next to `_run_highs_with_keyboard_interrupt` at `:189`) — +26
```python
def _run_cuopt_with_keyboard_interrupt(solve: Callable[[], Any]) -> Any:
    """
    Run a cuOpt solve while keeping Ctrl-C responsive.

    ``solver.Solve`` blocks in C++ for the whole solve and cuOpt exposes no
    cancel hook, so a SIGINT is deferred until the solve finishes. Running it in
    a worker thread lets the main thread raise ``KeyboardInterrupt`` immediately.
    The GPU work is not cancelled — it continues in the background until the
    process exits, so ``time_limit`` is the only hard bound cuOpt offers.
    """
    finished = threading.Event()
    result: Any = None
    run_error: BaseException | None = None

    def _target() -> None:
        nonlocal result, run_error
        try:
            result = solve()
        except BaseException as exc:  # pragma: no cover
            run_error = exc
        finally:
            finished.set()

    threading.Thread(target=_target, name="linopy-cuopt-solve", daemon=True).start()
    while not finished.wait(0.1):
        pass
    if run_error is not None:
        raise run_error
    return result
```
Shaped as "take a zero-argument callable" so it is testable with a dummy on a CPU-only machine,
mirroring `test/test_highs_keyboard_interrupt.py`. Deliberately **no cancel step** and **no
`finished.wait()` drain in the `except` branch**: exploration §8 measured a clean immediate
interpreter exit (`shell-observed exit=0`) with the daemon thread still inside cuOpt's C++ code,
so blocking on a solve the user just interrupted would reintroduce the 52.9 s hang the helper
exists to remove.

#### 2.1.6 `class cuOpt(Solver[None])` — placed immediately after `cuPDLPx` ends (`:4416`), before `_solver_class_for` (`:4419`) — ~+270
Override set, in this order (mirrors `cuPDLPx`, §5.1 of the charter):
`display_name`, `features`, `accepted_io_apis`, `_default_method` (new ClassVar),
`is_available`, `_license_probe`, `_build_file`, `_build_direct`, `_build_solver_model`,
`_run_direct`, `_solve`, `_set_solver_params`.

Class docstring documents: what cuOpt is, the driver/CC requirement, that linopy defaults
`method` to Barrier and why, and the notable solver options (`time_limit`, `method`,
`log_to_console`, `log_file`, `absolute_*/relative_*_tolerance`, `mip_relative_gap`,
`node_limit`) — matching the depth of `cuPDLPx`'s and `Highs`' docstrings.

```python
    display_name: ClassVar[str] = "cuOpt"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.DIRECT_API,
            SolverFeature.GPU_ACCELERATION,
            SolverFeature.GPU_ONLY,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.SEMI_CONTINUOUS_VARIABLES,
            # SolverFeature.QUADRATIC_OBJECTIVE added by the QP work package
        }
    )
    accepted_io_apis: ClassVar[frozenset[str]] = frozenset({"direct"})
    # cuOpt's own default (``method=0``, Concurrent) segfaults the interpreter on the
    # second or third solve in one process for models above ~1300 variables; Barrier
    # is stable, GPU-backed and the most accurate single method against HiGHS.
    _default_method: ClassVar[int] = 3
```

**`is_available` / `_license_probe`** — see §5c.

**`_build_direct`** (mirrors `cuPDLPx._build_direct`, `:4227`):
guards, then `self.solver_model = self._build_solver_model(model)`, `self.io_api = "direct"`,
`self.sense = model.sense`, `self._cache_model_labels(model)`. Guards, in order:

| Guard | Raise | Why |
|---|---|---|
| `model.type in ("MIQP", "IQP")` | `NotImplementedError("cuOpt does not support quadratic objectives with integer variables (MIQP). Use a solver that does (gurobi, highs).")` | exploration §12: MIQP returns `NoTermination`, `obj=0.0`, empty solution, `error_status=4` — a silent wrong answer. **Note `"IQP"`, which `model.type` produces for an all-integer quadratic model (`model.py:1666-1679`); exploration's "`model.type == 'MIQP'`" misses it.** |
| `kwargs.get("explicit_coordinate_names")` | `warnings.warn(..., UserWarning, stacklevel=2)` | verbatim shape of `cuPDLPx:4233-4239`; cuOpt is fed positionally, names are not set |

**`_build_solver_model(model) -> Any`** (`@staticmethod`, mirrors `cuPDLPx:4246`):

```python
        M = model.matrices
        if M.A is None:
            raise ValueError("Model has no constraints, cannot export to cuOpt.")
        A = M.A.tocsr()
        if A.nnz == 0:
            raise ValueError(
                "Model has no non-zero constraint coefficients; cuOpt cannot solve it."
            )
        if A.shape[1] != len(M.c):
            raise ValueError("Constraint matrix width does not match the objective length.")

        lower = np.where(M.sense != "<", M.b, -np.inf)
        upper = np.where(M.sense != ">", M.b, np.inf)

        vtypes = np.where(M.vtypes == "B", "I", M.vtypes)
        if not np.isin(vtypes, ("C", "I", "S")).all():
            raise ValueError(f"Unsupported variable types for cuOpt: {set(vtypes)}")

        # cuOpt returns negated duals for maximised models finished inside presolve,
        # so linopy always hands it the equivalent minimisation and negates the
        # objective and the duals afterwards (see `_solve`).
        c = -M.c if model.objective.sense == "max" else M.c

        dm = cuopt.data_model.DataModel()
        dm.set_csr_constraint_matrix(A.data, A.indices, A.indptr)
        dm.set_constraint_lower_bounds(lower)
        dm.set_constraint_upper_bounds(upper)
        dm.set_objective_coefficients(c)
        dm.set_variable_lower_bounds(M.lb)
        dm.set_variable_upper_bounds(M.ub)
        dm.set_variable_types(vtypes)
        return dm
```

Notes fixing this shape against exploration:
* `lower`/`upper` use the `!=` form from `Highs._build_solver_model` (`:1702-1703`) rather than
  cuPDLPx's `logical_or` pair — same result, and `Highs` is the idiom for the bounds-based
  encoding cuOpt needs.
* `np.inf` / `-np.inf` are passed **directly**; exploration §5 verified they work and no
  sentinel is needed.
* `set_row_types` is never called — there is no `'R'` row type, and when both are set the bounds
  win (`data_model.py:468-470`).
* `set_objective_offset` is never called: linopy rejects objective constants outright
  (`objective.py:192-193`, `"Constant values in objective function not supported."`), so there
  is nothing to offset. `set_objective_scaling_factor` is never called (exploration §3: it
  scales the reported objective without scaling the duals).
* `M.A.tocsr()` already yields `float64`/`int32`/`int32`, exactly what cuOpt stores — no cast,
  no `UserWarning` (exploration §6).
* `vtypes` is an `np.ndarray` (`<U1`) — never a list/str/bytes, which raise; and never
  user-supplied, because unrecognised characters are silently treated as continuous.
  `'B'→'I'` is safe because linopy already carries `lb=0, ub=1` for binaries.

**`_run_direct`** (mirrors `cuPDLPx:4284`): forwards to `self._solve(self.solver_model, ...)`
passing `log_fn`, `solution_fn`, `warmstart_fn`, `basis_fn`, `io_api=self.io_api`,
`sense=self.sense`.

**`_solve(dm, solution_fn=None, log_fn=None, warmstart_fn=None, basis_fn=None, io_api=None,
sense=None) -> Result`** — the core, structured exactly like `cuPDLPx._solve` (`:4303`):

1. `CONDITION_MAP` (see §4) — a module-scope-free local dict keyed by the **status name string**,
   with a `# see cuopt/linear_programming/solver/solver_wrapper.pyx` comment.
2. Refusals, in `cuPDLPx`'s order and phrasing:
   * `warmstart_fn is not None` → `NotImplementedError("Warmstarting is not yet implemented for cuOpt.")`
   * `basis_fn is not None` → `logger.warning("Basis files are not supported by cuOpt. Ignoring.")`
   * `solution_fn is not None` → `NotImplementedError("Solution file output is not yet implemented for cuOpt.")`
3. `settings = cuopt.solver_settings.SolverSettings()`; `self._set_solver_params(settings, log_fn)`.
4. `is_mip = _cuopt_has_integrality(dm.get_variable_types())` — a 3-line module-level helper
   accepting both `<U1` and `|S1` and testing for `I`/`S`. Read off the `DataModel` rather than
   the enum name so no unverified `ProblemCategory` spelling is relied on, and no instance state
   is added to the dataclass.
5. `sol = _run_cuopt_with_keyboard_interrupt(lambda: cuopt.solver.Solve(dm, settings))`, wrapped:
   ```python
   try:
       sol = _run_cuopt_with_keyboard_interrupt(lambda: cuopt.solver.Solve(dm, settings))
   except Exception as exc:
       if "CUOPT_REMOTE_HOST" in str(exc):
           raise RuntimeError(
               "cuOpt found no usable GPU. cuOpt requires an NVIDIA GPU with compute "
               "capability >= 7.0 and driver >= 525.60.13, and cannot solve on CPU. "
               "Use a CPU solver (e.g. 'highs') instead."
           ) from exc
       raise
   ```
   The check is narrow and the original exception is always chained or re-raised untouched, so a
   real error is never mislabelled (§5c).
6. Status:
   ```python
   name = sol.get_termination_status().name
   termination_condition = CONDITION_MAP.get(name, TerminationCondition.unknown)
   status = Status.from_termination_condition(termination_condition)
   status.legacy_status = name
   message = sol.get_error_message()
   if message:
       logger.error(f"cuOpt reported: {message}")
   ```
7. Solution:
   ```python
   def get_solver_solution() -> Solution:
       objective = sol.get_primal_objective()
       primal = np.asarray(sol.get_primal_solution(), dtype=float)
       # Duals are only available for continuous problems; get_dual_solution()
       # raises for a MILP solution. Reduced costs are not exposed: linopy has no
       # surface for them and cuOpt's values are wrong for maximised models.
       dual = (
           np.array([], dtype=float)
           if is_mip
           else np.asarray(sol.get_dual_solution(), dtype=float)
       )
       if sense == "max":
           # the model was handed to cuOpt in minimisation form
           objective = -objective
           dual = -dual
       primal = _solution_from_labels(primal, self._vlabels, self._n_vars)
       dual = _solution_from_labels(dual, self._clabels, self._n_cons)
       return Solution(primal, dual, objective)

   solution = self.safe_get_solution(status=status, func=get_solver_solution)
   solution = maybe_adjust_objective_sign(solution, io_api, sense)
   ```
   **No `if MAXIMIZE: dual = -dual` copied from cuPDLPx (`:4385-4386`)** — cuOpt's own convention
   already matches HiGHS, which is linopy's (`Highs._solve` passes `solution.row_dual` through
   unchanged, `:1871`). The negation here exists solely because M2 changed the problem we handed
   over; a code comment says so, because this is the single trap most likely to be "fixed" by a
   later reader following I5 literally.
   `maybe_adjust_objective_sign` is a no-op for `io_api="direct"` (`:396`) and is called only for
   consistency with every other `_solve`.
8. Report:
   ```python
   runtime: float | None = None
   with contextlib.suppress(Exception):
       runtime = float(sol.get_solve_time())
   self.io_api = io_api
   return self._make_result(
       status, solution, solver_model=dm, report=SolverReport(runtime=runtime)
   )
   ```
   `contextlib.suppress` around a report metric is the established idiom (`Highs._solve:1887-1892`),
   so an absent accessor degrades to `runtime=None` rather than failing the solve.

**`_set_solver_params(self, settings, log_fn=None) -> None`** — ~18 lines:
```python
        if log_fn is not None:
            self.solver_options["log_file"] = path_to_string(log_fn)
            logger.info(f"Log file at {self.solver_options['log_file']}")

        options: dict[str, Any] = {"method": self._default_method, **self.solver_options}
        if options["method"] == 0:
            logger.warning(
                "cuOpt's concurrent method (method=0) is known to crash on repeated "
                "solves in one process. Consider method=3 (barrier) instead."
            )
        for key, value in options.items():
            try:
                settings.set_parameter(key, value)
            except ValueError as exc:
                if isinstance(value, bool):
                    # cuOpt's integer-typed parameters reject Python bools
                    settings.set_parameter(key, int(value))
                    continue
                raise ValueError(
                    f"cuOpt does not accept the solver option {key!r}={value!r}: {exc}"
                ) from exc
```
Design points (§5f): `log_fn` wins over a user-supplied `log_file`, assigned into
`solver_options` before the loop — verbatim `Highs._set_solver_params` (`:1774-1784`), and the
parameter name is literally the same string. `log_to_console` is **not** touched, matching HiGHS
(which also leaves the console log on when `log_file` is set). The user's `method` always wins
over `_default_method` because `**self.solver_options` is spread second.

#### 2.1.7 `_SOLVER_PROBE_ORDER` (`:4443-4457`) — +1
```python
    "cupdlpx",
    "cuopt",
    "pips",
```
Appended after `cupdlpx`, keeping the "historical eager-probe order, append new entries" comment
honest. Consequence: if a machine has both GPU solvers and no CPU solver, `cuPDLPx` remains the
auto-default. I considered placing `"cuopt"` first on capability grounds (LP+MILP+QP vs LP) and
rejected it: reordering an existing entry's precedence is a behaviour change for existing users
of a list documented as historical, i.e. a drive-by (I4). Log as a `D`-entry.

#### 2.1.8 `QUADRATIC_SOLVERS` (`:4426-4431`) — no edit
Derived from the feature; declaring `QUADRATIC_OBJECTIVE` registers cuOpt automatically. Editing
the list would be wrong. **Zero-line diff** (C-R17 is satisfied by the QP build/solve path).

#### 2.1.9 `_build_file` override — +8
```python
    def _build_file(self, **kwargs: Any) -> None:
        logger.warning(
            "cuOpt does not support file IO. Building the model through the direct "
            "API instead."
        )
        self._build_direct(**kwargs)
```
Why an override of `_build_file` rather than of `_run_file` (cuPDLPx's choice): cuPDLPx's
`_run_file` runs *after* `_build_file` has already written and then discarded an LP file — for a
10⁵-variable model that is 0.23 s and 7.7 MB of pure waste. Intercepting one step earlier gets
the same behaviour with no IO. `_build_direct` sets `self.io_api = "direct"`, so
`Solver.solve`'s dispatch (`:858`) routes to `_run_direct` without further changes, and
`Model.solve`'s `finally` block (`:2145-2147`) simply finds no file to delete. `_run_file` is
consequently unreachable and is **not** overridden — the base implementation's
`NotImplementedError` (`:979-983`) is the correct fallback if it ever is reached.

### 2.2 `linopy/io.py` — +6 / −0

`:33` TYPE_CHECKING block:
```python
if TYPE_CHECKING:
    from cupdlpx import Model as cupdlpxModel
    from cuopt.linear_programming.data_model import DataModel as cuoptDataModel
```
After `to_cupdlpx` (`:792-795`):
```python
def to_cuopt(m: Model) -> cuoptDataModel:
    """Build the cuOpt DataModel for `m`."""
    solver = solvers.cuOpt.from_model(m, io_api="direct")
    return solver.solver_model
```
Byte-for-byte the shape of `to_cupdlpx`.

### 2.3 `linopy/model.py` — +2 / −0

`:73` import list (alphabetical within the `from linopy.io import (...)` block):
`to_cuopt,` before `to_cupdlpx,`. `:2517` binding: `to_cuopt = to_cuopt` next to
`to_cupdlpx = to_cupdlpx`.

### 2.4 `linopy/constants.py` — no edit

`TerminationCondition` (`:171-199`) already carries every value cuOpt needs: `optimal`,
`infeasible`, `unbounded`, `infeasible_or_unbounded`, `time_limit`, `iteration_limit`,
`suboptimal`, `internal_solver_error`, `unknown`. **Zero-line diff.** C-R10 is satisfied by the
mapping table in §4 plus its tests.

### 2.5 `linopy/solver_capabilities.py` — no edit

`SOLVER_REGISTRY` is a `_LazyRegistry` that derives `SolverInfo` from `SolverName` +
`cls.supported_features()` (`:78-100`). Nothing to register; editing it is a drive-by (I4).
**Zero-line diff.** C-R7 is satisfied by a V0 test (§8.2).

### 2.6 `test/conftest.py` — no edit

`pytest_collection_modifyitems` (`:53-79`) is generic on
`solver_supports(solver, SolverFeature.GPU_ONLY)` for any test whose callspec has a `solver`
param; cuPDLPx is not named. Declaring `GPU_ONLY` auto-marks and auto-skips cuOpt tests.
`LINOPY_RUN_GPU_TESTS` is already exported for import-time checks (`:47-50`). **Zero-line diff.**
C-R11 is satisfied by a verification step: run `.venv/bin/pytest -q -k cuopt` (no `--run-gpu`)
and paste output showing the cuOpt items **skipped**, then with `--run-gpu` showing them **run**.

### 2.7 `test/test_optimization.py` — +9 / −0

* `:562` `time_limit_option` dict → add `"cuopt": {"time_limit": 1}` (real snake_case name,
  exploration §7).
* `:~575` `method_options` dict → add `"cuopt": {"method": 1, "log_to_console": False}` — exercises
  a non-default `method` and a bool-valued parameter through `_set_solver_params`, and asserts the
  objective still equals 3.3.
* `:685` `test_infeasible_model` → **no cuOpt branch.** cuOpt returns `PrimalInfeasible` on this
  model (exploration §1), which maps to `infeasible` → `SolverStatus.warning`, satisfying both
  assertions. The cuPDLPx skip is a cuPDLPx hang, not a shared problem. Absence justified here so
  a later reviewer does not "restore symmetry".
* `:1110` `test_basis_and_warmstart` → add
  ```python
      if solver == "cuopt":
          pytest.skip("cuOpt does not support warmstart or basis files.")
  ```
* No change to `CPU_SOL_TOL`/`GPU_SOL_TOL` (`:79-80`): `"cuopt"` lands in `gpu_solvers` via
  `GPU_ONLY`, so every existing tolerance branch picks `GPU_SOL_TOL = 2.5e-4` automatically.
  Re-using the constant #516 added is the point of it existing.

### 2.8 `test/test_solvers.py` — +6 / −0

`test_solver_class_supports_feature` parametrization (`:447-464`) — the exact spot #516's four
lines went:
```python
        (solvers.cuOpt, SolverFeature.DIRECT_API, True),
        (solvers.cuOpt, SolverFeature.GPU_ONLY, True),
        (solvers.cuOpt, SolverFeature.INTEGER_VARIABLES, True),
        (solvers.cuOpt, SolverFeature.SEMI_CONTINUOUS_VARIABLES, True),
        (solvers.cuOpt, SolverFeature.SOS_CONSTRAINTS, False),
        (solvers.cuOpt, SolverFeature.READ_MODEL_FROM_FILE, False),
```
These run on CPU with no GPU and no cuOpt installed (pure class introspection) — the cheapest
possible guard on the declared feature set. The QP package appends
`(solvers.cuOpt, SolverFeature.QUADRATIC_OBJECTIVE, True)`.

### 2.9 `test/test_cuopt.py` — new, ~+215

The GPU-gated V1 differential suite. See §8.

### 2.10 `test/test_cuopt_keyboard_interrupt.py` — new, ~+35

CPU-only, dummy-driven test of `_run_cuopt_with_keyboard_interrupt`, modelled on
`test/test_highs_keyboard_interrupt.py`. See §8.

### 2.11 `pyproject.toml` — +6 / −0

See §5i.

### 2.12 `doc/gpu-acceleration.rst`, `doc/prerequisites.rst`, `doc/contributing.rst`, `doc/index.rst`, `doc/api.rst`, `doc/release_notes.rst`, `README.md`, `examples/using-solvers.ipynb`

See §9.

---

## 3. Feature-flag set

### 3.1 Declared

| Feature | Justification |
|---|---|
| `DIRECT_API` | `_build_direct` / `_run_direct` are the only path. Required for `("cuopt","direct",False)` to enter `params` (`test_optimization.py:52-56`). |
| `GPU_ACCELERATION` | cuOpt solves on the GPU; T4 verified (charter §4.1/§4.3). |
| `GPU_ONLY` | The in-process library **cannot** solve without a GPU: with `CUDA_VISIBLE_DEVICES=''` a tiny LP fails with `InputValidationError: A CPU-memory problem requires remote execution` (exploration §13). Also the mechanism by which `conftest.py:53-79` auto-skips and auto-marks cuOpt tests — without it, `pytest` without `--run-gpu` would try to solve on the GPU. |
| `SOLUTION_FILE_NOT_NEEDED` | No solution file is parsed; `Model.solve` then passes `solution_fn=None` unless `keep_files=True` (`model.py:2098-2106`). |
| `INTEGER_VARIABLES` | MILP verified (charter §4.3, exploration Q0/§1); puts cuOpt in `feasible_mip_solvers`. |
| `SEMI_CONTINUOUS_VARIABLES` | Verified supported, on a discriminating model, on both entry points (exploration §10: `vtype='S'` → obj `0.0` where `'C'`/`'I'` give `5.0`). Costs one `np.where` because `M.vtypes` already emits `'S'`. **Copying cuPDLPx's semi-continuous `NotImplementedError` (`:4249-4253`) would be a false claim** — exploration is explicit about this. Without the flag, `Solver._validate_model` (`:731-737`) would raise `ValueError` for a model cuOpt demonstrably solves. |
| `QUADRATIC_OBJECTIVE` | **Added by the QP work package only.** Verified against HiGHS (exploration §12, obj `-4.28571429` matching). Auto-registers cuOpt in `QUADRATIC_SOLVERS` (`:4426`) and `quadratic_solvers`, and unlocks `Solver._validate_model`'s quadratic gate (`:726-729`). |

### 3.2 Deliberately NOT declared

| Feature | Why not |
|---|---|
| `READ_MODEL_FROM_FILE` | §1.3. Not declared → cuOpt stays out of `file_io_solvers`, so `test_set_files_and_keep_files` / `test_set_files` (`test_optimization.py:643, 664`) skip it, and `_build_file` warn-and-redirects instead. |
| `LP_FILE_NAMES` | cuOpt is fed positionally in the direct path, so names are never written. Not declaring it also means `Solver._build_file` (`:775-780`) warns and disables `explicit_coordinate_names` automatically — free correct behaviour rather than a hand-written guard. |
| `SOS_CONSTRAINTS` | Verified absent: no `sos`/`ordered` attribute anywhere, the parser rejects an `SOS` section, and upstream's own test says "SOS is explicitly out of scope" (exploration §10). `Solver._validate_model` (`:739-744`) then raises the standard actionable `ValueError` naming `reformulate_sos=True`. |
| `INDICATOR_CONSTRAINTS` | Verified absent (exploration §10). `_validate_model` (`:746-752`) raises. |
| `IIS_COMPUTATION` | No IIS API on `DataModel`/`Solution` (exploration §10's full surface dump). |
| `SOLVER_ATTRIBUTE_ACCESS` | No Gurobi-style attribute surface. |
| `MIP_DUAL_BOUND_REPORT` | **No dual-bound accessor was verified in Phase 1.** Declaring a feature we cannot populate is a false capability claim; `SolverReport.dual_bound` would stay `None`. Charter §5 lists it as "possibly" — I resolve that to no. Recorded as a follow-up: if `Solution` exposes a MIP gap / dual bound, declare it then, with pasted evidence. |
| `supports_persistent_update` | `DataModel` has no update API; `Problem.reset_solved_values()` throws the whole model away anyway (exploration Q0). Left at the base `False`, so `Solver._apply_locked` (`:933-937`) rebuilds — the correct behaviour. |
| `supports_sign_update` | Follows from the above. |

---

## 4. Status-mapping table

**Every (problem_category, status) pair.** `Solution._set_termination_status`
(`solution/solution.py:224-237`) picks the LP or MILP enum from `problem_category`, so the same
*integer* means different things (value `2` = `PrimalInfeasible` for LP but `Infeasible` for
MILP; value `3` = `DualInfeasible` vs `Unbounded`). A map keyed on the raw int is therefore
wrong.

**Implementation choice: key the map on `sol.get_termination_status().name`.** The 12 distinct
names across the two enums are either unique to one enum or map to the same
`TerminationCondition` in both, so a single flat `dict[str, TerminationCondition]` is
unambiguous — and it needs no import of the enum classes from an unverified module path. Look-up
is `CONDITION_MAP.get(name, TerminationCondition.unknown)`, so a future cuOpt status degrades to
`unknown` rather than a `KeyError`.

| Category | Value | Status name | `TerminationCondition` | Evidence / note |
|---|---|---|---|---|
| LP | 0 | `NoTermination` | `internal_solver_error` | cuOpt's catch-all for malformed input and for the memory ceiling; empty primal array, `obj=nan`-or-`0`. Every reachable cause is pre-empted by the §2.1.6 guards, so reaching it means cuOpt could not run the problem → `SolverStatus.error`, and `safe_get_solution` correctly returns an empty `Solution()` instead of trying to parse an empty array (exploration §1, §5, §6, §9). |
| LP | 1 | `Optimal` | `optimal` | exploration §1 |
| LP | 2 | `PrimalInfeasible` | `infeasible` | exploration §1 |
| LP | 3 | `DualInfeasible` | `unbounded` | Dual-infeasible LP = primal unbounded, and cuOpt has a *separate* `UnboundedOrInfeasible`, so this is not the ambiguous case. **Never observed** (7 unbounded variants all returned 11) — mapping is by definition, not by measurement; carried in a code comment. |
| LP | 4 | `IterationLimit` | `iteration_limit` | exploration §1 |
| LP | 5 | `TimeLimit` | `time_limit` | `SolverStatus.ok`, and the primal may be all-zero — handled because `safe_get_solution` still calls the parser and linopy reports what cuOpt gave (exploration §1's warning about "solution available"). |
| LP | 6 | `NumericalError` | `internal_solver_error` | non-PSD Q, pathological scaling (exploration §1, §12) |
| LP | 7 | `PrimalFeasible` | `suboptimal` | `first_primal_feasible=True` (exploration §1) |
| LP | 11 | `UnboundedOrInfeasible` | `infeasible_or_unbounded` | exploration §1; what unbounded LPs actually return |
| MILP | 0 | `NoTermination` | `internal_solver_error` | as LP 0; also the MIQP case, which the §2.1.6 guard pre-empts |
| MILP | 1 | `Optimal` | `optimal` | exploration §1 |
| MILP | 2 | `Infeasible` | `infeasible` | exploration §1 |
| MILP | 3 | `Unbounded` | `unbounded` | **Never observed** (11 every time); mapping by definition |
| MILP | 5 | `TimeLimit` | `time_limit` | exploration §1 |
| MILP | 8 | `FeasibleFound` | `suboptimal` | exploration §1 |
| MILP | 11 | `UnboundedOrInfeasible` | `infeasible_or_unbounded` | exploration §1 |
| — | — | anything else | `unknown` | `.get()` default |

`ErrorStatus` (`0 Success`, `4 ValidationError`, `5 OutOfMemoryError`, `6 RuntimeError`) is **not**
mapped onto a `TerminationCondition`. It is read alongside: a non-empty `get_error_message()` is
logged via `logger.error`. `status.legacy_status` carries the status *name*, mirroring
`Highs._solve`'s `h.modelStatusToString(condition)` (`:1859`).

Two behavioural notes the implementer must not lose:
* `time_limit`/`iteration_limit` **can still return `Optimal`** — never infer a limit hit from the
  settings (exploration §1).
* Nothing new is added to `TerminationCondition` (charter §5 row 10 satisfied with a zero-line
  diff).

---

## 5. Design decisions on the hard findings

### 5a. Presolve + maximize negated duals → adopt M2 (always minimise)

**Decision: adopt M2 unconditionally. Never call `set_maximize(True)`.** For a `max` model,
build with `c → -c` (and `Q → -Q`, §5d), leave the sense at minimise, and negate the returned
objective and duals in `_solve`.

Why M2 and not M1 (`presolve=0/1`):
* M1 is verified **incomplete**: it fixes the equality cells but the ranged-row cells still land
  on `solved_by=Unset` and still return negated duals (exploration §2b).
* `presolve=0` makes the Concurrent segfault happen **one solve earlier** (exploration §9), so
  M1 interacts badly with §5b.
* `presolve=0` also disables presolve wholesale — a real performance cost imposed on every user
  to work around a sign bug.
* M2 is verified **21/21** against live HiGHS across 2-var/3-var × `<=`/`>=`/`==` × ranged ×3 ×
  QP ×4, versus 5/21 failures without it (exploration §2b).
* M2 keeps linopy off the buggy postsolve branch **by construction**, so it is robust to the
  unknown root cause (exploration's Unknowns: it was not determined whether the defect also
  affects a non-presolve path that was not constructed).

Refinement worth recording: `M.sense` only ever holds `<`, `>`, `=` (`matrices.py:118-127`), so
**a linopy model cannot produce a ranged row** — the ranged-row half of M2's evidence is
belt-and-braces rather than load-bearing. M2's load-bearing evidence is the reachable one:
`a_02f_linopy_reach.py` drove real `linopy.Model()`s (n = 4, 10, 40, 200, square equality
systems, `max`) to `max|Δdual| = 3.61` with a correct objective and status `Optimal`. That is the
case linopy users will hit.

Consequences that must not be forgotten:
* `maybe_adjust_objective_sign` is a **no-op for `io_api="direct"`** (`:396`), so the objective
  negation must be explicit in `_solve` — it is.
* Do **not** copy cuPDLPx's `if MAXIMIZE: dual = -dual` (`:4385-4386`). cuOpt's convention already
  matches HiGHS. The negation in `_solve` is M2's inverse transform and is commented as such.
* `set_objective_offset` is irrelevant (linopy forbids objective constants), so M2 needs no
  offset negation.

### 5b. Concurrent repeat-solve segfault → default `method=3` (Barrier)

**Decision: linopy defaults `method` to `3` (Barrier), user-overridable; warn if the user
selects `method=0`; add a repeat-solve regression test above the crash threshold.**

Facts (exploration §9): `method=0` (cuOpt's default, Concurrent) **hard-crashes with SIGSEGV** on
the 2nd–3rd sequential `Solve()` for LPs above n ≈ 1300, deterministically, reproduced in ≥12
fresh processes, insensitive to `num_cpu_threads=1`/`num_gpus=1`, and *worse* with `presolve=0`.
Methods 1 (PDLP), 2 (DualSimplex) and 3 (Barrier) all completed 25/25 varied solves with flat
device memory. Doing nothing is not defensible: linopy's test models are all under the threshold,
so the suite would stay green while a real user's second `model.solve("cuopt")` kills the
interpreter with no traceback.

Why Barrier over the two other stable options:
* **Accuracy.** Barrier's duals agree with HiGHS to **1.73e-09 – 2.47e-09**; PDLP reaches only
  **2.60e-06 – 4.71e-05** and drifts the primal objective (`2.80013079` vs `2.8`), and its
  default `1e-4` tolerances moved an objective by **5.52e-03 relative** on a 2000×1000 LP
  (exploration §2c, §7). Choosing PDLP would force either a loose frozen `rtol` or a
  linopy-imposed tolerance override — both worse interventions than choosing the method.
* **It is the method cuOpt itself trusts for the hardest class.** Every QP is solved by Barrier
  regardless of the requested `method` (exploration §12) — so `method=3` is the setting cuOpt
  already forces where it cares most.
* **DualSimplex is rejected** on principle: a solver declaring `GPU_ONLY` must not default to
  cuOpt's CPU dual simplex.
* Barrier's GPU backing is inferred, not measured (`nvidia-cudss-cu12` is in the dependency
  tree), but the decisive point is that the in-process library cannot solve at all without a
  GPU under *any* method (exploration §13), so Barrier is not a silent CPU fallback.

Implementation: `_default_method: ClassVar[int] = 3`, applied via
`{"method": self._default_method, **self.solver_options}` so a user setting always wins, plus a
`logger.warning` naming the hazard when the resolved method is `0`. Never refuse `method=0` — the
vendor default must stay reachable.

**Named contingency (pre-decided, so no agent has to improvise).** `method`'s interaction with
the MILP branch-and-bound path was not exercised in Phase 1. If WP-A's MILP tests fail or
`Solve` rejects `method=3` for a MILP, the coding agent **reports the failure verbatim** and
applies exactly this fallback: pass the default only when
`not _cuopt_has_integrality(dm.get_variable_types())`, and document the MILP repeat-solve hazard
in `doc/gpu-acceleration.rst`'s limitations list. It does not invent a third option.

### 5c. `is_available()` — import probe, with the GPU check moved to `_license_probe`

**Decision (an explicit, partial divergence from charter §4.6 constraint 3):**

```python
    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("cuopt")
```

**What happens on a GPU-less machine with `cuopt-cu12` installed:**

| Call | Result |
|---|---|
| `cuOpt.is_available()` | `True` |
| `linopy.available_solvers` | **contains** `"cuopt"` |
| `cuOpt.license_status()` | `LicenseStatus("cuopt", ok=False, message="RuntimeError: cuOpt probe solve returned ...")` — the message carries cuOpt's own text |
| `linopy.licensed_solvers` | **omits** `"cuopt"` |
| `check_solver_licenses("cuopt")` | reports the failure |
| `model.solve("cuopt")` | raises `RuntimeError("cuOpt found no usable GPU. cuOpt requires an NVIDIA GPU with compute capability >= 7.0 and driver >= 525.60.13, and cannot solve on CPU. Use a CPU solver (e.g. 'highs') instead.")`, chained from cuOpt's `InputValidationError` — **never a bare CUDA traceback** |
| `pytest` / `pytest --run-gpu` | cuOpt is absent from `params`, because `test_optimization.py` parametrizes on `licensed_solvers` (`:40-56`), not `available_solvers` |

`_license_probe` is the real GPU probe (~12 lines): build a 1-variable / 1-row `DataModel`
(`min x s.t. x >= 1`), a `SolverSettings` with `log_to_console=False`, call `solver.Solve`, and
raise `RuntimeError` unless the status name is `Optimal`. It mirrors
`cuPDLPx._license_probe` (`:4191-4193`), which likewise constructs a native model rather than
merely importing.

Why `is_available()` stays an import probe:
1. **Its contract is documented and shared.** `_AvailableSolvers`' docstring (`:4464-4470`):
   *"Membership means the solver's Python package or binary is importable — it does **not** mean a
   working license exists. Call `check_solver_licenses` for an opt-in eager license probe."*
   Making one solver's membership mean something else breaks a documented invariant of a public
   API for every reader of that list. `test_available_solvers.py:39` asserts
   `cls.is_available() == (sn.value in available_solvers)` for every solver — the two are one
   concept by design.
2. **Current `master` already has the right surface, and it post-dates PR #516.** `LicenseStatus`,
   `license_status()`, `_license_probe()` and `licensed_solvers` exist precisely to answer "is
   this solver *usable* right now", and their docstring says the probe "may acquire a license
   slot" — i.e. it is allowed to be expensive and to touch the runtime. Charter I5: master wins.
3. **A device probe inside `is_available()` has an import-time cost for CPU users.**
   `is_available()` is called on first access to `available_solvers`, i.e. during a plain
   `import linopy; linopy.available_solvers`. Any honest device check initialises some part of the
   CUDA runtime; putting it there taxes users who merely happen to have the wheel installed.
4. `_has_module("cuopt")` uses `importlib.util.find_spec` and does not execute
   `cuopt/__init__.py` at all, which is what keeps the probe genuinely cheap.

**What §4.6 constraint 3 asks that this does not deliver:** the literal sentence
"`linopy.available_solvers` must simply omit cuOpt". I judge the second half of that constraint
(the actionable message) to be the part with user impact and I satisfy it exactly; the first half
I satisfy through `licensed_solvers`. Exploration flagged this as "a real I5-vs-§4.6 fork for
Gate 3", so a reasoned position either way is what Gate 3 needs.

**Concrete alternative, if the lead rules §4.6 constraint 3 literally binding** (~10 lines, one
decision line, no other change to this plan):
```python
    @classmethod
    def _gpu_visible(cls) -> bool:
        try:
            from cuda.bindings import runtime
            err, count = runtime.cudaGetDeviceCount()
            return err == 0 and count > 0
        except Exception:
            return False

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        return _has_module("cuopt") and cls._gpu_visible()
```
(`cuda-bindings` is a verified member of the cuOpt dependency tree — exploration §13's install
transcript lists `cuda-bindings-12.9.7`.) I recommend against it for reasons 1–3, and note it
would also make `test_available_solvers.py`'s import-time contract solver-dependent.

### 5d. QP — `0.5 · M.Q`, full symmetric, CSR; MIQP/IQP rejected; non-PSD is an error

**The convention.** cuOpt minimises `cᵀx + xᵀQx` with effective Hessian `H = Q + Qᵀ`
(`data_model.py:287-288`, verified two independent ways plus live against HiGHS).
`linopy.expressions.QuadraticExpression.to_matrix` documents its own convention as
`0.5 xᵀQx` with doubled diagonal, i.e. **`model.matrices.Q` *is* the Hessian `H`** — which is why
`Highs` passes `triu(M.Q)` to `passHessian` (`:1716-1720`). Therefore:

```python
        Q = M.Q
        if Q is not None:
            if model.objective.sense == "max":
                Q = -Q            # M2: the model is handed over in minimisation form
            Qc = csr_matrix(0.5 * Q)
            dm.set_quadratic_objective_matrix(Qc.data, Qc.indices, Qc.indptr)
```
**Kept full-symmetric, halved, converted to CSR.** Verified: `0.5 * M.Q` reproduces HiGHS's
`-4.28571429` and `[1.571429, -0.142857, -2.0]`, while `M.Q` returns `Optimal` with
`-2.14285714` and `triu(0.5*M.Q)` returns `Optimal` with `-4.25806452` — the trap is real and
silent. Triangular forms are accepted by cuOpt too (only `Q + Qᵀ` is used), but full-symmetric is
the encoding that was verified against linopy's own matrix and needs no `triu`/`tril` reasoning.
`scipy.sparse.csr_matrix` is not currently imported in `solvers.py` (only `tril`, `triu` at
`:34`); the QP package extends that import line.

**MIQP guard.** `model.type in ("MIQP", "IQP")` → `NotImplementedError` in `_build_direct`
(§2.1.6). MIQP does not raise in cuOpt: it returns `NoTermination`, `obj=0.0`, an **empty**
solution array and `error_status=4` with
`"Presolve does not support optimization_problem with a quadratic objective"`. `"IQP"` matters:
`model.type` yields it for an all-integer quadratic model (`model.py:1666-1679`), and exploration
names only `"MIQP"`. `QUADRATIC_OBJECTIVE` is therefore declared **without** implying MIQP.

**Non-PSD Q.** cuOpt detects it and returns `NumericalError` with `obj=nan` — it does not crash
and does not return a bogus answer. Mapped to `internal_solver_error` (§4) →
`SolverStatus.error` → `safe_get_solution` returns an empty `Solution()`. Additionally, when the
model is quadratic and the status name is `NumericalError`, log
`logger.error("cuOpt reported a numerical error. For a quadratic objective this usually means the Hessian is not positive semi-definite.")`
— one line, actionable, and the honest thing to say. Under M2 the requirement inverts correctly
for `max` (verified: `max, c=[2], Q=[[-1]]` → `Optimal`; `Q=[[+1]]` → `NumericalError`).

**Other QP facts to respect:** duals *are* returned under QP and match HiGHS in all six sign
cells; non-binding QP duals come back at ~1e-10 rather than exactly 0, so the dual tolerance
needs an `atol` (§8.4); `lp_stats['dual_residual']` is meaningless under QP (51.0 on an optimal
model) and is never surfaced; `method` and `crossover` are silently overridden to Barrier for QP,
so no special handling. Quadratic **constraints** (`add_quadratic_constraint`) are a non-goal —
linopy has no model surface for them.

### 5e. Reduced costs — do nothing, and say why in one line

**Decision: never call `Solution.get_reduced_cost()`.** `grep -rn
"reduced_cost\|ReducedCost\|reduced cost" linopy/` returns no hits and
`linopy.constants.Solution` (`:274-286`) carries only `primal`, `dual`, `objective` — there is
nowhere to put them. Independently, cuOpt's values are **wrong** for maximised models with `<=`
rows on the default path (returning `−c − Aᵀy` instead of `c − Aᵀy`), and exploration could not
reduce the discrepancy to a single transformation across `max`/`<=` and `max`/`>=`.

The only artefact: a one-line comment where duals are read (§2.1.6 step 7), so that a future
reader adding reduced costs finds the warning instead of guessing a sign. No new public surface,
no `SolverReport` field, no docs promise. `doc/gpu-acceleration.rst` states plainly that cuOpt
returns duals for continuous problems and that linopy does not expose reduced costs for any
solver.

### 5f. Solver options — no blind coercion, retry bools, wrap the nameless error

1. **Pass-through, with the standard `for k, v in options.items(): settings.set_parameter(k, v)`
   loop** — the idiom of every other solver in the file.
2. **bool → int: retry, never guess.** `set_parameter("presolve", False)` raises
   `ValueError: Parameter presolve value False is not an integer`, while
   `set_parameter("log_to_console", False)` is accepted. Phase 1 did **not** establish whether
   `log_to_console=1` is accepted, so coercing every bool up front would be an unverified
   behaviour change on the one parameter class we know accepts bools. The retry (§2.1.6
   `_set_solver_params`) is correct without needing that fact: bool-typed parameters succeed on
   the first attempt, int-typed parameters succeed on the retry. Six lines, no guessing.
3. **Unknown parameter *name*.** `set_parameter` guards eagerly on membership of `solver_params`
   (`solver_settings.pyx:191-193`) and raises `ValueError: Invalid parameter. Please check
   documentation` — a message that **does not name the offending parameter**. Names are
   case-sensitive snake_case, so a user habitually writing `TimeLimit` (Gurobi/cuPDLPx style)
   gets a message they cannot act on. We re-raise as
   `ValueError(f"cuOpt does not accept the solver option {key!r}={value!r}: {exc}")` with
   `from exc`. This is the "consider wrapping it" exploration explicitly recommended, and the
   only place we add message text rather than pass cuOpt's through.
4. **Unknown/invalid *value*.** Raised late, at `Solve()`, and cuOpt's own message already names
   both parameter and value (`Parameter time_limit value -5.0 out of range`). Left untouched — no
   wrapping, no re-interpretation.
5. **`log_file` / `log_fn`.** `log_fn` wins, assigned into `self.solver_options` before the loop —
   verbatim `Highs._set_solver_params` (`:1779-1784`), and the parameter name is literally the
   same string, so no OS-level `dup2` redirect is needed (exploration §4 measured
   `redirect_stdout` capturing 0 chars and `log_file` capturing 1437). `path_to_string(log_fn)`
   even though cuOpt accepts a `Path`, for consistency with the rest of `solvers.py`. Semantics
   to document: the file is **truncated** per solve, not appended; an unwritable path raises from
   `Solve()` with an actionable message.
6. **`log_to_console` is not touched.** cuOpt writes to both file and console when `log_file` is
   set; so does HiGHS. Silencing the console implicitly would diverge from the repo's only
   same-named precedent. Documented in the class docstring as the way to quieten cuOpt.
7. **Nothing is set implicitly except `method`** (§5b). In particular: no default `time_limit`
   (a solver silently giving up is worse than one that runs), no tolerance overrides, and never
   `save_best_primal_so_far` (which segfaults in combination with `method=1` and a tiny
   `time_limit`).

### 5g. Interruptibility — adopt the HiGHS worker-thread pattern (minus the cancel step)

**Decision: adopt.** `solver.Solve()` is not KeyboardInterrupt-safe: exploration §8 measured
SIGINT at t=5.00 s surfacing as `KeyboardInterrupt` at t=57.92 s — **52.9 s of
unresponsiveness** — and confirmed there is no cancel/abort hook anywhere on the in-process LP
solver (only the gRPC client has `cancel`, and §4.2 rules that out). The worker-thread pattern
gives **0.00 s** latency, and the process exits cleanly and immediately even with the daemon
thread still inside cuOpt's C++ code.

The helper (§2.1.5) mirrors `_run_highs_with_keyboard_interrupt` (`:189-252`) in placement,
naming and shape, minus: the `HandleKeyboardInterrupt`/`HandleUserInterrupt` toggles (no
analogue), the `cancelSolve()` call (no analogue), and the post-interrupt
`while not finished.wait(0.1)` drain (deliberately dropped — draining would restore the 52.9 s
hang the helper removes, and exploration verified interpreter teardown is clean). Its docstring
says plainly that the GPU work continues until the process exits and that `time_limit` is the
only real bound. Taking a `Callable` rather than a `DataModel` makes it testable on CPU with a
dummy, mirroring `test/test_highs_keyboard_interrupt.py`.

Cost: ~26 lines + a ~35-line CPU test. Justified: it is the difference between Ctrl-C working
and a minute-long hang on every long solve, and it is repo precedent (I5).

### 5h. Warm start — defer, with a hard prohibition

**Decision: `warmstart_fn is not None` raises `NotImplementedError("Warmstarting is not yet
implemented for cuOpt.")`, exactly as cuPDLPx does (`:4352-4355`). `basis_fn` warns and is
ignored. `DataModel.set_initial_primal_solution` and `set_initial_dual_solution` are never
called, on any path.**

Reasons, in order:
1. **The linopy surface is a file path; cuOpt's warm start is an in-memory object.**
   `PDLPWarmStartData` is 17 host numpy arrays and scalars with no file format. Serialising it
   would be inventing a linopy-private format — new public surface, out of scope.
2. **Three non-default settings must line up simultaneously** for it to do anything:
   `method=1 (PDLP)`, `pdlp_solver_mode=1 (Stable2)`, `presolve=0`. Each is a material change:
   PDLP is the least accurate method (§5b), `presolve=0` makes the Concurrent segfault worse, and
   the default `pdlp_solver_mode=Stable3` silently produces `obj=nan` with warm-start data
   attached. Forcing all three behind a `warmstart_fn` argument would silently swap the solver
   the user asked for.
3. **The payload comes back in presolved coordinates**, so with the default `presolve=-1` it
   cannot be fed back into the *identical* model (`Invalid PDLPWarmStart data...`). That is an
   upstream defect, not something linopy can paper over.
4. **`set_initial_primal_solution` on an LP raises `cudaErrorIllegalAddress` and poisons the CUDA
   context for the rest of the process** — reproduced in a fresh process with all-default
   settings and correctly-sized C-contiguous `float64` input. One warm-start attempt takes down
   an entire pytest session or user script. This is why the prohibition is absolute and carries a
   code comment, not merely an omission.
5. PDLP warm start is LP-only anyway (`get_pdlp_warm_start_data` raises for a MILP solution).

Tested (§8.3): `warmstart_fn` raises `NotImplementedError`; `basis_fn` warns and still solves;
`test_basis_and_warmstart` skips cuOpt. Documented under "Limitations" with the three-setting
requirement stated, so a future contributor knows exactly what wiring it up would cost.

### 5i. The pyproject extra — `cuopt`, sibling of `solvers`, Linux marker

**Decision: adopt exploration item 13's verified candidate verbatim**, inserted immediately after
the `solvers = [...]` block (`pyproject.toml:113`):

```toml
# NVIDIA cuOpt (GPU). Deliberately NOT part of `solvers`: it pulls ~34 packages
# and several GB of CUDA wheels. Linux-only wheels, no sdist, so the marker keeps
# macOS/Windows resolution unaffected. Requires a CUDA 12 driver (>= 525.60.13)
# and a GPU of compute capability >= 7.0.
cuopt = [
    "cuopt-cu12>=26.8; platform_system == 'Linux'",
]
```

* **Name `cuopt`**, not `gpu`. It matches the solver name, the `SolverName` value and the file's
  existing lower-case single-word extras (`oetc`, `remote`, `docs`, `dev`, `solvers`,
  `benchmarks`). `gpu` reads as a category but is a promise the extra cannot keep — it installs
  one vendor's Linux-only CUDA-12 solver, and a future second GPU solver would want its own
  extra. Charter §5 writes `<gpu-extra>` as a placeholder, not a name.
* **Marker `platform_system == 'Linux'`** — the positive twin of the file's established
  `"cplex; platform_system != 'Darwin'"` (`:105`). It must be an equality, not `!= 'Darwin'`,
  because cuOpt has no Windows wheels either.
* **No `python_version` marker**: cuOpt ships cp311–cp314 and linopy already declares
  `requires-python = ">=3.11"`.
* **No `[tool.uv]` index entry**: the whole tree is on ordinary PyPI (§4.2), and index settings
  in `pyproject.toml` are workspace-local and never reach a downstream `pip install linopy[cuopt]`
  anyway (§4.6).
* **Floor `>=26.8`**: `26.6.0` has no cp311 wheel. No upper bound, matching the file's style for
  `gurobipy`/`mosek`.
* **Reachable from no other extra.** There is no `all` extra in this project; `solvers`, `dev`
  and `docs` are untouched. Verified: `pip install 'linopy[solvers]'` pulls 34 packages with
  *no* cuOpt/cuDF/CUDA (exploration §13 step 3).
* **Cross-platform resolution verified**: `uv pip compile --python-platform windows|macos
  --extra cuopt` both resolve with cuOpt absent; `linux` resolves 66 packages including
  `cuopt-cu12==26.8.0` (exploration §13 step 4).
* **I9's end date.** Once this lands, `uv sync --extra cuopt` becomes the correct command and
  keeps cuOpt installed — which is itself the extra's smoke test. `doc/contributing.rst` gets that
  sentence (§9).

---

## 6. Requirement-ID map

| ID | §5 map row | Satisfied by | WP |
|---|---|---|---|
| **C-R1** | `solvers.py:148` `SolverFeature` | §3 feature frozenset on `class cuOpt`; **zero-line diff to the enum** (every needed member exists). Declared: `DIRECT_API`, `GPU_ACCELERATION`, `GPU_ONLY`, `SOLUTION_FILE_NOT_NEEDED`, `INTEGER_VARIABLES`, `SEMI_CONTINUOUS_VARIABLES`; `+ QUADRATIC_OBJECTIVE` in WP-F. Proven by `test_solvers.py` rows (§2.8). | A, F |
| **C-R2** | `solvers.py:176` `TYPE_CHECKING` | `import cuopt` added alphabetically (§2.1.2) | A |
| **C-R3** | `solvers.py:304` `_LazyModule` | `cuopt = _LazyModule("cuopt.linear_programming")` + the `cuopt.routing` comment (§2.1.3). Deliberate refinement of the charter's literal `_LazyModule("cuopt")`, with a pasted resolution check and a named fallback. | A |
| **C-R4** | `solvers.py:346` `SolverName` | `cuOpt = "cuopt"` appended; member name == class name (§2.1.4) | A |
| **C-R5** | `solvers.py:4156` `class cuPDLPx` (template) | `class cuOpt(Solver[None])` after cuPDLPx: `display_name`, `features`, `accepted_io_apis`, `_default_method`, `is_available`, `_license_probe`, `_build_file`, `_build_direct`, `_build_solver_model`, `_run_direct`, `_solve`, `_set_solver_params` (§2.1.6) | A |
| **C-R6** | `solvers.py:4440` `_SOLVER_PROBE_ORDER` | `"cuopt"` appended after `"cupdlpx"` (§2.1.7) | A |
| **C-R7** | `solver_capabilities.py` `SOLVER_REGISTRY` | **Zero-line diff.** Proven by a V0 test asserting `SOLVER_REGISTRY["cuopt"].features == solvers.cuOpt.supported_features()`, `.display_name == "cuOpt"`, and `_solver_class_for("cuopt") is solvers.cuOpt` (§8.2) | C |
| **C-R8** | `io.py:33`, `:792` | `TYPE_CHECKING` import + `to_cuopt(m)` mirroring `to_cupdlpx` (§2.2) | A |
| **C-R9** | `model.py:73`, `:2517` | import + `to_cuopt = to_cuopt` binding (§2.3) | A |
| **C-R10** | `constants.py:171` `TerminationCondition` | **Zero-line diff**; the §4 mapping table + its status tests (§8.3) | A, C |
| **C-R11** | `test/conftest.py:57-79` | **Zero-line diff**; verified by pasting `pytest -k cuopt` (skipped) vs `pytest -k cuopt --run-gpu` (run) (§2.6) | C |
| **C-R12** | `test/test_optimization.py:562, 685, 1110` | `time_limit_option` + `method_options` entries, no branch at `:685` (justified), warmstart/basis skip at `:1110` (§2.7) | C |
| **C-R13** | `test/test_solvers.py` | six `test_solver_class_supports_feature` rows (§2.8), + one in WP-F | C, F |
| **C-R14** | `pyproject.toml` | the `cuopt` extra (§5i), acceptance-tested by a clean-venv install (§7 WP-B) | B |
| **C-R15** | `doc/gpu-acceleration.rst` | new `cuOpt` section: install, hardware/driver, usage, supported classes, limitations (§9) | D |
| **C-R16** | `doc/prerequisites.rst`, `contributing.rst`, `index.rst`, `api.rst`, `release_notes.rst`, `README.md` | §9's table, row by row. `api.rst` gains `solvers.cuOpt`; `to_cuopt` is deliberately **not** added because no `to_*` helper is listed there (justified in §9) | D, E |
| **C-R17** | `solvers.py:4426` `QUADRATIC_SOLVERS` | **Zero-line diff**; satisfied by the QP build/solve path, `QUADRATIC_OBJECTIVE`, the MIQP/IQP guard and the QP tests (§5d, §8) | F |

### 6.1 New requirements this plan implies

| ID | Requirement | Satisfied by | WP |
|---|---|---|---|
| **C-R18** | Every `max` model is handed to cuOpt in minimisation form (M2): `c → -c`, `Q → -Q`, `set_maximize` never called `True`; objective and duals negated on return | §5a; `_build_solver_model` + `_solve`; V1 six-cell matrix **plus** the presolve-solvable `max` cases (§8.3) | A, F |
| **C-R19** | cuOpt's crash-prone default method is not linopy's default; the hazard is regression-tested | §5b; `_default_method = 3`, `method=0` warning, subprocess repeat-solve test at n=2000 (§8.3) | A, C |
| **C-R20** | Honest degradation without a usable GPU | §5c; `_license_probe` GPU probe, `CUOPT_REMOTE_HOST` translation, `licensed_solvers` omission | A |
| **C-R21** | Ctrl-C interrupts a cuOpt solve promptly | §5g; `_run_cuopt_with_keyboard_interrupt` + CPU dummy test | A, C |
| **C-R22** | cuOpt's silent-failure modes cannot reach the user | §2.1.6 guards: `A is None`, `A.nnz == 0`, width mismatch, vtype whitelist, `'B'→'I'`, MIQP/IQP; tests in §8.3 | A, F |
| **C-R23** | Solver options fail loudly and informatively | §5f; bool retry, named unknown-option error, `log_fn` precedence; tests in §8.3 | A, C |
| **C-R24** | Warm start, basis files, solution files and reduced costs are refused or omitted with actionable messages, and `set_initial_primal_solution` is never called | §5e, §5h; tests in §8.3 | A, C |
| **C-R25** | A non-direct `io_api` request still solves (DoD: bare `model.solve("cuopt")`) | §2.1.9 `_build_file` override; test in §8.3 | A, C |

No `C-R*` ID is unmapped, reinterpreted, or satisfied by "documentation only" except C-R7,
C-R10, C-R11 and C-R17, each of which is a charter-declared zero-diff row and each of which gets
an executable check.

---

## 7. Work-package breakdown

Rules honoured: `solvers.py` core first and alone (§6.4 rule 1); disjoint parallel packages after
(rule 2); no two concurrent packages share a file (rule 3); QP last (rule 4); packaging is
first-class and early (rule 5).

| WP | Files (exclusive) | Requirement IDs | Runs | Commit |
|---|---|---|---|---|
| **WP-A** core | `linopy/solvers.py`, `linopy/io.py`, `linopy/model.py` | C-R1–R6, R8, R9, R10, R18–R25 | first, alone | 1 |
| **WP-B** packaging | `pyproject.toml` | C-R14 | in parallel with WP-A (disjoint file) | 2 |
| **WP-C** tests | `test/test_optimization.py`, `test/test_solvers.py`, `test/test_cuopt.py`, `test/test_cuopt_keyboard_interrupt.py` | C-R7, R11, R12, R13, and the test half of R19–R25 | after WP-A merges | 3 |
| **WP-D** docs | `doc/gpu-acceleration.rst`, `doc/prerequisites.rst`, `doc/contributing.rst`, `doc/index.rst`, `doc/api.rst`, `README.md`, `examples/using-solvers.ipynb` | C-R15, R16 (part) | in parallel with WP-C (disjoint files) | 4 |
| **WP-E** release notes | `doc/release_notes.rst` | C-R16 (part) | in parallel with WP-C/WP-D (disjoint file) | 5 |
| **WP-F** QP | `linopy/solvers.py`, `test/test_cuopt.py`, `test/test_solvers.py`, `doc/gpu-acceleration.rst`, `doc/release_notes.rst` | C-R1 (QP flag), R17, R18 (Q half), R22 (MIQP half) | **last**, after A–E have all merged | 6 |

Six substantive commits, one concern each, 1–5 files apiece — the shape of PR #516's real feature
arc. Message style: plain imperative, sentence case, no conventional prefix (D2).

### WP-A — cuOpt solver class and IO helpers
**Commit message:** `Implement interface to cuOpt solver and required IO`
**Acceptance criteria** (each needs pasted output):
1. `.venv/bin/python -c "from linopy.solvers import cuopt; print(cuopt.data_model.DataModel, cuopt.solver.Solve, cuopt.solver_settings.SolverSettings)"` resolves (§2.1.3), or the named fallback is applied and reported.
2. `"cuopt" in linopy.available_solvers` and `"cuopt" in linopy.licensed_solvers` on this T4.
3. `linopy.solver_capabilities.SOLVER_REGISTRY["cuopt"].features == linopy.solvers.cuOpt.supported_features()`.
4. A hand-run LP: `min 2y + x s.t. 2x+6y>=10, 4x+2y>=3` solved with `model.solve("cuopt", io_api="direct")` and with `model.solve("highs")` in the same process; objectives agree within `rtol=1e-6, atol=1e-8`, duals within `rtol=1e-5, atol=1e-6`.
5. The same model with `sense="max"` — duals agree in **sign and value** with HiGHS.
6. A 4-variable square-equality `max` model (exploration's `a_02f` shape) — duals agree with HiGHS and `sol.get_solved_by().name == "Unset"` is confirmed, proving the M2 path is exercised.
7. A MILP: objective agrees with HiGHS; `model.dual` is empty and no exception is raised.
8. A semi-continuous model (`lower=1, upper=10, semi_continuous=True`, `x <= 0.5`, `max x`) → objective `0`, agreeing with HiGHS.
9. `model.solve("cuopt")` **with no `io_api`** succeeds, emits the `_build_file` warning once, and writes no LP file.
10. `model.solve("cuopt", log_fn=tmp/"cuopt.log")` → the file exists and contains `cuOpt version`.
11. `model.solve("cuopt", TimeLimit=1)` raises `ValueError` naming `'TimeLimit'`; `model.solve("cuopt", presolve=False)` succeeds via the bool retry.
12. `/home/azureuser/.local/bin/ruff check .` and `.venv/bin/mypy` clean; `.venv/bin/pytest -q` and `.venv/bin/pytest --run-gpu -q` show **no regression** vs `logs/05-baseline*`.
**Explicitly out of scope for WP-A:** anything quadratic (no `set_quadratic_objective_matrix`
call, no `QUADRATIC_OBJECTIVE` flag) — but the MIQP/IQP guard **is** in WP-A, so a quadratic
model reaching the class raises the base `_validate_model` `ValueError` first, which is correct.

### WP-B — packaging
**Commit message:** `Add cuOpt package dependency`
**Acceptance criteria:**
1. `uv pip compile --no-cache --python-platform {linux,macos,windows} --python-version 3.11 --extra cuopt pyproject.toml` — all three exit 0; `cuopt-cu12` appears only for linux. Paste all three.
2. `pip install --dry-run --report` for `linopy[solvers]` in a clean venv shows **no** package matching `cuopt|cudf|libcuopt|rmm|pylibraft|nvidia|cuda`.
3. A **clean venv that has never seen cuOpt**: `pip install '<worktree>[cuopt]'` with **no** extra index and **no** resolver flags, exit 0, then `python -c "import cuopt; print(cuopt.__version__)"`. Paste both. (Q13 already proved this at `09c34dd`; the WP re-proves it against the committed tree, then deletes the venv.)
4. Disk: the temporary venv is removed and `df -h /` pasted afterwards.
**Note:** WP-B must **not** re-run `linopy.available_solvers` as its own acceptance test —
`"cuopt"` only appears there once WP-A has merged. That check belongs to WP-C.

### WP-C — tests
**Commit message:** `Update tests`
**Acceptance criteria:** the whole of §8, with pasted `pytest` output for
`.venv/bin/pytest -q`, `.venv/bin/pytest --run-gpu -q`, and
`.venv/bin/pytest --run-gpu -q test/test_cuopt.py -v` (the last showing every test **ran**, not
skipped); plus the conftest skip/run pair for C-R11; plus ruff and mypy clean.

### WP-D — documentation
**Commit message:** `Add cuOpt to the documentation`
**Acceptance criteria:** every row of §9 present; docs built in the cuOpt-free venv
(`PATH=/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin:$PATH`,
`.venv-docs/bin/sphinx-build -b html doc doc/_build/check`) with **no new warning** versus
`logs/05-baseline-docs.txt` — pasted diff of the warning sets. `examples/using-solvers.ipynb`
gains a **markdown-only** mention (no executable cuOpt cell — the notebook executes during the
docs build on a GPU-less machine).

### WP-E — release notes
**Commit message:** `Add comment for release notes`
**Acceptance criteria:** one bullet under `Upcoming Version`, naming the extra, the supported
problem classes, and the Linux/driver requirement; docs build still clean.

### WP-F — quadratic objectives
**Commit message:** `Add support for quadratic objectives`
**Dispatched only after WP-A–E have merged and Gate 4's checks pass on the LP/MILP core** (§4.5).
**Acceptance criteria:**
1. `SolverFeature.QUADRATIC_OBJECTIVE` declared; `"cuopt" in linopy.quadratic_solvers` (proves the
   derived `QUADRATIC_SOLVERS` registration, C-R17) — pasted.
2. The Q-convention test: `f = x0² + x1² + x0x1 − 3x0 − 3x1` → objective `-3.0` and `x = (1,1)`,
   agreeing with **live HiGHS** and with the analytic optimum, at `rtol=1e-6, atol=1e-8`. The
   `M.Q` (unhalved) and `triu(0.5·M.Q)` encodings must be *shown* to differ from HiGHS — the test
   asserts the correct one; the WP report pastes the wrong ones' values as evidence the test
   discriminates.
3. QP min and max, `<=`/`>=`/`==`, duals agreeing with live HiGHS (`rtol=1e-5, atol=1e-6`).
4. A MIQP model raises `NotImplementedError`; an all-integer quadratic model (`model.type ==
   "IQP"`) raises the same.
5. A non-PSD quadratic model returns `SolverStatus.error` with
   `termination_condition == "internal_solver_error"` and does not crash the process.
6. `pytest --run-gpu -q` green, ruff and mypy clean, docs still clean.

### Sequencing summary

```
WP-A  ──────────────► merge (commit 1)
WP-B  ──────────────► merge (commit 2)          [parallel with A: disjoint file]
                       └─► WP-C ─► merge (3)    [tests]
                       ├─► WP-D ─► merge (4)    [docs]   } parallel: disjoint files
                       └─► WP-E ─► merge (5)    [notes]  }
                                    └─► WP-F ─► merge (6) [QP, last]
```

---

## 8. Test plan

### 8.1 Where the tests live

| File | New/edited | Tier | Gating |
|---|---|---|---|
| `test/test_solvers.py` | edited (+6 rows) | V0 | none — pure class introspection, runs on CPU with cuOpt absent |
| `test/test_optimization.py` | edited (+9) | V2 | automatic via `GPU_ONLY` + `conftest.py:53-79` |
| `test/test_cuopt.py` | **new** | V0 + V1 | `pytestmark = pytest.mark.skipif("cuopt" not in licensed_solvers, ...)`, plus every test parametrized `@pytest.mark.parametrize("solver", ["cuopt"])` so `conftest` adds `pytest.mark.gpu` and skips without `--run-gpu` |
| `test/test_cuopt_keyboard_interrupt.py` | **new** | V0 | none — CPU-only, dummy callable, no cuOpt import |

The `solver=["cuopt"]` parametrization is deliberate: it reuses `conftest`'s **existing** generic
GPU gate (`callspec.params["solver"]` + `solver_supports(..., GPU_ONLY)`) instead of adding a
second gating mechanism, which is what keeps C-R11 a zero-line diff.

### 8.2 V0 — structural

Mechanical, every WP: `/home/azureuser/.local/bin/ruff check .`, `.venv/bin/mypy`. Plus, as real
tests:
* the six `test_solver_class_supports_feature` rows (§2.8), + `QUADRATIC_OBJECTIVE` in WP-F.
* `test_registry_resolves`: `SOLVER_REGISTRY["cuopt"].features == solvers.cuOpt.supported_features()`,
  `.display_name == "cuOpt"`, `_solver_class_for("cuopt") is solvers.cuOpt`,
  `SolverName.cuOpt.name == solvers.cuOpt.__name__` (C-R7, and it pins the enum-name/class-name
  invariant the shim depends on).
* `"cuopt" in solvers._SOLVER_PROBE_ORDER` and `"cuopt" in available_solvers` (guarded on
  `is_available()`).
* `test_run_cuopt_cancels_on_keyboard_interrupt` — `_run_cuopt_with_keyboard_interrupt` with a
  dummy callable that blocks on an `Event`; `_thread.interrupt_main()` from a helper thread;
  assert `KeyboardInterrupt` is raised **within 1 s** and that the dummy's own thread is still
  running (i.e. the helper did not drain). Mirrors `test_highs_keyboard_interrupt.py` exactly.

### 8.3 V1 — differential vs HiGHS, live, in the same process

Every numeric assertion compares against `model.solve("highs")` called **live on the identical
model in the same process**. No hard-coded expected value anywhere, with two deliberate
exceptions that are *mathematical* rather than *solver* facts and are labelled as such: the QP
closed-form optimum and the semi-continuous discriminating optimum.

| # | Test | What it proves | Notes |
|---|---|---|---|
| 1 | **Sign matrix, six cells** — `min`/`max` × `<=`/`>=`/`==`, on the exploration §2 model (`A=[[1,2,1],[3,1,1]]`, `b=[4,6]`, `0<=x<=10`, coefficients chosen so `x*=(1.6,1.2,0)` with both rows binding and a unique non-degenerate dual) | duals and objective agree with HiGHS in every cell | the charter's mandated matrix; parametrized over the six cells |
| 2 | **Presolve-solvable `max` models** — square equality systems at n=4 and n=10 (exploration `a_02f`) | the six-cell matrix *alone misses the negated-duals bug*; these are the cases where cuOpt finishes inside presolve and returns `dual = -dual_HiGHS` with `max|Δ| = 3.61` | asserts duals match HiGHS **and** `sol.get_solved_by().name == "Unset"`, which pins the test to the buggy code path so that losing coverage fails loudly rather than silently |
| 3 | **MILP** — the §4.3 model with integer variables, plus one knapsack | objective agrees with HiGHS; `model.dual` is empty and reading it raises nothing | MILP duals are unavailable by design (`get_dual_solution` raises for a MILP solution) |
| 4 | **Semi-continuous** — `x ∈ {0} ∪ [1,10]`, `x <= 0.5`, `max x` → `0`; and `x <= 7`, `max x` → `7` | the `'S'` mapping is real, in both discriminating directions | vs live HiGHS **and** the analytic value |
| 5 | **Repeat solves, above the crash threshold** — n=2000, m=1000, three sequential solves **in a subprocess** (`sub.run([sys.executable, "-c", ...])`), assert `returncode == 0` and three `optimal` lines | the `method=0` segfault cannot silently regress | run in a subprocess deliberately: a SIGSEGV in-process would kill the pytest session instead of failing one test. Subprocess precedent: `test_available_solvers.py:22-37` |
| 6 | **Status mapping** — infeasible (`x<=0, y<=0`), unbounded (`min -x, x>=0` unbounded above), `time_limit=1e-6` on a 2000×1000 LP, `iteration_limit=1` | `infeasible`, `infeasible_or_unbounded`, `time_limit`, `iteration_limit` | asserts on `model.termination_condition`; `DualInfeasible`/`Unbounded` (status 3) are **untestable** — never observed in Phase 1 — and are noted as such in a comment rather than asserted |
| 7 | **Guards** — zero-nnz model → `ValueError`; MIQP/IQP → `NotImplementedError` (WP-F); `warmstart_fn` → `NotImplementedError`; `solution_fn` → `NotImplementedError`; `basis_fn` → warns and still solves; SOS model → the base `ValueError` naming `reformulate_sos` | cuOpt's silent-failure modes are pre-empted (C-R22, C-R24) | `pytest.raises(..., match=...)` on the message |
| 8 | **Options** — `TimeLimit=1` → `ValueError` matching `TimeLimit`; `presolve=False` → solves (bool retry); `time_limit=1` → solves; `log_fn` → file exists and contains `cuOpt version`; `method=0` → emits the warning (`caplog`) and still solves | C-R23 | |
| 9 | **Default `io_api`** — `model.solve("cuopt")` with no `io_api`, `keep_files=True`, `problem_fn=tmp/"p.lp"` → solves, warns, and `p.lp` does **not** exist | C-R25 and the `_build_file` override | |
| 10 | **QP** (WP-F) — the cross-term Q-convention model, and QP min/max × `<=`/`>=`/`==` duals | the `0.5·M.Q` convention and QP dual signs | see WP-F acceptance criterion 2 |

### 8.4 V2 — linopy's own test models

The existing `params`-parametrized suite in `test_optimization.py`, which cuOpt joins
automatically via `DIRECT_API` + `licensed_solvers`, using the existing `GPU_SOL_TOL = 2.5e-4`
branch. No new expected values, no new constants. **This is where the `--run-gpu` run must be
shown to have actually executed** (charter §9's "GPU tests skipped but reported green" risk):
WP-C pastes `pytest --run-gpu -q -k cuopt -v` with visible `PASSED` lines, not just a summary.

### 8.5 Proposed NUMERIC tolerances (for `logs/15-verification.md` to freeze)

Defined as module constants in `test/test_cuopt.py` so they appear once and are greppable.

| Constant | Value | Rationale |
|---|---|---|
| `CUOPT_OBJ_RTOL` | `1e-6` | On small models cuOpt's objective matched HiGHS to `5.55e-17`–`4e-15` relative (`a_02f`: n=4 `13.754309` both; `a_q0_corr`: `-2.80000000` vs `-2.7999999999999994`). `1e-6` is ~9 orders above the observed residual and 4 orders **below** the smallest wrong-convention QP error (`-1.5` vs `-3.0`). |
| `CUOPT_OBJ_ATOL` | `1e-8` | Needed for objectives at or near zero (the semi-continuous `0` case). |
| `CUOPT_PRIMAL_RTOL` | `1e-6` | cuOpt returned `x=[1.6, 1.2]` exactly on the matrix model; the loosest observed primal was the QP `0.99999997` vs `1.0` = `3e-8` relative. `1e-6` is ~33× that. |
| `CUOPT_PRIMAL_ATOL` | `1e-7` | Variables resting at a bound came back as `0` exactly, but PDLP-solved components can sit at `~1e-8`; `1e-7` covers a value that should be `0`. |
| `CUOPT_DUAL_RTOL` | `1e-5` | Sized to the **worst measured non-PDLP dual residual** with ~5× margin: Barrier `1.73e-09`–`2.47e-09` (linopy's default method), Concurrent `8.18e-08`, DualSimplex `1.97e-07`. A **sign error is `2·|y| = 0.8`** on the matrix model — 5 orders of magnitude above this tolerance, which is what the matrix is for. |
| `CUOPT_DUAL_ATOL` | `1e-6` | Non-binding duals: HiGHS returns exact `0`, cuOpt returns `~1.07e-10` (QP) to `~3.2e-07` (LP reduced-cost-adjacent slack). An `atol` is mandatory here — a pure `rtol` against HiGHS's `0` can never pass. `1e-6` is ~3× the largest observed non-binding value. |
| `GPU_SOL_TOL` (existing, `test_optimization.py:80`) | `2.5e-4` | Reused unchanged for the shared V2 suite. Not tightened: it is #516's number for GPU solvers generally and re-tuning it for cuOpt would change cuPDLPx's tests too (I4). |

**The interlock that makes these numbers defensible.** They are only valid because linopy defaults
`method=3` (Barrier). Exploration §7 showed cuOpt's default `1e-4` tolerances moving an objective
by `5.52e-03` relative on a 2000×1000 LP under **PDLP**, and the four methods disagreeing by
`2.5e-5` relative on an n=5000 model at default tolerances. Therefore:
* **No test may force `method=1` (PDLP) and then assert against these tolerances.** The one test
  that sets `method=1` (`test_optimization.py`'s `method_options`) asserts only
  `np.isclose(objective, 3.3)` on a 2-variable model — inside numpy's default `rtol=1e-5`.
* **No test tightens cuOpt's own tolerance parameters.** If a differential check misses, the
  honest outcome is a reported miss with the actual numbers, **not** an injected
  `set_optimality_tolerance` and **not** a widened `rtol` (I12, §3.6).
* If WP-A's contingency (§5b) forces the default method away from Barrier, **every tolerance in
  this table must be re-derived at Gate 3 before it is frozen.** That dependency is stated here so
  it cannot be lost.

---

## 9. Docs plan

| Surface | What it will say |
|---|---|
| `doc/gpu-acceleration.rst` | A `cuOpt` subsection under **Supported GPU Solvers**, before `cuPDLPx` (alphabetical) or after it (append) — either, consistently. Content: what cuOpt is (NVIDIA, GPU concurrent dual simplex + barrier + PDLP) with a link to `docs.nvidia.com/cuopt`; **install** = `pip install "linopy[cuopt]"` / `uv pip install "linopy[cuopt]"` as the headline, with a warning that the bare `cuopt` package on PyPI is a **different project** and NVIDIA's is `cuopt-cu12`; **requirements** = Linux x86_64/aarch64 only, Python ≥ 3.11, NVIDIA GPU with **compute capability ≥ 7.0**, driver **≥ 525.60.13**, and the **`cu12` vs `cu13`** caveat (the extra installs `cu12`; `cu13` wheels need a newer driver, and choosing them is the user's own `pip install cuopt-cu13`); a **usage example** (`m.solve("cuopt", time_limit=60)`); **supported** = LP, MILP, QP, semi-continuous variables, duals for continuous problems, `log_fn`; **not supported** = MIQP (quadratic objective + integer variables), quadratic constraints, SOS, indicator constraints, warm start / basis files / solution files, reduced costs, in-place persistent updates, file `io_api`s (`io_api="lp"`/`"mps"` are accepted but served through the direct API with a warning); **limitations / notes** = linopy defaults `method=3` (barrier) because cuOpt's own default concurrent method (`method=0`) is known to crash on repeated solves in one process in cuOpt 26.08, MILP models return no duals, `log_file` truncates per solve, Ctrl-C returns control immediately but the GPU work runs to completion, and `time_limit` is the only hard bound. Also updates the file's own `pytest ... -k cupdlpx --run-gpu` example to mention `-k cuopt`. The page-level `.. warning:: This feature is experimental...` stays — it is accurate. |
| `doc/prerequisites.rst` | The **GPU-accelerated solvers** section (`:63-75`) gains cuOpt with the **one-command** install as the headline: `uv pip install "linopy[cuopt]"`. Immediately adjacent (not three pages away): **Linux only**, **CUDA 12 driver ≥ 525.60.13**, **compute capability ≥ 7.0**. `pypi.nvidia.com` is mentioned **at most** as an optional fallback, never as a required step, and **never** with `--index-strategy` — the tree is on ordinary PyPI. A pointer to `:doc:`gpu-acceleration``. |
| `doc/contributing.rst` | Beside the existing `--run-gpu` text (`:66-77`): `pytest --run-gpu -k cuopt` for cuOpt only; that the cuOpt extra makes `uv sync --extra cuopt` the way to get cuOpt for local GPU testing; and one honest caution — the `benchmarks` extra pins `pandas==3.0.5` while cuOpt's `cudf-cu12` caps pandas below `3.0.4`, so **benchmarks and cuOpt cannot share a virtualenv** (charter §4.4). Also updates `:46`'s `uv sync --extra dev --extra solvers` line **only if** it needs the cuopt extra mentioned; otherwise left alone (I4). |
| `doc/index.rst` | `:53`'s solver bullet list gains `- \`cuOpt <https://docs.nvidia.com/cuopt>\`__ (GPU-accelerated; LP, MILP, QP)` next to cuPDLPx. |
| `doc/api.rst` | `solvers.cuOpt` added to the **Implementations** autosummary (`:626-637`), placed before `solvers.cuPDLPx` to keep the list's sort order. **`to_cuopt` is deliberately NOT added**: the whole `to_*` family (`to_gurobipy`, `to_highspy`, `to_xpress`, `to_cupdlpx`) is absent from `api.rst` on current `master`, so adding only cuOpt's would be inconsistent, and adding all five would be a drive-by (I4). This is the §6.6b row I diverge on; the divergence and its reason go in `logs/50-review.md` per Gate 6's "satisfied **or its absence justified**". If the lead prefers the literal reading, the minimal change is one line, `model.Model.to_cuopt`, under **Input and output** — but then `to_cupdlpx` should follow in a separate, non-cuOpt PR. |
| `doc/release_notes.rst` | One bullet under `Upcoming Version` → `*Other*`: cuOpt support added (LP, MILP, QP, semi-continuous), installable with `pip install "linopy[cuopt]"` on Linux with a CUDA 12 GPU (compute capability ≥ 7.0, driver ≥ 525.60.13); `model.to_cuopt()` exposes the underlying `DataModel`. |
| `README.md` | `:153`'s **Supported solvers** list gains `* [cuOpt](https://docs.nvidia.com/cuopt) (GPU-accelerated)` next to cuPDLPx. |
| `pyproject.toml` | The extra carries the four-line comment in §5i explaining *why* it is a sibling of `solvers` and not part of it — that is "documented wherever extras are explained" for a file whose extras are documented by inline comment. |
| `examples/using-solvers.ipynb` | **Markdown-only** mention alongside the existing cuPDLPx mention, naming the extra and the Linux/GPU requirement. **No executable cuOpt cell**: `nbsphinx_execute = "auto"` runs this notebook during the docs build on a GPU-less machine (charter §6.0), so a cuOpt cell would either fail the build or need stored outputs. This decision is logged. |

**Mechanical gap check** (charter §6.6b) will show cuOpt touching everything #516 touched except
`CLAUDE.md` (nothing to add — `AGENTS.md` is the live file and needs no cuOpt content) and
`linopy/solver_capabilities.py` / `test/conftest.py` (charter-declared zero-diff rows), plus
`doc/api.rst` where #516 also made no change. Each difference is justified above.

---

## 10. Explicit non-goals

Anything below appearing in the diff is scope creep and should be challenged.

1. **File-based `io_api`s** (`lp`, `lp-polars`, `mps`) via `cuopt.linear_programming.io.Read`, and
   `SolverFeature.READ_MODEL_FROM_FILE` / `LP_FILE_NAMES` (§1.3). Recorded as an evidence-backed
   follow-up.
2. **`problem.Problem`** and anything reachable only through it (`relax()`, `readMPS`/`writeMPS`,
   `updateConstraint`, `getWarmstartData`).
3. **Warm start, initial points, MIP starts, basis files** — and in particular
   `DataModel.set_initial_primal_solution`, which must never be called (§5h).
4. **Reduced costs** — no new field on `Solution`, no `SolverReport` addition (§5e).
5. **MIQP / IQP**, quadratic constraints (`add_quadratic_constraint`), and any
   quadratically-constrained surface.
6. **SOS and indicator constraints** — verified absent upstream; the base `_validate_model`
   messages are the whole story.
7. **Persistent / in-place updates** (`supports_persistent_update`, `_apply_*` hooks).
8. **MIP callbacks** (`set_mip_callback`), `internals`, and any progress reporting.
9. **The remote/gRPC cuOpt server** (`cuopt-server-cu12`, `cuopt-sh-client`,
   `CUOPT_REMOTE_HOST`/`CUOPT_REMOTE_PORT`). linopy never sets them, and the "set them" text is
   what we translate *away* (§5c).
10. **`cuopt.routing`** — never imported.
11. **Setting variable/constraint names** in the direct path (`set_variable_names`,
    `set_row_names`), and honouring `explicit_coordinate_names`.
12. **`MIP_DUAL_BOUND_REPORT`**, `mip_gap` reporting, `lp_stats` surfacing, `get_solved_by()` in
    production code (it appears only in tests).
13. **Tuning cuOpt's tolerances or `mip_semi_continuous_big_m`** by default; setting a default
    `time_limit`; touching `pdlp_solver_mode`, `crossover`, `save_best_primal_so_far`.
14. **Any change to `linopy/solver_capabilities.py`, `linopy/constants.py`, `test/conftest.py`,
    `test/test_available_solvers.py`** — charter-declared zero-diff rows, verified not edited.
15. **Reordering `_SOLVER_PROBE_ORDER`'s existing entries** or changing the default-solver
    precedence for any existing solver.
16. **Fixing cuPDLPx's `if MAXIMIZE: dual = -dual`**, its `_run_file` netcdf detour, or any other
    pre-existing linopy defect noticed in passing — record in `logs/45-known-issues.md` (I13).
17. **cu13 wheels, aarch64, Python 3.12–3.14, multi-GPU (`num_gpus`)** — documented, not tested.
18. **Benchmarks** of any kind in this venv (pandas pin conflict, charter §4.4).
19. **Adding the `to_*` family to `doc/api.rst`** (§9).
20. **Widening any tolerance, skipping any test, or downgrading any check to make something
    green** (I12, §3.6). A red check reported with its numbers is the expected output when
    something misses.

---

## 11. Risks — what is most likely to go wrong with *this* plan

Ordered by expected damage.

1. **`method=3` (Barrier) as the default is the plan's biggest unverified bet.** Phase 1 never
   exercised `method` on the MILP branch-and-bound path, never timed Barrier at scale against
   Concurrent, and Barrier's GPU backing is inferred from `nvidia-cudss-cu12` being in the
   dependency tree rather than measured. Failure modes: `Solve` rejects `method=3` for a MILP;
   Barrier is materially slower than Concurrent on the LPs users care about; or Barrier turns out
   to be substantially CPU-side, which would make a `GPU_ONLY` solver embarrassing. Mitigated by
   the pre-decided contingency in §5b (apply the default only to non-integral models) and by the
   fact that *any* of methods 1/2/3 is a defensible replacement — but the frozen tolerances in
   §8.5 are derived from Barrier's accuracy, so a late switch means re-deriving them at Gate 3.
   **This is the item I would look at first.**
2. **`is_available()` is a deliberate partial divergence from charter §4.6 constraint 3.** If the
   lead reads that constraint as literally binding, the plan needs the ~10-line `_gpu_visible()`
   change in §5c. Cheap to apply, but it also changes what `test_available_solvers.py:39`'s
   `is_available() == (name in available_solvers)` invariant *means* for one solver, and it adds a
   CUDA-runtime touch to a plain `import linopy; linopy.available_solvers`.
3. **`_license_probe` runs a real GPU solve at test-collection time.** `test_optimization.py`
   imports `licensed_solvers` at module scope, so a plain `.venv/bin/pytest` on this T4 now creates
   a CUDA context and spends ~0.3 s before collecting anything — a behaviour change versus the
   Gate 0 baseline that is easy to mistake for a regression. It is also the only path by which
   cuOpt touches the GPU in a non-`--run-gpu` run. Acceptable (that is what `license_status()` is
   documented to do, and cuPDLPx's probe constructs a native model too), but it must be called
   out in the WP-A report so the Phase 5 reviewer does not flag it as a leak.
4. **Not declaring `READ_MODEL_FROM_FILE` plus overriding `_build_file` will read as a deviation.**
   A reviewer comparing side-by-side with cuPDLPx (charter §6.6a) will see cuPDLPx overriding
   `_run_file` and cuOpt overriding `_build_file` and may call it a stylistic outlier. The
   justification (no wasted LP file) is in a code comment and in §1.3, but it is the kind of
   difference that costs a review round. There is also a genuine ergonomic wart: a user who
   explicitly passes `io_api="lp"` gets the direct path with a warning rather than an error.
5. **`_LazyModule("cuopt.linear_programming")` assumes the package binds its submodules.**
   Exploration enumerated them but did not state whether via `dir()` on an imported package or via
   explicit imports. If `getattr` does not resolve, WP-A must apply the named fallback — a small
   change, but it touches every call site.
6. **The frozen dual tolerance (`rtol=1e-5, atol=1e-6`) could still miss.** It is sized from
   measurements on two small models. Test 2's square-equality systems at n=10 and the QP dual
   cells are the likeliest misses. The plan forbids widening, so a miss becomes a blocked item and
   a Gate 3 return — correct, but it is a real schedule risk, and I would rather absorb it than
   pre-inflate the number.
7. **Declaring `SEMI_CONTINUOUS_VARIABLES` exposes cuOpt's big-M semantics.** cuOpt implements
   `'S'` with `mip_semi_continuous_big_m` (default `1e10`); linopy's semantics are exact
   (`x = 0` or `lb ≤ x ≤ ub`). For a semi-continuous variable with an infinite upper bound the two
   are not the same problem, and the differential test could disagree with HiGHS. Mitigation: the
   V1 tests use finite bounds (the shapes exploration verified) and the big-M caveat is
   documented. If it does disagree, the honest fallback is to drop the feature flag — which then
   makes `_validate_model` raise for a model cuOpt *can* solve, i.e. a genuine trade-off, not a
   free out.
8. **A lingering daemon thread inside cuOpt's C++ code** after a `KeyboardInterrupt` may produce
   pytest warnings, or interact badly with a subsequent solve in the same process. Exploration
   verified only clean *interpreter shutdown*, not "interrupt, then solve again".
9. **Bulk.** My estimate is +770/−5 across 14 files, inside the §9 yardstick (#516: +771/−62 / 15
   files). The parts most likely to inflate it: the class docstring, the interrupt helper, the
   `_license_probe`, and `test_cuopt.py`. If a reviewer trims, the interrupt helper is the most
   defensible thing to keep and the vtype whitelist / width guard the cheapest to lose — but
   losing them re-opens cuOpt's silent-wrong-answer modes, so I would trim docstring prose first.
10. **`get_solved_by().name == "Unset"` is asserted in test 2.** It pins the test to cuOpt's
    presolve-only path deliberately, so a future cuOpt release that changes when presolve closes a
    model will turn that test red even though nothing is wrong. That is the failure direction I
    want (loud loss of coverage beats silent), but it will look like a flake to whoever hits it,
    so the test carries a comment saying exactly that.
11. **Status 3 (`DualInfeasible` / `Unbounded`) is mapped but untestable.** Seven unbounded
    variants all returned 11 in Phase 1. If a real user model does hit status 3, the `unbounded`
    mapping is reasoning, not evidence.
