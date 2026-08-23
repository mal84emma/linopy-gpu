# plan-2 — cuOpt × linopy implementation plan

*Planner k=2 · 2026-08-22 · model: Claude Opus 5 (`claude-opus-5[1m]`) · written against `master` @ `09c34dd`*

Sources: charter `PROJECT_LEAD.md` (§4 treated as fact), `AGENTS.md`, `logs/30-decisions.md`
rules block, `logs/10-exploration.md` (single source of empirical truth), and read-only reading
of `linopy/solvers.py`, `linopy/io.py`, `linopy/model.py`, `linopy/matrices.py`,
`linopy/constants.py`, `linopy/objective.py`, `linopy/solver_capabilities.py`, `test/conftest.py`,
`test/test_optimization.py`, `test/test_solvers.py`, `test/test_semi_continuous.py`,
`pyproject.toml`, `doc/*.rst`, `gh pr diff 516` (file list / shape only), and the installed
`cuopt` sources under `.venv/lib/python3.11/site-packages/cuopt/`.

Every claim below is either (a) cited to a charter/exploration finding, (b) cited to a
`file:line` in this repo or in the installed cuOpt package, or (c) explicitly flagged
**UNVERIFIED — WP must verify**. Nothing is asserted as measured that was not measured.

---

## 1. Entry-point decision

### Decision

| | Choice |
|---|---|
| **Primary entry point** | `cuopt.linear_programming.data_model.DataModel` + `cuopt.linear_programming.solver.Solve`, driven from `model.matrices` |
| **Secondary entry point** | **None.** No `problem.Problem`, no `io` submodule, no `BatchSolve`, no gRPC/server client. |
| **`accepted_io_apis`** | `frozenset({"direct"})` |
| **`SolverFeature.READ_MODEL_FROM_FILE`** | **not declared** |
| **`_run_file`** | overridden to raise `NotImplementedError` naming `io_api="direct"` |

### Position on the Phase 1 Q0 recommendation

Q0 recommends **`DataModel` + `Solve` primary, `io` submodule secondary for the `lp`/`mps`
`io_api`s**. I **agree on the primary and diverge on the secondary.**

Agreement on the primary is easy and evidence-driven: `DataModel` is 70× faster to build at
10⁵ variables (0.012–0.014 s vs 0.858–1.042 s for `Problem` and 0.831–0.951 s for the file
path), needs 13–16 lines of translation from `model.matrices`, and is the only candidate that
can express ranged rows. `Problem`'s apparent advantages (warm start, callbacks, options, log
control) were all shown to live on `SolverSettings`, which `DataModel` shares, and its one real
differentiator (`relax`/`readMPS`/`writeMPS`) is not something linopy needs. This is settled.

I diverge on declaring the file `io_api`s, for five reasons:

1. **It is a performance regression dressed as a capability.** The file path costs
   0.83–0.95 s of build time at 10⁵ variables against 0.012 s for `DataModel` — a 70×
   penalty on the one axis a GPU solver is chosen for. Every file-`io_api` solve is
   strictly worse than the `direct` solve of the same model. Offering a user a slower path
   is not a "win"; the win Q0 identified is a *capability parity* argument, not a
   user-outcome argument.
2. **`accepted_io_apis` is inert on current `master`.** `grep -rn "accepted_io_apis"
   linopy/ test/` returns exactly two hits: the ClassVar declaration at
   `linopy/solvers.py:466` and a test double at
   `test/test_persistent_solver_orchestrator.py:24`. **No production code reads it.** So
   declaring `{"lp","mps"}` there buys nothing functional — it is a comment with a type
   annotation. The thing that actually gates behaviour is `SolverFeature.READ_MODEL_FROM_FILE`
   (which builds the file-io test matrix at `test/test_optimization.py:45-50`) plus a working
   `_run_file`.
3. **Declaring `READ_MODEL_FROM_FILE` multiplies the GPU test matrix by ~6 and imports
   unverified risk.** `params` is `product(file_io_solvers, io_apis, explicit_coordinate_names)`
   — `lp` × `lp-polars` × `mps` × `{False, True}`, i.e. six extra parameterisations of ~40
   tests, each a real GPU solve. Q0's fidelity evidence covers `.lp`/`.mps` written by linopy's
   default writers only. It does **not** cover: the `lp-polars` writer, `explicit_coordinate_names`
   (which emits `name[coord]#label`, `linopy/io.py:112-122`, and would have to round-trip
   through cuOpt's parser and back through `_names_to_labels`), MPS `max`-sense interaction with
   `maybe_adjust_objective_sign` (`linopy/solvers.py:389-401`) and
   `read_sense_from_problem_file` (`:369-379`), or `.mps` writing requiring `highspy` to be
   present. Each is a plausible silent-wrong-answer site and none was measured.
4. **The charter's own bulk yardstick argues against it** (§9: #516 was +771/−62; materially
   past ~1.5× "warrants a hard look"). File IO is the single largest optional chunk available
   and it buys the least.
5. **Repo precedent is direct-only for GPU solvers** (`cuPDLPx`, `linopy/solvers.py:4177-4184`),
   and I5 says repo precedent outranks cuOpt's own surface where they disagree on linopy-side
   shape.

What I keep from Q0's secondary finding: it becomes an **explicitly named non-goal** (§10) with
the measured numbers attached, so a future contributor can pick it up knowing it was
considered and priced, not overlooked. `_run_file` gets a 4-line override whose message points
at `io_api="direct"` — better than the base class's generic "File-based API not implemented"
and much better than cuPDLPx's netcdf detour (`:4195-4225`), which we do **not** copy.

### Entry-point mechanics

```python
# module level, next to the other lazy imports (linopy/solvers.py:296-304)
cuopt = _LazyModule("cuopt.linear_programming")
```

Note the lazy module targets **`cuopt.linear_programming`**, not `cuopt`. Two reasons:
`cuopt/linear_programming/__init__.py` re-exports everything we need (`DataModel`, `Solve`,
`SolverSettings`, `SolverMethod`, `PDLPSolverMode`, `Solution`), and exploration item 10's
closing note records that importing **`cuopt.routing` installs a global `sys.excepthook`** that
writes `error_log.txt` into the cwd (`routing/vehicle_routing_wrapper.pyx:144-160`). Naming the
LP subpackage in the `_LazyModule` makes it structurally hard for a later change to widen the
import. A one-line comment records why.

`is_available()` still probes the top-level name (`_has_module("cuopt")`) — see §5c.

---

## 2. Full file-by-file change set

Line numbers are current `master` (`09c34dd`).

### 2.1 `linopy/solvers.py`

| Anchor | Change |
|---|---|
| `:148-164` `SolverFeature` | **No edit.** Every flag cuOpt needs already exists (§3). |
| `:175-181` `TYPE_CHECKING` | add `import cuopt` (alphabetical: before `cupdlpx`) |
| `:296-304` lazy modules | add `cuopt = _LazyModule("cuopt.linear_programming")` **above** `cupdlpx` (alphabetical), with the one-line `cuopt.routing` comment |
| `:346-359` `SolverName` | add `cuOpt = "cuopt"` immediately above `cuPDLPx = "cupdlpx"`. Vendor spelling; the enum member name must equal the class name (`solver_capabilities.py:52-57`). |
| after `:252` (`_run_highs_with_keyboard_interrupt` ends) | new module-level helper `_run_cuopt_with_keyboard_interrupt(data_model, settings) -> Solution` (§5g) |
| `:4156` (before `class cuPDLPx`) | new `class cuOpt(Solver[None])` — the bulk of the diff. Placed **before** `cuPDLPx` so the file's GPU-solver block reads in the same order as `_SOLVER_PROBE_ORDER`. |
| `:4419-4437` `_solver_class_for` / `QUADRATIC_SOLVERS` / `NO_SOLUTION_FILE_SOLVERS` | **No edit.** Both lists are derived from declared features. |
| `:4443-4457` `_SOLVER_PROBE_ORDER` | insert `"cuopt",` between `"copt",` and `"cupdlpx",` |

Class layout (ClassVar order and method order copied from `Highs` `:1564-1902` and `cuPDLPx`
`:4176-4416`, so the override set matches its siblings — §5.1 / §6.6a):

```
class cuOpt(Solver[None]):
    """<numpydoc class docstring: what cuOpt is, install pointer, the notable
       solver options (time_limit, method, log_to_console, *_tolerance), and
       the three limitations users must know: no warm start, no reduced costs,
       no MIQP.>"""

    display_name: ClassVar[str] = "cuOpt"
    features: ClassVar[frozenset[SolverFeature]] = frozenset({...})     # §3
    accepted_io_apis: ClassVar[frozenset[str]] = frozenset({"direct"})

    # cuOpt's default method (Concurrent) segfaults on repeat solves; see §5b.
    _DEFAULT_METHOD: ClassVar[int] = 3      # SolverMethod.Barrier

    is_available()            # classmethod, functools.cache — §5c
    _license_probe()          # classmethod — cheap, no solve
    _run_file(...)            # NotImplementedError pointing at io_api="direct"
    _build_direct(**kwargs)
    _build_solver_model(model) # staticmethod
    _run_direct(...)
    _solve(...)
    _set_solver_params(settings, log_fn=None)
```

Method bodies, in the order a reviewer will read them:

**`is_available`** — §5c. `_has_module("cuopt")` **and** a real CUDA-device probe, with a
one-shot `logger.warning` when the module is present but no device is.

**`_license_probe`** — mirrors `cuPDLPx._license_probe` (`:4191-4193`): construct the native
objects without solving.
```python
@classmethod
def _license_probe(cls) -> None:
    cuopt.DataModel()
    cuopt.SolverSettings()
```
Deliberately **not** a solve: `licensed_solvers` is evaluated at import time of
`test/test_optimization.py` (`:41`, `:52`, `:69`, `:74`), and a solve there would create a CUDA
context in every pytest process, including CPU-only runs. Constructing the objects proves the
compiled extensions and their CUDA libraries load, which is what the probe is for.

**`_run_file`**
```python
def _run_file(self, ..., **kw: Any) -> Result:
    raise NotImplementedError(
        "cuOpt does not support solving from LP/MPS files in linopy. "
        "Use io_api='direct'."
    )
```

**`_build_direct(self, **kwargs)`** — mirrors `cuPDLPx._build_direct` (`:4227-4244`):
reject MIQP/IQP (§5d), warn-and-ignore `explicit_coordinate_names`, build, set
`self.solver_model` / `self.io_api = "direct"` / `self.sense = model.sense`, then
`self._cache_model_labels(model)`. `set_names` and `log_fn` arrive in `**kwargs`
(`linopy/model.py:2116-2121`) and are ignored here; `log_fn` is honoured in `_solve`, as
cuPDLPx does.

**`_build_solver_model(model)`** — `@staticmethod`, the translation. Concretely:

```python
M = model.matrices
n = len(M.vlabels)
A = M.A.tocsr() if M.A is not None else None
lower = np.where((M.sense == ">") | (M.sense == "="), M.b, -np.inf)
upper = np.where((M.sense == "<") | (M.sense == "="), M.b, np.inf)
# ^ same construction as cuPDLPx :4259-4268; np.inf is accepted directly
#   (exploration §5: no sentinel needed).

# cuOpt cannot solve a model whose constraint matrix has no nonzero
# (exploration §5: nnz == 0 -> NoTermination, obj=nan, empty solution, no
# exception). Append one free row so unconstrained models still solve.
padded = A is None or A.nnz == 0
if padded:
    A, lower, upper = <one row: coefficient 1.0 on column 0, bounds -inf..+inf>

# M2 (§5a): never call set_maximize(True).
sign = -1.0 if model.objective.sense == "max" else 1.0

dm = cuopt.DataModel()
dm.set_csr_constraint_matrix(A.data, A.indices, A.indptr)
dm.set_constraint_lower_bounds(lower)
dm.set_constraint_upper_bounds(upper)
dm.set_objective_coefficients(sign * M.c)
dm.set_variable_lower_bounds(M.lb)
dm.set_variable_upper_bounds(M.ub)
dm.set_maximize(False)          # explicit: see M2 in the class docstring
if <model has binaries / integers / semi-continuous>:
    dm.set_variable_types(np.where(M.vtypes == "B", "I", M.vtypes))
if M.Q is not None:            # QP work package only
    Q = scipy.sparse.csr_matrix(sign * 0.5 * M.Q)
    dm.set_quadratic_objective_matrix(Q.data, Q.indices, Q.indptr)
return dm, padded
```

Notes the implementer must not lose:
* `M.A` is a `csr_array`; `.tocsr()` gives `data float64 / indices int32 / indptr int32`,
  exactly cuOpt's stored dtypes (exploration §6), so no conversion and no `UserWarning`.
* `set_variable_types` must receive an **`np.ndarray`** — a list/str/bytes raises
  `AttributeError` (exploration §6). `M.vtypes` is `<U1` (`linopy/matrices.py:94-100`), which
  works. The integrality guard mirrors `Highs._build_solver_model` (`:1681-1693`).
* `'B'` → `'I'` is required (cuOpt treats `'B'` as integer but does **not** clamp to `[0,1]`,
  exploration §6). No explicit clamping is needed: linopy already forces binary bounds to
  `0`/`1` at `linopy/model.py:862-870`, so `M.lb`/`M.ub` carry the box. **This is a
  simplification of exploration §6's advice, justified by that file:line.**
* `set_row_types` is **not** used — `data_model.py:468-470` gives bounds priority and there is
  no `'R'` ranged type (exploration §6).
* `set_objective_offset` is **not** used: linopy rejects objective constants outright
  (`linopy/objective.py:192-193`, `ValueError("Constant values in objective function not
  supported.")`), so there is never an offset to pass. `set_objective_scaling_factor` is never
  used (exploration §3: reporting-only, and it does not scale duals).
* No `set_variable_names` / `set_row_names`: solutions are mapped by label position via
  `_cache_model_labels` + `_solution_from_labels`, and printing 10⁵ names costs ~0.2 s for
  cosmetic value only. `explicit_coordinate_names=True` warns and is ignored, exactly as
  `cuPDLPx._build_direct` does (`:4233-4239`).
* `padded` is stored on the instance (`self._padded_rows: int`, 0 or 1) so `_solve` can drop
  the synthetic row's dual before mapping.

**`_run_direct`** — one call to `self._solve(...)`, signature identical to
`cuPDLPx._run_direct` (`:4284-4301`) but also forwarding `log_fn`.

**`_solve(...)`** — the status map, the M2 un-negation, and the report:

```python
# cuOpt reuses the same integer for different meanings on the LP and the MILP
# path (2 is PrimalInfeasible/Infeasible, 3 is DualInfeasible/Unbounded), so
# the map is keyed on the status *name*. The two enums' member names are
# globally distinct where they differ in meaning
# (solver/solver_wrapper.pyx:79-98), so one dict is unambiguous and we avoid
# importing that private module.
CONDITION_MAP: dict[str, TerminationCondition] = { ... }   # §4

settings = cuopt.SolverSettings()
self._set_solver_params(settings, log_fn)
if warmstart_fn is not None:
    raise NotImplementedError("Warmstarting not yet implemented for cuOpt.")
if basis_fn is not None:
    logger.warning("Basis files are not supported by cuOpt. Ignoring.")
if solution_fn is not None:
    logger.warning("Solution files are not supported by cuOpt. Ignoring.")

sol = _run_cuopt_with_keyboard_interrupt(self.solver_model, settings)

reason = sol.get_termination_reason()          # e.g. "Optimal" — .name
termination_condition = CONDITION_MAP.get(reason, TerminationCondition.unknown)
status = Status.from_termination_condition(termination_condition)
status.legacy_status = reason
if sol.get_error_status():                     # ErrorStatus != Success
    status.legacy_status = f"{reason}: {sol.get_error_message()}"

is_lp = sol.get_problem_category() == 0        # ProblemCategory.LP

def get_solver_solution() -> Solution:
    objective = float(sol.get_primal_objective())
    primal = np.asarray(sol.get_primal_solution(), dtype=float)
    dual = np.array([], dtype=float)
    if is_lp:
        dual = np.asarray(sol.get_dual_solution(), dtype=float)
        if self._padded_rows:
            dual = dual[: dual.size - self._padded_rows]
    if sense == "max":
        # M2: the model was handed to cuOpt as a minimisation, so the
        # objective and the duals come back negated. Note this is the
        # OPPOSITE of cuPDLPx (:4385-4386), which negates only for max
        # *because* it sets MAXIMIZE. Do not "harmonise" the two.
        objective = -objective
        dual = -dual
    primal = _solution_from_labels(primal, self._vlabels, self._n_vars)
    dual = _solution_from_labels(dual, self._clabels, self._n_cons)
    return Solution(primal, dual, objective)

solution = self.safe_get_solution(status=status, func=get_solver_solution)
solution = maybe_adjust_objective_sign(solution, io_api, sense)   # no-op for direct
```

Report block (mirrors `Highs._solve` `:1884-1902`), each read in its own
`contextlib.suppress(Exception)`:
* `runtime = float(sol.get_solve_time())`
* for MIP/IP: `stats = sol.get_milp_stats()`; `mip_gap = float(stats["mip_gap"])`;
  `dual_bound = float(stats["solution_bound"])`, **negated when `sense == "max"`** (M2 —
  `solution_bound` is in the negated objective's units). Keys verified at
  `solution/solution.py:213-222`; `get_milp_stats` raises for LP
  (`raise_if_lp_solution`, `:409`), hence the category guard.
* `simplex_iterations = int(stats["num_simplex_iterations"])` — free, and it is the only
  iteration count cuOpt reports that maps onto an existing `SolverReport` field
  (`linopy/constants.py:295-299`). LP's `lp_stats["nb_iterations"]` is method-agnostic and maps
  to neither `barrier_iterations` nor `simplex_iterations` honestly, so it is **not** surfaced.
  `lp_stats["dual_residual"]` is never surfaced (exploration §12: meaningless under QP).

**`_set_solver_params(self, settings, log_fn=None)`** — §5f.

### 2.2 `linopy/solver_capabilities.py`

**Zero-line diff**, as charter §5 says. `SOLVER_REGISTRY` is a `_LazyRegistry` deriving
`SolverInfo` from `SolverName` + the class's features (`:78-100`). Satisfied by a test, not an
edit (C-R7 → §8, `test_capability_shim_round_trips` at `test/test_solvers.py:479-483` already
parametrizes over every `SolverName` member, so adding the enum member automatically extends
it; the new registry assertion in `test_solvers.py` makes the coverage explicit).

### 2.3 `linopy/io.py`

* `:31-37` `TYPE_CHECKING`: add `from cuopt.linear_programming import DataModel as cuoptDataModel`
  above the `cupdlpx` line (matching the `from cupdlpx import Model as cupdlpxModel` idiom).
* after `:792-796` `to_cupdlpx`: add
  ```python
  def to_cuopt(m: Model) -> cuoptDataModel:
      """Build the cuopt DataModel for `m`."""
      solver = solvers.cuOpt.from_model(m, io_api="direct")
      return solver.solver_model
  ```
  Mirrors `to_cupdlpx` exactly (its nearest sibling), not `to_xpress`. ~7 lines.

### 2.4 `linopy/model.py`

* `:68-79` import block: add `to_cuopt,` (alphabetically before `to_cupdlpx`).
* `:2517` binding block: add `to_cuopt = to_cuopt` above `to_cupdlpx = to_cupdlpx`.

3 lines, exactly what #516 spent here.

### 2.5 `linopy/constants.py`

**Zero-line diff.** Every cuOpt status maps onto an existing `TerminationCondition` member
(§4). Charter §5: "add nothing new unless unavoidable" — it is avoidable.

### 2.6 `test/conftest.py`

**Zero-line diff expected.** `pytest_collection_modifyitems` (`:53-79`) keys generically on
`item.callspec.params["solver"]` + `solver_supports(solver, SolverFeature.GPU_ONLY)`; cuPDLPx
is not named anywhere. Declaring `GPU_ONLY` auto-marks and auto-skips. **Verified by
construction, and proved in WP-D** by running `pytest -q -k cuopt` (must report skips, not
runs) and `pytest --run-gpu -q -k cuopt` (must report runs). If a change turns out to be
needed, that is a finding to report, and #516's +50 lines here is a stale pattern (I5).

Consequence for test placement: any cuOpt test that is **not** parametrized on a `solver`
argument is invisible to that hook. Every new cuOpt test therefore takes `solver` as a
parameter (§8).

### 2.7 `test/test_optimization.py`

* `:558-570` `time_limit_option`: add `"cuopt": {"time_limit": 1}` (snake_case — cuOpt's names
  are case-sensitive, exploration §7).
* after `:58-60`: add the size-zero-safe selector, mirroring the existing
  `set_names_direct_solvers` idiom (`:58-60`, used at `:1201`):
  ```python
  cuopt_solvers = [solver for solver in ("cuopt",) if solver in licensed_solvers]
  ```
* `:1108-1111` `test_basis_and_warmstart`: extend the skip to cuOpt
  (`if solver in ("cupdlpx", "cuopt")`, with cuOpt's reason: PDLP warm-start state is not a
  file format — §5h).
* `:685-689` `test_infeasible_model`: **no change.** cuOpt returns `PrimalInfeasible` on
  infeasible LPs (exploration §1) — the cuPDLPx hang does not apply. If it turns out to hang,
  that is a finding, not a skip to add pre-emptively.
* new cuOpt-specific tests, all `@pytest.mark.parametrize("solver", cuopt_solvers)` (§8).

### 2.8 `test/test_solvers.py`

* `:448-465` feature table: add the cuOpt rows (§3 declares/denies; ~6 lines, #516 spent 4).
* one new test asserting the shim resolves (C-R7):
  ```python
  def test_cuopt_registered_in_capability_shim() -> None:
      info = SOLVER_REGISTRY["cuopt"]
      assert info.display_name == "cuOpt"
      assert info.features == solvers.cuOpt.supported_features()
  ```
  No GPU needed — the registry is derived from ClassVars.

### 2.9 `pyproject.toml`

Insert immediately after the `solvers = [...]` block (ends `:113`), as a sibling extra —
verbatim from the Q13-verified candidate (§5i).

### 2.10 Docs

`doc/gpu-acceleration.rst`, `doc/prerequisites.rst`, `doc/contributing.rst`, `doc/index.rst`,
`doc/api.rst`, `doc/release_notes.rst`, `README.md`, `examples/using-solvers.ipynb` — §9.

### 2.11 Not touched

`CLAUDE.md` (#516 edited it; that was #516's own harness note — not ours),
`linopy/matrices.py`, `linopy/constants.py`, `linopy/solver_capabilities.py`, `test/conftest.py`.

### Diff-size estimate

| File | est. |
|---|---|
| `linopy/solvers.py` | +250 (class ~190, interrupt helper ~35, imports/enum/probe ~5, QP ~20) |
| `linopy/io.py` | +9 |
| `linopy/model.py` | +3 |
| `pyproject.toml` | +7 |
| `test/test_optimization.py` | +115 / −3 |
| `test/test_solvers.py` | +14 |
| `doc/*` + `README.md` + notebook | +115 / −2 |
| **total** | **≈ +513 / −5 over 13 files** |

Comfortably inside the §9 yardstick (#516: +771/−62 / 15 files). Declaring the file `io_api`s
would have added an estimated +120 of code and a 6× GPU test matrix for negative user value.

---

## 3. Feature-flag set

```python
features: ClassVar[frozenset[SolverFeature]] = frozenset(
    {
        SolverFeature.INTEGER_VARIABLES,
        SolverFeature.QUADRATIC_OBJECTIVE,       # QP work package (WP-E) only
        SolverFeature.DIRECT_API,
        SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        SolverFeature.GPU_ACCELERATION,
        SolverFeature.GPU_ONLY,
        SolverFeature.SEMI_CONTINUOUS_VARIABLES,
        SolverFeature.MIP_DUAL_BOUND_REPORT,
    }
)
```

| Feature | Declared | Justification |
|---|---|---|
| `INTEGER_VARIABLES` | **yes** | MILP verified: charter §4.3 and exploration §1 (`MILP optimal status=1 Optimal cat=MIP obj=-2`). Gates `feasible_mip_solvers` in the test suite. |
| `QUADRATIC_OBJECTIVE` | **yes, in WP-E only** | QP verified against HiGHS through `model.matrices.Q` (exploration §12: `Q_cuopt = 0.5 * M.Q` reproduces HiGHS's `-4.28571429`). Declaring it auto-registers cuOpt in `QUADRATIC_SOLVERS` (`solvers.py:4426-4431`) — no manual edit. **Deliberately absent from the WP-A class** so the LP/MILP core is green before QP destabilises it (§4.5). |
| `DIRECT_API` | **yes** | The only io_api (§1). Gates the `("cuopt","direct",False)` row of `params`. |
| `SOLUTION_FILE_NOT_NEEDED` | **yes** | cuOpt returns arrays in-process; `Model.solve` then passes `solution_fn=None` (`model.py:2096-2102`). Same as cuPDLPx. |
| `GPU_ACCELERATION` | **yes** | It is a GPU solver. |
| `GPU_ONLY` | **yes** | No CPU fallback: with no visible device, `Solve` raises `InputValidationError: A CPU-memory problem requires remote execution…` (exploration §13). This is also the flag that drives conftest's auto-skip — declaring it is what keeps CPU CI green. |
| `SEMI_CONTINUOUS_VARIABLES` | **yes** | Exploration §10 measured it on a discriminating model (`vtype='S'` → `obj=0.0` vs `'C'`/`'I'` → `5.0`). Copying cuPDLPx's `NotImplementedError` guard (`:4249-4253`) would be a **false** claim of impossibility. Cost of declaring: zero extra code — `M.vtypes` already carries `'S'` and the `np.where` passes it through. Cost of *not* declaring: `_validate_model` (`:731-737`) would raise `ValueError` on a model cuOpt solves. Backed by one differential test (§8). |
| `MIP_DUAL_BOUND_REPORT` | **yes** | `Solution.get_milp_stats()["solution_bound"]` and `["mip_gap"]` exist (`solution/solution.py:213-222`). `test_milp_model` (`:826-834`) asserts `report.dual_bound is not None` whenever the flag is declared, so the flag is self-proving. Must be negated for `max` under M2. |
| `LP_FILE_NAMES` | no | No file path at all (§1). |
| `READ_MODEL_FROM_FILE` | no | §1 — deliberate, priced, non-goal. |
| `IIS_COMPUTATION` | no | No IIS API on `DataModel`/`Solution` (full public surface enumerated in exploration §10); `Model.compute_infeasibilities` (`model.py:2247+`) only knows gurobi/xpress. |
| `SOS_CONSTRAINTS` | no | Verified absent: no `sos` attribute anywhere, the LP/MPS parser rejects an `SOS` section, and upstream's own test says "SOS is explicitly out of scope" (exploration §10). |
| `INDICATOR_CONSTRAINTS` | no | Verified absent (exploration §10). |
| `SOLVER_ATTRIBUTE_ACCESS` | no | Gurobi-only surface (`linopy/variables.py:1157`). cuOpt's per-entity attribute story would be reduced costs, which are **wrong** for `max`/`<=` on the default path (§5e). |
| `supports_persistent_update` | **False** (default) | `DataModel` has no update API; `Problem.update*` exists but `reset_solved_values()` throws the `DataModel` away and rebuilds (`problem.py:1650-1651`), so even the other entry point offers nothing real. |
| `supports_sign_update` | **False** (default) | Follows from the above. |

---

## 4. Status-mapping table

Keyed on `Solution.get_termination_reason()`, which is `termination_status.name`
(`solution/solution.py:285-289`). The two status enums' member names are globally distinct
wherever they differ in meaning (`solver/solver_wrapper.pyx:79-98`: MILP has
`{NoTermination, Optimal, FeasibleFound, Infeasible, Unbounded, TimeLimit,
UnboundedOrInfeasible}`; LP has `{NoTermination, NumericalError, Optimal, PrimalInfeasible,
DualInfeasible, IterationLimit, TimeLimit, PrimalFeasible, UnboundedOrInfeasible}`), so **one
name-keyed dict is unambiguous** and no private module has to be imported. This satisfies
exploration §1's warning ("a `CONDITION_MAP` keyed on the raw int alone would be wrong")
without the two-dict split.

| problem_category | cuOpt status (int + name) | `TerminationCondition` | Evidence / how forced |
|---|---|---|---|
| LP | `0` `NoTermination` | `internal_solver_error` | Catch-all for malformed input and MIQP; exploration §1/§5/§6 (`err=4`, JSON message). `legacy_status` carries `get_error_message()`. Also the GPU-memory-ceiling outcome (§9 of exploration), so it is a *reachable* status, not theoretical. |
| LP | `1` `Optimal` | `optimal` | exploration §1 |
| LP | `2` `PrimalInfeasible` | `infeasible` | exploration §1 (`x>=2 & x<=1`) |
| LP | `3` `DualInfeasible` | `unbounded` | **Mapping proposed; status never observed** (exploration Unknowns: 7 unbounded variants all returned `11`). Mapped anyway because the name is unambiguous and an unmapped name would silently become `unknown`. Flagged in §11. |
| LP | `4` `IterationLimit` | `iteration_limit` | exploration §1 (`iteration_limit=1`) |
| LP | `5` `TimeLimit` | `time_limit` | exploration §1 (`time_limit=1e-6`) |
| LP | `6` `NumericalError` | `internal_solver_error` | exploration §1/§12 (non-PSD `Q`, `obj=nan`) |
| LP | `7` `PrimalFeasible` | `suboptimal` | exploration §1 (`first_primal_feasible=True`) |
| LP | `11` `UnboundedOrInfeasible` | `infeasible_or_unbounded` | exploration §1 |
| MIP / IP | `0` `NoTermination` | `internal_solver_error` | exploration §1/§12 (MIQP) — but MIQP is rejected pre-solve (§5d), so this is only reached on malformed input |
| MIP / IP | `1` `Optimal` | `optimal` | exploration §1 |
| MIP / IP | `2` `Infeasible` | `infeasible` | exploration §1 (`2x == 1`, x integer) |
| MIP / IP | `3` `Unbounded` | `unbounded` | **Mapping proposed; status never observed** (exploration Unknowns) — same reasoning as LP `3`. |
| MIP / IP | `5` `TimeLimit` | `time_limit` | exploration §1 (subset-sum, market-split) |
| MIP / IP | `8` `FeasibleFound` | `suboptimal` | exploration §1 (knapsack, `node_limit=1`) |
| MIP / IP | `11` `UnboundedOrInfeasible` | `infeasible_or_unbounded` | exploration §1 |
| any | anything not in the map | `unknown` | `.get(reason, TerminationCondition.unknown)` — matches `Highs._solve` (`:1855-1857`). cuPDLPx's `CONDITION_MAP.get(code, code)` (`:4374-4376`) leaks a raw int into a `TerminationCondition` slot; **do not copy that** (I5 applies to current `master`, and `Highs` is the better sibling here). |

`ErrorStatus` (`Success`/`ValidationError`/`OutOfMemoryError`/`RuntimeError`) is **not** a
`TerminationCondition`; it is read alongside and folded into `status.legacy_status` when
non-zero.

Two traps recorded in code comments:
* A `time_limit`/`iteration_limit` setting can still return `Optimal` — never infer a limit hit
  from the settings (exploration §1).
* LP `TimeLimit` returned `obj=+0` with a **zero** primal vector. `safe_get_solution`
  (`:1184-1201`) treats `time_limit` as `SolverStatus.ok` and will therefore parse that
  solution. That is the existing repo contract for every solver and we do not change it, but
  the docs' limitations section says so explicitly.

---

## 5. Design decisions on the hard findings

### 5a. Presolve + maximize negated duals → adopt **M2 (always minimise)**

**Decision: adopt M2 unconditionally. linopy never calls `set_maximize(True)`.**

For a `max` model, hand cuOpt `min (−c)ᵀx` (and `−Q` for QP), leave `set_maximize(False)`, then
negate `get_primal_objective()`, `get_dual_solution()` and the MIP `solution_bound` on the
linopy side.

Why, in order of weight:
1. It is the only mitigation verified to fix **all** the failing cases: `FAILING CASES: 5/21`
   without it, `0/21` with it (exploration §2b).
2. M1 (`presolve=0/1`) demonstrably does **not** fix ranged rows — `presolve=1` still lands on
   `solved_by=Unset` and still returns `dual=[-1.]` against HiGHS's `[1.]` (exploration §2a).
   A partial fix for a silent sign error is worse than none, because it makes the six-cell
   matrix pass.
3. Forcing `method=DualSimplex` also fixes it, and is rejected: it routes the solve onto the
   CPU simplex, i.e. it "fixes" a GPU solver by not using the GPU.
4. The root cause is **unknown** (exploration Unknowns: correlation with `solved_by == Unset`
   is 6/6 over 144 cells, but the C++ is a stripped `.so`). M2 avoids the entire branch **by
   construction**, which is the right shape of mitigation for an undiagnosed upstream bug —
   it does not depend on the bug's boundary being where we think it is.
5. `max|Δdual| = 3.61` on the failing cells: not a tolerance question, a correctness question.

Cost: two sign multiplications at build time, two negations at read time, and a comment. The
comment matters more than the code, because the **repo's own template does the opposite**:
`cuPDLPx._solve` negates duals for `max` (`:4385-4386`) *because* it sets `MAXIMIZE`. An
implementer following I5 by copying that line into a class that never sets `MAXIMIZE` would
introduce a systematic sign error in every maximised model. The code comment names cuPDLPx and
says "do not harmonise"; the class docstring says linopy's dual convention is HiGHS's
(`Highs._solve` passes `solution.row_dual` through unchanged, `:1871-1879`).

Note `maybe_adjust_objective_sign` (`:389-401`) is a no-op for `io_api="direct"`, so the
objective negation is explicit in `_solve`, not delegated.

**Residual risk, named:** M2 makes the *objective sign* and the *dual sign* depend on one
`sense == "max"` branch. If `self.sense` were ever wrong, both flip together and silently. The
V1 sign matrix (§8) is the guard, and it asserts objective *and* duals against live HiGHS in
all six cells.

### 5b. Concurrent-method repeat-solve segfault → default `method = 3` (Barrier)

**Decision: when the user has not set `method`, linopy sets `method = 3` (`SolverMethod.Barrier`).**

```python
# cuOpt's default method (0, Concurrent) crashes the interpreter with SIGSEGV
# on the second or third sequential solve for models above ~1300 variables
# (deterministic, reproduced in >=12 processes). Barrier is unaffected, runs
# on the GPU, and was the most accurate of the four methods when its duals
# were compared against HiGHS. Users can still override `method`.
self.solver_options.setdefault("method", self._DEFAULT_METHOD)
```

Why Barrier and not the alternatives:

| Option | Verdict |
|---|---|
| Do nothing (Concurrent) | **Rejected.** "linopy's own test models sit below the threshold, so the suite would go green while real user scripts die with SIGSEGV and no traceback" (exploration §9). Charter: "Doing nothing is not defensible." |
| `method=2` DualSimplex (exploration's own suggestion) | **Rejected.** It was the fastest *single* method in the spike runs, but dual simplex is the CPU leg of cuOpt's concurrent architecture. Defaulting a solver that declares `GPU_ONLY` and `GPU_ACCELERATION` onto a CPU algorithm is a false advertisement, and it would make every benchmark in the docs a lie. |
| `method=1` PDLP | **Rejected.** Worst measured accuracy: dual agreement with HiGHS only `4.71e-05` (vs Concurrent `8.18e-08`, Barrier `1.73e-09`), primal objective drift `2.80013079` vs `2.8`, `0` returned for `max`/`>=` reduced costs, plus the exploration's own §"Surprises": a badly-scaled LP under PDLP with no time limit **did not terminate** (13 min at 98 % GPU, killed by `timeout 900`), and `PDLP + time_limit=1e-6 + save_best_primal_so_far` segfaults. Too many sharp edges for a default. |
| `method=3` Barrier | **Chosen.** GPU-resident; 25/25 varied sequential solves with flat device memory (exploration §9, `b_q9d_out.txt`); **best measured dual agreement of all four methods** (`1.73e-09` / `2.47e-09`, exploration §2c); and it is already the method cuOpt silently forces for every QP (exploration §12), so LP, MILP and QP all run one code path. |

Supporting decisions:
* Implemented with `setdefault`, so `model.solve("cuopt", method=0)` still reaches Concurrent.
  We do **not** block the buggy method — that would be paternalistic and would break anyone
  reproducing cuOpt's own benchmarks. We change the default and document the hazard.
* The docs' "Known limitations" states the upstream bug, its threshold (~1300 variables, 2nd–3rd
  solve), and that `method=0` re-enables it.
* **Acceptance criterion (frozen, §8):** a GPU-gated test solves a 2000-variable LP **three
  times in one process** and compares each objective against live HiGHS. If it crashes, or if
  the objective misses its tolerance, the WP **reports the failure verbatim and stops on that
  item** — it does not try `method=2` instead. Changing the default method is a Gate-3
  decision, not a fix agent's call.
* **UNVERIFIED — WP-A must verify:** exploration's repeat-solve matrix used LPs only. WP-A must
  confirm (i) `method=3` solves MILP correctly (the existing `test_milp_model` covers this once
  cuopt is in `params`), and (ii) `method=3` is accepted, not rejected, for a MIP. If cuOpt
  errors on `method` for MIP, the fallback is to apply the default only when the model has no
  integers — and that is a reportable deviation, logged in `30-decisions.md`.

### 5c. `is_available()` on a GPU-less machine

**Decision: `is_available()` requires an importable `cuopt` package *and* a visible CUDA
device. §4.6 constraint 3 wins over the cuPDLPx/I5 precedent.**

The fork is real and exploration §13 states it fairly: `cuopt/__init__.py` is deliberately
GPU-free ("allows cuopt to be imported on CPU-only hosts when remote solve is configured"), so
`_has_module("cuopt")` is `True` with `CUDA_VISIBLE_DEVICES=''`, and the resulting error is
`InputValidationError: A CPU-memory problem requires remote execution. Set CUOPT_REMOTE_HOST
and CUOPT_REMOTE_PORT…`.

I resolve it for §4.6 because:
* §4.6 constraint 3 is stated by the **human lead** as binding and is a Definition-of-Done
  item: "`linopy.available_solvers` must simply omit cuOpt and `model.solve("cuopt")` must fail
  with an actionable message — never a bare CUDA traceback."
* I5 is about **linopy-side style** ("match repo precedent over external precedent"). It is not
  a licence to import a *functional* defect. cuPDLPx's import probe is correct for cuPDLPx,
  whose package cannot be imported without CUDA; it is the wrong probe for a package that
  deliberately imports fine without a GPU. Matching the *letter* of the precedent here breaks
  the *purpose* of the precedent, which is that `available_solvers` means "you can use this".
* `_AvailableSolvers`' docstring ("Membership means the solver's Python package or binary is
  importable — it does not mean a working **license** exists", `:4464-4470`) draws the line at
  licensing, and there is no license here. A missing GPU is closer to a missing *binary* than
  to a missing license.

Implementation (new module-level helper next to `_has_module`, `:307-314`):

```python
@functools.cache
def _has_cuda_device() -> bool:
    """
    True if at least one CUDA device is visible to this process.

    Uses the CUDA driver API, which does not create a context and therefore
    allocates no device memory. Returns False when the driver is missing or
    too old to initialise.
    """
```

Probe order (first that is importable wins), **all UNVERIFIED — WP-A must verify both the
positive and the negative case before landing**:
1. `from cuda.bindings import driver` → `driver.cuInit(0)`, `driver.cuDeviceGetCount()`.
   `cuda-bindings==12.9.7` is installed by the extra (exploration §13's install manifest), so
   it is present wherever cuOpt is. Preferred because it is a supported Python API.
2. Fallback `ctypes.CDLL("libcuda.so.1")` → `cuInit(0)`, `cuDeviceGetCount(&n)`. Zero new
   dependencies, stable driver ABI; a missing `libcuda.so.1` *is* the answer "no usable GPU".

Everything wrapped in `try/except Exception: return False`, so a probe that itself breaks
degrades to "unavailable" rather than to a traceback at import time.

```python
@classmethod
@functools.cache
def is_available(cls) -> bool:
    # ``import cuopt`` deliberately succeeds without a GPU (it supports a
    # remote-solve mode), so an import probe alone would advertise cuOpt on
    # CPU-only machines.
    if not _has_module("cuopt"):
        return False
    if not _has_cuda_device():
        logger.warning(
            "cuopt is installed but no CUDA device is visible, so the cuopt "
            "solver is unavailable. cuOpt needs a Linux host with an NVIDIA "
            "driver >= 525.60.13 and a GPU of compute capability >= 7.0; "
            "check CUDA_VISIBLE_DEVICES."
        )
        return False
    return True
```

**Exactly what happens on a GPU-less machine that has `cuopt-cu12` installed:**

| Call | Behaviour |
|---|---|
| `linopy.available_solvers` | does **not** contain `"cuopt"`. On first probe, one `logger.warning` naming the driver and compute-capability requirements and `CUDA_VISIBLE_DEVICES`. (`functools.cache` means at most once per process.) |
| `linopy.quadratic_solvers`, `licensed_solvers` | likewise omit it (both gate on `is_available()`, `:4479-4484`, `:4530-4535`). |
| `model.solve("cuopt")` | `Model.solve` asserts membership at `linopy/model.py:2078` → `AssertionError: Solver cuopt not installed`, preceded in the log by the warning above. |
| `solvers.cuOpt()` / `Solver.from_name("cuopt", …)` | `Solver.__post_init__` (`:475-480`) raises `ImportError: Solver package for 'cuopt' is not installed. Please install first to initialize solver instance.` |
| `to_cuopt(m)` | same `ImportError` (it goes through `from_model`). |
| `check_solver_licenses("cuopt")` | `LicenseStatus(ok=False, message="package not installed")`. |
| what a user actually sees | the `logger.warning` — which is the actionable part — followed by a generic "not installed" exception. |

**Known imperfection, declared rather than hidden:** the *exception text* says "not installed"
when the package **is** installed. Fixing that properly means touching
`Solver.__post_init__`/`Model.solve` messages shared by twelve other solvers, which is an I4
drive-by. So the actionable message is delivered by the warning, and this residual is
registered as **C-R18** and named in §11. If the Gate-3 reviewer disagrees, the alternative is a
one-line change at `model.py:2078`/`solvers.py:476` to append "or unavailable in this
environment" for all solvers — a shared-message change I decline to make on cuOpt's behalf.

Corollary for the docs: `doc/gpu-acceleration.rst` must state that `"cuopt" in
available_solvers` is `False` on a machine without a visible GPU **even when the package is
installed**, and that this differs from cuPDLPx.

### 5d. QP

**Decision: ship QP as its own work package (WP-E), with these four rules.**

1. **`Q_cuopt = 0.5 * model.matrices.Q`, kept full-symmetric, converted to CSR.** cuOpt
   minimises `cᵀx + xᵀQx` and internally symmetrises to `Q + Qᵀ`
   (`data_model/data_model.py:287-288`), so its effective Hessian is `Q + Qᵀ`; linopy's
   `matrices.Q` **is** the Hessian (`expressions.QuadraticExpression.to_matrix` docstring:
   "the quadratic term is 0.5 x^T Q x … the diagonal terms are doubled"), which is also why
   `Highs` passes `triu(M.Q)` to `passHessian` (`:1716-1720`). Verified live against HiGHS:
   `0.5*M.Q` → `-4.28571429` (matches), `M.Q` → `-2.14285714`, `triu(M.Q)` → `-2.12903226`
   (exploration §12). The near-misses return `Optimal` with a wrong objective, so this is a
   silent-wrong-answer site and gets both a comment and a dedicated test.
2. **`Q → −Q` under M2** for `max`-sense QP, alongside `c → −c` (validated 4/4 QP cells,
   exploration §12/§2b). Implemented by the single `sign` factor in `_build_solver_model`, so
   the two negations cannot drift apart.
3. **MIQP guard.** `model.type` is `"MIQP"` (mixed) or `"IQP"` (all-integer) —
   `linopy/model.py:1666-1679`. **Both** must be rejected:
   ```python
   if model.is_quadratic and model.type != "QP":
       raise NotImplementedError(
           "cuOpt does not support quadratic objectives with integer variables "
           f"(problem type {model.type}). Use a solver that supports MIQP "
           "(gurobi, xpress, copt), or relax the integrality."
       )
   ```
   Without the guard cuOpt returns `NoTermination`, `obj=0.0`, an **empty** solution array and
   `error_status=4` — no exception (exploration §12). Guarding on `model.type in ("MIQP",)`
   alone would miss the all-integer case; that is a concrete trap in exploration §12's wording
   ("linopy's `model.type` is `"MIQP"` for such a model") and this plan fixes it with the
   `model.py:1666-1679` citation.
4. **Non-PSD Q: no pre-check, honest status.** `Q + Qᵀ` must be PSD; a non-PSD Q gives
   `NumericalError` with `obj=nan`, no crash and no bogus answer (exploration §12). It maps to
   `internal_solver_error` (§4) and `legacy_status` carries `NumericalError` plus any error
   message. We do **not** compute eigenvalues to pre-detect it: that is O(n³), linopy does not
   do it for any other solver, and cuOpt's own detection is reliable. `test_quadratic_model_unbounded`
   (`:955-965`) accepts `["unbounded","unknown","infeasible_or_unbounded"]` — **UNVERIFIED — WP-E
   must check which of these cuOpt's unbounded-QP path produces**; if it produces
   `internal_solver_error`, that is a finding to report, and the honest resolution is a
   cuOpt-specific branch in that test with the measured status pasted, not a widened accept-list.

QP also inherits two facts worth a comment: `method` and `crossover` are **silently overridden**
to Barrier for every QP (exploration §12) — so a user's `method="PDLP"` is ignored, which the
docs state — and non-binding QP duals come back as `~1e-10` rather than exact `0`, which is why
every dual tolerance in §8 carries an `atol`, not just an `rtol`.

One more QP interaction found by reading the tests, not in the exploration:
**`test_quadratic_model_wo_constraint` (`test/test_optimization.py:929-944`) removes `con0` and
then requires `condition == "optimal"`.** That is a QP with **no constraints**, and
exploration §5 says cuOpt returns `NoTermination`/`obj=nan` whenever `nnz == 0`. The padding
row in §2.1 exists for exactly this — see §11 R3 for the fallback if it does not hold.

### 5e. Reduced costs

**Decision: do nothing. `get_reduced_cost()` is never called.**

`grep -rn "reduced_cost\|ReducedCost\|reduced cost" linopy/` returns no hits, and
`linopy.constants.Solution` (`:274-286`) carries only `primal`, `dual`, `objective`. There is
nowhere to put them. Independently, cuOpt's reduced costs are **wrong** for maximised models
with `<=` rows on the default path (returning `−c − Aᵀy` instead of `c − Aᵀy`), and exploration
found **no single transformation** covering `max`/`<=` and `max`/`>=` together — so anyone
"fixing" this by guessing a sign would make it worse.

Concretely:
* `SolverFeature.SOLVER_ATTRIBUTE_ACCESS` is **not** declared, so
  `model.variables.get_solver_attribute("RC")` raises `NotImplementedError` — which
  `test_solver_attribute_getter` (`:1153-1164`) already asserts for every non-gurobi solver.
* A one-line comment in `_solve` records that `get_reduced_cost()` exists and is deliberately
  unused, with a pointer to the defect, so nobody adds it later as an "easy win".
* The upstream defect is recorded in `logs/45-known-issues.md` by WP-A (it is a *cuOpt* bug, so
  I13's "record, don't fix" applies by analogy — and it must not be silently forgotten).

### 5f. Solver options

**Decision: follow `Highs._set_solver_params` (`:1774-1784`) with three additions —
type-directed bool coercion, an error message that names the offending option, and no silencing
of the console.**

```python
def _set_solver_params(self, settings: Any, log_fn: Path | None = None) -> None:
    if log_fn is not None:
        self.solver_options["log_file"] = path_to_string(log_fn)
        logger.info(f"Log file at {self.solver_options['log_file']}")
    # cuOpt's default method segfaults on repeat solves; see the class docstring.
    self.solver_options.setdefault("method", self._DEFAULT_METHOD)
    for key, value in self.solver_options.items():
        try:
            settings.set_parameter(key, self._coerce_option(settings, key, value))
        except ValueError as err:
            raise ValueError(
                f"Invalid solver option {key!r} for cuOpt: {err}. cuOpt option "
                "names are lower-case and snake_case, e.g. 'time_limit'."
            ) from err
```

* **`log_fn` / `log_file`.** cuOpt's parameter is spelled **identically to HiGHS's**
  (`log_file`), so the HiGHS line transfers verbatim, and cuOpt is strictly better off than
  cuPDLPx, which warns-and-ignores (`:4362-4363`). Verified: 1437 chars written, full banner +
  iteration log; `pathlib.Path` is also accepted but `path_to_string` is used for consistency
  (exploration §4).
  **Precedence, resolving exploration's open question:** `log_fn` **wins** over a user's
  `solver_options["log_file"]`, because that is exactly what `Highs._set_solver_params` does
  (it assigns into `solver_options` before the loop) and precedence should not differ between
  two solvers whose option is spelled the same. Documented in the class docstring.
* **`log_to_console`.** Left at cuOpt's default (`true`). HiGHS also prints to console by
  default; `log_fn` writes to both file and terminal (exploration §4 §A). The docstring names
  `log_to_console: False` as the way to silence it, mirroring how `cuPDLPx`'s docstring
  documents `LogToConsole` (`:4166`).
* **bool → int coercion.** `set_parameter` is type-strict: `presolve=False` raises
  `ValueError: Parameter presolve value False is not an integer`, while `log_to_console=False`
  is accepted (exploration §6, closing note). A blanket `bool → int` cast would risk breaking
  the genuinely boolean parameters, so the coercion is **type-directed**: ask cuOpt what type
  the parameter currently holds and only cast when it is an `int` that is not a `bool`.
  ```python
  @staticmethod
  def _coerce_option(settings: Any, key: str, value: Any) -> Any:
      # set_parameter is type-strict: integer-typed parameters reject Python
      # bools (`presolve=False` raises), while boolean ones accept them.
      if isinstance(value, bool):
          default = settings.get_parameter(key)
          if isinstance(default, int) and not isinstance(default, bool):
              return int(value)
      return value
  ```
  `get_parameter` guards on the same name list as `set_parameter`
  (`solver_settings.pyx:166-167`, `:191-193`), so an unknown name raises there and is caught by
  the same `except ValueError` — one error path, not two.
  **Acceptance criterion (WP-A):** `model.solve("cuopt", presolve=False)` and
  `model.solve("cuopt", log_to_console=False)` must both work, with output pasted.
* **Unknown option names.** cuOpt raises eagerly at `set_parameter` with
  `ValueError: Invalid parameter. Please check documentation` — which does **not** name the
  offending parameter (exploration §7). Since the most likely user error is Gurobi/cuPDLPx
  habit (`TimeLimit` instead of `time_limit`), the wrapper adds the name and the naming rule.
  Invalid *values* are raised late, inside `Solve`, with messages that already name parameter
  and value ("Parameter time_limit value -5.0 out of range") — those pass through untouched.
* **No default `time_limit`.** Tempting (exploration's §"Surprises" 3 saw a non-terminating
  PDLP solve), but a silent default time limit would turn "slow" into "wrong answer, status
  TimeLimit" for large models, and no other linopy solver imposes one. The non-termination was
  PDLP-specific; the default is Barrier. Documented as a limitation instead.
* **`save_best_primal_so_far` is never set by linopy** (it segfaults in combination with
  `method=PDLP` + a tiny `time_limit`). Users can still pass it; the docs warn.

### 5g. Interruptibility → **adopt the HiGHS worker-thread pattern**

**Decision: adopt it, minus the cancel step.**

Measured: SIGINT during `Solve()` is deferred for the whole remaining solve — 52.9 s of
unresponsiveness — and there is **no cancel API** anywhere on the in-process solver
(`grep -rni "cancel|abort|interrupt|sigint|keyboard"` matches only the gRPC client and the
routing module). Running `Solve()` in a daemon worker thread while the main thread waits on
`Event.wait(0.1)` gives **0.00 s** Ctrl-C latency, and the interpreter tears down cleanly with
`exit=0` even while the daemon thread is still inside cuOpt's C++ code (exploration §8).

```python
def _run_cuopt_with_keyboard_interrupt(data_model: Any, settings: Any) -> Any:
    """
    Run ``cuopt.Solve`` while ensuring Ctrl-C returns control immediately.

    cuOpt spends its whole runtime inside a C extension call and installs no
    signal handler of its own, so a SIGINT is only delivered once ``Solve``
    returns. Running it in a worker thread lets the main thread raise
    KeyboardInterrupt at once. cuOpt exposes no cancel API, so the GPU work
    continues in the background until the process exits; ``time_limit`` is the
    only bound cuOpt offers on solve duration.
    """
```

Shape mirrors `_run_highs_with_keyboard_interrupt` (`:189-252`) but is much shorter — no
`HandleKeyboardInterrupt` attribute dance, no `cancelSolve`: start thread, `while not
finished.wait(0.1)`, re-raise the worker's exception if any, and on `KeyboardInterrupt` re-raise
immediately **without** joining (verified safe: `exit=0`, "no hang, no crash, no CUDA teardown
error"). ~35 lines including the docstring.

Rejected alternative — call `Solve()` inline and document the hazard: a 53-second-unresponsive
Ctrl-C in an interactive session is precisely the failure users report as "linopy hangs", the
fix is verified and precedented in this very file, and 35 lines is a fair price. Adopting is
also the I5-consistent choice: a solver that runs long inside C gets the worker-thread
treatment here.

The docstring's honesty about "GPU work continues until process exit" is required — it is the
part a reader cannot infer from the HiGHS analogue.

### 5h. Warm start → **defer, and never touch the initial-point API**

**Decision: `warmstart_fn is not None` → `NotImplementedError`, mirroring `cuPDLPx._solve`
(`:4352-4355`). `basis_fn` → `logger.warning` and ignore. `DataModel.set_initial_primal_solution`
is never called, on any path.**

Why defer:
* linopy's warm-start surface is a **file path** (`warmstart_fn: Path`). cuOpt's warm-start
  payload is a live `PDLPWarmStartData` Python object with 17 host arrays — PDLP algorithm
  state, not a basis and not a file format. Exposing it through a `Path` parameter would mean
  inventing a serialisation format for another project's internal solver state. That is a
  design decision for a later PR with an API discussion, not a line in this one.
* Even in-process it needs **three simultaneous non-default settings**: `method=1` (PDLP),
  `pdlp_solver_mode=1` (Stable2), `presolve=0`. Each has a cost: PDLP is the least accurate
  method (§5b), and `presolve=0` is forced only because the payload comes back in **presolved
  coordinates** and otherwise fails validation against the identical model
  (`Exception: Invalid PDLPWarmStart data…`). Upstream's docstring is also wrong about which
  modes work — `Fast1` returns `NoTermination`/`nan`, and the *default* `Stable3` silently
  produces `nan` (exploration §11).
* PDLP warm start is LP-only (`get_pdlp_warm_start_data` raises for MILP).

Why the initial-point API is a hard prohibition, not a preference:
`DataModel.set_initial_primal_solution` on an **LP** raises
`RuntimeError: __copy:: D->H: failed: cudaErrorIllegalAddress` and **poisons the CUDA context
for the rest of the process** — every later `Solve()` then raises. Reproduced in a fresh
process with all-default settings and a correctly-sized, C-contiguous `float64` array
(exploration §11). One warm-start attempt would take down an entire pytest session or user
script. A code comment naming `cudaErrorIllegalAddress` sits next to the `NotImplementedError`
so a future contributor cannot mistake this for unexplored territory.

`test_basis_and_warmstart` gains cuopt to its existing skip (`:1108-1111`), with cuOpt's own
reason. Docs' limitations section says: no warm start, and why.

### 5i. The pyproject extra

**Decision: adopt exploration item 13's verified candidate verbatim, name `cuopt`.**

```toml
# NVIDIA cuOpt (GPU). Deliberately NOT part of `solvers`: it pulls ~34 packages
# and several GB of CUDA wheels. Linux-only wheels, no sdist, so the marker keeps
# macOS/Windows resolution unaffected. Requires a CUDA 12 driver (>= 525.60.13)
# and a GPU of compute capability >= 7.0.
cuopt = [
    "cuopt-cu12>=26.8; platform_system == 'Linux'",
]
```

Placed immediately **after** the `solvers = [...]` block (ends `pyproject.toml:113`), as a
sibling reachable from no other extra (this project has no `all` extra).

* **Name `cuopt`, not `gpu`.** It matches the solver name, the `SolverName` value and the
  file's existing lower-case single-word extras (`oetc`, `remote`, `docs`, `dev`, `solvers`,
  `benchmarks`); and `gpu` is a promise the extra cannot keep — it installs cuOpt specifically,
  on Linux only, for CUDA 12 only, and would collide the day a second GPU solver is added
  (`cupdlpx` is already commented out of `solvers`, `:112`). Charter §5's `<gpu-extra>` is a
  placeholder, not a name. I agree with the spike's recommendation.
* **Marker `platform_system == 'Linux'`** — the positive twin of the file's existing
  `"cplex; platform_system != 'Darwin'"` (`:105`). It must be an equality, not
  `!= 'Darwin'`, because there are no Windows wheels either. No `python_version` marker: cuOpt
  ships cp311–cp314 and linopy already declares `requires-python = ">=3.11"` (§4.2).
* **Floor-only version spec** matches the file's style (`gurobipy`, `highspy>=1.5.0`,
  `knitro>=15.1.0`). No upper cap: §4.2's note about `26.10.*` concerns NVIDIA's own index, not
  PyPI resolution, and a speculative cap would age badly.
* **No `[tool.uv]` index entry.** The whole tree is on PyPI (§4.2) and index settings are not
  propagated through published metadata anyway (§4.6).
* **Acceptance test (WP-B), all four §4.6 constraints, pasted:**
  1. clean venv → `pip install '<repo>[cuopt]'` with **no** extra index and **no** resolver
     flags → exit 0, then `python -c "import linopy; print(linopy.available_solvers)"` must now
     print a list **containing `"cuopt"`** (unlike exploration §13's `[]`, because the solver
     class exists by then — this is the acceptance test the spike explicitly deferred to us);
  2. `pip install --dry-run --report` for `linopy[solvers]` contains no
     `cuopt|cudf|libcuopt|rmm|pylibraft|nvidia|cuda` package;
  3. `uv pip compile --python-platform windows` and `--python-platform macos` with
     `--extra cuopt` both resolve, with no `cuopt-cu12`;
  4. `uv pip compile --python-platform linux --extra cuopt` resolves **with** `cuopt-cu12`.
* **After this lands, I9's end condition is met** for `uv sync --extra cuopt` — the lead
  decides when to relax it; the WP does not change its own invariants.

---

## 6. Requirement-ID map

C-R1…C-R17 are the §5 table rows, numbered top to bottom. C-R18+ are new.

| ID | §5 row | Satisfied by |
|---|---|---|
| **C-R1** | `solvers.py:148` `SolverFeature` | §3 feature set on `class cuOpt` (7 flags in WP-A, `QUADRATIC_OBJECTIVE` added in WP-E). No edit to the enum. **WP-A, WP-E** |
| **C-R2** | `solvers.py:176` `TYPE_CHECKING` | `import cuopt` added to the `TYPE_CHECKING` block. **WP-A** |
| **C-R3** | `solvers.py:304` lazy module | `cuopt = _LazyModule("cuopt.linear_programming")` + the `cuopt.routing` comment (§1). **WP-A** |
| **C-R4** | `solvers.py:346` `SolverName` | `cuOpt = "cuopt"` (vendor spelling; member name == class name). **WP-A** |
| **C-R5** | `solvers.py:4156` new class | `class cuOpt(Solver[None])` with the override set in §2.1. **WP-A** (+ QP path in **WP-E**) |
| **C-R6** | `solvers.py:4440` `_SOLVER_PROBE_ORDER` | `"cuopt"` inserted between `"copt"` and `"cupdlpx"`. **WP-A** |
| **C-R7** | `solver_capabilities.py` registry | **Zero-line diff, proved by test:** `test_cuopt_registered_in_capability_shim` + the existing `test_capability_shim_round_trips` (`test_solvers.py:479`) which auto-covers the new enum member. **WP-D** |
| **C-R8** | `io.py:33`, `:792` | `TYPE_CHECKING` alias + `to_cuopt(m)`. **WP-C** |
| **C-R9** | `model.py:73`, `:2517` | import + `to_cuopt = to_cuopt` binding. **WP-C** |
| **C-R10** | `constants.py:171` | **Zero-line diff**; §4's map uses only existing members. Proved by the status tests in **WP-D**. |
| **C-R11** | `test/conftest.py:57-79` | **Zero-line diff expected**, *proved* by running `pytest -k cuopt` (skips) and `pytest --run-gpu -k cuopt` (runs) with output pasted. **WP-D** |
| **C-R12** | `test_optimization.py:562/685/1110` | `time_limit_option["cuopt"]`, `cuopt_solvers` selector, warm-start skip, plus the new cuOpt tests. **WP-D** (+ QP tests **WP-E**) |
| **C-R13** | `test_solvers.py` | feature-table rows + the shim test. **WP-D** |
| **C-R14** | `pyproject.toml` | the `cuopt` extra (§5i) and its four-part clean-venv acceptance test. **WP-B** |
| **C-R15** | `doc/gpu-acceleration.rst` | new "cuOpt" section: install, hardware/driver, usage, supported classes, limitations. **WP-F** |
| **C-R16** | `prerequisites/contributing/index/api/release_notes/README` | §9 table. **WP-F** |
| **C-R17** | `solvers.py:4426` `QUADRATIC_SOLVERS` (QP) | **Read-only anchor.** `QUADRATIC_OBJECTIVE` + `0.5*M.Q` build path + MIQP guard + QP tests. **WP-E** |
| **C-R18** | *new* | **Honest GPU degradation** (§5c): `_has_cuda_device()`, the device-aware `is_available()`, the actionable `logger.warning`, and the docs sentence stating cuOpt is absent from `available_solvers` without a GPU. Source: §4.6 constraint 3 + exploration §13. **WP-A** (docs part **WP-F**) |
| **C-R19** | *new* | **M2 always-minimise** (§5a): sign factor at build, objective/dual/`solution_bound` negation at read, the "do not copy cuPDLPx" comment, and the V1 sign matrix incl. the presolve-solvable `max` case. Source: exploration §2a/§2b. **WP-A**, tests **WP-D** |
| **C-R20** | *new* | **Non-default `method`** (§5b): `_DEFAULT_METHOD = 3`, `setdefault`, comment, docs limitation, and the 3×-sequential-solve regression test above the ~1300-variable threshold. Source: exploration §9. **WP-A**, test **WP-D** |
| **C-R21** | *new* | **MIQP/IQP rejection** (§5d.3): `NotImplementedError` covering both `model.type` spellings, with a test. Source: exploration §12. **WP-E** |
| **C-R22** | *new* | **Interrupt-safe solve** (§5g): `_run_cuopt_with_keyboard_interrupt`. Source: exploration §8. **WP-A** |
| **C-R23** | *new* | **No-nonzero-constraint handling** (§2.1 padding row / §11 R3 fallback), so `nnz == 0` cannot silently return `NoTermination`/`nan`; plus mapping `NoTermination` to `internal_solver_error` with `get_error_message()` in `legacy_status`. Source: exploration §5/§6 and `test_optimization.py:929-944`. **WP-A** |
| **C-R24** | *new* | **Semi-continuous support declared honestly** (§3): `'B'→'I'` mapping, `'S'` passthrough, `SEMI_CONTINUOUS_VARIABLES` declared, one differential test, and cuPDLPx's guard deliberately **not** copied. Source: exploration §10. **WP-A**, test **WP-D** |

No ID is unmapped. C-R7, C-R10, C-R11 are satisfied by **proof of a zero-line diff**, which is
the charter's own reading of those rows.

---

## 7. Work-package breakdown

Rules honoured: `solvers.py` core first and alone; then disjoint parallel packages; QP last as
its own package; packaging first-class and early; no two packages holding one file at once.

| WP | Name | Files (exclusive) | Runs | Req. IDs | Commit |
|---|---|---|---|---|---|
| **WP-A** | solvers core (LP + MILP) | `linopy/solvers.py` | **first, alone** | C-R1…C-R6, C-R10, C-R18, C-R19, C-R20, C-R22, C-R23, C-R24 | 1 |
| **WP-B** | packaging | `pyproject.toml` | **parallel with WP-A** | C-R14 | 2 |
| **WP-C** | io + model | `linopy/io.py`, `linopy/model.py` | after WP-A merges | C-R8, C-R9 | 1 |
| **WP-D** | tests (LP/MILP) | `test/test_optimization.py`, `test/test_solvers.py`, (`test/conftest.py` read-only) | after WP-A merges; parallel with WP-C | C-R7, C-R11, C-R12, C-R13 | 3 |
| **WP-E** | QP | `linopy/solvers.py`, `test/test_optimization.py` | after WP-A, WP-C **and WP-D** merge | C-R1 (delta), C-R5 (delta), C-R17, C-R21 | 4 |
| **WP-F** | docs + release notes | `doc/*.rst`, `README.md`, `examples/using-solvers.ipynb` | last, after WP-E | C-R15, C-R16, C-R18 (docs part) | 5, 6 |

File-collision check: WP-A and WP-E both hold `linopy/solvers.py` → strictly sequential
(WP-E is last but one). WP-D and WP-E both hold `test/test_optimization.py` → sequential.
WP-B, WP-C, WP-F touch files no other package touches. WP-A ∥ WP-B is the only concurrency in
the first wave; WP-C ∥ WP-D the only one in the second.

Commit plan (~6 substantive commits, one concern each, 1–3 files apiece — #516's shape, plain
imperative style per D2):

| # | Message | Files | From |
|---|---|---|---|
| 1 | `Implement interface to cuOpt solver and required IO` | `solvers.py`, `io.py`, `model.py` | WP-A + WP-C |
| 2 | `Add cuOpt package dependency` | `pyproject.toml` | WP-B |
| 3 | `Update tests` | `test/test_optimization.py`, `test/test_solvers.py` | WP-D |
| 4 | `Add support for quadratic objectives with cuOpt` | `solvers.py`, `test/test_optimization.py` | WP-E |
| 5 | `Add to docs` | `doc/*`, `README.md`, `examples/using-solvers.ipynb` | WP-F |
| 6 | `Add comment for release notes` | `doc/release_notes.rst` | WP-F |

Commit 2 lands before 1 if WP-B finishes first; the lead orders at merge time. Either order
keeps six linear commits with no merges (§6.4a).

### Acceptance criteria per WP

**WP-A — solvers core.** Concrete, all with pasted output:
1. `/home/azureuser/.local/bin/ruff check .` and `.venv/bin/mypy` clean.
2. `python -c "import linopy; print(linopy.available_solvers)"` includes `"cuopt"`.
3. `python -c "…"` proving `SOLVER_REGISTRY["cuopt"].features == cuOpt.supported_features()`
   and that the 7 declared / 6 denied flags of §3 match.
4. A scratch script (not committed) solving the charter §4.3 LP through `model.solve("cuopt")`
   and printing objective + duals, cross-checked against `model.solve("highs")` **live in the
   same process**.
5. **Six-cell sign matrix** run by hand before writing tests: all six agree with live HiGHS
   within the §8 tolerances, *and* the presolve-solvable `max` case agrees.
6. `presolve=False` and `log_to_console=False` both accepted (§5f).
7. Three sequential solves of a 2000-variable LP in one process: no SIGSEGV (§5b).
8. `_has_cuda_device()` verified **both ways**: `True` normally, `False` under
   `CUDA_VISIBLE_DEVICES=''` — and in the latter case `available_solvers` omits `cuopt` and the
   warning text appears (§5c).
9. `python -c "from linopy.solvers import cuOpt; print(cuOpt.__doc__)"` — docstring documents
   `time_limit`, `method`, `log_to_console`, the tolerances, and the three limitations.
10. No `QUADRATIC_OBJECTIVE` in `features` yet; `model.solve("cuopt")` on a QP raises the base
    class's `ValueError` from `_validate_model` (`:726-729`).

**WP-B — packaging.** The four-part test of §5i, all pasted, from a venv that has never seen
cuOpt; plus `du -sh` on that venv; plus cleanup of the temporary venv/worktree.

**WP-C — io + model.** `to_cuopt(m)` returns an object whose
`get_objective_coefficients()` round-trips `M.c` (with the M2 sign for a `max` model — the WP
must assert the sign explicitly, because a silent sign flip here is invisible);
`model.to_cuopt()` works as a bound method; `ruff`/`mypy` clean; `doc/api.rst` entry deferred
to WP-F.

**WP-D — tests.** `pytest -q` (no GPU) and `pytest --run-gpu -q` both green with **no
regression vs `logs/05-baseline-pytest*.txt`**, output pasted; the cuOpt tests demonstrably
**ran** under `--run-gpu` (`-k cuopt -v` output pasted) and demonstrably **skipped** without
it; every V1 row of `logs/15-verification.md` has a test.

**WP-E — QP.** `test_quadratic_model*` pass for cuopt; the `0.5*M.Q` convention test passes and
**fails when `0.5` is removed** (the WP must demonstrate that once — charter §9's "a harness
check that cannot fire"); the MIQP and IQP guards both raise; the non-PSD status is recorded;
full `pytest --run-gpu` still green.

**WP-F — docs.** Every row of §9's table present; docs build in the **cuOpt-free**
`.venv-docs` with `PATH=…/.venv-docs/bin:$PATH` (D3) and **no new warning** vs
`logs/05-baseline-docs.txt`; the §6.6b mechanical gap check against #516's file list run and
every difference justified.

---

## 8. Test plan

### Where the tests live

| Test | File | Tier | Gating |
|---|---|---|---|
| existing suite, `("cuopt","direct",False)` row of `params` | `test/test_optimization.py` (unchanged bodies) | **V2** | conftest auto-skip on `GPU_ONLY` |
| feature-flag table rows | `test/test_solvers.py:448` | **V0** | none (ClassVars only) |
| capability-shim resolution | `test/test_solvers.py` (new) | **V0** | none |
| the six new cuOpt tests below | `test/test_optimization.py` | **V1** | `@pytest.mark.parametrize("solver", cuopt_solvers)` |

Everything cuOpt-specific is parametrized on `solver` so that conftest's
`pytest_collection_modifyitems` (`:53-79`) marks it `gpu` and skips it without `--run-gpu`.
`cuopt_solvers` is `[]` on a machine without cuOpt, so nothing is even collected there — the
same idiom `set_names_direct_solvers` (`:58-60`) already uses.

### The six V1 tests

**T1 — sign-convention matrix (the important one).** Parametrized over
`sense ∈ {min,max} × sign ∈ {<=,>=,==}` = the **six cells**. For each cell: build one
`linopy.Model`, solve it with `model.solve("cuopt")` **and** `model.solve("highs")` on an
identical freshly-built model **in the same process**, and compare objective, primal and
**dual** — never against a hard-coded number (§6.3a / risk register).

**T2 — presolve-solvable `max` model.** Exploration §"Surprises" 1 is explicit that the
six-cell matrix *alone would have missed* the negated-duals bug, and asks for "at least one
presolve-solvable `max` model (a square equality system…)" plus an assertion on
`get_solved_by()`. So T2 builds the reachable case exploration §2a demonstrated from real
linopy models — a **square equality system**, `max` sense, default settings, sizes `n ∈ {2, 4,
10}` — and asserts duals against live HiGHS. It additionally asserts that at least one size
reports `get_solved_by() == SolverMethod.Unset`, with a comment saying that a failure of *that*
assertion means the regression coverage has been lost and must be **reported, not deleted**.
`solved_by` is read off `model.solver.solver_model`… — note the `Solution` object is not
retained by `Result`; T2 therefore drives `Solver.from_name("cuopt", m, io_api="direct")`
directly and inspects the returned `Solution`, or asserts on `status.legacy_status`. **WP-D
decides which, and must not drop the `solved_by` assertion** — if `Solution` is unreachable
from the public path, the honest fix is a `logger.debug` plus a comment, reported as a partial
miss.
*Ranged rows are not testable through linopy:* `matrices._build_cons` produces only `<`, `>`,
`=` (`linopy/matrices.py:126-128`), so every row is one-sided or an equality. The ranged half
of exploration §2a is therefore unreachable from a linopy model and is not tested — recorded
here so its absence is deliberate, not an oversight.

**T3 — repeat solves above the segfault threshold (C-R20).** One process, a **2000-variable**
LP (above the measured ~1300 threshold), solved **three times**, each objective compared to
live HiGHS. Asserts the process survives and each solve is `optimal`.

**T4 — MILP differential + dual bound (C-R1, MIP_DUAL_BOUND_REPORT).** A small MILP solved by
cuOpt and HiGHS; objectives compared; `model.solver.report.dual_bound` is not `None` and, for a
`max` model, has the same sign as the objective (the M2 negation of `solution_bound`).

**T5 — semi-continuous differential (C-R24).** The discriminating model from exploration §10
(`min x`, `x <= 100`, `x` semi-continuous on `[5,10]` → optimum `0`, vs continuous → `5`),
solved by cuOpt and by HiGHS (which declares `SEMI_CONTINUOUS_VARIABLES`), objectives compared.

**T6 — QP convention (WP-E, C-R17).** The three-variable cross-term model from exploration §12
(`M.Q` dense `[[2,1,0],[1,4,0],[0,0,1]]`, `M.c = [-3,-1,2]`), solved by cuOpt and HiGHS. Must
agree. The WP demonstrates once that dropping the `0.5` makes it fail (it would return
`-2.14285714` against HiGHS's `-4.28571429`) — a check that cannot fail is not a check.
Plus: MIQP and IQP both raise `NotImplementedError`.

### Frozen numeric tolerances (proposed for `logs/15-verification.md`)

Every number below has a measured antecedent. The exploration's warning is taken seriously:
the default `1e-4` PDLP tolerances moved an objective by `5.52e-3` **relative** on a
2000×1000 LP, and the four `method` values disagree with each other by `~2.5e-5` relative on an
n=5000 model. Both of those are *large-model, loose-tolerance* numbers, so the tolerances are
**split by model size**, and the default method is the most accurate one (§5b).

| Check | Threshold | Rationale (measured antecedent) |
|---|---|---|
| T1/T2/T4/T5/T6 objective vs live HiGHS, **tiny models** (≤ 50 vars) | `rtol = 1e-6`, `atol = 1e-9` | On 2–3-variable models cuOpt agreed with HiGHS to 8+ significant figures (`obj=-2.80000000` vs `-2.7999999999999994`; QP `-4.28571429` vs `-4.285714285714258`), i.e. `≈1e-8` relative. `1e-6` is **100× looser than the worst observed**, which absorbs run-to-run jitter without being able to hide a sign error (whose signature is `O(1)` relative). |
| T1/T2 **duals** vs live HiGHS, tiny models | `rtol = 1e-6`, `atol = 1e-8` | Barrier (the default, §5b) agreed with HiGHS to `max|Δ| = 1.73e-09` (`<=`) and `2.47e-09` (`>=`). `atol = 1e-8` is 4× the worst observed and also covers the `~1e-10` non-binding QP duals that come back non-zero (exploration §12), so a genuinely-zero dual does not fail on noise. `rtol = 1e-6` gives ~400× headroom. **Deliberately tighter than the repo's `GPU_SOL_TOL = 2.5e-4`**: at `2.5e-4` a dual of `0.4` could be off by `1e-4` and still pass, which is not enough resolution to catch a partial sign or scaling error. |
| T1/T2 dual **sign** | exact: `np.sign(cuopt) == np.sign(highs)` wherever `abs(highs) > 1e-7` | The failure mode being guarded is a factor of `−1` (measured `max|Δdual| = 3.61`), so the sign check is a separate, tolerance-free assertion. The `1e-7` floor excludes duals HiGHS reports as structurally zero, where sign is meaningless. |
| T3 objective vs live HiGHS, **2000-var model** | `rtol = 2.5e-4` (= the repo's existing `GPU_SOL_TOL`) | The measured method-to-method spread at default tolerances on an n=5000 model was `2.5e-5` relative; `2.5e-4` is 10× that, and it reuses a number this repo already froze for GPU solvers (`test/test_optimization.py:80`, added by #516 commit 6 for exactly this reason) rather than inventing a new one. T3's purpose is *no crash*, not precision — the precision claim is T1's job on a model where it can be made tightly. |
| V2 (existing suite) | unchanged: `GPU_SOL_TOL = 2.5e-4` where the tests already use it | No change to a frozen repo number. |
| `pytest` / `pytest --run-gpu` | **zero** new failures vs `logs/05-baseline-pytest*.txt`, by name | charter §6.0 / Gate 4 |
| docs build | **zero** new warnings vs `logs/05-baseline-docs.txt` | charter §6.5 / Gate 6 |

**Named risk against these numbers, declared now rather than discovered later:** several
existing tests assert `.round(3)` equality (`test_quadratic_model` `:900-910`,
`test_quadratic_model_wo_constraint` `:929-944`) rather than using `GPU_SOL_TOL`. On the
tiny fixtures with Barrier I expect agreement at `~1e-8`, so `round(3)` should hold — but if it
does not, the WP **reports the actual numbers verbatim and stops**. The only legitimate
resolutions are (a) a `GPU_SOL_TOL`-based comparison for gpu solvers, following the precedent
#516 established in its own commit 6, proposed as a Gate-3 change; or (b) tightening cuOpt's
tolerance parameters for those models. Neither is a fix agent's unilateral call.

**Explicitly not proposed:** changing cuOpt's default `*_tolerance` parameters globally. The
`1e-4` defaults are cuOpt's own; tightening them to `1e-9` cost `140 → 34000` iterations and
`0.056 s → 1.941 s` in the exploration's own measurement. linopy does not second-guess any
other solver's tolerance defaults, and doing it here would trade the feature's headline benefit
for test convenience.

---

## 9. Docs plan

| Surface | What it will say |
|---|---|
| `doc/gpu-acceleration.rst` | New `cuOpt` section, placed **above** the existing `cuPDLPx` one (cuOpt is the more capable GPU solver): what cuOpt is (NVIDIA, concurrent PDLP + barrier + dual simplex on GPU) with a link to `docs.nvidia.com/cuopt`; **install**: `pip install 'linopy[cuopt]'` (and the `uv pip` equivalent) as the headline, with `uv pip install 'cuopt-cu12==26.8.*'` as a fallback below it; **requirements**: Linux only (no macOS/Windows wheels, no sdist), Python ≥ 3.11, NVIDIA driver ≥ 525.60.13, compute capability ≥ 7.0, CUDA 12 (`cu12`) — and a warning that `cu13` wheels need a newer driver and that **the bare `cuopt` package on PyPI is a different project**; **usage**: `m.solve("cuopt")` plus a `solver_options` example (`time_limit`, `method`, `log_to_console`, `log_file`); **supported**: LP, MILP, QP, semi-continuous variables, ranged-free row bounds, duals for LP/QP, MIP gap and dual bound; **limitations**, each one traceable to a finding: no warm start, no reduced costs, no MIQP, no SOS, no indicator constraints, no quadratic constraints, no LP/MPS file `io_api`, `method`/`crossover` silently ignored for QP, linopy defaults `method` to Barrier because cuOpt's default (Concurrent) crashes on repeat solves above ~1300 variables, Ctrl-C returns immediately but the GPU work continues until the process exits, and a time-limited LP can return a zero solution vector; **availability**: cuOpt is omitted from `linopy.available_solvers` on a machine with no visible GPU even when the package is installed (unlike cuPDLPx) — with the warning text quoted. |
| `doc/prerequisites.rst` | cuOpt added to the solver list (`:67` area) with the **one-command** extra install first, `platform`/driver/CC requirements stated next to it, and no `--index-strategy` and no `pypi.nvidia.com` as a required step (it may appear once as an optional fallback). |
| `doc/contributing.rst` | One short paragraph beside the existing `--run-gpu` text (`:63`): cuOpt tests are GPU-only and auto-skipped; `pytest test/test_optimization.py -k cuopt --run-gpu` runs them; **do not** run `uv sync` while cuOpt is installed from an unmerged tree; and the `benchmarks` extra pins `pandas==3.0.5` which cuOpt's `cudf` ceiling contradicts, so benchmarks need their own venv (§4.4). |
| `doc/index.rst` | `cuOpt <https://docs.nvidia.com/cuopt>`__ (GPU-accelerated) added to the solver bullet list (`:53` area). |
| `doc/api.rst` | `solvers.cuOpt` next to `solvers.cuPDLPx` (`:637`), and `to_cuopt` wherever `to_cupdlpx`/`to_highspy` appear. |
| `doc/release_notes.rst` | One bullet under the unreleased section: "Add support for the GPU-accelerated solver `cuOpt <…>`_, covering LP, MILP and QP via a new `linopy[cuopt]` extra." Commit 6. |
| `README.md` | `* [cuOpt](https://docs.nvidia.com/cuopt)` added to the solver list (`:153` area). |
| `pyproject.toml` | The extra carries the four-line comment in §5i explaining why it is separate from `solvers`; the docs surfaces above are where extras are explained for users. |
| `examples/using-solvers.ipynb` | **Decision: yes, minimally.** The notebook's opening cell already enumerates every solver and names "the GPU solver cuPDLPx" (`:11`); leaving cuOpt out of a list that claims to be complete is exactly the "name-drop vs explain" failure §6.6b warns about, in reverse. The change is **text-only in that one markdown cell** — "…or the GPU solvers cuPDLPx and cuOpt" — with **no new code cell**, because 17 of the 18 notebooks are executed during the docs build (`nbsphinx_execute="auto"`, `nbsphinx_allow_errors=False`) on a machine with **no GPU**, so any executable cuOpt cell would break the build. Logged as a decision so the "no new outputs" property is visible. |

Docs are built and diffed in the **cuOpt-free** `.venv-docs` (RTD has no GPU), with
`PATH=/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin:$PATH` (D3).

---

## 10. Explicit non-goals

Each is a thing a later reviewer might expect and should be able to see was *decided*, not
forgotten.

1. **File-based `io_api`s (`lp`, `lp-polars`, `mps`).** cuOpt can read them; we do not use it.
   §1 — 70× slower to build, 6× GPU test matrix, unverified `lp-polars` /
   `explicit_coordinate_names` / MPS-sense paths, `accepted_io_apis` inert on `master`.
   `READ_MODEL_FROM_FILE` not declared; `_run_file` raises.
2. **`problem.Problem` and `BatchSolve`.** ~70× slower, cannot express ranged rows, one
   already-deprecated accessor; `BatchSolve` has no linopy surface.
3. **The cuOpt server / gRPC client (`cuopt-server-cu12`, `cuopt-sh-client`) and remote
   solve.** `is_available()` is an in-process probe (§4.2); `CUOPT_REMOTE_HOST` is out of
   scope, and its error message is treated as a *symptom* in §5c, not a feature.
4. **Warm start / basis files.** §5h. `NotImplementedError` / warn-and-ignore.
5. **Reduced costs.** §5e. linopy has no surface, and cuOpt's are wrong for `max`/`<=`.
6. **MIQP / IQP.** Rejected up front (§5d.3).
7. **Quadratic *constraints*** (`add_quadratic_constraint`). linopy has no model surface for
   them, and cuOpt rejects equality quadratic constraints anyway.
8. **SOS, indicator constraints.** Verified absent upstream; the generic `_validate_model`
   guards (`:739-752`) already raise the right `ValueError`, so no cuOpt-specific code.
9. **Persistent / in-place updates** (`supports_persistent_update`, `supports_sign_update`).
   `DataModel` has no update API.
10. **MIP callbacks** (`set_mip_callback`), `internals` classes, solution/problem/presolve file
    outputs (`solution_file`, `user_problem_file`, `presolve_file`).
11. **`cu13` wheels.** Driver 535.274.02 is too old (§4.1). Documented as the user's choice,
    not offered by the extra.
12. **Changing cuOpt's default tolerances**, or imposing a default `time_limit`. §5f, §8.
13. **Fixing or working around cuOpt's upstream bugs beyond the two mitigations** (M2 and the
    `method` default). The negated-dual root cause, the Concurrent segfault, the
    `set_initial_primal_solution` CUDA corruption and the wrong `Fast1` docstring are recorded
    in `logs/45-known-issues.md` and reported, not patched.
14. **Any change to shared solver messages or to `TerminationCondition`.** §5c's residual and
    §2.5 — both would be I4 drive-bys.
15. **Benchmarks.** Blocked by the `pandas==3.0.5` pin vs cuOpt's `<3.0.4a0` ceiling (§4.4);
    a separate venv would be required and no benchmark claim is made in the docs.
16. **`CLAUDE.md`.** #516 edited it; that was its own harness note, not a cuOpt requirement.

---

## 11. Risks specific to *this* plan

Ordered by (probability × damage), with the trigger that would expose each and the fallback.

**R1 — The `is_available()` device probe is unverified and is the one novel mechanism in this
plan.** Everything else copies a measured pattern; `_has_cuda_device()` does not. Failure
modes: `cuda.bindings.driver.cuInit`'s Python signature differs from what I assumed (it returns
tuples, and the exact shape matters); `cuInit` in the parent process has a side effect I have
not measured; or it is slow enough to be felt in `available_solvers`. **Damage: high** — a
broken probe makes cuOpt invisible on a machine that *does* have a GPU, which reads exactly
like "the feature does not work". **Trigger:** WP-A's acceptance item 8 (both directions).
**Fallback if it cannot be made to work:** report verbatim, then fall back to
`_has_module("cuopt")` alone (the cuPDLPx/I5 behaviour) and record the §4.6-constraint-3 miss
as an escalation — *not* silently, and not by pretending the probe passed.

**R2 — Defaulting `method` to Barrier is a bet on one method across LP, MILP and QP, and the
exploration only stress-tested it on LPs.** If Barrier is slower than Concurrent on realistic
PyPSA-shaped models, we have traded the solver's headline performance for stability, and a
maintainer will (reasonably) ask for numbers we do not have — the exploration measured
*accuracy* per method and *stability* per method, but not comparative *speed* on a
representative model. **Trigger:** WP-A item 7 and `test_milp_model`. **Fallback:** the
`setdefault` shape means the decision is one constant, and the docs already tell users how to
override it; if MILP misbehaves, apply the default only for continuous models and report the
deviation. I would rather ship a stable default that can be overridden than the vendor default
that segfaults.

**R3 — The synthetic padding row is my invention, not a measured behaviour.** Exploration §5
verified that a row with a nonzero coefficient and `±inf` bounds solves (`row bounds both
+inf/-inf -> Optimal obj=0.0 dual=[0.]`), but it did **not** verify the specific shape I need:
a **single** row, one nonzero, both bounds infinite, on an otherwise box-only model. If that
returns `NoTermination`, `test_quadratic_model_wo_constraint` (`:929-944`) fails for cuopt.
**Trigger:** WP-A must test this shape *first*, before writing the class. **Fallback:** raise
`ValueError("Model has no constraints, cannot export to cuOpt. cuOpt requires at least one
constraint with a non-zero coefficient.")` — mirroring `cuPDLPx._build_solver_model`
(`:4256-4257`) — and add a cuopt branch to that one test, with the measured status pasted. That
is a worse user experience but an honest one, and it is a 3-line change either way.

**R4 — M2 concentrates two sign decisions on one branch.** `sense == "max"` drives the build-time
negation of `c` (and `Q`) and the read-time negation of objective, duals and `solution_bound`.
Five negations, one condition. A drift between them (e.g. someone adds a new `max` code path and
forgets `solution_bound`) is a silent wrong answer. **Mitigation:** the single `sign` variable at
build time; T1's six cells assert objective *and* duals; T4 asserts the dual bound's sign.
**Residual:** `solution_bound` for `max` MILP is covered only by a sign assertion, not a value
comparison against HiGHS's `mip_dual_bound` (whose definition may legitimately differ). I accept
that gap and name it.

**R5 — Declaring `SEMI_CONTINUOUS_VARIABLES` and `MIP_DUAL_BOUND_REPORT` widens the honest
claim surface beyond what #516 did for cuPDLPx.** Both are backed by measurements, but
semi-continuous carries cuOpt's `mip_semi_continuous_big_m = 1e10` default, which is a
modelling assumption linopy does not otherwise make; a semi-continuous variable with very
large bounds could in principle interact with it. **Trigger:** T5 (one discriminating model
only). **Fallback:** drop the flag and let `_validate_model` raise — a one-line change — but
that would be *less* honest than shipping it with one test, so I ship it.

**R6 — Two statuses in §4 are mapped but were never observed** (`LP 3 DualInfeasible`,
`MILP 3 Unbounded`; seven unbounded variants all returned `11`). If cuOpt's future versions
start returning `3`, the map is either right (my expectation) or produces `unbounded` where
`infeasible_or_unbounded` was meant. Damage is low (both are `SolverStatus.warning`) and the
alternative — omitting them so they fall through to `unknown` — is worse. Named so nobody
believes those two rows were tested.

**R7 — I declined the file `io_api`s, and that is the point in this plan most likely to draw a
"why not?"** If the Gate-3 reviewer or a maintainer wants them, the cost is real but bounded
(~+120 lines and a 6× GPU test matrix), and §1 states the price so the decision can be
reversed on evidence rather than re-litigated from scratch. The risk to *this* plan is not
technical: it is that "cuOpt can read LP/MPS and linopy doesn't use it" reads as an omission
unless the docs say plainly that it was measured and rejected. §9's limitations bullet does say
so.

**R8 — Existing `.round(3)` assertions in the QP tests.** Covered in §8's named risk. If they
fail, the resolution is a Gate-3 change, not a tolerance nudge (I12, §3.6).

**R9 — Six commits across six work packages means WP-A carries ~190 of the ~510 lines and every
hard decision.** If WP-A returns something structurally wrong (say, M2 implemented at read
time only), WP-C/D/E all build on it and the rework is expensive. **Mitigation:** WP-A's ten
acceptance items are deliberately front-loaded with the empirical checks (sign matrix, repeat
solves, device probe, padding row) *before* any test file is written, so a wrong foundation
surfaces inside WP-A rather than in WP-D.
