# plan-4 — cuOpt × linopy implementation plan

*Author: planner k=4 (Opus 5, `claude-opus-5[1m]`) · Phase 2 · 2026-08-22*
*Sources read: charter `PROJECT_LEAD.md` (full), `AGENTS.md`, `logs/30-decisions.md` rules block,
`logs/10-exploration.md` (full), current `master` source, PR #516 file list (shape only).*
*No sibling plan was read or listed.*

Throughout: "exploration §N" means the numbered item in `logs/10-exploration.md`.
Line numbers are current `master` (`09c34dd`).

---

## 0. Two facts I verified myself that change the design

Both are cheap reads of the installed package / current `master`, not re-runs of Phase 1
experiments. They are stated up front because two later sections depend on them.

**(0a) `LPTerminationStatus` and `MILPTerminationStatus` members collide as dict keys.**
Exploration §1 says a `CONDITION_MAP` may be keyed "on `(problem_category, status)` **or on the
enum member itself**". The second option is wrong. Both enums are `IntEnum`, whose MRO puts `int`
before `Enum`, so `__hash__`/`__eq__` come from `int`:

```
$ .venv/bin/python -c "..."          # two IntEnums each with Optimal = 1
True True                            # hash equal, == equal
{<A.Optimal: 1>: 'milp'} 1           # one key, silently overwritten
```

So a single dict keyed on enum members silently merges `LP.PrimalInfeasible` (2) with
`MILP.Infeasible` (2) and `LP.DualInfeasible` (3) with `MILP.Unbounded` (3). §4 below therefore
uses **two dicts keyed on the status *name* string** (`Solution.get_termination_reason()` returns
`termination_status.name`, `solution.py:285-289`), selected by problem category. Name keys also
avoid importing `cuopt.linear_programming.solver.solver_wrapper` (a `.pyx` module that
`solver/__init__.py` does not re-export) at all.

**(0b) `Solver.accepted_io_apis` is dead code on current `master`.**
`grep -rn accepted_io_apis linopy/ test/` → one declaration site (`solvers.py:466`) and one test
double (`test_persistent_solver_orchestrator.py:24`). **No production code reads it**, and no
shipped solver class declares it. What actually gates file IO is
`SolverFeature.READ_MODEL_FROM_FILE` (`test_optimization.py:45`, `Solver._build_file`) plus whether
`_run_file` is overridden. This matters for §1.

---

## 1. Entry-point decision

### Decision

| | Choice |
|---|---|
| **Primary (and only) entry point** | `cuopt.linear_programming.data_model.DataModel` + `cuopt.linear_programming.solver.Solve`, with `cuopt.linear_programming.solver_settings.SolverSettings` |
| **`io_api` surface** | **`direct` only.** `SolverFeature.READ_MODEL_FROM_FILE` is **not** declared; `SolverFeature.LP_FILE_NAMES` is **not** declared. |
| **`accepted_io_apis` ClassVar** | **Not declared** (see 0b and the justification below) |
| `cuopt.linear_programming.io` (file reader) | **Explicit non-goal for this PR** (§10, non-goal N1) |
| `problem.Problem` | Rejected |

`SolverName.cuOpt = "cuopt"`, `class cuOpt(Solver[None])`, `solver_model` = the `DataModel`
instance (the analogue of "the model"); the `SolverSettings` object is created per run inside
`_solve`, since it carries run-scoped state (`log_file` from `log_fn`).

### Agreement / divergence with the Phase 1 Q0 recommendation

**I agree with the primary half and diverge on the secondary half.**

*Agree — `DataModel` primary.* The measurements settle it and I do not re-litigate them:
0.012 s vs 0.86 s to build a 10⁵-variable model (~70×, exploration Q0 table), 13–16 lines of
translation against `model.matrices`, and it is the only candidate that can express ranged rows.
`problem.Problem`'s apparent advantages (`set_mip_callback`, warm-start, options, log control) all
live on `SolverSettings`, which every candidate shares, and its only genuine differentiators
(`relax`, `readMPS`, `writeMPS`) are things linopy does not need. It also funnels into the same
`DataModel` (`problem.py:1526-1630`), so it inherits the §2a dual defect without buying anything.

*Diverge — no `lp`/`mps` `io_api` in this PR.* Q0 calls the `io` path "a real, cheap capability
win … achievable in ~3 lines". The translation is 3 lines; the *feature* is not. Four reasons:

1. **It re-opens the highest-risk defect on a path where the mitigation does not reach.**
   Mitigation M2 (§5a) works because *we* build the `DataModel` and never call `set_maximize(True)`.
   `io.Read` sets the sense from the file (exploration Q0: `sense from file: maximize=False` for a
   min model), so a `max` model read from a `.lp`/`.mps` file arrives with `maximize=True` already
   set and lands on the buggy presolve postsolve. Recovering M2 there means reading the objective
   back out of the `DataModel`, negating it, and calling `set_maximize(False)` — a *second*
   implementation of the sign-critical transform, on a path with no build-time knowledge of the
   model. Charter §9 ranks the dual sign convention as the top risk; two code paths for it is the
   wrong trade for a capability nobody asked for.
2. **It multiplies the GPU test matrix by ~6.** Declaring `READ_MODEL_FROM_FILE` puts `cuopt` into
   `file_io_solvers` (`test_optimization.py:45-50`), i.e. `itertools.product(io_apis,
   explicit_coordinate_names)` = 4–6 extra parametrizations of ~40 tests, every one of them a real
   GPU solve, plus it opts cuOpt into `test_set_files`, `test_set_files_and_keep_files` and
   `test_solution_fn_parent_dir_doesnt_exist` (all guarded by `if solver in file_io_solvers`),
   which require honouring `solution_fn` — a surface exploration §7 flagged as unexercised
   (`solution_file` parameter, "UNKNOWN — whether `log_file` and `solution_file` interact").
3. **`mps` writing needs `highspy`** (`linopy/io.py`: "Package highspy not installed. This is
   required to exporting to MPS file."). A user who has highspy has HiGHS; the file path's marginal
   value is smallest exactly where it is available.
4. **It is a pure addition later.** Adding `READ_MODEL_FROM_FILE` + `_run_file` in a follow-up PR
   breaks nothing and needs no deprecation. Shipping LP/MILP/QP correctly through `direct` first is
   the ordering that keeps the diff proportionate (charter §9) and the sign story single-sourced.

*Why `accepted_io_apis` is not declared.* It is unread by production code (0b) and declared by no
shipped solver. Declaring it would be a stylistic outlier (Phase 6 §6.6a: "anything a reviewer
would flag as 'this file does not read like the rest of linopy'") **and** actively misleading — a
reader would reasonably assume it is enforced. The `direct`-only surface is enforced by the two
things that are actually read: the absence of `READ_MODEL_FROM_FILE`, and an overridden `_run_file`
that raises immediately (below). If Gate 3 prefers an explicit declaration, one line
(`accepted_io_apis: ClassVar[frozenset[str]] = frozenset({"direct"})`) satisfies it with no
behavioural change; I do not recommend it.

*What `model.solve("cuopt")` without `io_api="direct"` does.* Same as cuPDLPx today: `io_api` is
`None`, `_build` routes to `_build_file`, an `.lp` file is written, then `_run_file` raises. cuPDLPx
raises "linopy currently only supports reading models from netcdf files. Try using io_api='direct'
instead." after a warning and a netcdf detour. I **override `_run_file` with a direct
`NotImplementedError`** instead of copying the netcdf detour:

```python
def _run_file(self, **kw: Any) -> Result:
    raise NotImplementedError(
        "cuOpt does not support file-based IO. Use io_api='direct' instead."
    )
```

Rationale: the netcdf branch in `cuPDLPx._run_file` is reachable only for a `.netcdf` problem file,
which `Model.get_problem_file` never produces (`.lp`/`.mps` only), so copying it would add dead code
(I4) with a misleading warning. The base-class message ("File-based API not implemented for cuopt")
is nearly as good; overriding costs 4 lines and names the fix. That `io_api=None` still writes the
file before failing is a **pre-existing linopy rough edge shared with cuPDLPx** — it is
`Model.solve`'s io_api defaulting, outside the §5 map, and fixing it would change behaviour for
cuPDLPx too (I4/E4). Recorded, not fixed (see §11 R6); the docs always show `io_api="direct"`.

---

## 2. Full file-by-file change set

Estimated total: **+545 / −5 over 12 files** for LP+MILP+packaging+docs+tests, plus **+75** for QP
(§7 WP-F). Yardstick: #516 was +771/−62 over 15 files, so ~0.8× — comfortably inside charter §9.

### 2.1 `linopy/solvers.py` (~+230)

| Anchor | Change |
|---|---|
| `:148` `SolverFeature` | **No edit.** Every flag cuOpt needs already exists (§3). |
| `:175-181` `TYPE_CHECKING` | Add `import cuopt` — alphabetically first in the block, before `import cupdlpx`. |
| `:189-252` (after `_run_highs_with_keyboard_interrupt`) | New module-level helper `_run_cuopt_with_keyboard_interrupt(solve: Callable[[], Any]) -> Any` (~28 lines, §5g). |
| `:296-304` `_LazyModule` block | Add `cuopt = _LazyModule("cuopt")` **after** `cupdlpx` (keeps the existing block's "newest last" ordering), with a two-line comment: never import `cuopt.routing` — it installs a global `sys.excepthook` that writes `error_log.txt` into the cwd (`routing/vehicle_routing_wrapper.pyx:144-160`, exploration "Surprises" B). |
| `:307-314` (after `_has_module`) | New `_cuda_device_visible() -> bool` (~12 lines, §5c). |
| `:346-359` `SolverName` | Add `cuOpt = "cuopt"` after `cuPDLPx = "cupdlpx"`. Vendor spelling; enum member name must equal the class name for the `solver_capabilities` shim (`_solver_class`). |
| after `:4406` (`class cuPDLPx` ends at `:4416`) | New `class cuOpt(Solver[None])` placed **after** `cuPDLPx` and before `_solver_class_for` (~170 lines LP+MILP; +35 for QP in WP-F). Method order mirrors `cuPDLPx` exactly: docstring, `display_name`, `features`, `is_available`, `_run_file`, `_build_direct`, `_build_solver_model` (staticmethod), `_run_direct`, `_solve`, `_set_solver_params`. |
| `:4419` `_solver_class_for` | No edit. |
| `:4426` `QUADRATIC_SOLVERS` | **No edit** (derived from the feature; charter §5 last row). Verified by test, not by edit. |
| `:4443-4457` `_SOLVER_PROBE_ORDER` | Insert `"cuopt"` **immediately before** `"cupdlpx"`, i.e. `..., "copt", "cuopt", "cupdlpx", "pips"`. Two reasons: (i) it stays behind every CPU solver, so no existing user's default solver changes; (ii) ahead of cuPDLPx because cuOpt is a strict capability superset (LP+MILP+QP vs LP), so a user with only the two GPU solvers gets the more capable default. |
| `:4460-4470` `_AvailableSolvers` docstring | **No edit** (see §5c on why the docstring is not contradicted). |
| `_license_probe` | **Not overridden** (§5c). |

Class body, concretely (LP+MILP; QP deltas marked in §5d):

```python
class cuOpt(Solver[None]):
    """
    Solver subclass for the NVIDIA cuOpt solver. cuopt-cu12 must be installed
    with working GPU support for usage. Find the documentation at
    https://docs.nvidia.com/cuopt/.

    The full list of solver options is documented at
    https://docs.nvidia.com/cuopt/user-guide/latest/linear-programming/settings.html.
    Names are lower-case ``snake_case``; an unknown name raises.

    Some example options are:

    * time_limit : inf by default.
    * method : linopy defaults this to 3 (Barrier); 0 (Concurrent) is cuOpt's own
      default and crashes on repeated solves in one process (see below).
    * presolve : -1 (automatic) by default.
    * log_to_console : True by default.

    Notes
    -----
    Maximisation models are handed to cuOpt as the equivalent minimisation
    (negated objective) and the reported objective and duals are negated back.
    cuOpt 26.8 returns negated duals for maximised models that finish inside
    presolve; building in minimisation form avoids that code path entirely.
    Do **not** copy cuPDLPx's ``dual = -dual`` for ``max`` (:meth:`cuPDLPx._solve`) —
    cuOpt's dual convention already matches HiGHS, which is linopy's.

    ``method`` defaults to ``3`` (Barrier) rather than cuOpt's own ``0``
    (Concurrent), which segfaults the interpreter on the second or third
    sequential solve for models above roughly 1300 variables in cuOpt 26.8.
    Pass ``method=1`` (PDLP) for very large sparse LPs; ``method=0`` restores
    cuOpt's default and its crash.

    A ``KeyboardInterrupt`` returns control immediately, but the GPU solve
    continues in the background until the process exits — start a new process
    rather than re-solving.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "cuOpt"
    features: ClassVar[frozenset[SolverFeature]] = frozenset({...})   # §3

    @classmethod
    @functools.cache
    def is_available(cls) -> bool:
        if not _has_module("cuopt"):
            return False
        if not _cuda_device_visible():
            logger.warning(
                "cuopt-cu12 is installed but no CUDA device is visible; the 'cuopt' "
                "solver requires an NVIDIA GPU of compute capability >= 7.0 and "
                "driver >= 525.60.13, and is therefore not available."
            )
            return False
        return True

    def _run_file(...)          -> NotImplementedError            # §1
    def _build_direct(self, **kwargs) -> None                     # mirrors cuPDLPx:4227
    @staticmethod
    def _build_solver_model(model) -> cuopt.linear_programming.data_model.DataModel
    def _run_direct(...)        -> Result                          # mirrors cuPDLPx:4284
    def _solve(self, dm, ...)   -> Result
    def _set_solver_params(self, settings, log_fn=None) -> None
```

`_build_direct` (mirrors `cuPDLPx._build_direct`):

```python
def _build_direct(self, **kwargs: Any) -> None:
    model = self.model
    assert model is not None
    if kwargs.get("explicit_coordinate_names"):
        warnings.warn(
            "cuOpt does not support named variables/constraints. "
            "The explicit_coordinate_names parameter is ignored.",
            UserWarning, stacklevel=2,
        )
    dm = self._build_solver_model(model)
    self.solver_model = dm
    self.io_api = "direct"
    self.sense = model.sense
    self._cache_model_labels(model)
```

*(The MIQP guard lives in `_build_solver_model`, not here, so that `to_cuopt(m)` raises too — the
same placement `cuPDLPx._build_solver_model` uses for its semi-continuous guard.)*

`_build_solver_model` (the whole translation, ~35 lines):

```python
@staticmethod
def _build_solver_model(model: Model) -> Any:
    """Build a cuopt DataModel that mirrors the linopy `model`."""
    if model.type in ("MIQP", "IQP"):
        raise NotImplementedError(...)                     # as above
    M = model.matrices
    if M.A is None or M.A.nnz == 0:
        raise ValueError(
            "Model has no constraint coefficients, cannot export to cuOpt. "
            "cuOpt requires at least one non-zero in the constraint matrix."
        )
    A = M.A.tocsr()
    maximize = model.objective.sense == "max"
    sign = -1.0 if maximize else 1.0            # M2: always minimise

    dm = cuopt.linear_programming.data_model.DataModel()
    dm.set_csr_constraint_matrix(A.data, A.indices, A.indptr)
    dm.set_constraint_lower_bounds(np.where(M.sense != short_LESS_EQUAL, M.b, -np.inf))
    dm.set_constraint_upper_bounds(np.where(M.sense != short_GREATER_EQUAL, M.b, np.inf))
    dm.set_variable_lower_bounds(M.lb)
    dm.set_variable_upper_bounds(M.ub)
    dm.set_objective_coefficients(sign * M.c)
    # NOTE: set_maximize(True) is deliberately never called — see the class docstring.

    if len(model.binaries) + len(model.integers) + len(list(model.variables.semi_continuous)):
        # cuOpt has no binary code; linopy already clamps binaries to [0, 1].
        dm.set_variable_types(np.where(M.vtypes == "B", "I", M.vtypes))
    return dm
```

Notes on each line, all from the exploration (no new empiricism):
* row bounds use the `Highs._build_solver_model` formulation (`solvers.py:1702-1703`) which handles
  `<`, `>`, `=` in one expression; `np.inf` is accepted directly, no sentinel (exploration §5).
* `M.A` is already a `csr_array` with `float64`/`int32`/`int32` (exploration §6), so no dtype
  conversion and no `UserWarning`.
* `M.vtypes` is `<U1` and `np.where` preserves it; a `np.ndarray` is mandatory (a list/str/bytes
  raises, exploration §6). Only `C`/`I`/`S` are recognised and **anything else is silently
  continuous**, which is why the array is built here and user input never reaches it.
* the zero-nnz guard is the important one: exploration §5 shows a model with no constraint matrix,
  an explicit 0-row CSR, *or* one all-zero row returns `NoTermination` with `obj=nan` and an empty
  primal array **without raising**. `Model.solve` calls `constraints.sanitize_zeros()` by default,
  which makes `nnz == 0` reachable from an ordinary model whose only constraint is `0*x <= 1`.

`_solve` (~65 lines; status map + solution parse + report), and `_set_solver_params` (~18 lines):
see §4 for the maps, §5f for options, §5g for the interrupt wrapper.

### 2.2 `linopy/io.py` (~+8)

* `:32-37` `TYPE_CHECKING`: add
  `from cuopt.linear_programming.data_model import DataModel as cuoptDataModel`
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

The docstring paragraph is required, not decorative: `to_cuopt` is public API and returning a
minimising model for a `max` linopy model is surprising unless stated.

### 2.3 `linopy/model.py` (~+2)

* `:68-79` import block: add `to_cuopt,` (alphabetical: after `to_block_files`, before `to_cupdlpx`).
* `:2517` bindings: add `to_cuopt = to_cuopt` immediately before `to_cupdlpx = to_cupdlpx`.

### 2.4 `linopy/constants.py` (±0)

**No edit.** Every cuOpt status maps onto an existing `TerminationCondition` member (§4). Charter
§5: "add nothing new unless unavoidable" — it is avoidable.

### 2.5 `linopy/solver_capabilities.py` (±0)

**No edit** — it is a back-compat shim whose `_LazyRegistry` derives `SolverInfo` from
`SolverName` + the class. Satisfied by proof, not code: a test asserting
`SOLVER_REGISTRY["cuopt"].features == solvers.cuOpt.supported_features()` and
`.display_name == "cuOpt"` (§8, V0-4). PR #516's +68/−2 here is a stale pattern (I5).

### 2.6 `test/conftest.py` (±0)

**No edit.** `pytest_collection_modifyitems` (`:53-79`) is generic on
`solver_supports(solver, SolverFeature.GPU_ONLY)` and names no solver. Declaring `GPU_ONLY` gets
cuOpt auto-skipped without `--run-gpu` and auto-marked `gpu`. Satisfied by proof (§8, V0-5):
`pytest test/test_optimization.py -k cuopt` collects and skips with the reason string, and
`pytest --run-gpu -k cuopt` runs — both pasted.

### 2.7 `test/test_optimization.py` (~+4/−0)

Exactly three touches, matching the three cuPDLPx branches:

* `:562` `time_limit_option`: add `"cuopt": {"time_limit": 1},`.
* `:685` `test_infeasible_model`: **no change** — cuOpt is not skipped. It returns
  `PrimalInfeasible` on infeasible models (exploration §1) → `infeasible` → `status == "warning"`,
  and `compute_infeasibilities()` raises `NotImplementedError` (no `IIS_COMPUTATION`), so the
  existing `else` branch already fits. This is a deliberate capability win over cuPDLPx and an
  acceptance criterion of WP-D.
* `:1110` `test_basis_and_warmstart`: extend the skip to
  `if solver in ("cupdlpx", "cuopt")` with reason
  `"cuOpt does not support warmstart files."` (§5h).

No change to `CPU_SOL_TOL` / `GPU_SOL_TOL` — `GPU_SOL_TOL = 2.5e-4` already exists and cuOpt
inherits it automatically via `gpu_solvers` (derived from `GPU_ONLY`). **Introducing no new
tolerance constant in the shared suite is deliberate** (§8).

### 2.8 `test/test_solvers.py` (~+5)

Add to the `test_solver_class_supports_feature` parametrization (`:448-465`), mirroring the four
cuPDLPx rows:

```python
(solvers.cuOpt, SolverFeature.DIRECT_API, True),
(solvers.cuOpt, SolverFeature.GPU_ONLY, True),
(solvers.cuOpt, SolverFeature.INTEGER_VARIABLES, True),
(solvers.cuOpt, SolverFeature.READ_MODEL_FROM_FILE, False),
(solvers.cuOpt, SolverFeature.SOS_CONSTRAINTS, False),
```
plus, in WP-F, `(solvers.cuOpt, SolverFeature.QUADRATIC_OBJECTIVE, True)`.
These run on CPU (class-level, no GPU) and are the cheapest guard against a feature-flag typo.

### 2.9 `test/test_cuopt.py` (new, ~+190)

All cuOpt-specific correctness tests (§8). Precedent for a solver-specific module:
`test/test_highs_keyboard_interrupt.py`, `test/test_semi_continuous.py`. Kept out of
`test_optimization.py` because that file is a shared parametrized suite and ~190 lines of
cuOpt-specific differential tests there would be a merge magnet and hard to gate.

### 2.10 `pyproject.toml` (~+7)

Exploration §13's verified block, verbatim, inserted immediately after the `solvers = [...]` list
(`:113`) as a sibling extra:

```toml
# NVIDIA cuOpt (GPU). Deliberately NOT part of `solvers`: it pulls ~34 packages
# and several GB of CUDA wheels. Linux-only wheels, no sdist, so the marker keeps
# macOS/Windows resolution unaffected. Requires a CUDA 12 driver (>= 525.60.13)
# and a GPU of compute capability >= 7.0.
cuopt = [
    "cuopt-cu12>=26.8; platform_system == 'Linux'",
]
```

Do not "improve" the requirement string: this exact text was proven end-to-end from a clean venv
(exploration §13) and cross-compiled clean for Windows and macOS. Name `cuopt`, not `gpu` — §5i.

### 2.11 Docs (~+110 total)

`doc/gpu-acceleration.rst` (~+55), `doc/prerequisites.rst` (~+14), `doc/contributing.rst` (~+5),
`doc/index.rst` (+1), `doc/api.rst` (+1), `doc/release_notes.rst` (+1), `README.md` (+1),
`examples/using-solvers.ipynb` (+2 markdown words). Full content in §9.

### 2.12 Files PR #516 touched that this plan deliberately does not

Required by the §6.6b mechanical gap check; each is a justification, not a shrug.

| #516 file | Why not here |
|---|---|
| `CLAUDE.md` (+9/−1) | On current `master` it is a 5-line pointer to `AGENTS.md`. #516's addition was AI-contribution boilerplate that has since moved to `AGENTS.md`. Nothing cuOpt-specific. |
| `linopy/solver_capabilities.py` (+68/−2) | Now a derived shim (§2.5). Editing it is a drive-by change (I4) and charter §5 says to expect a zero-line diff. |
| `test/conftest.py` (+50) | The generic `GPU_ONLY` gating #516 added is already there (§2.6). |

---

## 3. Feature-flag set

```python
features: ClassVar[frozenset[SolverFeature]] = frozenset(
    {
        SolverFeature.INTEGER_VARIABLES,
        SolverFeature.QUADRATIC_OBJECTIVE,          # WP-F only
        SolverFeature.DIRECT_API,
        SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        SolverFeature.GPU_ACCELERATION,
        SolverFeature.GPU_ONLY,
        SolverFeature.SEMI_CONTINUOUS_VARIABLES,
        SolverFeature.MIP_DUAL_BOUND_REPORT,
    }
)
```

### Declared

| Feature | Justification |
|---|---|
| `DIRECT_API` | `_build_direct` / `_run_direct` implemented; the only supported `io_api` (§1). |
| `INTEGER_VARIABLES` | MILP verified (exploration §1: `MILP optimal status=1 Optimal cat=MIP obj=-2`, charter §4.3). Also gets cuOpt into `feasible_mip_solvers` so the existing MILP suite covers it. |
| `GPU_ACCELERATION` | It is a GPU solver. |
| `GPU_ONLY` | No in-process CPU fallback: without a visible device a solve raises `InputValidationError: A CPU-memory problem requires remote execution` (exploration §13). Declaring it is what drives conftest's auto-skip (C-R11) and `GPU_SOL_TOL` in the shared suite — both load-bearing, not cosmetic. |
| `SOLUTION_FILE_NOT_NEEDED` | The solution is returned in-process; `Model.solve` then passes `solution_fn=None`. Matches cuPDLPx. |
| `SEMI_CONTINUOUS_VARIABLES` | cuOpt **supports** `'S'`, proven on a discriminating model on both entry points (exploration §10: `vtype='S'` → `obj=0.0` vs `'C'`/`'I'` → `5.0`). Not declaring it makes `Solver._validate_model` (`solvers.py:731-737`) emit the **false** statement "Solver cuopt does not support semi-continuous variables", which exploration §10 explicitly warns against. Cost is one flag plus `M.vtypes` passed through, which we do anyway. **Contingency:** WP-D adds a V1 differential semi-continuous test (§8, V1-8); if it fails, report the numbers verbatim, drop this flag in the same WP, and log the decision — do not widen the test. |
| `MIP_DUAL_BOUND_REPORT` | `Solution.get_milp_stats()` returns `mip_gap` and `solution_bound` (`solution.py:367-410`), which populate `SolverReport(mip_gap=..., dual_bound=...)`. Declaring it activates the existing assertion at `test_optimization.py:826-834` (`report.dual_bound is not None`). **Contingency:** if `get_milp_stats()` does not populate `solution_bound` for linopy's tiny `milp_model`, report verbatim and drop this flag (one line) rather than weakening the assertion. |
| `QUADRATIC_OBJECTIVE` | WP-F only. QP verified against HiGHS (obj `-3.0`, and 6/6 dual cells, exploration §12/§3). Declaring it auto-registers cuOpt in `QUADRATIC_SOLVERS` (`solvers.py:4426`) — do not edit that list. |

### Deliberately NOT declared

| Feature | Why not |
|---|---|
| `READ_MODEL_FROM_FILE` | §1. Would opt cuOpt into `file_io_solvers` and the whole file-IO test matrix. |
| `LP_FILE_NAMES` | No file path, and `explicit_coordinate_names` is warn-and-ignored. `Solver._build_file` already downgrades it with a warning if someone asks. |
| `IIS_COMPUTATION` | No IIS/infeasibility-analysis surface anywhere in the package. Existing `test_infeasible_model` else-branch expects `NotImplementedError`. |
| `SOS_CONSTRAINTS` | Genuinely absent and explicitly out of scope upstream (exploration §10: the LP/MPS parser rejects an `SOS` section; upstream test comments "SOS is explicitly out of scope"). `_validate_model` then raises the actionable "reformulate first" message. |
| `INDICATOR_CONSTRAINTS` | Genuinely absent (exploration §10: no `indicator`/`genconstr` attribute anywhere). |
| `SOLVER_ATTRIBUTE_ACCESS` | Gurobi-only surface; `test_solver_attribute_getter` expects `NotImplementedError` for everyone else. Note cuOpt *does* expose `get_reduced_cost()`, but it is wrong for `max`/`<=` models (§5e) — declaring attribute access would be the one route by which that wrong number could reach a user. |
| *(no new `SolverFeature` member)* | Nothing cuOpt needs is unrepresentable, so `:148` gets a zero-line diff. |

`supports_persistent_update` / `supports_sign_update`: left at the base `False`. `DataModel` has no
incremental update API (exploration Q0 note 2), so a persistent path would rebuild anyway — which is
exactly what `_apply_locked` does for `supports_persistent_update = False`.

---

## 4. Status-mapping table

Keyed on `(problem category, status name)`. Two dicts, not one — see §0a. Module-level constants
inside `_solve` (HiGHS idiom, `solvers.py:1825-1845`).

```python
# cuOpt's LP and MILP status enums are distinct IntEnums whose members compare
# and hash equal by value (LP.PrimalInfeasible == MILP.Infeasible == 2), so a
# single dict keyed on the enum members silently merges them. Key on the status
# name and pick the map from the problem category instead.
```

**LP** (`problem_category == ProblemCategory.LP`, `LPTerminationStatus`):

| value | name | `TerminationCondition` | Justification (exploration §1 unless noted) |
|---|---|---|---|
| 0 | `NoTermination` | `internal_solver_error` | cuOpt's catch-all for malformed input, MIQP, and the GPU-memory ceiling. **Must not be `unknown`**: `unknown` sends `safe_get_solution` down its "try to parse anyway" branch (`solvers.py:1192-1200`), which would flip the status to `ok` on an empty/nan solution. `internal_solver_error` → `SolverStatus.error` → empty `Solution()`. |
| 1 | `Optimal` | `optimal` | |
| 2 | `PrimalInfeasible` | `infeasible` | |
| 3 | `DualInfeasible` | `unbounded` | **Never observed** (7 unbounded variants all returned 11). Mapped from the enum's meaning; flagged as untested in a code comment and in `logs/45-known-issues.md`. |
| 4 | `IterationLimit` | `iteration_limit` | |
| 5 | `TimeLimit` | `time_limit` | Note: returns `obj=+0` with a zero primal vector, not a partial solution. `time_limit` is `SolverStatus.ok` by linopy's own map — same semantics as HiGHS's `kTimeLimit`; the user must read `termination_condition`. Documented, not special-cased. |
| 6 | `NumericalError` | `internal_solver_error` | Non-PSD `Q`; `obj=nan`. |
| 7 | `PrimalFeasible` | `suboptimal` | Only reachable via `first_primal_feasible=True`. |
| 11 | `UnboundedOrInfeasible` | `infeasible_or_unbounded` | The status real unbounded LPs actually return. |

**MILP** (`problem_category in (MIP, IP)`, `MILPTerminationStatus`):

| value | name | `TerminationCondition` | Justification |
|---|---|---|---|
| 0 | `NoTermination` | `internal_solver_error` | MIQP (guarded before the solve anyway), malformed arrays. |
| 1 | `Optimal` | `optimal` | |
| 2 | `Infeasible` | `infeasible` | |
| 3 | `Unbounded` | `unbounded` | **Never observed** (11 every time). Same comment as LP 3. |
| 5 | `TimeLimit` | `time_limit` | |
| 8 | `FeasibleFound` | `suboptimal` | Feasible incumbent, limit hit. `suboptimal` is `SolverStatus.ok` and the solution *is* usable — correct. |
| 11 | `UnboundedOrInfeasible` | `infeasible_or_unbounded` | |

Fallback: `.get(name, TerminationCondition.unknown)` (HiGHS idiom, `solvers.py:1855-1857`), so a
status added by a future cuOpt release degrades to `unknown` rather than `KeyError`.

Alongside the mapping, in `_solve`:

```python
sol = _run_cuopt_with_keyboard_interrupt(
    lambda: cuopt.linear_programming.solver.Solve(dm, settings)
)
is_lp = sol.get_problem_category().name == "LP"
name = sol.get_termination_reason()
condition_map = _LP_CONDITION_MAP if is_lp else _MILP_CONDITION_MAP
termination_condition = condition_map.get(name, TerminationCondition.unknown)

if sol.get_error_status():                      # ErrorStatus.Success == 0
    logger.error(f"cuOpt reported an error: {sol.get_error_message()}")

n_active = 0 if self._vlabels is None else self._vlabels.size
if sol.get_primal_solution().size != n_active:
    # cuOpt returns an empty primal array for malformed input without raising.
    termination_condition = TerminationCondition.internal_solver_error

status = Status.from_termination_condition(termination_condition)
status.legacy_status = name
```

`ErrorStatus` / `ProblemCategory` are read by `.name` / truthiness rather than imported, so no
`.pyx`-module import path is hard-coded (see §0a).

Solution parse:

```python
def get_solver_solution() -> Solution:
    objective = sol.get_primal_objective()
    primal = np.asarray(sol.get_primal_solution(), dtype=float)
    if is_lp:
        dual = np.asarray(sol.get_dual_solution(), dtype=float)
    else:
        # get_dual_solution() raises AttributeError for MIP/IP solutions.
        dual = np.array([], dtype=float)
    if sense == "max":
        # The model was handed to cuOpt in minimisation form; see the class docstring.
        objective = -objective
        dual = -dual
    primal = _solution_from_labels(primal, self._vlabels, self._n_vars)
    dual = _solution_from_labels(dual, self._clabels, self._n_cons)
    return Solution(primal, dual, objective)
```

`_solution_from_labels` with an empty `dual` and `size == 0` short-circuits; for a MILP with
constraints, `self._n_cons > 0` and an empty values array — this is the same shape Gurobi's MILP
path produces (`solvers.py:4128-4131`), so it must be verified in WP-A that
`values_to_lookup_array` accepts it; if it does not, follow Gurobi and return
`np.array([], dtype=float)` unscattered. **Acceptance criterion, not a guess** (§7 WP-A AC-6).

Report:

```python
runtime = mip_gap = dual_bound = None
with contextlib.suppress(Exception):
    runtime = float(sol.get_solve_time())
if not is_lp:
    with contextlib.suppress(Exception):
        stats = sol.get_milp_stats()
        mip_gap = float(stats["mip_gap"])
        bound = float(stats["solution_bound"])
        dual_bound = -bound if sense == "max" else bound
```

The `solution_bound` negation under `max` is easy to miss and would report an upper bound as a
lower bound with the wrong sign.

---

## 5. Design decisions on the hard findings

### 5a. Presolve + `maximize` negated duals → adopt **M2 (always minimise)**

**Decision: adopt M2 exactly as validated (21/21 cases against live HiGHS).** For a `max` model,
build with `-M.c` (and `-0.5·M.Q` for QP), never call `set_maximize(True)`, then negate
`get_primal_objective()` and `get_dual_solution()` on the linopy side. No objective offset is
involved: linopy rejects objective constants outright
(`objective.py:192-193`, `ValueError: Constant values in objective function not supported`), so
`M.c` is the entire objective and `set_objective_offset` is never called.

Why M2 and not the alternatives:

* **M1 (`presolve` forced)** is rejected on evidence: it fixes the equality cells and **fails on
  ranged rows** (exploration §2b: `presolve=1` still lands on `Unset` and still negates). A fix that
  works on the shapes you tested and not on the ones you did not is the worst kind.
* **`method=DualSimplex`** would fix it by routing off the GPU entirely — self-defeating for a GPU
  solver, and (importantly) **my chosen default `method=3` (Barrier) does *not* avoid the bug**:
  exploration §2a's correlation table lists `Barrier / max / Unset → NEGATED`. M2 is load-bearing,
  not belt-and-braces.
* **`dual_postsolve`** is irrelevant (verified `NO` both ways).
* **Negating duals for `max` like cuPDLPx does** (`solvers.py:4385-4386`) is the trap: cuOpt's
  convention already matches HiGHS, which is linopy's convention (linopy passes HiGHS's `row_dual`
  through unchanged, `solvers.py:1871-1879`). An implementer "matching repo precedent" would inject
  a systematic sign error into every maximised model. This gets an explicit code comment and a
  paragraph in the class docstring, because it is the single most likely way this integration ships
  a silent wrong answer.

Reachability note that justifies the whole exercise: exploration §2a reached the defect from an
ordinary `linopy.Model()` through the natural translation, on square equality systems at n = 2, 4,
10, 40, 200, with `max|Δdual| = 3.61` and the objective always right. Ranged rows are *not*
reachable through `model.matrices` (linopy's `M.sense` only ever holds `<`, `>`, `=`), but the
equality case alone is enough.

Consequences carried forward: `to_cuopt` returns a minimising model (documented, §2.2); the MILP
`solution_bound` must be negated too (§4); the V1 sign matrix must include a presolve-solvable
`max` model and assert it still lands on `solved_by == Unset` (§8, V1-7).

### 5b. Concurrent repeat-solve SIGSEGV → **default `method` to 3 (Barrier)**

**Decision: when the user has not passed `method`, linopy sets `method = 3` (Barrier).** A
user-supplied `method` always wins.

```python
_CUOPT_DEFAULT_METHOD = 3   # SolverMethod.Barrier; see class docstring
...
if "method" not in self.solver_options:
    settings.set_parameter("method", _CUOPT_DEFAULT_METHOD)
```

*(Written as the literal 3 with the enum named in the comment, or as
`int(cuopt.linear_programming.solver_settings.SolverMethod.Barrier)` — the attribute chain resolves
through the lazy module. WP-A picks one and stays consistent; the literal keeps the module-level
constant free of a lazy-import side effect at definition time, which is why I prefer it.)*

Doing nothing is indefensible: cuOpt's own default (`0`, Concurrent) **hard-crashes the interpreter
with SIGSEGV on the 2nd–3rd sequential solve for LPs above ~1300 variables**, deterministically,
reproduced in ≥12 fresh processes (exploration §9). linopy's own test models are far below the
threshold, so the suite would stay green while a real user's second `model.solve("cuopt")` in one
script dies with no traceback — a false-green of exactly the kind charter §9 calls out.

Why Barrier over PDLP (the other stable GPU-native single method):

| | Barrier (3) | PDLP (1) | Concurrent (0) |
|---|---|---|---|
| Repeat solves, 25 varied shapes | 25/25, memory flat at 117 MiB | 25/25, flat 111 MiB | **crash** |
| Dual agreement vs HiGHS | **1.73e-09 – 2.47e-09** | 2.60e-06 – 4.71e-05 | 8.18e-08 |
| Objective drift at default tol | not observed | 2.80013079 vs 2.8; and 5.52e-3 relative on a 2000×1000 LP | — |
| Other pathologies | none observed | non-terminating on a badly scaled LP (13 min at 98% GPU, killed by `timeout 900`); segfault with `save_best_primal_so_far` + tiny `time_limit` | — |
| QP | always used anyway (`method` silently overridden) | overridden to Barrier | overridden to Barrier |

Barrier wins on accuracy by three to four orders of magnitude, showed no pathologies, and gives
**one uniform story across LP, MILP and QP** (QP is solved by Barrier whatever you ask for). It is
GPU-based (`nvidia-cudss-cu12` is in the dependency tree). It also lets §8's dual tolerance be
`rtol=1e-6` honestly instead of forcing tolerance overrides — which matters, because the
alternative route to a tight `rtol` is linopy silently tightening six tolerance parameters, i.e.
*more* hidden default-changing, not less.

The cost is honest and documented: Barrier is not the right method for the marquee
"millions of variables" sparse LP. The class docstring and `doc/gpu-acceleration.rst` both say so
and point at `method=1` (PDLP), which is itself repeat-solve-safe, so the advice is safe.

Also required (WP-A):
* **A repeat-solve regression test**, run in a **subprocess** so a regression cannot take down the
  pytest session: solve an n≈1600 LP three times in one interpreter with default options, assert
  `returncode == 0` and three `optimal` lines. Repo precedent for the pattern:
  `test_available_solvers.py:32-36`. Marked `gpu`. This test **fails loudly if anyone ever restores
  Concurrent as the default** — i.e. it is a check that can fire (charter §9).
* `save_best_primal_so_far` is never set by linopy, and is not documented as an option to use.

### 5c. `is_available()` on a GPU-less machine → **device probe in `is_available`, failing open**

**Decision: honour charter §4.6 constraint 3.** `cuOpt.is_available()` returns `True` only if
`cuopt` is importable **and** at least one CUDA device is visible.

```python
def _cuda_device_visible() -> bool:
    """
    True if at least one CUDA device is visible.

    ``import cuopt`` deliberately succeeds on CPU-only hosts (its ``__init__``
    lazy-imports every submodule so a remote solve can be configured), so an
    import probe alone would advertise a GPU-only solver on a machine that
    cannot run it. Falls back to True if the probe itself is unavailable, so a
    missing probe degrades to the import-only behaviour rather than hiding an
    installed solver.
    """
    try:
        from cuda.bindings import runtime          # ships with cuopt-cu12

        err, count = runtime.cudaGetDeviceCount()
    except Exception:
        return True
    return not err and count > 0
```

**Exact behaviour on a GPU-less machine with `cuopt-cu12` installed:**

* `linopy.available_solvers` → **does not contain `"cuopt"`** (§4.6 c3, first half).
* One `logger.warning` at the moment of the probe: *"cuopt-cu12 is installed but no CUDA device is
  visible; the 'cuopt' solver requires an NVIDIA GPU of compute capability >= 7.0 and driver >=
  525.60.13, and is therefore not available."*
* `model.solve("cuopt")` → `AssertionError("Solver cuopt not installed")` from
  `model.py:2077`, **immediately preceded by that warning line**, because `available_solvers` is
  evaluated on the line above. The pair is actionable; the bare CUDA traceback / "Set
  CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT" that §4.6 c3 forbids never appears.
* `linopy.solvers.cuOpt()` (direct construction) → `ImportError("Solver package for 'cuopt' is not
  installed. Please install first to initialize solver instance.")` from `Solver.__post_init__`.

**Two residual warts I am accepting rather than hiding:**

1. `model.solve`'s assertion text says "not installed" when the package *is* installed. Fixing it
   means editing `Model.solve` (`model.py:2077`), which is outside the §5 map and changes behaviour
   for every solver → I4/E4. Mitigated by the warning line. Recorded for
   `logs/45-known-issues.md`.
2. `check_solver_licenses("cuopt")` returns `message="package not installed"` (base
   `Solver.license_status`, `solvers.py:613-614`) for the same reason. Same mitigation, same record.

**Why this and not the I5 alternative.** The alternative (`is_available = _has_module("cuopt")`,
device check in `_license_probe`) matches `cuPDLPx.is_available` and the `_AvailableSolvers`
docstring literally, and I considered it seriously. It loses on three points: (i) §4.6 is a binding
human-lead requirement that names `available_solvers` explicitly, and it outranks a stylistic
precedent; (ii) `available_solvers[0]` is the **default solver** in `Model.solve` — on a GPU-less
box where cuOpt is the only installed solver, the import-probe version makes every default solve
fail with cuOpt's remote-host message; (iii) the `_AvailableSolvers` docstring's disclaimer is about
*licenses*, and a GPU-only solver with no GPU is not "installed but unlicensed", it is unusable. The
probe is a superset of the import probe, so nothing it reports is untrue.

`_license_probe` is therefore **not overridden** (base no-op): cuOpt needs no license, and the
runtime check already lives in `is_available`. Consequence: `licensed_solvers == available_solvers`
for cuOpt, so the whole parametrized suite in `test_optimization.py` (which keys off
`licensed_solvers`) picks cuOpt up. Deviating from cuPDLPx's override set here is deliberate and
gets a `logs/30-decisions.md` entry.

**Failing open is the point.** If `cuda.bindings.runtime.cudaGetDeviceCount()` is not the right API
in some future wheel, the `except` returns `True` and we degrade to today's cuPDLPx behaviour. The
opposite default would silently hide a working solver on a working T4 — a far worse failure. WP-A's
acceptance criteria require **both** branches to be demonstrated on this machine and pasted:
`available_solvers` contains `cuopt` normally, and does not under `CUDA_VISIBLE_DEVICES=''`. That is
the "make the check fail on purpose once" discipline from charter §9.

### 5d. QP: `0.5·M.Q`, MIQP guard, non-PSD

**Decision (WP-F):**

```python
Q = M.Q
if Q is not None:
    # linopy's M.Q is the Hessian H (QuadraticExpression.to_matrix: "the
    # quadratic term is 0.5 x^T Q x ... diagonal terms are doubled"), while
    # cuOpt minimises c'x + x'Qx with Q internally symmetrised as Q + Q'.
    # Passing M.Q directly halves the solution and returns a wrong objective
    # with status Optimal.
    Qc = scipy.sparse.csr_matrix(sign * 0.5 * Q)
    dm.set_quadratic_objective_matrix(Qc.data, Qc.indices, Qc.indptr)
```

* **Convention:** `Q_cuopt = 0.5 · M.Q`, kept **full symmetric**, converted to CSR. Verified three
  independent ways in exploration §12, including a live HiGHS cross-check through
  `model.matrices.Q` (`0.5*M.Q` → `-4.28571429` = HiGHS; `M.Q` → `-2.14285714`, `Optimal`,
  silently wrong). Triangular encodings that satisfy `Q + Qᵀ = H` also work, but full-symmetric is
  what the empirical check used, needs no `triu` import, and is the least error-prone.
* **`max` sense:** `sign = -1` applies to `Q` as well as `c` (M2; validated 4/4 QP cells).
* **MIQP:** rejected up front in `_build_solver_model`, since cuOpt returns `NoTermination`,
  `obj=0.0`, an **empty** solution and `error_status=4` without raising. Guard on
  `model.type in ("MIQP", "IQP")` — note `Model.type` (`model.py:1666-1679`) yields `"IQP"` for an
  all-integer quadratic model and `"MIQP"` only when continuous variables are also present, so
  checking `"MIQP"` alone would miss the pure-integer case. Message: *"cuOpt does not support
  quadratic objectives with integer variables (MIQP). Use a solver that supports them (gurobi,
  xpress)."* Declaring `QUADRATIC_OBJECTIVE` therefore does **not** imply MIQP support, which is
  why the guard is `NotImplementedError` and not a feature flag.
* **Non-PSD `Q`:** no pre-check. cuOpt detects it and returns `NumericalError` with `obj=nan`
  (exploration §12) → `internal_solver_error` → `SolverStatus.error` → empty `Solution()`. Writing
  our own PSD test would mean an eigenvalue computation on every quadratic solve (expensive,
  and wrong at the margins) to produce a slightly nicer message. Honest degradation for free is the
  better trade; the docs state the PSD requirement.
* **Quadratic *constraints*** (`add_quadratic_constraint`) are out of scope: linopy has no model
  surface for them (non-goal N4).
* **`method`/`crossover` are silently overridden for QP** by cuOpt (always Barrier). Since §5b
  already defaults to Barrier, a user passing `method=...` on a QP gets what they would have got
  anyway — no new surprise, and one sentence in the docs.
* **`lp_stats['dual_residual']` is meaningless under QP** (51.0 on a solved model). Never surfaced
  or asserted on.

### 5e. Reduced costs → **do nothing, deliberately**

**Decision: `get_reduced_cost()` is never called.** linopy has no reduced-cost surface —
`grep -rn "reduced_cost\|ReducedCost" linopy/` returns nothing and `constants.Solution`
(`constants.py:273-286`) carries only `primal`, `dual`, `objective` — so there is nowhere to put
them. And cuOpt's are wrong where it matters: for a **maximised** model with `<=` rows the default
path returns `−c − Aᵀy` instead of `c − Aᵀy`, and forced `PDLP` returns a flat `0` for `max`/`>=`
(exploration §2d); exploration could not reduce it to one transformation and explicitly declined to
propose one.

Two positive actions instead of zero:
1. A one-line comment next to the solution parse: *"cuOpt's `get_reduced_cost()` is not surfaced:
   linopy has no reduced-cost surface, and the values are wrong for maximised models with `<=`
   rows in cuOpt 26.8."* — so nobody "helpfully" adds it later.
2. `SolverFeature.SOLVER_ATTRIBUTE_ACCESS` is not declared (§3), which closes the other route
   (`variables.get_solver_attribute("RC")`) by which the bad numbers could surface.

The upstream defect goes into `logs/45-known-issues.md` (record, don't fix — I13).

### 5f. Solver options

**Decision: pass through unchanged; no bool→int coercion; wrap the unknown-*name* error only.**

```python
def _set_solver_params(self, settings: Any, log_fn: Path | None = None) -> None:
    if log_fn is not None:
        self.solver_options["log_file"] = path_to_string(log_fn)
        logger.info(f"Log file at {self.solver_options['log_file']}")
    if "method" not in self.solver_options:
        settings.set_parameter("method", _CUOPT_DEFAULT_METHOD)   # §5b
    for k, v in self.solver_options.items():
        try:
            settings.set_parameter(k, v)
        except ValueError as e:
            raise ValueError(
                f"Invalid solver option '{k}' for cuOpt: {e}. Parameter names are "
                "lower-case snake_case, e.g. 'time_limit'."
            ) from e
```

* **`log_file` / `log_to_console`:** this is the `Highs._set_solver_params` idiom verbatim
  (`solvers.py:1774-1784`) — cuOpt's parameter is spelled *identically* to HiGHS's. `log_fn` wins
  over a user-supplied `solver_options["log_file"]`, matching HiGHS's precedence exactly, which
  settles exploration's open "UNKNOWN — precedence" question by repo precedent (§3.2: follow HiGHS
  and move on). **No OS-level `dup2` is needed** — the charter's prediction that
  `redirect_stdout` cannot catch cuOpt's C++ output is confirmed (0 chars), but the native
  `log_file` parameter writes the full banner and iteration log (1437 chars). cuOpt is strictly
  better off than cuPDLPx here, which warn-and-ignores `log_fn`. `log_to_console` is left alone
  (default `True`), matching HiGHS, so a `log_fn` user still sees the console log.
* **No bool→int coercion.** `presolve=False` raises `ValueError: Parameter presolve value False is
  not an integer` while `log_to_console=False` is accepted — the type expectation is per-parameter.
  A blanket `bool → int` cast is unverified for the genuinely boolean parameters (would
  `log_to_console=1` be accepted? exploration did not test it) and a per-parameter allow-list is a
  109-name table that will rot. cuOpt's own value-error messages **name the parameter and the
  value** and are actionable as-is; the docs say "pass `0`/`1` for integer-valued parameters such as
  `presolve`".
* **Unknown *name*** is the one message that fails the actionable bar: `set_parameter('TimeLimit',
  1)` → `ValueError: Invalid parameter. Please check documentation`, which names neither the
  parameter nor the convention. It is raised eagerly at `set_parameter`, so wrapping it is exact and
  cheap. The `TimeLimit` spelling is precisely what a cuPDLPx or Gurobi user will type
  (`test_optimization.py:562` uses `"cupdlpx": {"TimeLimit": 1}`), so this wrapper will earn its
  keep.
* **Bad *values*** raise late, from `Solve()`, and are left unwrapped: those messages already name
  the parameter and value.
* **No default `time_limit`.** Exploration raised the question ("consider whether a default
  `time_limit` is warranted") after a non-terminating PDLP run. I decline: no other linopy solver
  imposes one, a silent cap would turn a slow solve into a wrong-looking `time_limit` status, and
  the pathology was specific to `method=PDLP`, which is not the default (§5b). Documented instead.

### 5g. Interruptibility → **adopt the worker-thread pattern**

**Decision: yes, with a `Callable`-shaped helper and no cancel step.**

```python
def _run_cuopt_with_keyboard_interrupt(solve: Callable[[], Any]) -> Any:
    """
    Run a cuOpt solve while keeping Ctrl-C responsive.

    cuOpt's ``Solve()`` blocks in C++ for the whole solve and exposes no cancel
    hook, so a SIGINT is deferred until it returns. Running it in a daemon
    thread lets the main thread raise KeyboardInterrupt immediately. The GPU
    work cannot be stopped: it continues in the background until the process
    exits, so ``time_limit`` is the only real bound on a running solve.
    """
```

Measured justification (exploration §8): SIGINT at t=5.0 s raised `KeyboardInterrupt` at t=57.9 s —
**52.9 s of unresponsiveness** — versus **0.00 s** with the worker thread; and the interpreter exits
cleanly (`exit=0`, no CUDA teardown error) with the daemon thread still inside cuOpt's C++ code.
cuOpt installs no SIGINT handler of its own and there is no `cancelSolve()` analogue (the only
`cancel` in the package is the gRPC client, which §4.2 rules out).

Differences from `_run_highs_with_keyboard_interrupt` (`solvers.py:189-252`), each deliberate:
* **No `HandleKeyboardInterrupt` / `HandleUserInterrupt` fiddling** — no such attributes exist.
* **No cancel step** — no API to call.
* **Re-raise immediately without joining.** HiGHS joins *after* cancelling, which is bounded;
  joining cuOpt would mean waiting out the entire solve, defeating the purpose. The orphaned daemon
  thread is documented in the class docstring, including the advice not to re-solve in the same
  process (a second concurrent cuOpt solve is unsafe and `Solver._lock` is released by the
  unwinding exception).
* **Signature is a `Callable`**, not `(dm, settings)`. This is what makes it testable **on CPU with
  a dummy callable**, exactly as `test/test_highs_keyboard_interrupt.py` tests the HiGHS helper with
  a `DummyHighs` — a V1-tier test that needs no GPU and no cuOpt (§8, V1-9).

### 5h. Warm start → **defer**

**Decision: `warmstart_fn` raises `NotImplementedError`; `basis_fn` warns and is ignored;
`DataModel.set_initial_primal_solution` is never called, on any path.**

```python
if warmstart_fn is not None:
    raise NotImplementedError("Warmstarting is not yet implemented for cuOpt.")
if basis_fn is not None:
    logger.warning("Basis files are not supported by cuOpt. Ignoring.")
if solution_fn is not None:
    logger.warning("Solution files are not supported by cuOpt. Ignoring.")
```

(cuPDLPx raises for `solution_fn`; a warning is friendlier and is what it does for `basis_fn`. Since
`SOLUTION_FILE_NOT_NEEDED` means `Model.solve` passes `None`, this is only reachable when a user
passes one explicitly, and failing a solve over an unwanted output file is disproportionate. Either
is defensible; I pick warn-and-ignore and log the choice.)

Three independent reasons to defer, all measured:
1. **Usable warm start needs three simultaneous non-default settings** — `method=1` (PDLP),
   `pdlp_solver_mode=1` (Stable2), `presolve=0`. `Stable3` (the default mode) and even `Fast1`
   (which the upstream docstring claims is supported) return `NoTermination` with `obj=nan`. Wiring
   it up means silently overriding three defaults, including one that contradicts §5b's method
   choice.
2. **The payload is not a file.** `warmstart_fn: Path` is linopy's only warm-start surface, and
   `PDLPWarmStartData` is a 17-field Python object in *presolved* coordinates that only round-trips
   with `presolve=0`. There is no honest way to spell it as a path in this PR.
3. **`set_initial_primal_solution` on an LP poisons the CUDA context** —
   `cudaErrorIllegalAddress`, and *every later solve in the process* raises. Reproduced with
   all-default settings, correctly sized C-contiguous `float64` input. One warm-start attempt takes
   down a whole pytest session or user script. This is not a "careful, use it right" hazard; the
   only safe rule is never call it. A code comment says so at the point where a future contributor
   would be tempted.

Test consequence: `test_basis_and_warmstart` skips cuOpt (§2.7). Docs list warm start under
limitations with the reason.

### 5i. The pyproject extra

**Decision: `linopy[cuopt]`, contents `"cuopt-cu12>=26.8; platform_system == 'Linux'"`, sibling of
`solvers`.** Exploration §13's verified candidate, adopted unchanged.

* **Name `cuopt`, not `gpu`.** It matches the solver name, the `SolverName` value and the file's
  existing lower-case single-word extras (`oetc`, `remote`, `docs`, `dev`, `solvers`,
  `benchmarks`), and — the decisive argument — `gpu` is a promise the extra cannot keep: it installs
  cuOpt specifically, on Linux only, for CUDA 12 only. A future second GPU solver gets its own
  extra, which is *more* honest than silently changing what `linopy[gpu]` installs. Charter §5's
  `<gpu-extra>` is a placeholder, not a name.
* **Marker `platform_system == 'Linux'`** — a positive equality, not `!= 'Darwin'` (the file's
  `cplex` style), because cuOpt has no Windows wheels either. No `python_version` marker: cuOpt
  ships cp311–cp314 and linopy already requires `>=3.11`.
* **Sibling of `solvers`, reachable from nothing else.** There is no `all` extra in this project, so
  a sibling is sufficient to keep 6.7 GB out of `pip install linopy[solvers]` — verified: that
  resolution pulls 34 packages with "GPU/CUDA-related packages present: NONE".
* **No `[tool.uv]` index entry.** The whole tree is on PyPI (§4.2) and workspace index settings do
  not propagate through published metadata anyway (§4.6).
* **`>=26.8`, not `==26.8.*`.** pip/uv select per-interpreter wheels and backtrack, so a future
  release without a wheel for the running Python is skipped rather than fatal; pinning would
  freeze users on a 2026 release. `cu13` is deliberately *not* offered by the extra (driver too old
  on much current hardware); the docs state the `cu12`/`cu13` split next to the install command.
* Already proven end-to-end from a clean venv with no extra index and no resolver flags
  (`exit=0`, `import cuopt → 26.08.00`, a real T4 solve matching §4.3), and cross-compiled clean for
  Windows (25 pkgs) and macOS (23 pkgs) with cuOpt absent. The re-run after WP-A merges is the
  acceptance test (`available_solvers` must then contain `cuopt`).

---

## 6. Requirement-ID map

§5 table rows numbered top to bottom as C-R1…C-R17, then my additions.

| ID | §5 row | Work that satisfies it | WP |
|---|---|---|---|
| C-R1 | `solvers.py:148` `SolverFeature` | Feature set chosen (§3); **zero-line diff to the enum** — no new member needed. Proven by the `test_solver_class_supports_feature` rows (§2.8). | A, F |
| C-R2 | `solvers.py:176` `TYPE_CHECKING` | `import cuopt` added to the block. | A |
| C-R3 | `solvers.py:304` `_LazyModule` | `cuopt = _LazyModule("cuopt")` + the "never import `cuopt.routing`" comment. | A |
| C-R4 | `solvers.py:346` `SolverName` | `cuOpt = "cuopt"` (vendor spelling; member name == class name for the shim). | A |
| C-R5 | `solvers.py:4156` `class cuPDLPx` | New `class cuOpt(Solver[None])` with the override set of §2.1 (LP+MILP). | A |
| C-R6 | `solvers.py:4440` `_SOLVER_PROBE_ORDER` | `"cuopt"` inserted before `"cupdlpx"`. Proven by the existing `test_is_available_matches_membership` (`test_available_solvers.py:39-45`), which fails if the entry is missing. | A |
| C-R7 | `solver_capabilities.py` `SOLVER_REGISTRY` | **Zero-line diff**, satisfied by proof: V0-4 asserts `SOLVER_REGISTRY["cuopt"]` resolves with the declared features and `display_name == "cuOpt"`. | D |
| C-R8 | `io.py:33`, `:792` | `TYPE_CHECKING` alias + `to_cuopt(m)` with the minimisation-form docstring. | B |
| C-R9 | `model.py:73`, `:2517` | import + `to_cuopt = to_cuopt` binding. | B |
| C-R10 | `constants.py:171` `TerminationCondition` | **Zero-line diff**; the §4 mapping uses only existing members. Proven by V1-4 (status mapping test). | A, D |
| C-R11 | `test/conftest.py:57-79` | **Zero-line diff**, satisfied by proof: V0-5 pastes the collect-and-skip without `--run-gpu` and the run with it. | D |
| C-R12 | `test_optimization.py:562/685/1110` | `time_limit` option row added; `test_infeasible_model` deliberately **not** skipped (capability win — cuOpt returns `PrimalInfeasible`); `test_basis_and_warmstart` skip extended. | D |
| C-R13 | `test_solvers.py` | 5 feature-matrix rows (+1 in WP-F). | D, F |
| C-R14 | `pyproject.toml` | The `cuopt` extra (§5i), plus the clean-venv acceptance test. | C |
| C-R15 | `doc/gpu-acceleration.rst` | New "cuOpt" section: install, hardware/driver, usage, supported classes, limitations (§9). | E |
| C-R16 | `prerequisites.rst`, `contributing.rst`, `index.rst`, `api.rst`, `release_notes.rst`, `README.md` | All six updated per §9. | E |
| C-R17 | `solvers.py:4426` `QUADRATIC_SOLVERS` | **Read-only.** QP build/solve path + tests; the list stays derived. Proven by V0-6 (`"cuopt" in linopy.solvers.QUADRATIC_SOLVERS`). | F |

New requirements this plan implies (next free IDs; for the lead to record in
`logs/30-decisions.md`):

| ID | Requirement | Origin | WP |
|---|---|---|---|
| C-R18 | **Honest degradation without a GPU**: `is_available()` device probe (failing open) + the one-shot actionable `logger.warning`; both branches demonstrated. | §4.6 c3 vs exploration §13 | A |
| C-R19 | **Always-minimise (M2)**: build with `-c` (`-0.5·Q` for QP), never `set_maximize(True)`, negate objective, duals and MILP `solution_bound` on return. | exploration §2a/§2b | A, F |
| C-R20 | **Default `method = 3` (Barrier)** when the user has not set one, plus the subprocess repeat-solve regression test. | exploration §9 | A |
| C-R21 | **KeyboardInterrupt worker-thread helper** `_run_cuopt_with_keyboard_interrupt`, plus its CPU dummy test. | exploration §8 | A |
| C-R22 | **MIQP guard** (`NotImplementedError` for `model.type in ("MIQP","IQP")`) and documented non-PSD behaviour. | exploration §12 | F |
| C-R23 | **Malformed-input pre-guards**: `M.A is None or M.A.nnz == 0` → `ValueError`; primal-length check downgrading to `internal_solver_error`; `get_error_message()` logged. | exploration §5/§6 | A |
| C-R24 | **Solver-option ergonomics**: unknown-parameter-name `ValueError` wrapped to name the option and the naming convention. | exploration §7 | A |
| C-R25 | **`log_fn` honoured** via the native `log_file` parameter (HiGHS precedence), not warn-and-ignored. | exploration §4 | A |
| C-R26 | **`examples/using-solvers.ipynb`** mentions cuOpt in its solver list (charter §6.6b's last row: "decide and log"). | charter §6.6b | E |

Every C-R1…C-R17 is mapped; four of them (C-R7, C-R10, C-R11, and C-R1's enum) are satisfied by
**proof of a zero-line diff**, which is stated explicitly so a reviewer cannot mistake them for
omissions.

---

## 7. Work-package breakdown

Rules honoured: solvers.py core first and alone among *dependent* packages; QP last as its own
package; packaging first-class and early; no two concurrent packages share a file. Each WP is one
worktree cut from the tip of `feat/cuopt-solver`, with the base SHA in its brief.

| WP | Files (exclusive) | Runs | C-R IDs | Commit |
|---|---|---|---|---|
| **A — solvers core (LP+MILP)** | `linopy/solvers.py` | first | C-R1…C-R6, C-R18–C-R21, C-R23–C-R25, and C-R10/C-R19 in part | 1 |
| **C — packaging** | `pyproject.toml` | **concurrent with A** (disjoint file) | C-R14 | 3 |
| **B — io + model** | `linopy/io.py`, `linopy/model.py` | after A merges | C-R8, C-R9 | 1 |
| **D — tests** | `test/test_cuopt.py` (new), `test/test_optimization.py`, `test/test_solvers.py` | after A+B merge; concurrent with E | C-R7, C-R10, C-R11, C-R12, C-R13 | 2 |
| **E — docs** | `doc/*.rst`, `README.md`, `examples/using-solvers.ipynb` | concurrent with D | C-R15, C-R16, C-R26 | 3 |
| **E2 — release notes** | `doc/release_notes.rst` | after E | part of C-R16 | 4 |
| **F — QP** | `linopy/solvers.py`, `test/test_cuopt.py`, `test/test_solvers.py`, `doc/gpu-acceleration.rst` | **last, alone** | C-R17, C-R22, and C-R1/C-R13/C-R19 deltas | 5 |

`E2` is split out only so the release-notes line can be its own commit, mirroring #516's commit 4;
it can be the same agent as E, dispatched twice, or folded in if the lead prefers 5 commits.

**Commit plan (§6.4a shape — plain imperative, no conventional prefix, per D2):**

1. `Implement interface to cuOpt solver and required IO` — WP-A + WP-B (3 files, mirrors #516's
   commit 1)
2. `Update tests` — WP-D
3. `Add to docs & add package dependency` — WP-E + WP-C
4. `Add comment for release notes` — WP-E2
5. `Add support for quadratic objectives` — WP-F
6. *(reserved)* Phase 5/6 fix commits, honestly labelled as the follow-up they are. I am not
   inventing a sixth substantive commit to hit a count.

Note commits 3 and 4 land **before** QP, so **the docs written in commit 3 must describe LP and
MILP only**, and commit 5 adds the QP paragraph. This keeps every intermediate commit truthful
(a reviewer reading commit 3 does not see docs promising an unimplemented feature) and is why WP-F
owns a docs file — sequentially, so no collision.

### WP-A — solvers core. Acceptance criteria

Every one requires **pasted output** in the slice report (I7).

* **AC-1** `/home/azureuser/.local/bin/ruff check .` and `.venv/bin/mypy` clean.
* **AC-2** `linopy.available_solvers` contains `"cuopt"`; `linopy.solvers.SOLVER_REGISTRY["cuopt"]`
  resolves; `linopy.quadratic_solvers` does **not** yet contain it (QP is WP-F).
* **AC-3** `_SOLVER_PROBE_ORDER` proof: `.venv/bin/pytest test/test_available_solvers.py -q` green
  (this suite's `test_is_available_matches_membership` is the mechanical check).
* **AC-4** Honest degradation, both branches pasted:
  `python -c "import linopy; print(linopy.available_solvers)"` contains `cuopt`; and
  `CUDA_VISIBLE_DEVICES='' python -c "..."` does **not**, and emits the warning line.
* **AC-5** LP and MILP end-to-end: a 2-variable `min` and `max` LP and the MILP variant solved via
  `model.solve("cuopt", io_api="direct")`, objectives and duals compared against
  `model.solve("highs")` **live in the same process**, `rtol=1e-6, atol=1e-7`.
* **AC-6** MILP dual shape: confirm `_solution_from_labels(np.array([]), self._clabels, n_cons)`
  behaves (returns NaN-filled or is bypassed); if it raises, follow Gurobi
  (`solvers.py:4128-4131`) and return the unscattered empty array. Report which.
* **AC-7** Repeat-solve subprocess test green: 3 sequential n≈1600 LP solves, `returncode == 0`.
* **AC-8** `_run_cuopt_with_keyboard_interrupt` dummy test green (CPU only).
* **AC-9** Guards: zero-nnz model → `ValueError`; unknown option name → the wrapped `ValueError`;
  `warmstart_fn` → `NotImplementedError`; `log_fn` produces a non-empty log file containing
  `cuOpt version`.
* **AC-10** `.venv/bin/pytest -q` and `.venv/bin/pytest --run-gpu -q` with **no regression** vs
  `logs/05-baseline-pytest*.txt`. (Test-file edits are WP-D's, so cuOpt is only lightly exercised
  by the shared suite here — that is expected, and AC-5/AC-7 carry the weight.)

### WP-B — io + model. Acceptance criteria

* `ruff`/`mypy` clean. `m.to_cuopt()` returns a `DataModel` whose
  `get_objective_coefficients()` equals `M.c` for a `min` model and `-M.c` for a `max` model
  (the M2 invariant, checked at the public surface).
* `doc/api.rst`'s `to_cuopt` entry is WP-E's; WP-B must not touch docs.
* `pytest -q` no regression.

### WP-C — packaging. Acceptance criteria

* `pip install '<worktree>[cuopt]'` in a **fresh venv that has never seen cuOpt**, with **no**
  extra index and **no** resolver flags → `exit=0`, `import cuopt` works. Paste the transcript tail.
* `pip install --dry-run --report` for `linopy[solvers]` shows **no** `cuopt`/`cudf`/`nvidia-*`/`rmm`
  package. Paste the filter result.
* `uv pip compile --python-platform windows|macos --extra cuopt` both resolve with cuOpt absent;
  `--python-platform linux` resolves with it present. Paste the three summary lines.
* **Deferred final check** (Phase 5 conformance row, after WP-A has merged): in that same clean
  venv, `python -c "import linopy; print(linopy.available_solvers)"` must contain `cuopt` — the
  end-to-end DoD line. WP-C cannot satisfy this alone and must say so rather than claim it.
* The venv is deleted afterwards (disk: charter §4.4).

### WP-D — tests. Acceptance criteria

* All tests of §8 present, each labelled with its verification-plan row in the slice report.
* `.venv/bin/pytest -q` and `.venv/bin/pytest --run-gpu -q` both green, **no regression** vs
  baseline, and the GPU tests demonstrably **ran** rather than skipped: paste the
  `--run-gpu -k cuopt` summary line showing non-zero passes.
* `pytest -k cuopt` **without** `--run-gpu` shows the tests skipped with conftest's reason string
  (this is the C-R11 proof).
* `test_infeasible_model[cuopt-direct-False]` passes **without a skip** (the capability win).
* Every V1 differential test calls HiGHS **live in the same process**; no hard-coded expected value
  anywhere. The slice report states this explicitly.

### WP-E / E2 — docs. Acceptance criteria

* Every row of charter §6.6b satisfied, itemised in the slice report.
* Docs build in the cuOpt-free venv with
  `PATH=/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin:$PATH
  /home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin/sphinx-build -b html doc doc/_build/check`,
  **no new warning** vs `logs/05-baseline-docs.txt` (diff the warning lists, paste the delta).
* Commit-3 docs claim **LP and MILP only**; no sentence promises QP.
* The install command shown first is `pip install 'linopy[cuopt]'`; Linux-only, driver ≥ 525.60.13
  and CC ≥ 7.0 appear **next to it**; the bare `cuopt` PyPI package is called out as a different
  project; `pypi.nvidia.com` appears at most as an optional fallback and never with
  `--index-strategy`.

### WP-F — QP. Acceptance criteria

* QP objective and duals match HiGHS live, all six sign cells, `rtol=1e-6, atol=1e-7`.
* The convention guard: a test asserting that `0.5·M.Q` (not `M.Q`) is what reaches cuOpt, by
  comparing objectives — i.e. the test would fail with the naive encoding, which returns `Optimal`
  and a wrong number.
* MIQP → `NotImplementedError`; non-PSD `Q` → `SolverStatus.error` with `internal_solver_error`.
* `"cuopt" in linopy.solvers.QUADRATIC_SOLVERS` **without** editing that list.
* The existing `feasible_quadratic_solvers` tests in `test_optimization.py` pass for cuOpt.
* `pytest --run-gpu -q` green, no regression.

---

## 8. Test plan

Tiers per charter §6.3a. Every V1 row calls HiGHS **live, in-process, on the identical model** —
never a baked-in expectation.

### V0 — structural (mechanical, every WP)

| ID | Check | Where |
|---|---|---|
| V0-1 | `ruff check .` clean | WP-A/B/C/D/E/F |
| V0-2 | `mypy` clean | same |
| V0-3 | `"cuopt" in linopy.available_solvers`; `SolverName("cuopt").name == "cuOpt"` | `test_cuopt.py` |
| V0-4 | `SOLVER_REGISTRY["cuopt"].features == cuOpt.supported_features()`, `display_name == "cuOpt"` (C-R7) | `test_cuopt.py` |
| V0-5 | GPU gating: `-k cuopt` skips without `--run-gpu`, runs with it (C-R11) | pasted CLI output |
| V0-6 | `"cuopt" in QUADRATIC_SOLVERS` after WP-F; `"cuopt" in NO_SOLUTION_FILE_SOLVERS` | `test_cuopt.py` |
| V0-7 | Feature-flag matrix rows | `test_solvers.py` (CPU) |

### V1 — differential vs live HiGHS (`test/test_cuopt.py`, `@pytest.mark.gpu` via conftest)

| ID | Model | Asserts | Tolerance |
|---|---|---|---|
| **V1-1…V1-6** | **The six-cell sign matrix**: `{min, max} × {<=, >=, ==}` on `A = [[1,2,1],[3,1,1]]`, `b = [4,6]`, `0 <= x <= 10`, objective per cell chosen so the primal optimum is `x = (1.6,1.2,0)` with both rows binding and a unique non-degenerate dual (exploration §2's model, so the numbers are already known-good) | primal, objective **and duals** vs live HiGHS | primal/objective `rtol=1e-6`; duals `rtol=1e-6, atol=1e-7` |
| **V1-7** | **The presolve-solvable `max` model** — a square equality system (`n = 2` and `n = 4`), the case exploration §2a reached from an ordinary `linopy.Model()` with `max|Δdual| = 3.61`. Two assertions: (a) duals vs live HiGHS; (b) `Solve(m.to_cuopt(), SolverSettings()).get_solved_by().name == "Unset"` — the guard that this model still exercises the presolve-only branch. Without (b) the test can silently stop covering the bug. | as above + the `solved_by` guard | duals `rtol=1e-6, atol=1e-7` |
| **V1-8** | Semi-continuous MILP (`x = 0` or `5 <= x <= 10`, `min x` s.t. `x <= 100`) | primal + objective vs live HiGHS (HiGHS has `SEMI_CONTINUOUS_VARIABLES`) | `rtol=1e-6` |
| **V1-9** | `_run_cuopt_with_keyboard_interrupt` with a dummy callable — **CPU only, no GPU mark**: interrupts the main thread mid-"solve" and asserts `KeyboardInterrupt` is raised promptly and the worker still completes. Modelled on `test_highs_keyboard_interrupt.py`. | timing + exception | — |
| **V1-10** | **Repeat solves in one process**, subprocess-isolated: 3 sequential solves of an n≈1600 / m≈800 LP with default options; assert `returncode == 0` and 3 `optimal` lines. Above the ~1300-variable crash threshold, so it fires if the Concurrent default ever returns. | `returncode`, statuses | — |
| **V1-11** | Status mapping: infeasible (`x >= 2` and `x <= 1`) → `infeasible`/`warning`; `time_limit=1e-6` on a larger LP → `time_limit`; a zero-nnz model → `ValueError` | `termination_condition` strings | exact |
| **V1-12** | MILP: `milp_model`-shaped model vs live HiGHS on objective; `report.dual_bound is not None`; duals are the empty/NaN shape, not garbage | objective `rtol=2.5e-4` (`GPU_SOL_TOL`, MILP objectives are exact integers here but the constant is the repo's own) | |
| **V1-13** *(WP-F)* | QP six sign cells + the `0.5·M.Q` convention guard + MIQP `NotImplementedError` + non-PSD → error status | objective + duals vs live HiGHS | `rtol=1e-6, atol=1e-7` |
| **V1-14** | Options: `log_fn` writes a log containing `cuOpt version`; unknown option name raises the wrapped `ValueError` naming the option; `time_limit=1` accepted | messages / file content | — |

### V2 — end-to-end on linopy's own models (Phase 5)

`pytest --run-gpu` over the whole shared suite, with cuOpt parametrized through
`params`/`feasible_mip_solvers`/`feasible_quadratic_solvers`/`gpu_solvers`. This is where cuOpt
meets `test_optimization.py`'s existing expected values at the existing `GPU_SOL_TOL = 2.5e-4`, and
where `test_infeasible_model` and `test_milp_model`'s `dual_bound` assertion run for real.

### Proposed NUMERIC tolerances and rationale

Freezing these is Gate 3's job; these are my proposals with the numbers behind them.

| Quantity | Proposed | Rationale |
|---|---|---|
| **Duals, V1 small models** | `rtol=1e-6, atol=1e-7` | Measured dual agreement with HiGHS on the exploration model: **1.73e-09 – 2.47e-09 absolute** under Barrier (this plan's default `method`), 8.18e-08 under Concurrent. `np.isclose` allows `atol + rtol·|y| = 1e-7 + 4e-7 = 5e-7` at `|y| = 0.4` — a 200× margin for Barrier and a 6× margin even if Barrier is ever swapped for Concurrent. The `atol` is *required*, not cosmetic: non-binding QP duals come back as ~1e-10 rather than exact 0, where `rtol` is meaningless. A sign error is O(1) relative, so this catches it by five orders of magnitude. |
| **Objective, V1 small models** | `rtol=1e-6` | On these models cuOpt agrees with HiGHS to ~1e-15 (`-2.80000000` vs `-2.7999999999999994`). Note what this number is **not** protecting against: exploration §7's 5.52e-3 relative objective shift came from `method=PDLP` at the default 1e-4 tolerance on a 2000×1000 random LP. This plan neither uses PDLP by default nor asserts an objective on a large model — the large-model test (V1-10) asserts *process survival and status only*. That is the deliberate way to keep a defensible tight tolerance without silently overriding cuOpt's tolerance parameters. |
| **Shared suite (`test_optimization.py`)** | `GPU_SOL_TOL = 2.5e-4`, **unchanged** | Already in the file (`:81`), chosen by #516 for GPU precision, and cuOpt inherits it automatically via `GPU_ONLY`. Introducing a second cuOpt-specific constant would be exactly the "widen a tolerance for my solver" move I12 forbids. Calibration: the four `method` values disagree with each other by ~2.5e-5 relative at default tolerances on an n=5000 LP (exploration §7), an order of magnitude inside 2.5e-4. |
| **QP objective and duals** | `rtol=1e-6, atol=1e-7` | Same as LP. QP is always solved by Barrier, and the measured six-cell QP duals matched HiGHS to ~1e-10 absolute. |
| **`solved_by` / status / termination** | exact string equality | Not tolerances; they must be exact. |

**No tolerance in this plan is to be widened.** If any of these misses, the responsible agent
reports the actual numbers verbatim and stops on that item (I12, §3.6). Two specific traps I want
named in `logs/15-verification.md` so nobody "fixes" them: (i) a dual comparison that passes only
after `set_optimality_tolerance` is tightened is a *changed default*, not a passing test; (ii) a
sign-matrix cell that passes because both solvers returned ~0 for a non-binding row is not
evidence — V1-1…V1-6's model is constructed so every dual is non-zero and unique.

---

## 9. Docs plan

Charter §6.6b's table, row by row. Commit 3 covers LP+MILP; commit 5 adds the QP sentences.

| Surface | Content |
|---|---|
| **`doc/gpu-acceleration.rst`** | New `cuOpt` subsection under "Supported GPU Solvers", **before** the existing `cuPDLPx` one (more capable solver first). ~55 lines: one-paragraph description (NVIDIA cuOpt, concurrent dual simplex + barrier + PDLP, LP/MILP/QP); **install**: `pip install 'linopy[cuopt]'` first, with "Linux x86_64/aarch64 only; NVIDIA GPU of compute capability ≥ 7.0; driver ≥ 525.60.13 (CUDA 12)" immediately beneath, then the manual `uv pip install 'cuopt-cu12>=26.8'` fallback and a warning that the bare `cuopt` package on PyPI is an unrelated project; a `cu12` vs `cu13` note (`cu13` needs a newer driver — use `cu12` unless you know otherwise); a **usage example** (`m.solve("cuopt", io_api="direct")`, plus one with `time_limit`/`method`); **supported problem classes** (LP, MILP, semi-continuous variables; QP added in commit 5); **limitations**: `io_api="direct"` only (no LP/MPS file IO), no SOS, no indicator constraints, no warm start / basis files, no reduced costs, no IIS, MIQP unsupported; **notable behaviours**: linopy defaults `method=3` (Barrier) because cuOpt's own default crashes on repeated solves in one process — pass `method=1` (PDLP) for very large sparse LPs; Ctrl-C returns immediately but the GPU solve continues until the process exits; option names are lower-case `snake_case` and integer-valued parameters need `0`/`1`, not `False`/`True`; a link to NVIDIA's option reference. Also extend the existing "Run specific GPU test" example with `-k cuopt`. |
| **`doc/prerequisites.rst`** | In the "GPU-accelerated solvers" list (`:65-73`), add cuOpt **above** cuPDLPx: *"NVIDIA cuOpt — GPU-accelerated LP / MILP / QP solver (Linux only; NVIDIA GPU with compute capability ≥ 7.0 and driver ≥ 525.60.13)"*, and add the code block `pip install 'linopy[cuopt]'` (equivalently `uv pip install 'linopy[cuopt]'`) with the requirements repeated on the adjacent line, plus the "the bare `cuopt` package on PyPI is a different project" caution. **No `--extra-index-url`, no `--index-strategy`, no `pypi.nvidia.com`** — the whole tree is on PyPI. |
| **`doc/contributing.rst`** | In the existing "GPU Testing" section (`:60-78`), one sentence: cuOpt's GPU tests are auto-marked like any `GPU_ONLY` solver, plus `pytest test/test_cuopt.py --run-gpu` and a note that the repeat-solve test spawns a subprocess. **Do not** touch the `uv sync --extra dev --extra solvers` line at `:46` (I9's trap (a) is about our own runs, not the docs). |
| **`doc/index.rst`** | Add `- NVIDIA cuOpt (GPU-accelerated)` with the docs link to the solver list (`:52-53`), above cuPDLPx. |
| **`doc/api.rst`** | Add `solvers.cuOpt` to the Implementations autosummary (`:637`, before `solvers.cuPDLPx`) — one line. **`to_cuopt` is deliberately not added**: I checked, and `api.rst`'s IO section (`:108-116`) lists only `Model.to_file`, `Model.to_netcdf` and `io.read_netcdf` — `to_gurobipy`, `to_highspy`, `to_xpress` and `to_cupdlpx` appear **nowhere** in `api.rst`. Listing only cuOpt's helper would be a stylistic outlier, and listing all five is a drive-by change (I4). Charter §6.6b asks api.rst to carry "`to_cuopt` and any other new public surface", so this is an **absence to justify in `logs/50-review.md`** at Gate 6, not an omission — the one-line escape hatch if the reviewer disagrees is `model.Model.to_cuopt` in the IO autosummary. |
| **`doc/release_notes.rst`** | Commit 4 adds one bullet under "Upcoming Version": *"Add support for the GPU-accelerated NVIDIA cuOpt solver (LP and MILP), installable with ``pip install 'linopy[cuopt]'`` on Linux."* Commit 5 (WP-F) amends that same line to read "(LP, MILP and QP)". Written this way so every commit is truthful at the moment it lands; if QP does not ship, the line already says the truth and only the docs' QP paragraph needs an honest "not currently supported, because …". |
| **`README.md`** | Add `* [NVIDIA cuOpt](https://docs.nvidia.com/cuopt/)` to the solver list (`:152-153`), above cuPDLPx. |
| **`pyproject.toml`** | The extra is documented in `doc/prerequisites.rst` (which is where extras are explained for users) and carries the inline comment block of §2.10 for anyone reading the TOML. |
| **`examples/using-solvers.ipynb`** | **Decision: yes, mention it** (C-R26). The notebook's intro sentence enumerates every backend ("…, or the GPU solver cuPDLPx"); leaving cuOpt out of an enumeration that names its sibling is the "name-drop vs explain" failure inverted. Change is two words in one markdown cell — *"…or the GPU solvers cuOpt and cuPDLPx"* — with **no new code cell**, because the notebook is executed during the docs build (`nbsphinx_execute = "auto"`, `nbsphinx_allow_errors = False`) on a machine with **no GPU**, where any cuOpt solve would fail the build. This constraint is the reason the change is deliberately minimal, and it must be stated in the WP-E brief. |

Cross-cutting statements the docs must make plainly (charter §6.6b): supported classes (LP, MILP,
QP, semi-continuous); unsupported surfaces (SOS, indicator constraints, MIQP, warm start, reduced
costs, file `io_api`); requirements (CC ≥ 7.0, driver ≥ 525.60.13, Linux); the `cu12`/`cu13` caveat.

---

## 10. Explicit non-goals

| # | Non-goal | Why |
|---|---|---|
| N1 | **File `io_api`s (`lp`, `mps`) via `cuopt.linear_programming.io`** | §1. Re-opens the sign-critical transform on a second path and multiplies the GPU test matrix by ~6. Additive later. |
| N2 | **Warm start / basis files** | §5h. Three non-default settings must align, the payload is not a file, and the LP initial-point API poisons the CUDA context. |
| N3 | **Reduced costs** | §5e. No linopy surface, and cuOpt's values are wrong for `max`/`<=`. |
| N4 | **Quadratic *constraints*** (`add_quadratic_constraint`) | linopy has no model surface for them; cuOpt also rejects equality quadratic constraints. |
| N5 | **`supports_persistent_update` / in-place updates** | `DataModel` has no incremental update API. |
| N6 | **MIQP** | Unsupported upstream (guarded, §5d). |
| N7 | **MIP callbacks** (`set_mip_callback`) | No linopy callback surface. |
| N8 | **IIS / `compute_infeasibilities()`** | No cuOpt surface. |
| N9 | **`solution_fn` / `presolve_file` / `user_problem_file` outputs** | Warn-and-ignore; the in-process solution is what linopy consumes. |
| N10 | **Diagnosing the upstream Concurrent SIGSEGV or the presolve dual-sign bug** | Both characterised, both mitigated, neither fixable from linopy. Record in `logs/45-known-issues.md`; an upstream issue report is the human's call. |
| N11 | **Changing `Model.solve`'s `io_api` defaulting or its "not installed" assertion text** | Outside the §5 map, changes behaviour for every solver (I4/E4). Recorded as a known rough edge (§5c, §1). |
| N12 | **Touching `cupdlpx`'s code, features or tests** | I4. cuOpt's arrival must not change one line of cuPDLPx behaviour. |
| N13 | **Benchmarks** | §4.4: the `benchmarks` extra pins `pandas==3.0.5`, incompatible with cuOpt's ceiling. Not run in this venv. |
| N14 | **A default `time_limit`** | §5f. No other linopy solver imposes one. |
| N15 | **A `close()` override / teardown** | No teardown API exists and memory returns to baseline after every solve, including a 14.2 GiB one. |

---

## 11. Risks specific to this plan

Ordered by how much I would worry about each, with the specific thing that would go wrong.

| # | Risk | Why it bites *this* plan | Early warning / mitigation |
|---|---|---|---|
| **R1** | **The `cuda.bindings` device probe is wrong for the installed wheel** — either the import path or the `(err, count)` return shape. | §5c is the one place I specify an API the exploration did not exercise. A wrong probe that raised would fail *open* (by design), so the visible symptom is not a crash but §4.6 c3 silently unmet. | WP-A AC-4 demands **both** branches pasted, including `CUDA_VISIBLE_DEVICES=''`. If the negative branch does not fire, the agent reports it and the fallback is `rmm.mr.available_device_memory()` or a tiny-solve probe moved into `_license_probe` — a decision for the lead, not a silent patch. |
| **R2** | **`method=3` (Barrier) is wrong for MILP.** I force it for every problem type; the exploration only swept LPs for the segfault and only measured method behaviour on LPs and QPs. | If `method` affects the LP relaxations inside cuOpt's branch-and-bound, MILP could get slower or (worst case) less accurate, and I would have made that trade blind. | The shared MILP suite (`test_milp_model`, `test_milp_model_r`, `test_model_with_inf`, the binary models) runs under the forced method in V2 — a real, existing oracle. WP-A AC-5 plus V1-12 cover it earlier. If MILP misbehaves, the narrow fix is to force the method only when the model has no integrality; that contingency is written into WP-A's brief so the agent does not have to invent it. |
| **R3** | **`MIP_DUAL_BOUND_REPORT` cannot be honoured on tiny models.** | Declaring it activates `assert report.dual_bound is not None` at `test_optimization.py:834`. If cuOpt solves linopy's 2-variable `milp_model` inside presolve and returns empty `milp_stats`, that assertion goes red — and the *wrong* fix (a `try/except` that leaves `dual_bound=None` while keeping the flag) would leave a lying feature declaration. | §3's explicit contingency: report the observation verbatim, drop the flag in the same WP, do not weaken the assertion. |
| **R4** | **`SEMI_CONTINUOUS_VARIABLES` is honest but unproven through linopy.** Exploration verified `'S'` via the raw API, not via `model.matrices` + a linopy semi-continuous variable (which also carries a positive-lower-bound requirement, `model.py:873`). | A declared feature that produces a wrong optimum is worse than an undeclared one. `mip_semi_continuous_big_m = 1e10` is also untuned. | V1-8 is a differential test against HiGHS on a discriminating model (continuous optimum 5, semi-continuous optimum 0). Same contingency as R3: fail → report numbers → drop the flag. |
| **R5** | **The `direct`-only decision is the one I most expect a sibling planner to have made differently.** If Gate 3 wants the file `io_api`s, WP-A's `_run_file` and the feature set change, and a second sign-transform path appears. | It is a scope call, not a correctness call, so it will not show up as a red test — it shows up as a reviewer saying "why not?". | §1 states the reasons in full so the selector can overrule it cheaply: adding `READ_MODEL_FROM_FILE` + a `_run_file` that reads back and un-maximises is a self-contained follow-up WP, not a rework of WP-A. |
| **R6** | **`model.solve("cuopt")` without `io_api="direct"` still writes an LP file and then fails.** Shared with cuPDLPx, out of scope (N11), but every doc example must remember `io_api="direct"` or users hit it immediately. | A reviewer could reasonably call this a bad first-run experience for a headline feature. | Every docs example carries `io_api="direct"`; the failure message names the fix; recorded in `logs/45-known-issues.md`. |
| **R7** | **Orphaned GPU thread after Ctrl-C.** `Solver._lock` is released as the `KeyboardInterrupt` unwinds, so a user who catches it and re-solves runs two concurrent cuOpt solves. | Undefined behaviour on the GPU, plausibly a crash — and it would look like *our* bug. | Documented in the class docstring with the explicit "start a new process" instruction. I deliberately do **not** add a guard flag: it would be unverifiable state with no way to clear it, and the alternative (joining the thread) costs 52.9 s of unresponsiveness. |
| **R8** | **Docs-build fragility from `examples/using-solvers.ipynb`.** The notebook executes during the build on a GPU-less machine. | A single cuOpt code cell added "for completeness" fails the docs build with `nbsphinx_allow_errors = False` — and the failure would look unrelated to cuOpt. | §9 restricts the change to a markdown sentence and says so in the WP-E brief; WP-E's acceptance criterion is a warning-list diff against the Gate 0 baseline. |
| **R9** | **The `0.5·M.Q` factor gets "corrected" by a later reader.** Passing the Hessian directly returns `Optimal` with a silently halved solution. | Nothing in the type system or the status catches it; only the differential test does. | The multi-line comment at the call site names both conventions, and WP-F's acceptance requires a test that *fails under the naive encoding* — so the guard is executable, not just a comment. |
| **R10** | **Bulk creep in `_solve`.** My plan adds an interrupt helper, a device probe, an error-message surfacer, a primal-length guard and an option-error wrapper on top of the cuPDLPx shape. Each is justified; together they are ~60 lines of "extra" over the template. | Phase 6 could reasonably flag the class as heavier than its sibling. | The estimate (§2) still lands at ~0.8× #516's total. Each addition is traceable to a numbered exploration finding and to a C-R ID (C-R18, C-R21, C-R23, C-R24), so the review conversation is about evidence rather than taste. If the lead needs to cut, the ordered give-up list is: the option-error wrapper (C-R24), then the primal-length guard (C-R23, partly), then the interrupt helper (C-R21). The device probe (C-R18) and M2 (C-R19) are not negotiable. |
