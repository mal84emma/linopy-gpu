# cuOpt support — pre-existing issues noticed, deliberately NOT fixed (I13)

## KI1 — `test/remote/` red at Gate 0 baseline (environmental)

*Phase 0, lead, 2026-08-22.* 46 FAILED in `test/remote/test_oetc.py`, 28 ERROR in
`test/remote/test_oetc_job_polling.py`: `ImportError: The 'google-cloud-storage' and
'requests' packages are required for OetcHandler` — the `oetc` extra is not installed in
`.venv`. Pre-existing and unrelated to solvers; excluded from all later comparisons by
directory (`test/remote/`). See `logs/05-baseline.md`.

## KI2 — `cupdlpx` not installed on this machine

*Phase 0, lead, 2026-08-22.* The existing GPU solver cuPDLPx is absent from `.venv`
(`ModuleNotFoundError`), so at baseline `pytest --run-gpu` exercises nothing GPU-specific
(counts identical to the CPU run). Not a defect to fix (I13); recorded so Phase 5 compares
"no new failure" rather than "identical counts".

## KI3 — `mypy` reports 28 pre-existing errors on `master`, all in the `xpress` paths

*Phase 4, WP-AB agent (Opus 5, 1M), 2026-08-22.* `/home/azureuser/linopy-gpu/.venv/bin/mypy linopy`
on untouched `master` (`09c34dd`, clean tree) prints `Found 28 errors in 1 file (checked 30 source
files)`. Every one is in `linopy/solvers.py`'s `Xpress` class and is a stale-API mismatch against
the installed `xpress` stubs, e.g.:

```
linopy/solvers.py:2985: error: "problem" has no attribute "chgobjsense"; maybe "chgObjSense"?  [attr-defined]
linopy/solvers.py:2983: error: Module has no attribute "maximize"  [attr-defined]
linopy/solvers.py:3103: error: "problem" has no attribute "readbasis"; maybe "readBasis"?  [attr-defined]
linopy/solvers.py:3054: error: Need type annotation for "m"  [var-annotated]
```

Consequence for this run: `logs/15-verification.md` **V0-2's threshold ("exit code 0, 0 errors")
is unreachable on this branch through no fault of the cuOpt work.** The cuOpt diff adds **zero**
new errors — the message sets on `master` and `wp/wp-ab` are identical (see `logs/40-blockers.md`
B11 for the pasted diff). Not fixed (I13): the xpress interface is unrelated to cuOpt and
repairing it would be a large drive-by diff (I4).

## KI4 — `linopy-problem-*` temp files accumulate in the default `solver_dir`

*Phase 4, WP-AB agent (Opus 5, 1M), 2026-08-22.* `/tmp` (the default `solver_dir`,
`gettempdir()`) held **176** `linopy-problem-*.lp`/`.mps` files timestamped `17:13` and `17:22` on
2026-08-22 — the Gate-0 baseline `pytest` runs, hours before any cuOpt code existed. A plain
`pytest -q` run on `wp/wp-ab` (cuOpt cells skipped, no GPU flag) added more, including two
**zero-byte** `-rw-------` files, which is the signature of
`Model.get_problem_file()`'s `NamedTemporaryFile(delete=False)` never being unlinked.

Not attributed to a specific call site and **not fixed** (I13, and out of WP-A/B scope). Recorded
because it made a frozen check vacuous until worked around: any future assertion of the form
`list(Path(m.solver_dir).glob("linopy-problem-*")) == []` must use a fresh `solver_dir`
(`tempfile.mkdtemp()`), never the process default — see `logs/40-blockers.md` B12. WP-D's V1-17
implementation needs this.

## KI5 — `doc/api.rst` documents only two of linopy's `to_*` solver bridges

*Phase 4, WP-E agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* The *IO* autosummary in
`doc/api.rst` (`:111-116` on `master`) lists `model.Model.to_file`, `model.Model.to_netcdf` and
`io.read_netcdf`. The other bridges bound the same way in `linopy/model.py` —
`to_gurobipy`, `to_highspy`, `to_cupdlpx`, `to_mosek`, `to_xpress`, `to_block_files` — are
**not** listed, so they have no API page.

Pre-existing gap, **not fixed** (I13, I4, and an explicit refusal in `plans/FINAL.md` §10 item 13).
WP-E added only `model.Model.to_cuopt`, which charter §6.6b names explicitly; the entry builds
(`doc/generated/linopy.model.Model.to_cuopt.rst` generated, page rendered, zero new warnings), so
the missing siblings are a documentation omission and not a Sphinx limitation.

**Resolved the other way (2026-08-23, post-run, human instruction):** the human lead read the
omission of the sibling bridges as a deliberate repo choice and had the `model.Model.to_cuopt`
line removed from `doc/api.rst` for consistency (amended into commit 3; branch rewritten
55ca033 → 7adc7e8). `solvers.cuOpt` stays in the *Implementations* list, where `solvers.cuPDLPx`
is also listed. `Model.to_cuopt` itself is unchanged — it is now undocumented in the API
reference exactly like every other `to_*` bridge.

## KI6 — `doc/prerequisites.rst` states "Python 3.9 or later"; `pyproject.toml` requires `>=3.11`

*Phase 4, WP-E agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* `doc/prerequisites.rst:8` tells
the reader to have "Python 3.9 or later installed", while `pyproject.toml` declares
`requires-python = ">=3.11"`. Noticed while editing the solver lists in the same file.

**Not fixed** (I13): it is unrelated to cuOpt, and correcting it would put an unrelated line in the
cuOpt diff (I4). Worth a one-line follow-up PR upstream.

## KI7 — `keep_files=True` forces a solution file even for `SOLUTION_FILE_NOT_NEEDED` solvers, so it raises for both GPU solvers

*Phase 4, WP-E agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* `Model.solve`
(`linopy/model.py:2098-2106`) suppresses the solution file only when the solver declares
`SolverFeature.SOLUTION_FILE_NOT_NEEDED` **and** `keep_files=False`:

```python
if solution_fn is None:
    if (solver_class.supports(SolverFeature.SOLUTION_FILE_NOT_NEEDED)
            and not keep_files):
        solution_fn = None
    else:
        solution_fn = self.get_solution_file()
```

So `keep_files=True` hands a `solution_fn` to a solver that cannot write one. cuOpt raises
`NotImplementedError("Solution file output is not yet implemented for cuOpt.")` — and cuPDLPx has
the byte-identical guard (`solvers.py:4448-4451`, "not yet implemented for cuPDLPx"), so this is a
pre-existing interaction, not a cuOpt regression. Measured on `wp/wp-e` (base `a2db13f`), fresh
`solver_dir` per case:

```
RESULT | solve(**{})                                          | OK ('ok', 'optimal') | files left: []
RESULT | solve(**{'io_api': 'direct'})                        | OK ('ok', 'optimal') | files left: []
RESULT | solve(**{'keep_files': True})                        | NotImplementedError: Solution file output is not yet implemented for cuOpt. | files left: [('linopy-problem-6ighyb4z.lp', 0), ('linopy-solve-n0r3q5_8.sol', 0)]
RESULT | solve(**{'io_api': 'direct', 'keep_files': True})    | NotImplementedError: Solution file output is not yet implemented for cuOpt. | files left: [('linopy-problem-y8i3a4vj.lp', 0), ('linopy-solve-47ceavae.sol', 0)]
```

**Not fixed** (I13, and `plans/FINAL.md` §10 item 2 makes solution-file output an explicit
non-goal). Two consequences recorded rather than acted on:

1. `plans/FINAL.md` §9's prescribed sentence for `doc/gpu-acceleration.rst` — "with
   `keep_files=True` an empty problem file remains" — does **not** describe the merged code: the
   solve *raises*. WP-E documented the measured behaviour instead (`keep_files=True` unsupported,
   raises `NotImplementedError`). See the WP-E return report, item 4.
2. Note also that `io_api="direct"` still creates a `linopy-problem-*.lp` (0 bytes) when
   `keep_files=True` — the same unlinked-temp-file surface as KI4.

## KI8 — cuOpt leaks per-thread OpenMP state on its mixed-integer path, so a new thread per solve crashes the process

*Phase 4, WP-D agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* Upstream cuOpt defect, recorded
rather than worked around (I13). With **no linopy involved** — raw
`cuopt.linear_programming.Solve` on a 20-variable MILP, `method=3`, ten times in one process:

```
called on the main thread                 -> 10/10 Optimal, exit=0
one NEW daemon thread per Solve() call    -> 2-5 Optimal, then
                                             Assertion failure at kmp_alloc.cpp(2725) ...
                                             OMP: Error #13   exit=134
one thread reused for all ten calls       -> 10/10 Optimal, exit=0
```

Pure-LP solves are unaffected (30/30 through linopy's thread-per-solve seam). Consequence for
linopy: `_run_cuopt_with_keyboard_interrupt` (`solvers.py:256-283`) starts a fresh thread per
solve, so the 4th–6th MILP solve in a process dies — see `logs/40-blockers.md` B17 for the full
matrix and why neither of FINAL.md's pre-decided contingencies covers it. Not fixed by WP-D:
`solvers.py` is out of its scope, and the shape of the fix (reuse one worker, or stay on the main
thread for integral models) is a Gate-3 decision.

## KI9 — cuOpt's `mip_gap` / `solution_bound` come back `±inf` on an unbounded MILP, and the objective `nan`

*Phase 4, WP-D agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* Noticed while building V1-4's
bounded `max` MILP; not a defect to fix and not asserted anywhere. Maximising `2x + y` with `x`
unbounded above returns `('ok', ...)` with `solution.objective = nan`, `report.dual_bound = -inf`
and `report.mip_gap = 0.0`. `logs/15-verification.md` V1-4 already rules `±inf` in-spec ("the
threshold is `is not None`, deliberately"), so V1-4's model was built bounded in both senses
instead (`x` in `[0, 5]`, `y` integer in `[0, 9]`, one `<=` row) and the bound is asserted to be a
bound. Recorded so nobody reads the `nan` as a linopy bug later.

## KI10 — V1-6b and V1-13's tolerance constants are not in `test_cuopt.py` yet (WP-F must add them)

*Phase 4, WP-D agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* Not a defect — a handover note so
the numbers are not re-invented. `logs/15-verification.md`'s constants table lists
`CUOPT_PRIMAL_ATOL_DEGENERATE = 1e-4` and `CUOPT_DUAL_ATOL_QP = 1e-6`; both are used only by rows
that need quadratic support (V1-6b's `quadratic_model` minus `con0`, and V1-13a/b/c), which are
WP-F's. WP-D left them out rather than commit two unused module constants (I4). **WP-F must add
them verbatim from the frozen table** — `CUOPT_PRIMAL_ATOL_DEGENERATE: float = 1e-4`,
`CUOPT_DUAL_ATOL_QP: float = 1e-6` — next to the constants already in `test/test_cuopt.py:47-63`,
and must not pick different numbers.

Also note V1-6b itself is **not** implemented by WP-D: its fixture (`quadratic_model` with `con0`
removed) is a QP, so on `a2db13f` it raises `ValueError: Solver cuopt does not support quadratic
problems.` from `Solver._validate_model` before any pad-row code runs. The non-quadratic halves of
V1-6 are covered (`test_model_without_constraints`, `test_model_without_constraints_or_bounds_raises`,
`test_duals_stay_aligned_with_the_constraints`).

### KI8 addendum — linopy no longer triggers it

*Phase 4, WP-B17 agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* The upstream cuOpt defect
recorded above is unchanged and unfixed (it is NVIDIA's). What changed is linopy's exposure to it:
`_run_cuopt_with_keyboard_interrupt` now hands every solve to **one** persistent daemon thread
instead of starting a fresh one per call, so no solve ever lands on a new thread. Evidence and
before/after matrix in `logs/40-blockers.md` under "B17 — FIXED". Keep this entry: if a future
change reintroduces a thread per solve, this is the reason it will start aborting again.

**KI8 addendum (lead, 2026-08-23T00:55Z):** filed upstream as https://github.com/NVIDIA/cuopt/issues/1768 (corrected scope: per-solve THREAD CREATION leaks OpenMP per-thread state in the bundled LLVM libomp — LP and MILP both affected, MILP fastest; our '30/30 LP clean' was under-threshold at N=10, refuted at N=30). linopy solve path fixed by the persistent worker (D12); disposal path still exposed — see B19.

### KI8 addendum 2 — the disposal path is routed too, and what remains outside linopy's reach

*Phase 4, WP-B19 agent (Opus 5, `claude-opus-5[1m]`), 2026-08-23.* B19 was the same upstream
defect reached through **disposal** rather than solving: `Solver.__del__` -> `close()` dropped the
cuOpt `DataModel` on whatever thread the cyclic collector happened to run on. `cuOpt.close()` now
hands the native model to the persistent `linopy-cuopt-solve` worker instead
(`_dispose_on_cuopt_worker`, `solvers.py`); evidence in `logs/40-blockers.md` under
"B19 — FIXED".

Two surfaces are **deliberately left alone** because linopy does not own the references:

1. **User-held native objects.** `Result.solver_model`, `model.solver.solver_model` and the
   `DataModel` returned by `linopy.io.to_cuopt` are the caller's; whichever thread drops the last
   of those references runs the C++ teardown there. `plans/FINAL.md` §2 requires `solver_model` to
   hold the real `DataModel` (and `to_cuopt` returns it), so this cannot be removed without
   breaking C-R6/C-R7 — see the WP-B19 report, item 2.
2. **The cuOpt `Solution` object.** `sol` in `cuOpt._solve` is created on the worker but is a local
   of `_solve`, so it is released on the thread that called `model.solve()`. That is the main
   thread in every measured configuration (20 sequential MILP solves and the full `--run-gpu`
   suite are clean), and routing it would mean restructuring `_solve` rather than a surgical
   change (I4). A user who calls `model.solve("cuopt")` from inside a dask task is still exposed.

Both disappear when NVIDIA fixes https://github.com/NVIDIA/cuopt/issues/1768.

## KI11 — cuOpt 26.08.00 can destabilise the host process after any in-process solve (upstream, B21)

A CPython process that has solved with cuOpt (any problem class, `io_api="direct"`) can crash
LATER, at an unrelated native call — another solver, numpy, garbage collection — because cuOpt's
bundled LLVM OpenMP runtime intermittently corrupts glibc's main-arena large-bin metadata
(diagnosed under gdb: `_int_malloc` follows a NULL `fd_nextsize`; full evidence B21,
`logs/b21-diag-*.txt`). Measured 8 crashes in 12 full `pytest --run-gpu` suite runs at `26546ee`;
six distinct crash sites, all after `test_cuopt.py` ran in-process. linopy is exonerated: the
DataModel deep-copies its inputs (pyx audit), all buffer-pinning and thread-affinity mitigations
were built and measured to have no effect, and 420 solves + allocator-debug tooling produced zero
linopy-attributable corruption. **This is a user-facing statement about cuOpt 26.08.00, not a test
artifact**: long-lived processes (dask workers, notebooks, servers) that solve with cuOpt are
exposed until NVIDIA fixes it. Upstream: NVIDIA/cuopt#1768 (+ evidence comment). Consequence for
this project: V2-7 is permanently red with cause (D16/ESC-2); every *completed* suite run remains
byte-identical to baseline, so correctness is unaffected — only process stability.

## KI12 — `Model.copy()` silently drops a quadratic objective (pre-existing linopy defect)

*Phase 4, WP-F agent (Opus 5, `claude-opus-5[1m]`), 2026-08-23.* Not a cuOpt defect and **not
fixed here** (I13, I4 — it is production behaviour in a file this work package does not own).

`linopy/io.py:1270` (on master `09c34dd`, unchanged by this arc) rebuilds the objective
unconditionally as a `LinearExpression`:

```python
obj_expr = LinearExpression(m.objective.expression.data.copy(deep=deep), new_model)
```

A `QuadraticExpression` objective is therefore downgraded on copy: the copy reports
`model.type == "LP"`, `model.is_quadratic == False`, `matrices.Q is None`, and
`matrices.c` raises `IndexError: boolean index did not match indexed array along axis 0;
size of axis is 7 but size of corresponding boolean axis is 14` (the quadratic term's
two-variable rows are still in the flattened coefficient array).

Measured on master, both `deep=True` and the `copy.copy`/`copy.deepcopy` protocols route through
the same function:

```
orig QP True   ->   copy LP False
```

**Consequence for WP-F, and why it matters beyond a nuisance:** a differential test that builds its
HiGHS oracle with `model.copy()` compares cuOpt's QP answer against **HiGHS solving the LP
relaxation**. Measured on the V1-13a three-variable model: the copied oracle returned
`objective = -55.0`, `x = [10, -10, -5]` where the true QP optimum is `-4.2857142857`,
`x = [1.5714, -0.1429, -2.0]` — a wrong expectation, off by 12x, that no assertion in the test could
distinguish from a solver bug. `test/test_cuopt.py`'s QP rows therefore use `solve_qp_with_both`,
which **builds** the oracle twice instead of copying it; the helper's docstring records why, so the
next reader does not "simplify" it back to `.copy()`. `solve_with_both` (LP/MILP) is unaffected and
keeps using `.copy()`.

A user-facing fix belongs in `linopy/io.py:copy` (pick `type(m.objective.expression)` the way
`_copy_expr` already does for named expressions) with its own test in `test/test_io.py`; it is a
separate change from cuOpt support and is left to the maintainers.

**Reported upstream (2026-08-23, post-run, on the human lead's instruction):**
https://github.com/PyPSA/linopy/issues/903. Filed by a dispatched agent after a clean duplicate
check (five REST `search/issues` variants + two pages of recent titles, open and closed, zero hits)
and after re-replicating against a fresh `upstream/master` (`09c34dd`) worktree — the offending
line is unchanged since `Model.copy` landed in PR #623 and is byte-identical at `v0.9.1`. The issue
follows the repo's `bug_report.yaml` structure, opens with the AGENTS.md `> [!NOTE]` AI-generated
marking, and cites permalink `linopy/io.py#L1270` at `09c34dd`. It does not mention the cuOpt
branch beyond "differential testing of quadratic-objective support".
