# cuOpt support — blockers and difficulties (append-only)

## B1 — bare `uv pip install` targeted the wrong Python (resolved)

*Phase 0, lead, 2026-08-22.* Bare `uv pip install 'cuopt-cu12==26.8.*'` resolved against the ambient conda env (`/anaconda/envs/azureml_py38`, Python 3.10) and failed: no cp310 wheel for cuopt-cu12 26.8.0 (cp311–cp314 only, matching §4.2). Resolved by passing `--python /home/azureuser/linopy-gpu/.venv/bin/python`. See D1.

## B2 — cuOpt segfaults the interpreter on a specific parameter combination (worked around, not fixed)

*Phase 1, spike agent A (Opus 5), 2026-08-22.* `a_01_statuses.py` died with `Segmentation fault` (exit 139) part-way through, losing all buffered stdout. Isolated in `spikes/a_01b_segfault_repro.py`: `method=PDLP` + `time_limit=1e-6` + `save_best_primal_so_far=True` crashes the process **deterministically, 3/3 attempts**; each parameter alone returns normally (`no_timelimit -> Optimal`, `no_savebest -> TimeLimit`). Worked around by removing that case from the status sweep and running everything else with `python -u`. Not fixable from linopy — recorded as an upstream cuOpt 26.08.00 defect; the plan should not expose `save_best_primal_so_far` unguarded. Remaining consequence: none for this phase's findings.

## B3 — a pathologically scaled LP with `method=PDLP` and no time limit does not terminate (worked around)

*Phase 1, spike agent A (Opus 5), 2026-08-22.* The `badly scaled LP (1e18 coefficients)` case in `a_01c_statuses_more.py` never returned: the process held the T4 at 98% utilisation for ~13 minutes and was killed by its own `timeout 900`, truncating the run before the NaN/inf and non-PSD-Q cases. Worked around by re-running that block as `spikes/a_01d_numerical.py` with `time_limit=10`, which returns `TimeLimit` for PDLP and `Optimal obj=-2` for DualSimplex/Barrier/Concurrent. All intended statuses were subsequently forced. No evidence lost.

## B4 — `presolve=False` rejected; boolean-vs-int typing on `set_parameter` (resolved)

*Phase 1, spike agent A (Opus 5), 2026-08-22.* Six cases in `a_01c_statuses_more.py` failed with `ValueError: Parameter presolve value False is not an integer`, even though `log_to_console=False` is accepted. Resolved by passing `presolve=0` / `presolve=1`. This cost one re-run of the unbounded-status sweep; it also matters for the plan (the options dict must not naively forward Python bools).

## B5 — a background spike was truncated by a task notification (resolved by re-running in the foreground)

*Phase 1, spike agent A (Opus 5), 2026-08-22.* `a_02e_presolve_dual_sign.py` launched with `run_in_background` produced only 133 of ~150 output lines and never reached its summary table, despite the wrapper reporting exit 0. Re-run in the foreground it completed (197 lines, full correlation table). All quantitative claims in the findings file come from the complete foreground run; the truncated output was discarded, not used.

## B6 — `nvidia-smi` via `subprocess.run` after a cuOpt solve segfaulted; the real cause was cuOpt's default `method` (resolved, and it became a finding)

*Phase 1, spike agent B (Opus 5, 1M), 2026-08-22.* `b_q9_repeat.py` died with `Segmentation fault` (exit 139) on its 3rd sequential solve, losing all buffered output. First hypothesis was `fork()`-after-CUDA from my `subprocess.run(["nvidia-smi", ...])` memory sampler. Isolated with `b_q9_fork_segv.py` (subprocess vs no-subprocess, 2 runs each): **both** crash, so subprocess was not the cause. Narrowed with `b_q9b_segv_narrow.py` / `b_q9e_sweep.txt`: `method=0` (Concurrent, cuOpt's **default**) segfaults deterministically on the 2nd-3rd `Solve()` in one process for LPs above ~1300 variables, while `method=1/2/3` complete 25 varied sequential solves cleanly. Worked around for my own measurements by (a) reading GPU memory in-process via `rmm.mr.available_device_memory()` instead of `nvidia-smi`, and (b) running the repeat-solve sections under `method=2`. Not fixable from linopy — recorded as an upstream cuOpt 26.08.00 defect and flagged as the most consequential finding in `spikes/exploration-B.md` §9/Surprises: linopy's test models sit *below* the threshold, so the suite would stay green while user scripts crash.

## B7 — `DataModel.set_initial_primal_solution` poisoned the CUDA context mid-spike (not fixed; avoided)

*Phase 1, spike agent B (Opus 5, 1M), 2026-08-22.* Section H of `b_q11_warmstart.py` raised `RuntimeError: __copy:: D->H: failed: cudaErrorIllegalAddress`, after which **every** subsequent solve in the process raised too, aborting the rest of the run. Isolated in a fresh process with `b_q11b_initialpoint.py` (5 modes): the LP `set_initial_primal_solution` path crashes with all-default settings and with `presolve=0`/`method=1`, on a correctly-sized C-contiguous float64 array; `set_initial_dual_solution` alone is fine; the MILP path is fine. No workaround exists — the API is unusable for LPs. Consequence recorded: linopy's cuOpt class must never call it on an LP, since one attempt takes down an entire pytest session.

## B8 — Q13 clean-venv install vs 18 GB of free disk (resolved)

*Phase 1, spike agent B (Opus 5, 1M), 2026-08-22.* `pip install '<worktree>[cuopt]'` needs ~6.7 GB installed plus a similar wheel cache, against 18 GB free. Resolved by passing `--no-cache-dir` (a cache flag, not a resolver or index flag — the charter's "no resolver flags, no extra index" constraint is untouched: no `--extra-index-url`, `--index-strategy`, `--find-links` or `--pre` was used), and by deleting `.venv-q13`, the `q13-install` worktree and the `wp/q13-install` branch immediately after the evidence was captured. Disk back to 18 GB free; main checkout clean.

## B9 — WP-C clean-venv install vs 18 GB free disk, second time (resolved, same fix as B8)

*Phase 4, WP-C agent (Opus 5, 1M), 2026-08-22.* Re-running Q13 against the real
`pyproject.toml` edit needs the same ~6.7 GB venv on the same 18 GB of free disk, plus a
591 MB `libcuopt-cu12` wheel pip builds locally in `TMPDIR`. Resolved exactly as B8:
`--no-cache-dir` (a *cache* flag; no `--extra-index-url`, `--index-strategy`, `--find-links`
or `--pre` was passed, so the "no extra index, no resolver flags" constraint is intact).
Peak usage was 11 GB free / 91% during the install; back to 18 GB / 86% after cleanup.

**Throwaway venvs deleted — confirmed.** `/home/azureuser/linopy-gpu/dev-scripts/cuopt/worktrees/wp-c-venvs/`
(containing `.venv-wpc`, 6.7 GB) was removed with `rm -rf` after the evidence was captured;
`ls` on `worktrees/` now shows only `wp-ab` and `wp-c`, and `df -h /` reports `18G` avail /
`86%` — identical to the pre-run figure. The build by-products the pip install left inside the
`wp-c` worktree (`build/`, `linopy.egg-info/`, `linopy/version.py`, `.ruff_cache/` — all
gitignored) were also deleted; `git status --porcelain --ignored` in the worktree is now
**0 lines**. Evidence transcripts were copied to
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/spikes/c_wpc_*` before deletion.

## B10 — V3-1's `available_solvers` half cannot pass on `wp/wp-c` (expected; deferred, not widened)

*Phase 4, WP-C agent (Opus 5, 1M), 2026-08-22.* `logs/15-verification.md` V3-1 requires
`linopy.available_solvers` to contain `cuopt` (occurrence count >= 1) after the clean-venv
install. On `wp/wp-c` the measured value is `[]` — occurrence count **0**, i.e. the frozen
threshold is **not met**, reported verbatim rather than widened (§3.6/I12). Cause is
structural, not packaging: the branch is `09c34dd` + a `pyproject.toml`-only commit, so
`hasattr(linopy.solvers, "cuOpt")` is `False` and `_SOLVER_PROBE_ORDER` is
`('gurobi', 'highs', 'glpk', 'cbc', 'scip', 'cplex', 'xpress', 'knitro', 'mosek', 'mindopt',
'copt', 'cupdlpx', 'pips')` — no `"cuopt"` entry. This is exactly the ordering caveat carried
verbatim in the WP-C brief and in `plans/FINAL.md` §7.2: the packaging half passes
(`pip install` exit 0, `import cuopt` -> `26.08.00`, charter §4.3 LP solved on the T4 from the
fresh venv) and the membership half **must be re-verified after WP-A merges**. No workaround
attempted and none is possible from `pyproject.toml`; the item is deferred, not closed.

## B11 — `mypy` is not clean on `master`, so V0-2's threshold is unreachable

*Phase 4, WP-AB agent (Opus 5, 1M), 2026-08-22.* `logs/15-verification.md` V0-2 requires
`mypy` → "exit code `0`, `0` errors". Measured on **untouched `master`** in the main checkout
(`/home/azureuser/linopy-gpu`, clean tree, `09c34dd`):

```
$ /home/azureuser/linopy-gpu/.venv/bin/mypy linopy
...
Found 28 errors in 1 file (checked 30 source files)
```

All 28 are pre-existing `xpress` errors in `linopy/solvers.py` (`"problem" has no attribute
"chgobjsense"; maybe "chgObjSense"?`, `Module has no attribute "maximize"`, …), i.e. the
threshold is unreachable for **any** agent on this branch. Resolve-first steps taken: (1) grepped
`doc/contributing.rst`, `AGENTS.md` and `.pre-commit-config.yaml` — **no** documented mypy
invocation and mypy is not a pre-commit hook, so no "official" narrower target exists; a bare
`.venv/bin/mypy` errors with `Missing target module, package, files, or command`; (2) ran the same
command on `master` and on `wp/wp-ab` and diffed the message sets with line numbers stripped.

**What I did instead of widening anything:** reported the miss verbatim in the slice report and
proved the *delta* is zero rather than declaring the absolute threshold met —

```
$ diff <(master mypy output, line numbers stripped, sorted) <(wp/wp-ab same)
1c1
< Found 28 errors in 1 file (checked 30 source files)
---
> Found 28 errors in 1 file (checked 29 source files)
```

(the 30-vs-29 file count is `linopy/version.py`, a setuptools-scm artifact that exists only in the
main checkout). Zero new errors, zero new error kinds. Recorded as KI3; **not fixed** (I13).

## B12 — criterion 8's temp-file leak check was vacuous against the default `solver_dir`

*Phase 4, WP-AB agent (Opus 5, 1M), 2026-08-22.* FINAL.md §7.2 WP-A item 8 / V1-17 assert
`list(Path(m.solver_dir).glob("linopy-problem-*")) == []`. The default `solver_dir` is
`gettempdir()` = `/tmp`, which already held **176** `linopy-problem-*` files from the Gate-0
baseline runs (timestamps `2026-08-22_17:13` and `17:22`, i.e. hours before this worktree
existed). Against that directory the check can only ever fail, and would have failed for a reason
that has nothing to do with the `_build_file` override — the charter §9 "a harness check that
cannot fire" failure mode, in its opposite form.

Overcome by pointing the model at a fresh `tempfile.mkdtemp()` directory before the bare solve, so
the assertion measures exactly the file this solve created. Result: `leftover
linopy-problem-* files = []` **and** `all files left in solver_dir = []`. The pre-existing `/tmp`
litter is recorded as KI4 and not investigated further (out of scope, I4/I13).

## B13 — fork-harness child evidence was being swallowed by `os._exit`

*Phase 4, WP-AB agent (Opus 5, 1M), 2026-08-22.* The V0-7/V1-12 harness prints the child's
outcome and then calls `os._exit()`, which does not flush stdio, so the first run showed only the
wait status and none of the diagnostic line. Added an explicit `sys.stdout.flush()` before
`os._exit`, after which the naive-probe variant prints the actual upstream failure —
`child: FAILED InputValidationError: A CPU-memory problem requires remote execution. Set
CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT to solve on a remote GPU server.` with wait status `768`
— matching the exploration's W4 measurement exactly. Worth recording because the check *passed*
both times: without the flush the evidence was an exit code with no explanation, which is weak
evidence for a check whose whole purpose is to be able to fire.

## B14 — a docs build inside a worktree documents the MAIN checkout's `linopy`, not the worktree's (resolved here; a trap for later WPs)

*Phase 4, WP-E agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.*
`.venv-docs` was created with `uv pip install --python .venv-docs/bin/python -e '.'` **from
`/home/azureuser/linopy-gpu`**, so its editable finder resolves `import linopy` to
`/home/azureuser/linopy-gpu/linopy` regardless of the cwd sphinx-build runs in. Measured from the
worktree:

```
$ .venv-docs/bin/python -c "import linopy;print(linopy.__file__)"          # cwd = worktree root
/home/azureuser/linopy-gpu/dev-scripts/cuopt/worktrees/wp-e/linopy/__init__.py   # only because '' is on sys.path for -c
$ cd doc && .venv-docs/bin/python -c "import linopy;print(linopy.__file__)"
/home/azureuser/linopy-gpu/linopy/__init__.py
```

Consequence: `doc/api.rst`'s autosummary rows (`solvers.cuOpt`, `model.Model.to_cuopt`) are
resolved against the main checkout's working tree, **not** the worktree's branch. Harmless for
WP-E — the main checkout was on `feat/cuopt-solver` @ `a2db13f`, byte-identical in `linopy/` to
`wp/wp-e`, and both stubs generated and rendered (`doc/generated/linopy.solvers.cuOpt.rst`,
`doc/generated/linopy.model.Model.to_cuopt.rst`).

**Later agents (WP-F especially): before trusting a docs build run from a worktree, check that
`/home/azureuser/linopy-gpu` is on a branch that already contains the code your new autosummary
rows reference.** If it is not, the build silently documents the wrong tree, or emits a "failed to
import" warning that looks like a docs bug. `PYTHONPATH=<worktree>` does not fix it reliably
because the editable finder wins for a top-level package already installed.

## B15 — `NotebookEdit` rewrites a whole cell's `source` to one JSON string and drops the file's trailing newline (avoided)

*Phase 4, WP-E agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* Editing the one markdown cell of
`examples/using-solvers.ipynb` with `NotebookEdit` produced a 13-line diff for a one-sentence
change: the cell's `"source"` list-of-lines collapsed into a single `"...\n..."` string and the
final newline of the file was removed (`\ No newline at end of file`). That is a reformatting
drive-by (I4) in a file the change barely touches.

Resolved by `git checkout examples/using-solvers.ipynb` and doing the edit as a plain string
replacement on the raw `.ipynb` text (with `json.load` afterwards as the validity check). Final
diff: 2 lines changed, 2 added, list format and trailing newline preserved. **Any later agent
touching a notebook should edit the raw JSON text, not use `NotebookEdit`.**

## B16 — `plans/FINAL.md` §9 prescribes a docs sentence the merged code does not honour (`keep_files=True`)

*Phase 4, WP-E agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* FINAL.md §9's row for
`doc/gpu-acceleration.rst` requires the note "with `keep_files=True` an empty problem file
remains". Measured on `wp/wp-e` (base `a2db13f`), `keep_files=True` does not leave a file and
carry on — it **raises** `NotImplementedError: Solution file output is not yet implemented for
cuOpt.`, both with and without `io_api="direct"` (full four-case table in
`logs/45-known-issues.md` KI7).

Not a threshold miss (no number in `logs/15-verification.md` covers it), and not fixed in code
(I13; solution-file output is FINAL.md §10 non-goal 2). Resolved by documenting the **measured**
behaviour: the limitations list now says `keep_files=True` is unsupported and raises. Flagged to
the lead in the WP-E return report, item 4, because it is a deviation from a Gate-3 contract row.

## B17 — repeated MILP solves crash the interpreter: `_run_cuopt_with_keyboard_interrupt` starts a new thread per solve (NOT fixed — production code, out of WP-D scope)

*Phase 4, WP-D agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* `pytest --run-gpu` **cannot be
green on `a2db13f`**, with or without the WP-D test files: the 4th–6th mixed-integer solve in one
process kills the process (SIGSEGV `exit 139`, or SIGABRT `exit 134` with
`OMP: Error #13: Assertion failure at kmp_alloc.cpp(2725)`). LP solves are unaffected.

Measured on `wp/wp-d` @ `a2db13f`, every line a separate fresh process:

```
# through linopy (Model.solve -> cuOpt._solve -> _run_cuopt_with_keyboard_interrupt)
30 sequential LP solves (3 vars)                 -> 30/30 optimal, exit=0
20 sequential MILP solves (20 vars, method=3)    -> 3 optimal then exit=139
12 LP solves then 2 MILP solves                  -> 12 LPs + 1st MILP ok, 2nd MILP exit=134
10 MILP solves, method=0 (cuOpt's own default)   -> exit=139
10 MILP solves, method=1 (PDLP)                  -> exit=139
10 MILP solves, method=2 (DualSimplex)           -> 10/10 optimal, exit=0

# raw cuopt.linear_programming, same 20-var MILP, method=3, no linopy involved
10 solves called on the MAIN thread              -> 10/10 Optimal, exit=0
10 solves, ONE NEW daemon thread PER SOLVE       -> 2-5 Optimal then exit=134 (kmp_alloc assertion)
10 solves, ONE thread reused for all ten         -> 10/10 Optimal, exit=0
```

Root cause is therefore not `method` and not linopy's model translation: it is
`solvers.py:256-283` starting a **fresh** `threading.Thread` for every `Solve()`. cuOpt's
branch-and-bound path leaks per-thread OpenMP state, and the allocator assertion fires once enough
threads have come and gone. A single long-lived worker thread survives; the main thread survives.

Independent of the WP-D files — reproduced with plain scripts and with the *existing* suite:

```
$ pytest test/test_optimization.py -q --run-gpu -k "cuopt and (milp or inf)"
......Fatal Python error: Aborted
exit=134
```

**Not fixed here:** `linopy/solvers.py` is WP-A's/WP-F's file and WP-D may not touch production
code. Also *not* covered by FINAL.md §8.5 contingency 5 / §5b's named contingency G8: G8's only
authorised fallback is "pass the default only when the model has no integrality", i.e. hand MILPs
cuOpt's own `method=0` — which is measured above to crash as well; and `method=2`, the only method
that survives, is explicitly forbidden ("never invents a third option"). So this is a
**report-and-stop** for the lead: it needs a decision at the `solvers.py` level (reuse one worker
thread, or run the solve on the main thread when `model.type` has integrality), which is a Gate-3
question, not a WP-D one.

WP-D's own 100 tests all pass — 84 in one process with the 16 mixed-integer cells deselected, and
each of those 16 green in its own process (pasted in the WP-D report). The cells are kept in the
file rather than skipped: skipping them would hide this defect.

## B18 — `milp_time_limit`'s frozen forcing recipe does not produce `TimeLimit` through linopy (reported, not substituted)

*Phase 4, WP-D agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* `logs/15-verification.md`'s status
table forces MILP `TimeLimit` (5) with "120-variable subset-sum with a fractional rhs,
`time_limit=1.0`". Rebuilt as a linopy model with the exploration script's exact numbers
(`a_01c_statuses_more.py:69-76`: `rng = default_rng(3)`, `w = rng.integers(1_000_000, 9_999_999,
120)`, `target = float(w.sum() // 2) + 0.5`, all-integer `x` in `[0,1]`):

```
[milp_time_limit] ok suboptimal 2.23s      # FeasibleFound, not TimeLimit
```

The exploration measured this recipe under cuOpt's own default `method=0`; linopy defaults to
`method=3` (Barrier, FINAL.md §5b), under which the same model terminates `FeasibleFound`.

The second recipe from the same exploration row (6x50 market-split, `time_limit=2.0`) *does* give
`time_limit` through linopy — but only once per process, and it is one of the crash triggers in
B17:

```
[market_split tl=2.0] ok time_limit 3.29s
[tiny lp after]        ok optimal            # sometimes
# repeat the same market-split solve twice in one process:
[market_split tl=2.0] ok time_limit
OMP: Error #13: Assertion failure at kmp_alloc.cpp(2725).   exit=134
```

Per charter §3.6 / I12 the frozen recipe's miss is **reported verbatim and work stops on that
item**: `test_status_map[milp_time_limit]` is **not** in `test/test_cuopt.py`. No substitute recipe
was quietly adopted, no threshold widened, and no test skipped to paper over it. Every other row of
the frozen status table is implemented and green.

### B17 addendum — the acceptance evidence, per criterion

*WP-D agent, 2026-08-22.* What the crash does and does not cost, measured on `wp/wp-d` @ `e5344f8`:

| Run | Result |
|---|---|
| `pytest -q` (no GPU flag) | `46 failed, 8545 passed, 889 skipped, 817 warnings, 28 errors in 471.05s` — FAILED/ERROR **set identical** to `logs/05-baseline-pytest.txt` (`diff` empty over all 80 lines), `+18 passed`, `+198 skipped` |
| `pytest -q -k cuopt` (no GPU flag) | `18 passed, 198 skipped` — every GPU cuOpt cell skipped; the 18 are `test_cuopt_interrupt.py` (6) and the class-level feature rows in `test_solvers.py` (12) |
| `pytest -k cuopt --run-gpu -v` | 28 cells PASSED, then `Fatal Python error: Aborted` in `test_milp_differential[v1-min]` (B17) |
| `pytest --run-gpu -q` | `Fatal Python error: Segmentation fault` at 7%, same test, same cause (B17) |
| `test_cuopt.py --run-gpu -k "not milp and not semi_continuous"` | `84 passed, 16 deselected in 78.27s` |
| the 16 deselected cells, one pytest process each | 16/16 `1 passed` |
| `test_optimization.py test_solvers.py --run-gpu -k "cuopt and not milp and not binary and not inf"` | `85 passed, 9 skipped` |
| the 16 excluded existing-suite cuOpt cells (`milp*`, `binary`, `*_with_inf`, `infeasible_model`), one process each | 16/16 `1 passed` |

So every V1 and V2 row implemented by WP-D is green; only the *aggregate* `--run-gpu` run is red,
and it is red for B17's production-side reason. V2-3 needed **no** contingency:
`test_from_name_set_names_false[legacy-cuopt]` and `[v1-cuopt]` both PASSED at `pytest.approx`'s
default `rel=1e-6`, so FINAL.md §8.5 item 2 was **not** applied.

### B17 — FIXED: one persistent worker thread for every cuOpt solve

*Phase 4, WP-B17 agent (Opus 5, `claude-opus-5[1m]`), 2026-08-22.* Fixed in `linopy/solvers.py`
only, on `wp/wp-b17` (base `a2db13f`, `LINEAGE OK`; folded into commit 1 by `--amend` per D12 →
`67675e4`). `_run_cuopt_with_keyboard_interrupt` no longer starts a thread per call: a lazily
started daemon thread (`linopy-cuopt-solve`) takes `_CuoptJob`s off a module-level
`queue.SimpleQueue` (`_cuopt_solve_queue`, `@functools.cache`) and writes the result or the
exception back into the job. Call sites unchanged. `+53 / −21`, one file.

Before → after, every line a separate fresh process, `PYTHONPATH` pinned to the worktree:

```
                                                    BEFORE (a2db13f)      AFTER (67675e4)
20 sequential MILP model.solve("cuopt")             1 optimal, exit=134   20/20 optimal, exit=0
                                                    (OMP #13 kmp_alloc.cpp(2725))
pytest test_optimization.py --run-gpu               7 dots then           12 passed,
  -k "cuopt and (milp or inf)"                      Fatal Python error:   3446 deselected
                                                    Aborted, exit=134     in 3.88s, exit=0
10 LP / 10 MILP alternating, one process             (not run)            20/20 optimal, exit=0
3 x n=2000/m=1000 LP (V1-8 / C-R20 guard)            (not run)            3 optimal lines, exit=0
```

The BEFORE abort's faulthandler traceback names the seam directly — `solvers.py:272 in _target`
(the per-call worker) under `solvers.py:279 in _run_cuopt_with_keyboard_interrupt` — so the fix
target was confirmed on the crash, not inferred.

Interrupt semantics retained (CPU, no cuOpt call, dummy callables through the helper):

```
[control] uninterrupted 3.0s callable returned after 3.000s (must be >= 3.0)
[interrupt] KeyboardInterrupt reached the caller after 0.200s -- under 1s: True
[interrupt] abandoned solve still running on the worker: True
[interrupt] abandoned solve completed on the worker: True
[after] subsequent call returned 'second-call-ok' in 0.000s
[after] ran on the SAME worker thread as the interrupted solve: True
[error] worker exception propagated to the caller: ValueError('worker blew up')
[error] helper still usable: 'reusable-after-error', same worker thread: True
[threads] after 25 calls, live 'linopy-cuopt-solve' threads: 1
[queued] abandoned solve still in flight: True
[queued] next call returned 'queued-behind-ok' after 1.800s (waited for the abandoned solve: True),
         abandoned finished: True
```

The `[control]` line is there to prove the `< 1 s` check **can** fire (charter §9): without the
`interrupt_main()` the same helper blocks for the full 3.0 s.

No behavioural regression on the solve path — charter §4.3 LP, cuOpt vs **live in-process** HiGHS,
frozen V1-1 thresholds (obj/primal `rtol=1e-6, atol=1e-6`; duals `rtol=1e-6, atol=1e-7`):

```
cuopt : obj=2.7999999944487737 primal=[1.5999999924202215, 1.2000000020285524] duals=[0.40000000077826353, 0.19999999827276196]
highs : obj=2.7999999999999994 primal=[1.6, 1.1999999999999993] duals=[0.4, 0.19999999999999996]
|dobj|     = 5.551e-09  (tol 3.800e-06)
max|dx|    = 7.580e-09  (tol 2.200e-06)
max|ddual| = 1.727e-09  (tol 3.000e-07)
```

`ruff check .` → `All checks passed!`; `ruff format --check linopy/solvers.py` → `1 file already
formatted`; `mypy linopy` → `Found 28 errors in 1 file (checked 29 source files)`, all in the
`Xpress` paths (lowest line 2838), i.e. exactly KI3's pre-existing 28 and **zero** new.

Still red and **not** B17's: `pytest test_optimization.py test_solvers.py --run-gpu -k cuopt` gives
`4 failed, 89 passed, 7 skipped` — `test_solver_time_limit_options[{legacy,v1}-cuopt-direct-False]`
(`KeyError: 'cuopt'`) and `test_basis_and_warmstart[{legacy,v1}-cuopt-direct-False]`
(`NotImplementedError: Warmstarting…`). Verified pre-existing: the identical 4 failures reproduce
with the fix stashed on the untouched base. These are the test-side solver branches WP-D adds
(charter §5, `test_optimization.py` row) and are green on `wp/wp-d` — nothing for B17 to do.

KI8 stands as the upstream defect record; the linopy-side trigger it names is now removed.

## B19 — Intermittent GC-time abort: cuOpt native teardown on a foreign (dask) thread [OPEN — blocking, fix deferred to the timer-gated round]

*Found by: lead, Phase-4 step-3 stability runs · 2026-08-23*

Full --run-gpu suite on the merged 4-commit branch: 2 aborts in 4 runs (attempt 1: crash, identity lost to a broken capture pipeline; run 2 + proper re-run: clean, failure set = GPU baseline; run 3: exit 134 with full faulthandler dump, logs/phase4-gpu-stability-run3.txt line 39). Run-3 attribution: the crashing thread is a DASK WORKER executing linopy graph tasks; mid-task garbage collection finalizes an unreachable solver instance — Solver.__del__ → close() (solvers.py:1241/1246) — which drops the last references to cuOpt native objects (close() sets solver_model=None), running their C++ teardown on the dask thread → SIGABRT. The persistent cuOpt solve worker (solvers.py:282) is idle at crash time, so B17's solve-path fix holds; this is the DISPOSAL path of the same upstream per-thread OpenMP defect (KI8; shape matches upstream #1219's teardown SIGSEGV). Intermittent: requires a dead cuOpt solver instance + GC firing on a non-main thread. Candidate fixes for the fix round (NOT chosen yet): route cuOpt disposal through the persistent worker (enqueue a drop-references job from close()); or avoid retaining native cuOpt objects in solver_model. Any fix is a solvers.py change → fix agent + §6.4a re-checks + GPU stability re-runs.

### B19 — FIXED: `cuOpt.close()` releases the native model on the persistent solve worker

*Phase 4, WP-B19 agent (Opus 5, `claude-opus-5[1m]`), 2026-08-23.* Fixed in `linopy/solvers.py`
only, on `wp/wp-b19` (base `9e1e96e`, `LINEAGE OK`; folded into commit 1 by the §6.4a
amend-and-rebase recipe → `67675e4` becomes `95c0fcb`, tip `f2d1188`). `+27 / −0`, one file:
a module-level `_dispose_on_cuopt_worker(native)` next to `_cuopt_solve_queue`, and a `close()`
override on `cuOpt` that takes `solver_model` out of the instance and hands it to that helper
before calling `super().close()`.

**Design choice — candidate (a), route disposal through the persistent worker.** Candidate (b)
("never retain native objects in `solver_model`") was rejected on the plan, not on taste:
`plans/FINAL.md` §2 prescribes `_make_result(status, solution, solver_model=dm, …)` "mirroring
`cuPDLPx._solve`'s tail" (FINAL.md:527-528), `linopy/io.py:799-808`'s `to_cuopt` **returns**
`solver.solver_model` and is annotated `-> cuoptDataModel` (C-R6/C-R7), and
`test/test_cuopt.py:228` assigns a real `DataModel` to `instance.solver_model`. (b) would have
broken all three. No deviation from FINAL.md was needed for (a): `solver_model` still holds the
real `DataModel` after this change, and `close()` still sets it to `None` synchronously — only
the thread the C++ destructor runs on changes.

Handover properties, all required by the brief: `SimpleQueue.put` never blocks, so a `close()`
inside `__del__` does not wait; a collection firing **on** the worker itself just queues the drop
behind the job it is running (no deadlock); and `contextlib.suppress(Exception)` degrades to a
plain local drop when no worker can be reached (interpreter shutdown), which is what `__del__`
would have done anyway.

**The crash itself would not reproduce on demand.** Six reproducer shapes on the unfixed base,
each a fresh process, every one exit 0 — logs `b19-prefix-v{1..7}-*.txt`:

```
fresh gc thread per iteration, Model<->Solver cycle collected there:
  25 LP                                        -> exit 0   (b19-prefix-v1-lp.txt)
  25 MILP                                      -> exit 0   (b19-prefix-v2-milp.txt)
  25 mixed LP/MILP                             -> exit 0   (b19-prefix-v2-mixed.txt)
 200 LP                                        -> exit 0   (b19-prefix-v3-lp200.txt)
1313 mixed LP/MILP (stopped deliberately at    -> 0 aborts, 0 OMP lines
      1313 of 2000 to free wall clock)                     (b19-prefix-v4-mixed2000.txt)
ONE long-lived collector thread (dask-pool-shaped), n=2000/m=500, 6 solved
models kept alive for GPU pressure:
  20 mixed                                     -> exit 0   (b19-prefix-v6-big.txt)
 200 mixed                                     -> exit 0   (b19-prefix-v7-big200.txt)
```

Reported plainly rather than papered over: **no targeted reproducer was found that aborts
deterministically pre-fix.** Note this refutes the "thread churn" reading of B19 — 1313 distinct
collector threads each ran a `DataModel` destructor without a `kmp_alloc` assertion — and it is
consistent with the run-3 dump carrying **no** OMP diagnostic line, unlike B17's aborts. Whatever
the C++ cause (a stream-ordered RMM free under per-thread default streams is the better fit; the
build links `cuda.bindings._bindings.cyruntime_ptds`), the *observable* is the same and is what
the fix targets: the destructor must not run on a foreign thread.

**So the property is checked directly instead of waiting for the abort**
(`dev-scripts/b19_teardown_thread.py`, two deterministic checks):

* **A — routing.** `cuOpt.close()` called from a foreign thread, with a Python sentinel wrapping a
  real `DataModel` in `solver_model`, must release it on `linopy-cuopt-solve`. The sentinel's
  refcount reaching zero names the releasing thread. **Base: 5/5 on the foreign thread (FAIL).
  Fixed: 0/5 foreign, 5/5 on `linopy-cuopt-solve` (PASS), 5 runs in a row.**
* **B — GC path.** A solved model dropped and collected on a foreign thread must reach
  `_dispose_on_cuopt_worker` from `__del__`. Fixed: 5/5 handovers, none from `MainThread`.

```
--- UNFIXED base (logs/b19-prefix-teardown-thread.txt) ---
[A0] native model released on thread: foreign-close-0
...
[A] releases on a foreign thread: 5/5
(check B cannot run: `_dispose_on_cuopt_worker` does not exist on the base)
EXIT=1

--- FIXED (logs/b19-postfix-teardown-thread.txt) ---
[A0] native model released on thread: linopy-cuopt-solve
...
[A] releasing threads: ['linopy-cuopt-solve']
[A] releases on a foreign thread: 0/5
[B0] __del__ handed the DataModel over from thread: foreign-gc-0, ...
[B] iterations whose foreign-thread collection handed over: 5/5
A (routing) : PASS
B (gc path) : PASS
EXIT=0
```

An earlier probe using `weakref.finalize` on the `DataModel` was **discarded as vacuous** and is
recorded here so nobody rebuilds it: CPython's collector runs `handle_weakrefs` *before*
`finalize_garbage`, so the callback fires on the collector thread whether or not the object is
resurrected into the queue — it reported "foreign-gc-N" identically before and after the fix.
Same for `__del__` on a garbage wrapper. Only a wrapper that is **resurrected out of** the
unreachable set (check A's shape) measures the deallocating thread.

**Full-suite stability — the frozen threshold for this item, 3/3.** `pytest --run-gpu -q` from the
worktree, `PYTHONPATH` pinned, three consecutive runs, exit code captured with an explicit
`PYTEST_EXIT=$?` append (never through a pipe):

```
run1: Fatal=0 | 46 failed, 8734 passed, 700 skipped, 827 warnings, 28 errors in 554.23s | PYTEST_EXIT=1
run2: Fatal=0 | 46 failed, 8734 passed, 700 skipped, 827 warnings, 28 errors in 551.14s | PYTEST_EXIT=1
run3: Fatal=0 | 46 failed, 8734 passed, 700 skipped, 827 warnings, 28 errors in 550.36s | PYTEST_EXIT=1
```

`diff` of the 80-line FAILED/ERROR set against `logs/05-baseline-pytest-gpu.txt` is **empty for all
three** — the pre-existing KI1 `test/remote/` set, nothing else. `PYTEST_EXIT=1` is that set, not
an abort (an abort is 134/139). Pre-fix this same suite aborted in 2 of 4 runs.

No behavioural regression: B17's checks re-run green (20 sequential MILP `20/20 optimal exit 0`;
mixed 10 LP/10 MILP `20/20 optimal exit 0`), the interrupt contract holds (`< 1 s`, worker
completes the abandoned solve, helper reusable, 1 live worker after 25 calls), and the charter §4.3
LP differential against **live in-process** HiGHS reproduces B17's numbers to the digit at the
frozen V1-1 thresholds (`|dobj| 5.551e-09` vs tol `3.8e-06`; `max|dx| 7.580e-09` vs `2.2e-06`;
`max|ddual| 1.727e-09` vs `3.0e-07`). `ruff check .` clean; `mypy linopy` = the 28 pre-existing
KI3 Xpress errors, message set **identical** to the base modulo line numbers (0 new).

Residual exposure linopy cannot own (user-held `solver_model` / `to_cuopt` return value, and the
cuOpt `Solution` local in `_solve`) is recorded in `logs/45-known-issues.md`, KI8 addendum 2.

#### B19 — FIXED, addendum: V2-7 executed at its frozen count of FIVE runs (not three)

*WP-B19 agent, 2026-08-23.* The entry above was written after three runs, which is what my brief
asked for. The frozen plan is stricter: `logs/15-verification.md` **V2-7** (added in round 5, digest
`feef773d8fface670eda923095ee216de32a8fedf50b0561caaebe9687a3f820`, which matches the file on disk
and both recorded copies) requires **five consecutive** runs and says in terms that "fewer runs is a
**weakening** and requires a return to Gate 3, not a judgement call at execution time". Two further
runs were therefore executed on the committed tip (`f2d1188`, porcelain clean) and **all five** are
reported against V2-7's four exact thresholds:

```
run1: Fatal=0 | summary-line-with-passed=1 | symdiff-vs-baseline=0 | 46 failed, 8734 passed, 700 skipped, 827 warnings, 28 errors in 554.23s | PYTEST_EXIT=1
run2: Fatal=0 | summary-line-with-passed=1 | symdiff-vs-baseline=0 | 46 failed, 8734 passed, 700 skipped, 827 warnings, 28 errors in 551.14s | PYTEST_EXIT=1
run3: Fatal=0 | summary-line-with-passed=1 | symdiff-vs-baseline=0 | 46 failed, 8734 passed, 700 skipped, 827 warnings, 28 errors in 550.36s | PYTEST_EXIT=1
run4: Fatal=0 | summary-line-with-passed=1 | symdiff-vs-baseline=0 | 46 failed, 8734 passed, 700 skipped, 827 warnings, 28 errors in 550.67s | PYTEST_EXIT=1
run5: Fatal=0 | summary-line-with-passed=1 | symdiff-vs-baseline=0 | 46 failed, 8734 passed, 700 skipped, 827 warnings, 28 errors in 551.32s | PYTEST_EXIT=1

TOTAL 'Fatal Python error' across the five logs: 0        (threshold: exactly 0)
runs whose exit status is 134 or 139:            0        (threshold: exactly 0)
logs containing a pytest summary line:           5/5      (threshold: >= 1 each)
FAILED/ERROR symmetric difference vs baseline:   0 for all five (threshold: 0)
```

Logs: `b19-gpu-run{1,2,3,4,5}.txt`. `PYTEST_EXIT=1` is the pre-existing KI1 set, which V2-7 names as
expected and not a miss.

Also confirmed against round 5's other tightening, **V1-8 at N=20** (raised from 3 because 3 sat
below the 5–13-solve abort threshold measured upstream): 20 sequential `n=2000 / m=1000` LP solves
in one process on the fixed tip → `20/20 optimal`, `DONE 20/20`, exit 0
(`b19-postfix-v18-n2000x20.txt`). The WP-D test file still asserts 3, which the frozen row already
flags as lagging code — not this slice's file.

## B20 — frozen node ids in `15-verification.md` omit master's `legacy`/`v1` convention prefix (documentation mismatch, no code defect)

*Phase 4, WP-TESTS agent (Opus 5, `claude-opus-5[1m]`), 2026-08-23.* The frozen status table names its
tests as e.g. `test/test_cuopt.py::test_status_map[milp_time_limit]`, and my brief's acceptance item 2
used that literal node id. It does not exist and cannot be selected:

```
$ pytest 'test/test_cuopt.py::test_status_map[milp_time_limit]' --run-gpu -v --collect-only
collecting ... ERROR: not found: .../test/test_cuopt.py::test_status_map
(no match in any of [<Module test_cuopt.py>])
collected 0 items
```

master `d9805ed` ("feat: v1 semantic convention") added a suite-wide parametrization, so every cuOpt
test carries a convention prefix — the real ids are:

```
test/test_cuopt.py::test_status_map[legacy-milp_time_limit]
test/test_cuopt.py::test_status_map[v1-milp_time_limit]
```

This is why the cuOpt file collects 100 items for 50 tests (102 for 51 after this slice). Nothing is
wrong with the code or the recipe: **each frozen cell is executed twice, once per convention**, which
is strictly more coverage than the row asks for. Recorded so a conformance reviewer copying node ids
straight out of the frozen table does not read `ERROR: not found` as a missing test. `15-verification.md`
is chmod 444 and was not edited; the fix, if any is wanted, is a lead-side note, not a test change.

### B18 — CLOSED: the re-measured recipe is now implemented in the test file

*WP-TESTS agent (Opus 5, `claude-opus-5[1m]`), 2026-08-23.* Round 5 struck the 120-variable
subset-sum recipe and froze the 6x50 market-split one; this slice implements it as the
`milp_time_limit` cell of `test_status_map` (commit 2, now `2720ee1`). Re-measured through linopy
before writing the test, one fresh process:

```
ERROR:linopy.solvers:cuOpt returned 0 primal values for 50 variables (TimeLimit).
Status: ok / Termination condition: time_limit / Solution: 0 primals, 0 duals
A[0,:6] = [81.0, 8.0, 17.0, 23.0, 18.0, 80.0]   A.sum() = 14702.0
rhs = [1182.5, 1292.5, 1369.5, 1122.5, 1279.5, 1106.5]
STATUS ok CONDITION time_limit WALL 3.31   nvars 50   primal.size 0
```

Every frozen fingerprint reproduced exactly (`rhs` matches the row digit for digit, wall 3.31 s vs the
row's 3.30 s in-process figure). The two fingerprint assertions are carried inside the builder so a
numpy stream change fails loudly instead of swapping the instance. Both "code lags this row" caveats
in `15-verification.md` — this one and V1-8's `range(3)` — are now discharged.

## B21 — one full-suite `--run-gpu` run segfaulted inside SCIP's `optimize()` (new signature, under investigation)

**Status:** OPEN — reproduction runs in flight.
**Found:** 2026-08-23 ~02:45Z, first Gate-4 pre-WP-F full GPU run at `26546ee` (`logs/gate4-pre-wpf-gpu.txt`).

**What happened.** `pytest test/ --run-gpu` exited 139 (SIGSEGV) at ~37%, during
`test/test_optimization.py::test_default_settings_small_slices` — faulthandler's current thread is
the MAIN thread inside `SCIP._run_file` → `m.optimize()` (`solvers.py:2692`, PySCIPOpt native call).
This is NOT the B19 signature (that was cuOpt teardown on a dask worker thread; B19's fix is present
and its deterministic check passes 5/5). The cuOpt persistent worker (`solvers.py:282 _worker`) was
idle-parked at crash time. No OMP:/kmp_alloc diagnostic in the dump.

**Why it may still be ours.** The GPU baseline (master) never imports cuOpt, so SCIP never shares a
process with cuOpt's bundled LLVM libomp there. On our branch, `test_cuopt.py` runs first and loads
it. A second OpenMP runtime colliding with SCIP/Soplex's is a plausible intermittent mechanism —
same family as B17/NVIDIA/cuopt#1768, different victim. Against that: five consecutive full runs at
`f2d1188` (identical solver code; only `test/test_cuopt.py` differs at `26546ee`, +2 in-process
cuOpt MILP time-limit solves) were abort-free, and the CPU run in the same batch was baseline-identical
(74-id failure set, symdiff 0). Prior probability of a one-off environmental flake is nonzero.

**Evidence:** `logs/gate4-pre-wpf-gpu.txt` (faulthandler dump from line ~72); comparison logs
`b19-gpu-run{1..5}.txt` (clean at f2d1188); `logs/gate4-pre-wpf-cpu.txt` (clean).

**Next:** two reproduction runs at `26546ee` (`gate4-pre-wpf-gpu-run{2,3}.txt`, in flight). If clean →
intermittent; decide between more runs / targeted SCIP+cuOpt coexistence reproducer before WP-F.
If it reproduces at the same frame → bisect the test-file delta (the two new in-process MILP
time-limit solves are the prime suspect for changed OMP state history).

**B21 update (03:07Z):** Reproduced — 2 crashes in 3 runs at `26546ee`. Run 2 clean (symdiff 0);
run 3 exit 139 with a SECOND distinct frame: main-thread GC executing generic `Solver.close()`
:1259 (`self.solver_model = None`) from `__del__` :1264 at ~47% of test_optimization — the decref
of a non-cuOpt solver's native model segfaults (cuOpt's B19 `close` override at :4923 is not in the
stack). Two different native-crash sites + 2/3 frequency vs 0/5 at `f2d1188` (test-file-only delta)
points at process-wide native/OpenMP state corruption sourced from the branch-new in-process cuOpt
work. Prime suspect: the two `milp_time_limit` cells — in-process time-limited cuOpt MILP solves
(upstream #1768's fast-trigger path was MILP). Bisect in flight: 3 full `--run-gpu` runs with both
cells deselected (`b21-bisect-no-mtl-run{1..3}.txt`).

**B21 update (03:58Z) — CAUSATION ESTABLISHED, source narrowed to in-process cuOpt activity.**
Bisect matrix at `26546ee` (each cell = full `--run-gpu` suite):
- with all cuOpt tests: **4 crashes / 6 runs**, four DISTINCT native crash sites (SCIP optimize; GC decref
  of a non-cuOpt solver_model; numpy `_get_ufunc_and_otypes` ×2 during fixture setup), all in
  test_optimization AFTER test_cuopt ran in-process;
- with only the 2 milp_time_limit cells deselected: 2 crashes / 3 runs → those cells exonerated;
- with ALL cuOpt activity removed (`--ignore=test/test_cuopt.py -k "not cuopt"`, 116 deselected):
  **0 crashes / 3 runs**, counts identical to the master GPU baseline (46F/8527P/691S/28E).
Machine health clean (GPU 43°C idle, 36 Gi RAM free). Interpretation: cuOpt in-process solving
intermittently corrupts the process heap; the crash then fires at a random later native call.
f2d1188's earlier 5/5 clean is now read as sampling luck, and B19's dask-thread abort may have been
one manifestation of the same corruption. Open hypotheses for the diagnosis agent: (H-a) linopy-side
buffer-lifetime bug — cuOpt's `DataModel.set_*` may hold zero-copy views of arrays `_build_direct`
lets die, later heap reuse corrupts; (H-b) upstream cuOpt native defect (#1768 family); (H-c) worker
accrual. Diagnosis agent dispatched with allocator-debugging tools (MALLOC_CHECK_, MALLOC_PERTURB_,
PYTHONMALLOC=debug) to move the abort next to the corrupting write.

### B21 — DIAGNOSIS: H-a REFUTED, verdict H-b (upstream cuOpt native defect)

*Phase 4, B21 diagnosis agent (Opus 5, `claude-opus-5[1m]`), 2026-08-23.* Diagnosis only —
no production code touched, `git status --porcelain` = 0 lines throughout, tip `26546ee`.
Scripts in `dev-scripts/cuopt/spikes/b21_diag_*`, evidence in `logs/b21-diag-*`.

**Verdict: H-b, high confidence. H-a is refuted twice over — at the API boundary and
behaviourally. H-c is not supported.**

**1. H-a refuted at the API boundary (`logs/b21-diag-boundary.txt`).** cuOpt's
`DataModel.set_*` does NOT hold zero-copy views. `data_model_wrapper.pyx`'s `type_cast`
ends in `np_obj.astype(np.dtype(np_type))`, i.e. an unconditional copy, and stores it in a
cdef attribute of the `DataModel`; `set_data_model_view()` then takes
`get_data_ptr(self.get_<field>())` — a pointer into the DataModel's OWN buffer, never into
the caller's. Measured for every array `_build_solver_model` passes:

```
field                        in dtype   held dtype same buffer?  owned
A_values                     float64    float64    False         True
A_indices                    int32      int32      False         True
A_offsets                    int32      int32      False         True
constraint_lower_bounds      float64    float64    False         True
constraint_upper_bounds      float64    float64    False         True
variable_lower_bounds        float64    float64    False         True
variable_upper_bounds        float64    float64    False         True
objective_coefficients       float64    float64    False         True
variable_types               <U1        <U1        True          True

-- mutate every array we passed in, then read the DataModel back --
A_values                     unchanged after caller mutation: True
...  (all eight numeric fields: True)
variable_types               unchanged after caller mutation: False
```

`variable_types` is the only aliasing setter (`self.variable_types = variable_types`, no
copy) — and it cannot dangle either: the DataModel holds the only strong reference until
`solver_wrapper.Solve` replaces it with `type_cast(..., "S1")`, an astype copy, BEFORE
`set_data_model_view()` runs. That cast is also what makes the `<const char*>` count
correct (`<U1` is 4 bytes/elem, `S1` is 1):

```
variable_types dtype BEFORE Solve-time cast : dtype('<U1')
variable_types dtype AFTER  Solve-time cast : dtype('S1')
  itemsize <U1 -> S1 : 4 -> 1 bytes/elem
set_data_model_view() -> ok
```

Every length/index invariant the C++ `data_model_view_t` relies on also holds for all seven
model shapes the suite uses (2x2 charter LP, 20-var MILP, 30 binaries, equality block, the
no-constraint pad-row path, 200x100 random LP, mixed binary/continuous): `len(lb)==len(ub)
==len(vtypes)==len(c)`, `len(cl)==len(cu)==len(offsets)-1`, `offsets[-1]==nnz`,
`indices.max() < n_vars`, all C-contiguous and aligned, no views. `INVARIANTS OK` on all
seven rows. So there is no out-of-bounds handed to cuOpt either.

**2. H-a refuted behaviourally — the suite still crashes with NOTHING freed**
(`logs/b21-diag-arm-pinsettings-run{1,2,3}.txt`, summary
`logs/b21-diag-arm-pinsettings-summary.txt`). A pytest plugin
(`spikes/b21_pin_plugin.py`, loaded with `-p`, production code untouched) pinned FOREVER
every argument of every `DataModel.set_*` call, every `DataModel`, every `SolverSettings`
and every `Solution` for the whole session, and additionally constructed `SolverSettings`
on the persistent worker so no cuOpt native object crossed a thread boundary at all:

```
ARM=pinsettings NTRIALS=3
B21_PIN=1 B21_SETTINGS_ON_WORKER=1
git: 26546ee porcelain=0
run1: EXIT=139 Fatal=1 |
run2: EXIT=139 Fatal=1 |
run3: EXIT=1 Fatal=0 | 46 failed, 8736 passed, 700 skipped, 827 warnings, 28 errors in 731.95s
```

**2 crashes / 3 runs, versus 4 / 6 unpinned** — indistinguishable. The clean run is
baseline-identical (FAILED/ERROR symmetric difference vs `logs/05-baseline-pytest-gpu.txt`
= **0** over 77 ids), so the pinning changes nothing except that it cannot prevent the
crash. Both crashes are a NEW native site, the same one twice, again in the MAIN thread in
pure numpy under the `test_optimization.py` `model` fixture:

```
Current thread 0x00007b021ef43740 (most recent call first):
  File ".../xarray/core/duck_array_ops.py", line 428 in where
  File ".../xarray/core/duck_array_ops.py", line 434 in where_method
  File ".../xarray/core/duck_array_ops.py", line 409 in sum_where
  File ".../xarray/computation/nanops.py", line 101 in nansum
  File ".../xarray/core/duck_array_ops.py", line 551 in f
  File ".../xarray/namedarray/core.py", line 928 in reduce
  ...
  File "/home/azureuser/linopy-gpu/linopy/expressions.py", line 3162 in merge
  File "/home/azureuser/linopy-gpu/linopy/expressions.py", line 2278 in __add__
  File "/home/azureuser/linopy-gpu/test/test_optimization.py", line 102 in model
```

Both fired at `[ 36%]`, i.e. ~7 min in — so the full `--run-gpu` suite is now a ~7-minute
reproducer, not a 12-minute one.

**3. The one linopy-side thread-affinity hole that DOES exist, and is NOT the cause**
(`logs/b21-diag-settings-thread.txt`). B19 routed `DataModel` disposal to the worker but
left `SolverSettings` asymmetric. Measured:

```
   cuOpt._solve runs on 'MainThread'
   SolverSettings() constructed  -> new solver_settings_t on 'MainThread'
   set_solver_setting/prepare_solver_settings on 'linopy-cuopt-solve'
   -> c_solver_settings.reset(new solver_settings_t()) ran here: the CALLER-allocated
      object was DELETED on this thread and a NEW one ALLOCATED on this thread
   [__del__] SolverSettings(A) finalized on thread 'MainThread'
```

i.e. **two cross-thread native new/delete pairs per solve** (`solver_settings.pyx:116` new
on caller, `solver_wrapper.pyx:186` delete+new on worker, caller-thread local delete). The
arm in (2) eliminated both and the crash rate did not move, so this is a latent
inconsistency worth tidying, **not** B21's cause. (Also filed in the return report.)

**4. What could NOT be reproduced outside pytest.** Three standalone reproducers, each many
fresh subprocesses, all clean — so the corrupting condition needs the pytest process:

| reproducer | dose | tooling | result |
|---|---|---|---|
| `b21_diag_repro.py drop 12 2000` | 156 cuOpt solves + numpy/`maybe_replace_signs` churn | `MALLOC_CHECK_=3 MALLOC_PERTURB_=165 LD_PRELOAD=libc_malloc_debug.so.0` | `TOTAL canary corruptions: 0`, `EXIT=0` (`b21-diag-mallocdbg-drop.txt`) |
| `b21_diag_repro2.py drop 10 400`, 3 trials | 140 cuOpt solves + 3 600 non-cuOpt native solves (highs/scip/copt/xpress/cplex/gurobi, file+direct, models closed) + 5 live ThreadPoolExecutor workers | above **plus** `PYTHONMALLOC=debug` | 3/3 `TOTAL canary corruptions: 0`, `OK`, `TRIAL_EXIT=0` (`b21-diag-v2-drop-dbg.txt`) |
| `pytest test/test_cuopt.py test/test_optimization.py --run-gpu` | the real cells | none | `3334 passed, 226 skipped in 479.13s`, `EXIT=0`, `Fatal=0` (`b21-diag-pair-run1.txt`) |

4 000 half-freed 496-byte heap canaries bracketing every phase were never touched, and
neither glibc's `MALLOC_CHECK_=3` nor CPython's `PYTHONMALLOC=debug` guard bytes ever fired
— so whatever the corruption is, it is not a detectable overflow of a glibc chunk or of a
pymalloc block in these doses. H-c gets no support either: 420 cuOpt solves on one
persistent worker in one process produced nothing.

**5. Environmental fact for the upstream report** (`logs/b21-diag-omp-map.txt`).
`import cuopt.linear_programming` brings **two vendored OpenMP runtimes** into the process
at once, neither able to dedupe with the other (both auditwheel-mangled private SONAMEs):

```
    .../libraft_cu12.libs/libgomp-855c301a.so.1.0.0     <- DT_NEEDED of libraft.so   (GNU)
    .../libcuopt_cu12.libs/libomp-8fe85495.so           <- DT_NEEDED of libcuopt.so  (LLVM)
```

The LLVM one ships the full GNU shim (256 `GOMP_*` entry points, `omp_*@OMP_1.0`, version
nodes `GOMP_1.0`..`GOMP_5.0`; 33 dynamic symbols in common with libgomp). Symbol
interposition was checked and is NOT happening — `LD_DEBUG=bindings` shows every call
staying in its own runtime (`libcuopt.so -> libomp-8fe85495.so` for `GOMP_parallel`,
`omp_set_lock`, …; `libgomp-855c301a.so -> itself`) — but two independent OMP runtimes with
two thread pools and two sets of TLS destructors in one process is the same family as
B17/B19/NVIDIA/cuopt#1768 and belongs in the upstream issue.

**Recommended direction (not implemented):** upstream-report-plus-KI. There is no
linopy-side lifetime or thread-affinity change that fixes this — the two candidate ones
were tested and did not move the rate. The only linopy-side items worth doing are (a)
optional tidy-up: keep `SolverSettings`' whole C++ lifecycle on `linopy-cuopt-solve` for
consistency with B19's `DataModel` handling, and (b) a KI entry saying an in-process cuOpt
solve can destabilise the rest of a long-lived CPython process on cuOpt 26.08.00, with the
suite-level consequence recorded honestly rather than papered over.

#### B21 — DIAGNOSIS addendum: the corruption is glibc main-arena large-bin metadata

*B21 diagnosis agent, 2026-08-23.* Two full `--run-gpu` suites were run under
`gdb --batch` (`logs/b21-diag-gdb-run{1,2}.txt`). Run 1 was clean
(`46 failed, 8736 passed, 700 skipped, 827 warnings, 28 errors in 767.16s`, baseline set).
Run 2 crashed and gives the first NATIVE backtrace of B21:

```
Thread 1 "python" received signal SIGSEGV, Segmentation fault.
0x00007ffff7ca45c2 in _int_malloc (av=av@entry=0x7ffff7e1ac80 <main_arena>,
                                   bytes=bytes@entry=1048592) at ./malloc/malloc.c:4133

#0  0x00007ffff7ca45c2 in _int_malloc (av=0x7ffff7e1ac80 <main_arena>, bytes=1048592)
#1  0x00007ffff7ca5139 in __GI___libc_malloc (bytes=1048592) at ./malloc/malloc.c:3329
#2  0x00007ffc9070de3c in _ce2c849dea36cd50f814ed0fad2731f8 () from .../cplex/_internal/libcplex2220.so
...
#10 0x00007ffd1ac3e921 in _wrap_CPXXreadcopyprob () from .../cplex/_internal/py311_cplex2220.so
#11 0x000055555589647f in cfunction_call ()

rip 0x7ffff7ca45c2 <_int_malloc+3090>    rax 0x0    rsi 0x55556c33b988
si_signo = 11, si_code = 1, _sigfault = {si_addr = 0x20}

=> 0x7ffff7ca45c2 <_int_malloc+3090>:   cmp    %rsi,0x20(%rax)
   0x7ffff7ca45c6 <_int_malloc+3094>:   jne    0x7ffff7ca46f2 <_int_malloc+3394>
   0x7ffff7ca45cc <_int_malloc+3100>:   mov    %rdx,0x28(%rsi)
   0x7ffff7ca45d0 <_int_malloc+3104>:   mov    0x28(%rdx),%rax
```

This settles what "corruption" means here. `0x20` and `0x28` are the `fd_nextsize` /
`bk_nextsize` fields of an x86-64 `malloc_chunk`; `_int_malloc` is walking the **main
arena's large-bin doubly-linked skip list** to satisfy a 1 MiB request, follows a pointer
that is `NULL` (`rax = 0`), and dies dereferencing `0x20`. So B21 is **glibc heap free-list
metadata corruption in the main arena**, not a stale Python object pointer and not a numpy
or xarray defect. That is exactly why the crash site is always a different, innocent
library: the first party to walk the broken bin dies, and who that is depends on
allocation history. It also explains why the 4 000 heap canaries and CPython's
`PYTHONMALLOC=debug` guard bytes never fired — the damage is to glibc's own bin pointers,
not to the payload of any block either tool watches.

This is a **fifth** distinct victim: CPLEX `CPXXreadcopyprob` (previously SCIP `optimize()`,
a generic `Solver.close()` decref, numpy `_get_ufunc_and_otypes`, xarray/np `where` under
`LinearExpression.merge`).

Thread landscape at the crash — 76 live threads, of which **7 are cuOpt's LLVM OpenMP team,
still parked** long after `test_cuopt.py` finished:

```
     42 python            (7 of them inside libomp, 6 frames each)
     10 tokio-rt-worker   (polars)
      8 async-executor-
      5 jemalloc_bg_thd   (polars' bundled jemalloc)
      2 cuda-EvtHandlr
      9 polars-*

Thread 187 (... "python"):
#5  __kmp_suspend_64<false, true>(int, kmp_flag_64<false, true>*)   from .../libcuopt_cu12.libs/libomp-8fe85495.so
#6  kmp_flag_64<false, true>::wait(kmp_info*, int, void*)           from .../libomp-8fe85495.so
#7  __kmp_hyper_barrier_release(barrier_type, kmp_info*, ...)       from .../libomp-8fe85495.so
#8  __kmp_fork_barrier(int, int)                                    from .../libomp-8fe85495.so
#9  __kmp_launch_thread ()                                          from .../libomp-8fe85495.so
#10 __kmp_launch_worker(void*)                                      from .../libomp-8fe85495.so
```

Those seven `__kmp_launch_worker` threads are the persistent LLVM-OpenMP team cuOpt's
solves created, and B17's abort was an assertion inside that runtime's own allocator
(`kmp_alloc.cpp(2725)`). A thread-caching allocator inside `libomp` writing through stale
bin pointers into the shared glibc main arena is the mechanism that fits every observation
here, and it is upstream's to fix (NVIDIA/cuopt#1768). Nothing in linopy's translation,
buffer lifetimes or thread routing can prevent it — see the pinning arm above.

#### B21 — DIAGNOSIS addendum 2: `MALLOC_CHECK_=3` never fires; a sixth site, on a worker thread

*B21 diagnosis agent, 2026-08-23.* One more full `--run-gpu` suite under
`gdb --batch` **with** `MALLOC_CHECK_=3 MALLOC_PERTURB_=165
LD_PRELOAD=/lib/x86_64-linux-gnu/libc_malloc_debug.so.0`
(`logs/b21-diag-gdb-mallocchk.txt`). It crashed at `[ 36%]` — same neighbourhood as the
pinning arm's two crashes — and glibc printed **no** diagnostic at all
(`grep -cE "malloc\(\)|free\(\)|corrupted|glibc detected"` → `0`), so this is not a
chunk-header overflow that `MALLOC_CHECK_` can catch at `free()` time.

The crash is the **sixth** distinct site, and the first on a non-main thread — CPython's
cyclic GC, triggered by a `dict` allocation, dereferencing a bad pointer while traversing a
dict:

```
Thread 91 "python" received signal SIGSEGV, Segmentation fault.
0x000055555595e5a4 in visit_decref ()

#0  0x000055555595e5a4 in visit_decref ()
#1  0x0000555555890962 in dict_traverse ()
#2  0x000055555595cf94 in gc_collect_main ()
#3  0x00005555558b212f in PyType_GenericAlloc ()
#4  0x00005555559d3e61 in dict_new ()
#5  0x00005555558ad705 in type_call ()
...
#24 0x0000555555b16d3a in thread_run ()
#25 0x0000555555a5a5c4 in pythread_wrapper ()
#26 0x00007ffff7c94ac3 in start_thread (...) at ./nptl/pthread_create.c:442

si_signo = 11, si_code = 1, _sigfault = {si_addr = 0xaa}
```

So the damage reaches the Python object graph too, not only glibc's bins: a live container
held a pointer that is no longer a valid object. Taken with addendum 1's `_int_malloc`
large-bin fault, the picture is a single process-wide heap corruption with two observable
faces — broken glibc bin pointers and dangling object pointers — and six unrelated
victims across five libraries. Consistent with everything above, and with none of it being
linopy's to fix.

**Crash-site census for B21 (all in one process, all after in-process cuOpt solves):**

| # | site | thread | log |
|---|---|---|---|
| 1 | SCIP `m.optimize()` (`solvers.py:2692`) | main | `gate4-pre-wpf-gpu.txt` |
| 2 | generic `Solver.close():1259` decref of a non-cuOpt `solver_model`, from `__del__` | main | `gate4-pre-wpf-gpu-run3.txt` |
| 3 | numpy `_get_ufunc_and_otypes` listcomp under `apply_ufunc`/`maybe_replace_signs` (x2) | main | `b21-bisect-no-mtl-run{2,3}.txt` |
| 4 | xarray/numpy `where` under `LinearExpression.merge` (x2, pinning arm) | main | `b21-diag-arm-pinsettings-run{1,2}.txt` |
| 5 | glibc `_int_malloc` large-bin walk, from CPLEX `CPXXreadcopyprob` | main | `b21-diag-gdb-run2.txt` |
| 6 | CPython `gc_collect_main` -> `dict_traverse` -> `visit_decref` | worker thread | `b21-diag-gdb-mallocchk.txt` |

**Rates measured in this diagnosis, all at `26546ee`, all full `test/ --run-gpu`:**

```
unpinned baseline (lead's bisect)                     4 crashes / 6 runs
everything pinned + SolverSettings on the worker      2 crashes / 3 runs
under gdb, no allocator flags                         1 crash   / 2 runs
under gdb + MALLOC_CHECK_=3 + MALLOC_PERTURB_=165     1 crash   / 1 run
------------------------------------------------------------------------
combined                                              8 crashes / 12 runs
```

Clean runs are always baseline-identical (`b21-diag-arm-pinsettings-run3.txt`: FAILED/ERROR
symmetric difference vs `05-baseline-pytest-gpu.txt` = 0 over 77 ids;
`b21-diag-gdb-run1.txt`: `46 failed, 8736 passed, 700 skipped, ..., 28 errors`).

## B22 — F8's "concrete types" are documentary only: mypy erases both `cuopt.*` and `cupdlpx.*` annotations to `Any` (recorded, not fixed)

**Status:** open, cosmetic. Found while satisfying acceptance item 7 of WP-FIX5
("confirm the F8 annotations are checked, not `Any`-erased"). They are **not**
checked.

F8 asked for the three `Any` annotations on `cuOpt._build_solver_model`,
`cuOpt._solve` and `cuOpt._set_solver_params` to be replaced with the concrete
cuOpt types, citing `linopy/io.py:33` and the cuPDLPx precedent at
`linopy/solvers.py:4331`/`:4432`. That was done (commit `b0b18f4`) and the names
now read correctly, but mypy gains nothing:

```
$ mypy probe.py   # reveal_type on the three cuOpt members
Revealed type is "def (model: linopy.model.Model) -> Any"
Revealed type is "def (self: ..., dm: Any, ...) -> linopy.constants.Result"
Revealed type is "def (self: ..., settings: Any, log_fn: pathlib.Path | None =)"
$ mypy probe2.py  # the pre-existing cuPDLPx precedent, for comparison
Revealed type is "def (self: ..., cu_model: Any, ...) -> linopy.constants.Result"
```

**Cause:** `cuopt` ships no `py.typed`
(`.venv/.../site-packages/cuopt/py.typed` does not exist) and `pyproject.toml`
sets `ignore_missing_imports = true`, so every symbol under `cuopt.*` — and
under `cupdlpx.*`, which is not installed at all here — resolves to `Any`. The
`TYPE_CHECKING` import at `linopy/solvers.py:177` and `linopy/io.py:33` are in
the same position.

**Consequence:** the annotations are readable documentation and are consistent
with the surrounding cuPDLPx convention, which is what F8 was really asking for;
they are not, and cannot currently be, type-checked. Nothing to fix in linopy —
it would need upstream stubs or a local stub package. Recorded so the review
trail does not carry the stronger claim.
