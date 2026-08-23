# Gate 0 baselines — untouched `master` code (2026-08-22)

Taken on `feat/cuopt-solver` at its branch point from `master` (tip `09c34dd`), with the
final `.venv` state for this run: cuopt-cu12==26.8.0 installed, pandas 3.0.3 (§4.4
pre-authorised downgrade). No production file modified. Raw output in the sibling
`05-baseline-*.txt` files.

## 1. `.venv/bin/pytest -q` → `logs/05-baseline-pytest.txt`

```
46 failed, 8527 passed, 691 skipped, 817 warnings, 28 errors in 464.01s (0:07:44)
```

**All 46 failures and all 28 errors are in `test/remote/`** — every FAILED is in
`test/remote/test_oetc.py` (46), every ERROR in `test/remote/test_oetc_job_polling.py` (28,
plus 6 collection-related `linopy.remote.oetc` lines). Root cause, pasted from the log:

```
ImportError: The 'google-cloud-storage' and 'requests' packages are required
for OetcHandler. Install them with: pip install linopy[oetc]
```

Environmental (the `oetc` extra is not installed in `.venv`), pre-existing, unrelated to
solvers. **Excluded from later comparisons by directory: `test/remote/`.** No failure exists
outside it. Full name list: `grep -E '^(FAILED|ERROR)' logs/05-baseline-pytest.txt`.

## 2. `.venv/bin/pytest --run-gpu -q` → `logs/05-baseline-pytest-gpu.txt`

```
46 failed, 8527 passed, 691 skipped, 817 warnings, 28 errors in 462.91s (0:07:42)
```

Identical to the CPU run, including the skip count. Explanation, verified:
`cupdlpx` is **not installed** in `.venv` (`ModuleNotFoundError`), and
`linopy.available_solvers` = `['gurobi', 'highs', 'scip', 'cplex', 'xpress', 'knitro',
'mosek', 'mindopt', 'copt']`. The only GPU solver's tests skip on unavailability regardless
of `--run-gpu`, so at baseline the flag exercises nothing GPU-specific on this machine.
Consequence for Phase 5: after cuOpt lands, `--run-gpu` runs strictly MORE tests than at
baseline — the comparison is "no new failure outside test/remote/", not "identical counts".

## 3. Docs build (cuOpt-free `.venv-docs`) → `logs/05-baseline-docs.txt`

```
build succeeded, 19 warnings.
```

- All 18 example notebooks executed (nbsphinx `auto`); **none failed** — including the two
  gurobi notebooks (`infeasible-model.ipynb`, `solve-on-oetc.ipynb`), which ran under the
  restricted gurobi license ("Restricted license - for non-production use only - expires
  2027-11-29").
- The 19 warnings (docutils inline-markup issues in autosummary/docstrings, 2 duplicate-object
  warnings for `linopy.piecewise.PiecewiseFormulation`) are pre-existing; full list in the txt.
  Later builds must show **no warning not in this file**.
- Note: first attempt aborted on missing system pandoc; fixed contained via pypandoc_binary in
  `.venv-docs` (D3). The 19-warning result above is the complete post-fix build.
- IPKernelApp stderr noise (TCP-encryption warning, startup-files warning) appears per
  notebook; it is not a sphinx warning and not counted.

## Gate 0 verdict

- `import cuopt` in `.venv`: **OK** (cuopt-cu12 26.8.0).
- ruff 0.15.20 at `/home/azureuser/.local/bin/ruff`: **OK**.
- `git check-ignore dev-scripts/`: **hit** (I3 holds).
- Three baselines captured with pasted output, every red item named by file: **done**.
- Red at baseline (recorded in `45-known-issues.md`): the 74 `test/remote/` items above.
