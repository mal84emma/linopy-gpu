# cuOpt exploration — findings

*Written by: {{agent id}} · Phase 1 · cuOpt version: {{x.y.z}} · GPU: {{name}} · {{YYYY-MM-DD}}*

<!-- HOW TO USE (delete this comment block when filling in):
     Copy to logs/10-exploration.md. FIVE planning agents read this file as their only source
     of empirical truth about cuOpt, so keep the headings EXACTLY — they parse it by heading.
     Three rules:
       1. Every claim carries PASTED output or a `file:line` in the installed package
          (.venv/lib/python3.11/site-packages/cuopt/). An unevidenced claim is an I7 violation.
       2. An unknown is written as "UNKNOWN — <what you tried, why it did not resolve>".
          A guess recorded as a finding is the single worst thing this file can contain:
          five plans will be built on it.
       3. Write "None" rather than deleting a section. Absence must be explicit. -->

## Q0 — Entry-point recommendation

<!-- Charter §4.3 / §6.1. Prototype AT LEAST TWO candidates. RECOMMEND, do not decide —
     Gate 3 fixes the choice. -->

**Recommended:** {{candidate}} {{· secondary: <candidate> for <io_api>, or "none"}}

**Why, in one paragraph:** …

| Candidate | LP | MILP | QP | Duals | Reduced costs | Warm start | Options | Log control | Translation LOC from `model.matrices` | Build time @ ~10⁴ vars | Build time @ ~10⁵ vars | Usable `io_api`s | Stability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `data_model.DataModel` + `solver.Solve` | | | | | | | | | | | | | |
| `problem.Problem` | | | | | | | | | | | | | |
| `io` submodule | | | | | | | | | | | | | |

**Measurement method:** <!-- the scripts under spikes/, how timings were taken, how many runs -->

**Pasted evidence:**

```
{{ timings and prototype output }}
```

**What would change this recommendation:** <!-- name the finding that would flip it -->

## 1. Status mapping

| cuOpt status (value + name) | Proposed `TerminationCondition` | How it was forced | Evidence |
|---|---|---|---|

## 2. Dual and reduced-cost sign convention

<!-- HIGHEST-RISK ITEM (charter §9). Compare against HiGHS live on the same model.
     All six cells, measured — not reasoned about. -->

| Sense | Constraint | cuOpt dual | HiGHS dual | Same sign? | cuOpt RC | HiGHS RC | Same sign? |
|---|---|---|---|---|---|---|---|
| min | `<=` | | | | | | |
| min | `>=` | | | | | | |
| min | `==` | | | | | | |
| max | `<=` | | | | | | |
| max | `>=` | | | | | | |
| max | `==` | | | | | | |

**Transformation linopy must apply:** …

## 3. Objective offset and `set_maximize`

## 4. Log capture

<!-- cuOpt prints from C++. Does `contextlib.redirect_stdout` catch it? If not, the working
     pattern (OS-level dup2 fd redirect?) with pasted proof that output was captured. -->

## 5. Edge cases

| Case | Behaviour | Evidence |
|---|---|---|
| no constraints | | |
| empty / degenerate model | | |
| single variable | | |
| infinite bounds (`np.inf` vs sentinel) | | |
| ranged constraints | | |
| equality via `lb == ub` | | |
| free variables | | |

## 6. Array contracts

<!-- dtypes, CSR index width, C-contiguity, what set_variable_types accepts. State what
     FAILS as well as what works — the guards depend on it. -->

## 7. Solver options

| Parameter (real name) | Type | Default | Effect | Behaviour on unknown name |
|---|---|---|---|---|

## 8. Interruptibility

<!-- Is Solve() KeyboardInterrupt-safe? cf. _run_highs_with_keyboard_interrupt, solvers.py:189 -->

## 9. Repeat solves and teardown

## 10. Unsupported surfaces

| Surface | Genuinely absent? | How confirmed |
|---|---|---|
| semi-continuous | | |
| SOS | | |
| indicator constraints | | |

## 11. Warm start

## 12. QP — how, not whether

<!-- Charter §4.5 commits us to QP. The Q-matrix convention is the trap: full vs triangular,
     and whether the diagonal carries a factor of 2. Determine it EMPIRICALLY — solve a QP
     with a known closed-form optimum and check the objective, both conventions. -->

**Q-matrix format cuOpt expects:** …

**Proof:** <!-- the known-optimum QP, both conventions tried, the objective each returned -->

```
{{ paste }}
```

**Mapping from linopy's `model.objective` quadratic form:** …

**PSD requirement / behaviour on non-PSD Q:** …

**Are duals returned under QP?** …

**Crossover unsupported for QP — confirmed?** …

## 13. Installability from a clean venv

<!-- Charter §4.6. This is the ONLY check that runs outside the hand-built dev venv, so it is
     the only one that can catch "works here, unavailable to users". Paste, do not summarise. -->

**Candidate extra:** `linopy[{{name}}]` — contents: …

```
{{ paste: pip install '/home/azureuser/linopy-gpu[<extra>]' in a FRESH venv,
   no --extra-index-url, no --index-strategy }}
```

```
{{ paste: python -c "import linopy; print(linopy.available_solvers)" — cuopt must appear }}
```

**cuOpt must NOT appear here:**

```
{{ paste: resolution of linopy[solvers] — confirm no cuopt/cudf/libcuopt }}
```

**macOS / Windows resolution unaffected:**

```
{{ paste: uv pip compile --python-platform windows / macos with the marker set }}
```

**Verdict:** one-command install works / **FAILS — <why>**

## Unknowns

<!-- Everything that stayed unresolved, each with what was tried. "None" if none. -->

## Surprises

<!-- Anything true that the charter's §4 did not anticipate. If it CONTRADICTS §4, that is an
     E2 escalation, not a quiet correction — say so here and raise it. "None" if none. -->
