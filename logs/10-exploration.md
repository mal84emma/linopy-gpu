# cuOpt exploration — findings

*Written by: spike agents A and B (both Opus 5, `claude-opus-5[1m]`) · Phase 1 · cuOpt version: 26.08.00 (banner reports `26.8.0, git hash 400863c1`) · GPU: Tesla T4 (15.57 GiB, CUDA 12.9 runtime on 535.274.02) · 2026-08-22*

> Provenance: agent A wrote Q0 and items 1, 2, 3, 5, 6, 12; agent B wrote items 4, 7, 8, 9,
> 10, 11, 13; Unknowns and Surprises carry both scopes. Mechanically merged from
> `spikes/exploration-A.md` + `spikes/exploration-B.md` by the lead (assembly only, D5) —
> no content was altered.
>
> Scripts: `/home/azureuser/linopy-gpu/dev-scripts/cuopt/spikes/a_*.py` and `b_*`; captured
> output in the matching `*_out.txt` / `*.txt` files in the same directory.

---

## Q0 — Entry-point recommendation
**Recommended:** `data_model.DataModel` + `solver.Solve` · secondary: the `io` submodule (`cuopt.linear_programming.io.Read`) for the `lp` and `mps` file `io_api`s

**Why, in one paragraph:** `DataModel` is the only candidate that is simultaneously fast, complete
and expressive. It ingests `model.matrices` almost verbatim (13–16 lines of translation), builds a
10⁵-variable model in **0.012 s** versus **0.86 s** for `problem.Problem` (**~70× slower**) and
**0.83–0.95 s** for the file path, and it is the only candidate that can express **ranged rows**
(`problem.Problem` structurally cannot — `Problem._to_data_model` uses `set_row_types` plus a
*single* `set_constraint_bounds` vector, `problem.py:1593-1594`, and `Problem._from_data_model`
raises `Exception("Couldn't initialize constraints")` on a row whose bound is neither its lower nor
its upper, `problem.py:1524`). Everything that looked like a `Problem`-only advantage in charter
§4.3 turns out to live on `SolverSettings`, which all three candidates share: `set_mip_callback`
(`solver_settings.pyx:256`), `set_pdlp_warm_start_data` (`:229`), `set_parameter` (`:178`) and log
control (`log_to_console`, `log_file`); `Problem.getWarmstartData` (`problem.py:1907`) is a thin
wrapper over `Solution.get_pdlp_warm_start_data`, which the `DataModel` path already returns, and
`Problem.Slack/DualValue/ReducedCost` are just `populate_solution` copying the same
`Solution` arrays onto Python objects (`problem.py:2164-2201`). `Problem`'s only genuine
differentiator is `relax()` / `readMPS` / `writeMPS`, none of which linopy needs. Meanwhile the
`io` path is a real, cheap capability win that cuPDLPx cannot offer at all: cuOpt's reader
round-trips linopy's own `.lp` and `.mps` output with **full fidelity** — variable and row names,
column order, bounds, integrality, objective sense and the quadratic objective all survive — so
declaring `accepted_io_apis ⊇ {"lp", "mps"}` is achievable in ~3 lines. None of the three is marked
experimental; `internals` (the callback classes) is the only private-looking module and is not a
candidate.

| Candidate | LP | MILP | QP | Duals | Reduced costs | Warm start | Options | Log control | Translation LOC from `model.matrices` | Build time @ ~10⁴ vars | Build time @ ~10⁵ vars | Usable `io_api`s | Stability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `data_model.DataModel` + `solver.Solve` | yes (obj `-2.8`, dual `[-0.4,-0.2]`) | yes (obj `-2.0`, `cat=MIP`) | **yes** (obj `-3.0` = HiGHS) | yes | yes (but see §2 — unreliable for `max`) | yes — `SolverSettings.set_pdlp_warm_start_data` + `DataModel.set_initial_primal/dual_solution` | yes — `SolverSettings.set_parameter`, 117 names | yes — `log_to_console`, `log_file` | **13** (LP+MILP) / **16** (with QP); + 4 for the §2 `max` mitigation | **0.002 s** | **0.012–0.014 s** | `direct` | public, documented, no experimental marker |
| `problem.Problem` | yes (obj `-2.8`, dual `[-0.4,-0.2]`) | yes (obj `-2.0`) | yes (obj `-3.0`, `ObjValue -3.0`) but needs `QuadraticExpression` built from `Variable*Variable` products, not a matrix | yes (`Constraint.DualValue`) | yes (`Variable.ReducedCost`) | same `SolverSettings` API + `getWarmstartData()` wrapper | same `SolverSettings` | same | **21** and **cannot express ranged rows at all**; +~10 more for QP | 0.069–0.146 s (**~35–70×** A) | **0.858–1.042 s** (**~70×** A) | `direct` | public; `getQcsr()` already `DeprecationWarning`-ed (`problem.py:2140`) |
| `io` submodule (`io.Read` on a linopy-written file) | yes (obj `-2.8`, dual `[-0.4,-0.2]`) | yes (`cat=MIP`, 250/250 integers preserved) | yes (obj `-3.0`; `.lp` gives triangular Q `[1,1,1]`, `.mps` gives symmetric `[1,0.5,0.5,1]` — both correct) | yes | yes | same `SolverSettings` | same `SolverSettings` | same | **3** (`to_file` + `Read`), reverse mapping already exists in linopy (`_solution_from_names`, `solvers.py:105`) | 0.063 s (mps) / 0.117 s (lp) | 0.831 s (lp) / 0.951 s (mps) | **`lp`, `mps`** — plus `direct` is unaffected | public; `.mps` writing needs `highspy` (`linopy/io.py`: `from highspy.highs import Highs`, "Package highspy not installed. This is required to exporting to MPS file.") |

**Measurement method:** `/home/azureuser/linopy-gpu/dev-scripts/cuopt/spikes/a_q0_prototypes.py`.
One linopy model per size (`N` variables, `N/2` constraints, 3 nonzeros per row, `nnz = 3N/2`),
built once; then each candidate's translation timed with `time.perf_counter()`, **2 repetitions**
each, after `model.matrices` had already been materialised (materialisation timed separately so it
is not charged to any candidate). Correctness compared against `model.solve("highs")` live in the
same process. Fidelity of the file path measured separately in `a_q0_io_fidelity.py`.
Build-time numbers are wall clock on an otherwise-shared T4 (a sibling agent held the GPU part of
the time); the *build* step is pure CPU/host work, so GPU contention does not affect it.

**Pasted evidence:**

```
SCALE — build time at N = 10000 variables (5000 constraints)
  linopy model construction        :    0.514 s (nvars=10000, ncons=5000)
  model.matrices materialisation   :    0.002 s (A shape (5000, 10000), nnz 15000)
  [rep0] A DataModel build       :    0.002 s
  [rep1] A DataModel build       :    0.002 s
  [rep0] B Problem addVar/addCon :    0.131 s   + _to_data_model    0.015 s   =    0.146 s
  [rep1] B Problem addVar/addCon :    0.055 s   + _to_data_model    0.014 s   =    0.069 s
  C io lp : linopy write    0.060 s (0.7 MB)  + cuopt Read    0.057 s  =    0.117 s
  C io mps: linopy write    0.039 s (1.3 MB)  + cuopt Read    0.024 s  =    0.063 s
     A obj = -89982.85816112 (Optimal)

SCALE — build time at N = 100000 variables (50000 constraints)
  linopy model construction        :    0.382 s (nvars=100000, ncons=50000)
  model.matrices materialisation   :    0.013 s (A shape (50000, 100000), nnz 150000)
  [rep0] A DataModel build       :    0.014 s
  [rep1] A DataModel build       :    0.012 s
  [rep0] B Problem addVar/addCon :    0.782 s   + _to_data_model    0.260 s   =    1.042 s
  [rep1] B Problem addVar/addCon :    0.707 s   + _to_data_model    0.151 s   =    0.858 s
  C io lp : linopy write    0.231 s (7.7 MB)  + cuopt Read    0.600 s  =    0.831 s
  C io mps: linopy write    0.567 s (12.8 MB)  + cuopt Read    0.384 s  =    0.951 s
     A obj = -899328.10789928 (Optimal)
```

Same model, all candidates, correctness (`a_q0_corr.txt`):

```
CORRECTNESS — same small linopy model through all three candidates
HiGHS via linopy : obj = -2.7999999999999994  x = [1.6 1.2]  duals = {'c0': -0.4, 'c1': -0.19999999999999996}
A DataModel      : status=Optimal obj=-2.80000000 x=[1.6 1.2] dual=[-0.40000007 -0.20000008]
B Problem        : status=Optimal obj=-2.80000000 x=[1.6 1.2] dual=[-0.40000007 -0.20000008]
C io.Read(lp )   : status=Optimal obj=-2.80000000 x=[1.6 1.2] dual=[-0.40000007 -0.20000008]
      var names from file : ['x0' 'x1']
      row names from file : ['c0' 'c1']
      sense from file     : maximize=False
      row types           : ['L' 'L']
C io.Read(mps)   : status=Optimal obj=-2.80000000 x=[1.6 1.2] dual=[-0.40000007 -0.20000008]

--- MILP through each candidate ---
A DataModel MILP : Optimal obj=-2.000000 cat=MIP
B Problem MILP   : Optimal obj=-2.000000 cat=MIP
C io.Read(lp ) MILP: Optimal obj=-2.000000 cat=MIP vtypes=[b'I' b'I']
C io.Read(mps) MILP: Optimal obj=-2.000000 cat=MIP vtypes=[b'I' b'I']

--- QP through each candidate ---
HiGHS via linopy QP: obj = -2.9999999999999964 x = [0.99999997 0.99999997]
A DataModel QP   : Optimal obj=-3.00000000 x=[1. 1.]
B Problem QP     : build_B translation drops Q entirely (LinearExpression only); Problem needs QuadraticExpression objects built from Variable*Variable products
C io.Read(lp ) QP : Optimal obj=-3.00000000 x=[1. 1.]  Qvals=[1. 1. 1.]
C io.Read(mps) QP : Optimal obj=-3.00000000 x=[1. 1.]  Qvals=[1.  0.5 0.5 1. ]
```

`problem.Problem` does solve QP when the objective is expressed with its own operators
(`a_12_out.txt`, TEST 9):

```
Problem QP: status=Optimal obj=-3.00000000 x=(0.9999999996735056, 0.9999999996735003)  [target obj -3, x=(1,1)]
  Problem ObjValue    : -3.0
  Problem constr dual : 7.236946777227376e-11
  Problem var RC      : -1.0518583022699716e-09 -1.0518583022699676e-09
```

File-path fidelity at 750 variables / 250 rows / 250 integers / `max` sense
(`a_q0_io.txt` plus the dtype-corrected recheck; the `n integer read : 0` line in
`a_q0_io.txt` is a bug in *my* comparison — the file reader returns `<U1`, not `|S1`):

```
linopy n_integer = 250 vtypes dtype <U1 uniq ['C' 'I']
lp:  vt dtype=<U1 uniq=['C' 'I'] n_I=250
   Solving a problem with 250 constraints, 750 variables (250 integers), and 750 nonzeros
   cat=MIP status=Optimal obj=3770.188931383...
mps: vt dtype=<U1 uniq=['C' 'I'] n_I=250
   Solving a problem with 250 constraints, 750 variables (250 integers), and 750 nonzeros
   cat=MIP status=Optimal obj=3770.188931383...
linopy/HiGHS objective : +3770.18893138

### does cuOpt's reader preserve the FILE's column order?
  linopy variable labels : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  file-read var names    : ['x0', 'x1', ..., 'x11']
  file-read c            : [1.0, 2.0, ..., 12.0]
  linopy M.c             : [1.0, 2.0, ..., 12.0]
  c arrays identical     : True
```

**What would change this recommendation:**

1. If the `max`-sense dual-sign defect of §2 turned out to be **absent** on the `Problem` path or
   the `io` path while present on `DataModel`. It is not: the defect is in cuOpt's presolve
   postsolve and `Problem._to_data_model` funnels into the very same `DataModel`
   (`problem.py:1526-1630`), so all three inherit it.
2. If linopy ever needs per-variable incremental model updates (`supports_persistent_update`).
   `Problem.updateConstraint` / `updateObjective` exist and `DataModel` has no equivalent — but
   `Problem.reset_solved_values()` throws the whole `DataModel` away and rebuilds
   (`problem.py:1650-1651`), so even there the advantage is nominal.
3. If ranged constraints were declared out of scope *and* a 70× build-time penalty were acceptable,
   `Problem` would become defensible on readability grounds alone. Neither is true.

## 1. Status mapping
Enumerated from
`.venv/lib/python3.11/site-packages/cuopt/linear_programming/solver/solver_wrapper.pyx:79-111`
(`MILPTerminationStatus`, `LPTerminationStatus`, `ErrorStatus`, `ProblemCategory`) and confirmed at
runtime by `a_00_enumerate.py`. `Solution._set_termination_status`
(`solution/solution.py:224-237`) picks the LP or the MILP enum from `problem_category`, so **the
same integer means different things for LP and MILP** — value `2` is `PrimalInfeasible` (LP) or
`Infeasible` (MILP), value `3` is `DualInfeasible` (LP) or `Unbounded` (MILP), value `8` exists
only for MILP. A `CONDITION_MAP` keyed on the raw int alone would be wrong; key it on
`(problem_category, status)` or on the enum member itself.

| cuOpt status (value + name) | Proposed `TerminationCondition` | How it was forced | Evidence |
|---|---|---|---|
| LP `0` `NoTermination` | `internal_solver_error` (and surface `get_error_message()`) | quadratic objective + integer variables (MIQP); also any wrong-length input array | `a_01d_out.txt`: `MIQP  status= 0 NoTermination  err=4 obj=+0 msg='{"CUOPT_ERROR_TYPE": "ValidationError", "msg": "Presolve does not support optimization_problem with a quadratic objective"}'`; `a_06_out.txt`: `c too short (1 for 2 vars) -> NoTermination obj=nan` |
| LP `1` `Optimal` | `optimal` | `min -x-y s.t. x+2y<=4, 3x+y<=6` | `a_01_out.txt`: `LP optimal  status= 1 Optimal  cat=LP err=0 obj=-2.8` |
| LP `2` `PrimalInfeasible` | `infeasible` | `x>=2` and `x<=1` | `a_01_out.txt`: `LP primal infeasible (x>=2 & x<=1)  status= 2 PrimalInfeasible` |
| LP `3` `DualInfeasible` | `unbounded` (mapping proposed, status **not observed** — see Unknowns) | not reachable: every unbounded LP tried returned `11` instead | `a_01_out.txt`, `a_01c_out.txt` — 7 unbounded variants, all `status=11` |
| LP `4` `IterationLimit` | `iteration_limit` | `iteration_limit=1` on a 400×300 dense LP | `a_01_out.txt`: `LP iteration_limit=1 (method=PDLP)  status= 4 IterationLimit  obj=-37.6926`; same for `Concurrent` |
| LP `5` `TimeLimit` | `time_limit` | `time_limit=1e-6` on the same LP | `a_01_out.txt`: `LP time_limit=1e-6 (method=PDLP)  status= 5 TimeLimit  obj=+0` |
| LP `6` `NumericalError` | `internal_solver_error` | non-PSD `Q` (`Q=[[-1]]`, or `Q=diag(1,-1)`) | `a_01d_out.txt`: `min, Q=[[-1]] (concave)  status= 6 NumericalError  obj=+nan`; `min, Q=diag(1,-1) (indefinite)  status= 6 NumericalError` |
| LP `7` `PrimalFeasible` | `suboptimal` | `first_primal_feasible=True`, `method=PDLP` | `a_01_out.txt`: `LP first_primal_feasible=True (method=PDLP)  status= 7 PrimalFeasible  obj=-3.54817` |
| LP `11` `UnboundedOrInfeasible` | `infeasible_or_unbounded` | `min -x, x∈[0,∞)`; survives `infeasibility_detection=False` and `strict_infeasibility=True` | `a_01_out.txt`: `LP unbounded (min -x, x in [0,inf))  status=11 UnboundedOrInfeasible`; `a_01c_out.txt`: `unbounded, strict_infeasibility=True  status=11` |
| MILP `0` `NoTermination` | `internal_solver_error` | MIQP (as above) | `a_01d_out.txt` (same line — `cat=MIP` in `a_12_out.txt` TEST 8) |
| MILP `1` `Optimal` | `optimal` | same LP with `set_variable_types(['I','I'])` | `a_01_out.txt`: `MILP optimal  status= 1 Optimal  cat=MIP err=0 obj=-2` |
| MILP `2` `Infeasible` | `infeasible` | `2x == 1`, `x` integer | `a_01_out.txt`: `MILP infeasible (2x=1, x integer)  status= 2 Infeasible  cat=MIP` |
| MILP `3` `Unbounded` | `unbounded` (mapping proposed, status **not observed** — see Unknowns) | not reachable: unbounded MILPs returned `11` | `a_01_out.txt`: `MILP unbounded (min -x, x integer, no ub)  status=11`; `a_01c_out.txt`: `MILP unbounded, 2 vars, one bounded  status=11` |
| MILP `5` `TimeLimit` | `time_limit` | 120-variable subset-sum with a fractional rhs, `time_limit=1.0`; and a 6×50 market-split, `time_limit=2.0` | `a_01c_out.txt`: `MILP subset-sum eq, time_limit=1.0  status= 5 TimeLimit  cat=MIP obj=+0`; `MILP market-split 6x50 fractional rhs, time_limit=2.0  status= 5 TimeLimit` |
| MILP `8` `FeasibleFound` | `suboptimal` | 60-item knapsack with `time_limit=0.05`, `time_limit=1e-6`, or `node_limit=1` | `a_01_out.txt`: `MILP time_limit=0.05 (knapsack n=60)  status= 8 FeasibleFound  obj=-1939`; `MILP node_limit=1 ... status= 8 FeasibleFound obj=-1962` |
| MILP `11` `UnboundedOrInfeasible` | `infeasible_or_unbounded` | unbounded MILP; also unbounded-and-infeasible together | `a_01_out.txt`: `MILP unbounded+infeasible  status=11 UnboundedOrInfeasible  cat=MIP` |
| `ErrorStatus` `0` `Success` / `4` `ValidationError` / `5` `OutOfMemoryError` / `6` `RuntimeError` | not a `TerminationCondition` — read alongside it | `get_error_status()` is `4` with a JSON `get_error_message()` on the MIQP case | `a_01d_out.txt` MIQP line above; enum values from `a_00_enumerate.py` |

Notes the implementer must not miss:

- `time_limit`/`iteration_limit` **can still return `Optimal`.** `MILP iteration_limit=1 (knapsack n=60)`
  returned `status= 1 Optimal obj=-2020` (`a_01_out.txt`) — `iteration_limit` is not a MILP node
  budget. Do not infer a limit hit from the settings.
- `TimeLimit` on the LP path returned `obj=+0` with a zero primal vector, not a partial solution.
  `safe_get_solution` must therefore not treat `TimeLimit` as "solution available" without checking.
- `NoTermination` returns an **empty** `primal_solution` array. See §5 and §6.

## 2. Dual and reduced-cost sign convention
Script: `a_02_signs.py` (six cells), `a_02b_signs_precise.py` (full precision, 3 repetitions,
`solved_by`), `a_02c_rc_probe.py` (reduced-cost sweep), `a_02d_equality_max.py`,
`a_02e_presolve_dual_sign.py` (144-row matrix), `a_02f_linopy_reach.py` (via a real linopy model),
`a_02g_dual_postsolve.py` (which setting fixes it), `a_02h_mitigation.py` / `a_02i_m2_full.py`
(mitigation validation). HiGHS is called **live, in the same process, on the identical model** in
every one; linopy's own HiGHS path passes `solution.row_dual` straight through with no sign
adjustment (`solvers.py:1871-1879` — no negation for `max`, unlike cuPDLPx at `solvers.py:4385-4386`),
so HiGHS's convention **is** linopy's convention.

Model: `A = [[1,2,1],[3,1,1]]`, `b = [4,6]`, `0 <= x <= 10`, objective coefficients per cell chosen
so the primal optimum is always `x = (1.6, 1.2, 0)` with both rows binding, `x2` at its lower bound
with a non-zero reduced cost, and a unique non-degenerate dual.

| Sense | Constraint | cuOpt dual | HiGHS dual | Same sign? | cuOpt RC | HiGHS RC | Same sign? |
|---|---|---|---|---|---|---|---|
| min | `<=` | `[-0.40000007, -0.20000008]` | `[-0.4, -0.2]` | **yes — identical** | `[-3.16e-07, -2.24e-07, +1.59999985]` | `[0, 0, +1.6]` | yes |
| min | `>=` | `[+0.40000007, +0.20000008]` | `[+0.4, +0.2]` | **yes — identical** | `[+3.16e-07, +2.24e-07, +0.40000015]` | `[0, 0, +0.4]` | yes |
| min | `==` | `[+0.40000007, +0.20000008]` | `[+0.4, +0.2]` | **yes — identical** | `[+3.16e-07, +2.24e-07, +0.40000015]` | `[0, 0, +0.4]` | yes |
| max | `<=` | `[+0.40000007, +0.20000008]` | `[+0.4, +0.2]` | **yes — identical** | `[-3.16e-07, -2.24e-07, **+0.39999985**]` | `[0, 0, **-1.6**]` | **NO — wrong value, off by `2·c_j`** |
| max | `>=` | `[-0.40000007, -0.20000008]` | `[-0.4, -0.2]` | **yes — identical** | `[+3.16e-07, +2.24e-07, -0.39999985]` | `[0, 0, -0.4]` | yes |
| max | `==` | `[-0.40000007, -0.20000008]` | `[-0.4, -0.2]` | **yes — identical** | `[+3.16e-07, +2.24e-07, -0.39999985]` | `[0, 0, -0.4]` | yes |

On this model the duals need **no transformation at all** and agree to `max|Δ| = 8.18e-08`
(default `Concurrent` method). But the six-cell table is **not sufficient**, and this is the single
most important finding in this file:

### 2a. `max` + `solved_by == Unset` → duals come back NEGATED (silently, status `Optimal`)

Widening the sweep to a second model shape exposed a real defect. Whenever cuOpt finishes a model
**inside presolve** — reported as `solved_by == SolverMethod.Unset`, log line
`Optimal solution found during presolve.` — **and** `set_maximize(True)` was used, the duals are
returned with the **wrong sign**. `a_02e_presolve_dual_sign.py` ran 144 cells
(2 model shapes × {`<=`,`>=`,`==`} × {min,max} × presolve∈{unset,0,1} × 4 methods) against HiGHS.
The correlation is perfect:

```
CORRELATION TABLE — verdict vs solved_by
solved_by     sense  verdict     count
Barrier       max    OK          17
Barrier       min    OK          17
DualSimplex   max    OK          35
DualSimplex   min    OK          35
PDLP          max    DIFFERENT   17
PDLP          min    DIFFERENT   17
Unset         max    NEGATED     3
Unset         min    OK          3
```

(`PDLP  DIFFERENT` is tolerance only, not sign — see §2c.) The three `NEGATED` cells:

```
2var  ==   max   None      Concurrent  Unset        [0.4 0.2]   HiGHS [-0.4 -0.2]   NEGATED
2var  ==   max   None      PDLP        Unset        [0.4 0.2]   HiGHS [-0.4 -0.2]   NEGATED
2var  ==   max   None      Barrier     Unset        [0.4 0.2]   HiGHS [-0.4 -0.2]   NEGATED
```

**It is reachable from an ordinary linopy model through the natural translation.**
`a_02f_linopy_reach.py` builds real `linopy.Model()`s, solves each with `model.solve("highs")` and
with `model.matrices → DataModel → solver.Solve` in the same process:

```
[1] min, 2 vars (square equality system), default settings
    linopy/HiGHS : obj=+2.80000000 dual=[0.4 0.2]
    linopy/cuOpt : obj=+2.80000000 dual=[0.4 0.2] solved_by=Unset status=Optimal
    -> OK
[2] max, 2 vars (square equality system), default settings
    linopy/HiGHS : obj=+2.80000000 dual=[0.4 0.2]
    linopy/cuOpt : obj=+2.80000000 dual=[-0.4 -0.2] solved_by=Unset status=Optimal
    -> *** NEGATED — WRONG SIGN ***
[6] max, 3 vars (extra free-ish var), default settings
    linopy/HiGHS : obj=+12.80000000 dual=[0.4 0.2]
    linopy/cuOpt : obj=+12.80000000 dual=[-0.4 -0.2] solved_by=Unset status=Optimal
    -> *** NEGATED — WRONG SIGN ***

n=4     sense=min  solved_by=Unset  cuOptobj=+13.754309 HiGHSobj=+13.754309  dual verdict=OK       max|d|=5.551e-17
n=4     sense=max  solved_by=Unset  cuOptobj=+26.384320 HiGHSobj=+26.384320  dual verdict=NEGATED  max|d|=3.610e+00
n=10    sense=max  solved_by=Unset  cuOptobj=+39.842325 HiGHSobj=+39.842325  dual verdict=NEGATED  max|d|=3.610e+00
n=40    sense=max  solved_by=Unset  cuOptobj=+84.740420 HiGHSobj=+84.740420  dual verdict=NEGATED  max|d|=3.610e+00
n=200   sense=max  solved_by=Unset  cuOptobj=+240.674144 HiGHSobj=+240.674144  dual verdict=NEGATED  max|d|=3.610e+00
```

The **objective is always right**; only the duals flip. `max|Δdual| = 3.61` — this is not a
tolerance question.

**It is not limited to equality rows.** `a_05b_ranged_signs.py` shows the same defect on **ranged**
rows, where the `presolve` workaround does *not* help:

```
### settings: presolve=1
ranged 1<=x+y<=3, max x+y  (UPPER binds)
    cuOpt solved_by=Unset        obj=+3.000000 x=[3. 0.] dual=[-1.]
    HiGHS                       obj=+3.000000 x=[3. 0.] dual=[1.]   -> *** NEGATED ***
ranged 1<=x+y<=3, max -x-y (LOWER binds)
    cuOpt solved_by=Unset        obj=-1.000000 x=[1. 0.] dual=[1.]
    HiGHS                       obj=-1.000000 x=[1. 0.] dual=[-1.]   -> *** NEGATED ***
2 ranged rows, max
    cuOpt solved_by=Unset        obj=+3.000000 x=[ 4. -1.] dual=[-1. -1.]
    HiGHS                       obj=+3.000000 x=[ 4. -1.] dual=[1. 1.]   -> *** NEGATED ***
```

### 2b. Which settings fix it (`a_02g_dual_postsolve.py`, `a_02i_m2_full.py`)

```
ALL SIX CELLS CORRECT?  (rtol=1e-5, atol=1e-7)   [2-var square model]
  default                NO
  presolve=0             YES
  presolve=1             YES
  dual_postsolve=0       NO
  dual_postsolve=1       NO
  method=DualSimplex     YES
  method=Barrier         NO
  crossover=1            NO
```

`presolve=0/1` and `method=DualSimplex` work on that model only because they route the solve away
from the presolve-only path; on the **ranged** cases above `presolve=1` still lands on `Unset` and
still returns the wrong sign. So `presolve` is **not** a general fix, `dual_postsolve` is
irrelevant, and forcing `method=DualSimplex` would throw away the GPU.

Two mitigations were validated over all 21 cases that exercise the defect (2-var/3-var, `<=`/`>=`/`==`,
ranged ×3, QP ×4), each against HiGHS live:

- **M1 — set `presolve` explicitly.** Fixes the equality cells, **fails** on ranged. Rejected.
- **M2 — never call `set_maximize(True)`.** For a `max` model hand cuOpt the equivalent
  minimisation (`c → -c`, `Q → -Q`, `set_maximize(False)`), then negate the returned objective and
  the returned duals on the linopy side. This keeps linopy off the buggy postsolve branch **by
  construction**.

```
### mitigation M2 disabled (baseline)
  FAILING CASES: 5 / 21
    2var == max        Unset  cuOpt [0.4 0.2]     HiGHS [-0.4 -0.2]   NEGATED
    2var == max (c=+1) Unset  cuOpt [-0.4 -0.2]   HiGHS [0.4 0.2]     NEGATED
    ranged max x+y     Unset  cuOpt [-1.]         HiGHS [1.]          NEGATED
    ranged max -x-y    Unset  cuOpt [1.]          HiGHS [-1.]         NEGATED
    2 ranged rows max  Unset  cuOpt [-1. -1.]     HiGHS [1. 1.]       NEGATED

### mitigation M2 ENABLED
  FAILING CASES: 0 / 21
```

### 2c. Tolerance, per method — matters for the frozen `rtol`

`a_02c_rc_probe.py`, `max|cuOpt dual − HiGHS dual|`:

```
DUAL agreement across the same sweeps (max|cuOpt-HiGHS| per method):
  max/<=: Concurrent=8.18e-08  PDLP=2.60e-06  DualSimplex=1.97e-07  Barrier=1.73e-09
  max/>=: Concurrent=8.18e-08  PDLP=4.71e-05  DualSimplex=8.18e-08  Barrier=2.47e-09
  min/<=: Concurrent=8.18e-08  PDLP=2.60e-06  DualSimplex=1.97e-07  Barrier=1.73e-09
  min/>=: Concurrent=8.18e-08  PDLP=4.71e-05  DualSimplex=8.18e-08  Barrier=2.47e-09
```

The default (`Concurrent`) method holds duals to **~8.2e-08 absolute**. `method=PDLP` alone reaches
only **4.7e-05** and its primal objective drifts too (`2.80013079` vs `2.8`). A dual tolerance
tighter than ~1e-4 is therefore **not** safe if PDLP is ever forced; at the default it is.

### 2d. Reduced costs are separately unreliable — and linopy does not use them

`a_02c_rc_probe.py` sweeps `c_2` on `max c·x s.t. A x <= b` (`y = [0.4, 0.2]`), comparing the
correct `rc = c − Aᵀy` against the "negated-`c`" value `−c − Aᵀy`:

```
MAX, rows <= , sweeping c_2 (c_0=c_1=1 fixed)
    c2 |    HiGHS rc2 |   correct  negated-c |     Concurrent           PDLP    DualSimplex        Barrier
  -1.0 |  -1.60000000 |   -1.6000    +0.4000 |    +0.39999985    +0.40000008    -1.60000003    +0.40000000
  -3.0 |  -3.60000000 |   -3.6000    +2.4000 |    +2.39999985    +2.40000008    -3.60000003    +2.40000000
 -0.25 |  -0.85000000 |   -0.8500    -0.3500 |    -0.35000015    -0.34999992    -0.85000003    -0.35000000

MIN, rows <= , same sweep on c_2 (sanity: min should always agree)
   1.0 |  +1.60000000 |    +1.60000015    +1.59999992    +1.60000003    +1.60000000
   3.0 |  +3.60000000 |    +3.60000015    +3.59999992    +3.60000003    +3.60000000

MAX, rows >= (c=[-1,-1,c2]) — the cell that DID agree; sweep to confirm
  -1.0 |  -0.40000000 |   -0.4000    +1.6000 |    -0.39999985    -0.00000000    -0.39999985    -0.40000000
```

So for a **maximised** model with `<=` rows, the default `Concurrent` path (and `PDLP`, and
`Barrier`) return `−c − Aᵀy` instead of `c − Aᵀy`, while a forced `DualSimplex` returns the correct
value; and `PDLP` returns a flat `0` for `max`/`>=`. I could not reduce this to one transformation
and I am **not** proposing one.

**This does not block the integration:** `grep -rn "reduced_cost\|ReducedCost\|reduced cost" linopy/`
returns **no hits**, and `linopy.constants.Solution` (`constants.py:274-286`) carries only
`primal`, `dual`, `objective`. linopy has no reduced-cost surface. Record the defect; do not expose
`get_reduced_cost()`.

**Transformation linopy must apply:**

1. **Duals: none — pass `Solution.get_dual_solution()` through unchanged**, *provided* the model was
   handed to cuOpt in **minimisation form**. Concretely: do **not** copy cuPDLPx's
   `if ModelSense == MAXIMIZE: dual = -dual` (`solvers.py:4385-4386`) — cuOpt's convention already
   matches HiGHS, which is linopy's.
2. **`max` handling: apply M2.** Never call `set_maximize(True)`. Build with `-M.c` (and `-Q`),
   leave `set_maximize(False)`, then negate `get_primal_objective()` and `get_dual_solution()`.
   This is both the sign fix *and* the objective fix, and it is verified 21/21 against HiGHS. Note
   `maybe_adjust_objective_sign` (`solvers.py:389-401`) is a no-op for `io_api="direct"`, so the
   objective negation must be done explicitly in `_solve`.
3. **Reduced costs: do not surface them.** cuOpt's `get_reduced_cost()` is wrong for maximised
   models with `<=` rows on the default path (§2d); linopy has nowhere to put them anyway.
4. **`get_dual_solution()` raises for MILP** (`AttributeError: Attribute get_dual_solution is not
   supported for milp solution`, `solution/solution.py:262`). Guard on
   `get_problem_category()`/`ProblemCategory.LP` before reading duals, `dual_objective`,
   `reduced_cost` or `lp_stats`.

## 3. Objective offset and `set_maximize`
Script `a_03_offset_maximize.py`, output `a_03_out.txt`. Base model
`min -x-y s.t. x+2y<=4, 3x+y<=6, 0<=x,y<=10` → `obj -2.8` at `x=(1.6,1.2)`.

**`set_objective_offset(k)` is a plain additive constant, and it IS included in the reported
objective** (both primal and dual), exactly as `data_model.py:55-58` claims:

```
min, no offset                       obj=-2.80000000  dual_obj=-2.80000000  dual=[-0.40000007 -0.20000008] rc=[3.16e-07 2.24e-07]
min, set_objective_offset(10.0)      obj=+7.20000000  dual_obj=+7.20000000  dual=[-0.40000007 -0.20000008] rc=[3.16e-07 2.24e-07]
min, set_objective_offset(-10.0)     obj=-12.80000000 dual_obj=-12.80000000 dual=[-0.40000007 -0.20000008] rc=[3.16e-07 2.24e-07]
min, set_objective_offset(0.5)       obj=-2.30000000  dual_obj=-2.30000000  dual=[-0.40000007 -0.20000008] rc=[3.16e-07 2.24e-07]
max, no offset                       obj=+2.80000000  dual_obj=+2.80000000  dual=[0.40000007 0.20000008]
max, set_objective_offset(10.0)      obj=+12.80000000 dual_obj=+12.80000000 dual=[0.40000007 0.20000008]
max, set_objective_offset(-10.0)     obj=-7.20000000  dual_obj=-7.20000000  dual=[0.40000007 0.20000008]
```

- **Duals and reduced costs are unaffected by the offset** — identical to 8 significant figures in
  every row above. Correct, and what linopy needs.
- The offset also works for MILP and QP:
  `MILP, no offset obj=-2.00000000` → `MILP, offset(10) obj=+8.00000000`;
  `QP min ... no offset obj=-3.00000000` → `QP min, same + offset(7) obj=+4.00000000`.
- **Sign convention of the offset under `max`:** it is added *after* the sense is applied, so under
  M2 (§2) the offset must be negated along with `c`: build with `-M.c` and offset `-k`, then negate
  the reported objective, giving `-(-(c·x) - k) = c·x + k`. Equivalently: leave the offset out of
  the model and add it in Python.
- Under M2 the offset can be dropped entirely — linopy's `M.c` already excludes the constant term
  and linopy adds `model.objective.constant` itself, so `set_objective_offset` is optional. Passing
  it is still the cleaner option because it makes `get_primal_objective()` directly comparable.

**`set_objective_scaling_factor` is a reporting-only multiplier — DO NOT use it to flip the sense.**
It scales the *reported* objective but does **not** change what is optimised:

```
min, scaling_factor(2.0)                                       obj=-5.60000000  x=[1.6 1.2]
min, scaling_factor(-1.0)                                      obj=+2.80000000  x=[1.6 1.2]   <-- x is still the MINIMISER of c.x
min, scaling_factor(0.5)                                       obj=-1.40000000  x=[1.6 1.2]
min, offset(10) AND scaling(2)  [docs: scale*(c.x + offset)]    obj=+14.40000000 x=[1.6 1.2]
```

`2*(-2.8 + 10) = 14.4` confirms the documented `scale * (c·x + offset)` formula. Duals and reduced
costs are **not** scaled (still `[-0.40000007, -0.20000008]`), so with a scaling factor ≠ 1 the
reported objective and the duals are in different units. Do not set it.

**Does `set_maximize(True)` flip the sign of duals and reduced costs?** No — not as an independent
transformation. Writing the same optimisation two ways (`min c·x` vs `max (-c)·x`) yields duals that
are exact negatives of each other, which is the mathematically correct outcome for two different
objective functions, and cuOpt matches HiGHS in both forms:

```
(i)  min  c.x    rows <=   c=[-1. -1.]   obj=-2.80000000  dual=[-0.40000007 -0.20000008] rc=[3.16e-07 2.24e-07]
(ii) max (-c).x  rows <=                 obj=+2.80000000  dual=[0.40000007 0.20000008]  rc=[-3.16e-07 -2.24e-07]
     -> dual(ii) == -dual(i) ? True   rc(ii) == -rc(i) ? True

(i)  min  c.x    rows >=   c=[1. 1.]     obj=+2.80000000  dual=[0.40000007 0.20000008]
(ii) max (-c).x  rows >=                 obj=-2.80000000  dual=[-0.40000007 -0.20000008]
     -> dual(ii) == -dual(i) ? True

(i)  min  c.x    rows ==   c=[1. 1.]     obj=+2.80000000  dual=[0.4 0.2]
(ii) max (-c).x  rows ==                 obj=-2.80000000  dual=[0.4 0.2]
     -> dual(ii) == -dual(i) ? False   dual(ii) == dual(i) ? True     <-- THE §2a DEFECT
```

The `==` row is the §2a bug appearing again: cuOpt failed to negate. So the honest statement is:
**`set_maximize(True)` is *supposed* to leave duals in the user's sense convention (matching HiGHS),
and does so on every code path except the presolve-only one, where it forgets to negate.** Under
M2 the question is moot because `set_maximize(True)` is never used.

QP duals under `max` were also checked with a genuinely binding constraint (a_03 §3f) and agree in
all six cells:

```
QP min/<=  cuOpt obj=-2.25000000 x=[0.5 0.5] dual=[-1.5]   HiGHS dual=[-1.49999995]  same=True
QP min/>=  cuOpt obj=-3.00000000 x=[1. 1.]   dual=[1.20e-09] HiGHS dual=[0.]         same=True
QP min/==  cuOpt obj=-2.25000000 x=[0.5 0.5] dual=[-1.5]   HiGHS dual=[-1.49999995]  same=True
QP max/<=  cuOpt obj=+2.25000000 x=[0.5 0.5] dual=[1.5]    HiGHS dual=[1.49999995]   same=True
QP max/>=  cuOpt obj=+3.00000000 x=[1. 1.]   dual=[-1.20e-09] HiGHS dual=[-0.]       same=True
QP max/==  cuOpt obj=+2.25000000 x=[0.5 0.5] dual=[1.5]    HiGHS dual=[1.49999995]   same=True
```

## 4. Log capture
**Script:** `b_q4_logcapture.py` → `b_q4_out.txt`; `b_q4b_logfile_detail.py` → `b_q4b_out.txt`

### Headline

The charter's prediction is **confirmed**: `contextlib.redirect_stdout` captures **0 bytes**.
But an OS-level `dup2` redirect is **not needed** — cuOpt has a **native `log_file` solver
parameter**, and its name is *literally the same string HiGHS uses*, so linopy's existing idiom
transfers verbatim.

**Recommended pattern (matches `Highs._set_solver_params`, `solvers.py:1779-1781`):**

```python
def _set_solver_params(self, settings, log_fn: Path | None = None) -> None:
    if log_fn is not None:
        self.solver_options["log_file"] = path_to_string(log_fn)
        logger.info(f"Log file at {self.solver_options['log_file']}")
    for k, v in self.solver_options.items():
        settings.set_parameter(k, v)
```

`log_to_console` (bool, default `true`) is the companion parameter that silences the terminal.

### Evidence — A/B/C/D/E from `b_q4_out.txt`

```
======================================================================
B) contextlib.redirect_stdout(StringIO)
======================================================================
redirect_stdout captured 0 chars
--- captured begin ---

--- captured end ---
VERDICT B: NOT CAPTURED (leaked to terminal)

======================================================================
C) solver parameter log_file=<path>  (+ log_to_console=False)
======================================================================
log_file exists=True size=1437
--- log_file begin ---
Setting parameter log_file to /tmp/tmpx2lnpyui/cuopt.log
Setting parameter log_to_console to false
cuOpt version: 26.8.0, git hash: 400863c1, host arch: x86_64, device archs: ...
CPU: AMD EPYC 7V12 64-Core Processor, threads (physical/logical): 8/8, RAM ...
CUDA 12.9, device: Tesla T4 (ID 0), VRAM: 15.57 GiB
Solving a problem with 2 constraints, 2 variables (0 integers), and 4 nonzeros
...
Status: Optimal   Objective: 2.80000000e+00  Iterations: 2  Time: 0.011s
--- log_file end ---
VERDICT C: log_file WORKS

======================================================================
D) log_to_console=False alone -> console should be silent
======================================================================
(above this line there should be NO cuOpt banner) obj = 2.8000000000000003

======================================================================
E) OS-level dup2 fd redirect of fd 1 into a temp file
======================================================================
dup2 captured 1338 chars
...
VERDICT E: CAPTURED
```

So all three mechanisms are characterised:

| Mechanism | Captures cuOpt's C++ output? |
|---|---|
| `contextlib.redirect_stdout(StringIO())` | **No — 0 chars.** Output leaks straight to the terminal. |
| solver parameter `log_file=<path>` | **Yes** — 1437 chars written, full banner + iteration log. |
| solver parameter `log_to_console=False` | Console silenced (no banner printed in section D). |
| OS-level `os.dup2` on fd 1 | **Yes** — 1338 chars. Works, but unnecessary given `log_file`. |

### `log_file` semantics (from `b_q4b_out.txt`)

| Property | Behaviour | Pasted proof |
|---|---|---|
| `log_file` + default `log_to_console=true` | Writes to **both** file and terminal (`file size=1394 chars; file has banner: True`, and the banner also appeared on the terminal in section A) | `b_q4b_out.txt` §A |
| Two solves, same `log_file` | **Truncates**, does not append: `after 1st solve: 1437 chars; after 2nd: 1437 chars; 'cuOpt version' occurrences=1` → `VERDICT: TRUNCATES` | §B |
| Unwritable path | Raises from `Solve()` with an actionable message: `RuntimeError: Failed opening file /nonexistent-dir-xyz/cuopt.log for writing: No such file or directory`; `/proc/version` → `Permission denied` | §C |
| `log_file=''` (the default) | Nothing written anywhere; no stray files (`new files in tmpdir: []`) | §D |
| `pathlib.Path` value | **Accepted** (`Path accepted: exists=True size=1439`) — `set_c_solver_settings` stringifies via `str(value)` (`solver_settings.pyx:390`). `path_to_string()` is still the right thing for consistency with the rest of `solvers.py`. | §E |

### Repo idiom survey (what the new class should look like)

| `solvers.py` line | Class | How it handles `log_fn` |
|---|---|---|
| `1779-1781` | `Highs` | `self.solver_options["log_file"] = path_to_string(log_fn)` — **exact same parameter name as cuOpt** |
| `2269-2270` | `Gurobi` | `m.setParam("logfile", path_to_string(log_fn))` |
| `2409-2410` | `Cplex` | `log_f = open(path_to_string(log_fn), "w")` + stream redirect |
| `2558-2559` | `SCIP` | `m.setLogfile(...)` |
| `3014-3016` | `Xpress` | `m.setLogFile(...)` |
| `3702-3703` | `Mosek` | `m.linkfiletostream(mosek.streamtype.log, ...)` |
| `3944-3945` | `COPT` | `m.setLogFile(...)` |
| `4074-4075` | `MindOpt` | `mindoptpy.Env(path_to_string(log_fn) if log_fn else "")` |
| `4362-4363` | `cuPDLPx` | `logger.warning("Log files are not supported by cuPDLPx. Ignoring.")` |

**cuOpt is strictly better off than cuPDLPx here** — it can honour `log_fn` properly rather than
warn-and-ignore. Following the HiGHS line is the smallest, most idiomatic diff.

**Caveat for the plan:** because a user-supplied `solver_options={"log_file": ...}` would be
overwritten by `log_fn` (and vice versa depending on ordering), decide the precedence explicitly.
HiGHS resolves it by having `log_fn` win (it assigns into `solver_options` before the loop).

---

## 5. Edge cases
Script `a_05_edge_cases.py`, output `a_05_out.txt`; ranged-row signs in `a_05b_ranged_signs.py`.

| Case | Behaviour | Evidence |
|---|---|---|
| no constraints | **Unsolvable, and it fails SILENTLY.** Every variant returns `NoTermination` with `obj=nan` and an **empty** `primal_solution`, no exception. The trigger is `nnz == 0`, not `nrows == 0`: a model with one genuine all-zero row also fails. cuOpt needs a constraint matrix with at least one nonzero. | `no A set at all, 2 bounded vars, min x+y -> status=NoTermination cat=LP obj=nan x=array([], dtype=float64)`; `explicit 0-row CSR (indptr=[0]) 2 vars min x+y -> NoTermination`; `one all-zero row, 0 <= 0*x <= 1 -> NoTermination`; `one all-zero row, 1 <= 0*x <= 2 (INFEASIBLE row) -> NoTermination` |
| empty / degenerate model | `DataModel()` with nothing set → `NoTermination`, `obj=nan`, empty arrays, no exception. All-empty arrays (0 vars, 0 rows) → same. An all-zero **objective** over a feasible box is fine. | `completely empty DataModel() -> status=NoTermination cat=LP obj=nan`; `0 vars, 0 rows (all arrays empty) -> NoTermination`; `all-zero objective, feasible box -> Optimal obj=0.0 x=[0. 0.] dual=[0.]` |
| single variable | Works when there is a row with a nonzero; fails like "no constraints" otherwise. | `1 var 1 row: min x, x>=2 -> Optimal obj=2.0 x=[2.] dual=[1.]`; `1 var, no rows: min x, x in [3,9] -> NoTermination`; `1 var, 0-row CSR -> NoTermination` |
| infinite bounds (`np.inf` vs sentinel) | **`np.inf` / `-np.inf` work directly — no sentinel needed, and `M.lb`/`M.ub`/derived row bounds can be passed as-is.** Large finite sentinels (`1e20`, `1e30`, `DBL_MAX`) are accepted and behave like infinity when they do not bind, so they are *not* a distinguishing test — but there is no reason to use them. Pathological bounds degrade honestly: `lb=+inf` and `lb>ub` both give `PrimalInfeasible`, not a crash. | `var ub = np.inf (min x, x>=0) -> Optimal obj=0.0`; `var lb = -np.inf, ub = np.inf, min x s.t. x>=2 (free var) -> Optimal obj=2.0 x=[2.] dual=[1.]`; `row ub = sentinel 1e+30 -> Optimal obj=-3.0`; `row ub = sentinel 1.79769e+308 -> Optimal obj=-3.0`; `row bounds both +inf/-inf -> Optimal obj=0.0 dual=[0.]`; `var lb=+inf (nonsense) -> PrimalInfeasible`; `var lb > ub (2 > 1) -> PrimalInfeasible` |
| ranged constraints | **Fully supported via `set_constraint_lower_bounds` + `set_constraint_upper_bounds`** with `lo != hi`, both finite — LP and MILP, primal and duals. **But `max` + ranged is a live instance of the §2a dual-sign defect and `presolve=1` does not fix it; M2 does.** Also note `problem.Problem` cannot express ranged rows at all. `lo > hi` gives `PrimalInfeasible`. | `ranged 1 <= x+y <= 3, min x+y -> Optimal obj=1.0 x=[0. 1.] dual=[1.]`; `ranged, 2 rows, min ... -> Optimal obj=-2.0 x=[1. -3.] dual=[1. 1.]`; `ranged + MILP -> Optimal cat=MIP obj=2.0`; `ranged row lo > hi (3 <= x+y <= 1) INFEASIBLE -> PrimalInfeasible`; and a_05b: `ranged 1<=x+y<=3, max x+y  cuOpt dual=[-1.]  HiGHS dual=[1.]  *** NEGATED ***` |
| equality via `lb == ub` | Two distinct meanings, both work. **Row** equality via `constraint_lower == constraint_upper` gives the same answer and the same dual as `set_row_types(['E'])`. **Variable** fixing via `variable_lower == variable_upper` works, including all-variables-fixed; if the fixed point violates a row the result is `PrimalInfeasible`. | `row lo == hi == 2, min x -> Optimal obj=2.0 x=[2.] dual=[1.]`; `equality via set_row_types(['E']) -> Optimal obj=2.0 x=[2.] dual=[1.]`; `x fixed at 2.5 by lb==ub -> Optimal obj=2.5 x=[2.5 0.] dual=[0.]`; `ALL vars fixed by lb==ub -> Optimal obj=5.0 x=[2. 3.]`; `all vars fixed but row VIOLATED -> PrimalInfeasible` |
| free variables | Work for both continuous and integer variables. A genuinely unbounded free-variable model is reported as `UnboundedOrInfeasible`, correctly. | `free var, bounded objective: min x s.t. x >= -3, x free -> Optimal obj=-3.0 x=[-3.] dual=[1.]`; `free INTEGER variable, min x s.t. x >= -3.5 -> Optimal cat=MIP obj=-3.0 x=[-3.]`; `free var in equality: min x+y, x-y == 1, both free -> UnboundedOrInfeasible` |
| **variable bounds never set** (extra case) | **The installed docstring is wrong.** `data_model.py:379-381` says the default variable lower bound "for all is 0" and `:397-398` says the default upper bound is `+infinity`. Empirically a model with the bounds never set is `UnboundedOrInfeasible`, i.e. the lower bound is **not** 0. linopy always passes `M.lb`/`M.ub`, so this is documentation-only — but never rely on the defaults. | `variable bounds never set (docs say lb=0, ub=+inf) -> status=UnboundedOrInfeasible cat=LP obj=nan` |

## 6. Array contracts
Script `a_06_array_contracts.py`, output `a_06_out.txt`. Ground truth for the casting behaviour:
`data_model/data_model_wrapper.pyx:25-36` (`type_cast`: `np.asarray` → `astype`, `warnings.warn` on
a float↔int class change), `:150-153` (A → float64/int32/int32), `:168-170` (Q → float64/int32/int32),
`:207-208` (`set_variable_types` stores the object **verbatim**, no cast).

**Stored dtypes after `set_*` (what cuOpt actually holds):**

```
A_values dtype      : float64
A_indices dtype     : int32
A_offsets dtype     : int32
c dtype             : float64
var lb dtype        : float64
scipy csr native    : data float64 indices int32 indptr int32
```

`scipy.sparse.csr_matrix` already produces exactly this, so `M.A.tocsr()` needs no conversion.

**What WORKS (all `-> Optimal obj=-2.800000`):**

```
OK    CSR with int64 indices/indptr
OK    CSR indices/indptr passed as float64      WARN[Casting A_indices from float64 to int32; Casting A_offsets from float64 to int32]
OK    CSR values float32
OK    objective coefficients int64              WARN[Casting c from int64 to float64]
OK    objective coefficients python list
OK    A_values non-C-contiguous strided view                         (vals C_CONTIGUOUS=True)
OK    variable_upper_bounds negative-stride view                     (ub C_CONTIGUOUS=False, strides=(-8,))
```

- **int64 CSR indices are silently accepted** (cast to int32 with no warning, because `type_cast`
  only warns on a float↔int *class* change). scipy switches `indices`/`indptr` to int64 above
  2³¹ nonzeros; there is **no overflow guard**, so a genuinely >2³¹-nnz model would be silently
  truncated. Not reachable on a 16 GB T4, but worth a note.
- **Non-C-contiguity is a non-issue** — `type_cast`'s `astype` copies. A negative-stride view worked.
- Floats where ints are expected, and ints where floats are expected, both work but emit a
  `UserWarning`. linopy should pass exact dtypes so no warning ever reaches a user.

**`set_variable_types` — what works and what FAILS:**

```
OK    np.array(['I','I'])            (dtype <U1)     -> cat=MIP Optimal obj=-2.000000 stored dtype=|S1
OK    np.array(['C','C'])            (dtype <U1)     -> cat=LP  Optimal obj=-2.800000 stored dtype=|S1
OK    np.array(['I','C'])            (dtype <U1)     -> cat=MIP Optimal obj=-2.500000 stored dtype=|S1
OK    np.array([b'I',b'I'])          (dtype |S1)     -> cat=MIP Optimal obj=-2.000000 stored dtype=|S1
OK    np.array(['I','I'], dtype='S1')                -> cat=MIP Optimal obj=-2.000000 stored dtype=|S1
FAIL  python list ['I','I']                          -> AttributeError: 'list' object has no attribute 'astype'
FAIL  python list [b'I',b'I']                        -> AttributeError: 'list' object has no attribute 'astype'
FAIL  python str 'II'                                -> AttributeError: 'str' object has no attribute 'astype'
FAIL  python bytes b'II'                             -> TypeError: a bytes-like object is required, not 'str'
```

**Silent-wrong-answer traps in the same setter** — these do NOT raise:

```
OK    np.array([73,73], int8) (ascii 'I')            -> cat=LP  Optimal obj=-2.800000   <-- integrality SILENTLY IGNORED
OK    lowercase np.array(['i','i'])                  -> cat=LP  Optimal obj=-2.800000   <-- case-sensitive, silently continuous
OK    np.array(['Z','Z'])  (nonsense)                -> cat=LP  Optimal obj=-2.800000   <-- unknown code silently continuous
OK    np.array(['B','B'])  (binary?)                 -> cat=MIP Optimal obj=-2.000000   <-- treated as integer; does NOT clamp to [0,1] (ub was 10)
OK    np.array(['S','S'])  (semi-continuous)         -> cat=MIP FeasibleFound obj=-2.800000
OK    EMPTY np.array([], '<U1')                      -> cat=LP  Optimal obj=-2.800000
OK    WRONG LENGTH np.array(['I']) for 2 vars        -> cat=MIP NoTermination obj=0.000000
OK    WRONG LENGTH np.array(['I','I','I']) for 2 vars-> cat=MIP NoTermination obj=0.000000
```

Consequences for the implementation:
- Pass a **`np.ndarray`**, never a list/str/bytes. `<U1` and `|S1` both work; `<U1` matches
  `M.vtypes`'s own dtype so `dm.set_variable_types(M.vtypes)` is the natural call.
- Only uppercase `'C'`, `'I'`, `'S'` (and `'B'` ≡ `'I'`) are recognised. **Anything else is silently
  continuous.** linopy's `M.vtypes` uses `'C'`/`'I'`/`'B'`; `'B'` must be mapped to `'I'` *and*
  the `[0,1]` bounds enforced through `lb`/`ub` (cuOpt does not clamp `'B'`).
- The dtype `get_variable_types()` returns depends on the path: `|S1` after `set_variable_types`,
  `<U1` after `io.Read`. Guards must accept both.
- `is_mip()` in `solver/solver.py:85-100` already handles `str`/`bytes` and mixed lists, and looks
  only for `'I'`/`'S'` — consistent with the above.

**Wrong-length numeric arrays fail SILENTLY too:**

```
OK    c too short (1 for 2 vars)                     -> NoTermination obj=nan
OK    c too long (3 for 2 vars)                      -> NoTermination obj=nan
OK    var lb too short                               -> NoTermination obj=nan
OK    constraint ub too short                        -> NoTermination obj=nan
```

No exception, no error message, `NoTermination` with `nan`. So **`NoTermination` is cuOpt's
catch-all "your input was malformed"**, and `get_error_status()` / `get_error_message()` are the
only way to learn why (they were populated for the MIQP case, §1). Validate array lengths on the
linopy side before calling `Solve`.

**`set_row_types` is more permissive than `set_variable_types`** (it iterates characters rather than
calling `astype`, `data_model_wrapper.pyx:190-195`):

```
OK    set_row_types np.array(['L','L']) + set_constraint_bounds      -> Optimal obj=-2.800000 x=[1.6 1.2]
OK    set_row_types np.array([b'L',b'L'])                            -> Optimal obj=-2.800000 x=[1.6 1.2]
OK    set_row_types python list ['L','L']                            -> Optimal obj=-2.800000 x=[1.6 1.2]
OK    set_row_types ['G','G']                                        -> Optimal obj=-20.000000 x=[10. 10.]
OK    set_row_types ['E','E']                                        -> Optimal obj=-2.800000 x=[1.6 1.2]
OK    set_row_types ['R','R'] (ranged? nonsense?)                    -> NoTermination obj=nan x=[]
```

There is **no `'R'` (ranged) row type** — ranged rows must go through
`set_constraint_lower_bounds`/`set_constraint_upper_bounds`. Since linopy has to support `<`, `>`,
`=` and (via `M.b` pairs) ranges anyway, the bounds-based route is the one to use throughout and
`set_row_types` should be left alone (`data_model.py:468-470`: if both are set, the bounds win).

**Also worth recording:** `set_parameter` is type-strict on booleans — `presolve=False` raises
`ValueError: Parameter presolve value False is not an integer` (`a_01c_out.txt`, 6 occurrences),
whereas `log_to_console=False` is accepted. Pass ints for the int-typed parameters.

## 7. Solver options
**Scripts:** `b_q7_params.txt` (full name/default dump), `b_q7_options.py` → `b_q7_out.txt`

### Enumeration

`cuopt.linear_programming.solver_settings.solver_settings.get_solver_parameter_names()`
(`solver_settings.pyx:35-49`) returns **117 entries, 109 unique**. `solver_params` is the module-level
list; the module also injects `CUOPT_<UPPERCASE>` string constants for each
(`solver_settings.pyx:52-53`, re-exported by `solver/solver_parameters.py`).

```
  117 names, 109 unique
  DUPLICATED names: ['log_file', 'log_to_console', 'num_gpus', 'presolve', 'presolve_file',
                     'solution_file', 'time_limit', 'user_problem_file']
  tolerances: ['mip_relative_tolerance', 'relative_gap_tolerance', 'primal_infeasible_tolerance',
               'absolute_dual_tolerance', 'relative_primal_tolerance', 'absolute_primal_tolerance',
               'relative_dual_tolerance', 'absolute_gap_tolerance', 'mip_integrality_tolerance',
               'mip_absolute_tolerance', 'dual_infeasible_tolerance']
  limits: ['iteration_limit', 'mip_hyper_diving_backtrack_limit',
           'mip_hyper_diving_iteration_limit_factor', 'mip_hyper_diving_node_limit',
           'mip_hyper_heuristic_related_vars_time_limit', 'mip_hyper_heuristic_relaxed_lp_time_limit',
           'mip_hyper_heuristic_rins_max_time_limit', 'mip_hyper_heuristic_rins_time_limit',
           'mip_hyper_submip_iteration_limit_ratio', 'mip_hyper_submip_node_limit_base',
           'mip_strong_branching_simplex_iteration_limit', 'node_limit', 'time_limit', 'work_limit']
```

(Eight names are registered twice in the C++ layer — harmless duplicates, but code that iterates
`solver_params` should not assume uniqueness.)

### The parameters linopy actually needs

| Parameter (real name) | Type | Default | Effect | Behaviour on unknown name |
|---|---|---|---|---|
| `time_limit` | float (seconds) | `inf` | Wall-clock cap; terminates with status `5 TimeLimit` | n/a |
| `iteration_limit` | int | `2147483647` | Terminates with status `4 IterationLimit` | n/a |
| `node_limit` | int | `2147483647` | MILP node cap | n/a |
| `work_limit` | float | `inf` | deterministic work cap | n/a |
| `method` | int (`SolverMethod`) | `0` = Concurrent | `0` Concurrent / `1` PDLP / `2` DualSimplex / `3` Barrier / `4` Unset | n/a |
| `pdlp_solver_mode` | int (`PDLPSolverMode`) | `4` = Stable3 | `0` Stable1, `1` Stable2, `2` Methodical1, `3` Fast1, `4` Stable3 | n/a |
| `presolve` | int | `-1` (auto) | `0` disables; see §11 — it also determines whether warm-start data round-trips | n/a |
| `log_to_console` | bool | `true` | Console log on/off | n/a |
| `log_file` | str | `''` | Log file path (see §4) | n/a |
| `absolute_primal_tolerance` | float | `1e-4` | LP primal feasibility (abs) | n/a |
| `relative_primal_tolerance` | float | `1e-4` | LP primal feasibility (rel) | n/a |
| `absolute_dual_tolerance` | float | `1e-4` | LP dual feasibility (abs) | n/a |
| `relative_dual_tolerance` | float | `1e-4` | LP dual feasibility (rel) | n/a |
| `absolute_gap_tolerance` | float | `1e-4` | LP duality gap (abs) | n/a |
| `relative_gap_tolerance` | float | `1e-4` | LP duality gap (rel) | n/a |
| `mip_absolute_tolerance` | float | `1e-6` | MILP objective abs | n/a |
| `mip_relative_tolerance` | float | `0.0` | MILP objective rel | n/a |
| `mip_integrality_tolerance` | float | `1e-5` | integrality slack | n/a |
| `mip_absolute_gap` | float | `0.0` | MILP abs gap | n/a |
| `mip_relative_gap` | float | `1e-4` | MILP rel gap | n/a |
| `primal_infeasible_tolerance` | float | `0.0` | infeasibility detection | n/a |
| `dual_infeasible_tolerance` | float | `0.0` | infeasibility detection | n/a |
| `crossover` | bool | `false` | barrier→simplex crossover | n/a |
| `num_cpu_threads` | int | `-1` | CPU threads for the CPU-side methods | n/a |
| `random_seed` | int | `-1` | | n/a |
| `solution_file` / `user_problem_file` / `presolve_file` | str | `''` | file outputs — possible route for linopy's `solution_fn` | n/a |
| `mip_semi_continuous_big_m` | float | `1e10` | big-M used for semi-continuous vars (see §10) | n/a |

The full 117-name dump with every default is in `b_q7_params.txt`.

### Unknown parameter **name** → `ValueError`, raised eagerly at `set_parameter`

`SolverSettings.set_parameter` guards on membership of `solver_params`
(`solver_settings.pyx:191-193`). `get_parameter` does the same (`:166-167`).

```
======================================================================
A) UNKNOWN PARAMETER NAME -> set_parameter
======================================================================
  set_parameter('TimeLimit', 1) -> ValueError: Invalid parameter. Please check documentation
  set_parameter('time-limit', 1) -> ValueError: Invalid parameter. Please check documentation
  set_parameter('', 1) -> ValueError: Invalid parameter. Please check documentation
  set_parameter('log_TO_console', 1) -> ValueError: Invalid parameter. Please check documentation
  set_parameter('not_a_param', 1) -> ValueError: Invalid parameter. Please check documentation

B) UNKNOWN PARAMETER NAME -> get_parameter
  get_parameter('not_a_param') -> ValueError: Invalid parameter. Please check documentation
```

Notes for the plan: **names are case-sensitive and snake_case** — a user habitually writing
`TimeLimit` (Gurobi/cuPDLPx style) gets a `ValueError` whose message does not name the offending
parameter. Consider wrapping it.

### Unknown/invalid **value** → `ValueError`, raised **late, at `Solve()`**

`set_parameter` only stores into `settings_dict`; validation happens in C++ during
`set_c_solver_settings` (`solver_settings.pyx:388`) which `Solve` calls
(`solver_wrapper.pyx:203`, `:461`):

```
C) WRONG *VALUE* for a known parameter (type/range abuse)
  time_limit='not_a_number' -> ValueError: Parameter time_limit value not_a_number is not a float
  time_limit=-5.0 -> ValueError: Parameter time_limit value -5.0 out of range
  method=999 -> ValueError: Parameter method value 999 out of range
  log_to_console='banana' -> ValueError: Parameter log_to_console value banana must be true or false
  absolute_primal_tolerance='1e-9' -> accepted; status=1 (Optimal) obj=2.8000000000000003 err=0 msg=''
```

Isolated traceback confirming the raise point (`b_q7_out.txt` sibling run):

```
  File "cuopt/linear_programming/solver/solver_wrapper.pyx", line 461, in ...Solve
  File "cuopt/linear_programming/solver/solver_wrapper.pyx", line 203, in ...set_solver_setting
  File "cuopt/linear_programming/solver_settings/solver_settings.pyx", line 388, in ...set_c_solver_settings
ValueError: Parameter time_limit value not_a_number is not a float
set_parameter with bad VALUE returned fine; settings_dict = {'time_limit': 'not_a_number'}
```

Value messages **do** name the parameter and the value — good enough to surface directly.
Numeric strings are coerced (`'1e-9'` accepted).

### Limits actually bite

```
D) TIME LIMIT actually bites (large LP, time_limit=0.05)
  status=5 name=TimeLimit reason='TimeLimit'
  solve_time=0.111991887
  same model, no time limit, PDLP:
  status=1 name=Optimal solve_time=0.405075665

E) ITERATION LIMIT bites (iteration_limit=1, PDLP)
  status=4 name=IterationLimit reason='IterationLimit'
```

### Tolerances

`SolverSettings.set_optimality_tolerance(eps)` is a convenience that fans out to the six LP
tolerances only (it skips `mip*` and `*infeasible*`, `solver_settings.pyx:224-227`):

```
F) TOLERANCES: set_optimality_tolerance + individual tolerance params
  set_optimality_tolerance(1e-9) touched:
    absolute_dual_tolerance = 1e-09
    relative_dual_tolerance = 1e-09
    absolute_primal_tolerance = 1e-09
    relative_primal_tolerance = 1e-09
    absolute_gap_tolerance = 1e-09
    relative_gap_tolerance = 1e-09

  round-trip get_parameter after set_parameter:
    time_limit: set 12.5 -> get 12.5
    absolute_primal_tolerance: set 1e-08 -> get 1e-08
    log_to_console: set False -> get False
    method: set 2 -> get 2
    log_file: set '/tmp/x.log' -> get '/tmp/x.log'
```

**⚠️ Directly relevant to the frozen verification thresholds (§6.3a).** The **default `1e-4`
tolerance is loose enough to move the objective by ~0.55 %** on a 2000×1000 random LP:

```
  tolerance actually changes result quality (PDLP, loose vs tight):
    tol=0.01: status=Optimal obj=4916.2143590849 iters=140 time=0.056
    tol=1e-10: status=Optimal obj=4889.2173861980 iters=34000 time=1.941
```

`|4916.214 - 4889.217| / 4889.217 = 5.52e-3`. A differential test at `rtol=1e-6` on the PDLP path
therefore **requires** tightening the tolerances explicitly. For calibration, the four `method`
values disagree with each other on the same model at default tolerances by ~1e-5 relative
(`b_q9b_out.txt`, n=5000 m=2500, seed 100: Concurrent `12506.8355`, PDLP `12506.8538`,
DualSimplex `12506.8355`, Barrier `12506.8226` → spread `2.5e-5`), so the chosen `rtol` interacts
with the chosen `method`. I did **not** compare any of these against HiGHS — that is item 2's job.
Whoever writes `logs/15-verification.md` needs both numbers.

---

## 8. Interruptibility
**Scripts:** `b_q8_interrupt.py` → `b_q8_out.txt`; `b_q8b_exit.py` → `b_q8b_out.txt`

### Verdict: `solver.Solve()` is **NOT** KeyboardInterrupt-safe. There is **no cancel API**.

SIGINT delivered while the C++ `Solve()` is running is **deferred for the entire remaining solve
duration** (~53 s in the run below), then raised at the first Python bytecode boundary — which lands
*inside* `create_solution`, i.e. after the solve has already finished:

```
===== MODE direct =====
[main] building model n=40000 m=20000
[main] model built at t=41.85s
[helper] sending SIGINT to pid 208096 at t=5.00s
[main] KeyboardInterrupt raised at t=57.92s
Traceback (most recent call last):
  File ".../b_q8_interrupt.py", line 63, in <module>
    sol = solve(dm, st)
  File ".../cuopt/linear_programming/solver/solver.py", line 102, in Solve
    s = solver_wrapper.Solve(
  File "cuopt/linear_programming/solver/solver_wrapper.pyx", line 472, in ...Solve
  File "cuopt/linear_programming/solver/solver_wrapper.pyx", line 254, in ...create_solution
  File ".../data_model/utilities/exception_handler.py", line 21, in func
KeyboardInterrupt
[main] still alive, exiting at t=57.92s
```

SIGINT at t=5.00 s → `KeyboardInterrupt` at t=57.92 s. **52.9 s of unresponsiveness.**

### cuOpt installs no SIGINT handler of its own

```
===== MODE handler =====
SIGINT handler BEFORE import cuopt: <built-in function default_int_handler>
SIGINT handler AFTER  import cuopt: <built-in function default_int_handler>
SIGINT handler AFTER  a solve       : <built-in function default_int_handler>
```

### No cancel/abort hook exists on the in-process LP solver

`grep -rni "cancel|abort|interrupt|sigint|keyboard"` over
`.venv/lib/python3.11/site-packages/cuopt/**/*.{py,pyx,pxd}` (excluding `tests/`) matches only:

* `grpc/linear_programming/grpc_client.pyx:302` `def cancel(self, str job_id)` — the **out-of-process
  gRPC/server** client, which §4.2 rules out for linopy.
* `routing/vehicle_routing_wrapper.pyx:145` — a `sys.excepthook` clause in the *routing* module.

`DataModel` / `SolverSettings` / `solver.Solve` expose **nothing**. There is no `cancelSolve()`
counterpart to HiGHS's, so `_run_highs_with_keyboard_interrupt`'s cancel step has no analogue.

### The HiGHS worker-thread pattern still works — and is the right precedent

Running `Solve()` in a worker thread while the main thread waits on `Event.wait(0.1)`
(`_run_highs_with_keyboard_interrupt`, `solvers.py:189-250`) gives **immediate** Ctrl-C
responsiveness. The solve itself cannot be stopped and runs to completion in the background:

```
===== MODE threaded =====
[main] building model n=40000 m=20000
[main] model built at t=41.78s
[helper] sending SIGINT to pid 208843 at t=5.00s
[main] KeyboardInterrupt in waiter at t=5.00s (worker still running in background)
[main] waiting for worker to finish so we can report its fate...
[main] worker finished at t=58.21s err=None status=Optimal
```

**Latency 0.00 s** vs 52.9 s. And the process **exits cleanly and immediately** if the interpreter
tears down while the daemon thread is still inside cuOpt's C++ code (`b_q8b_out.txt`):

```
[main] model built t=41.0s
[main] KeyboardInterrupt at t=5.00s; exiting NOW without joining
[main] falling off the end at t=5.00s (interpreter shutdown begins)
shell-observed exit=0
```

No hang, no crash, no CUDA teardown error. So the recommended shape is a
`_run_cuopt_with_keyboard_interrupt(dm, settings)` helper mirroring `solvers.py:189`, minus the
cancel step — with a docstring saying plainly that the GPU work continues until the process exits,
and that `time_limit` is the only real bound cuOpt offers.

---

## 9. Repeat solves and teardown
**Scripts:** `b_q9_repeat.py` → `b_q9_out.txt`; `b_q9_fork_segv.py` → `b_q9_segv_out.txt`;
`b_q9b_segv_narrow.py` → `b_q9b_out.txt` / `b_q9e_sweep.txt`; `b_q9c_segv_trace.py` → `b_q9c_out.txt`;
`b_q9d_confirm.py` → `b_q9d_out.txt`; `b_q9f_bigmem.py` → `b_q9f_out.txt`; peak sampler → `b_q9g_peak.txt`

### 🔴 BLOCKING FINDING — the DEFAULT solver method segfaults on the 2nd/3rd sequential solve

`method=0` (**Concurrent**, cuOpt's default) **hard-crashes the process with SIGSEGV** on the second
or third sequential `Solve()` in one interpreter, for LPs above roughly **n ≈ 1300** variables.
Single-method modes (`1` PDLP, `2` DualSimplex, `3` Barrier) are **unaffected**.

Deterministic, reproduced in ≥12 fresh processes. Size sweep (`b_q9e_sweep.txt`, 8 reps each,
`method=0`, m = n/2):

```
--- method=0 n=200  m=100 reps=8 --- -> 8 solves, exit=0
--- method=0 n=500  m=250 reps=8 --- -> 8 solves, exit=0
--- method=0 n=1000 m=500 reps=8 --- -> 8 solves, exit=0
--- method=0 n=1100 m=550 reps=8 --- -> 8 solves, exit=0
--- method=0 n=1200 m=600 reps=8 --- -> 8 solves, exit=0
--- method=0 n=1300 m=650 reps=8 --- -> 6 solves, exit=139
--- method=0 n=1400 m=700 reps=8 --- -> 3 solves, exit=139
--- method=0 n=1500 m=750 reps=8 --- -> 2 solves, exit=139
```

Method comparison at the same shape (`b_q9b_out.txt`, 6 reps, n=5000 m=2500):

```
### method=0 ###  solve 0 only, then exit=139
### method=1 ###  solves 0..5 all Optimal, COMPLETED, exit=0
### method=2 ###  solves 0..5 all Optimal, COMPLETED, exit=0
### method=3 ###  solves 0..5 all Optimal, COMPLETED, exit=0
```

25 sequential solves cycling five different shapes (`b_q9d_out.txt`):

```
########## method=0 ##########
    0 n= 1500 m=  750 obj=    4128.5983 solved_by=2 wall= 0.389s dev_used=117 MiB
exit=139
########## method=1 ##########  25/25 COMPLETED, dev_used flat at 111 MiB, exit=0
########## method=2 ##########  25/25 COMPLETED, dev_used 107 MiB (transient 271), exit=0
########## method=3 ##########  25/25 COMPLETED, dev_used flat at 117 MiB, exit=0
```

`faulthandler` stack (`b_q9c_out.txt`) puts the fault inside `Solve`, and shows the crash is
insensitive to `num_cpu_threads=1` and `num_gpus=1`; setting `presolve=0` makes it crash **one solve
earlier**:

```
###  ###                       solves 0,1 then SIGSEGV
### num_cpu_threads=1 ###      solves 0,1 then SIGSEGV
### presolve=0 ###             solve 0 then SIGSEGV
### method=0 num_gpus=1 ###    solves 0,1 then SIGSEGV

Thread 0x000076358a2a7740 (most recent call first):
  File ".../cuopt/linear_programming/solver/solver.py", line 102 in Solve
  File ".../cuopt/utilities/exception_handler.py", line 24 in func
  File ".../b_q9c_segv_trace.py", line 33 in <module>
```

**Why this is a first-class risk for linopy, not a curiosity:** `pytest` runs many solves in one
process. linopy's own test models are tiny (well under 1300 vars), so the suite would likely stay
green while **a real user's second `model.solve("cuopt")` in a script crashes the interpreter with
no traceback**. Mitigations the plan should weigh: (a) default `method` to a single-method mode
(`2` DualSimplex was the fastest and most stable single method in my runs — 25/25 varied solves,
flat memory; I did **not** compare its answers against HiGHS, that is item 2's job), (b) document
the hazard, (c) add a repeat-solve regression test with a model above the threshold. Doing nothing
is not defensible.

*(An earlier version of the repeat-solve script sampled memory via `subprocess.run("nvidia-smi")`;
`b_q9_segv_out.txt` shows the crash reproduces identically **with and without** any subprocess call,
so `fork()`-after-CUDA is **not** the cause.)*

### Sequential solves under a safe method: stable, no leak

`b_q9_out.txt`, 20 sequential LPs at n=5000 m=2500 with `method=2`:

```
    #     status             obj   wall_s  dev_used_MiB
    0    Optimal    12506.835477    0.309           107
    1    Optimal    12708.990798    0.201           107
   ...
   19    Optimal    12408.167976    0.290           107
```

Device usage **flat at 107 MiB across all 20 solves**. 10 sequential MILPs (n=300 m=150,
`time_limit=10`) hovered at 191 MiB with transient peaks to 363 MiB and returned to 191 MiB.
Alternating LP/MILP with `del` + `gc.collect()`:

```
  0    LP: after solve=191 MiB, after del+gc=191 MiB
  1  MILP: after solve=191 MiB, after del+gc=191 MiB
  ... (through 5)
```

i.e. **`gc` makes no difference** — cuOpt releases its own device allocations at the end of each
`Solve()`, so there is nothing for Python to free.

### GPU context lifetime / first-solve overhead

* `import cuopt` and `import cuopt.linear_programming` allocate **no** device memory; the CUDA
  context appears on the first `Solve()`.
* First-solve overhead is modest: first LP 0.309 s vs ~0.20-0.29 s steady state (n=5000 m=2500),
  and 0.372 s vs ~0.04 s at n=1000 m=500. Order **0.3 s** of one-off init.
* **No explicit teardown API exists:**

```
F) Explicit teardown API?
  cuopt: teardown-ish attributes -> []
  cuopt.linear_programming: teardown-ish attributes -> []
```

So there is nothing for a `close()` override to call — unlike `Gurobi`, which manages an `Env`
stack (`solvers.py:1951-1970`).

### Behaviour near the T4's 16 GB

Peak device memory sampled every 20 ms by a background thread while `Solve()` ran
(`b_q9g_peak.txt`; PDLP, `time_limit=15`):

```
baseline device_used=105 MiB / 15948 MiB (rmm.mr.available_device_memory -> cudaMemGetInfo, process-global)
  n=  1000000 m=   500000 nnz=   4000000: status=Optimal        wall= 4.95s PEAK device_used=  707 MiB / 15948 MiB, after=119 MiB
  n= 20000000 m= 10000000 nnz=  80000000: status=TimeLimit      wall=35.95s PEAK device_used=12857 MiB / 15948 MiB, after=119 MiB
  n= 60000000 m= 30000000 nnz= 240000000: status=NoTermination  wall=36.50s PEAK device_used=14231 MiB / 15948 MiB, after=117 MiB
```

* Memory returns to the ~117-119 MiB baseline after **every** solve, including the 14.2 GiB one — no leak.
* At 14231 / 15948 MiB the solver **degrades to `NoTermination` rather than raising an OOM**. That
  status must map to something honest (`TerminationCondition.unknown` at minimum); a user hitting the
  memory ceiling gets a *silently useless* answer, not an error. **The plan should treat
  `NoTermination` as a real, reachable status, not a theoretical one.**
* **UNKNOWN — I did not force an actual CUDA OOM.** cuOpt absorbed 60 M variables / 240 M nonzeros
  without raising; going further would have needed host RAM beyond what was prudent on a GPU shared
  with a sibling agent. The failure mode *at* the ceiling is characterised (`NoTermination`); the
  failure mode *past* it is not.
* Process remained usable after the largest attempt: `yes: obj=2412.585908 dev_used=191 MiB`.

---

## 10. Unsupported surfaces
**Script:** `b_q10_unsupported.py` → `b_q10_out.txt`, plus greps over the installed package.

| Surface | Genuinely absent? | How confirmed |
|---|---|---|
| semi-continuous | **NO — cuOpt SUPPORTS it, on both entry points** | `VType.SEMI_CONTINUOUS = "S"` at `linear_programming/problem.py:30`; `mip_semi_continuous_big_m` is a real solver parameter (default `1e10`); solved empirically both ways (below) |
| SOS (SOS1/SOS2) | **YES — absent, and explicitly out of scope** | No `sos`/`ordered` attribute on `DataModel`, `Problem` or `cuopt.linear_programming`; the LP/MPS parser rejects an `SOS` section with `InputValidationError: LP section 'SOS' is not supported (scope is LP/MIP/QP only)`; upstream test `tests/linear_programming/test_parser.py:154` comments "SOS is explicitly out of scope" |
| indicator constraints | **YES — absent** | No `indicator`/`genconstr` attribute anywhere; `grep -rni indicator` over `*.py`/`*.pyx`/`*.pxd` matches only an unrelated `_REMOTE_INDICATORS` list in `tests/linear_programming/test_cpu_only_execution.py:349` |

### Semi-continuous — pasted proof

Discriminating model: `min x` s.t. `x <= 100`, `x` bounded `[5, 10]`. Continuous/integer optimum is
**5**; semi-continuous (`x = 0` **or** `5 <= x <= 10`) optimum is **0**.

```
======================================================================
A) SEMI-CONTINUOUS via DataModel.set_variable_types
======================================================================
  vtype='C': status=1 obj=5.0 x=[5.0]
  vtype='I': status=1 obj=5.0 x=[5.0]
  vtype='S': status=1 obj=0.0 x=[0.0]

======================================================================
B) SEMI-CONTINUOUS via problem.Problem
======================================================================
  CONTINUOUS: x=np.float64(5.0) obj=np.float64(5.0)
  SEMI_CONTINUOUS: x=np.float64(0.0) obj=np.float64(0.0)
```

**Consequence for the plan:** cuPDLPx's guard
(`solvers.py:4249-4253`, "Semi-continuous variables are not supported by cuPDLPx…") **must not be
copied into the cuOpt class**. Doing so would raise `NotImplementedError` for something cuOpt
demonstrably solves — the exact opposite of "unsupported features raise honestly". Whether the
first cuOpt slice *implements* semi-continuous is a scope decision (it needs a `'S'` in the
`set_variable_types` array wherever `model.variables.semi_continuous` is set, and a decision about
`mip_semi_continuous_big_m`), but the class must not claim it is impossible.

### ⚠️ Invalid variable-type characters are silently accepted

```
  bogus variable type characters:
  vtype='B': ACCEPTED status=1 obj=5.0
  vtype='Z': ACCEPTED status=1 obj=5.0
  vtype='s': ACCEPTED status=1 obj=5.0
  vtype='N': ACCEPTED status=1 obj=5.0
```

`DataModel.set_variable_types` performs **no validation** (`data_model_wrapper.pyx:207-208` just
stores the array; the C++ side takes a raw `const char*`, `:412-415`). Every unrecognised character
is treated as continuous — lower-case `'s'` silently loses semi-continuous semantics. **linopy must
build that array itself and never pass user input through**, and the mapping must be exactly
`'C'`/`'I'`/`'S'`.

### Full public surface, for the record

```
DataModel: ['add_quadratic_constraint', 'clear_quadratic_constraints', 'get_ascii_row_types',
 'get_constraint_bounds', 'get_constraint_lower_bounds', 'get_constraint_matrix_indices',
 'get_constraint_matrix_offsets', 'get_constraint_matrix_values', 'get_constraint_upper_bounds',
 'get_initial_dual_solution', 'get_initial_primal_solution', 'get_objective_coefficients',
 'get_objective_name', 'get_objective_offset', 'get_objective_scaling_factor', 'get_problem_name',
 'get_quadratic_constraints', 'get_quadratic_objective_indices', 'get_quadratic_objective_offsets',
 'get_quadratic_objective_values', 'get_row_names', 'get_row_types', 'get_sense',
 'get_variable_lower_bounds', 'get_variable_names', 'get_variable_types',
 'get_variable_upper_bounds', 'set_constraint_bounds', 'set_constraint_lower_bounds',
 'set_constraint_upper_bounds', 'set_csr_constraint_matrix', 'set_data_model_view',
 'set_initial_dual_solution', 'set_initial_primal_solution', 'set_maximize',
 'set_objective_coefficients', 'set_objective_name', 'set_objective_offset',
 'set_objective_scaling_factor', 'set_problem_name', 'set_quadratic_objective_matrix',
 'set_row_names', 'set_row_types', 'set_variable_lower_bounds', 'set_variable_names',
 'set_variable_types', 'set_variable_upper_bounds', 'writeMPS']

Problem: ['IsMIP', 'NumConstraints', 'NumNZs', 'NumVariables', 'Obj', 'ObjValue', 'addConstraint',
 'addVariable', 'dict_to_object', 'getCSR', 'getConstraint', 'getConstraints', 'getIncumbentValues',
 'getObjective', 'getQCSR', 'getQcsr', 'getQuadraticConstraints', 'getVariable', 'getVariables',
 'getWarmstartData', 'get_incumbent_values', 'get_pdlp_warm_start_data', 'populate_solution',
 'read', 'readMPS', 'relax', 'reset_solved_values', 'setObjective', 'solve', 'update',
 'updateConstraint', 'updateObjective', 'writeMPS']
```

(`DataModel.writeMPS` exists — relevant to Q0's `io_api` question, which is agent A's item.)

---

## 11. Warm start
**Scripts:** `b_q11_warmstart.py` → `b_q11_out.txt`; `b_q11b_initialpoint.py` → `b_q11b_out.txt`

### Two distinct mechanisms exist; one works, one crashes the CUDA context

| Mechanism | API | Works? |
|---|---|---|
| Full PDLP state | `Solution.get_pdlp_warm_start_data()` → `SolverSettings.set_pdlp_warm_start_data()` | **Yes — but only with `presolve=0` and `pdlp_solver_mode=Stable2`** |
| Initial point (primal) | `DataModel.set_initial_primal_solution()` on an **LP** | 🔴 **NO — `cudaErrorIllegalAddress`, poisons the CUDA context** |
| Initial point (dual) | `DataModel.set_initial_dual_solution()` on an LP | Yes (but made the solve *slower*: 2100 vs 1500 iterations) |
| MILP start | `DataModel.set_initial_primal_solution()` on a **MILP** (routed to `add_initial_mip_solution`, `solver_wrapper.pyx:218-222`) | Yes |
| `Problem` path | `Problem.getWarmstartData()` (`problem.py:1907`) returns the same `PDLPWarmStartData`; `get_pdlp_warm_start_data()` is deprecated (`problem.py:1926-1932`) | same constraints |

### What the payload carries

`cuopt.linear_programming.solution.solution.PDLPWarmStartData` — a plain Python object, **17 fields,
all host `numpy.float64` arrays or Python scalars** (`cuda_iface=False`, so it is *not* device
memory and can be held/pickled freely):

```
    current_ATY                                      ndarray    shape=(2000,) dtype=float64 cuda_iface=False
    current_dual_solution                            ndarray    shape=(1000,) dtype=float64 cuda_iface=False
    current_primal_solution                          ndarray    shape=(2000,) dtype=float64 cuda_iface=False
    initial_dual_average                             ndarray    shape=(1000,) dtype=float64 cuda_iface=False
    initial_primal_average                           ndarray    shape=(2000,) dtype=float64 cuda_iface=False
    initial_primal_weight                            float      0.7776658991021892
    initial_step_size                                float      1.5424617156721339
    iterations_since_last_restart                    int        760
    last_candidate_kkt_score                         float      0.003926767546857462
    last_restart_duality_gap_dual_solution           ndarray    shape=(1000,) dtype=float64 cuda_iface=False
    last_restart_duality_gap_primal_solution         ndarray    shape=(2000,) dtype=float64 cuda_iface=False
    last_restart_kkt_score                           float      0.00513994391319315
    sum_dual_solutions                               ndarray    shape=(1000,) dtype=float64 cuda_iface=False
    sum_primal_solutions                             ndarray    shape=(2000,) dtype=float64 cuda_iface=False
    sum_solution_weight                              float      1308.4969077275582
    total_pdhg_iterations                            int        3202
    total_pdlp_iterations                            int        3200
```

It is PDLP algorithm state (primal/dual iterates, running averages, restart bookkeeping, step size,
primal weight) — **not** a simplex basis. Nothing in it is usable by `DualSimplex` or `Barrier`.

### 🔴 The payload comes back in **presolved** space — so the default settings cannot round-trip it

`set_solver_setting` validates the warm-start arrays against the **user** model's dimensions
(`solver_wrapper.pyx:187-201`), but the arrays returned are sized to the **presolved** model:

```
======================================================================
A2) SHAPE of the payload vs the `presolve` parameter
======================================================================
  presolve= 0: primal=(2000,) dual=(1000,)  (user model n=2000 m=1000)
  presolve= 1: primal=(1845,) dual=(994,)  (user model n=2000 m=1000)
  presolve=-1: primal=(1634,) dual=(942,)  (user model n=2000 m=1000)
```

With the **default `presolve=-1`**, feeding the data straight back into the *identical* model fails:

```
Exception: Invalid PDLPWarmStart data. Passed problem and PDLPWarmStart data should have the same amount of variables.
```

Only `presolve=0` makes it round-trip. That is a genuine upstream defect, and it means "warm start"
in cuOpt costs you presolve.

### With `presolve=0` it works, and works well

```
======================================================================
B) WARM restart on the SAME model
======================================================================
  warm (same model): status=Optimal obj=5232.12682726 iters=40 time=0.016s
  cold iters=3200  warm iters=40  ratio=0.0125

======================================================================
C) WARM restart on a PERTURBED model of the same shape
======================================================================
  perturbed cold: status=Optimal obj=5231.92959385 iters=7600 time=0.629s
  perturbed warm: status=Optimal obj=5231.92968460 iters=4880 time=0.405s
  objective: cold=5231.92959385 warm=5231.92968460 reldiff=1.735e-08

======================================================================
D) REUSE the same warm start object a second time
======================================================================
  warm (2nd reuse of same ws object): status=Optimal obj=5232.12682726 iters=40 time=0.016s
```

* Same model: **3200 → 40 iterations (80×)**, same objective to 7 significant figures.
* 1 %-perturbed objective vector: 7600 → 4880 iterations (1.56×), objectives agree to `1.7e-08`.
* The object is **reusable** — a second solve from the same payload gives the identical result, so it
  is not consumed.
* Size mismatch raises clearly:

```
  n+10, m+0: Exception: Invalid PDLPWarmStart data. Passed problem and PDLPWarmStart data should have the same amount of variables.
  n+0, m+10: Exception: Invalid PDLPWarmStart data. Passed problem and PDLPWarmStart data should have the same amount of constraints.
```

### Only `pdlp_solver_mode=Stable2` honours it — the docstring is wrong about `Fast1`

`SolverSettings.set_pdlp_warm_start_data`'s docstring (`solver_settings.pyx:248`) says
*"Only supported solver modes are Stable2 and Fast1."* Measured:

```
F) Warm start under other solver methods / pdlp modes
  method=Concurrent: status=Optimal obj=5232.126827
  method=DualSimplex: status=Optimal obj=5232.127448
  method=Barrier: status=Optimal obj=5232.126037
  pdlp_solver_mode=Stable1     : status=NoTermination iters=     -1 obj=nan
  pdlp_solver_mode=Stable2     : status=Optimal    iters=     40 obj=5232.126827
  pdlp_solver_mode=Methodical1 : status=NoTermination iters=     -1 obj=nan
  pdlp_solver_mode=Fast1       : status=NoTermination iters=     -1 obj=nan
  pdlp_solver_mode=Stable3     : status=NoTermination iters=     -1 obj=nan
```

`Stable2` (=1) works. **`Fast1` returns `NoTermination` with `obj=nan`** — contradicting the
docstring. `Stable3` is the *default* `pdlp_solver_mode` (4), so warm start silently produces `nan`
unless the mode is changed too. `method=DualSimplex/Barrier/Concurrent` ignore the warm start
(they solved fine, just from cold).

Net: **usable warm start requires three non-default settings simultaneously** —
`method=1 (PDLP)`, `pdlp_solver_mode=1 (Stable2)`, `presolve=0`. That is a big enough ask that
"raise `NotImplementedError` for `warmstart_fn`, as cuPDLPx does (`solvers.py:4352-4355`)" is a
defensible first cut. If it *is* wired up, the three settings must be forced together and documented.

### MILP

```
G) get_pdlp_warm_start_data on a MILP solution
  AttributeError: Attribute get_pdlp_warm_start_data is not supported for milp solution
```

(`Solution.raise_if_milp_solution`, `solution.py:365`.) PDLP warm start is LP-only.

### 🔴 `set_initial_primal_solution` on an LP corrupts the CUDA context

Isolated in a fresh process with nothing else run first (`b_q11b_out.txt`):

```
==================== control ====================
reference solve: obj=1232.38787290 iters=1500 primal.shape=(500,) dual.shape=(250,) primal.flags.c_contiguous=True
get_initial_primal_solution -> <class 'numpy.ndarray'> (0,)
RESULT: status=Optimal obj=1232.3878729013568 iters=1500
follow-up plain solve: OK obj=1157.829590

==================== primal ====================
get_initial_primal_solution -> <class 'numpy.ndarray'> (500,)
RAISED: RuntimeError: __copy:: D->H: failed: cudaErrorIllegalAddress: an illegal memory access was encountered
follow-up plain solve: POISONED -> RuntimeError: CUDA error at: /__w/rmm/rmm/cpp/src/cuda_stream.cpp:20: cudaErrorIllegalAddress an illegal memory access was encountered

==================== dual ====================
get_initial_dual_solution   -> <class 'numpy.ndarray'> (250,)
RESULT: status=Optimal obj=1232.387939439497 iters=2100
follow-up plain solve: OK obj=1157.829590

==================== both ====================
RAISED: RuntimeError: __copy:: D->H: failed: cudaErrorIllegalAddress: an illegal memory access was encountered
follow-up plain solve: POISONED -> RuntimeError: CUDA error at: ...cuda_stream.cpp:20: cudaErrorIllegalAddress...

==================== mip ====================
get_initial_primal_solution -> <class 'numpy.ndarray'> (500,)
RAISED: ValueError: <MILPTerminationStatus.FeasibleFound: 8> is not a valid LPTerminationStatus
follow-up plain solve: OK obj=1157.829590
```

Also reproduced with **all-default settings** (default `method=0`, default `presolve=-1`):

```
ref obj 1232.387892867366
DEFAULT settings + set_initial_primal_solution: RAISED RuntimeError __copy:: D->H: failed: cudaErrorIllegalAddress: an illegal memory access was encountered
```

The arrays passed were `float64`, C-contiguous, correctly sized (500 = n) — see the printed
`primal.shape=(500,) … c_contiguous=True`. The MILP row's `ValueError` is my script casting a
`MILPTerminationStatus` through `LPTerminationStatus`; the MILP **solve itself succeeded** and the
context stayed healthy.

**Consequence: `linopy`'s cuOpt class must never call `DataModel.set_initial_primal_solution` on an
LP.** It is not a soft failure — every later `Solve()` in the process raises, so one warm-start
attempt takes down an entire pytest session or user script.

---

## 12. QP — how, not whether
Script `a_12_qp.py`, output `a_12_out.txt`; sign cross-checks in `a_03_offset_maximize.py` §3f and
`a_02i_m2_full.py`.

**Q-matrix format cuOpt expects:** CSR (`Q_values` float64, `Q_indices` int32, `Q_offsets` int32 of
length `n_variables + 1`), and the objective is

```
    minimize    cᵀx  +  xᵀ Q x            (plus objective_offset)
```

with the **effective Hessian `H = Q + Qᵀ`** — i.e. `obj = cᵀx + ½ xᵀ H x`. This matches the
docstring at `data_model/data_model.py:287-288` ("The input Q matrix is internally symmetrized as
Q + Q^T") and was confirmed empirically two independent ways. Consequences:

- **Full symmetric and triangular are BOTH accepted**, and give the same answer, because only
  `Q + Qᵀ` is used. What must be right is `Q + Qᵀ = H`.
- **There is no factor-of-2 on the diagonal to apply on top of that** — but there *is* one hiding
  in the linopy interface, because linopy's `Q` is already the Hessian (see below). Getting it wrong
  produces a **silently `Optimal`, silently wrong** objective.

**Proof:**

Test 1 — one variable, `f(x) = x² − 2x`, closed-form optimum `x* = 1`, `f* = −1`; `c = [-2]`,
`Q = [[q]]`. The three candidate conventions predict different answers:

```
#   obj = c'x +      x'Qx      ->  q=1 gives x=1.0, obj=-1.0
#   obj = c'x + 0.5  x'Qx      ->  q=1 gives x=2.0, obj=-2.0
#   obj = c'x +      x'(Q+Q')x ->  q=1 gives x=0.5, obj=-0.5

Q=[[1.0]] c=[-2]     status=Optimal      obj=-1.00000000  x=[1.]
     implied Hessian h from x*=c/-h : h = 2.000000 (Q+Q'=2.000, Q=1.000, 0.5Q=0.500)
Q=[[2.0]] c=[-2]     status=Optimal      obj=-0.50000000  x=[0.5]
     implied Hessian h from x*=c/-h : h = 4.000000 (Q+Q'=4.000, Q=2.000, 0.5Q=1.000)
Q=[[0.5]] c=[-2]     status=Optimal      obj=-2.00000000  x=[2.]
     implied Hessian h from x*=c/-h : h = 1.000000 (Q+Q'=1.000, Q=0.500, 0.5Q=0.250)
```

The implied Hessian equals `Q + Qᵀ` in all three rows → `obj = cᵀx + xᵀQx`.

Test 2 — two variables with a **cross term**: `f = x0² + x1² + x0x1 − 3x0 − 3x1`, optimum
`x* = (1,1)`, `f* = −3`, Hessian `H = [[2,1],[1,2]]`. Six candidate encodings of `Q`:

```
(a) Q = H            (full sym)                     status=Optimal obj=-1.50000000 x=[0.5 0.5]  [target -3 -> no]
(b) Q = 0.5*H        (full sym)                     status=Optimal obj=-3.00000000 x=[1. 1.]    [target -3 -> MATCH]
(c) Q = triu(H)                                     status=Optimal obj=-1.80000000 x=[0.6 0.6]  [target -3 -> no]
(d) Q = triu(0.5*H)                                 status=Optimal obj=-3.60000000 x=[1.2 1.2]  [target -3 -> no]
(e) Q = [[1,1],[0,1]] (tri, diag=coeff, offdiag=coeff) status=Optimal obj=-3.00000000 x=[1. 1.] [target -3 -> MATCH]
(f) Q = tril(0.5*H)                                 status=Optimal obj=-3.60000000 x=[1.2 1.2]  [target -3 -> no]
```

Exactly the two encodings with `Q + Qᵀ = H` match: `(b) 0.5·H` and
`(e) strictly-upper(H) + ½·diag(H)`. Note how badly the near-misses fail *silently*: `(a)` (the
naive "pass the Hessian") returns `Optimal` with objective `−1.5` instead of `−3`.

Test 3 — live cross-check against HiGHS through linopy's own `model.matrices.Q`:

```
linopy/HiGHS objective : -4.285714285714258
linopy/HiGHS solution  : [ 1.57142848 -0.14285712 -1.9999998 ]
M.c   = [-3. -1.  2.]
M.Q (dense, == Hessian per linopy docstring) =
 [[2. 1. 0.]
  [1. 4. 0.]
  [0. 0. 1.]]
  Q_cuopt = 0.5 * M.Q (full sym)           status=Optimal obj=-4.28571429 x=[ 1.571429 -0.142857 -2.      ]  MATCHES HiGHS
  Q_cuopt = M.Q            (full sym)      status=Optimal obj=-2.14285714 x=[ 0.785714 -0.071429 -1.      ]
  Q_cuopt = triu(M.Q)                      status=Optimal obj=-2.12903226 x=[ 0.741935  0.032258 -1.      ]
  Q_cuopt = triu(0.5*M.Q)                  status=Optimal obj=-4.25806452 x=[ 1.483871  0.064516 -2.      ]
```

**Mapping from linopy's `model.objective` quadratic form:**

`linopy.expressions.QuadraticExpression.to_matrix` (`expressions.py:2945-2970`) states its own
convention in its docstring: *"the quadratic term is 0.5 x^T Q x. The matrix Q is therefore
symmetric and the diagonal terms are doubled."* So **`model.matrices.Q` IS the Hessian `H`** (it is
a `csc_matrix`; `matrices.py:161-170`), which is also why HiGHS is fed `triu(M.Q)` via
`passHessian` (`solvers.py:1718-1720`). Therefore:

```python
Q = M.Q                              # csc, == Hessian H, full symmetric
Qc = scipy.sparse.csr_matrix(0.5 * Q)
dm.set_quadratic_objective_matrix(Qc.data, Qc.indices, Qc.indptr)
```

**`Q_cuopt = 0.5 * model.matrices.Q`, kept full-symmetric, converted to CSR.** Passing `M.Q`
directly halves the solution and returns a wrong objective with status `Optimal` — this is the trap
the charter warned about, and it is real. Under mitigation M2 (§2) a `max`-sense QP also needs
`Q → -Q` alongside `c → -c` (validated 4/4 QP cells in `a_02i_out.txt`).

**PSD requirement / behaviour on non-PSD Q:** `Q + Qᵀ` must be PSD (for minimisation). A non-PSD
`Q` is detected and reported as **`NumericalError`** with `obj=nan` — it does not crash and does not
return a bogus answer:

```
min with Q=[[-1]] (concave, non-PSD)           status=NumericalError obj=+nan  x=[0.]
min with Q=diag(1,-1) (indefinite)             status=NumericalError obj=+nan  x=[0. 0.]
min with asymmetric Q=[[1,3],[-1,1]] (Q+Q' PSD) status=Optimal      obj=-0.25000000  x=[0.25 0.25]
    Q+Q' = [[2. 2.]
            [2. 2.]]
```

The third row is the important one: a **wildly asymmetric** `Q` is accepted so long as `Q + Qᵀ` is
PSD — further confirmation that only `Q + Qᵀ` is used. For `max`, the requirement inverts as
expected (`max, c=[2], Q=[[-1]] -> Optimal obj=+1.00000000 x=[1.]` matching the closed form;
`max, c=[2], Q=[[+1]] -> NumericalError`). Under M2 the `-Q` negation preserves this.

**Are duals returned under QP?** **Yes**, and they match HiGHS in all six `min`/`max` × `<=`/`>=`/`==`
cells (pasted in §3 above). `get_reduced_cost()` also returns values under QP. Two caveats:

```
cuOpt QP solved_by : Barrier
cuOpt QP duals     : [ 1.07473390e-10 -1.30209876e-10]
HiGHS  QP duals    : [0. 0.]
cuOpt lp_stats     : {'primal_residual': 1.06e-10, 'dual_residual': 51.00000000023794, 'gap': 0.0, 'nb_iterations': 12}
```

- Non-binding QP duals come back as `~1e-10` rather than exact `0`, so an `atol` is required, not
  just an `rtol`.
- **`lp_stats['dual_residual']` is meaningless under QP** (`51.0` on a solved-to-optimality model).
  Do not surface or assert on it for QP.

**Crossover unsupported for QP — confirmed?** **Not confirmed as an error — it is silently
irrelevant.** Setting `crossover` either way, and forcing any `method`, never raises and never
changes the answer; every QP is solved by **Barrier** regardless:

```
crossover=True  method=Concurrent   -> Optimal      obj=-3.00000000 solved_by=Barrier
crossover=True  method=Barrier      -> Optimal      obj=-3.00000000 solved_by=Barrier
crossover=True  method=PDLP         -> Optimal      obj=-3.00000000 solved_by=Barrier
crossover=True  method=DualSimplex  -> Optimal      obj=-3.00000000 solved_by=Barrier
crossover=False method=Concurrent   -> Optimal      obj=-3.00000000 solved_by=Barrier
crossover=False method=Barrier      -> Optimal      obj=-3.00000000 solved_by=Barrier
crossover=False method=PDLP         -> Optimal      obj=-3.00000000 solved_by=Barrier
crossover=False method=DualSimplex  -> Optimal      obj=-3.00000000 solved_by=Barrier
```

So `method` and `crossover` are **silently overridden** for QP. A user passing `method="PDLP"` with
a quadratic objective gets Barrier and no warning. Practical upshot: QP never takes the buggy
presolve-only dual path (§2a), which is why all four QP cells pass even without M2 — but M2 is still
required for LP/MILP and does no harm to QP.

**MIQP (quadratic objective + integer variables) is NOT supported and must be rejected up front.**
It does not raise; it returns `NoTermination`, `obj=0.0`, an **empty** solution array, and
`error_status=4`:

```
MIQP, Q=0.5H, integer vars                     status=NoTermination obj=+0.00000000  x=[]
        problem_category = MIP
MIQP  status= 0 NoTermination  err=4 obj=+0
      msg='{"CUOPT_ERROR_TYPE": "ValidationError", "msg": "Presolve does not support optimization_problem with a quadratic objective"}'
```

linopy's `model.type` is `"MIQP"` for such a model, so the guard is cheap. `SolverFeature.QUADRATIC_OBJECTIVE`
must therefore be declared **without** implying MIQP support, and `_build_direct` must raise
`NotImplementedError` for `model.type == "MIQP"`.

**What `problem.Problem` offers for QP:** a working but matrix-hostile path. It has no
`set_quadratic_objective_matrix` equivalent; the objective must be built from
`Variable * Variable` products into a `QuadraticExpression`, which `setObjective`
(`problem.py:1818-1839`) turns into a `scipy.sparse.coo_matrix` and then CSR — so the same
convention applies, but you have to synthesise `n_terms` Python objects to express a matrix linopy
already has. It does solve correctly and reports duals and reduced costs (TEST 9 output pasted in
Q0 above). It also exposes `add_quadratic_constraint` / `getQuadraticConstraints`
(`data_model.py:309-365`) for quadratically-constrained problems — `'L'`/`'G'` only, equality
explicitly rejected (`raise ValueError("Equality constraints are not supported.")`,
`data_model.py:349-350`) — which linopy has no model surface for and which is out of scope.

## 13. Installability from a clean venv
**Artifacts:** `b_q13_pyproject.diff`, `b_q13_install.txt` (267 lines, full pip transcript),
`b_q13_solvers_report.json` + `b_q13_solvers_resolution.txt`, `b_q13_crossplatform.txt`,
`b_q13_verify.txt`, `b_q13b_nogpu.py` → `b_q13b_out.txt`

**Candidate extra:** `linopy[cuopt]` — contents: `cuopt-cu12>=26.8` under a Linux platform marker.

Exact TOML, inserted immediately after the `solvers = [...]` block (`pyproject.toml:113`), i.e. a
**sibling** of `solvers`, reachable from no other extra (there is no `all` extra in this project):

```toml
# NVIDIA cuOpt (GPU). Deliberately NOT part of `solvers`: it pulls ~34 packages
# and several GB of CUDA wheels. Linux-only wheels, no sdist, so the marker keeps
# macOS/Windows resolution unaffected. Requires a CUDA 12 driver (>= 525.60.13)
# and a GPU of compute capability >= 7.0.
cuopt = [
    "cuopt-cu12>=26.8; platform_system == 'Linux'",
]
```

**Naming note for the planners.** I used `cuopt`, matching the solver name, the `SolverName` value,
and the file's existing lower-case single-word extras (`oetc`, `remote`, `docs`, `dev`, `solvers`,
`benchmarks`). The alternative the charter hints at (`gpu`) reads better as a *category* and would
survive a future second GPU solver, but it is a promise the extra cannot keep — it would install
cuOpt specifically, on Linux only, for CUDA 12 only. Charter §5 writes `<gpu-extra>` as a
placeholder, not a name. My recommendation is **`cuopt`**; this is a D-entry for Gate 3, not
something I decided.

**Marker style.** `platform_system == 'Linux'` is the positive twin of the file's established
`"cplex; platform_system != 'Darwin'"` (`pyproject.toml:105`) — it must be an equality, not
`!= 'Darwin'`, because cuOpt has no Windows wheels either. No `python_version` marker is needed:
cuOpt ships cp311-cp314 and linopy already declares `requires-python = ">=3.11"` (§4.2).

### Step 1-2 — clean-venv install, **no extra index, no resolver flags**

Worktree cut from `feat/cuopt-solver` (`09c34dd`), `pyproject.toml` edited, **never committed**
(`git status --short` → ` M pyproject.toml`); venv created with `uv venv --python 3.11`, real pip
installed into it, cuOpt confirmed absent first:

```
$ .venv-q13/bin/pip --version
pip 26.2.1 from /home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-q13/lib/python3.11/site-packages/pip (python 3.11)
$ .venv-q13/bin/python -c "import cuopt"
ModuleNotFoundError: No module named 'cuopt'
```

No index configuration anywhere that could smuggle in `pypi.nvidia.com`:

```
$ .venv-q13/bin/pip config list          # (empty)
$ env | grep -iE "PIP_|UV_|INDEX"        # (none)
$ cat ~/.config/pip/pip.conf /etc/pip.conf   # (no such files)
$ grep -n "\[tool.uv\]" <worktree>/pyproject.toml   # (no [tool.uv] in pyproject)
```

The command, verbatim:

```
.venv-q13/bin/pip install --no-cache-dir \
  '/home/azureuser/linopy-gpu/dev-scripts/cuopt/worktrees/q13-install[cuopt]'
```

(`--no-cache-dir` is a *cache* flag, not a resolver or index flag — used only because the machine
had 18 GB free and the wheel cache would have doubled the footprint. No `--extra-index-url`, no
`--index-strategy`, no `--find-links`, no `--pre`.)

Result — `exit=0`:

```
Successfully installed bottleneck-1.6.0 cachetools-7.1.7 click-8.4.2 cloudpickle-3.1.2
cuda-bindings-12.9.7 cuda-core-1.1.1 cuda-pathfinder-1.6.1 cuda-toolkit-12.9.2.0 cudf-cu12-26.8.0
cuopt-cu12-26.8.0 cupy-cuda12x-14.2.0 dask-2026.7.1 deprecation-2.1.0 fsspec-2026.7.0
importlib_metadata-9.0.0 libcudf-cu12-26.8.0 libcuopt-cu12-26.8.0 libkvikio-cu12-26.8.0
libraft-cu12-26.8.0 librmm-cu12-26.8.0 linopy-0.9.1.post1.dev3+g09c34dd9d.d20260822 llvmlite-0.46.0
locket-1.0.0 markdown-it-py-4.2.0 mdurl-0.1.2 numba-0.64.0 numba-cuda-0.30.4 numba-cuda-mlir-0.5.0
numexpr-2.14.2 numpy-2.4.6 nvidia-cublas-cu12-12.9.2.10 nvidia-cuda-cccl-cu12-12.9.27
nvidia-cuda-nvcc-cu12-12.9.86 nvidia-cuda-nvrtc-cu12-12.9.86 nvidia-cuda-runtime-cu12-12.9.79
nvidia-cudss-cu12-0.7.1.6 nvidia-cufft-cu12-11.4.1.4 nvidia-cufile-cu12-1.14.1.1
nvidia-curand-cu12-10.3.10.19 nvidia-cusolver-cu12-11.7.5.82 nvidia-cusparse-cu12-12.5.10.65
nvidia-libnvcomp-cu12-5.3.0.16 nvidia-nccl-cu12-2.31.2 nvidia-nvjitlink-cu12-12.9.86
nvidia-nvtx-cu12-12.9.79 nvtx-0.2.16 packaging-26.3 pandas-3.0.3 partd-1.4.2 polars-1.43.2
polars-runtime-32-1.43.2 pyarrow-23.0.1 pygments-2.21.0 pylibcudf-cu12-26.8.0 pylibraft-cu12-26.8.0
python-dateutil-2.9.0.post0 pyyaml-6.0.3 rapids-logger-0.2.3 rich-15.0.0 rmm-cu12-26.8.0
scipy-1.17.1 six-1.17.0 toolz-1.1.0 tqdm-4.70.0 typing-extensions-4.16.0 xarray-2026.7.0 zipp-4.1.0
exit=0
```

Note `pandas-3.0.3` — the §4.4 ceiling from `cudf-cu12` reproduces exactly in a clean venv, so it is
an upstream property, not an artefact of our dev venv. Also note pip **built `libcuopt-cu12` from a
591 MB wheel** it created locally (`Created wheel for libcuopt-cu12: … size=591670555`) — pip does
this for the `-cu12` metapackage indirection; it does not need an sdist toolchain for cuOpt itself.

### Step 2 (verification) — both asked-for probes

```
### python -c 'import linopy; print(linopy.available_solvers)' ###
0.9.1.post1.dev3+g09c34dd9d.d20260822
[]

### python -c 'import cuopt; print(cuopt.__version__)' ###
26.08.00
```

`available_solvers` is `[]` — **expected and correct for today's tree**: this branch has no `cuOpt`
solver class and no `"cuopt"` entry in `_SOLVER_PROBE_ORDER` (`solvers.py:4443-4457`), and the venv
contains no other solver either. `import cuopt` → `26.08.00` proves the extra delivered a working
package. Re-run this check after the `solvers.py` work package lands; it is the acceptance test for
that slice.

And cuOpt genuinely *works* from this from-scratch venv on the T4 (the charter §4.3 model, solved
through the raw API inside `.venv-q13`):

```
### end-to-end: solve the charter 4.3 LP through the raw cuOpt API in this fresh venv ###
status 1 obj 2.8000000000000003 x [1.5999999999999999, 1.2000000000000002] duals [0.40000007091852546, 0.20000008176974685]
```

Matches §4.3 exactly (`x=[1.6, 1.2]`, obj `2.8`, duals `[0.4, 0.2]`, status `1`). Installed size:
**6.7 GB** (`du -sh` on the venv) — consistent with §4.4's 6.3 GB estimate.

### Step 3 — `linopy[solvers]` pulls **no** cuOpt/cuDF/CUDA

`pip install --dry-run --report` in the clean venv, then filtered:

```
pip install 'linopy[solvers]' would install 34 packages:
  Bottleneck==1.6.0
  Mosek==11.2.3
  PySCIPOpt==6.2.1
  PyYAML==6.0.3
  click==8.4.2
  cloudpickle==3.1.2
  coptpy==8.0.6
  cplex==22.2.0.1
  dask==2026.7.1
  deprecation==2.1.0
  fsspec==2026.7.0
  gurobipy==13.0.2
  highspy==1.15.1
  importlib_metadata==9.0.0
  knitro==16.0.0
  linopy==0.9.1.post1.dev3+g09c34dd9d.d20260822
  locket==1.0.0
  mindoptpy==2.3.0
  numexpr==2.14.2
  numpy==2.4.6
  packaging==26.3
  pandas==3.0.5
  partd==1.4.2
  polars-runtime-32==1.43.2
  polars==1.43.2
  python-dateutil==2.9.0.post0
  scipy==1.17.1
  six==1.17.0
  toolz==1.1.0
  tqdm==4.70.0
  xarray==2026.7.0
  xpress==9.9.1
  xpresslibs==9.9.1
  zipp==4.1.0

GPU/CUDA-related packages present: NONE
```

The filter tested for `cuopt`, `cudf`, `libcuopt`, `rmm`, `pylibraft`, `nvidia`, `cuda`, `cupdlpx`.
Note `pandas==3.0.5` here vs `3.0.3` above — further confirmation that the downgrade is caused
solely by the cuOpt tree and does not leak into `solvers`.

### Step 4 — macOS / Windows resolution unaffected

```
$ uv pip compile --no-cache --python-platform windows --python-version 3.11 --extra cuopt <wt>/pyproject.toml
Resolved 25 packages in 196ms      exit=0     # no cuopt-cu12, no nvidia-*; adds colorama, tzdata

$ uv pip compile --no-cache --python-platform macos   --python-version 3.11 --extra cuopt <wt>/pyproject.toml
Resolved 23 packages in 190ms      exit=0     # no cuopt-cu12, no nvidia-*

$ uv pip compile --no-cache --python-platform linux   --python-version 3.11 --extra cuopt <wt>/pyproject.toml
Resolved 66 packages in 460ms      exit=0     # cuopt-cu12==26.8.0, libcuopt-cu12==26.8.0,
                                             # cudf-cu12==26.8.0, pylibraft-cu12, rmm-cu12,
                                             # nvidia-* ... , pandas==3.0.3
```

All three succeed. Full resolutions in `b_q13_crossplatform.txt`. This is the CI matrix check
(`.github/workflows/test.yml` runs ubuntu/macos/windows-latest) and it passes without any marker
gymnastics beyond the single `platform_system == 'Linux'`.

### ⚠️ §4.6 constraint 3 ("degrades honestly") is **not** satisfied by an import probe

Not asked for, but it falls out of this item and the plan needs it. `b_q13b_out.txt`:

```
########## with GPU visible ##########
import cuopt              -> OK, version 26.08.00
_has_module("cuopt")      -> True
solve tiny LP             -> OK obj = 2.8000000000000003
########## CUDA_VISIBLE_DEVICES='' ##########
import cuopt              -> OK, version 26.08.00
import cuopt.linear_programming -> OK
import submodules         -> OK
_has_module("cuopt")      -> True
solve tiny LP             -> FAILED InputValidationError: A CPU-memory problem requires remote execution. Set CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT to solve on a remote GPU server.
########## CUDA_VISIBLE_DEVICES='-1' ##########
  (identical)
```

`cuopt/__init__.py` is deliberately GPU-free — its own comment says *"Lazy imports … allows cuopt to
be imported on CPU-only hosts when remote solve is configured"* — so **`_has_module("cuopt")`
returns `True` on a machine with no usable GPU**, and `available_solvers` would list `cuopt`. §4.6
constraint 3 says it "must simply omit cuOpt". That is achievable (put a real device probe in
`is_available`) but it **departs from repo precedent**: `cuPDLPx.is_available` is
`_has_module("cupdlpx")` (`solvers.py:4188-4189`) and `_AvailableSolvers`' own docstring says
*"Membership means the solver's Python package or binary is importable — it does not mean a working
license exists"* (`solvers.py:4464-4470`). This is an I5-vs-§4.6 fork for Gate 3, and the error
message a GPU-less user currently gets ("Set CUOPT_REMOTE_HOST…") is *not* the actionable message
§4.6 asks for.

### Verdict

**One-command install works.** `pip install 'linopy[cuopt]'` from a venv that had never seen cuOpt,
with no extra index and no resolver flags, installed a working `cuopt-cu12==26.8.0` that solves on
the T4; `linopy[solvers]` pulls nothing CUDA-related; macOS and Windows resolution both succeed with
cuOpt absent. The only open sub-item is the honest-degradation question above, which is a
*code* question (where the GPU probe lives), not a packaging one.

### Cleanup performed (charter's disk constraint)

```
$ rm -rf .../dev-scripts/cuopt/.venv-q13
$ git worktree remove --force .../dev-scripts/cuopt/worktrees/q13-install
$ git branch -D wp/q13-install
Deleted branch wp/q13-install (was 09c34dd).
--- worktrees ---
/home/azureuser/linopy-gpu  09c34dd [feat/cuopt-solver]
--- branches ---
  backup/pre-upstream-sync-master-2026-08-22
* feat/cuopt-solver
  master
--- main checkout status ---            (clean)
/dev/root       119G  101G   18G  86% /
```

No tracked file in the main checkout was touched.

---

## Unknowns
### Agent A scope
- **`LPTerminationStatus.DualInfeasible` (3) — never observed.** Seven unbounded-LP variants all
  returned `11 UnboundedOrInfeasible`: default settings, `infeasibility_detection=False`,
  `strict_infeasibility=True`, an unbounded model with a binding row, and `presolve` at every
  setting I could pass. `UNKNOWN — cannot force status 3; the proposed `unbounded` mapping is
  therefore untested. (`presolve=False` raises `ValueError: Parameter presolve value False is not an
  integer`; `presolve=0`/`1` were tried in `a_02e`/`a_02g` and neither produced status 3 either.)`
- **`MILPTerminationStatus.Unbounded` (3) — never observed.** Same story: `11` every time, including
  a 2-variable MILP with one bounded and one unbounded variable.
  `UNKNOWN — cannot force MILP status 3.`
- **The root cause of the `max` + presolve dual-sign defect.** The correlation with
  `solved_by == Unset` is 6/6 exact over 144 cells, and the log line is
  `Optimal solution found during presolve.` (Papilo presolve), but I did not read C++ source — the
  `.so` is the only artefact. `UNKNOWN — whether the bug is in Papilo's dual postsolve or in cuOpt's
  sense handling around it; whether it also affects some non-presolve path I did not construct.`
  M2 avoids the whole branch, which is why I recommend it over any narrower workaround.
- **A single transformation for `get_reduced_cost()` under `max`.** The `max`/`<=` cells return
  `−c − Aᵀy` and the `max`/`>=` cells return the correct `c − Aᵀy`; I could not find one rule
  covering both. `UNKNOWN — swept c_2 over four values × four methods × three row senses
  (a_02c_rc_probe.py) and found no consistent single formula.` Moot for linopy (no RC surface) but
  recorded so nobody "fixes" it by guessing a sign.
- **Whether cuOpt's int32 CSR indices can be overflowed in practice.** `type_cast` casts int64→int32
  with no warning and no range check, but a >2³¹-nonzero model does not fit in the T4's 16 GB.
  `UNKNOWN — not testable on this hardware; recorded as a latent risk only.`
- **Whether `'B'` is a distinct variable type from `'I'`.** `np.array(['B','B'])` produced
  `cat=MIP` and the same objective as `'I'` on a model with `ub=10`, so `'B'` did **not** clamp to
  `[0,1]`. `UNKNOWN — whether 'B' is a documented code that happens to alias 'I' here, or an
  unrecognised code that `is_mip()` nonetheless classifies as a MIP.` Either way, map linopy's
  `'B'` to `'I'` and set `lb=0, ub=1` explicitly.

### Agent B scope
* **UNKNOWN — the true GPU-OOM failure mode.** I characterised behaviour *at* the ceiling
  (peak 14231 / 15948 MiB → status `NoTermination`, memory returned afterwards) but never provoked
  an actual `MemoryError`/`std::bad_alloc` from a legitimately oversized model. cuOpt absorbed
  60 M variables / 30 M constraints / 240 M nonzeros without raising; pushing past that needed host
  RAM I judged imprudent on a GPU shared with a sibling agent. Tried: three escalating sizes with a
  20 ms peak sampler (`b_q9g_peak.txt`).
* **UNKNOWN — the root cause of the `method=0` (Concurrent) repeat-solve SIGSEGV.** Characterised
  precisely (deterministic, n ≳ 1300, 2nd-3rd solve, insensitive to `num_cpu_threads`/`num_gpus`,
  absent under `method=1/2/3`) but not diagnosed: `faulthandler` only reaches
  `solver.py:102 in Solve`, and there are no debug symbols in the shipped `.so`. Tried:
  `faulthandler`, method matrix, size sweep, subprocess-vs-not, same-model-vs-different,
  `presolve=0`. Not tried: `gdb`/`cuda-gdb` on the stripped binary, or an upstream issue search.
* **UNKNOWN — whether `set_initial_primal_solution`'s LP crash has a working input shape.** I proved
  it crashes with a correctly-sized, C-contiguous `float64` array under both default and
  `presolve=0`/`method=1` settings, but did not try device arrays (`cupy`/`__cuda_array_interface__`),
  `float32`, or a fresh (non-round-tripped) vector. The safe conclusion — do not call it — holds
  regardless.
* **UNKNOWN — whether `log_file` and `solution_file` interact.** `solution_file` and
  `user_problem_file`/`presolve_file` exist as parameters and might serve linopy's `solution_fn`, but
  I only exercised `log_file`. That belongs with whoever owns the `io_api` question (Q0).
* **UNKNOWN — precedence when a user passes `solver_options={"log_file": ...}` *and* `log_fn`.**
  Mechanically both write to `settings_dict`, so last-writer-wins; which one linopy should let win is
  a plan decision, not a measurement.

## Surprises
### Agent A scope
Nothing here contradicts charter §4. Everything in §4 that this agent exercised held up: the §4.3
snippet reproduced exactly (LP `x=[1.6,1.2]`, obj `2.8`, duals `[0.4,0.2]`, status `Optimal`; MILP
obj `2.0`), the concurrent dual-simplex/barrier/PDLP architecture is visible in `get_solved_by()`,
and `cuopt.__version__` is `26.08.00` on the T4 with a CUDA 12.9 runtime, as §4.1/§4.2 state. Four
things surprised *me*, none of them a §4 contradiction:

1. **A silent dual-sign defect in cuOpt 26.08.00 for maximised models finished by presolve** (§2a).
   This is the exact failure mode charter §6.1 item 2 was written to catch, and the six-cell matrix
   *alone would have missed it* — all six cells pass on a model whose presolve does not close it.
   It needed a second model shape and the `solved_by` field to surface. **Recommendation for the
   verification plan: the V1 sign matrix must include at least one presolve-solvable `max` model
   (a square equality system, and a ranged row) in addition to the six canonical cells, and should
   assert on `get_solved_by()`.** Mitigation M2 is verified 21/21; `presolve=0/1` is verified to be
   an incomplete fix and should not be adopted.

2. **cuOpt's dual convention is the *opposite* of the repo's own template.** cuPDLPx negates duals
   for `max` (`solvers.py:4385-4386`: `if cu_model.ModelSense == cupdlpx.PDLP.MAXIMIZE: dual = -dual`).
   cuOpt must **not**. An implementer following I5 ("match repo precedent") by copying cuPDLPx's
   `_solve` line-for-line would introduce a systematic sign error in every maximised model. This is
   a trap that should be called out in the plan and in a code comment.

3. **A deterministic interpreter SEGFAULT and a non-terminating solve, both from ordinary parameter
   combinations.** `method=PDLP` + `time_limit=1e-6` + `save_best_primal_so_far=True` segfaults the
   process (exit 139, 3/3 attempts; `a_01b_segfault_repro.py`) — each parameter alone is fine.
   Separately, a badly-scaled LP (`1e18` coefficients) with `method=PDLP` and no time limit did not
   terminate: it held the T4 at 98% utilisation for ~13 minutes and was killed by `timeout 900`
   (`a_01c_out.txt` stops mid-run). With `time_limit=10` the same model returns `TimeLimit`, and
   `DualSimplex`/`Barrier`/`Concurrent` all solve it to `Optimal obj=-2` in seconds
   (`a_01d_out.txt`). Implications: never expose `save_best_primal_so_far` unguarded, and consider
   whether a default `time_limit` is warranted.

4. **`NoTermination` is cuOpt's silent catch-all for malformed input** (§5, §6): no constraint
   matrix, a zero-nnz matrix, any wrong-length array, or a MIQP all return `NoTermination` with
   `obj=nan`-or-`0` and an **empty** primal array, without raising. `get_error_status()` /
   `get_error_message()` sometimes carry a JSON explanation and sometimes do not. Every guard on the
   linopy side has to be a pre-check, because the post-check is indistinguishable from "solver
   failed".

Also worth noting, though minor: the installed docstring at `data_model.py:379-381` claims the
default variable lower bound is `0`, and empirically it is not (§5, last row) — a cuOpt
documentation error, harmless for linopy because `M.lb`/`M.ub` are always passed.

### Agent B scope
Nothing here **contradicts** charter §4 — §4.1-§4.6 all held up, and §4.4's pandas ceiling and
§4.3's LP answer both reproduced exactly in a from-scratch venv. But five things §4 did not
anticipate, in descending order of consequence:

1. **🔴 cuOpt's default solver method (`method=0`, Concurrent) segfaults on repeat solves.**
   Deterministic above ~1300 variables, on the 2nd-3rd `Solve()` in one process. §4.3 celebrates
   that "cuOpt runs concurrent dual simplex + barrier + PDLP, not PDLP alone" as a feature; that
   code path is the one that crashes. linopy's tiny test models sit *below* the threshold, so the
   suite would go green while real user scripts die with SIGSEGV and no traceback. Full evidence in
   §9. This needs a plan decision (default `method`, a guard, a regression test above the threshold),
   and it is the single most consequential thing I found.
2. **🔴 `DataModel.set_initial_primal_solution` on an LP raises `cudaErrorIllegalAddress` and
   poisons the CUDA context for the rest of the process.** Reproduced in a fresh process with
   all-default settings. Any code path that touches it takes down the whole session. §11.
3. **Semi-continuous variables ARE supported** (`VType.SEMI_CONTINUOUS = 'S'`, verified 0.0 vs 5.0
   on a discriminating model, on both entry points). §6.1's item 10 asks us to "confirm each is
   genuinely absent"; for semi-continuous the honest answer is the opposite. Copying cuPDLPx's
   semi-continuous `NotImplementedError` guard would be a *false* claim. Related: bogus variable-type
   characters are silently accepted as continuous, with no validation anywhere. §10.
4. **Warm-start data is returned in presolved coordinates**, so it cannot be fed back into the same
   model unless `presolve=0` — and it only takes effect under `pdlp_solver_mode=Stable2`, not the
   default `Stable3` and not `Fast1` (which the upstream docstring claims is supported, but which
   returns `obj=nan`). Three non-default settings must line up for warm start to do anything. §11.
5. **`import cuopt` succeeds with no GPU** — by design, per `cuopt/__init__.py`'s own comment — so
   an import-probe `is_available()` would advertise cuOpt on GPU-less machines, and the resulting
   error is "Set CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT to solve on a remote GPU server". That
   collides with §4.6 constraint 3, while matching `cuPDLPx.is_available` and the documented
   semantics of `available_solvers`. A real I5-vs-§4.6 fork for Gate 3. §13.

Two smaller notes: **`log_file` makes an OS-level `dup2` redirect unnecessary** (§6.1 item 4
anticipated needing one) — cuOpt's parameter is even spelled identically to HiGHS's option; and
importing **`cuopt.routing` installs a global `sys.excepthook`** that writes `error_log.txt` into the
cwd (`routing/vehicle_routing_wrapper.pyx:144-160`). `import cuopt` and
`import cuopt.linear_programming` do **not** trigger it — verified — so linopy simply must never
import `cuopt.routing`. Worth a comment next to the `_LazyModule` line so nobody "helpfully" widens
the import later.
