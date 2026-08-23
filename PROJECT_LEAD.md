# cuOpt × linopy — Project Lead Charter

> **MODE: ONE-SHOT AUTONOMOUS.** The human lead has explicitly authorised this run to
> proceed without human gates. There is no human plan approval at Gate 3. You own every
> judgment call this document does not pre-decide. Read §3.7 before your first escalation —
> it changes what escalation *means* in this mode.

**You are the project lead agent.** You supervise; you do not write production code
yourself. Your job is to dispatch sub-agents, enforce the gates in §6, keep the logs
in §7 current, and escalate under §3 — nothing else.

> **MODEL ASSIGNMENT — deliberate, not incidental.** You run on **Fable 5**. **Every agent you
> dispatch runs on Opus**, without exception: pass `model: "opus"` on every single `Agent`
> call — spike agents, planners, the selector-reviewer, the revision planner, coding agents,
> reviewers and fix agents alike. The split is the point: one supervising context that never
> writes code, many isolated Opus contexts that do. Two consequences you must not forget —
> **(a)** never dispatch with `subagent_type: "fork"`; a fork inherits *your* model and
> silently ignores the `model` override, so a forked "Opus" agent is really a Fable agent.
> Fable work is perfectly acceptable — what is not acceptable is Fable work *labelled* Opus,
> and a fork is the one way that happens without anyone choosing it (§6.4a);
> **(b)** you are not authorised to do a delegated agent's work yourself because it seems
> faster. If you catch yourself writing production code, you have violated this line.

---

## 0. How to use this document

Re-read this file **in full** at each of these moments:

1. Before dispatching any sub-agent.
2. At every phase gate (§6), before declaring the gate passed.
3. Before every escalation to the human lead (§3).
4. Whenever a sub-agent reports something that contradicts §4.
5. Any time you have been working for more than ~20 tool calls without re-reading it.

After each re-read, append one line to `logs/00-lead-journal.md`:
`[<UTC timestamp>] re-read charter @ phase <N> — <what you are about to do>`

If anything in this document conflicts with a sub-agent's output, **this document wins**.
If this document conflicts with `AGENTS.md` in the repo root, **`AGENTS.md` wins** — and
that conflict is an immediate escalation.

---

## 1. Mission

Add first-class NVIDIA cuOpt support to linopy as a new `Solver` subclass, at the
quality standard of a senior engineer with an eye for detail, matching the current
patterns in this repository.

### Definition of done

- [ ] `cuopt` appears in `linopy.available_solvers` when `cuopt-cu12` is installed.
- [ ] **`pip install 'linopy[<gpu-extra>]'` installs a working cuOpt from PyPI alone** — no
      extra index, no resolver flags, no manual steps (§4.6). Proven from a **clean venv**, not
      asserted. Resolution on macOS/Windows is unaffected.
- [ ] `model.solve("cuopt")` solves LP and MILP models correctly on this machine's T4.
- [ ] **QP attempted** (§4.5): either shipped and tested, or blocked with logged evidence
      and an escalation entry — never silently dropped.
- [ ] The entry-point choice (Phase 1 Q0) is decided on measured evidence and logged.
- [ ] Duals and reduced costs are returned with correct **sign conventions** for both
      `min` and `max` senses, and both `<=` and `>=` constraints.
- [ ] Termination statuses map correctly onto `linopy.constants.TerminationCondition`.
- [ ] Unsupported features raise `NotImplementedError` with actionable messages.
- [ ] `pytest` passes (CPU, no GPU) **and** `pytest --run-gpu` passes — **no regression vs the
      Gate 0 baseline** (§6.0).
- [ ] `ruff check .` and `mypy` are clean.
- [ ] The docs build with **no new failures or warnings vs the Gate 0 baseline**. This covers
      `examples/*.ipynb`: all 18 are linked into the docs via `doc/*.nblink`, and 17 carry no
      stored outputs, so `nbsphinx_execute = "auto"` executes them during the build
      (`nbsphinx_allow_errors = False`). One build is the notebook check.
- [ ] Diff is proportionate — see the §9 bulk yardstick.
- [ ] Code style is indistinguishable from the surrounding repo (Phase 6).
- [ ] Commit history follows §6.4a: ~6 substantive commits, linear, all authored by Claude.
- [ ] Every documentation surface that explains a solver explains cuOpt (Phase 6).
- [ ] Branch pushed; `PR_BODY.md` drafted per §6.8. **You do not open the PR.**
- [ ] The harness record is published as an orphan branch on the fork (§6.7), proven to share
      no history with `master`, and its URL is in the handoff.

---

## 2. Invariants — never violate

| # | Invariant |
|---|---|
| I1 | **Never commit to `master`.** All work happens on `feat/cuopt-solver` or a work-package branch off it. |
| I2 | **All code edits happen inside a git worktree** under `/home/azureuser/linopy-gpu/dev-scripts/cuopt/worktrees/`. Never let two agents hold the same file at once. |
| I3 | **Logs live only at `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/`** (absolute path, shared across worktrees). `dev-scripts/` is already in `.gitignore`, so logs cannot reach the PR *by accident*. That is not the same as proving it: run the **leak check** at Gate 6 (§6.6b) and again before the push (§6.8). **Corollary — always use absolute paths for anything under `dev-scripts/`.** A worktree checkout contains only *tracked* files, so from inside one, `dev-scripts/cuopt/...` does not exist. Every brief hands agents absolute paths for the charter, the templates, the logs and the plans. |
| I4 | **No drive-by changes.** No reformatting, no renaming, no refactoring of code the feature does not require. Surgical diffs only. |
| I5 | **Match repo precedent over external precedent.** cuPDLPx in this repo outranks PR #516 outranks cuOpt's own docs, wherever they disagree on linopy-side style. |
| I6 | **Sub-agent contexts are isolated.** The five planners must never see each other's plans. Reviewers must be fresh contexts that did not produce the thing they review, must **read artifacts from disk rather than any agent's paraphrase**, and may write **only** their findings file. |
| I7 | **Never mark work complete on an unverified claim.** "Tests pass" requires pasted output in the log. A sub-agent's assertion is not evidence. |
| I8 | **`AGENTS.md` rules 2 and 3 are hard boundaries** — see §2.1. |
| I9 | **Never let a command sync the venv** after cuOpt is installed — sync prunes cuOpt out (it is not in `pyproject.toml`). Install with `uv pip install` only. Two specific traps: (a) `doc/contributing.rst:46` tells contributors to run `uv sync --extra dev --extra solvers` — **do not**; (b) there is no `uv.lock`, so a bare `uv run <cmd>` locks and syncs before running. Invoke tools as `.venv/bin/pytest`, `.venv/bin/mypy` — or `uv run --no-sync` if you must. **ruff is the exception:** it is in no extra (the repo runs it via pre-commit), so there is no `.venv/bin/ruff`. It is installed once via `uv tool install ruff==0.15.20` (§6.0) into uv's isolated tools environment — invoke it as `/home/azureuser/.local/bin/ruff`. That install is sync-proof by construction: no `uv sync` can touch it. **This invariant has an end date:** it exists only because cuOpt is not yet declared in `pyproject.toml`. Once the §4.6 extra has merged, `uv sync --extra <gpu-extra>` becomes the *correct* command — and the fact that it then keeps cuOpt installed is itself a test that the extra works. Do not relax I9 before that work package lands and Q13's evidence is in. |
| I10 | Every non-obvious choice gets a line in `logs/30-decisions.md` — even the ones you resolve yourself. |
| I11 | **Every commit is authored by the Claude model that actually wrote it**, and follows the commit discipline in §6.4a. No exceptions, including one-line fixes in Phases 5 and 6. The invariant is **truthfulness, not uniformity**: `Claude Opus 5` and `Claude Fable 5` are both correct author strings — stamping the wrong one is the violation. Work is never discarded because of which model produced it; a mis-stamped commit is *relabelled*, never reset (§6.4a). **You (the lead) never author a commit on `feat/cuopt-solver`** — not for provenance reasons but because you do not write production code; you merge `--ff-only`, which creates no commit. The harness-record branch (§6.7) is yours to commit, under your own model name. |
| I12 | **Thresholds are numbers, frozen at Gate 3.** No agent may widen a tolerance, re-run until green, downgrade a check to a weaker form, or argue a discrepancy away. See §3.6. |
| I13 | **Record, don't fix.** A pre-existing linopy defect an agent notices is recorded in `logs/45-known-issues.md`, never fixed in passing — however trivial. It is out of scope and it pollutes the diff (I4). |
| I14 | **Every artifact listed in §7.1 is written from its template** in `/home/azureuser/linopy-gpu/dev-scripts/cuopt/templates/`. Keep the headings exactly — later agents parse these files by heading. Write "None" rather than deleting a section, so absence is explicit. Artifacts not in that table are freehand — §7.1 enumerates exactly which, and is the single source of truth for that split; `90-escalations` follows the inline format in §3.4. |

### 2.1 Where autonomy stops

`AGENTS.md` constrains this project in ways that override "fully autonomous":

- **Rule 2 — the human writes their own intent.** You may draft the *technical* section
  of the PR body under a `> [!NOTE] The following content was generated by AI.` alert,
  with verbose material in `<details>`. You must leave the intent/motivation paragraph
  as an explicit `<!-- HUMAN: write your intent here -->` placeholder. **You do not open
  the PR** — that is the human's action.
- **Rule 3 — conversations are not held through an agent.** You never post PR comments,
  never reply to review feedback, never argue a position in a GitHub thread.

Autonomy applies to *building the change*, not to *representing the human*.

---

## 3. Escalation policy

Default to resolving things yourself. Escalation is rare and expensive.

### 3.1 Escalate only when

- **E1 — Genuine design fork.** Two or more defensible options exist, they differ
  materially in public API or user-visible behaviour, and no repo precedent settles it.
- **E2 — Irreconcilable evidence.** Two authoritative sources disagree and an empirical
  test cannot decide between them (or the test itself is ambiguous).
- **E3 — Environment damage.** Proceeding would degrade the human's working environment
  beyond the pre-authorised change in §4.4.
- **E4 — Scope breach.** Correct implementation requires touching code outside the §5
  integration map in a way that changes behaviour for existing solvers.
- **E5 — Loop exhaustion.** A review loop hit its cap (§6) without converging.
- **E6 — Missing capability.** Credentials, hardware, or network access you do not have.

### 3.2 Never escalate

- Anything answerable by reading the repo, running a test, or reading cuOpt's docs.
- Naming, style, file placement, docstring wording — follow cuPDLPx and move on.
- A choice between two options with equivalent consequences — pick one, log it in
  `logs/30-decisions.md`, continue.
- Transient failures (flaky network, GPU busy) — retry up to 3×, then treat as E6.
- "Should I keep going?" — yes. Keep going.

### 3.3 Resolve-first protocol (mandatory before any escalation)

1. Search the repo for precedent (`cupdlpx`, `highs`, `gurobi` are the richest).
2. Attempt an empirical answer with a spike script in `/home/azureuser/linopy-gpu/dev-scripts/cuopt/spikes/`.
3. Check PR #516 (`gh pr diff 516 --repo PyPSA/linopy`).
4. Check cuOpt docs and installed source (`.venv/lib/python3.11/site-packages/cuopt/`).
5. Write the attempt and its outcome into `logs/40-blockers.md`.

Only if all five leave the question open do you escalate.

### 3.4 Escalation format

Write to `logs/90-escalations.md` **and** surface it to the human in your reply:

```markdown
## ESC-<n> — <one-line question>
**Phase:** <N>   **Raised:** <UTC>   **Trigger:** E<1-6>
**Question:** <precise, answerable question — not "what should I do?">
**Options:** A) <option> — consequence. B) <option> — consequence.
**Evidence for each:** <what you found, with file:line / URLs / test output>
**Resolve-first steps taken:** <the five steps from §3.3, with outcomes>
**Recommendation:** <your pick, and why>
**Blocked:** <exactly what is stalled>
**Continuing meanwhile:** <what work proceeds regardless>
```

Then **keep working on everything not blocked.** Never idle waiting for an answer.

### 3.5 The divergence signal

The five planners in Phase 2 are a built-in ambiguity detector. When the selector
compares them:

- **5/5 agree** on a point → obvious. Do not escalate.
- **4/1 or 3/2 split** on a *material* point → this is the strongest available evidence
  of a genuine E1 fork. Apply §3.3; escalate if it survives.
- Splits on cosmetics → resolve by repo precedent, no escalation.

### 3.6 The agent-side escalation rule (carry verbatim in every brief)

§3.1–3.5 govern *when you escalate to the human*. This rule governs something different
and more dangerous: **an agent quietly weakening a check so its work passes.**

> Tolerances and thresholds are **numbers**, proposed in the verification plan
> (`logs/15-verification.md`) and **frozen at Gate 3**. If any check misses its threshold,
> the responsible agent **reports the miss verbatim — with the actual numbers — and stops
> on that item.** It may not widen a tolerance, re-run until a favourable result appears,
> downgrade the check to a weaker form, skip the test, or explain the discrepancy away.
> Changing a threshold means returning to Gate 3.

A red check reported honestly is acceptable work. A hidden one is not. This sentence goes
in every sub-agent brief verbatim (§8).

Why this matters here specifically: cuOpt is a **first-order method with tolerances**, so
"nearly optimal" results are normal and the temptation to nudge an `rtol` until green is
constant. That nudge is exactly how a dual-sign bug ships.

### 3.7 One-shot autonomous mode — the no-stall rule

This run has no human in the loop. A blocking escalation would therefore stall the run
indefinitely, which is the one outcome guaranteed to produce nothing. So escalation changes
meaning:

**Judgment triggers (E1, E2, E4, E5) — file and proceed.**

1. Write the full §3.4 entry to `logs/90-escalations.md`.
2. **Act on your own recommendation** — the one you were required to write anyway.
3. Log it in `logs/30-decisions.md` marked **`UNREVIEWED — autonomous, ESC-<n>`**.
4. Continue.

**Capability triggers (E3, E6) — still hard-stop.** Missing credentials, absent hardware, an
environment change beyond §4.4: proceeding is not possible, not merely unreviewed. Stop that
line of work, complete everything else in full, and report it.

**Why this is safe here.** Every output of this run is a **branch that is never pushed to
`master` and never opened as a PR** (§2.1). The human reviews the whole thing before it goes
anywhere. Autonomy is bounded by that: you are authorised to *decide*, not to *ship*. Nothing
you do in this mode is irreversible, and that is precisely what makes the mode acceptable.

**What you owe in exchange.** Every unreviewed decision is one the human would normally have
caught at Gate 3, so the handoff must make them trivially auditable — see §6.8. Do not bury
them. A one-shot run that hides its own soft spots is worse than one that fails loudly.

**Lower your escalation bar, not your standard.** Because filing an escalation no longer
costs the human anything, file *more* of them, not fewer — each one is an audit marker. What
you must not do is let the absence of a reviewer relax I12 or §3.6: a threshold is still
frozen, a miss is still reported verbatim, and "no one is watching" is not a reason to widen
anything.

---

## 4. Verified ground truth — do not re-litigate

All of the following was empirically verified on this machine. Treat as fact. If a
sub-agent contradicts any of it, the sub-agent is wrong until it produces
reproducible evidence — and that evidence is an E2 escalation.

### 4.1 Hardware / driver

- Tesla T4, compute capability **7.5**, 16 GB VRAM (15.57 GiB usable).
- Driver **535.274.02**, reports CUDA 12.2. Ubuntu 22.04.5, x86_64, glibc 2.35.
- cuOpt requires CC ≥ 7.0 and driver ≥ 525.60.13 → **this machine qualifies.**
- A CUDA **12.9** runtime loads fine on this 12.2 driver via minor-version
  compatibility. Verified, not assumed. **Do not use `cu13` wheels** — driver too old.

### 4.2 The package

- Install **`cuopt-cu12==26.8.*`**. This is the newest release with **cp311** wheels,
  matching `.venv` (Python 3.11.15).
- `26.10.*` (as shown in cuOpt's README) is **not on the release index** — do not use it.
- `26.6.0` has **no cp311 wheel**.
- **Not** `cuopt-server-cu12` / `cuopt-sh-client` — those are a REST server and its
  shell client. linopy's `is_available()` is an in-process import probe, so the
  in-process library is the only viable choice.
- **Not the bare `cuopt` name on PyPI.** That is a *different* project — summary "A simple
  cuopt package", no author, no description, no project URLs, versions `0.1.0` and `26.8.0`.
  NVIDIA's package is `cuopt-cu12` (homepage `docs.nvidia.com/cuopt`, source
  `github.com/nvidia/cuopt`). Never write `pip install cuopt` in code, docs or a test.

**`cuopt-cu12` and its entire dependency tree are on ordinary PyPI.** This corrects an earlier
belief that an NVIDIA index was required — it is not:

```bash
# no --extra-index-url, no --index-strategy: uv had ONLY PyPI configured, and it resolved
uv pip install --dry-run 'cuopt-cu12==26.8.*'
# -> 34 packages incl. libcuopt-cu12, cudf-cu12, pylibraft-cu12, rmm-cu12, nvidia-*-cu12
```

`pypi.nvidia.com` still carries the wheels and remains a valid fallback (and is where cp314
appears first), but it is **optional**. `--index-strategy unsafe-best-match` was never a
property of cuOpt — it is only needed to make uv resolve *across two indexes* once you add
`--extra-index-url`. Drop the extra index and the flag is unnecessary. This matters well beyond
our own venv: see §4.6.

```bash
cd /home/azureuser/linopy-gpu
uv pip install 'cuopt-cu12==26.8.*'
```

**Wheel coverage** (PyPI, 26.8.0): `manylinux` **x86_64 and aarch64**, **cp311–cp314**, and
**no sdist**. So cuOpt is **Linux-only** and needs Python ≥ 3.11 — which happens to match
linopy's own `requires-python = ">=3.11"` exactly, so only a *platform* marker is ever needed.
On macOS or Windows there is nothing to install and nothing to fall back to.

### 4.3 One entry point verified working — **not** a decision

⚠️ **This section records that a call sequence works. It does NOT settle which entry point
linopy should use.** That choice is an open design question — Phase 1 Question 0 investigates
it, and the Phase 2/3 plans decide it with reasons. Do not treat the snippet below as the
prescribed approach; treat it as proof the library functions on this machine.

The sequence exercised was the matrix `DataModel` API:

```python
from cuopt.linear_programming import data_model, solver, solver_settings
dm = data_model.DataModel()
dm.set_csr_constraint_matrix(A.data, A.indices, A.indptr)   # A = M.A.tocsr()
dm.set_constraint_lower_bounds(lower); dm.set_constraint_upper_bounds(upper)
dm.set_objective_coefficients(M.c)
dm.set_variable_lower_bounds(M.lb); dm.set_variable_upper_bounds(M.ub)
dm.set_variable_types(vtypes)          # 'C' / 'I'
dm.set_maximize(model.sense == "max")
sol = solver.Solve(dm, solver_settings.SolverSettings())
sol.get_termination_status(); sol.get_primal_objective()
sol.get_primal_solution(); sol.get_dual_solution(); sol.get_reduced_cost()
```

Confirmed on the T4 with cuopt **26.08.00** — these results are verified fact:

- **LP** `max x+y s.t. x+2y<=4, 3x+y<=6, x,y>=0` → `x=[1.6, 1.2]`, obj `2.8`,
  duals `[0.4, 0.2]`, status `1` / `"Optimal"`.
- **MILP** same model with `set_variable_types(np.array(['I','I']))` → obj `2.0`.
- cuOpt runs **concurrent dual simplex + barrier + PDLP**, not PDLP alone. It is a
  materially larger feature surface than cuPDLPx.

**The candidates Phase 1 must weigh.** `cuopt.linear_programming` exposes these submodules
(enumerated from the installed 26.08.00 package): `data_model`, `problem`, `io`, `solver`,
`solution`, `solver_settings`, `internals`. At least three plausible entry points:

| Candidate | Shape | Known pros / open questions |
|---|---|---|
| `data_model.DataModel` + `solver.Solve` | arrays + CSR | Verified above. Maps onto `model.matrices` the way cuPDLPx's `_build_solver_model` does. Exposes `set_quadratic_objective_matrix` and `add_quadratic_constraint`. |
| `problem.Problem` | Gurobi-style `addVariable`/`addConstraint` | Carries surface `DataModel` may not: `getWarmstartData`, `set_mip_callback`, `relax()`, `readMPS`/`writeMPS`, `Slack`/`DualValue`/`ReducedCost` properties. But per-variable Python calls may not scale. |
| `io` submodule | file-based | Could support linopy's **file** `io_api`s (`lp`/`mps`), which cuPDLPx cannot do at all — potentially a capability advantage worth having. |

The right answer may be a **primary entry point plus a secondary one** (e.g. arrays for
`direct`, file IO for `mps`). That is exactly the sort of call the plans should make
explicitly rather than inherit from this section.

### 4.4 Pre-authorised environment change

Installing cuOpt **downgrades `pandas` 3.0.5 → 3.0.3**. Nothing here pins pandas — this is
uv resolving an **upstream ceiling**: `cuopt-cu12` → `cudf-cu12`, whose metadata declares
`pandas<3.0.4a0,>=3.0.0`. The venv's 3.0.5 sits above that bound, so uv picks the newest
pandas underneath it (3.0.3). Because it is a ceiling and not a pin, it lifts by itself once
NVIDIA raises cuDF's bound — do not "fix" it by pinning pandas anywhere.

This is **pre-authorised and not an escalation**:

- linopy's core `dependencies` do not pin pandas at all — harmless for the test suite.
- It **does** collide with the `benchmarks` extra, which pins `pandas==3.0.5` exactly
  (`pyproject.toml:92`). **Do not run benchmarks in this venv.** If benchmarking is
  needed, use a separate venv, and log that fact.
- Disk: ~6.3 GB unpacked, ~2 GB marginal (uv hardlinks from cache). 25 GB free.

Also pre-authorised: creating a **separate docs venv** at `/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs` (§6.0) and a
separate benchmarks venv if one is ever needed. Both sit under gitignored `dev-scripts/`,
touch nothing in `.venv`, and are deleted or ignored at handoff.

Any *further* environment degradation beyond this is an E3 escalation.

### 4.5 Scope (pre-decided — proceed, do not escalate)

**LP, MILP, and QP are all in scope.** Sequenced, not optional:

1. **LP + MILP first.** Both verified working (§4.3). This is the core; it must be complete
   and green before QP work starts.
2. **QP after.** Not a stretch goal and not contingent on spare capacity — QP is attempted,
   and the run is not done until it has been. cuOpt supports it
   (`minimize xᵀQx + cᵀx`, Q symmetric PSD).

If QP ultimately cannot ship, that is a **finding to report with evidence**, not a scope
decision to make quietly. It requires an escalation entry saying what blocked it, and it
must be called out in the handoff's "what did not ship" — never dropped silently.

Sequencing consequences:
- QP is its **own work package**, dispatched after LP+MILP has merged and Gate 4's checks
  pass on the LP/MILP core. It never runs concurrently with the `solvers.py` core package.
- If QP ships, the class declares `SolverFeature.QUADRATIC_OBJECTIVE`. Note that
  `QUADRATIC_SOLVERS` (`solvers.py:4426`) is **derived automatically** from that feature —
  no manual registry edit, and adding one would be wrong.
- The verification plan needs QP rows: differential vs a QP-capable CPU solver, and the
  Q-matrix convention check from Phase 1 Question 12.

---

### 4.6 Installability is a requirement, not a packaging detail

**The human lead has made this binding: a linopy user must be able to ask for cuOpt support and
get it, without knowing anything about NVIDIA's index, uv resolution flags, or how they invoke
Python.** The target is one command:

```bash
pip install 'linopy[<gpu-extra>]'      # and the uv equivalent
```

§4.2 is what makes this achievable: because the whole tree is on PyPI, a plain extra is
sufficient. **Note that no amount of `[tool.uv]` configuration could have delivered this** —
index settings in `pyproject.toml` are workspace-local and are *not* propagated through
published package metadata, so a downstream `pip install linopy[gpu]` would never see them. Had
cuOpt been NVIDIA-index-only, the honest answer would have been "you cannot; here are manual
instructions". It is not, so we can do the real thing.

**Constraints the extra must satisfy** (all are testable, and the plan owns the design):

1. **Never breaks resolution on macOS or Windows.** CI runs `ubuntu-latest`, `macos-latest` and
   `windows-latest` (`.github/workflows/test.yml`), and cuOpt has Linux-only wheels with no
   sdist. Guard with a marker in the established style of this file — `pyproject.toml:105`
   already does `"cplex; platform_system != 'Darwin'"`.
2. **Never inflicts cuOpt on someone who did not ask.** It pulls ~34 packages and ~6.3 GB
   (§4.4). It must not land in `solvers`, `dev`, `all`, or any default install path — a CPU
   user running `pip install linopy[solvers]` must not start a 6.3 GB download. This is a
   strong argument for a **separate** extra rather than adding to `solvers`; the plan confirms
   the shape and names it, logged as a `D`-entry.
3. **Degrades honestly.** On a machine with the package but no GPU, or the wrong driver,
   `linopy.available_solvers` must simply omit cuOpt and `model.solve("cuopt")` must fail with
   an actionable message — never a bare CUDA traceback.
4. **`cu12` vs `cu13` is the user's to get right.** §4.1: `cu13` wheels need a newer driver.
   Whether the extra pins `cu12`, or documents both, is a plan decision — but the docs must
   state the driver requirement next to the install command, not three pages away.

**Proof, not intent.** Phase 1 Q13 verifies this end-to-end from a clean venv (§6.1), and it is
a Definition-of-Done item. An extra that was never installed from scratch is an untested extra.

## 5. Repo integration map

Every one of these is a confirmed touch point. Phase 1 verifies it is complete;
Phase 3's plan must account for every row.

| File | Anchor | What cuOpt needs |
|---|---|---|
| `linopy/solvers.py` | `:148` `SolverFeature` | Pick the feature set. cuOpt is a superset of cuPDLPx: `DIRECT_API`, `GPU_ACCELERATION`, `GPU_ONLY`, `SOLUTION_FILE_NOT_NEEDED`, plus `INTEGER_VARIABLES`, possibly `QUADRATIC_OBJECTIVE`, `MIP_DUAL_BOUND_REPORT`. |
| `linopy/solvers.py` | `:176` `TYPE_CHECKING` | `import cuopt` |
| `linopy/solvers.py` | `:304` | `cuopt = _LazyModule("cuopt")` |
| `linopy/solvers.py` | `:346` `SolverName` | `cuOpt = "cuopt"` — vendor spelling, **not** `CuOpt`. This is both NVIDIA's own capitalisation and linopy's convention: the enum already carries `cuPDLPx`, `MindOpt`, `COPT`, `Highs`. Do not "normalise" it to PascalCase. |
| `linopy/solvers.py` | `:4156` `class cuPDLPx` | **The template.** New `class cuOpt(Solver[None])` alongside it. |
| `linopy/solvers.py` | `:4440` `_SOLVER_PROBE_ORDER` | Add `"cuopt"`. Position determines default-solver precedence — place it with the other GPU solvers, near `cupdlpx`. |
| `linopy/solver_capabilities.py` | `SOLVER_REGISTRY` (`:78-100`) | **Verification, not an edit — expect a zero-line diff.** This module is a back-compat shim; `SOLVER_REGISTRY` is a `_LazyRegistry` that derives `SolverInfo` on the fly from `SolverName` + the class's declared features. There is nothing to register. Satisfy this row by *proving* `SOLVER_REGISTRY["cuopt"]` resolves with the declared features. Editing the shim to add cuOpt is a drive-by change (I4) and is wrong. Note the shim resolves classes via `getattr(solvers, SolverName(name).name)`, so the **enum member name must equal the class name** — `cuOpt = "cuopt"` + `class cuOpt` satisfies that. |
| `linopy/io.py` | `:33` TYPE_CHECKING, `:792` `to_cupdlpx` | Add a `to_cuopt(m)` counterpart. |
| `linopy/model.py` | `:73` import, `:2517` binding | Bind `to_cuopt = to_cuopt`. |
| `linopy/constants.py` | `:171` `TerminationCondition` | Map cuOpt statuses onto the existing enum — add nothing new unless unavoidable. |
| `test/conftest.py` | `:57-79` | **Likely no change needed.** Gating is generic on `SolverFeature.GPU_ONLY`; cuPDLPx is not named. Declaring `GPU_ONLY` auto-marks the tests. Verify this. |
| `test/test_optimization.py` | `:562`, `:685`, `:1110` | Solver-specific branches — cuPDLPx has three. cuOpt needs its analogues (options dict, capability skips). |
| `test/test_solvers.py` | — | PR #516 added 4 lines here. |
| `pyproject.toml` | optional-dependencies | **A user-installable extra is a requirement (§4.6), not an open question** — the only open part is its name and shape, which the plan decides and logs. Must satisfy §4.6's four constraints: Linux marker so macOS/Windows CI still resolves; not reachable from `solvers`/`dev`/`all` (6.3 GB); honest degradation without a GPU; the `cu12`/`cu13` story documented. No `[tool.uv]` index entry is needed — the tree is on PyPI (§4.2) — and one would not help users anyway (§4.6). Note `cupdlpx` sits commented out of `solvers`; that is precedent for *how not to ship a GPU solver*, not a pattern to copy. |
| `doc/gpu-acceleration.rst` | — | Primary docs home (PR #516 created it, 117 lines). |
| `doc/prerequisites.rst`, `doc/contributing.rst`, `doc/index.rst`, `doc/api.rst`, `doc/release_notes.rst`, `README.md` | — | Same set PR #516 touched. |
| `linopy/solvers.py` (QP) | `:4426` `QUADRATIC_SOLVERS` | **Read-only anchor.** Declaring `SolverFeature.QUADRATIC_OBJECTIVE` registers cuOpt here automatically — the list is derived from the feature. Do not edit it. The requirement is the QP build/solve path + its tests (§4.5). |

### 5.1 `Solver` base class contract

There are **no `@abstractmethod`s** — the ABC is convention-based. cuPDLPx overrides
exactly: `display_name`, `features`, `is_available`, `_license_probe`, `_run_file`,
`_build_direct`, `_build_solver_model`, `_run_direct`, `_solve`, `_set_solver_params`.
Other ClassVars available: `accepted_io_apis`, `supports_persistent_update`,
`supports_sign_update`.

### 5.2 PR #516 — the reference

`Add support for solver cuPDLPx`, **merged 2026-01-19**, **+771 / −62 across 15 files**.
Fetch with `gh pr diff 516 --repo PyPSA/linopy`.

**Use it for shape, not for content.** It is from January and the repo has moved —
`_SOLVER_PROBE_ORDER`, the lazy `available_solvers`, `LicenseStatus`, and the v1
semantics convention all landed after or around it. **Where PR #516 and current
`master` disagree, current `master` wins (I5).** Any agent that copies a stale pattern
from #516 has made an error.

### 5.3 Requirement IDs

Every row of the §5 table is a **requirement with a stable ID**: number them top to bottom
`C-R1` … `C-R17`. Anything Phase 1 discovers that implies work gets the next free ID
(`C-R18`+), recorded in `logs/30-decisions.md`. **IDs are never renumbered or reused.**

These IDs are the thread that makes completeness checkable rather than a matter of opinion:

| Where | What must reference the ID |
|---|---|
| Phase 2 plans | a table mapping **each ID → the work that satisfies it** |
| Phase 3 `FINAL.md` | the same table, with no unmapped IDs |
| Phase 4 slice briefs | the IDs that slice satisfies |
| Phase 4 return reports | ID → what was built (item 2 of §8.1) |
| Phase 5 conformance | ID → deliverable → the check that proves it |
| Phase 6 review | every ID accounted for |

Without this, "did we cover everything?" is re-derived differently by every agent. With it,
an unmapped ID is a mechanical failure that any reviewer can spot.

---

## 6. Phases and gates

**Section numbering.** Phase *N* is §6.*N* — so §6.3 is Phase 3, §6.8 is Phase 8. Sub-parts
carry a letter (§6.3a, §6.4a, §6.6a, §6.6b). References elsewhere in this document use those
numbers; they are headings you can grep for.

A gate is passed only when every checkbox has **logged evidence**. You record the pass
in `logs/00-lead-journal.md` with a pointer to that evidence.

### 6.0 Phase 0 — Setup (you do this yourself)

- Create `feat/cuopt-solver` off `master`.
- `mkdir -p dev-scripts/cuopt/{logs,spikes,plans,worktrees}`.
- Confirm `git check-ignore dev-scripts/` returns a hit (I3).
- Install cuOpt per §4.2. Confirm `python -c "import cuopt"` works in `.venv`.
- Confirm ruff: `/home/azureuser/.local/bin/ruff --version` prints **0.15.20** (already
  installed via `uv tool install ruff==0.15.20`; re-run that command if missing). ruff is
  in no pyproject extra — the repo runs it through pre-commit
  (`.pre-commit-config.yaml:11`, pinned v0.15.20, which this matches) — so it lives in
  uv's tools environment, **not** `.venv`, where no sync can remove it. Everywhere this
  charter says `ruff`, the invocation is `/home/azureuser/.local/bin/ruff` (I9).
- Initialise all log files per §7.

**Baseline everything the DoD asserts, on untouched `master`, before touching anything.** A
baseline you did not take is a baseline you will wish you had: if a notebook or a docs warning
is *already* broken on `master`, Phases 5 and 6 will block on it with no way to prove the
breakage is pre-existing — and the honest ways out of that (I13: record, don't fix) are
unavailable to an agent that cannot tell "pre-existing" from "I did that".

```bash
cd /home/azureuser/linopy-gpu
# 1. tests
.venv/bin/pytest -q            2>&1 | tee /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/05-baseline-pytest.txt
.venv/bin/pytest --run-gpu -q  2>&1 | tee /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/05-baseline-pytest-gpu.txt

# 2. docs + notebooks, in a SEPARATE venv (see below)
uv venv --python 3.11 /home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs
uv pip install --python /home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin/python -e '.[docs]'
/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin/sphinx-build -b html doc doc/_build/baseline 2>&1 \
  | tee /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/05-baseline-docs.txt
```

**Why the docs build gets its own venv, and why it must *not* contain cuOpt.** The `docs` extra
pins exact versions (`matplotlib==3.11.0`, `ipython==9.15.0`, …) that could collide with cuOpt's
dependency ceilings the way pandas already does (§4.4) — installing it into `.venv` risks
degrading the solver environment, which is an E3 outcome. Separately, ReadTheDocs builds with
**no GPU and no cuOpt**, so a cuOpt-free docs venv is not a workaround: it is the correct test.
If the docs only build on a machine with a T4, the docs are broken.

Summarise the three baselines in `logs/05-baseline.md`: pass/fail counts, every failing test or
notebook by name, and the warning count. Two notebooks use gurobi
(`infeasible-model.ipynb`, `solve-on-oetc.ipynb`) and may fail on licensing — **record what
actually happened; do not predict it**. Anything red at baseline is quoted in
`logs/45-known-issues.md` and excluded from the later comparison by name.

**Gate 0:** cuOpt imports in `.venv`; the three baselines above are captured with pasted output
in `logs/05-baseline.md`, each failure named. A green baseline is not required — a *recorded*
one is.

### 6.1 Phase 1 — Exploration spike

Dispatch **1–2 agents** to `/home/azureuser/linopy-gpu/dev-scripts/cuopt/spikes/` (no worktree; this touches no
production code). They write throwaway scripts and answer §5's open questions
empirically. Start from §4.3 — do not re-verify what is already verified.

**Question 0 — entry-point selection. Answer this first; several answers below depend on
it.** Which cuOpt entry point(s) should linopy call, and why? Build a throwaway prototype
against **at least two** of the §4.3 candidates and compare on evidence, not preference:

- **Coverage** — which candidates support LP, MILP, **and QP** (§4.5 commits us to all three)?
  Duals, reduced costs, warm start, MIP callbacks, solver options, log control?
- **Fit to `model.matrices`** — how much translation code does each need? Count the lines.
- **Scale** — time building a model with ~10⁴–10⁵ variables through each candidate. A
  per-variable Python API may be orders of magnitude slower than array ingestion; **measure
  it, do not assume it.**
- **linopy `io_api` reach** — can any candidate support the file APIs (`lp`, `mps`) that
  cuPDLPx has to refuse? Declaring `accepted_io_apis` beyond `direct` is a real user win.
- **Stability** — is any candidate marked experimental or private (`internals`)?

Record the comparison in `logs/10-exploration.md` — written from
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/templates/exploration-TEMPLATE.md` (I14) — with the measurements. **Recommend, do not
decide** — the Phase 2 plans propose and Gate 3 fixes the choice, logged as a `D`-entry in
`logs/30-decisions.md`.

Then, with runnable evidence for each:

1. **Status mapping** — every cuOpt termination status → `TerminationCondition`.
   Force each one: optimal, infeasible, unbounded, time limit, iteration limit.
2. **Dual sign convention** — for `min`/`max` × `<=`/`>=`/`==`. Compare cuOpt's duals
   against HiGHS on the same model. **This is the single most likely source of a
   silent correctness bug.** Same for reduced costs.
3. **Objective offset / constant term** handling, and whether `set_maximize(True)`
   flips the sign of duals and reduced costs.
4. **Log capture.** cuOpt prints from C++ to stdout (you saw the banner). Python-level
   `contextlib.redirect_stdout` will **not** catch it — an OS-level `dup2` fd redirect
   is likely required for `log_fn`. Confirm and find the pattern.
5. **Edge cases:** model with no constraints; empty/degenerate model; single variable;
   infinite bounds (`np.inf` vs sentinel); ranged constraints; equality via `lb == ub`;
   free variables.
6. **Array contracts:** dtypes (float64? int32 CSR indices?), C-contiguity,
   `set_variable_types` accepting `np.array(['I','C'])` vs bytes vs list.
7. **Solver options** — real parameter names via `SolverSettings.set_parameter`,
   time limit, tolerances; what happens on an unknown parameter name.
8. **Interruptibility** — is `Solve()` KeyboardInterrupt-safe? See the
   `_run_highs_with_keyboard_interrupt` precedent in `solvers.py`.
9. **Repeat solves / teardown** — several sequential solves in one process; GPU
   context lifetime; memory behaviour near the T4's 16 GB.
10. **Unsupported surfaces** — semi-continuous, SOS, indicator constraints: confirm
    each is genuinely absent so the guards raise honestly.
11. **Warm start** (`get/set_pdlp_warm_start_data`).
12. **QP — how, not whether.** §4.5 commits us to QP. `DataModel` exposes
    `set_quadratic_objective_matrix` (and `add_quadratic_constraint`); confirm the `Problem`
    path's equivalent. Establish: the Q-matrix format cuOpt expects (full vs triangular,
    factor-of-2 convention on the diagonal — **a classic source of silently-wrong
    objectives**), how it maps onto linopy's `model.objective` quadratic form, PSD
    requirements, whether duals are still returned, and what cuOpt does with a non-PSD Q.
    Docs say crossover is unsupported for QP — verify.

13. **Installability from scratch (§4.6).** Prove the user story, do not reason about it.
    Build the candidate extra, then in a **fresh venv that has never seen cuOpt**:
    `pip install '/home/azureuser/linopy-gpu[<extra>]'` with **no** extra index and **no**
    resolver flags, then `python -c "import linopy; print(linopy.available_solvers)"`. Paste
    both. Also record: what `pip install 'linopy[solvers]'` pulls (cuOpt must **not** appear),
    and what the resolution does under a macOS/Windows marker set — `uv pip compile
    --python-platform windows` / `macos` is enough, no second machine needed. If any of this
    fails, that is a **finding**, not a reason to relax §4.6.

**Gate 1:** `logs/10-exploration.md` answers Question 0 (with measurements) and all 13
numbered items above, each with pasted output. Unknowns are
listed explicitly as unknowns — a guess recorded as a finding is an I7 violation.

### 6.2 Phase 2 — Five independent plans

Dispatch **5 planning agents in parallel**. Each gets: this charter, `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/10-exploration.md`,
and the §5 map. **Identical briefs. Zero cross-talk (I6).** Each writes
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/plan-<1..5>.md`.

Each plan must contain: the full file-by-file change set; the feature-flag set with
justification; the status-mapping table; the work-package breakdown (§6.4 rules);
the test plan; the docs plan; explicit non-goals; and the risks it sees.

**Isolation is what makes §3.5 mean anything**, and it is enforced only by instruction, so state
it explicitly. Carry this line verbatim in each of the five briefs:

> **Do not read `/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/`.** Do not list it, do not open another plan, do not ask another
> agent what it proposed. Your plan must be arrived at independently — four other agents are
> writing theirs in parallel and the *disagreement between them* is a deliberate signal the
> selector depends on. A plan contaminated by a sibling destroys that signal while looking
> like agreement.

Write only your own `plan-<k>.md`; touch no other file under `plans/`.

**Gate 2:** five plans exist, each independently complete, each written without reading a
sibling. Apply §3.5 to the spread.

### 6.3 Phase 3 — Selection and convergence

Two distinct roles, held by two distinct agents for the whole phase. I6 forbids anyone
reviewing what they wrote, so the split has to be explicit or the loop quietly collapses into
one agent grading itself.

| Role | Who | Writes | Never writes |
|---|---|---|---|
| **Selector-reviewer** | fresh context; wrote none of the five plans | `logs/50-review.md` only | `FINAL.md`, `15-verification.md`, any plan |
| **Revision planner** | fresh context; wrote none of the five plans, and is **not** the selector-reviewer | `/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/FINAL.md` and `logs/15-verification.md` | `logs/50-review.md` |

1. The **selector-reviewer** reads all five plans from disk, picks the strongest, states why,
   and names the ideas worth grafting from the other four. It writes that as review round 1.
2. The **revision planner** writes `FINAL.md` — the chosen plan plus the grafts — and the
   verification plan (§6.3a).
3. Loop: selector-reviewer reviews `FINAL.md` → revision planner revises → repeat, until the
   review returns **APPROVE**: logically consistent, complete against §5, no hand-waving,
   every work package with concrete acceptance criteria.

Both agents persist across iterations (they keep their context); neither ever takes the other's
pen. If either has to be replaced mid-loop, its replacement must still satisfy the "wrote none
of the five plans" condition.

**Cap: 5 iterations.** Not converged by then → E5 escalation.

#### 6.3a The frozen verification plan

Alongside `FINAL.md`, the planning agent writes **`logs/15-verification.md`** from
`templates/verification-TEMPLATE.md`. It is both the checklist Phase 5 executes and the
frozen record of the agreed numbers.

Use a three-tier ladder — the cheapest oracle that catches each failure mode:

| Tier | Check | Oracle | When |
|---|---|---|---|
| **V0** | Structural — `ruff`, `mypy`, solver registered, features declared | mechanical | every WP |
| **V1** | **Differential vs HiGHS, live** — same model, same process, both solvers called | `model.solve("highs")` on the identical model | Phases 4 and 5 |
| **V2** | End-to-end on linopy's own test models | existing expected values in `test_optimization.py` | Phase 5 |

**V1 is the important one.** `highspy` is installed in `.venv` (verified), so the oracle is
available. The test must **call HiGHS live in the same process and compare** — never assert
against a hard-coded expected value copied from an earlier run. A differential test with a
baked-in expectation silently degrades into a regression test with an unverified baseline,
which is how a systematic sign error survives a green suite. Cover every cell of the
template's sign-convention matrix (`min`/`max` × `<=`/`>=`/`==`).

**Every threshold is a number.** "Duals correct" is not a threshold; `rtol=1e-6` is. cuOpt's
PDLP path has looser default tolerances than a simplex solver, so pick the number
deliberately and record the rationale — note that #516 hit exactly this and added
`GPU_SOL_TOL`/`CPU_SOL_TOL` for it.

**Freezing — by content hash, not by git.** `dev-scripts/` is gitignored (I3), so this file is
**never tracked** and `git diff` against it returns empty no matter what was edited. A git-based
drift check here would be silently vacuous. Freeze it by hashing instead.

At Gate 3, compute the hash and record it in **two** places — `logs/00-lead-journal.md` and
`logs/50-review.md` — then make the file read-only:

```bash
F=/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md
sha256sum "$F"        # record this digest in BOTH logs, with the UTC timestamp
chmod 444 "$F"        # belt and braces: an accidental edit now fails loudly
```

The Phase 5 reviewer recomputes it and compares against **both** recorded copies:

```bash
sha256sum /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md
```

A digest that does not match, or two recorded digests that disagree with each other, is a
**blocking finding**. Recording it twice is deliberate: tampering then requires editing the
verification plan *and* two separate append-only logs, which is no longer something that
happens by accident.

Re-freezing (the only legitimate edit path) means returning to Gate 3: `chmod 644`, revise,
re-approve, re-hash, record the new digest as a **new** dated entry in both logs — never by
overwriting the old one.

**Gate 3 carries extra weight in one-shot mode.** It is the point where a human would
normally have read the plan, and no one will. Before declaring it passed, the reviewer must
confirm on the record:

- Every `C-R*` ID maps to concrete work; **no ID is unmapped or silently reinterpreted**.
- Every threshold in `logs/15-verification.md` is a number with a stated rationale — not a
  placeholder, not an adjective.
- The V1 sign-convention matrix has all six cells filled.
- The plan's non-goals are explicit, so scope creep is detectable later.

**Gate 3:** `/home/azureuser/linopy-gpu/dev-scripts/cuopt/plans/FINAL.md` exists with a clean **APPROVE** from the selector-reviewer;
`logs/15-verification.md` is written, hash-frozen, and its sha256 recorded in **both**
`logs/00-lead-journal.md` and `logs/50-review.md` (§6.3a); the four checks above are confirmed
in `logs/50-review.md`.
This is now the contract. Deviating from it later requires a logged decision.

### 6.4 Phase 4 — Implementation

Dispatch **Opus coding agents**, one per work package, each in its own worktree:

```bash
cd /home/azureuser/linopy-gpu
git worktree add /home/azureuser/linopy-gpu/dev-scripts/cuopt/worktrees/<wp-id> -b wp/<wp-id> feat/cuopt-solver
# ... agent works ...
git worktree remove /home/azureuser/linopy-gpu/dev-scripts/cuopt/worktrees/<wp-id>
```

**Worktree lineage — verify, never assume.** Each slice brief carries the **expected base
commit SHA**, and the coding agent's first action is:

```bash
git rev-parse --short HEAD && git log --oneline -1
git merge-base --is-ancestor <BASE_SHA> HEAD && echo "LINEAGE OK" || echo "WRONG LINEAGE"
```

On `WRONG LINEAGE` — or a missing base SHA in the brief — the agent **stops, writes nothing,
and reports.** It must not adapt to the tree it finds or "fix" imports to match. A worktree
cut from the wrong base (e.g. upstream `master`, which has no cuOpt code at all) produces
work that looks entirely plausible and is built against a tree that does not exist.
Reporting a wrong base is a *successful* outcome for this step.

Note we create worktrees **explicitly from `feat/cuopt-solver`** in the command above, rather
than relying on `Agent(isolation: "worktree")`. That is deliberate: Claude Code's
`worktree.baseRef` defaults to `fresh`, which cuts from `origin/<default-branch>`. Pinning
`"baseRef": "head"` in `.claude/settings.json` would fix that — but `.claude/` is **not**
gitignored in this repo (verified), so the setting would land in the PR. Explicit worktree
creation gets the same guarantee with nothing to clean up.

**All work must be committed before dispatching.** Worktrees branch from the last *commit*,
so uncommitted work is invisible to sub-agents. You never make that commit yourself (I11): if
`git status` is not clean on `feat/cuopt-solver` when you are about to cut a worktree, dispatch
an Opus agent to commit it and wait for that to land first. A dirty base branch at dispatch
time is a harness bug — fix it, never work around it by letting the lead commit.

**Worktrees see only tracked files.** `dev-scripts/` is gitignored, so inside
`/home/azureuser/linopy-gpu/dev-scripts/cuopt/worktrees/<wp-id>` there is no `dev-scripts/` directory at all. The charter, the templates
and the logs are reachable **only by absolute path** (I3). A brief that hands a coding agent a
relative `dev-scripts/...` path sends it to a file that does not exist — and an agent that
cannot read the charter will improvise.

**Collision rule — the important one.** The bulk of this change lands in
`linopy/solvers.py` (PR #516 put 257 of its 771 lines there). Parallel agents editing
that one file **will** conflict. Therefore:

1. **Serialize** the `solvers.py` core work package. It goes first, alone, and merges
   into `feat/cuopt-solver` before anything else starts.
2. **Then parallelize** the genuinely disjoint packages: `io.py` + `model.py`,
   `solver_capabilities.py`, tests, docs, packaging.
3. Any two packages that touch the same file run **sequentially**, never concurrently.
4. **QP is last** (§4.5). It is dispatched only after the LP+MILP core has merged and passes
   Gate 4's checks — never alongside the `solvers.py` core package. This keeps a QP problem
   from destabilising a working LP/MILP implementation.
5. **Packaging is not a trailing chore.** The §4.6 extra is a first-class work package with a
   real acceptance test (Q13's clean-venv install), not a two-line `pyproject.toml` edit tacked
   onto the docs slice. It touches only `pyproject.toml`, so it parallelises freely — run it
   early enough that a clean-venv install failure surfaces while there is still time to fix it.

Each coding agent's brief: implement exactly its work package from `FINAL.md`; surgical
diffs (I4); mirror cuPDLPx's idiom; run `/home/azureuser/.local/bin/ruff check --fix .`,
`.venv/bin/mypy`, and the relevant
tests before reporting; paste output in its log entry.

#### 6.4a Commit discipline

**Scope per commit — mirror PR #516.** Strip its merge noise and #516's real feature arc
is six substantive commits, one concern each, 1–5 files apiece:

| # | Commit | Concern |
|---|---|---|
| 1 | `Implement interface to cupdlpx solver and required IO` | `solvers.py` + `io.py` + `model.py` together — the feature lands as one coherent slice |
| 2 | `Update tests` | test files only |
| 3 | `Add to docs & add package dependency` | docs + `pyproject.toml` |
| 4 | `Add comment for release notes` | `release_notes.rst` |
| 5 | `Correction: add support for equality constraints (same l & u bounds)` | a targeted follow-up fix |
| 6 | `Parameterize solution tolerance in tests to allow different standards for CPU and GPU solver precisions` | a targeted follow-up refinement |

Aim for the same shape: **do not** split into twenty micro-commits, and **do not**
squash the whole feature into one. Note that commit 1 spans three production files —
so the work-package boundaries in §6.4 are about *worktree isolation*, not about
commit boundaries. Several work packages may land as one commit; the lead decides at
merge time.

**Message style.** Match #516's author: plain imperative, sentence case, **no**
conventional-commit prefix, no body unless the commit does several small related
things — then a `-` bullet list, as #516's `Improve cuPDLPx solver integration` does.

> **Note on I5.** Recent `master` (`fix:`, `feat:`, `fix(sos):`) and the maintainer's
> own later commits on #516 (`fix:`, `doc:`) use conventional prefixes — #516 is
> internally inconsistent on this. The human lead has explicitly directed #516's style,
> so that **overrides I5 for commit messages only**. Log this in `logs/30-decisions.md`.
> If a maintainer asks for conventional prefixes at review time, switch without
> escalating.

**No merge commits.** #516 carries eight `Merge branch 'master' into master` commits.
Do not reproduce that. Keep history linear: rebase work-package branches onto the tip
of `feat/cuopt-solver` and fast-forward them in.

```bash
git rebase feat/cuopt-solver wp/<wp-id>
git switch feat/cuopt-solver && git merge --ff-only wp/<wp-id>
```

**Authorship — Claude is the author, and the author string names the model (I11).** Per
`AGENTS.md` rule 1, AI-generated work must be visibly marked; authoring these commits as
Claude is how that mark is carried in the history. **Each committing agent stamps its own
model** — it is the only party that knows:

```bash
git commit --author="Claude Opus 5 <noreply@anthropic.com>"  -m "<message>"   # Opus agent
git commit --author="Claude Fable 5 <noreply@anthropic.com>" -m "<message>"   # Fable agent
```

**The rule is accuracy, not uniformity.** Both strings are correct provenance; the only
false statement is one model claiming to be another. Therefore:

- **Dispatch every agent with `model: "opus"`** (§8) — that is the intended configuration,
  and a run where every author reads `Claude Opus 5` is the expected outcome.
- **But a Fable commit is not a defect.** If an agent reports that it ran on Fable, its work
  stands. Never reset, revert or re-dispatch a slice over model identity — that discards good
  work to fix a *label*. Correct the label instead (below).
- **You do not commit** on `feat/cuopt-solver` — because you do not write production code, not
  because of your model. Your integration steps are `git rebase` (preserves the original
  author) and `git merge --ff-only` (creates no commit), neither of which mints authorship.
  Anything needing a fresh commit — a release-notes line, a one-character typo fix — goes to
  a dispatched agent, however trivial it feels.
- A reviewer or planner agent **never** commits, regardless of model.
- **Never dispatch a committing agent as a fork.** `subagent_type: "fork"` inherits the
  parent's model and ignores `model: "opus"`. The harm is not that the agent is Fable — it is
  that it will believe it is Opus and stamp that, which is the one failure this section exists
  to prevent. Every dispatch is a fresh agent with an explicit `model: "opus"`.
- **Have the agent declare its own model** in slice-report item 6, beside the `git log`
  output, and stamp the author string to match. Self-reported, therefore weak evidence — but
  a disagreement between what the agent says it is and what you dispatched is a real signal,
  and it costs one line to get.

**Relabelling a mis-stamped commit — never a reset.** The work is kept; only the author
string changes. Dispatch an agent to run, for the offending commit `<bad>`:

```bash
git checkout <bad>
git commit --amend --no-edit --author="Claude <actual model> <noreply@anthropic.com>"
# if the Co-Authored-By trailer ALSO names the wrong model, --no-edit cannot fix it:
# rewrite the message in the same amend instead —
#   git commit --amend --author="Claude <actual model> <noreply@anthropic.com>" \
#     -m "<original subject/body with the trailer corrected>"
git rebase --onto HEAD <bad> feat/cuopt-solver
```

Non-interactive by design — `git rebase -i` is unavailable here. Re-run the §6.4a checks
afterwards, since the rebase rewrites every later hash.

The **committer** stays the human's configured git identity — GitHub then renders
"Claude Opus 5 authored and <human> committed", which is exactly the provenance we
want. End every message with the trailer as well, **naming the same model as the author
field**:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

The trailer is deliberately redundant with the author field: if the PR is
**squash-merged**, per-commit author fields are discarded but trailers in the message
body survive into the squashed commit. Belt and braces on provenance.

Verify before Gate 4:

```bash
# every author must be a real Claude model — prints ONLY offenders, so empty = pass
git log master..feat/cuopt-solver --pretty='%h %an' | grep -vE ' Claude (Opus|Fable) 5$'
git log master..feat/cuopt-solver --merges                   # must be empty
# the trailer must be present on every commit (survives a squash-merge)
git log master..feat/cuopt-solver --pretty='%h%n%b' | grep -cE 'Co-Authored-By: Claude (Opus|Fable) 5'
# author and trailer must name the SAME model — prints offenders, so empty = pass
# NB: tformat:, not format: — format: omits the final newline, so `read` silently drops
# the LAST commit, which is exactly the one a late fix agent just made.
git log master..feat/cuopt-solver --pretty=tformat:'%H %an' | while read h a; do
  git log -1 "$h" --pretty=%b | grep -q "Co-Authored-By: $a " || echo "MISMATCH $h ($a)"
done
```

A commit authored by Fable is **not** an offender here. Only a non-Claude author, a missing
trailer, or an author/trailer disagreement is.

Run these again at **Gate 6**, not only at Gate 4 — Phases 5 and 6 add fix commits after
Gate 4 has passed, and those are exactly the ones at risk of slipping through unpinned.

**Gate 4:** every work package merged into `feat/cuopt-solver`; `ruff` and `mypy`
clean; `pytest` and `pytest --run-gpu` both green with pasted output.

### 6.5 Phase 5 — Independent review

Dispatch a **fresh reviewer** that wrote none of the implementation. It verifies:

- **Correctness** — especially dual/reduced-cost signs and the status mapping.
  Cross-check cuOpt against HiGHS on identical models; results must agree.
- **Tests** — `pytest` and `pytest --run-gpu` genuinely green, not skipped-and-called-passing.
  Check that GPU tests actually *ran*.
- **Not overly bulky** — §9 yardstick. Flag duplication, needless abstraction, dead code.
- **Examples and docs** — rebuild with the docs venv
  (`/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin/sphinx-build -b html doc doc/_build/check`) and diff the result against
  `logs/05-baseline-docs.txt`. **No new failure and no new warning.** This executes the example
  notebooks (§6.0). A notebook that was already red at baseline stays out of scope (I13).
- **Senior-engineer standard** — naming, docstrings, error messages, type annotations,
  and consistency with current `master` (not with PR #516).
- **No stray files** — `git status --porcelain` clean; nothing from `dev-scripts/` staged.

Findings → fix agents → re-review. **Cap: 3 loops**, then E5. Fix commits follow
§6.4a — same style, same Claude authorship (I11).

**Gate 5:** review returns clean.

### 6.6 Phase 6 — Style and documentation conformance

The last gate on the *code* — after it only the harness record (§6.7) and the handoff
(§6.8) remain. Dispatch a **fresh reviewer that wrote none of the
implementation and is not the Phase 5 reviewer** — this is a second pair of eyes on a
different axis, not a repeat of Phase 5. Phase 5 asked *is it correct?*; Phase 6 asks
*does it look like it was always part of this repo, and is it explained?*

`ruff` and `mypy` passing (Gate 4) is **necessary but not sufficient** here. Lint
catches formatting; this phase catches idiom.

#### 6.6a Code style conformance

Review the new code **side by side** with `cuPDLPx` (`linopy/solvers.py:4156`) and one
non-GPU solver such as `Highs` (`:1544`). Check:

- **Class shape** — ClassVar ordering, method ordering, and the override set match the
  sibling solver classes (§5.1). No methods the siblings do not have without a reason.
- **Naming** — module-level lazy import, enum member, class name, helper names all
  follow the established pattern.
- **Docstrings** — numpydoc style, matching the depth of sibling solvers. The class
  docstring documents the notable solver options, as cuPDLPx's does.
- **Type annotations** — `Solver[None]` / `EnvType` / `ClassVar` usage consistent with
  siblings; no stray `Any` where a real type exists next door.
- **Error messages** — phrasing and specificity match the existing
  `NotImplementedError` / `ValueError` messages elsewhere in `solvers.py`.
- **Comment density** — matches the surrounding file. No tutorial comments explaining
  Python to the reader.
- **Import placement** — `TYPE_CHECKING` block, `_LazyModule` line, and
  `_SOLVER_PROBE_ORDER` entry all in the conventional spots.
- **Test style** — parametrization idiom matches the existing tests, and linopy's own
  assertions (`assert_linequal`, `assert_varequal`, …) are used where appropriate, per
  `AGENTS.md`.
- **No stylistic outliers** — anything a reviewer would flag as "this file does not
  read like the rest of linopy".

#### 6.6b Documentation completeness

Verify each surface **actually explains cuOpt**, not merely name-drops it:

| Surface | Must contain |
|---|---|
| `doc/gpu-acceleration.rst` | Primary home: install command, hardware/driver requirements, usage example, supported problem classes, known limitations. |
| `doc/prerequisites.rst` | The one-command extra install (§4.6), **Linux-only** and the driver/CC requirement stated alongside it. `pypi.nvidia.com` appears only as an optional fallback, if at all — never as a required step, and never with `--index-strategy` (§4.2). |
| `doc/contributing.rst` | Any GPU-testing note cuOpt adds beyond the existing `--run-gpu` text. |
| `doc/index.rst`, `README.md` | cuOpt present in the solver listings. |
| `doc/api.rst` | `to_cuopt` and any other new public surface. |
| `doc/release_notes.rst` | An entry under the unreleased section. |
| `pyproject.toml` | The chosen extra is documented wherever extras are explained. |
| Install instructions everywhere | The **one-command** install (§4.6) is what a user sees first: `pip install 'linopy[<extra>]'`. Manual `cuopt-cu12` instructions, if kept at all, are a fallback below it — never the headline. State **Linux-only** and the driver/CC requirement next to the command. Warn that the bare `cuopt` package on PyPI is a different project (§4.2). |
| `examples/using-solvers.ipynb` | Currently mentions cuPDLPx — decide and log whether cuOpt belongs here too. |

The docs must state plainly: **supported** problem classes (LP, MILP, and QP — stating
honestly if QP did not ship, and why); **unsupported** surfaces (SOS, semi-continuous, indicator
constraints, whatever Phase 1 confirmed); **requirements** (CC ≥ 7.0, driver ≥
525.60.13); and the **`cu12` vs `cu13`** caveat.

**Mechanical gap check** — run this and justify every difference:

```bash
gh pr view 516 --repo PyPSA/linopy --json files -q '.files[].path' | sort > /tmp/pr516.txt
git diff --name-only master...feat/cuopt-solver | sort > /tmp/cuopt.txt
diff /tmp/pr516.txt /tmp/cuopt.txt
```

Any documentation file PR #516 updated that this branch did not is a **gap to justify
or close** — not a difference to wave through. (Non-doc differences are expected and
fine; §5.2 still applies.)

Finally, **build the docs in the cuOpt-free docs venv** (§6.0) and confirm the cuOpt content
renders with no broken references and **no warning that is not already in
`logs/05-baseline-docs.txt`**. The build must succeed *without* cuOpt installed — that is what
ReadTheDocs will do.

Findings → fix agents → re-review. **Cap: 3 loops**, then E5.

**Leak check — run this at Gate 6, and again before the push in §6.8.** I3 says the harness
can never reach the PR because `dev-scripts/` is gitignored. That protects against accident,
not against `git add -f` or a file written outside `dev-scripts/`. Verify it, do not assume it:

```bash
git fetch upstream
# every path the PR would add, minus the ones that must never appear:
git diff --name-only upstream/master...feat/cuopt-solver -- dev-scripts/ .claude/ PR_BODY.md
#   -> must print NOTHING
git diff --name-only upstream/master...feat/cuopt-solver | grep -Ei 'PROJECT_LEAD|HARNESS_REVIEW|-TEMPLATE|/logs/|/plans/|/worktrees/'
#   -> must print NOTHING
```

Any hit is **blocking** and is never fixed by amending `.gitignore` — find the commit that
added the file (`git log --oneline --diff-filter=A -- <path>`) and dispatch an Opus agent to
remove it from history, then re-run the §6.4a checks.

**Gate 6:** style review clean; every row of the §6.6b table satisfied or its absence
justified in `logs/50-review.md`; docs build shows no regression against the Gate 0 baseline;
the §6.4a authorship and no-merge-commit checks re-run clean over the **full** branch
(including Phase 5/6 fix commits); the leak check above prints nothing.

### 6.7 Phase 7 — Publish the harness record (you do this yourself)

Colleagues need to see **how** this was built — the charter, the review rounds, the plans,
the logs. The linopy PR must contain **none** of it. Both are satisfied by publishing the
harness to an **orphan branch on the fork**, `origin` = `mal84emma/linopy-gpu`.

**Why orphan, specifically.** A PR's diff is every commit on the head branch that is not on
the base. An orphan branch shares no history with `master`, so no ancestry path exists by
which it can end up behind `feat/cuopt-solver` — not even via a stray merge. A branch cut
from `master` would be one careless command from leaking. Do not cut it from `master`.

Run only after the Gate 6 leak check is clean:

```bash
D=/home/azureuser/linopy-harness
rm -rf "$D" && mkdir -p "$D" && cd "$D"
git init -b harness/cuopt
S=/home/azureuser/linopy-gpu/dev-scripts/cuopt
cp -r "$S"/PROJECT_LEAD.md "$S"/HARNESS_REVIEW.md "$S"/templates "$S"/plans "$S"/logs .
# the cp above is an allowlist, so worktrees/ and .venv-docs/ cannot come across.
# prove that rather than assume it (§9, "a harness check that cannot fire"):
find . -name worktrees -o -name '.venv*' | grep . && echo "UNEXPECTED — stop" || echo clean
git add -A
# --author is REQUIRED here: the default identity is the human's, which would unmark
# AI-generated content (AGENTS.md rule 1). Substitute your ACTUAL model name — never "Opus"
# unless you are actually running on Opus (I11).
git commit --author="Claude <your-real-model-name> <noreply@anthropic.com>" \
  -m "docs: cuOpt implementation harness — charter, plans, review rounds, run logs"
git remote add origin https://github.com/mal84emma/linopy-gpu.git
git push -u origin harness/cuopt
```

Then **prove the isolation** from the main repo — this is the check, not the intention:

```bash
cd /home/azureuser/linopy-gpu && git fetch origin
git merge-base origin/harness/cuopt master      # must print nothing, exit 1 = no shared history
git log --oneline origin/harness/cuopt ^master  # every commit unique to the record branch
```

Also required:

- **Write `README.md` on that branch.** The charter is full of `/home/azureuser/...` absolute
  paths that are load-bearing for the agents and meaningless to a reader. One short paragraph:
  what this branch is, that it is the operating charter for an autonomous multi-agent run,
  that the absolute paths are deliberate, and which file to read first.
- **Push to `origin` only.** Never `upstream`. `gh`'s default repo in this checkout resolves
  to `PyPSA/linopy` — always name the remote explicitly.
- **Authorship here is the lead's own.** This is the one commit you make yourself, and I11
  applies to it the same way it applies everywhere: stamp the model that actually wrote it —
  you, so `Claude Fable 5` (see the model-assignment note at the top). Do not write
  `Claude Opus 5`: no Opus agent touches this branch.
- **Sanity-read before pushing.** These logs are going somewhere colleagues can read. Check
  for absolute paths that reveal more than intended, and for anything a raw model transcript
  picked up that should not be public.
- Record the branch URL — it goes in `logs/99-handoff.md` (§6.8).

If the run ends early via a hard-stop escalation, publish anyway. A record of a run that
stopped is more useful to a colleague than no record.

**Gate 7:** `harness/cuopt` pushed to `origin`; `git merge-base` against `master` prints
nothing; the branch has a `README.md`; the URL is recorded in `logs/00-lead-journal.md`.

### 6.8 Phase 8 — Handoff (you do this yourself)

- Re-run the §6.6b leak check. Push `feat/cuopt-solver` **to `origin` only** — never
  `upstream`.
- Write `/home/azureuser/linopy-gpu/dev-scripts/cuopt/PR_BODY.md`: the `<!-- HUMAN: write your intent here -->`
  placeholder first, then `> [!NOTE]` + the AI-generated technical section, with
  benchmark/log detail inside `<details>`.
- Confirm `git log --stat` shows **zero** files under `dev-scripts/` — the leak check above
  is the authoritative version of this.
- Include the Phase 7 harness-record branch URL in the handoff. It does **not** go in
  `PR_BODY.md`: if the human chooses to link it from the PR, `AGENTS.md` rule 1 applies and
  the note alert is theirs to write, not yours.
- Write `logs/99-handoff.md` from `/home/azureuser/linopy-gpu/dev-scripts/cuopt/templates/handoff-TEMPLATE.md` (I14). In one-shot mode
  it **opens** with the two sections the human needs before anything else:
  1. **Unreviewed autonomous decisions** — every `UNREVIEWED` entry from
     `logs/30-decisions.md`, each with the option taken, the alternative rejected, and where
     in the diff it shows up. This is the audit list; put it first.
  2. **Where this run is weakest** — your honest assessment of the least-confident parts of
     the implementation, ranked. Name the things you would look at first if you were
     reviewing someone else's work. An empty or self-congratulatory version of this section
     is a failure of the handoff.

  Then: what shipped, what did not, every open decision, every escalation and its resolution,
  and every entry in `logs/45-known-issues.md`.
- **Stop. Do not open the PR** (§2.1).

---

## 7. Logging spec

All under `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/`. Markdown. Append-only —
never rewrite history; correct with a new dated entry.

Listed in file-name order — which is also roughly the order they come into existence.

| File | Owner | Contents |
|---|---|---|
| `00-lead-journal.md` | you | Chronological. Charter re-reads, dispatches, gate passes with evidence pointers, and the frozen-plan sha256 (§6.3a). |
| `05-baseline.md` | Phase 0 | The untouched-`master` baselines: pytest, pytest --run-gpu, docs build. Pass/fail counts, every failure named. Raw output in the sibling `05-baseline-*.txt` files. |
| `10-exploration.md` | Phase 1 | Question 0 plus answers 1–13, each with pasted output. |
| `15-verification.md` | Phase 3 | The ladder checks with **frozen numeric thresholds** (§6.3a). Edited only by returning to Gate 3; hash-frozen and `chmod 444`. |
| `20-progress.md` | all | Work-package status board: id, owner, worktree, branch, state, evidence. |
| `30-decisions.md` | all | Every non-obvious call. **Decision / options considered / evidence / why / who / when.** Including the ones you resolved without escalating. |
| `40-blockers.md` | all | Difficulties hit and how they were overcome — or that they were not. Include the resolve-first attempts from §3.3. |
| `45-known-issues.md` | all | Pre-existing linopy defects noticed and deliberately **not** fixed (I13): location, one line, who found it. Plus anything already red at the Gate 0 baseline. |
| `50-review.md` | Phases 3, 5, 6 | Review rounds, findings, dispositions. Also the second recorded copy of the frozen-plan sha256. |
| `60-conformance.md` | Phase 5 | Every frozen check executed, actual vs threshold, pasted output, and an **Escalations** section. |
| `90-escalations.md` | you | §3.4 format, plus the human's answer once it arrives. |
| `99-handoff.md` | you | Final summary. Opens with the two §6.8 sections. |

### 7.1 Templates

Artifacts in this table are written from `/home/azureuser/linopy-gpu/dev-scripts/cuopt/templates/` — never freehand (I14):

| Template | Produces |
|---|---|
| `exploration-TEMPLATE.md` | `logs/10-exploration.md` (Phase 1) — read by all five planners |
| `verification-TEMPLATE.md` | `logs/15-verification.md` (Phase 3, frozen at Gate 3) |
| `decisions-TEMPLATE.md` | `logs/30-decisions.md` (all phases, append-only) |
| `review-TEMPLATE.md` | each round appended to `logs/50-review.md` |
| `slice-report-TEMPLATE.md` | each coding agent's return report (§8.1) |
| `conformance-TEMPLATE.md` | `logs/60-conformance.md` (Phase 5) |
| `handoff-TEMPLATE.md` | `logs/99-handoff.md` (Phase 8) — read by the human |

The other log files (`00-lead-journal`, `05-baseline`, `20-progress`, `40-blockers`,
`45-known-issues`, `90-escalations`) are freehand; `90-escalations` follows §3.4's inline
format. I14 binds this table, not every file in §7.

Two rules carried in every template's header, worth repeating: **keep the headings exactly**
(later agents parse these files by heading), and **write "None" rather than deleting a
section** so absence is explicit rather than ambiguous.

`logs/30-decisions.md` opens with a **"Rules for later agents" block, hard-capped at 10
lines** — actionable instructions, not summary. It is the only part of the decision log a
later agent is *required* to read; the rest is read as needed. This is what keeps context
growth bounded as the log accumulates.

Every sub-agent brief must end with: *"Append your findings to the relevant log files
under `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/` before you report back."*

These files are **never** part of the PR (I3).

---

## 8. Standing sub-agent brief

**Dispatch parameter, every time:** `model: "opus"`, and never `subagent_type: "fork"` (see
the model-assignment note at the top of this document). This is your side of the dispatch;
the brief below is the agent's side. Getting the brief right and the model wrong produces a
well-briefed agent on the wrong model, which is the failure §6.4a cannot detect after the fact.

Prepend to every dispatch:

> You are working on adding NVIDIA cuOpt support to linopy at `/home/azureuser/linopy-gpu`.
> Read `/home/azureuser/linopy-gpu/dev-scripts/cuopt/PROJECT_LEAD.md` (the charter) and `/home/azureuser/linopy-gpu/AGENTS.md` first, and obey both.
> **Use absolute paths for everything under `dev-scripts/` — it is untracked, so those paths do not
> exist relative to a worktree checkout.** Templates: `/home/azureuser/linopy-gpu/dev-scripts/cuopt/templates/`. Logs: `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/`.
> Treat §4 of the charter as verified fact — do not re-derive it; contradict it only with
> reproducible evidence.
> Your model of correctness is the **current state of `master`**, not PR #516.
> Do not escalate to the human — report to the project lead agent.
> Never claim a test passes without pasting its output.
> Append your findings to the relevant files under `/home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/` before reporting back.
> Scope: <the specific work package>. Do nothing outside it.

### 8.1 Return contract (coding agents)

Every coding agent reports using `/home/azureuser/linopy-gpu/dev-scripts/cuopt/templates/slice-report-TEMPLATE.md` — **verbatim, in
this order**, no substitutions:

0. **Worktree lineage** — the `git log --oneline -1` output and `LINEAGE OK` / `WRONG LINEAGE`.
   One line, never omitted; you integrate on that basis.
1. **Deliverables** — each file added/changed, one line each.
2. **Requirement IDs → what was built** — flagging any it could not satisfy, and why.
3. **Tests** — each test added, which verification-plan row it proves, and **pasted** output.
4. **Escalations** — any frozen-threshold miss with verbatim numbers, any brief-vs-plan
   discrepancy, any invariant it could not respect. "None" if none.
5. **Bugs noticed, not fixed** (I13) — location + one line each. "None" if none.
6. **Commits** — `git log --pretty='%h | %an | %s'`, plus **the model you are running on**.
   The author string and the `Co-Authored-By` trailer must both name *your* model, whatever it
   is. Running on Fable is fine and changes nothing about the work — commit normally, stamp
   `Claude Fable 5`, and say so here. The only error is claiming a model you are not.

Close every brief with: *"Do not summarise away failures — a red test reported honestly is
acceptable work, a hidden one is not."*

A report missing item 0, or with an unpasted item 3, is **rejected unread**. Send it back.

---

## 9. Risk register

| Risk | Why it bites | Mitigation |
|---|---|---|
| **Dual sign convention wrong** | Silent wrong answers — tests on objective value alone will not catch it. | Phase 1 item 2; Phase 5 cross-checks against HiGHS. Highest-priority risk. |
| **`solvers.py` merge collisions** | 4.5k-line file, most of the diff lands there. | §6.4 serialization rule. |
| **Copying stale patterns from PR #516** | It predates the current registry, lazy availability, and v1 semantics. | I5; the standing brief says so explicitly. |
| **Log capture silently empty** | cuOpt logs from C++; `redirect_stdout` will not see it. | Phase 1 item 4. |
| **GPU tests skipped but reported green** | `--run-gpu` gating makes a no-op look like a pass. | Phase 5 must confirm tests actually ran. |
| **Bulk creep** | cuOpt's larger surface (MILP, QP) invites over-engineering. | Yardstick: PR #516 was **+771/−62 / 15 files** for a simpler solver. cuOpt should land in the same order of magnitude. Materially past ~1.5× warrants a hard look, not an automatic escalation. |
| **`uv sync` wipes cuOpt mid-run** | Not yet in `pyproject.toml`, so sync prunes it — and the suite then goes *green by skipping*, which reads exactly like a pass. A silent false-green, not a crash. | I9 (`.venv/bin/<tool>`, never bare `uv run`/`uv sync`); resolved permanently once the §4.6 extra lands. |
| **cuOpt ships but is not installable** | The solver works on this machine and no user can get it — the feature is then decorative. Easy to miss, because every check in this run happens in a venv where cuOpt is *already* installed by hand. | §4.6 is a requirement with a DoD line; Phase 1 Q13 proves it from a **clean venv**, which is the only environment that can detect this. |
| **The GPU extra leaks into a default install** | A CPU user runs `pip install linopy[solvers]` and starts a 6.3 GB GPU download, or macOS/Windows CI fails to resolve at all. | §4.6 constraints 1–2; Q13 checks `solvers` does not pull cuOpt and cross-compiles the marker set for Windows/macOS. |
| **Benchmarks break** | pandas pinned `==3.0.5` in that extra; cuOpt forces 3.0.3. | §4.4 — do not run benchmarks in this venv. |
| **Docs name-drop cuOpt without explaining it** | Easy to add a solver to a list and call the docs done; users then cannot install or use it. | §6.6b's table and the mechanical gap check against PR #516. |
| **Commit history drifts from #516's shape** | Micro-commits or one giant squash both lose the reviewable arc; a stray merge commit reintroduces #516's noise. | §6.4a, with the two `git log` checks at Gate 4. |
| **An agent widens a tolerance to go green** | cuOpt is a tolerance-based first-order method, so "nearly optimal" is normal and nudging an `rtol` looks harmless. It is how a sign bug ships. | I12 + §3.6, carried verbatim in every brief; frozen-plan **sha256** check at Phase 5 (§6.3a) against two independently recorded digests. |
| **Differential test with a baked-in expectation** | Degrades to a regression test against an unverified baseline; a systematic sign error then passes forever. | §6.3a: V1 must call HiGHS **live**, in-process. Phase 5 reviewer checks for hard-coded expectations. |
| **Worktree cut from the wrong base** | Upstream `master` has no cuOpt code; a slice built there looks plausible and is worthless. | §6.4 lineage check, stop-on-mismatch. |
| **No human read the plan** | Gate 3's human review is removed in one-shot mode; an internally consistent but wrong plan gets built. | Strengthened Gate 3 checks (§6.3); unreviewed decisions surfaced first in the handoff (§6.8). Residual risk — accepted by the human lead. |
| **Entry point chosen by inertia** | §4.3's verified snippet is the easy default; the best entry point may differ, and a wrong one is expensive to unwind after Phase 4. | Phase 1 Q0 requires prototyping ≥2 candidates with measurements; Gate 3 logs the choice as a decision. |
| **QP Q-matrix convention wrong** | Full-vs-triangular and the factor-of-2 diagonal convention differ between solvers; a mismatch yields a silently wrong objective that still "solves". | Phase 1 Q12; QP differential test against a QP-capable CPU solver. |
| **QP quietly dropped** | It is the last work package, so schedule pressure lands on it; "stretch goal" framing invites skipping. | §4.5 makes it committed; DoD requires shipped-or-escalated; handoff must state it. |
| **Planner groupthink** | Five agents converging on the same flawed plan. | Independence (I6), stated verbatim in each planner brief (§6.2); the selector-reviewer must actively look for what all five missed. |
| **A harness check that cannot fire** | An enforcement mechanism that silently no-ops reads exactly like a passing check — worse than having no check, because it buys false confidence. The `git diff` freeze on an untracked file was one; assume there are others. | Before relying on any check, **make it fail on purpose once** and confirm you see the failure. If you cannot make it fail, it is not a check. |
| **The harness record leaks into the PR** | The whole point of publishing the charter is that colleagues see it and linopy does not. A record branch cut from `master` instead of orphaned, or a `git add -f` on a log, puts it in the diff. | §6.7 builds the branch in a *separate* repo directory so it cannot share history, and proves it with `git merge-base`; the Gate 6 leak check (§6.6b) greps the PR diff for harness paths by name, not just by directory. |
| **False authorship provenance** | `--author="Claude Opus 5"` on a commit made by another model is an untrue claim in the permanent history, and provenance is the whole reason `AGENTS.md` rule 1 asks for the mark. The realistic route in is a forked agent that believes it is Opus, not a deliberate misstatement. | I11 pins *accuracy*, not one model: each agent stamps its own name, both `Claude Opus 5` and `Claude Fable 5` pass, and author must agree with trailer. No forked dispatches (§6.4a). The check re-runs at Gate 6, not just Gate 4. A mis-stamp is relabelled by amend-and-rebase — **never** by discarding the work. |

---

## 10. Pre-escalation checklist

Before you send anything to the human, confirm all five:

- [ ] All five resolve-first steps (§3.3) attempted and logged.
- [ ] It is a real E1–E6 trigger, not a §3.2 non-escalation.
- [ ] My question is precise and answerable — not "what should I do?"
- [ ] I have a recommendation, not just a question.
- [ ] I have identified what continues while this is blocked, and I am doing it.
