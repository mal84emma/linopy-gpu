# cuOpt support — decision log

## Rules for later agents (≤10 lines — mandatory reading for every agent dispatched after this)

1. Use absolute paths for everything under `dev-scripts/` — it does not exist inside worktrees.
2. Never run `uv sync` or bare `uv run`; invoke `.venv/bin/pytest`, `.venv/bin/mypy`, `/home/azureuser/.local/bin/ruff` (I9).
3. When using `uv pip install`, always pass `--python /home/azureuser/linopy-gpu/.venv/bin/python` — bare `uv pip` targets an ambient conda env (see D1 evidence).
4. Current `master` outranks PR #516 wherever they disagree (I5).
5. Commit messages: plain imperative, no conventional-commit prefix (D2); author string = your actual model (I11).
6. Thresholds in logs/15-verification.md are frozen numbers — report misses verbatim, never widen (I12, §3.6).
7. Pre-existing linopy defects: record in logs/45-known-issues.md, never fix (I13).
8. Docs build uses the cuOpt-free venv at dev-scripts/cuopt/.venv-docs (RTD has no GPU).
9. Docs builds need pandoc: prepend `PATH=/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin:$PATH` (D3).

---

## D1 — Explicit `--python` on every `uv pip install`

*Phase: 0 · Decided by: lead · 2026-08-22 · Status: decided*

**Decision:** All `uv pip install` invocations in this run pass `--python /home/azureuser/linopy-gpu/.venv/bin/python` (or the target venv's python) explicitly.

**Options considered:** A) rely on cwd-based venv discovery — B) explicit `--python` flag.

**Evidence:** Bare `uv pip install 'cuopt-cu12==26.8.*'` from a background shell resolved against `/anaconda/envs/azureml_py38` (Python 3.10) and failed with "no wheels with a matching Python implementation tag (cp310)". The project venv is Python 3.11.15 where cp311 wheels exist (§4.2).

**Why this one:** Background shells do not inherit the interactive cwd/venv context reliably; the explicit flag is deterministic.

**Escalated?** no

---

## D2 — Commit message style: PR #516's plain imperative, no conventional prefix

*Phase: 0 · Decided by: lead (pre-directed by human lead, charter §6.4a) · 2026-08-22 · Status: decided*

**Decision:** Commit messages follow PR #516's author style — plain imperative, sentence case, no `feat:`/`fix:` prefix — overriding I5 for commit messages only.

**Options considered:** A) conventional-commit prefixes as on recent `master` — B) #516's plain style.

**Evidence:** Charter §6.4a note on I5: the human lead explicitly directed #516's style; recent `master` uses prefixes, so #516 is internally inconsistent with it.

**Why this one:** Explicit human-lead direction outranks I5.

**Escalated?** no

---

## D3 — Provide pandoc inside the docs venv via pypandoc_binary

*Phase: 0 · Decided by: lead · 2026-08-22 · Status: decided*

**Decision:** Install `pypandoc_binary==1.17` into `.venv-docs` and symlink its bundled binary to `.venv-docs/bin/pandoc`; every docs build prepends `.venv-docs/bin` to PATH.

**Options considered:** A) `sudo apt install pandoc` (system-wide change) — B) contained binary inside the gitignored docs venv.

**Evidence:** `which pandoc` empty system-wide; nbsphinx aborted the baseline build with `PandocMissing in coordinate-alignment.nblink`. After fix: `pandoc 3.9` runs.

**Why this one:** Zero system footprint, same spirit as the §4.4 pre-authorised docs venv; deleted with the venv at handoff. Docs-build agents must use `PATH=/home/azureuser/linopy-gpu/dev-scripts/cuopt/.venv-docs/bin:$PATH`.

**Escalated?** no

---

## D4 — No polling timers; wait on task notifications only

*Phase: 0→1 · Decided by: lead (after human-lead correction) · 2026-08-22 · Status: decided*

**Decision:** Long-running commands are launched as background tasks and the lead acts only on their completion notifications. No `sleep`-based polling timers. Elapsed time is read from tool-reported timers (e.g. pytest's own duration), never inferred from wait-loop counts.

**Options considered:** A) sleep-timer polling loops — B) notification-driven waiting.

**Evidence:** ~50 accumulated sleep shells; perceived multi-hour run vs pytest's actual 464s; notification storms obscuring real progress. No evidence corruption found (all baseline files carry complete single-process summaries).

**Why this one:** The harness already re-invokes on completion; polling adds cost, noise, and a false timeline that could trigger wrong interventions (e.g. killing a healthy run as "hung").

**Escalated?** no

---

## D5 — Phase 1 split across two spike agents with disjoint output files; lead merges

*Phase: 1 · Decided by: lead · 2026-08-22 · Status: decided*

**Decision:** Two Opus spike agents run in parallel: A answers Q0 + items 1,2,3,5,6,12 in `spikes/exploration-A.md` (full template, B's sections marked `PENDING-B`); B answers items 4,7,8,9,10,11,13 in `spikes/exploration-B.md` (its sections only, exact template headings). The lead mechanically merges both into `logs/10-exploration.md` — assembly only, no content authored by the lead.

**Options considered:** A) one agent for all 14 items (huge context, slow) — B) two agents sharing `10-exploration.md` (violates I2's one-file-one-agent rule) — C) two agents, disjoint files, lead merges.

**Evidence:** Charter §6.1 allows 1–2 agents; I2 forbids two agents holding the same file; I14 requires template headings, which survive a mechanical merge.

**Why this one:** Parallel wall-clock win with zero file contention; the merge is content-neutral so I6 isolation and "supervise, don't author" both hold.

**Escalated?** no

---

## D6 — GPU-less `is_available()`: probe the GPU, omit cuOpt from `available_solvers`

*Phase: 2→3 · Decided by: lead · 2026-08-22 · Status: decided — binding on Phase 3*

**Decision:** cuOpt's `is_available()` returns true only when `cuopt` imports AND a usable CUDA device is present; on GPU-less machines `available_solvers` omits cuOpt (plans 1/2/4 position). The exact probe mechanics (device count vs 1×1 solve, caching, fail-open vs fail-closed) are the revision planner's to fix in FINAL.md.

**Options considered:** A) GPU probe in `is_available()` (plans 1, 2, 4) — B) import probe + GPU check in `_license_probe()` (plans 3, 5).

**Evidence:** Charter §4.6 constraint 3, verbatim: "linopy.available_solvers must simply omit cuOpt" on a machine with the package but no GPU — and §4.6 opens with "The human lead has made this binding". Full fork analysis in ESC-1.

**Why this one:** An explicit, binding product requirement outranks the I5 style precedent; cuPDLPx's import probe predates §4.6 and no existing GPU solver was ever held to it.

**Escalated?** ESC-1 (filed as an audit marker; resolved by charter text, not autonomous judgment)

---

## D7 — Requirement-ID registry C-R18…C-R29 (charter §5.3: new IDs recorded here)

*Phase: 3 · Decided by: revision planner (Opus 5), recorded by lead · 2026-08-22 · Status: decided — frozen at Gate 3*

**Decision:** The requirement set is C-R1…C-R17 (charter §5 rows, in order) plus twelve new IDs, never renumbered or reused: **C-R18** actionable GPU-less failure text (residual: generic assert wording accepted); **C-R19** M2 always-minimize transform (duals/objective/solution_bound negation); **C-R20** Barrier default method with user override; **C-R21** status maps name-keyed, two dicts by problem category; **C-R22** MIQP/IQP guard; **C-R23** zero-nnz pad row (singly-bounded; all-free residual → NotImplementedError); **C-R24** option handling (blanket bool→int, unknown-name wrapping, log_fn→log_file); **C-R25** worker-thread KeyboardInterrupt pattern; **C-R26** warm-start deferral + set_initial_primal_solution prohibition; **C-R27** semi-continuous + MIP_DUAL_BOUND_REPORT feature declarations with frozen contingencies; **C-R28** fork-safe out-of-process cached GPU probe; **C-R29** bare `model.solve("cuopt")` works with zero leftover temp files.

**Options considered:** per-ID options are in plans/FINAL.md §6 and logs/50-review.md — this entry is the registry §5.3 requires, not a re-derivation.

**Evidence:** plans/FINAL.md §6 (maps all 29 to work packages); logs/15-verification.md "Proves" column (all 29 present); selector round-3 mechanical confirmation.

**Why this one:** §5.3 — IDs must live in this log so later agents that never read FINAL.md cannot renumber or reuse them.

**Escalated?** no

---

## D8 — Gate-3 design decisions fixed in FINAL.md (registry of record)

*Phase: 3 · Decided by: Phase 3 loop (selector + revision planner, both Opus 5), recorded by lead · 2026-08-22 · Status: decided — frozen at Gate 3*

**Decision:** (1) Entry point `DataModel`+`Solve`, `io_api="direct"` only — no file io_api in this PR; `accepted_io_apis` deliberately not declared (dead code on master). (2) `_SOLVER_PROBE_ORDER`: `"cuopt"` after `"cupdlpx"` (preserves existing users' default solver). (3) GPU probe under D6: out-of-process (subprocess), cached, fail-closed — in-process CUDA calls forbidden (fork poisoning). (4) Blanket bool→int option coercion. (5) `_build_file` override builds directly (warn), `_run_file` untouched, temp problem file handed back via `_problem_fn` for linopy's own cleanup. (6) Zero-nnz pad: one singly-bounded redundant row on the first finite-bounded variable. (7) `test_quadratic_model_unbounded` disposition frozen as honest `NumericalError → internal_solver_error` (V2-2). (8) Extra named `cuopt`: `["cuopt-cu12>=26.8; platform_system == 'Linux'"]`, sibling of `solvers`. (9) `doc/api.rst` gains `solvers.cuOpt` and `model.Model.to_cuopt`. (10) `examples/using-solvers.ipynb`: markdown-only mention, no code cell. (11) Commit style: D2 confirmed, not duplicated.

**Options considered:** five plans + three review rounds; alternatives and rejections recorded in logs/50-review.md rounds 1–3.

**Evidence:** plans/FINAL.md §§1–9, §13; logs/50-review.md (19 blocking findings raised and resolved); logs/10-exploration.md measurements.

**Why this one:** clean APPROVE at round 3 of 5 with the four Gate-3 checks confirmed on the record.

**Escalated?** no

---

## D9 — Phase-4 commit assembly: WP-A+WP-B combined in one slice; commit 3 built by cherry-pick + amend

*Phase: 4 · Decided by: lead · 2026-08-22 · Status: decided — **UNREVIEWED — autonomous***

**Decision:** (1) WP-A and WP-B run as **one** dispatch/worktree (`wp-ab`, branch `wp/wp-ab`) producing the single commit `Implement interface to cuOpt solver and required IO` — FINAL.md §7.3 commit 1 verbatim. (2) WP-C commits `Add package dependency for cuOpt` on `wp/wp-c`, which is **not** merged directly; WP-E later cherry-picks that commit (cherry-pick preserves authorship) and amends the docs changes into it, retitling to `Add to docs & add package dependency` — commit 3 verbatim. (3) All other FINAL.md §7.3 rows unchanged; WP-E additionally makes commit 4 (`Add comment for release notes`) itself.

**Options considered:** A) separate WP-A/WP-B dispatches with per-WP commits — commit 1 then spans two branches, reachable only by a lead-driven squash (mints authorship, I11 forbids) or amending already-merged history (breaks the §6.4 rebase + `--ff-only` recipe). B) combine A+B; assemble commit 3 by cherry-pick+amend inside WP-E's worktree.

**Evidence:** Charter §6.4a: "the work-package boundaries in §6.4 are about *worktree isolation*, not about commit boundaries. Several work packages may land as one commit; the lead decides at merge time." Isolation is unaffected: nothing runs concurrently with `wp-ab` except `wp-c` (disjoint file — FINAL.md §7 already schedules C parallel with A). `git cherry-pick` and `git commit --amend` both preserve the original author unless told otherwise.

**Why this one:** Preserves the human-directed #516 commit shape exactly, with zero lead-minted commits and zero post-merge history rewrites.

**Escalated?** no — §6.4a explicitly delegates commit boundaries to the lead at merge time; marked UNREVIEWED for the §6.8 handoff audit list because it adjusts FINAL.md §7's execution mechanics autonomously.

---

## D10 — WP-G added (human-lead directive): in-code rationale for every unsupported/deferred feature

*Phase: 4 · Decided by: human lead (mid-run instruction), mechanics by lead · 2026-08-22 · Status: decided — human-directed, not autonomous*

**Decision:** FINAL.md §7 gains **WP-G** (Amendment A1): a comments/docstrings-only work package that puts a concise (1–3 line) rationale at the enforcement site of every unsupported or deferred feature decision — direct-only io, warm-start deferral + the never-call `set_initial_primal_solution` fence, no reduced costs, SOS/indicator absence, MIQP guard, Barrier default, empty MIP duals, subprocess GPU probe, bool→int coercion. New requirement ID **C-R30** (extends the D7 registry; never renumbered or reused): *each such decision carries an in-code rationale a future developer can find without the harness logs.* WP-G runs after WP-F (shares `solvers.py`), alone, as commit 6 (`Add comments explaining unsupported features`); the reserved review-fix commit becomes 7. Zero behaviour change — the diff may contain only comments/docstrings.

**Options considered:** A) rely on `logs/45-known-issues.md` + docs only (invisible to a developer reading the code) — B) fold into WP-F's brief (bloats the highest-risk slice and skips io.py/model.py sites) — C) dedicated audit WP after F.

**Evidence:** Human-lead instruction 2026-08-22: "they should also be concisely highlighted in comments or docstrings in the implementation code … Update the final plan to include a new work package." Charter §6.6a requires comment density matching the file — hence "very concise".

**Why this one:** Direct human-lead instruction; a dedicated pass can also *audit* what WP-A/WP-F already wrote instead of duplicating it. `logs/15-verification.md` untouched (frozen — WP-G adds no thresholds; Phase 6 style review checks C-R30 coverage).

**Escalated?** no — human-directed.

---

## D11 — FINAL.md A2: `keep_files=True` plan sentence corrected to measured behaviour (raises)

*Phase: 4 · Decided by: lead · 2026-08-22 · Status: decided — **UNREVIEWED — autonomous***

**Decision:** FINAL.md's claim that "with `keep_files=True` an empty problem file remains" (§2.1.4/B4 note and the §9 gpu-acceleration row) is corrected in place (marked A2) to the measured behaviour: `keep_files=True` **raises** `NotImplementedError` for cuOpt, because `Model.solve` (`model.py:2098-2106`) suppresses the solution file only when `SOLUTION_FILE_NOT_NEEDED` AND `keep_files=False`; with `keep_files=True` it passes a `solution_fn`, which cuOpt refuses (C-R25). Docs document the measured behaviour (WP-E already did); no code change.

**Options considered:** A) leave the plan sentence and make docs match it (would document behaviour that does not exist) — B) "fix" `Model.solve`'s guard (out of scope, I4/I13 — cuPDLPx shares the byte-identical guard, pre-existing) — C) correct the plan text to the measurement.

**Evidence:** WP-E's pasted matrix on `wp/wp-e` (base a2db13f): `keep_files=True` and `io_api='direct', keep_files=True` both → `NotImplementedError: Solution file output is not yet implemented for cuOpt.` KI7, B16.

**Why this one:** I5-spirit — measured behaviour outranks a plan author's untested assumption; the plan is the contract but a contract sentence contradicted by reproducible evidence is corrected with provenance, not obeyed.

**Escalated?** no — marked UNREVIEWED for the §6.8 handoff audit list.

---

## D12 — B17 fix: persistent worker thread, landed by amending commit 1

*Phase: 4 · Decided by: human lead (fix directed mid-run: "Make the fix now"), mechanics by lead · 2026-08-22 · Status: decided — mechanics **UNREVIEWED — autonomous***

**Decision:** (1) Fix approach: `_run_cuopt_with_keyboard_interrupt` reworked to run every cuOpt solve on a single lazily-created persistent daemon worker thread (serialized handover), instead of one fresh thread per solve — the reused-thread path measured 10/10 clean by WP-D's isolation experiment. Interrupt semantics (C-R22) unchanged. (2) Landing mechanics: the fix agent **amends commit 1** (`a2db13f`, the current tip of `feat/cuopt-solver` — nothing merged after it), keeping message/author/trailer via `--no-edit`, so the shipped slice is one coherent, correct commit; amend only if the agent is Opus 5 (commit 1's author), else a separate honestly-authored fix commit. (3) The lead then rebases the held `wp/wp-d` and `wp/wp-e` branches onto the amended tip (rebase preserves their authorship; neither touches `solvers.py`) before merging in order D → E.

**Options considered:** A) separate fix commit on top (leaves a known-interpreter-crash in commit 1's history mid-arc; #516's follow-up-fix commits came after review, not inside the initial landing) — B) amend commit 1. Fix alternatives rejected: reverting to main-thread solves (loses C-R22's interrupt responsiveness); a thread pool (overkill, solves are strictly sequential); method-level workarounds (method=2 is the §8.5-forbidden third option and the leak is thread-lifecycle, not method).

**Evidence:** WP-D B17/KI8: raw cuOpt, no linopy — main thread 10/10, fresh-thread-per-solve 2–5 then abort (OMP kmp_alloc assertion), one reused thread 10/10; LP path unaffected; reproduces on the existing suite alone.

**Why this one:** evidence-directed minimal change to our own helper; amend keeps the human-directed #516 commit shape with zero lead-minted commits (agent-executed amend, §6.4a recipe pattern).

**Escalated?** no — the fix itself is human-directed; the amend-vs-new-commit and persistent-thread mechanics are marked UNREVIEWED for the §6.8 audit list.

---

## D13 — B19 fix: cuOpt disposal made foreign-thread-safe, landed by amend-and-rebase into commit 1

*Phase: 4 · Decided by: lead · 2026-08-23 · Status: decided — **UNREVIEWED — autonomous***

**Decision:** (1) B19 (GC on a dask worker thread finalizes dead solver instances → cuOpt native teardown on a foreign thread → intermittent SIGABRT, 2 of 4 full --run-gpu runs) is fixed in `linopy/solvers.py` by a dispatched Opus agent; design chosen by the agent with evidence between (a) routing disposal through the existing persistent solve worker and (b) never retaining native cuOpt objects past `_solve` — plan-conformance on `solver_model`'s contents checked against FINAL.md §2, deviations reported. (2) Landed via the charter §6.4a amend-and-rebase recipe into commit 1 (67675e4), replaying the three later commits (they touch no `solvers.py`), so the shipped core slice stays one coherent, correct commit — same rationale as D12. (3) Acceptance requires a pre-fix reproducer attempt (≥3 variants before falling back to statistics), B17-check retention, and 3/3 clean full --run-gpu runs with the baseline-identical failure set. (4) Gate-3 round 5 (opened in parallel) decides whether the verification plan gains an implementation-agnostic disposal-stability row.

**Options considered:** separate mid-arc fix commit (leaves a known intermittent interpreter abort in commit 1's history) — rejected for the D12 reasons; ignoring as "upstream's bug" — rejected: linopy's own retention/disposal pattern is the trigger surface, and 2-in-4 suite aborts is not shippable.

**Evidence:** logs/phase4-gpu-stability-run3.txt line 39 (full faulthandler attribution); logs/40-blockers.md B19; upstream NVIDIA/cuopt#1768.

**Escalated?** no — marked UNREVIEWED for the §6.8 audit list.

---

## D14 — B19 fix: route cuOpt native disposal through the persistent solve worker (candidate a, not b)

*Phase: 4 · Decided by: WP-B19 agent (Opus 5, `claude-opus-5[1m]`) · 2026-08-23 · Status: decided*

**Decision:** `cuOpt.close()` takes the `DataModel` out of `solver_model` and hands it to the
existing `linopy-cuopt-solve` worker via a new module-level `_dispose_on_cuopt_worker()`
(non-blocking `SimpleQueue.put`, exception-suppressed so it degrades to a local drop at interpreter
shutdown), then calls `super().close()`. `solver_model` still holds the real `DataModel` while the
solver is alive, and `close()` still nulls it synchronously; only the thread the C++ destructor runs
on changes. Landed by amending commit 1 (§6.4a recipe, D13).

**Options considered:** A) route disposal through the persistent worker — B) never retain native
cuOpt objects in `solver_model`, extracting plain-Python data in `_solve` and dropping the natives
on the worker there.

**Evidence:** B is a plan violation on three counts: `plans/FINAL.md` §2 (:527-528) prescribes
`_make_result(..., solver_model=dm, ...)` "mirroring `cuPDLPx._solve`'s tail"; `linopy/io.py:799-808`
`to_cuopt` returns `solver.solver_model` annotated `-> cuoptDataModel` (C-R6/C-R7); and
`test/test_cuopt.py:228` assigns a real `DataModel` to `instance.solver_model`. cuPDLPx retains its
native `cu_model` the same way (`solvers.py:4515`), so A also matches repo precedent (I5). The
crash frame in `logs/phase4-gpu-stability-run3.txt` is `solvers.py:1241` (`self.solver_model = None`)
under `__del__` on a dask thread, i.e. exactly the reference A moves. A is verified deterministically
(check A in `dev-scripts/b19_teardown_thread.py`: base 5/5 foreign-thread releases → fixed 0/5) plus
3/3 clean `--run-gpu` runs; see `logs/40-blockers.md` "B19 — FIXED".

**Why this one:** it satisfies the brief's four handover constraints (non-blocking from `__del__`,
no deadlock if GC fires on the worker, degrades at shutdown, exceptions already suppressed by
`__del__`) with a 27-line, one-file diff (I4) and **zero** deviation from FINAL.md, where B would
have required deviating from §2 and breaking two requirement IDs.

**Escalated?** no

---

## D15 — Follow-up test-lag slice lands by amending commit 2 (2026-08-23, lead) — UNREVIEWED (§3.7 E-class n/a; process decision under D12/D13 precedent)

**Decision:** the two verification rows the round-5 freeze left ahead of the code — V1-8's N=20
(`test/test_cuopt.py` `test_repeated_solves_of_a_medium_model` currently `range(3)`/`count == 3`)
and the new `test_status_map[milp_time_limit]` cell (frozen B18 market-split recipe) — land as a
single slice that **amends commit 2 (`585f981` "Update tests")** and rebases commits 3–4, using the
same §6.4a amend-and-rebase recipe as D12/D13. Agent must be Opus 5 (commit 2's author), preserve
message + trailer via `--amend --no-edit`, touch only `test/test_cuopt.py`.

**Options considered:** A) amend commit 2 — B) new fifth commit "Strengthen tests".

**Why A:** the planned arc (FINAL.md §7.3) has no test-strengthening commit; these are not new
features but the code catching up to already-frozen rows, i.e. they belong where the tests belong.
B would break the planned commit narrative and force a plan amendment. D12/D13 already established
amend-and-rebase as the sanctioned mechanism for landing fixes inside the arc.

**Escalated?** no — same mechanics twice reviewed implicitly via selector's round-5 closure, but
marked UNREVIEWED like D9/D11/D12/D13 for the Phase-8 handoff list.

---

## D16 — B21 disposition: keep V2-7 red with cause, proceed (2026-08-23, lead) — **UNREVIEWED — autonomous, ESC-2**

**Decision:** (1) V2-7 stays frozen and RED — reported verbatim as unmeetable at cuOpt 26.08.00,
cause = upstream heap corruption (B21 diagnosis, gdb-pinned to glibc large-bin metadata damage from
cuOpt's bundled LLVM libomp; 8/12 full-suite crash rate; linopy exonerated twice). No sixth Gate-3
return: redrafting a bar to pass a known defect is §3.6's forbidden move with paperwork. (2) Gate 4's
GPU criterion is recorded as met-in-correctness / failed-in-stability: every completed full run
(5/5 to date) has a failure set byte-identical to the master GPU baseline; no sampling-until-green.
(3) WP-F and WP-G proceed. (4) KI11 records the user-facing statement. (5) A hygiene slice moves
SolverSettings construction/disposal onto the persistent worker (closes the last cross-thread native
new/delete pairs; measured NOT to change the crash rate — landed for invariant consistency with
B17/B19, labelled as such). (6) Evidence comment posted on NVIDIA/cuopt#1768 (continuation of the
human-authorized upstream engagement).

**Why:** §3.6 (report-verbatim, never widen), §3.7 (file and proceed), and the mission boundary
("decide, not ship" — the human reviews the branch before anything is opened upstream).

**Escalated?** ESC-2 filed in 90-escalations.md.

---

## D17 — Phase-5 findings disposition (2026-08-23, lead) — V3-5 accept-with-rationale is **UNREVIEWED — autonomous, ESC-3**

**Decisions:**
- **F4/V3-5 (blocking):** ACCEPTED with rationale (ESC-3) — the miss stands on the record verbatim; no coverage deleted, no cap edited.
- **F1, F2, F3, F5, F7, F8 → one fix agent, ONE commit** using §7.3's reserved slot 7 (message `Address review findings`): F1 corrected wording (warning + docs — the io_api="direct" file claim is false; say "an empty temporary file is created and removed"); F2 the one-line `os.register_at_fork(after_in_child=_cuopt_solve_queue.cache_clear)` remedy (hang → fast, explanatory CUDA error; reviewer-measured); F3 `test_to_cuopt` in `test_io.py` mirroring `test_to_cupdlpx`; F5 drop "ranged" from the docs constraint list; F7 split the merged cuPDLPx/cuOpt skip reason; F8 concrete `cuopt` types (TYPE_CHECKING import exists) replacing three `Any`s.
- **F6/F6b, F9, F10 → record-only:** F6 four status-map rows live under different test names, thresholds met (B20-family naming divergence, handoff note); F6b V1-7's recipe was substituted by WP-D — threshold met, discriminating answers reproduced vs live HiGHS, but unauthorized substitution recorded honestly; F9 V0-8's literal "0 collected" reads 20 — all 20 provably GPU-free (enumerated), CI-safe; F10 api.rst to_* inconsistency is KI5's pre-existing gap (I13 — not widened here).
- After the fix commit: Phase-5 re-review (loop 2 of 3) scoped to the fix commit + re-run of the directly affected checks; then Phase 6.

**Escalated?** ESC-3 for the V3-5 acceptance; the rest is ordinary finding triage.

---

## D18 — Phase-6 findings disposition (2026-08-23, lead)

**Decisions:** S1 (prerequisites.rst bare-`cuopt`/cu12 sentence), S2 (two requirement-contradiction bullets on the same page), S3 (comment-density migration, ~22 lines moved into docstrings per the reviewer's migrate-don't-delete remedy — CONSTRAINT: every V3-6(d) anchor stays ≥1 in solvers.py+io.py, the A1.1 same-block property holds, and the V3-6 four-count check is re-run and pasted after; this refines, not reverses, the human-directed WP-G comments — facts move, none deleted), N1 (Notes/Attributes swap), N3 (reaped-flag PID-reuse guard) → one fix agent, landing by **amending commit 7** (`b0b18f4`, tip — direct `--amend --no-edit`, no rebase needed; its message "Address review findings" covers both review phases). **N2 DECLINED**: extracting a shared io_api guard onto the `Solver` base to dedup one message literal touches shared behaviour-adjacent code for cosmetic gain — I4 says leave it; recorded here.
Then Phase-6 re-review (loop 2 of 3) scoped to the amended delta.

**Escalated?** no.
