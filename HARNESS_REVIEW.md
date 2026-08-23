# Harness consistency review — cuOpt autonomous-run infrastructure

*Reviewed: 2026-08-22 · Scope: `dev-scripts/cuopt/PROJECT_LEAD.md` + `templates/*` · Reviewer: Claude (Fable 5), independent pass before run start*

---

## FINAL REVIEW, PASS 4 — 2026-08-22 17:0x (after the authorship-policy redesign)

**Scope of the delta:** charter 1218 → 1273 lines; `slice-report-TEMPLATE.md` updated. The
planner replaced the Opus-only authorship rule with a **truthfulness rule**: a new MODEL
ASSIGNMENT block at the top (lead = Fable 5, every dispatch `model: "opus"`, never
`subagent_type: "fork"`); I11 rewritten ("authored by the model that actually wrote it" —
both `Claude Opus 5` and `Claude Fable 5` valid, mis-stamps relabelled, work never
discarded); §6.4a rewritten (own-model stamping, amend-and-rebase relabel procedure,
author-must-equal-trailer check with a `tformat:` fix for the dropped-last-commit bug);
§8 gains the dispatch-parameter rule; slice-report item 6 + "Model I am running on" line;
risk-register row rewritten to match.

**Verified:**
- The fork claim is factually correct: `subagent_type: "fork"` inherits the parent model
  and ignores a `model` override (per the Agent tool contract) — the one route to a
  mislabelled commit, and the charter bans it at dispatch.
- **All three §6.4a checks fire-tested on a scratch repo** (per the charter's own "make it
  fail on purpose once" rule): the author grep catches a non-Claude author; the mismatch
  loop catches an author/trailer disagreement; a correctly-trailered *Fable* commit passes;
  and the LAST commit is genuinely checked — the `tformat:` comment is right that `format:`
  would silently drop it. The multiword author name survives `read h a` via
  remainder-of-line splitting, and the trailing space in `"Co-Authored-By: $a "` matches
  the ` <email>` that follows.
- The relabel procedure (`checkout <bad>` → `--amend --author` → `rebase --onto HEAD <bad>
  feat/cuopt-solver`) is git-correct and leaves the branch rebased; the instruction to
  re-run the §6.4a checks afterwards is necessary and present.
- §8, §8.1, DoD ("all authored by Claude" — model-neutral), Phase 5 fix-agent wording, and
  every remaining "dispatch an Opus agent" instruction are consistent with the new policy.
- Pass-3 fixes (F4 `--author` in §6.7, M5, typo) all survived the planner's edits.

### F5 — handoff-TEMPLATE.md still asserted the OLD policy (FIXED)

`handoff-TEMPLATE.md` §8 read "All authored by Claude Opus 5: {{yes/no}}" — under new I11 a
legitimate Fable-authored commit would force the lead to answer "no" on a clean run, or
worse, treat the commit as a defect. Now: "Every author a real Claude model, author
matching its Co-Authored-By trailer (I11)" + a "Fable-authored commits: {{list, or none}}"
line, which also feeds the audit surface the handoff exists for.

### F6 — §6.7 still framed the harness commit as an exception to "Opus pinning" (FIXED)

"I11's Opus pinning does not apply to it" described the *old* I11. Under the new invariant
the harness-record commit is not an exception at all — it follows the same rule (stamp the
model that wrote it). Reworded: I11 applies the same way everywhere; the lead stamps
`Claude Fable 5` per the model-assignment note.

### M6 — relabel snippet contradicted itself on the trailer (FIXED)

The comment said "fix the Co-Authored-By trailer in the same amend" while the command used
`--no-edit`, which cannot change the message. The snippet now says explicitly: a wrong
trailer means rewriting the message in the amend (drop `--no-edit`, pass the corrected
message).

### M7 — garbled rationale in the MODEL ASSIGNMENT block (FIXED)

"cheap wide supervision, expensive narrow execution" had both adjectives backwards
however you read it (supervision is one context, execution is the fan-out; and the lead
runs the higher-tier model). Replaced with the actual rationale: "one supervising context
that never writes code, many isolated Opus contexts that do."

**Verdict: consistent.** The authorship redesign is a genuine improvement — it converts the
old policy's worst failure mode (a fork silently mislabelling Fable work as Opus) from an
invariant violation into a relabelling chore, and its enforcement checks are now proven to
fire. With F5/F6/M6/M7 fixed, charter and all seven templates agree with each other again.

---

## FINAL REVIEW, PASS 3 — 2026-08-22 16:4x (after planner's Phase-7 additions)

**Scope of the delta reviewed:** charter grew 1129 → 1213 lines. New: §6 section-numbering
note; DoD item "harness record published as orphan branch (§6.7)"; §6.7 Phase 7 (publish
harness record) + Gate 7; handoff renumbered/kept as §6.8 (Phase 8); I11 gained the §6.7
lead-commit exception; §6.4 worktree-baseRef note; three new risk rows (harness check that
cannot fire; harness record leaks into PR; false authorship provenance);
handoff-TEMPLATE.md header now carries the harness-record URL line.

**Verified live:**
- All §6.x / Gate-N / §7.1 / §8.1 cross-references in the 1213-line charter resolve to the
  right sections after the renumbering — no stale pointer found (grep over every reference).
- `origin` = `mal84emma/linopy-gpu` (https), `upstream` = `PyPSA/linopy` — matches §6.7.
- `gh` default repo resolves to **PyPSA/linopy** — confirms §6.7's "always name the remote
  explicitly" warning is factual, not hypothetical.
- gh authenticated as mal84emma over https with `credential.helper=store` → the §6.7
  https push from a freshly-`git init`ed directory will authenticate.
- Global git identity exists (`mal84emma`) → the fresh-repo commit in §6.7 will not fail
  on missing identity.
- `/home/azureuser/linopy-harness` does not exist → `rm -rf "$D"` destroys nothing.
- `.claude/settings.local.json` IS gitignored (line 56) but `.claude/settings.json` is NOT
  → the §6.4 baseRef claim is accurate as written. (A baseRef pin in settings.local.json
  would technically be leak-safe, but explicit worktree creation remains the more robust
  choice — no charter change needed.)
- `git merge-base A B` with disjoint histories prints nothing / exits 1 — the Gate 7
  isolation proof works as described; the `find`-based allowlist check in the §6.7 script
  can genuinely fire.
- Gate 7's URL-in-journal + §6.7's URL-in-handoff + handoff-TEMPLATE header line are
  consistent (all three, not contradictory).

### F4 — the §6.7 commit command contradicted its own authorship rule (FIXED)

The §6.7 script had bare `git commit -m ...`, so a lead following it verbatim would author
the harness-record commit as the **human** (global git identity `mal84emma`) — while the
bullet four lines later demands the lead's *real model name*, and AGENTS.md rule 1 demands
AI-generated content be marked. An unmarked, human-attributed commit of 100% AI-generated
content is exactly the provenance failure I11 exists to prevent.

**Fix applied:** the command now reads
`git commit --author="Claude <your-real-model-name> <noreply@anthropic.com>" -m ...`
with a comment explaining why and warning not to write "Opus" unless actually on Opus.

### M5 — stale "last gate before handoff" (FIXED)

§6.6's opening line predated Phase 7: Gate 6 is no longer the last gate before handoff —
Gate 7 sits between. Reworded to "The last gate on the *code* — after it only the harness
record (§6.7) and the handoff (§6.8) remain."

### Cosmetic (FIXED)

Risk-register typo "a untrue claim" → "an untrue claim".

### Observations, no change required

- `spikes/` is deliberately excluded from the §6.7 harness-record allowlist. Consistent
  with the section's own intro (charter/plans/reviews/logs), and Phase-1 evidence survives
  as pasted output in `logs/10-exploration.md`. If colleagues should see the throwaway
  prototypes too, add `"$S"/spikes` to the `cp` allowlist — a judgment call, not a defect.
- The harness-record commit message uses a `docs:` prefix; §6.4a's no-prefix rule binds
  only `feat/cuopt-solver` ("overrides I5 for commit messages only" on the PR branch), so
  no conflict.
- The §6.6b leak-check pathspec includes root-level `PR_BODY.md` while §6.8 writes it under
  gitignored `dev-scripts/cuopt/` — the pathspec is belt-and-braces, not a contradiction.

**Verdict: consistent.** With F4 and M5 fixed, the Phase-7 additions integrate cleanly:
no stale cross-reference, every new command verified runnable against the real remotes,
auth and filesystem state of this machine.

---

## FINAL REVIEW, PASS 2 — 2026-08-22 16:1x (exhaustive pre-implementation)

**F1 and F2 are fixed and verified** (charter @ 16:13): §7 row now "answers 1–13"; Gate 1
now "all 13 numbered items above"; the `doc/prerequisites.rst` row now mandates the
one-command extra with `pypi.nvidia.com` as optional fallback and "never with
`--index-strategy`".

**Every factual claim re-verified live against the repo and machine** (not from memory):
all ten `solvers.py` anchors (148/176/189/304/346/359/1544/4156/4426/4440); `io.py:33,792`;
`model.py:73,2517`; `constants.py:171`; `conftest.py:57-79` generic GPU gating;
`test_optimization.py:80-81,562,685,1110`; `pyproject.toml` 92/105/112/126; the §5.1
override set matches `cuPDLPx` exactly (all 10 members, nothing else);
`QUADRATIC_SOLVERS`/`NO_SOLUTION_FILE_SOLVERS` are feature-derived comprehensions;
`_LazyRegistry` shim as described; `contributing.rst:46` exact; `conf.py:104-105`; 18
notebooks = 18 nblinks, only `solve-on-remote.ipynb` has stored outputs, gurobi notebooks
are `infeasible-model.ipynb`/`solve-on-oetc.ipynb`; CI matrix is
ubuntu/macos/windows-latest; all five §6.6b doc files exist; `using-solvers.ipynb`
mentions cuPDLPx. Environment: Python 3.11.15, highspy OK, cuopt not yet installed,
pandas 3.0.5, 25 GB free, T4 / driver 535.274.02 / CC 7.5, `dev-scripts/` gitignored
(line 8), `.claude/` NOT gitignored, gh authenticated (mal84emma), git identity set,
uv 0.11.23, master clean, no `feat/*`/`wp/*` branches yet.

### F3 — BLOCKING: `ruff` is not installed anywhere on this machine

`.venv/bin/ruff` does not exist, `python -m ruff` fails, and there is no system `ruff` on
PATH. The repo does not put ruff in any extra — it runs it via **pre-commit**
(`.pre-commit-config.yaml:11`, `ruff-pre-commit rev v0.15.20`); the `dev` extra carries
only mypy/pytest/pre-commit. Yet the charter assumes `.venv/bin/ruff` throughout: I9's
tool list, every Phase 4 brief (`ruff check --fix .`), the V0 tier, Gate 4, §6.5, and
`conformance-TEMPLATE.md` (`.venv/bin/ruff check .`). Every gate that runs ruff will fail
— and the natural agent workaround (`uv run ruff`, or `uv sync --extra dev`) is exactly
the I9 trap that prunes cuOpt from the venv.

**Fix (one line in §6.0 Phase 0 setup):** `uv pip install ruff==0.15.20` — pin to the
pre-commit rev so lint results match the repo's hooks, installed with `uv pip install`
(no sync, I9-safe). Then `.venv/bin/ruff` exists as the charter assumes. Add "confirm
`.venv/bin/ruff --version`" to Gate 0. (Alternative: `uvx ruff@0.15.20`, but that changes
every invocation site; installing into `.venv` changes one.)

**Verdict: one blocking one-liner (F3) stands between the plan and implementation.**
Everything else — invariants, gates, templates, cross-references, commands, and every
factual claim — verifies clean.

---

## FINAL REVIEW — 2026-08-22 (pre-implementation)

Full re-read of the charter (now 1122 lines) and all 7 templates.

**R1 and R2 are fixed and verified.** §6.4 now says the lead never makes the pre-dispatch
commit and dispatches an Opus agent instead (consistent with I11); I14 now defers to §7.1
as the single source of truth, and §7.1's freehand list matches (six files).

**Two new defects, both from the Q13 addition:**

- **F1 — §7 table stale on answer count.** `PROJECT_LEAD.md:1002` says `10-exploration.md`
  holds "Question 0 plus answers 1–12", but Phase 1 now has 13 items (Q13 installability),
  Gate 1 (line 590) says "all 13", and `exploration-TEMPLATE.md` has a `## 13.` section.
  Change to "answers 1–13". (Nit in the same area: Gate 1 says "all 13 **below**" but the
  items sit above the gate line — say "the 13 numbered items".)
- **F2 — §6.6b contradicts §4.2/§4.6 on install docs.** The `doc/prerequisites.rst` row
  (`PROJECT_LEAD.md:930`) requires "Install instructions incl. the NVIDIA index and
  `--index-strategy unsafe-best-match`" — but §4.2 established the whole tree is on plain
  PyPI, that flag "was never a property of cuOpt", and §6.6b's own "Install instructions
  everywhere" row makes the one-command extra the headline. As written, a Phase 6 agent
  would be *required* to document an index and flag users must not need. Reword to:
  "One-command extra install; Linux-only + driver requirement stated; `pypi.nvidia.com`
  mentioned only as an optional fallback, if at all."

Everything else checked clean: I1–I14 internally consistent; §6.3 role split vs I6;
§6.3a sha256 freeze flow end-to-end (charter ↔ verification-, conformance-, review-TEMPLATE
all agree, "all three match" logic correct); §6.4 lineage + explicit-worktree rationale;
§6.4a authorship pinning incl. Gate 6 re-run and the slice-report "do NOT commit if not
Opus" instruction; I9 (`.venv/bin/*` everywhere, no bare `uv run`/`uv sync` outside
warnings); absolute paths throughout §8 and all templates; §7.1 template table matches the
7 files on disk; all §6.x cross-references resolve to real headings; risk register rows
consistent with the mechanisms they cite.

**Verdict: ready for implementation once F1 and F2 (two one-line edits) are applied.**

---

## RECHECK — 2026-08-22 (after planner revision)

All 12 issues were addressed; new factual claims introduced by the revision were
re-verified against the repo (docs extra exists; `contributing.rst` `uv sync` line
confirmed; gurobi notebooks are `infeasible-model.ipynb` + `solve-on-oetc.ipynb`; all
18 notebooks nblink-linked; exactly 17 carry no stored outputs). Status:

| Issue | Status |
|---|---|
| B1 drift check | ✅ sha256 dual-record + `chmod 444`; templates updated; "make every check fail once" risk row added |
| B2 stale C-R7 | ✅ rewritten as zero-diff verification of the `_LazyRegistry` shim |
| B3 `uv run` sync | ✅ I9 rewritten; conformance template uses `.venv/bin/*` |
| S1 Opus authorship | ⚠️ fixed, but see R1 below |
| S2 relative paths | ✅ absolute paths throughout charter + templates |
| S3 "11 answers" | ✅ §7 now "Question 0 plus answers 1–12" |
| S4 I14 overreach | ⚠️ scoped to §7.1; exploration + handoff templates added; see R2 |
| S5 baseline gap | ✅ §6.0 baselines incl. separate cuOpt-free docs venv; DoD rescoped to no-regression-vs-baseline |
| M1–M4 | ✅ all fixed (roles table, verbatim isolation line, §6.N numbering, table order) |

### Remaining (new/residual)

- **R1 — §6.4 contradicts I11.** "A phase-lead must commit before dispatching" (§6.4,
  ~line 674) conflicts with I11's "You (the lead) never author a commit." Reword to:
  "all work must be committed before dispatching — dispatch an Opus fix agent if a
  commit is needed."
- **R2 — I14's freehand list is incomplete.** I14 names `20-progress, 40-blockers,
  90-escalations` as freehand; §7.1 correctly lists six (adding `00-lead-journal`,
  `05-baseline`, `45-known-issues`). Make the enumerations match (or have I14 defer to
  §7.1).

**Verdict:** The harness is well-constructed and nearly all of its factual claims check
out against the repo — every §5 line anchor, the PR #516 stats (+771/−62, 15 files,
merged 2026-01-19), and the environment facts (Python 3.11.15, highspy present,
pandas 3.0.5, 25 GB free, `.claude/` not gitignored, gh authenticated, conftest GPU
gating generic on `SolverFeature.GPU_ONLY`) were verified. But there are **3 defects
that would break the harness's own enforcement mechanisms mid-run**, plus several
smaller inconsistencies.

---

## Blocking — the mechanism doesn't work as written

### B1. The frozen-verification-plan drift check is vacuous

`dev-scripts/` is gitignored (`.gitignore:8`, per I3), so `logs/15-verification.md` is
never tracked by git. Consequences:

- `git diff <approval-sha> -- dev-scripts/cuopt/logs/15-verification.md` (§6.3a, and
  the review-TEMPLATE checklist) returns empty **for an untracked file no matter what
  was edited** — the anti-tampering check central to I12/§3.6 can never fire.
- There is no "approving commit SHA" that contains the file at all.

**Fix:** freeze by content hash — at Gate 3 record
`sha256sum dev-scripts/cuopt/logs/15-verification.md` in `00-lead-journal.md` and
`50-review.md`; the Phase 5 reviewer recomputes and compares. Update §6.3a, the risk
register row, and the review-TEMPLATE checklist line accordingly.
(Also: the review-TEMPLATE uses the relative path `logs/15-verification.md`, which is
wrong from repo root.)

### B2. §5's `solver_capabilities.py` row (C-R7) is stale

The file is now a back-compat shim: `SOLVER_REGISTRY` is a `_LazyRegistry` that derives
`SolverInfo` on the fly from `SolverName` + the solver class's declared features
(`linopy/solver_capabilities.py:78-100`). **There is nothing to register.** A planner
following the map literally will either edit the shim (a drive-by change, violating I4)
or stall. This is the exact "stale pattern" failure mode the charter warns against
(I5/§5.2), baked into the charter itself.

**Fix:** reword C-R7 as a verification, not an edit: "`SOLVER_REGISTRY['cuopt']`
resolves with the declared features — no code change needed."

Side note (verified consistent): the shim resolves classes via
`getattr(solvers, SolverName(name).name)`, so the enum member name must equal the class
name — §5's `CuOpt = "cuopt"` + `class CuOpt` satisfies this.

### B3. `uv run pytest` in conformance-TEMPLATE violates I9

There is no `uv.lock` in the repo; `uv run` locks and syncs the project environment by
default, which **prunes cuOpt out of the venv** — precisely what I9 exists to prevent —
and the template instructs the Phase 5 agent to paste `uv run pytest -q` /
`uv run pytest --run-gpu -q` output.

**Fix:** change to `uv run --no-sync pytest ...` or `.venv/bin/pytest ...` in
`conformance-TEMPLATE.md` (and anywhere else `uv run` appears in briefs).

---

## Should fix

### S1. "Claude Opus 5" authorship is hard-coded but not guaranteed

Phase 4 dispatches Opus agents, fine — but Phase 5/6 fix agents and any lead-made
commit are not pinned to Opus (the lead runs on Fable 5), while I11, §8.1 item 6, the
slice-report template, and the Gate 4 check all demand every author be literally
`Claude Opus 5`. Misattributing the model defeats the provenance purpose (AGENTS.md
rule 1).

**Fix:** either pin *all* committing agents (including fix agents) to Opus explicitly,
or check the pattern `Claude .* <noreply@anthropic.com>` and use the actual authoring
model's name in `--author` and the `Co-Authored-By:` trailer.

### S2. Charter and templates are unreadable from inside worktrees

Worktree checkouts contain only tracked files; `dev-scripts/` is untracked, so a coding
agent working in `dev-scripts/cuopt/worktrees/<wp>` cannot read
`dev-scripts/cuopt/PROJECT_LEAD.md` or `templates/` at the relative paths the standing
brief (§8) and I14 use. The logs instruction already uses an absolute path.

**Fix:** use absolute paths (`/home/azureuser/linopy-gpu/dev-scripts/cuopt/...`) for
the charter and templates in §8 and I14.

### S3. Phase-1 answer-count mismatch

§7 says `10-exploration.md` holds "The 11 answers"; Phase 1 actually has Question 0 +
items 1–12, and Gate 1 says "Question 0 … and all 12 below."

**Fix:** correct §7's row (agents told structure is load-bearing will trip on this).

### S4. I14 overreaches

I14 says "every artifact is written from its template," but only 5 of the 11 §7 log
files have templates (`10-exploration`, `20-progress`, `40-blockers`,
`45-known-issues`, `00-lead-journal`, `90-escalations`, `99-handoff` are freehand; the
escalation format is inline in §3.4).

**Fix:** either scope I14 to the §7.1 table, or add the missing templates —
`10-exploration.md` (consumed by five planners) and `99-handoff.md` (consumed by the
human, with a mandated two-section opening) are the two most worth templating.

### S5. Gate 0 baseline is narrower than the DoD

The DoD requires all 18 `examples/*.ipynb` to execute (2 use gurobi —
license-dependent) and the docs to build (`nbsphinx_execute = "auto"`,
`nbsphinx_allow_errors = False`), but Gate 0 only baselines `pytest`. If any notebook
or docs warning is already broken on untouched `master`, Phases 5/6 will block with no
recorded proof the breakage is pre-existing.

**Fix:** add docs-build and notebook-execution baselines to Gate 0, and scope those DoD
items to "no regression vs the Gate 0 baseline."

---

## Minor

- **M1 — Phase 3 role ambiguity:** unclear whether the selector writes `FINAL.md`
  (then per I6 it cannot also review it) or the "separate planning agent" does. Name
  the roles explicitly: selector-reviewer vs revision planner, with `FINAL.md` written
  by the planner.
- **M2 — I6 planner isolation is instruction-only:** five parallel planners write into
  the same `plans/` dir and could read siblings. Add "do not read `plans/`" to the
  Phase 2 brief.
- **M3 — Dangling section references:** "§6.7" (DoD, §3.7, risk register) and "§6.3"
  (Gate 3 risk-register row) do not exist as headings — subsections are 6.3a/6.4a and
  phases are named. Cosmetic, but the document tells agents headings are parsed.
- **M4 — §7 table ordering:** `15-verification.md` is listed after `40-blockers.md` —
  ordering nit.

---

## Verified consistent (no action needed)

- All §5 line anchors: `solvers.py` 148/176/304/346/1544/4156/4426/4440; `io.py`
  33/792; `model.py` 73/2517; `constants.py` 171; `conftest.py` 57–79;
  `test_optimization.py` 562/685/1110; `pyproject.toml:92` (`pandas==3.0.5`).
- PR #516: "Add support for solver cuPDLPx", merged 2026-01-19, +771/−62, 15 files.
- Environment: `.venv` Python 3.11.15; highspy importable; cuopt not yet installed
  (Phase 0 installs it); pandas 3.0.5; 25 GB free disk; `gh` authenticated
  (mal84emma); git identity configured; `.claude/` not gitignored (so the §6.4
  explicit-worktree rationale holds); `dev-scripts/` gitignored (I3 holds).
- `QUADRATIC_SOLVERS` and `NO_SOLUTION_FILE_SOLVERS` are derived from features —
  §4.5's "no manual registry edit" is correct.
- `GPU_SOL_TOL`/`CPU_SOL_TOL` exist in `test/test_optimization.py:80-81`;
  `_run_highs_with_keyboard_interrupt` exists at `solvers.py:189`.
- conftest GPU gating is generic on `SolverFeature.GPU_ONLY` via `solver_supports` —
  "likely no change needed" is right.
- `testpaths = ["test"]` in `pyproject.toml` neutralizes any risk of pytest collecting
  duplicate tests from worktrees under `dev-scripts/`.
- Collision/serialization rules, worktree lineage checks, §3.7 escalation split
  (file-and-proceed vs hard-stop), requirement-ID threading, and commit-shape
  discipline (rebase preserves authorship; `--ff-only` adds no merge commits) are all
  internally consistent.
