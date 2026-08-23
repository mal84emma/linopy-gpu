# cuOpt support — handoff

*Run completed: {{YYYY-MM-DD}} · Branch: `feat/cuopt-solver` @ {{sha}} · Lead: {{agent id}}*
*Harness record (§6.7): {{https://github.com/mal84emma/linopy-gpu/tree/harness/cuopt}} — the
charter, plans, review rounds and logs behind this run. Not part of the PR.*

<!-- HOW TO USE (delete this comment block when filling in):
     This is logs/99-handoff.md. THE HUMAN READS THIS FIRST AND MAY READ LITTLE ELSE.
     One-shot mode removed every human gate, so this file is the entire audit surface for a
     run nobody supervised. The first two sections are mandatory and must come first —
     they are the price of the autonomy (charter §3.7).
     Write it for a reviewer who is looking for what you got WRONG. Make that easy. -->

## 1. Unreviewed autonomous decisions

<!-- Every `UNREVIEWED — autonomous, ESC-n` entry from logs/30-decisions.md. One row each.
     These are the calls a human would have made at Gate 3 and did not. "None" only if
     logs/30-decisions.md genuinely contains no UNREVIEWED entry. -->

| ID | Decision taken | Alternative rejected | Where it shows in the diff | Cost to reverse |
|---|---|---|---|---|

## 2. Where this run is weakest

<!-- Ranked, most-suspect first. Your honest read of the least-confident parts — the things
     you would look at first if this were someone else's branch.
     A short, cheerful, or empty version of this section is a FAILED handoff, not a clean run.
     Nothing here is held against the run; a hidden soft spot is. -->

| # | Area | Why I am not confident | What would settle it |
|---|---|---|---|

## 3. What shipped

| Requirement ID | Delivered | Where |
|---|---|---|

## 4. What did not ship

<!-- Every requirement ID not fully satisfied, and why. QP has its own row here if it did not
     land (charter §4.5) — it may never be silently absent. "None" if everything shipped. -->

## 5. Verification summary

- Frozen verification plan: `logs/15-verification.md`, sha256 `{{digest}}`
- Checks executed / passed: {{k}} / {{m}} — detail in `logs/60-conformance.md`
- Any check that missed its threshold, with the actual numbers: …
- Baseline comparison (§6.0): tests …, docs build …

## 6. Escalations

| ID | Question | Trigger | How it was resolved | Still open? |
|---|---|---|---|---|

## 7. Known issues not fixed

<!-- Everything in logs/45-known-issues.md — pre-existing linopy defects seen and deliberately
     left alone (I13), plus anything already red at the Gate 0 baseline. "None" if none. -->

## 8. State of the tree

- Branch pushed: {{yes/no}} · Commits: {{k}} · Every author a real Claude model, author
  matching its Co-Authored-By trailer (I11): {{yes/no}} · Fable-authored commits: {{list, or none}}
- Merge commits present: must be **none** — {{confirm}}
- `git log --stat` shows zero files under `dev-scripts/`: {{confirm}}
- `PR_BODY.md` drafted with the `<!-- HUMAN: write your intent here -->` placeholder: {{confirm}}
- **PR not opened** (charter §2.1): {{confirm}}

## 9. What the human should do next

<!-- Concrete and ordered. Start with reviewing section 1. -->
