# cuOpt support — {{plan | implementation | style-and-docs}} review, round {{k}}

*Reviewer: {{agent id}} · {{YYYY-MM-DD}} · Reviewing: {{artifact path or commit range}}*

<!-- HOW TO USE (delete this comment block when filling in):
     Append to /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/50-review.md. This is the ONLY file you may write — do not edit the
     plan, the code, the verification plan, or any other log. You read artifacts FROM DISK; you
     do not accept anyone's paraphrase of them. End with an explicit verdict.
     Use absolute paths under dev-scripts/ — it is untracked, so relative paths do not resolve
     from inside a worktree. -->

## Verdict

**APPROVE** | **REVISE**

<!-- REVISE if any blocking finding stands. Do not approve with "minor nits remain". -->

## Blocking findings

| # | Finding | Where (file:line) | Why it blocks | Suggested fix |
|---|---|---|---|---|

## Non-blocking observations

## Checklist

<!-- Tick every row or state why it does not apply. An unticked row with no explanation
     means the review is incomplete, not that the item passed. -->

- [ ] Every requirement ID in scope is covered
- [ ] No requirement silently dropped or reinterpreted
- [ ] Frozen verification plan unchanged — `sha256sum /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md` matches the
      digest recorded at Gate 3 in **both** `logs/00-lead-journal.md` and `logs/50-review.md`
      (`git diff` cannot check this: `dev-scripts/` is gitignored, so the file is untracked)
- [ ] No threshold widened, no check downgraded to a weaker tier
- [ ] Tests actually ran (GPU tests not silently skipped)
- [ ] Diff is surgical — no drive-by reformatting or refactoring
- [ ] Matches repo idiom (current `master`, not PR #516)
