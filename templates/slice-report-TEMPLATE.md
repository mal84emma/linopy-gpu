# Work package {{wp-id}} — return report

<!-- HOW TO USE: this is the coding agent's return contract (charter §8.1). Report these
     items VERBATIM and IN THIS ORDER. Do not summarise away failures — a red test reported
     honestly is acceptable work; a hidden one is not. -->

**0. Worktree lineage.**
```
{{ paste: git log --oneline -1 }}
{{ paste: git merge-base --is-ancestor <BASE_SHA> HEAD && echo "LINEAGE OK" || echo "WRONG LINEAGE" }}
```

**1. Deliverables.** <!-- each file added/changed, one line on what it does -->

**2. Requirement IDs → what was built.** <!-- flag any you could NOT satisfy, and why -->

**3. Tests.** <!-- each test added, which verification-plan row it proves, PASTED output -->

**4. Escalations.** <!-- frozen-threshold miss (verbatim numbers), brief-vs-plan discrepancy,
     invariant you could not respect. "None" if none. -->

**5. Bugs noticed, not fixed.** <!-- pre-existing defects seen in or near your slice:
     location + one line. Never fix in passing. "None" if none. -->

**6. Commits.** <!-- paste: git log --pretty='%h | %an | %s' for your slice.
     Stamp YOUR OWN model in the --author field and in the Co-Authored-By trailer; the two
     must name the same model. "Claude Opus 5" and "Claude Fable 5" are both correct.
     Running on Fable changes nothing about the work — commit normally and say so below.
     The only error is claiming a model you are not: a false mark is worse than no commit. -->

Model I am running on: {{model}}
