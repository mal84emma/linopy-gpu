# cuOpt support — conformance evidence

*Written by: {{agent id}} · Phase 5 · Verification plan: `logs/15-verification.md` @ sha256 `{{digest}}`*

<!-- HOW TO USE (delete this comment block when filling in):
     Copy to logs/60-conformance.md. Execute EVERY row of the frozen verification plan and
     record each result against its frozen threshold. Paste real output — never a summary,
     never a retyped number. Any miss goes in Escalations VERBATIM and stops there.
     Invoke tools as .venv/bin/<tool>. NEVER a bare `uv run` or `uv sync`: there is no
     uv.lock, so uv would sync the project env and prune cuOpt straight out of it (I9).
     Exception: ruff is not in .venv — it is a uv tool install (charter §6.0), invoked as
     /home/azureuser/.local/bin/ruff. -->

## Frozen-plan integrity

<!-- dev-scripts/ is gitignored, so `git diff` on the verification plan is vacuous — it
     returns empty for an untracked file however much it was edited. Hash it instead. -->

```
{{ paste: sha256sum /home/azureuser/linopy-gpu/dev-scripts/cuopt/logs/15-verification.md }}
```

- Digest recorded in `logs/00-lead-journal.md` at Gate 3: `{{digest}}`
- Digest recorded in `logs/50-review.md` at Gate 3: `{{digest}}`
- **All three match:** yes / **NO — blocking finding**

## Verification plan results

| Check ID | Threshold (frozen) | Actual | Pass? | Evidence (pasted below / path) |
|---|---|---|---|---|

## Requirement → evidence map

| Req ID | Deliverable (file:symbol) | Evidence (check ID + pasted test) |
|---|---|---|

## Test output (pasted verbatim)

```
{{ paste: .venv/bin/pytest -q }}
```

```
{{ paste: .venv/bin/pytest --run-gpu -q  — confirm GPU tests RAN, not skipped }}
```

```
{{ paste: /home/azureuser/.local/bin/ruff check . && .venv/bin/mypy }}
```

## Escalations

<!-- Any check that missed its frozen threshold, VERBATIM with the actual numbers. Do not
     widen a tolerance, do not re-run until green, do not downgrade the check, do not argue
     the discrepancy away. Report and stop on that item. "None" if none. -->

None
