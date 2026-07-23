# Personality

- No sycophancy. Value honest, objective feedback over agreeableness — push back on weak ideas, biases, or poorly-thought-out requests, and explain better alternatives, while staying friendly.

# Git commit workflow

- Never run `git commit` unless explicitly asked that turn. Drafting/staging is fine anytime.
- Every commit needs a `Co-Authored-By: Claude <model-name> <noreply@anthropic.com>` trailer (via HEREDOC), one per contributing model.

# Fix scope

- If a bug report floats a bigger speculative fix alongside a smaller one matching existing patterns, don't default to the bigger one — surface the choice explicitly before planning.
- Fold in same-sized adjacent bugs found in passing (flag in summary); a structurally bigger one needs an explicit scope question first.

# Breaking changes

- lokigi is pre-1.0: prefer fixing an inconsistency properly (rename/remove the bad param or default) over a compatibility shim. No deprecation period (alias + `FutureWarning`) unless asked, or the break is silent and hard to notice — say why if added.
- When removing/renaming public API, make failures legible: if it'd otherwise be absorbed by `**kwargs` and surface as an unrelated error, add an explicit check naming the replacement.
- Breaking = anything that changes results without the caller changing code, not just signature changes. Each needs a `**BREAKING:**` HISTORY.md bullet plus a line in that version's `### ⚠️ Breaking changes` summary (create if absent).

# Testing

- Don't monkeypatch core objects (e.g. `pd.DataFrame`) to cover a trivial/unreachable branch — skip it.
- For subtle-bug fixes (wrong direction, silent no-op, wrong ordering), prove the new test catches the regression by reverting the fix and confirming it fails, then restoring. "Tests pass" alone isn't proof.

# Deferred work

- Structural fixes that get deferred need a handover: context, repro, proposed design/trade-offs, implementation notes, testing guidance.

# Reporting

- State uncertainty plainly — distinguish directly-verified from inferred (e.g. "confirmed via code inspection, but couldn't trigger through the public API").
- Named external citations (papers, APIs, specs) need a primary-source check before landing in code/docs — a recalled name or detail can be subtly wrong even when the surrounding content is right.

# Third-party reuse

- Reusing external code, data, or test fixtures — even a small amount — needs a licence check and attribution in THIRD_PARTY_LICENCES.md as part of the same change, proposed proactively rather than waiting to be asked.

# Plotting

- In any method that draws geometry (`.plot()`/`.explore()`) and also adds text labels via `adjust_text`, draw the geometry first. Labelling before the axes have real data limits lets `adjust_text` reposition labels against the still-default (0,1) range; once the real extent is applied afterward, those stale positions read as data coordinates far outside it. This doesn't error or show up in a normal render — it only surfaces as a huge, near-empty image on a `bbox_inches="tight"` save, which is what Jupyter's inline display uses. `plot_sites()` gets the order right; copy that order, not just its calls.

# Example notebooks

- Lives at `examples/<category>/<name>/index.ipynb` (`location`, `eda`, `travel_time_matrices`, `routing`, `other`). Front matter markdown cell: `title`, `toc: true`, `execute: {enabled: true}`, `image:`.
- `image:` is expected in practice, not truly optional -- it's the listing grid's thumbnail. Generate it from a representative static plot via `savefig`. If the notebook has no static representative plot to use, omit `image:`/`image.png` and say so rather than fabricating one.
- Must be wired into `examples/examples.qmd` (both the `listing` metadata's `contents` list and the matching `:::{#id}:::` div) or it's invisible.
- Prefer extending an existing example's setup (sample data, site/matrix registration) over a new one; cross-link with relative markdown links.
- Committed with real executed outputs. After editing code cells, run `python -m jupyter nbconvert --to notebook --execute --inplace <path>` from the notebook's own directory and check for error outputs.
- Don't fabricate numbers in prose — derive by running the scenario. Re-run and check any notebook whose narrated numbers a core-code change could have shifted.
- Revert re-execution diffs that are pure timestamp/widget-ID/execution-count churn with no value change.
- Hand-editing prose in an already-executed notebook (no re-run needed): edit `source` as a JSON list of lines (each ending `\n` except the last), matching existing style, for a line-granular diff.

# HISTORY.md

- Every user-facing change needs an entry — check as part of the task, don't wait to be asked.
- Add to the top section if it's unreleased; only start a new `## vX.Y.Z` (bumping `pyproject.toml`) for the first change since the last release.
- One top-level bullet per feature/fix, nested (4-space) sub-bullets for specifics.
- Versions with breaking changes open with `### ⚠️ Breaking changes` (scannable one-liners) then `### Notes` for full bullets — see v0.7.0. Otherwise a plain flat list.
- Docs/example-only changes to already-documented features usually don't need a new bullet — refine existing wording instead.
