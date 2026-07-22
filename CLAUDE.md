# Git commit workflow

- Never run `git commit` unless the user explicitly asks for it that turn. Drafting a message and staging files is fine anytime; committing is not.
- Every commit must include a `Co-Authored-By: Claude <model-name> <noreply@anthropic.com>` trailer (via HEREDOC), even for a short "please commit this." If multiple models contributed this session, include a trailer for each.

# Fix scope

- If a bug report floats a bigger, speculative fix alongside a smaller one consistent with existing patterns, don't default to the bigger one just because it was mentioned last. Surface the choice explicitly before planning — the smaller fix usually wins.
- A same-sized adjacent bug found in passing is fine to fold in and flag in the summary. A structurally bigger one needs an explicit scope question first, even mid-task.

# Breaking changes

- lokigi is pre-1.0, so breaking changes are acceptable in any minor release. Prefer fixing an inconsistency properly over carrying a compatibility shim: rename the bad parameter, remove the wrong default, drop the deprecated alias.
- Don't add a deprecation period (alias + `FutureWarning`) by default. Only do so if asked, or if the change is both silent and hard for a caller to notice — and say why in the summary.
- When removing or renaming public API, make the failure legible. A bare removal is fine when Python raises a clear `TypeError` naming the method, but not when the argument would be absorbed by `**kwargs` and surface as an unrelated error from a dependency — add an explicit check that names the replacement.
- Every breaking change needs a `**BREAKING:**` prefix on its HISTORY.md bullet and a line in that version's `### ⚠️ Breaking changes` summary at the top of the section (create it if absent). Treat as breaking anything that changes results without the caller changing code — a renamed or removed argument, a metric that changes meaning, a different solution being selected — not just signature changes.

# Testing

- Don't monkeypatch core objects (e.g. `pd.DataFrame`) just to cover an unreachable or trivial branch — skip it instead.
- For fixes to subtle bugs (wrong direction, silent no-op, wrong ordering), prove the new test catches the regression: revert the fix, confirm it fails, then restore. "Tests pass" alone isn't proof.

# Deferred work

- When a fix needs a structural change and gets deferred, write a handover: context, a concrete repro, the proposed design and trade-offs, implementation notes, and testing guidance.

# Reporting

- State uncertainty plainly. Distinguish what's directly verified from what's inferred, e.g. "confirmed via code inspection, but couldn't trigger through the public API."

# Example notebooks

- Each example lives at `examples/<category>/<name>/index.ipynb` (categories: `location`, `eda`, `travel_time_matrices`, `routing`, `other`). Front matter is a markdown cell: `title`, `toc: true`, `execute: {enabled: true}`, and optionally `image: image.png` if a card image exists.
- A new example is invisible until it's wired into `examples/examples.qmd`: add its path to the `contents` list of the right `listing` block (or a new one), under the heading section that already matches its topic. Check both the `listing` metadata at the top of the file and the `:::{#id}:::` div in the body — both reference the same `id`.
- Prefer extending an existing example's problem setup (same sample data, same site/travel-matrix registration) over inventing a new one, so the reader isn't re-learning unrelated setup. Link back to the example you borrowed from, and forward to yours from it, with relative markdown links (`../other_example/index.ipynb`).
- Notebooks are committed **with real, executed outputs**, not authored output. After writing or editing code cells, run `python -m jupyter nbconvert --to notebook --execute --inplace <path>` from the notebook's own directory (its relative sample-data paths assume that cwd), and check the result for error outputs before committing.
- Don't fabricate specific numbers in prose ("coverage of 62%", "grows from three solutions to six") — derive them by actually running the scenario first, then write the sentence to match. If a change to core code could have shifted a number an existing notebook already narrates, re-run that notebook and check, rather than assuming the prose still holds.
- If re-executing a notebook produces a diff that's purely execution timestamps / widget IDs / cell-execution-count churn with no actual value change, revert it — that's noise, not signal. Only keep a re-execution diff when it reflects a real output change.
- When hand-editing prose in an already-executed `.ipynb` (no code change, so no need to re-run), edit the cell's `source` as a JSON list of lines (each ending `\n` except the last), matching the file's existing style — not a single string — so the diff stays line-granular and reviewable.

# HISTORY.md

- Every user-facing change (new parameter, new method, behavior change, bugfix) needs an entry — check whether one is needed as part of the task, don't wait to be asked.
- If the top section is an unreleased version (not yet tagged/published), add new bullets to it rather than starting a new `## vX.Y.Z` heading. Only start a new version section, and bump the version in `pyproject.toml` to match, for the first change since the last release.
- Match the existing structure: one top-level bullet per feature/fix as a one-line summary, with nested (4-space-indented) sub-bullets for specifics — defaults, edge cases, caveats, what stays unchanged.
- A version section that contains breaking changes opens with a `### ⚠️ Breaking changes` summary — one scannable line each, saying what changed and what to do about it — followed by `### Notes` for the full bullets. See v0.7.0. Versions with no breaking changes keep the plain flat list.
- A docs-only or example-only change to an already-documented feature usually doesn't need a new bullet — refine the existing one if the wording needs to change (e.g. mentioning a new warning), otherwise leave it.
