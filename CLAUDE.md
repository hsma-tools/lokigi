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

# HISTORY.md

- Every user-facing change (new parameter, new method, behavior change, bugfix) needs an entry — check whether one is needed as part of the task, don't wait to be asked.
- If the top section is an unreleased version (not yet tagged/published), add new bullets to it rather than starting a new `## vX.Y.Z` heading. Only start a new version section, and bump the version in `pyproject.toml` to match, for the first change since the last release.
- Match the existing structure: one top-level bullet per feature/fix as a one-line summary, with nested (4-space-indented) sub-bullets for specifics — defaults, edge cases, caveats, what stays unchanged.
- A version section that contains breaking changes opens with a `### ⚠️ Breaking changes` summary — one scannable line each, saying what changed and what to do about it — followed by `### Notes` for the full bullets. See v0.7.0. Versions with no breaking changes keep the plain flat list.
- A docs-only or example-only change to an already-documented feature usually doesn't need a new bullet — refine the existing one if the wording needs to change (e.g. mentioning a new warning), otherwise leave it.
