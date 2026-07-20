# Git commit workflow

- Never run `git commit` unless the user explicitly asks for it that turn. Drafting a message and staging files is fine anytime; committing is not.
- Every commit must include a `Co-Authored-By: Claude <model-name> <noreply@anthropic.com>` trailer (via HEREDOC), even for a short "please commit this." If multiple models contributed this session, include a trailer for each.

# Fix scope

- If a bug report floats a bigger, speculative fix alongside a smaller one consistent with existing patterns, don't default to the bigger one just because it was mentioned last. Surface the choice explicitly before planning — the smaller fix usually wins.
- A same-sized adjacent bug found in passing is fine to fold in and flag in the summary. A structurally bigger one needs an explicit scope question first, even mid-task.

# Testing

- Don't monkeypatch core objects (e.g. `pd.DataFrame`) just to cover an unreachable or trivial branch — skip it instead.
- For fixes to subtle bugs (wrong direction, silent no-op, wrong ordering), prove the new test catches the regression: revert the fix, confirm it fails, then restore. "Tests pass" alone isn't proof.

# Deferred work

- When a fix needs a structural change and gets deferred, write a handover: context, a concrete repro, the proposed design and trade-offs, implementation notes, and testing guidance.

# Reporting

- State uncertainty plainly. Distinguish what's directly verified from what's inferred, e.g. "confirmed via code inspection, but couldn't trigger through the public API."
