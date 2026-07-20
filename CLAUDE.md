# Git commit workflow

- Never run `git commit` unless the user explicitly asks for it in that turn. Preparing a diff, drafting a commit message, and staging files is fine at any time — running the actual `git commit` command is not, until the user says to commit.
- Every commit made by Claude must include a `Co-Authored-By: Claude <model-name> <noreply@anthropic.com>` trailer in the commit message (e.g. `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`), using a HEREDOC so the message formats correctly. Do not omit this trailer even when the user's request is a short "please commit this."
- If a change was worked on across a model switch within the same session, include a `Co-Authored-By` trailer for each model that contributed, not just the one active when the commit happens.

# Fix scope

- When a bug report floats a bigger, speculative fix (a new concept/feature/column) alongside a smaller fix that's consistent with existing patterns in the codebase, don't default to the bigger one just because it was mentioned last or most elaborately. Surface the scope as an explicit choice (e.g. via a clarifying question) before planning. Expect the smaller, consistent-with-existing-patterns fix to usually be preferred.
- A same-sized adjacent bug found while fixing another one is fine to fold in and flag in the summary. An architecturally bigger one (needs a structural change, not a local fix) should be surfaced as an explicit scope question before proceeding, even mid-task.

# Testing

- Don't monkeypatch core/foundational objects (e.g. `pd.DataFrame`) just to force coverage of an effectively-unreachable or trivial code path. If reaching a branch requires that kind of risk, it's usually not worth testing -- skip it, even if the patch would clean up via fixture teardown.
- For a fix addressing a subtle bug (wrong direction, silent no-op, wrong pruning order — not a trivial typo), prove the new tests actually catch the regression: temporarily revert the production change, confirm they fail, then restore it. "Tests pass" alone isn't proof a fix is correct.

# Deferred work

- When a fix needs a structural change rather than a local one and gets deliberately deferred, write a structured handover (history/context, the bug with a concrete reproduction, the proposed fix design and trade-offs, implementation notes, testing guidance) rather than cramming it into the current session.

# Reporting

- State uncertainty plainly rather than asserting with more confidence than the evidence supports. Distinguish what you've directly verified from what you're inferring — e.g. "confirmed via code inspection, but couldn't trigger through the public API" rather than presenting an inference as settled fact.
