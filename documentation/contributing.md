# Contributing: branches, pull requests and the Ready state

Every change starts from a GitHub issue and goes through a pull request linked to it (`Closes #N` in the body).

## Workflow

1. One change = one branch = one pull request. Work in a separate git worktree, never on `main`.
2. Before each push, run what CI runs (see [Code quality](code-quality.md)): `uv sync --locked`, `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, `uv run pytest`.
3. Open the pull request as a **draft**. Its description explains each change and why, and lists the local commands and their results.

## Who marks a pull request "Ready for review"

**Only the author, by hand** (issue #198). Green tests are a precondition, never a trigger:

- The CI workflow `.github/workflows/code-quality.yml` only reports results. It runs with `permissions: contents: read`, so it cannot change the state of a pull request. `tests/test_workflows.py` checks that no workflow gets write access to pull requests and that none calls `gh pr ready`.
- Automated helpers (bots, Claude sessions, scripts) report the local and CI results on the pull request and leave it in draft. They never call `gh pr ready`, the `markPullRequestReadyForReview` mutation, or the equivalent API, even when every check is green.
- The author marks the pull request as ready when they consider the work complete and want a human review. A pull request is only marked ready when the local checks and the CI are green on its last commit.
