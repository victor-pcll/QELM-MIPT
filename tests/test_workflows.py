"""GitHub workflows report results and never change a pull request (#198).

A pull request stays in draft until its author marks it ready by hand, so no
workflow may get write access to pull requests or call ``gh pr ready``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = sorted(
    (Path(__file__).resolve().parents[1] / ".github" / "workflows").glob("*.y*ml")
)
# What would let a workflow move a draft to ready.
READY_CALLS = ("gh pr ready", "markPullRequestReadyForReview")


def _permissions(workflow: dict[str, Any]) -> list[Any]:
    """The top-level permissions and those of every job."""
    jobs = workflow.get("jobs") or {}
    return [workflow.get("permissions")] + [
        job.get("permissions") for job in jobs.values()
    ]


def test_there_are_workflows() -> None:
    assert WORKFLOWS, "no workflow found under .github/workflows"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_cannot_write_pull_requests(path: Path) -> None:
    """Read-only permissions: ``write-all`` or ``pull-requests: write`` would
    let the workflow mark a draft as ready."""
    workflow = yaml.safe_load(path.read_text())
    assert workflow.get("permissions") is not None, (
        f"{path.name}: set top-level permissions explicitly (read-only)"
    )
    for permissions in _permissions(workflow):
        if permissions is None:
            continue
        assert permissions not in ("write-all", "write"), path.name
        if isinstance(permissions, dict):
            assert permissions.get("pull-requests") != "write", path.name
            assert all(value != "write" for value in permissions.values()), (
                f"{path.name}: {permissions}"
            )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_never_marks_a_pull_request_ready(path: Path) -> None:
    text = path.read_text()
    for call in READY_CALLS:
        assert call not in text, f"{path.name} calls {call}"
