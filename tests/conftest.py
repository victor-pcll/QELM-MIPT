"""Shared pytest setup.

Several tests run ``main.py``, the experiment scripts or their bash launchers
in a subprocess. Those children import ``QELM`` through the editable install,
i.e. the ``.pth`` file uv writes into ``.venv``. On macOS that file can carry
the ``hidden`` flag, and Python 3.12+ skips hidden ``.pth`` files, so the
child fails with ``ModuleNotFoundError: No module named 'QELM'`` (issue #80).
Putting ``src/`` on ``PYTHONPATH`` here makes every subprocess find the
package whatever the state of the install.
"""

from __future__ import annotations

import os
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")

_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
if SRC not in _paths:
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in [SRC, *_paths] if p)
