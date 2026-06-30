"""Apply the run-local PyAV override in every torchrun worker when requested."""

from __future__ import annotations

import os

if os.environ.get("G05_FORCE_VIDEO_BACKEND") == "pyav":
    from force_pyav import activate

    activate()
