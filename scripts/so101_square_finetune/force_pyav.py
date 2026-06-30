"""Run-local video-backend override for an installed-but-unusable TorchCodec.

The repository selects TorchCodec solely because its Python package is present.
On this server its shared CUDA dependency is unavailable, while PyAV is working.
Keeping this patch in the run directory avoids changing the repository default
or any other project that may legitimately use TorchCodec.
"""

from __future__ import annotations


def activate() -> None:
    from g05.data.lerobot.datasets import video_utils

    video_utils.get_safe_default_codec = lambda: "pyav"
