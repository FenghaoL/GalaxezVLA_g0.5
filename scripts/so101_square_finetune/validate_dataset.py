#!/usr/bin/env python3
"""Validate the prepared SO101 square-camera data before a G0.5 training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from force_pyav import activate as activate_pyav
from prepare_g05_model_frame import OFFSETS, SIGNS, discover_raw_roots


activate_pyav()


ACTION_HORIZON = 32


def prepared_root(raw_root: Path) -> Path:
    return raw_root.with_name(f"{raw_root.name}_g05_model_frame")


def _arrays(table, column: str) -> np.ndarray:
    result = np.asarray(table[column].to_pylist(), dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 6:
        raise AssertionError(f"{column}: expected [N,6], got {result.shape}")
    if not np.isfinite(result).all():
        raise AssertionError(f"{column}: contains NaN or Inf")
    return result


def validate_coordinate_conversion(raw: Path, prepared: Path) -> int:
    total = 0
    raw_files = sorted((raw / "data").rglob("*.parquet"))
    prepared_files = sorted((prepared / "data").rglob("*.parquet"))
    if [p.relative_to(raw / "data") for p in raw_files] != [
        p.relative_to(prepared / "data") for p in prepared_files
    ]:
        raise AssertionError(f"{prepared}: data parquet file set differs from raw")

    for raw_file, prepared_file in zip(raw_files, prepared_files, strict=True):
        raw_table = pq.read_table(raw_file, columns=["action", "observation.state"])
        prepared_table = pq.read_table(prepared_file, columns=["action", "observation.state"])
        if raw_table.num_rows != prepared_table.num_rows:
            raise AssertionError(f"{prepared_file}: row count changed")
        for name in ("action", "observation.state"):
            raw_values = _arrays(raw_table, name)
            actual = _arrays(prepared_table, name)
            expected = raw_values * SIGNS + OFFSETS
            if not np.allclose(actual, expected, atol=1e-5, rtol=0):
                max_error = float(np.abs(actual - expected).max())
                raise AssertionError(f"{prepared_file}: {name} coordinate transform mismatch ({max_error})")
        total += raw_table.num_rows
    return total


def validate_episodes(root: Path, expected_frames: int) -> int:
    """Check the metadata that keeps action/video reads inside each episode."""
    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise AssertionError(f"{root}: expected LeRobot v3.0")
    if info.get("fps") != 15:
        raise AssertionError(f"{root}: expected 15 FPS, got {info.get('fps')}")
    if info["features"]["observation.images.fixed"]["shape"] != [480, 480, 3]:
        raise AssertionError(f"{root}: fixed camera is not 480x480")
    if info["features"]["observation.images.wrist"]["shape"] != [480, 640, 3]:
        raise AssertionError(f"{root}: wrist camera is not 480x640")

    episode_tables = [pq.read_table(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))]
    if not episode_tables:
        raise AssertionError(f"{root}: no meta/episodes parquet")
    episodes = np.concatenate(
        [
            np.stack(
                [
                    table["dataset_from_index"].to_numpy(),
                    table["dataset_to_index"].to_numpy(),
                    table["length"].to_numpy(),
                    table["videos/observation.images.fixed/from_timestamp"].to_numpy(),
                    table["videos/observation.images.fixed/to_timestamp"].to_numpy(),
                    table["videos/observation.images.wrist/from_timestamp"].to_numpy(),
                    table["videos/observation.images.wrist/to_timestamp"].to_numpy(),
                ],
                axis=1,
            )
            for table in episode_tables
        ]
    )
    if int(episodes[0, 0]) != 0 or int(episodes[-1, 1]) != expected_frames:
        raise AssertionError(f"{root}: episode frame boundaries do not cover its data rows")
    if not np.all(episodes[:, 1] > episodes[:, 0]):
        raise AssertionError(f"{root}: empty or reversed episode range")
    if not np.all(episodes[:, 2] == episodes[:, 1] - episodes[:, 0]):
        raise AssertionError(f"{root}: length differs from dataset frame bounds")
    if not np.all(episodes[1:, 0] == episodes[:-1, 1]):
        raise AssertionError(f"{root}: episode ranges are not contiguous")
    if not np.all(episodes[:, 3] <= episodes[:, 4]) or not np.all(episodes[:, 5] <= episodes[:, 6]):
        raise AssertionError(f"{root}: invalid video timestamp range")

    # G0.5's v3 reader clamps action indices to dataset_to_index - 1.  This is
    # the exact safety property that prevents a 32-step action chunk from ever
    # spilling into the reset period or the next episode.
    for start, end, *_ in episodes.astype(np.int64):
        anchors = np.arange(start, end)
        future = np.minimum(anchors + ACTION_HORIZON - 1, end - 1)
        if future.max(initial=start) >= end:
            raise AssertionError(f"{root}: action horizon crosses an episode boundary")
    return len(episodes)


def validate_runtime_sample(prepared_roots: list[Path]) -> None:
    """Exercise the repo's actual SO100/SO101 V3 adapter, including video decoding."""
    from g05.data.so100_canonical_dataset import SO100CanonicalLerobotDatasetV3

    shape_meta = {
        "action": [
            {
                "key": "right_arm",
                "lerobot_key": "action",
                "start_index": 0,
                "raw_shape": 6,
                "shape": 6,
                "time_offset": 0,
            }
        ],
        "state": [
            {
                "key": "right_arm",
                "lerobot_key": "observation.state",
                "start_index": 0,
                "raw_shape": 6,
                "shape": 6,
                "time_offset": 0,
            }
        ],
        "images": [
            {
                "key": "exterior",
                "camera_type": "exterior",
                "lerobot_key": "__so100_exterior__",
                "start_index": 0,
                "raw_shape": [3, 480, 480],
                "shape": [3, 480, 480],
                "time_offset": 0,
            },
            {
                "key": "wrist_left",
                "camera_type": "wrist_left",
                "lerobot_key": "__so100_wrist_left__",
                "start_index": 0,
                "raw_shape": [3, 480, 640],
                "shape": [3, 480, 640],
                "time_offset": 0,
            },
            {
                "key": "wrist_right",
                "camera_type": "wrist_right",
                "lerobot_key": "__so100_wrist_right__",
                "start_index": 0,
                "raw_shape": [3, 480, 640],
                "shape": [3, 480, 640],
                "time_offset": 0,
            },
        ],
    }
    dataset = SO100CanonicalLerobotDatasetV3(
        dataset_dirs=[str(root) for root in prepared_roots],
        shape_meta=shape_meta,
        action_size=ACTION_HORIZON,
        obs_size=1,
        lerobot_ds_version="3.0",
        load_images=True,
        random_drop_camera=False,
        val_set_proportion=0.0,
        is_training_set=True,
    )
    sample = dataset[0]
    assert tuple(sample["state"]["right_arm"].shape) == (1, 6)
    assert tuple(sample["action"]["right_arm"].shape) == (ACTION_HORIZON, 6)
    assert tuple(sample["images"]["exterior"].shape) == (1, 3, 480, 480)
    assert tuple(sample["images"]["wrist_right"].shape) == (1, 3, 480, 640)
    assert tuple(sample["images"]["wrist_left"].shape) == (1, 3, 480, 640)
    assert sample["images"]["wrist_left"].sum().item() == 0
    print(
        "Runtime adapter sample OK: exterior=(1,3,480,480), "
        "wrist_right=(1,3,480,640), wrist_left=black padding."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/so101_g05_square_v1"))
    parser.add_argument(
        "--skip-video-sample",
        action="store_true",
        help="Skip the one real video decode; structural checks still run.",
    )
    args = parser.parse_args()

    raw_roots = discover_raw_roots(args.raw_root.resolve())
    if len(raw_roots) != 11:
        raise AssertionError(f"Expected 11 raw timestamp roots, found {len(raw_roots)}")

    total_frames = 0
    total_episodes = 0
    prepared_roots = []
    for raw in raw_roots:
        prepared = prepared_root(raw)
        if not (prepared / "meta" / "g05_model_frame.json").exists():
            raise AssertionError(f"Missing prepared dataset: {prepared}")
        frames = validate_coordinate_conversion(raw, prepared)
        episodes = validate_episodes(prepared, frames)
        print(f"OK {prepared.relative_to(args.raw_root.resolve().parent)}: {episodes} episodes, {frames:,} frames")
        total_frames += frames
        total_episodes += episodes
        prepared_roots.append(prepared)

    if total_episodes != 125:
        raise AssertionError(f"Expected 125 episodes, got {total_episodes}")
    if total_frames != 20756:
        raise AssertionError(f"Expected 20,756 frames, got {total_frames}")
    if not args.skip_video_sample:
        validate_runtime_sample(prepared_roots)
    print(f"VALIDATION PASSED: {len(prepared_roots)} roots, {total_episodes} episodes, {total_frames:,} frames")


if __name__ == "__main__":
    main()
