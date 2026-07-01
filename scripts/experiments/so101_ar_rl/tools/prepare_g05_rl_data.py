#!/usr/bin/env python
"""Prepare SO101 RL raw LeRobot roots for G0.5 model-frame training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.so101_square_finetune.prepare_g05_model_frame import (  # noqa: E402
    MANIFEST_NAME,
    OFFSETS,
    SIGNS,
    transform_data_parquet,
    transform_episode_parquet,
    transform_global_stats,
)


def discover_run_roots(raw_root: Path) -> list[Path]:
    return sorted(path.parent.parent for path in raw_root.glob("*/meta/info.json"))


def symlink_directory(source: Path, destination: Path) -> None:
    if source.exists():
        os.symlink(source.resolve(), destination, target_is_directory=True)


def rewrite_labels(source_root: Path, destination_root: Path) -> int:
    source_labels = source_root / "rl_rollout_labels.jsonl"
    if not source_labels.exists():
        return 0
    rows = []
    for line in source_labels.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["raw_dataset_dir"] = str(source_root.resolve())
        row["prepared_dataset_dir"] = str(destination_root.resolve())
        row["dataset_dir"] = str(destination_root.resolve())
        rows.append(row)

    for name in ("rl_rollout_labels_prepared.jsonl", "rl_rollout_labels.jsonl"):
        out = destination_root / name
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def prepare_one(source_root: Path, destination_root: Path, force: bool) -> tuple[Path, int, int]:
    manifest = destination_root / "meta" / MANIFEST_NAME
    if destination_root.exists():
        if manifest.exists() and not force:
            labels = rewrite_labels(source_root, destination_root)
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            print(f"SKIP {destination_root} (already prepared)")
            return destination_root, int(existing.get("total_frames", 0)), labels
        if not force:
            raise FileExistsError(f"Destination exists without reusable manifest: {destination_root}")
        shutil.rmtree(destination_root)

    temporary_root = destination_root.with_name(f".{destination_root.name}.tmp-{os.getpid()}")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)

    try:
        temporary_root.mkdir(parents=True)
        shutil.copytree(source_root / "meta", temporary_root / "meta")
        symlink_directory(source_root / "videos", temporary_root / "videos")
        symlink_directory(source_root / "recording_context", temporary_root / "recording_context")

        total_rows = 0
        for parquet in sorted((source_root / "data").rglob("*.parquet")):
            relative = parquet.relative_to(source_root / "data")
            total_rows += transform_data_parquet(parquet, temporary_root / "data" / relative)

        for parquet in sorted((source_root / "meta" / "episodes").rglob("*.parquet")):
            relative = parquet.relative_to(source_root / "meta" / "episodes")
            transform_episode_parquet(parquet, temporary_root / "meta" / "episodes" / relative)

        transform_global_stats(temporary_root / "meta" / "stats.json")
        (temporary_root / "meta" / MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "source_root": str(source_root.resolve()),
                    "prepared_root": str(destination_root.resolve()),
                    "coordinate_transform": {
                        "formula": "q_model = signs * q_arm + offsets",
                        "signs": SIGNS.astype(int).tolist(),
                        "offsets_degrees": OFFSETS.astype(int).tolist(),
                    },
                    "videos": "symlinked from source; no re-encoding",
                    "total_frames": total_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_root, destination_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    labels = rewrite_labels(source_root, destination_root)
    print(f"READY {destination_root} ({total_rows:,} frames, {labels} labels)")
    return destination_root, total_rows, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    prepared_root = args.prepared_root.resolve()
    run_roots = discover_run_roots(raw_root)
    if not run_roots:
        raise SystemExit(f"No run roots found under {raw_root}")

    prepared_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    total_frames = 0
    total_labels = 0
    for source in run_roots:
        dest = prepared_root / source.name
        path, frames, labels = prepare_one(source, dest, force=args.force)
        prepared.append(str(path))
        total_frames += frames
        total_labels += labels

    summary = {
        "raw_root": str(raw_root),
        "prepared_root": str(prepared_root),
        "runs": prepared,
        "total_frames": total_frames,
        "total_labels": total_labels,
    }
    summary_path = prepared_root / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(prepared)} roots, {total_frames:,} frames, {total_labels} labels.")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
