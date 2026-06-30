#!/usr/bin/env python3
"""Prepare local SO101 LeRobot v3 datasets for G0.5 fine-tuning.

The raw recordings use the physical SO101 motor frame.  G0.5's SO100/SO101
adapter instead expects the model frame used by its deployment client:

    q_model = signs * q_arm + offsets
    signs   = [ 1, -1, 1, 1, 1, 1]
    offsets = [ 0, 90,90, 0, 0, 0]

Only parquet state/action values and their metadata statistics are rewritten.
Videos are symlinked, never re-encoded.  The source recording is therefore
left untouched, while the prepared sibling remains a valid self-contained
LeRobot v3 root from the reader's point of view.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SIGNS = np.asarray([1, -1, 1, 1, 1, 1], dtype=np.float64)
OFFSETS = np.asarray([0, 90, 90, 0, 0, 0], dtype=np.float64)
VECTOR_COLUMNS = ("action", "observation.state")
STAT_NAMES = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
OPPOSITE_QUANTILE = {
    "q01": "q99",
    "q10": "q90",
    "q50": "q50",
    "q90": "q10",
    "q99": "q01",
}
MANIFEST_NAME = "g05_model_frame.json"


def discover_raw_roots(raw_parent: Path) -> list[Path]:
    """Return timestamp-run roots, excluding any already-prepared siblings."""
    roots = []
    for info_path in raw_parent.rglob("meta/info.json"):
        root = info_path.parent.parent
        if root.name.endswith("_g05_model_frame"):
            continue
        roots.append(root)
    return sorted(roots)


def model_frame(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != 6:
        raise ValueError(f"Expected final dimension 6, got {values.shape}")
    return values * SIGNS + OFFSETS


def _transform_stats_record(stats: dict[str, Any]) -> dict[str, Any]:
    """Transform a LeRobot stats dictionary while preserving quantile ordering."""
    transformed = copy.deepcopy(stats)
    if not all(name in stats for name in ("min", "max", "mean", "std")):
        return transformed

    lo = np.asarray(stats["min"], dtype=np.float64)
    hi = np.asarray(stats["max"], dtype=np.float64)
    if lo.shape[-1] != 6 or hi.shape[-1] != 6:
        return transformed

    transformed["min"] = np.where(SIGNS > 0, model_frame(lo), model_frame(hi)).tolist()
    transformed["max"] = np.where(SIGNS > 0, model_frame(hi), model_frame(lo)).tolist()
    transformed["mean"] = model_frame(np.asarray(stats["mean"], dtype=np.float64)).tolist()
    transformed["std"] = (np.asarray(stats["std"], dtype=np.float64) * np.abs(SIGNS)).tolist()

    for quantile, opposite in OPPOSITE_QUANTILE.items():
        if quantile not in stats or opposite not in stats:
            continue
        own = np.asarray(stats[quantile], dtype=np.float64)
        mirrored = np.asarray(stats[opposite], dtype=np.float64)
        if own.shape[-1] != 6 or mirrored.shape[-1] != 6:
            continue
        transformed[quantile] = np.where(SIGNS > 0, model_frame(own), model_frame(mirrored)).tolist()
    return transformed


def transform_global_stats(path: Path) -> None:
    """Rewrite meta/stats.json, which is not read during training but should stay honest."""
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for key in VECTOR_COLUMNS:
        if isinstance(data.get(key), dict):
            data[key] = _transform_stats_record(data[key])
    path.write_text(json.dumps(data, indent=2) + "\n")


def _replace_vector_column(table: pa.Table, name: str) -> pa.Table:
    field_index = table.schema.get_field_index(name)
    if field_index < 0:
        raise KeyError(f"Missing required column: {name}")
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"{name} must have shape [N, 6], got {values.shape}")
    converted = model_frame(values).astype(np.float32)
    field = table.schema.field(field_index)
    column = pa.array(converted.tolist(), type=field.type)
    return table.set_column(field_index, field, column)


def transform_data_parquet(source: Path, destination: Path) -> int:
    table = pq.read_table(source)
    for column in VECTOR_COLUMNS:
        table = _replace_vector_column(table, column)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="snappy")
    return table.num_rows


def _transform_episode_stats_columns(table: pa.Table, prefix: str) -> pa.Table:
    """Transform flattened stats/action/* or stats/observation.state/* parquet columns."""
    values = table.to_pydict()
    names = set(values)
    required = {f"{prefix}/{name}" for name in ("min", "max", "mean", "std")}
    if not required.issubset(names):
        return table

    lo = np.asarray(values[f"{prefix}/min"], dtype=np.float64)
    hi = np.asarray(values[f"{prefix}/max"], dtype=np.float64)
    if lo.ndim != 2 or lo.shape[-1] != 6 or hi.shape != lo.shape:
        return table

    values[f"{prefix}/min"] = np.where(SIGNS > 0, model_frame(lo), model_frame(hi)).tolist()
    values[f"{prefix}/max"] = np.where(SIGNS > 0, model_frame(hi), model_frame(lo)).tolist()
    values[f"{prefix}/mean"] = model_frame(
        np.asarray(values[f"{prefix}/mean"], dtype=np.float64)
    ).tolist()
    values[f"{prefix}/std"] = (
        np.asarray(values[f"{prefix}/std"], dtype=np.float64) * np.abs(SIGNS)
    ).tolist()

    for quantile, opposite in OPPOSITE_QUANTILE.items():
        current_name = f"{prefix}/{quantile}"
        opposite_name = f"{prefix}/{opposite}"
        if current_name not in names or opposite_name not in names:
            continue
        own = np.asarray(values[current_name], dtype=np.float64)
        mirrored = np.asarray(values[opposite_name], dtype=np.float64)
        values[current_name] = np.where(SIGNS > 0, model_frame(own), model_frame(mirrored)).tolist()

    return pa.Table.from_pydict(values, schema=table.schema)


def transform_episode_parquet(source: Path, destination: Path) -> None:
    table = pq.read_table(source)
    for prefix in ("stats/action", "stats/observation.state"):
        table = _transform_episode_stats_columns(table, prefix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="snappy")


def _symlink_directory(source: Path, destination: Path) -> None:
    if source.exists():
        os.symlink(source.resolve(), destination, target_is_directory=True)


def prepare_one(source_root: Path, force: bool) -> tuple[Path, int]:
    destination_root = source_root.with_name(f"{source_root.name}_g05_model_frame")
    manifest = destination_root / "meta" / MANIFEST_NAME
    if destination_root.exists():
        if manifest.exists() and not force:
            existing = json.loads(manifest.read_text())
            if Path(existing.get("source_root", "")).resolve() == source_root.resolve():
                print(f"SKIP {destination_root} (already prepared)")
                return destination_root, int(existing.get("total_frames", 0))
        raise FileExistsError(
            f"Prepared destination already exists: {destination_root}. "
            "Refusing to overwrite it; remove it deliberately or rerun with --force."
        )

    temporary_root = destination_root.with_name(f".{destination_root.name}.tmp-{os.getpid()}")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)

    try:
        temporary_root.mkdir(parents=True)
        shutil.copytree(source_root / "meta", temporary_root / "meta")
        _symlink_directory(source_root / "videos", temporary_root / "videos")
        _symlink_directory(source_root / "recording_context", temporary_root / "recording_context")

        total_rows = 0
        for parquet in sorted((source_root / "data").rglob("*.parquet")):
            relative = parquet.relative_to(source_root / "data")
            total_rows += transform_data_parquet(parquet, temporary_root / "data" / relative)

        for parquet in sorted((source_root / "meta" / "episodes").rglob("*.parquet")):
            relative = parquet.relative_to(source_root / "meta" / "episodes")
            transform_episode_parquet(parquet, temporary_root / "meta" / "episodes" / relative)

        transform_global_stats(temporary_root / "meta" / "stats.json")
        manifest_path = temporary_root / "meta" / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(
                {
                    "source_root": str(source_root.resolve()),
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
            + "\n"
        )
        os.replace(temporary_root, destination_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(f"READY {destination_root} ({total_rows:,} frames)")
    return destination_root, total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/so101_g05_square_v1"),
        help="Parent directory containing task/timestamp LeRobot v3 roots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing prepared sibling. This removes only the prepared sibling, never raw data.",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    roots = discover_raw_roots(raw_root)
    if not roots:
        raise RuntimeError(f"No LeRobot v3 dataset roots found under {raw_root}")

    if args.force:
        for source in roots:
            destination = source.with_name(f"{source.name}_g05_model_frame")
            if destination.exists():
                shutil.rmtree(destination)

    prepared, total_frames = [], 0
    for source in roots:
        destination, frames = prepare_one(source, force=args.force)
        prepared.append(str(destination))
        total_frames += frames

    print(f"Prepared {len(prepared)} roots, {total_frames:,} total frames.")


if __name__ == "__main__":
    main()
