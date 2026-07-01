#!/usr/bin/env python
"""Build source-agnostic success/failure SO101 trajectory pairs for AR-DPO."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BAD_EPISODE_UID = "20260701_114422_ep00003"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def find_label_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    by_parent: dict[Path, Path] = {}
    for path in sorted(root.glob("**/rl_rollout_labels.jsonl")):
        by_parent[path.parent] = path
    for path in sorted(root.glob("**/rl_rollout_labels_prepared.jsonl")):
        by_parent[path.parent] = path
    return sorted(by_parent.values())


def episode_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_uid": row["episode_uid"],
        "dataset_dir": row.get("prepared_dataset_dir") or row.get("dataset_dir"),
        "episode_index": int(row["episode_index"]),
        "source": row.get("source"),
        "success": bool(row.get("success")),
        "frame_count": row.get("frame_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-episode-uid", action="append", default=[BAD_EPISODE_UID])
    parser.add_argument("--expect-pairs", type=int, default=None)
    args = parser.parse_args()

    label_files = find_label_files(args.labels_root)
    if not label_files:
        raise SystemExit(f"No label files found under {args.labels_root}")

    exclude = set(args.exclude_episode_uid or [])
    rows: list[dict[str, Any]] = []
    for path in label_files:
        for row in read_jsonl(path):
            if row.get("episode_uid") in exclude:
                continue
            row["_label_file"] = str(path)
            rows.append(row)

    buckets: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"success": [], "failure": []}
    )
    for row in rows:
        instruction = str(row.get("instruction") or "")
        init_config_id = str(row.get("init_config_id") or row.get("bucket_label") or "")
        if not instruction or not init_config_id:
            continue
        key = (instruction, init_config_id)
        if bool(row.get("success")):
            buckets[key]["success"].append(row)
        else:
            buckets[key]["failure"].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for (instruction, init_config_id), group in sorted(buckets.items()):
        local_pairs = 0
        for chosen in group["success"]:
            for rejected in group["failure"]:
                pair_id = f"{init_config_id}_{local_pairs:05d}"
                pairs.append(
                    {
                        "format": "g05_so101_ar_dpo_pair/v1",
                        "pair_id": pair_id,
                        "instruction": instruction,
                        "init_config_id": init_config_id,
                        "chosen": episode_ref(chosen),
                        "rejected": episode_ref(rejected),
                    }
                )
                local_pairs += 1
        summary.append(
            {
                "instruction": instruction,
                "init_config_id": init_config_id,
                "success": len(group["success"]),
                "failure": len(group["failure"]),
                "pairs": local_pairs,
                "success_sources": dict(Counter(str(r.get("source") or "") for r in group["success"])),
                "failure_sources": dict(Counter(str(r.get("source") or "") for r in group["failure"])),
            }
        )

    with args.output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_doc = {
        "labels_root": str(args.labels_root),
        "label_files": [str(path) for path in label_files],
        "excluded_episode_uids": sorted(exclude),
        "usable_labels": len(rows),
        "success_episodes": sum(1 for row in rows if bool(row.get("success"))),
        "failure_episodes": sum(1 for row in rows if not bool(row.get("success"))),
        "pairs": len(pairs),
        "buckets": summary,
    }
    summary_path.write_text(json.dumps(summary_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(pairs)} pairs to {args.output}")
    print(f"Wrote summary to {summary_path}")
    if args.expect_pairs is not None and len(pairs) != args.expect_pairs:
        raise SystemExit(f"Expected {args.expect_pairs} pairs, got {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

