#!/usr/bin/env python
"""Check prepared SO101 RL labels and generated DPO pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BAD_EPISODE_UID = "20260701_114422_ep00003"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    args = parser.parse_args()

    label_files = sorted(args.prepared_root.glob("*/rl_rollout_labels_prepared.jsonl"))
    if len(label_files) != 5:
        raise SystemExit(f"Expected 5 prepared run label files, got {len(label_files)}")

    raw_rows = []
    for path in label_files:
        raw_rows.extend(read_jsonl(path))
    filtered = [row for row in raw_rows if row.get("episode_uid") != BAD_EPISODE_UID]
    success = [row for row in filtered if bool(row.get("success"))]
    failure = [row for row in filtered if not bool(row.get("success"))]
    pairs = read_jsonl(args.pairs)
    pair_text = "\n".join(json.dumps(pair, ensure_ascii=False) for pair in pairs)

    checks = {
        "raw_labels": len(raw_rows),
        "filtered_labels": len(filtered),
        "success_episodes": len(success),
        "failure_episodes": len(failure),
        "pairs": len(pairs),
        "bad_uid_in_success": any(row.get("episode_uid") == BAD_EPISODE_UID for row in success),
        "bad_uid_in_pairs": BAD_EPISODE_UID in pair_text,
    }
    print(json.dumps(checks, indent=2, ensure_ascii=False))

    expected = {
        "raw_labels": 53,
        "filtered_labels": 52,
        "success_episodes": 27,
        "failure_episodes": 25,
        "pairs": 85,
        "bad_uid_in_success": False,
        "bad_uid_in_pairs": False,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise SystemExit(f"Check failed: {key} expected {value}, got {checks[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

