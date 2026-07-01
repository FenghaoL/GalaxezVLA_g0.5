"""SO101 AR-SFT / AR-DPO dataset wrappers.

These wrappers deliberately reuse the existing LeRobot dataset and processor.
They only change which already-processed samples are exposed to the trainer:

* SuccessFrameDataset: all frames from successful labelled episodes.
* PreferencePairDataset: fixed anchor frames from success/failure episode pairs.

The raw/prepared LeRobot roots remain untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from g05.data.mixture_lerobot_dataset import MixtureLerobotDataset
from g05.utils.data.data_utils import collate_fn_pad_sequences

logger = logging.getLogger(__name__)


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
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_label_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    by_parent: dict[Path, Path] = {}
    for path in sorted(root.glob("**/rl_rollout_labels.jsonl")):
        by_parent[path.parent] = path
    for path in sorted(root.glob("**/rl_rollout_labels_prepared.jsonl")):
        by_parent[path.parent] = path
    return sorted(by_parent.values())


def load_labels(root: str | Path, exclude_episode_uids: Optional[set[str]] = None) -> list[dict[str, Any]]:
    root = Path(root)
    exclude_episode_uids = exclude_episode_uids or set()
    rows: list[dict[str, Any]] = []
    for label_file in find_label_files(root):
        for row in read_jsonl(label_file):
            if row.get("episode_uid") in exclude_episode_uids:
                continue
            row = dict(row)
            row["_label_file"] = str(label_file)
            rows.append(row)
    return rows


def _episode_table(root: Path) -> list[dict[str, Any]]:
    episode_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No meta/episodes parquet found under {root}")
    rows: list[dict[str, Any]] = []
    for path in episode_files:
        table = pq.read_table(
            path,
            columns=["episode_index", "dataset_from_index", "dataset_to_index", "length"],
        )
        rows.extend(table.to_pylist())
    return rows


def _path_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _root_from_inner_dataset(inner: Any) -> Path:
    root = getattr(inner, "root", None)
    if root is None:
        root = getattr(inner, "repo_id", None)
    if root is None:
        raise ValueError(f"Cannot resolve dataset root for {inner!r}")
    return Path(root)


@dataclass(frozen=True)
class EpisodeRange:
    episode_uid: str
    dataset_dir: str
    episode_index: int
    dataset_idx: int
    local_start: int
    local_end: int
    start: int
    end: int
    length: int
    instruction: str
    init_config_id: str
    source: str
    success: bool


class EpisodeIndex:
    """Map label episode references to global indices in an existing Mixture dataset."""

    def __init__(self, base_dataset: Dataset, labels: list[dict[str, Any]]):
        if not isinstance(base_dataset, MixtureLerobotDataset):
            raise TypeError(
                "SO101 RL wrappers expect the existing MixtureLerobotDataset. "
                f"Got {type(base_dataset).__name__}."
            )
        if getattr(base_dataset, "use_weight_for_sampling", False):
            raise ValueError(
                "SO101 RL wrappers require data.use_weight_for_sampling=false so label "
                "episode indices map one-to-one to dataset indices."
            )

        self.base_dataset = base_dataset
        self.labels = labels
        self._ranges_by_uid: dict[str, EpisodeRange] = {}
        self._ranges_by_ref: dict[tuple[str, int], EpisodeRange] = {}
        self.skipped_episode_uids: list[str] = []
        self._build()

    @staticmethod
    def _label_dataset_dir(row: dict[str, Any]) -> str:
        dataset_dir = row.get("prepared_dataset_dir") or row.get("dataset_dir")
        if not dataset_dir:
            raise KeyError(f"Label lacks dataset_dir/prepared_dataset_dir: {row}")
        return _path_key(dataset_dir)

    def _build_root_offsets(self) -> dict[str, tuple[int, int, int]]:
        """Return root -> (mixture_dataset_idx, mixture_start, episode_start_offset)."""
        root_offsets: dict[str, tuple[int, int, int]] = {}
        for mixture_dataset_idx, inner_group in enumerate(self.base_dataset.datasets):
            group_start = int(self.base_dataset.effective_starts[mixture_dataset_idx])
            episode_offset = 0
            for shard_idx, shard in enumerate(inner_group.multi_dataset._datasets):
                root = _path_key(_root_from_inner_dataset(shard))
                root_offsets[root] = (mixture_dataset_idx, group_start, episode_offset)
                episode_offset += int(getattr(shard, "num_episodes", 0))
        return root_offsets

    def _build(self) -> None:
        root_offsets = self._build_root_offsets()

        for row in self.labels:
            dataset_dir = self._label_dataset_dir(row)
            if dataset_dir not in root_offsets:
                known = "\n  ".join(sorted(root_offsets)[:10])
                raise KeyError(
                    f"Label dataset_dir is not present in the instantiated dataset: {dataset_dir}\n"
                    f"Known roots begin with:\n  {known}"
                )

            episode_index = int(row["episode_index"])
            mixture_dataset_idx, mixture_group_start, episode_start_offset = root_offsets[dataset_dir]
            inner_group = self.base_dataset.datasets[mixture_dataset_idx]
            inner_episode_index = episode_start_offset + episode_index
            local_start = int(inner_group.episode_data_index["from"][inner_episode_index].item())
            local_end = int(inner_group.episode_data_index["to"][inner_episode_index].item())
            local_end = min(local_end, len(inner_group))
            label_count = row.get("frame_count")
            if label_count is not None:
                local_end = min(local_end, local_start + int(label_count))

            start = mixture_group_start + local_start
            end = mixture_group_start + local_end
            if end <= start:
                episode_uid = str(row.get("episode_uid") or f"{dataset_dir}:{episode_index}")
                self.skipped_episode_uids.append(episode_uid)
                logger.warning(
                    "[SO101 RL] skipping episode with no indexable training frames: "
                    "episode_uid=%s range=%s..%s dataset_dir=%s episode_index=%s",
                    episode_uid,
                    start,
                    end,
                    dataset_dir,
                    episode_index,
                )
                continue

            ep_range = EpisodeRange(
                episode_uid=str(row["episode_uid"]),
                dataset_dir=dataset_dir,
                episode_index=episode_index,
                dataset_idx=mixture_dataset_idx,
                local_start=local_start,
                local_end=local_end,
                start=start,
                end=end,
                length=end - start,
                instruction=str(row.get("instruction") or ""),
                init_config_id=str(row.get("init_config_id") or ""),
                source=str(row.get("source") or ""),
                success=bool(row.get("success")),
            )
            self._ranges_by_uid[ep_range.episode_uid] = ep_range
            self._ranges_by_ref[(dataset_dir, episode_index)] = ep_range

    def by_uid(self, episode_uid: str) -> EpisodeRange:
        return self._ranges_by_uid[episode_uid]

    def by_pair_ref(self, ref: dict[str, Any]) -> EpisodeRange:
        uid = ref.get("episode_uid")
        if uid and uid in self._ranges_by_uid:
            return self._ranges_by_uid[str(uid)]
        dataset_dir = ref.get("prepared_dataset_dir") or ref.get("dataset_dir")
        if dataset_dir is None:
            raise KeyError(f"Pair ref lacks episode_uid or dataset_dir: {ref}")
        return self._ranges_by_ref[(_path_key(dataset_dir), int(ref["episode_index"]))]


def _get_mixture_inner_sample(base_dataset: MixtureLerobotDataset, dataset_idx: int, local_idx: int):
    sample = base_dataset.datasets[int(dataset_idx)][int(local_idx)]
    emb_type = base_dataset.embodiments2types[base_dataset.embodiments[int(dataset_idx)]]
    sample["embodiment"] = emb_type
    sample["embodiment_type"] = emb_type
    if "samples" in sample and "action" in sample["samples"] and "proprio" in sample["samples"]:
        sample["samples"]["action"]["embodiment"] = emb_type
        sample["samples"]["proprio"]["embodiment"] = emb_type
    return sample


class SuccessFrameDataset(Dataset):
    """Expose every frame from successful labelled episodes only."""

    def __init__(
        self,
        base_dataset: Dataset,
        labels_root: str | Path,
        exclude_episode_uids: Optional[Iterable[str]] = None,
    ):
        self.base_dataset = base_dataset
        self.exclude_episode_uids = set(exclude_episode_uids or [])
        self.labels = load_labels(labels_root, self.exclude_episode_uids)
        self.episode_index = EpisodeIndex(base_dataset, self.labels)

        self.success_labels = [row for row in self.labels if bool(row.get("success"))]
        self.indices: list[tuple[int, int]] = []
        self.success_episode_uids: list[str] = []
        for row in self.success_labels:
            episode_uid = str(row["episode_uid"])
            if episode_uid not in self.episode_index._ranges_by_uid:
                continue
            ep_range = self.episode_index.by_uid(episode_uid)
            self.success_episode_uids.append(ep_range.episode_uid)
            self.indices.extend(
                (ep_range.dataset_idx, local_idx)
                for local_idx in range(ep_range.local_start, ep_range.local_end)
            )

        self.summary = {
            "success_episodes": len(self.success_labels),
            "trainable_success_episodes": len(self.success_episode_uids),
            "frames": len(self.indices),
            "excluded_episode_uids": sorted(self.exclude_episode_uids),
            "skipped_episode_uids": sorted(self.episode_index.skipped_episode_uids),
            "source_success_counts": dict(
                Counter(str(row.get("source") or "") for row in self.success_labels)
            ),
        }
        logger.info("[SO101 AR-SFT] %s", self.summary)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dataset_idx, local_idx = self.indices[int(idx)]
        return _get_mixture_inner_sample(self.base_dataset, dataset_idx, local_idx)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)


def _anchor_indices(ep_range: EpisodeRange, anchor_count: int) -> list[tuple[int, int]]:
    anchor_count = max(1, int(anchor_count))
    if ep_range.length <= anchor_count:
        return [
            (ep_range.dataset_idx, local_idx)
            for local_idx in range(ep_range.local_start, ep_range.local_end)
        ]
    # Avoid the very first/last frames when possible; tails often contain padding.
    rel = np.linspace(0.15, 0.85, anchor_count)
    offsets = np.rint(rel * (ep_range.length - 1)).astype(int)
    offsets = np.clip(offsets, 0, ep_range.length - 1)
    # np.rint can collide for very short episodes; de-duplicate then fill.
    unique = []
    for off in offsets.tolist():
        if off not in unique:
            unique.append(off)
    cursor = 0
    while len(unique) < anchor_count:
        if cursor not in unique:
            unique.append(cursor)
        cursor += 1
    return [
        (ep_range.dataset_idx, ep_range.local_start + off)
        for off in unique[:anchor_count]
    ]


class PreferencePairDataset(Dataset):
    """Return chosen/rejected fixed-anchor sample groups for AR-DPO."""

    def __init__(
        self,
        base_dataset: Dataset,
        pairs_path: str | Path,
        labels_root: str | Path,
        anchor_count: int = 4,
        exclude_episode_uids: Optional[Iterable[str]] = None,
        ref_logps_path: str | Path | None = None,
    ):
        self.base_dataset = base_dataset
        self.pairs_path = Path(pairs_path)
        self.labels_root = Path(labels_root)
        self.anchor_count = int(anchor_count)
        self.exclude_episode_uids = set(exclude_episode_uids or [])
        self.labels = load_labels(labels_root, self.exclude_episode_uids)
        self.episode_index = EpisodeIndex(base_dataset, self.labels)
        self.pairs = read_jsonl(self.pairs_path)
        self.ref_logps_path = Path(ref_logps_path) if ref_logps_path else None
        self.ref_logps: dict[str, dict[str, float]] = {}
        if self.ref_logps_path and self.ref_logps_path.exists():
            self.load_ref_logps(self.ref_logps_path)

        self._chosen_indices: list[list[tuple[int, int]]] = []
        self._rejected_indices: list[list[tuple[int, int]]] = []
        kept_pairs: list[dict[str, Any]] = []
        skipped_pairs: list[str] = []
        for pair in self.pairs:
            pair_id = str(pair.get("pair_id") or f"pair_{len(kept_pairs):05d}")
            try:
                chosen_range = self.episode_index.by_pair_ref(pair["chosen"])
                rejected_range = self.episode_index.by_pair_ref(pair["rejected"])
            except KeyError:
                skipped_pairs.append(pair_id)
                continue
            kept_pairs.append(pair)
            self._chosen_indices.append(_anchor_indices(chosen_range, self.anchor_count))
            self._rejected_indices.append(_anchor_indices(rejected_range, self.anchor_count))
        self.raw_pair_count = len(self.pairs)
        self.pairs = kept_pairs
        self.skipped_pair_ids = skipped_pairs

        self.summary = {
            "raw_pairs": self.raw_pair_count,
            "trainable_pairs": len(self.pairs),
            "anchor_count": self.anchor_count,
            "pairs_path": str(self.pairs_path),
            "ref_logps_loaded": len(self.ref_logps),
            "skipped_episode_uids": sorted(self.episode_index.skipped_episode_uids),
            "skipped_pair_ids": self.skipped_pair_ids,
            "buckets": dict(Counter(str(p.get("init_config_id") or "") for p in self.pairs)),
        }
        logger.info("[SO101 AR-DPO] %s", self.summary)

    def __len__(self) -> int:
        return len(self.pairs)

    def _pair_id(self, idx: int) -> str:
        pair = self.pairs[int(idx)]
        return str(pair.get("pair_id") or f"pair_{int(idx):05d}")

    def load_ref_logps(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            self.ref_logps = {}
            return
        refs: dict[str, dict[str, float]] = {}
        for row in read_jsonl(path):
            refs[str(row["pair_id"])] = {
                "ref_chosen_logp": float(row["ref_chosen_logp"]),
                "ref_rejected_logp": float(row["ref_rejected_logp"]),
            }
        self.ref_logps = refs
        logger.info("[SO101 AR-DPO] loaded %s cached reference logps from %s", len(refs), path)

    def has_complete_ref_logps(self) -> bool:
        return all(self._pair_id(i) in self.ref_logps for i in range(len(self.pairs)))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = int(idx)
        pair = self.pairs[idx]
        pair_id = self._pair_id(idx)
        ref = self.ref_logps.get(pair_id, {})
        chosen = [
            _get_mixture_inner_sample(self.base_dataset, dataset_idx, local_idx)
            for dataset_idx, local_idx in self._chosen_indices[idx]
        ]
        rejected = [
            _get_mixture_inner_sample(self.base_dataset, dataset_idx, local_idx)
            for dataset_idx, local_idx in self._rejected_indices[idx]
        ]
        return {
            "pair_id": pair_id,
            "chosen": chosen,
            "rejected": rejected,
            "ref_chosen_logp": float(ref.get("ref_chosen_logp", 0.0)),
            "ref_rejected_logp": float(ref.get("ref_rejected_logp", 0.0)),
            "pair_meta": {
                "pair_id": pair_id,
                "instruction": pair.get("instruction"),
                "init_config_id": pair.get("init_config_id"),
                "chosen_uid": pair.get("chosen", {}).get("episode_uid"),
                "rejected_uid": pair.get("rejected", {}).get("episode_uid"),
                "chosen_source": pair.get("chosen", {}).get("source"),
                "rejected_source": pair.get("rejected", {}).get("source"),
            },
        }

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)


def collate_preference_pairs(batch: list[dict[str, Any]], padding_input_id: int = 0) -> dict[str, Any]:
    chosen_samples: list[dict[str, Any]] = []
    rejected_samples: list[dict[str, Any]] = []
    anchor_counts: list[int] = []
    pair_meta: list[dict[str, Any]] = []
    pair_ids: list[str] = []
    ref_chosen: list[float] = []
    ref_rejected: list[float] = []

    for item in batch:
        chosen = list(item["chosen"])
        rejected = list(item["rejected"])
        if len(chosen) != len(rejected):
            raise ValueError("Chosen and rejected anchor counts must match")
        anchor_counts.append(len(chosen))
        chosen_samples.extend(chosen)
        rejected_samples.extend(rejected)
        pair_meta.append(item["pair_meta"])
        pair_ids.append(str(item["pair_id"]))
        ref_chosen.append(float(item["ref_chosen_logp"]))
        ref_rejected.append(float(item["ref_rejected_logp"]))

    return {
        "chosen": collate_fn_pad_sequences(chosen_samples, padding_input_id=padding_input_id),
        "rejected": collate_fn_pad_sequences(rejected_samples, padding_input_id=padding_input_id),
        "anchor_counts": torch.tensor(anchor_counts, dtype=torch.long),
        "ref_chosen_logp": torch.tensor(ref_chosen, dtype=torch.float32),
        "ref_rejected_logp": torch.tensor(ref_rejected, dtype=torch.float32),
        "pair_ids": pair_ids,
        "pair_meta": pair_meta,
    }


def _aggregate_anchor_logps(logps: torch.Tensor, anchor_counts: torch.Tensor) -> torch.Tensor:
    chunks = torch.split(logps, [int(x) for x in anchor_counts.detach().cpu().tolist()])
    return torch.stack([chunk.sum() for chunk in chunks], dim=0)


@torch.no_grad()
def cache_reference_logps(
    *,
    policy,
    dataset: PreferencePairDataset,
    output_path: str | Path,
    padding_input_id: int,
    batch_size: int = 1,
    num_workers: int = 0,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Compute and write fixed reference logps for every preference pair."""

    output_path = Path(output_path)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")

    if output_path.exists() and meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_meta = {}
        if metadata is None or existing_meta == metadata:
            dataset.load_ref_logps(output_path)
            if dataset.has_complete_ref_logps():
                logger.info("[SO101 AR-DPO] reference cache is already complete: %s", output_path)
                return

    was_training = policy.training
    policy.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=lambda rows: collate_preference_pairs(
            rows, padding_input_id=padding_input_id
        ),
    )
    rows: list[dict[str, Any]] = []
    device = next(policy.parameters()).device

    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, 1):
        anchor_counts = batch["anchor_counts"].to(device)
        chosen = {k: v for k, v in batch["chosen"].items()}
        rejected = {k: v for k, v in batch["rejected"].items()}
        chosen_out = policy.compute_ar_action_logps(chosen)
        rejected_out = policy.compute_ar_action_logps(rejected)
        chosen_logp = _aggregate_anchor_logps(chosen_out["logps"], anchor_counts)
        rejected_logp = _aggregate_anchor_logps(rejected_out["logps"], anchor_counts)
        for i, pair_id in enumerate(batch["pair_ids"]):
            rows.append(
                {
                    "pair_id": pair_id,
                    "ref_chosen_logp": float(chosen_logp[i].detach().cpu()),
                    "ref_rejected_logp": float(rejected_logp[i].detach().cpu()),
                    "anchor_count": int(anchor_counts[i].detach().cpu()),
                }
            )
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % 5 == 0:
            logger.info(
                "[SO101 AR-DPO] cached reference logps for %s/%s pair batches",
                batch_idx,
                total_batches,
            )

    write_jsonl(output_path, rows)
    if metadata is not None:
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset.load_ref_logps(output_path)
    policy.train(was_training)


def aggregate_anchor_logps(logps: torch.Tensor, anchor_counts: torch.Tensor) -> torch.Tensor:
    return _aggregate_anchor_logps(logps, anchor_counts)
