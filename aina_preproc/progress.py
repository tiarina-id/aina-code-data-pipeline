from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .pack import PackStats, ShardInfo
from .sft import SftShardInfo, SftStats


@dataclass
class ProgressState:
    config_hash: str
    processed_samples: dict[str, int] = field(default_factory=dict)
    source_file_indices: dict[str, int] = field(default_factory=dict)
    source_row_offsets: dict[str, int] = field(default_factory=dict)
    source_tokens: dict[str, int] = field(default_factory=dict)
    normalized_counts: dict[str, int] = field(default_factory=dict)
    filtered_counts: dict[str, int] = field(default_factory=dict)
    rejected_counts: dict[str, int] = field(default_factory=dict)
    deduplicated_count: int = 0
    pack_stats: PackStats = field(default_factory=PackStats)
    sft_stats: SftStats = field(default_factory=SftStats)
    train_bytes: int = 0
    val_bytes: int = 0
    shard_bytes: dict[str, int] = field(default_factory=dict)
    normalized_bytes: dict[str, int] = field(default_factory=dict)
    filtered_bytes: dict[str, int] = field(default_factory=dict)
    completed: bool = False

    @property
    def actual_tokens(self) -> int:
        return self.pack_stats.total_tokens


def load_progress(
    path: str | Path,
    expected_config_hash: str,
    *,
    allow_hash_upgrade: bool = False,
) -> ProgressState | None:
    progress_path = Path(path)
    if not progress_path.exists():
        return None
    raw = json.loads(progress_path.read_text())
    if raw.get("config_hash") != expected_config_hash and not allow_hash_upgrade:
        raise RuntimeError(
            "Existing progress file belongs to a different config. "
            "Use --no-resume, delete progress.json, or restore the matching config."
        )
    if raw.get("config_hash") != expected_config_hash and allow_hash_upgrade:
        raw["config_hash"] = expected_config_hash
    pack_stats_raw = raw.pop("pack_stats", {})
    pack_stats_raw["shards"] = [
        shard if isinstance(shard, ShardInfo) else ShardInfo(**shard)
        for shard in pack_stats_raw.get("shards", [])
    ]
    pack_stats = PackStats(**pack_stats_raw)
    sft_stats_raw = raw.pop("sft_stats", {})
    sft_stats_raw["shards"] = [
        shard if isinstance(shard, SftShardInfo) else SftShardInfo(**shard)
        for shard in sft_stats_raw.get("shards", [])
    ]
    sft_stats = SftStats(**sft_stats_raw)
    return ProgressState(pack_stats=pack_stats, sft_stats=sft_stats, **raw)


def save_progress(path: str | Path, state: ProgressState) -> None:
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))
    tmp_path.replace(progress_path)


def file_size(path: str | Path) -> int:
    file_path = Path(path)
    return file_path.stat().st_size if file_path.exists() else 0


def truncate_file(path: str | Path, size: int) -> None:
    file_path = Path(path)
    if file_path.exists():
        with file_path.open("r+b") as handle:
            handle.truncate(size)


def truncate_jsonl_files(paths_to_sizes: dict[str, int]) -> None:
    for path, size in paths_to_sizes.items():
        truncate_file(path, size)


def state_to_report_counts(state: ProgressState) -> dict[str, Any]:
    return {
        "actual_tokens": state.actual_tokens,
        "train_tokens": state.pack_stats.train_tokens,
        "val_tokens": state.pack_stats.val_tokens,
        "filtered_count": sum(state.rejected_counts.values()),
        "deduplicated_count": state.deduplicated_count,
    }
