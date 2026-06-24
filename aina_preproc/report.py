from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .pack import PackStats


def write_metadata(
    output_dir: str | Path,
    *,
    vocab_size: int,
    dtype: str,
    sequence_length: int,
    stats: PackStats,
    tokenizer_source: str,
    sources: list[dict[str, Any]],
    output_mode: str = "pretrain",
    tokens_per_sample: int | None = None,
    loss_shift: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    metadata = {
        "vocab_size": vocab_size,
        "dtype": dtype,
        "sequence_length": sequence_length,
        "tokens_per_sample": tokens_per_sample or sequence_length,
        "loss_shift": loss_shift,
        "output_mode": output_mode,
        "total_tokens": stats.total_tokens,
        "train_tokens": stats.train_tokens,
        "val_tokens": stats.val_tokens,
        "train_sequences": stats.train_sequences,
        "val_sequences": stats.val_sequences,
        "dropped_remainder_tokens": stats.dropped_remainder_tokens,
        "tokenizer_source": tokenizer_source,
        "sources": sources,
        "shards": [
            {
                "split": shard.split,
                "index": shard.index,
                "path": shard.path,
                "tokens": shard.tokens,
                "sequences": shard.sequences,
                "closed": shard.closed,
            }
            for shard in stats.shards
            if shard.tokens > 0
        ],
        "output_files": [
            *[shard.path for shard in stats.shards if shard.tokens > 0],
            "manifest.json",
            "metadata.json",
            "tokenizer/",
        ],
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def write_dataset_report(
    path: str | Path,
    config: PipelineConfig,
    *,
    stats: PackStats,
    sources: list[dict[str, Any]],
    filtered_count: int,
    deduplicated_count: int,
    output_files: list[str],
) -> dict[str, Any]:
    report = {
        "project": config.project_name,
        "target_tokens": config.target_tokens,
        "actual_tokens": stats.total_tokens,
        "output_mode": config.output_mode,
        "sequence_length": config.sequence_length,
        "train_tokens": stats.train_tokens,
        "val_tokens": stats.val_tokens,
        "sources": sources,
        "filtered_count": filtered_count,
        "deduplicated_count": deduplicated_count,
        "output_files": output_files,
    }
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def log_progress(message: str) -> None:
    print(message, flush=True)
