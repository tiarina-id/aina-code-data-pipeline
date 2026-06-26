from __future__ import annotations

import json
import math
import shutil
import sys
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


_PROGRESS_ACTIVE = False
_PROGRESS_WIDTH = 0


def log_progress(message: str) -> None:
    clear_progress_line()
    print(message, flush=True)


def emit_progress_line(message: str) -> None:
    global _PROGRESS_ACTIVE, _PROGRESS_WIDTH
    if not sys.stdout.isatty():
        print(message, flush=True)
        return
    line = fit_terminal(message)
    padding = " " * max(0, _PROGRESS_WIDTH - len(line))
    sys.stdout.write(f"\r{line}{padding}")
    sys.stdout.flush()
    _PROGRESS_ACTIVE = True
    _PROGRESS_WIDTH = len(line)


def clear_progress_line() -> None:
    global _PROGRESS_ACTIVE, _PROGRESS_WIDTH
    if not sys.stdout.isatty() or not _PROGRESS_ACTIVE:
        return
    sys.stdout.write("\r" + (" " * _PROGRESS_WIDTH) + "\r")
    sys.stdout.flush()
    _PROGRESS_ACTIVE = False
    _PROGRESS_WIDTH = 0


def progress_bar(value: int | float, total: int | float, *, width: int = 24) -> str:
    if total <= 0:
        return "[------------------------]   0.0%"
    ratio = min(1.0, max(0.0, float(value) / float(total)))
    filled = int(round(width * ratio))
    return f"[{'#' * filled}{'-' * (width - filled)}] {ratio * 100:5.1f}%"


def fit_terminal(line: str) -> str:
    columns = shutil.get_terminal_size((120, 20)).columns
    max_width = max(40, columns - 1)
    if len(line) <= max_width:
        return line
    return line[: max_width - 3] + "..."


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
