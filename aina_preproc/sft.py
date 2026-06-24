from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SftShardInfo:
    split: str
    index: int
    path: str
    samples: int = 0
    tokens: int = 0
    closed: bool = False


@dataclass
class SftStats:
    train_samples: int = 0
    val_samples: int = 0
    train_tokens: int = 0
    val_tokens: int = 0
    train_shard_index: int = 0
    val_shard_index: int = 0
    shards: list[SftShardInfo] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return self.train_samples + self.val_samples

    @property
    def total_tokens(self) -> int:
        return self.train_tokens + self.val_tokens


class SftJsonlShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        samples_per_shard: int,
        val_ratio: float,
        seed: int,
        append: bool = False,
        initial_stats: SftStats | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self.val_ratio = val_ratio
        self.seed = seed
        self.stats = initial_stats or SftStats()
        self.handles = {
            "train": self._open_current_shard("train", append=append),
            "val": self._open_current_shard("val", append=append),
        }
        self.write_manifest()

    def add_record(self, record: dict[str, Any], token_count: int) -> None:
        sample_index = self.stats.total_samples
        split = "val" if is_val_sample(sample_index, self.seed, self.val_ratio) else "train"
        shard = self._current_shard(split)
        if shard.samples >= self.samples_per_shard:
            self._rotate_shard(split)
            shard = self._current_shard(split)

        self.handles[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self.handles[split].write("\n")
        shard.samples += 1
        shard.tokens += token_count
        if split == "val":
            self.stats.val_samples += 1
            self.stats.val_tokens += token_count
        else:
            self.stats.train_samples += 1
            self.stats.train_tokens += token_count

    def flush_handles(self) -> None:
        for handle in self.handles.values():
            handle.flush()
        self.write_manifest()

    def close(self) -> None:
        self.flush_handles()
        for handle in self.handles.values():
            handle.close()

    def _rotate_shard(self, split: str) -> None:
        current = self._current_shard(split)
        current.closed = True
        self.handles[split].flush()
        self.handles[split].close()
        if split == "val":
            self.stats.val_shard_index += 1
        else:
            self.stats.train_shard_index += 1
        self.handles[split] = self._open_current_shard(split, append=False)
        self.write_manifest()

    def _open_current_shard(self, split: str, *, append: bool) -> object:
        shard = self._ensure_current_shard(split)
        mode = "a" if append and shard.samples else "w"
        return (self.output_dir / shard.path).open(mode, encoding="utf-8")

    def _ensure_current_shard(self, split: str) -> SftShardInfo:
        index = self.stats.val_shard_index if split == "val" else self.stats.train_shard_index
        path = sft_shard_path(split, index)
        for shard in self.stats.shards:
            if shard.split == split and shard.index == index:
                return shard
        shard = SftShardInfo(split=split, index=index, path=path)
        self.stats.shards.append(shard)
        return shard

    def _current_shard(self, split: str) -> SftShardInfo:
        return self._ensure_current_shard(split)

    def write_manifest(self) -> None:
        manifest = {
            "format": "jsonl_messages",
            "samples_per_shard": self.samples_per_shard,
            "train_samples": self.stats.train_samples,
            "val_samples": self.stats.val_samples,
            "total_samples": self.stats.total_samples,
            "train_tokens": self.stats.train_tokens,
            "val_tokens": self.stats.val_tokens,
            "total_tokens": self.stats.total_tokens,
            "shards": [asdict(shard) for shard in self.stats.shards if shard.samples > 0],
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def __enter__(self) -> "SftJsonlShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def sft_shard_path(split: str, index: int) -> str:
    return f"{split}-{index:05d}.jsonl"


def is_val_sample(sample_index: int, seed: int, val_ratio: float) -> bool:
    payload = f"sft:{seed}:{sample_index}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < val_ratio

