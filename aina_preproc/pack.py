from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ShardInfo:
    split: str
    index: int
    path: str
    tokens: int = 0
    sequences: int = 0
    closed: bool = False


@dataclass
class PackStats:
    train_tokens: int = 0
    val_tokens: int = 0
    train_sequences: int = 0
    val_sequences: int = 0
    dropped_remainder_tokens: int = 0
    train_shard_index: int = 0
    val_shard_index: int = 0
    shards: list[ShardInfo] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.train_tokens + self.val_tokens


class PackedDatasetWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        sequence_length: int,
        dtype: str,
        val_ratio: float,
        seed: int,
        shard_sequences: int = 100_000,
        append: bool = False,
        initial_stats: PackStats | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length
        self.dtype = normalize_dtype(dtype)
        self.val_ratio = val_ratio
        self.seed = seed
        self.shard_sequences = shard_sequences
        self.buffer: list[int] = []
        self.stats = initial_stats or PackStats()
        self.handles = {
            "train": self._open_current_shard("train", append=append),
            "val": self._open_current_shard("val", append=append),
        }
        self.write_manifest()

    def add_tokens(self, tokens: list[int]) -> None:
        self.buffer.extend(tokens)
        while len(self.buffer) >= self.sequence_length:
            block = self.buffer[: self.sequence_length]
            del self.buffer[: self.sequence_length]
            self._write_block(block)

    def flush_remainder(self) -> None:
        self.stats.dropped_remainder_tokens += len(self.buffer)
        self.buffer.clear()
        self.write_manifest()

    def flush_handles(self) -> None:
        for handle in self.handles.values():
            handle.flush()
        self.write_manifest()

    def close(self) -> None:
        self.flush_handles()
        for handle in self.handles.values():
            handle.close()

    def current_output_files(self) -> list[Path]:
        return [self.output_dir / shard.path for shard in self.stats.shards]

    def _write_block(self, block: list[int]) -> None:
        values = array(array_typecode(self.dtype), block)
        sequence_index = self.stats.train_sequences + self.stats.val_sequences
        split = "val" if is_val_sequence(sequence_index, self.seed, self.val_ratio) else "train"
        shard = self._current_shard(split)
        if shard.sequences >= self.shard_sequences:
            self._rotate_shard(split)
            shard = self._current_shard(split)

        values.tofile(self.handles[split])
        shard.tokens += len(block)
        shard.sequences += 1
        if split == "val":
            self.stats.val_tokens += len(block)
            self.stats.val_sequences += 1
        else:
            self.stats.train_tokens += len(block)
            self.stats.train_sequences += 1

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
        mode = "ab" if append and shard.tokens else "wb"
        return (self.output_dir / shard.path).open(mode)

    def _ensure_current_shard(self, split: str) -> ShardInfo:
        index = self.stats.val_shard_index if split == "val" else self.stats.train_shard_index
        path = shard_path(split, index)
        for shard in self.stats.shards:
            if shard.split == split and shard.index == index:
                return shard
        shard = ShardInfo(split=split, index=index, path=path)
        self.stats.shards.append(shard)
        return shard

    def _current_shard(self, split: str) -> ShardInfo:
        return self._ensure_current_shard(split)

    def write_manifest(self) -> None:
        manifest = {
            "sequence_length": self.sequence_length,
            "dtype": self.dtype,
            "shard_sequences": self.shard_sequences,
            "train_tokens": self.stats.train_tokens,
            "val_tokens": self.stats.val_tokens,
            "total_tokens": self.stats.total_tokens,
            "shards": [asdict(shard) for shard in self.stats.shards if shard.tokens > 0],
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def __enter__(self) -> "PackedDatasetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def validate_shards(output_dir: str | Path, stats: PackStats, dtype: str, sequence_length: int) -> tuple[int, bool]:
    total_tokens = 0
    for shard in stats.shards:
        if shard.tokens == 0:
            continue
        path = Path(output_dir) / shard.path
        tokens, ok = validate_bin(path, dtype, sequence_length)
        total_tokens += tokens
        if not ok or tokens != shard.tokens:
            return total_tokens, False
    return total_tokens, total_tokens == stats.total_tokens


def validate_bin(path: str | Path, dtype: str, sequence_length: int) -> tuple[int, bool]:
    file_path = Path(path)
    if not file_path.exists():
        return 0, False
    itemsize = dtype_itemsize(dtype)
    token_count = file_path.stat().st_size // itemsize
    return token_count, token_count % sequence_length == 0


def shard_path(split: str, index: int) -> str:
    return f"{split}-{index:05d}.bin"


def is_val_sequence(sequence_index: int, seed: int, val_ratio: float) -> bool:
    payload = f"{seed}:{sequence_index}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < val_ratio


def normalize_dtype(dtype: str) -> str:
    if dtype not in {"uint16", "uint32"}:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype


def array_typecode(dtype: str) -> str:
    return {"uint16": "H", "uint32": "I"}[normalize_dtype(dtype)]


def dtype_itemsize(dtype: str) -> int:
    return {"uint16": 2, "uint32": 4}[normalize_dtype(dtype)]
