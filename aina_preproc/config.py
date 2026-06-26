from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class SourceConfig:
    name: str
    type: str
    hf_id: str
    target_tokens: int
    mix_role: str | None = None
    config_name: str | None = None
    data_dir: str | None = None
    data_files: tuple[str, ...] | None = None
    data_file_limit: int | None = None
    split: str | None = None
    language: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    project_name: str
    target_tokens: int
    sequence_length: int
    val_ratio: float
    seed: int
    tokenizer_path: str
    output_dir: str
    sources: tuple[SourceConfig, ...]
    output_mode: str = "pretrain"
    artifact_format: str | None = None
    tokens_per_sample: int | None = None
    loss_shift: str = "model_internal"
    fallback_tokenizer: str | None = None
    work_dir: str = "/data/aina-code/work"
    progress_path: str = "/data/aina-code/work/progress.json"
    report_path: str = "reports/dataset_report.json"
    s3_output: str | None = None
    base_ratio: float = 0.9
    instruct_ratio: float = 0.1
    mixed_instruct_ratio: float = 0.0
    checkpoint_interval_tokens: int = 10_000_000
    s3_upload_interval_tokens: int | None = None
    shard_sequences: int = 100_000
    sft_samples_per_shard: int = 10_000
    write_intermediate_jsonl: bool = True
    num_workers: int = 1
    worker_batch_size: int = 32
    worker_start_method: str = "fork"
    log_interval_seconds: int = 60

    @property
    def sample_length(self) -> int:
        return self.tokens_per_sample or self.sequence_length

    @property
    def normalized_dir(self) -> Path:
        return Path(self.work_dir) / "normalized"

    @property
    def filtered_dir(self) -> Path:
        return Path(self.work_dir) / "filtered"

    @property
    def tokenized_dir(self) -> Path:
        return Path(self.work_dir) / "tokenized"

    @property
    def packed_dir(self) -> Path:
        return Path(self.output_dir)

    def with_overrides(
        self,
        *,
        target_tokens: int | None = None,
        output_dir: str | None = None,
        sequence_length: int | None = None,
        work_dir: str | None = None,
        progress_path: str | None = None,
        report_path: str | None = None,
        num_workers: int | None = None,
        worker_batch_size: int | None = None,
        worker_start_method: str | None = None,
        log_interval_seconds: int | None = None,
    ) -> "PipelineConfig":
        return dataclasses.replace(
            self,
            target_tokens=target_tokens if target_tokens is not None else self.target_tokens,
            output_dir=output_dir if output_dir is not None else self.output_dir,
            sequence_length=sequence_length if sequence_length is not None else self.sequence_length,
            work_dir=work_dir if work_dir is not None else self.work_dir,
            progress_path=progress_path if progress_path is not None else self.progress_path,
            report_path=report_path if report_path is not None else self.report_path,
            num_workers=num_workers if num_workers is not None else self.num_workers,
            worker_batch_size=worker_batch_size if worker_batch_size is not None else self.worker_batch_size,
            worker_start_method=worker_start_method if worker_start_method is not None else self.worker_start_method,
            log_interval_seconds=log_interval_seconds if log_interval_seconds is not None else self.log_interval_seconds,
        )


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    sources = tuple(SourceConfig(**source) for source in raw.pop("sources"))
    return PipelineConfig(sources=sources, **raw)


def config_hash(config: PipelineConfig) -> str:
    payload = dataclasses.asdict(config)
    for operational_key in [
        "checkpoint_interval_tokens",
        "s3_upload_interval_tokens",
        "s3_output",
        "write_intermediate_jsonl",
        "num_workers",
        "worker_batch_size",
        "worker_start_method",
        "log_interval_seconds",
    ]:
        payload.pop(operational_key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_dirs(config: PipelineConfig) -> None:
    for path in [
        Path(config.work_dir),
        config.normalized_dir,
        config.filtered_dir,
        config.tokenized_dir,
        config.packed_dir,
        Path(config.report_path).parent,
        Path(config.progress_path).parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def dataclass_to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value
