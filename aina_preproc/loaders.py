from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath
from typing import Any

from .config import SourceConfig


def load_hf_stream(source: SourceConfig) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with `python3 -m pip install -e .`."
        ) from exc

    if source.data_files:
        return load_data_files(source, list(source.data_files))

    kwargs: dict[str, Any] = {
        "path": source.hf_id,
        "streaming": True,
        "trust_remote_code": source.trust_remote_code,
    }
    if source.data_dir:
        kwargs["data_dir"] = source.data_dir
    if source.config_name:
        kwargs["name"] = source.config_name
    if source.revision:
        kwargs["revision"] = source.revision

    split = source.split or "train"
    try:
        return load_dataset(**kwargs, split=split)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        return load_data_files(source, discover_hf_data_files(source))


def resolve_data_files(source: SourceConfig) -> list[str] | None:
    if source.data_files:
        return list(source.data_files)
    try:
        return discover_hf_data_files(source)
    except Exception:
        return None


def iter_data_files_from_cursor(
    source: SourceConfig,
    data_files: list[str],
    *,
    start_file_index: int = 0,
    start_row_offset: int = 0,
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    for file_index, data_file in enumerate(data_files[start_file_index:], start=start_file_index):
        row_offset = start_row_offset if file_index == start_file_index else 0
        rows = load_data_files(source, [data_file])
        for row_index, row in enumerate(rows):
            if row_index < row_offset:
                continue
            yield file_index, row_index + 1, row


def load_data_files(source: SourceConfig, data_files: list[str]) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with `python3 -m pip install -e .`."
        ) from exc

    if not data_files:
        raise RuntimeError(f"No parquet/json data files found for dataset source {source.name}.")

    builder = detect_builder(data_files)
    split = source.split or "train"
    return load_dataset(
        builder,
        data_files={split: data_files},
        split=split,
        streaming=True,
    )


def discover_hf_data_files(source: SourceConfig) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. It is installed with `datasets`."
        ) from exc

    api = HfApi()
    files = api.list_repo_files(source.hf_id, repo_type="dataset", revision=source.revision)
    candidates = [
        path
        for path in files
        if path.endswith((".parquet", ".jsonl", ".json"))
        and (not source.data_dir or path.startswith(source.data_dir.rstrip("/") + "/"))
    ]

    if source.config_name:
        lowered = source.config_name.lower()
        subset_matches = [path for path in candidates if lowered in path.lower()]
        if subset_matches:
            candidates = subset_matches

    candidates = sorted(candidates, key=natural_file_key)
    if source.data_file_limit is not None:
        candidates = candidates[: source.data_file_limit]

    return [f"hf://datasets/{source.hf_id}/{path}" for path in candidates]


def detect_builder(data_files: list[str]) -> str:
    first = data_files[0].lower()
    if first.endswith(".parquet"):
        return "parquet"
    if first.endswith((".jsonl", ".json")):
        return "json"
    raise ValueError(f"Unsupported data file type: {data_files[0]}")


def natural_file_key(path: str) -> tuple[str, int]:
    name = PurePosixPath(path).name
    digits = "".join(char for char in name if char.isdigit())
    return path, int(digits or 0)


def skip_seen(rows: Iterable[dict[str, Any]], count: int) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        if index >= count:
            yield row
