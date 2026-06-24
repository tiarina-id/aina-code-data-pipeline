from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Iterable

from .config import PipelineConfig, SourceConfig, config_hash, ensure_dirs
from .dedup import ExactDeduper
from .filters import should_keep
from .jsonl import append_jsonl
from .loaders import iter_data_files_from_cursor, load_hf_stream, resolve_data_files, skip_seen
from .normalize import normalize_row, render_training_text
from .pack import PackedDatasetWriter, validate_shards
from .progress import (
    ProgressState,
    file_size,
    load_progress,
    save_progress,
    truncate_file,
    truncate_jsonl_files,
)
from .report import log_progress, write_dataset_report, write_metadata
from .sft import SftJsonlShardWriter
from .tokenize import copy_tokenizer_artifacts, load_tokenizer
from .upload_s3 import download_files, download_prefix, get_json, put_json, upload_directory, upload_files


def run_pipeline(
    config: PipelineConfig,
    *,
    resume: bool = True,
    skip_upload: bool = False,
    max_samples_per_source: int | None = None,
    streams: dict[str, Iterable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if config.output_mode == "sft_jsonl":
        return run_sft_jsonl_pipeline(
            config,
            resume=resume,
            skip_upload=skip_upload,
            max_samples_per_source=max_samples_per_source,
            streams=streams,
        )
    if config.output_mode != "pretrain":
        raise ValueError(f"Unsupported output_mode: {config.output_mode}")
    return run_pretrain_pipeline(
        config,
        resume=resume,
        skip_upload=skip_upload,
        max_samples_per_source=max_samples_per_source,
        streams=streams,
    )


def run_pretrain_pipeline(
    config: PipelineConfig,
    *,
    resume: bool = True,
    skip_upload: bool = False,
    max_samples_per_source: int | None = None,
    streams: dict[str, Iterable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    ensure_dirs(config)
    if resume and not skip_upload and config.s3_output:
        restore_checkpoint_from_s3(config)
    cfg_hash = config_hash(config)
    if not resume:
        cleanup_fresh_run_files(config)
    state = load_progress(
        config.progress_path,
        cfg_hash,
        allow_hash_upgrade=resume_checkpoint_files_exist(config),
    ) if resume else None
    if state is None:
        state = ProgressState(config_hash=cfg_hash)
    elif resume:
        restore_outputs_to_checkpoint(config, state)

    tokenizer = load_tokenizer(config.tokenizer_path, config.fallback_tokenizer)
    source_reports: list[dict[str, Any]] = []
    dedup_path = Path(config.work_dir) / "dedup" / "exact_hashes.sqlite"
    next_checkpoint = state.actual_tokens + config.checkpoint_interval_tokens

    with ExactDeduper(dedup_path) as deduper, PackedDatasetWriter(
        config.output_dir,
        sequence_length=config.sample_length,
        dtype=tokenizer.dtype,
        val_ratio=config.val_ratio,
        seed=config.seed,
        shard_sequences=config.shard_sequences,
        append=resume and state.actual_tokens > 0,
        initial_stats=state.pack_stats,
    ) as writer:
        for source in config.sources:
            if state.actual_tokens >= config.target_tokens:
                break
            if not source_included(config, source):
                continue

            source_token_limit = min(source.target_tokens, config.target_tokens - state.actual_tokens)
            source_report = process_source(
                config,
                source,
                state,
                deduper,
                writer,
                tokenizer,
                token_limit=source_token_limit,
                max_samples=max_samples_per_source,
                stream=(streams or {}).get(source.name),
                upload_enabled=not skip_upload,
            )
            source_reports.append(source_report)

            writer.flush_handles()
            update_output_file_state(config, state)
            save_progress(config.progress_path, state)
            deduper.checkpoint()
            log_source_progress(config, source.name, state)

            if state.actual_tokens >= next_checkpoint:
                save_progress(config.progress_path, state)
                deduper.checkpoint()
                next_checkpoint = state.actual_tokens + config.checkpoint_interval_tokens

        writer.flush_remainder()
        state.pack_stats = writer.stats
        writer.flush_handles()
        update_output_file_state(config, state)
        state.completed = state.actual_tokens >= config.target_tokens
        save_progress(config.progress_path, state)
        deduper.checkpoint()

    copy_tokenizer_artifacts(config.tokenizer_path, config.output_dir, tokenizer)
    total_count, outputs_ok = validate_shards(config.output_dir, state.pack_stats, tokenizer.dtype, config.sample_length)
    if not outputs_ok:
        raise RuntimeError(
            f"Packed output validation failed: file_tokens={total_count}, expected_tokens={state.pack_stats.total_tokens}, "
            f"tokens_per_sample={config.sample_length}"
        )

    sources_report = build_sources_report(config, state, source_reports)
    metadata = write_metadata(
        config.output_dir,
        vocab_size=tokenizer.vocab_size,
        dtype=tokenizer.dtype,
        sequence_length=config.sequence_length,
        stats=state.pack_stats,
        tokenizer_source=tokenizer.source,
        sources=sources_report,
        output_mode=config.output_mode,
        tokens_per_sample=config.sample_length,
        loss_shift=config.loss_shift,
    )
    output_files = [str(Path(config.output_dir) / shard.path) for shard in state.pack_stats.shards if shard.tokens > 0]
    output_files.extend(str(Path(config.output_dir) / name) for name in ["manifest.json", "metadata.json"])
    report = write_dataset_report(
        config.report_path,
        config,
        stats=state.pack_stats,
        sources=sources_report,
        filtered_count=sum(state.rejected_counts.values()),
        deduplicated_count=state.deduplicated_count,
        output_files=output_files,
    )

    uploaded: list[str] = []
    if not skip_upload and config.s3_output:
        upload_checkpoint(config)
        uploaded = upload_packed_outputs(config, state)

    log_progress(
        "finished "
        f"tokens={state.actual_tokens} train={state.pack_stats.train_tokens} val={state.pack_stats.val_tokens} "
        f"output={config.output_dir}"
    )
    return {"metadata": metadata, "report": report, "uploaded": uploaded}


def process_source(
    config: PipelineConfig,
    source: SourceConfig,
    state: ProgressState,
    deduper: ExactDeduper,
    writer: PackedDatasetWriter,
    tokenizer,
    *,
    token_limit: int,
    max_samples: int | None,
    stream: Iterable[dict[str, Any]] | None,
    upload_enabled: bool,
) -> dict[str, Any]:
    processed_before = state.processed_samples.get(source.name, 0)
    cursor_rows: Iterable[tuple[int | None, int | None, dict[str, Any]]]
    if stream is not None:
        rows = stream
        if processed_before:
            rows = skip_seen(rows, processed_before)
        cursor_rows = ((None, None, row) for row in rows)
    else:
        data_files = resolve_data_files(source)
        if data_files:
            cursor_rows = iter_data_files_from_cursor(
                source,
                data_files,
                start_file_index=state.source_file_indices.get(source.name, 0),
                start_row_offset=state.source_row_offsets.get(source.name, 0),
            )
        else:
            rows = load_hf_stream(source)
            if processed_before:
                rows = skip_seen(rows, processed_before)
            cursor_rows = ((None, None, row) for row in rows)

    normalized_path = config.normalized_dir / f"{source.name}.jsonl"
    filtered_path = config.filtered_dir / f"{source.name}.jsonl"
    accepted_tokens = state.source_tokens.get(source.name, 0)
    sample_count = 0

    log_progress(f"source={source.name} start processed_offset={processed_before} target_tokens={token_limit}")
    for file_index, next_row_offset, row in cursor_rows:
        if max_samples is not None and sample_count >= max_samples:
            break
        if accepted_tokens >= token_limit or state.actual_tokens >= config.target_tokens:
            break

        sample_count += 1
        state.processed_samples[source.name] = state.processed_samples.get(source.name, 0) + 1
        if file_index is not None and next_row_offset is not None:
            state.source_file_indices[source.name] = file_index
            state.source_row_offsets[source.name] = next_row_offset
        record = normalize_row(source, row)
        if record is None:
            increment(state.rejected_counts, "normalize_failed")
            continue

        if config.write_intermediate_jsonl:
            append_jsonl(normalized_path, record)
            state.normalized_counts[source.name] = state.normalized_counts.get(source.name, 0) + 1
            state.normalized_bytes[str(normalized_path)] = file_size(normalized_path)

        filter_result = should_keep(record)
        if not filter_result.keep:
            increment(state.rejected_counts, filter_result.reason or "filtered")
            continue

        if deduper.is_duplicate(record):
            state.deduplicated_count += 1
            continue

        if config.write_intermediate_jsonl:
            append_jsonl(filtered_path, record)
            state.filtered_counts[source.name] = state.filtered_counts.get(source.name, 0) + 1
            state.filtered_bytes[str(filtered_path)] = file_size(filtered_path)

        text = render_training_text(record)
        tokens = tokenizer.encode_with_eos(text)
        if not tokens:
            increment(state.rejected_counts, "empty_tokens")
            continue

        writer.add_tokens(tokens)
        state.pack_stats = writer.stats
        accepted_tokens += len(tokens)
        state.source_tokens[source.name] = accepted_tokens

        if state.actual_tokens >= config.target_tokens:
            break

        if state.actual_tokens and state.actual_tokens % config.checkpoint_interval_tokens < len(tokens):
            writer.flush_handles()
            update_output_file_state(config, state)
            save_progress(config.progress_path, state)
            deduper.checkpoint()
            log_source_progress(config, source.name, state)

        if (
            upload_enabled
            and config.s3_upload_interval_tokens
            and state.actual_tokens
            and state.actual_tokens % config.s3_upload_interval_tokens < len(tokens)
        ):
            writer.flush_handles()
            update_output_file_state(config, state)
            save_progress(config.progress_path, state)
            deduper.checkpoint()
            write_partial_artifacts(config, state)
            upload_checkpoint(config)

    return {
        "name": source.name,
        "type": source.type,
        "processed_samples": state.processed_samples.get(source.name, 0),
        "accepted_tokens": state.source_tokens.get(source.name, 0),
        "normalized_count": state.normalized_counts.get(source.name, 0),
        "filtered_count": state.filtered_counts.get(source.name, 0),
    }


def run_sft_jsonl_pipeline(
    config: PipelineConfig,
    *,
    resume: bool = True,
    skip_upload: bool = False,
    max_samples_per_source: int | None = None,
    streams: dict[str, Iterable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    ensure_dirs(config)
    if resume and not skip_upload and config.s3_output:
        restore_checkpoint_from_s3(config)
    cfg_hash = config_hash(config)
    if not resume:
        cleanup_fresh_run_files(config)
    state = load_progress(
        config.progress_path,
        cfg_hash,
        allow_hash_upgrade=resume_checkpoint_files_exist(config),
    ) if resume else None
    if state is None:
        state = ProgressState(config_hash=cfg_hash)
    elif resume:
        restore_outputs_to_checkpoint(config, state)

    tokenizer = load_tokenizer(config.tokenizer_path, config.fallback_tokenizer)
    dedup_path = Path(config.work_dir) / "dedup" / "exact_hashes.sqlite"
    source_reports: list[dict[str, Any]] = []

    with ExactDeduper(dedup_path) as deduper, SftJsonlShardWriter(
        config.output_dir,
        samples_per_shard=config.sft_samples_per_shard,
        val_ratio=config.val_ratio,
        seed=config.seed,
        append=resume and state.sft_stats.total_samples > 0,
        initial_stats=state.sft_stats,
    ) as writer:
        for source in config.sources:
            if state.sft_stats.total_tokens >= config.target_tokens:
                break
            if not source_included(config, source):
                continue
            source_token_limit = min(source.target_tokens, config.target_tokens - state.sft_stats.total_tokens)
            source_report = process_sft_source(
                config,
                source,
                state,
                deduper,
                writer,
                tokenizer,
                token_limit=source_token_limit,
                max_samples=max_samples_per_source,
                stream=(streams or {}).get(source.name),
                upload_enabled=not skip_upload,
            )
            source_reports.append(source_report)
            writer.flush_handles()
            update_sft_file_state(config, state)
            save_progress(config.progress_path, state)
            deduper.checkpoint()
            log_progress(
                f"source={source.name} sft_tokens={state.sft_stats.total_tokens}/{config.target_tokens} "
                f"samples={state.sft_stats.total_samples} output={config.output_dir}"
            )

        writer.flush_handles()
        update_sft_file_state(config, state)
        state.completed = state.sft_stats.total_tokens >= config.target_tokens
        save_progress(config.progress_path, state)
        deduper.checkpoint()

    sources_report = build_sources_report(config, state, source_reports)
    metadata = write_sft_metadata(config, state, sources_report)
    report = write_sft_report(config, state, sources_report)
    uploaded: list[str] = []
    if not skip_upload and config.s3_output:
        upload_checkpoint(config)
        uploaded = upload_packed_outputs(config, state)

    log_progress(
        "finished "
        f"sft_tokens={state.sft_stats.total_tokens} samples={state.sft_stats.total_samples} "
        f"output={config.output_dir}"
    )
    return {"metadata": metadata, "report": report, "uploaded": uploaded}


def process_sft_source(
    config: PipelineConfig,
    source: SourceConfig,
    state: ProgressState,
    deduper: ExactDeduper,
    writer: SftJsonlShardWriter,
    tokenizer,
    *,
    token_limit: int,
    max_samples: int | None,
    stream: Iterable[dict[str, Any]] | None,
    upload_enabled: bool,
) -> dict[str, Any]:
    cursor_rows = iter_source_rows(config, source, state, stream)
    accepted_tokens = state.source_tokens.get(source.name, 0)
    sample_count = 0
    log_progress(
        f"source={source.name} start processed_offset={state.processed_samples.get(source.name, 0)} "
        f"target_tokens={token_limit}"
    )
    for file_index, next_row_offset, row in cursor_rows:
        if max_samples is not None and sample_count >= max_samples:
            break
        if accepted_tokens >= token_limit or state.sft_stats.total_tokens >= config.target_tokens:
            break

        sample_count += 1
        state.processed_samples[source.name] = state.processed_samples.get(source.name, 0) + 1
        if file_index is not None and next_row_offset is not None:
            state.source_file_indices[source.name] = file_index
            state.source_row_offsets[source.name] = next_row_offset

        record = normalize_row(source, row)
        if record is None or record.get("type") != "instruct":
            increment(state.rejected_counts, "normalize_failed")
            continue
        filter_result = should_keep(record)
        if not filter_result.keep:
            increment(state.rejected_counts, filter_result.reason or "filtered")
            continue
        if deduper.is_duplicate(record):
            state.deduplicated_count += 1
            continue

        token_count = len(tokenizer.encode_with_eos(render_training_text(record)))
        if token_count <= 1:
            increment(state.rejected_counts, "empty_tokens")
            continue

        writer.add_record(sft_public_record(record), token_count)
        state.sft_stats = writer.stats
        accepted_tokens += token_count
        state.source_tokens[source.name] = accepted_tokens

        if (
            state.sft_stats.total_tokens
            and state.sft_stats.total_tokens % config.checkpoint_interval_tokens < token_count
        ):
            writer.flush_handles()
            update_sft_file_state(config, state)
            save_progress(config.progress_path, state)
            deduper.checkpoint()

        if (
            upload_enabled
            and config.s3_upload_interval_tokens
            and state.sft_stats.total_tokens
            and state.sft_stats.total_tokens % config.s3_upload_interval_tokens < token_count
        ):
            writer.flush_handles()
            update_sft_file_state(config, state)
            save_progress(config.progress_path, state)
            deduper.checkpoint()
            write_partial_artifacts(config, state)
            upload_checkpoint(config)

    return {
        "name": source.name,
        "type": source.type,
        "mix_role": source.mix_role,
        "processed_samples": state.processed_samples.get(source.name, 0),
        "accepted_tokens": state.source_tokens.get(source.name, 0),
        "normalized_count": state.normalized_counts.get(source.name, 0),
        "filtered_count": state.filtered_counts.get(source.name, 0),
    }


def iter_source_rows(
    config: PipelineConfig,
    source: SourceConfig,
    state: ProgressState,
    stream: Iterable[dict[str, Any]] | None,
) -> Iterable[tuple[int | None, int | None, dict[str, Any]]]:
    processed_before = state.processed_samples.get(source.name, 0)
    if stream is not None:
        rows = stream
        if processed_before:
            rows = skip_seen(rows, processed_before)
        return ((None, None, row) for row in rows)
    data_files = resolve_data_files(source)
    if data_files:
        return iter_data_files_from_cursor(
            source,
            data_files,
            start_file_index=state.source_file_indices.get(source.name, 0),
            start_row_offset=state.source_row_offsets.get(source.name, 0),
        )
    rows = load_hf_stream(source)
    if processed_before:
        rows = skip_seen(rows, processed_before)
    return ((None, None, row) for row in rows)


def source_included(config: PipelineConfig, source: SourceConfig) -> bool:
    role = source.mix_role
    if config.output_mode == "pretrain":
        return role in {None, "base", "mixed_instruct"}
    if config.output_mode == "sft_jsonl":
        return role in {"instruct_only", None} and source.type == "instruct"
    return False


def sft_public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": record["messages"],
        "source": record.get("source"),
    }


def write_sft_metadata(
    config: PipelineConfig,
    state: ProgressState,
    sources_report: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = {
        "project": config.project_name,
        "output_mode": config.output_mode,
        "artifact_format": config.artifact_format or "jsonl_messages",
        "target_tokens": config.target_tokens,
        "actual_tokens": state.sft_stats.total_tokens,
        "train_tokens": state.sft_stats.train_tokens,
        "val_tokens": state.sft_stats.val_tokens,
        "train_samples": state.sft_stats.train_samples,
        "val_samples": state.sft_stats.val_samples,
        "total_samples": state.sft_stats.total_samples,
        "sft_samples_per_shard": config.sft_samples_per_shard,
        "sources": sources_report,
        "shards": [
            dataclasses.asdict(shard)
            for shard in state.sft_stats.shards
            if shard.samples > 0
        ],
        "output_files": [
            *[shard.path for shard in state.sft_stats.shards if shard.samples > 0],
            "manifest.json",
            "metadata.json",
        ],
    }
    path = Path(config.output_dir) / "metadata.json"
    path.write_text(dataclasses_json(metadata))
    return metadata


def write_sft_report(
    config: PipelineConfig,
    state: ProgressState,
    sources_report: list[dict[str, Any]],
) -> dict[str, Any]:
    output_files = [
        *[str(Path(config.output_dir) / shard.path) for shard in state.sft_stats.shards if shard.samples > 0],
        str(Path(config.output_dir) / "manifest.json"),
        str(Path(config.output_dir) / "metadata.json"),
    ]
    report = {
        "project": config.project_name,
        "output_mode": config.output_mode,
        "target_tokens": config.target_tokens,
        "actual_tokens": state.sft_stats.total_tokens,
        "train_tokens": state.sft_stats.train_tokens,
        "val_tokens": state.sft_stats.val_tokens,
        "train_samples": state.sft_stats.train_samples,
        "val_samples": state.sft_stats.val_samples,
        "sources": sources_report,
        "filtered_count": sum(state.rejected_counts.values()),
        "deduplicated_count": state.deduplicated_count,
        "output_files": output_files,
    }
    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(dataclasses_json(report))
    return report


def restore_outputs_to_checkpoint(config: PipelineConfig, state: ProgressState) -> None:
    for relative_path, size in state.shard_bytes.items():
        truncate_file(Path(config.output_dir) / relative_path, size)
    truncate_jsonl_files(state.normalized_bytes)
    truncate_jsonl_files(state.filtered_bytes)


def cleanup_fresh_run_files(config: PipelineConfig) -> None:
    paths = [
        Path(config.progress_path),
        Path(config.output_dir) / "train.bin",
        Path(config.output_dir) / "val.bin",
        Path(config.output_dir) / "manifest.json",
        Path(config.output_dir) / "metadata.json",
        Path(config.output_dir) / "metadata.partial.json",
        Path(config.work_dir) / "dedup" / "exact_hashes.sqlite",
        Path(config.work_dir) / "dedup" / "exact_hashes.sqlite-wal",
        Path(config.work_dir) / "dedup" / "exact_hashes.sqlite-shm",
    ]
    for source in config.sources:
        paths.append(config.normalized_dir / f"{source.name}.jsonl")
        paths.append(config.filtered_dir / f"{source.name}.jsonl")
    paths.extend(Path(config.output_dir).glob("train-*.bin"))
    paths.extend(Path(config.output_dir).glob("val-*.bin"))
    paths.extend(Path(config.output_dir).glob("train-*.jsonl"))
    paths.extend(Path(config.output_dir).glob("val-*.jsonl"))
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()


def resume_checkpoint_files_exist(config: PipelineConfig) -> bool:
    return any(Path(config.output_dir).glob("train-*.bin"))


def build_sources_report(
    config: PipelineConfig,
    state: ProgressState,
    latest_source_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_by_name = {row["name"]: row for row in latest_source_reports}
    reports = []
    for source in config.sources:
        row = latest_by_name.get(source.name, {})
        reports.append(
            {
                "name": source.name,
                "type": source.type,
                "mix_role": source.mix_role,
                "target_tokens": source.target_tokens,
                "processed_samples": state.processed_samples.get(source.name, row.get("processed_samples", 0)),
                "accepted_tokens": state.source_tokens.get(source.name, row.get("accepted_tokens", 0)),
                "normalized_count": state.normalized_counts.get(source.name, row.get("normalized_count", 0)),
                "filtered_count": state.filtered_counts.get(source.name, row.get("filtered_count", 0)),
            }
        )
    return reports


def increment(mapping: dict[str, int], key: str) -> None:
    mapping[key] = mapping.get(key, 0) + 1


def log_source_progress(config: PipelineConfig, source_name: str, state: ProgressState) -> None:
    train_mb = sum(size for path, size in state.shard_bytes.items() if path.startswith("train-")) / 1024 / 1024
    val_mb = sum(size for path, size in state.shard_bytes.items() if path.startswith("val-")) / 1024 / 1024
    log_progress(
        f"source={source_name} tokens={state.actual_tokens}/{config.target_tokens} "
        f"train={state.pack_stats.train_tokens} val={state.pack_stats.val_tokens} "
        f"size_mb=train:{train_mb:.1f},val:{val_mb:.1f} output={config.output_dir}"
    )


def upload_checkpoint(config: PipelineConfig) -> None:
    if not config.s3_output:
        return
    uploaded = upload_packed_outputs(config, None)
    checkpoint_id = str(int(time.time() * 1000))
    checkpoint_prefix = f"checkpoint/staging/{checkpoint_id}/"
    uploaded.extend(upload_files(checkpoint_artifacts(config, checkpoint_prefix), config.s3_output))
    ready_uri = put_json(
        "checkpoint/READY.json",
        {
            "checkpoint_id": checkpoint_id,
            "checkpoint_prefix": checkpoint_prefix,
            "created_unix_ms": int(time.time() * 1000),
            "output_mode": config.output_mode,
            "project": config.project_name,
        },
        config.s3_output,
    )
    uploaded.append(ready_uri)
    log_progress(f"s3_upload files={len(uploaded)} checkpoint={checkpoint_id} destination={config.s3_output}")


def write_partial_artifacts(config: PipelineConfig, state: ProgressState) -> None:
    if config.output_mode == "sft_jsonl":
        partial_metadata = {
            "project": config.project_name,
            "output_mode": config.output_mode,
            "target_tokens": config.target_tokens,
            "actual_tokens": state.sft_stats.total_tokens,
            "train_tokens": state.sft_stats.train_tokens,
            "val_tokens": state.sft_stats.val_tokens,
            "train_samples": state.sft_stats.train_samples,
            "val_samples": state.sft_stats.val_samples,
            "completed": state.completed,
            "shards": [
                dataclasses.asdict(shard)
                for shard in state.sft_stats.shards
                if shard.samples > 0
            ],
        }
        partial_path = Path(config.output_dir) / "metadata.partial.json"
        partial_path.write_text(dataclasses_json(partial_metadata))
        write_sft_report(config, state, build_sources_report(config, state, []))
        return

    partial_metadata = {
        "project": config.project_name,
        "target_tokens": config.target_tokens,
        "actual_tokens": state.actual_tokens,
        "sequence_length": config.sequence_length,
        "train_tokens": state.pack_stats.train_tokens,
        "val_tokens": state.pack_stats.val_tokens,
        "train_sequences": state.pack_stats.train_sequences,
        "val_sequences": state.pack_stats.val_sequences,
        "shards": [dataclasses.asdict(shard) for shard in state.pack_stats.shards],
        "completed": state.completed,
    }
    partial_path = Path(config.output_dir) / "metadata.partial.json"
    partial_path.write_text(dataclasses_json(partial_metadata))
    write_dataset_report(
        config.report_path,
        config,
        stats=state.pack_stats,
        sources=build_sources_report(config, state, []),
        filtered_count=sum(state.rejected_counts.values()),
        deduplicated_count=state.deduplicated_count,
        output_files=[
            *[str(Path(config.output_dir) / shard.path) for shard in state.pack_stats.shards if shard.tokens > 0],
            str(Path(config.output_dir) / "manifest.json"),
            str(Path(config.output_dir) / "metadata.partial.json"),
        ],
    )


def upload_packed_outputs(config: PipelineConfig, state: ProgressState | None) -> list[str]:
    include = {"manifest.json", "metadata.json", "metadata.partial.json", "tokenizer/"}
    if state is not None:
        include.update(shard.path for shard in state.pack_stats.shards if shard.tokens > 0)
        include.update(shard.path for shard in state.sft_stats.shards if shard.samples > 0)
    else:
        include.update(path.name for path in Path(config.output_dir).glob("train-*.bin"))
        include.update(path.name for path in Path(config.output_dir).glob("val-*.bin"))
        include.update(path.name for path in Path(config.output_dir).glob("train-*.jsonl"))
        include.update(path.name for path in Path(config.output_dir).glob("val-*.jsonl"))
    return upload_directory(config.output_dir, config.s3_output or "", include=include)


def checkpoint_artifacts(config: PipelineConfig, prefix: str = "checkpoint/") -> list[tuple[Path, str]]:
    dedup_dir = Path(config.work_dir) / "dedup"
    artifacts = [
        (Path(config.progress_path), f"{prefix}progress.json"),
        (Path(config.report_path), f"{prefix}dataset_report.json"),
        (Path(config.output_dir) / "metadata.partial.json", f"{prefix}metadata.partial.json"),
        (Path(config.output_dir) / "manifest.json", f"{prefix}manifest.json"),
        (dedup_dir / "exact_hashes.sqlite", f"{prefix}dedup/exact_hashes.sqlite"),
        (dedup_dir / "exact_hashes.sqlite-wal", f"{prefix}dedup/exact_hashes.sqlite-wal"),
        (dedup_dir / "exact_hashes.sqlite-shm", f"{prefix}dedup/exact_hashes.sqlite-shm"),
    ]
    return artifacts


def restore_checkpoint_from_s3(config: PipelineConfig) -> None:
    if Path(config.progress_path).exists():
        return
    ready = get_json("checkpoint/READY.json", config.s3_output)
    if not ready or not ready.get("checkpoint_prefix"):
        log_progress(f"s3_restore skipped: no READY marker at {config.s3_output}checkpoint/READY.json")
        return
    checkpoint_prefix = str(ready["checkpoint_prefix"])
    output_downloaded = download_files(
        [
            ("metadata.json", Path(config.output_dir) / "metadata.json"),
            ("metadata.partial.json", Path(config.output_dir) / "metadata.partial.json"),
            ("manifest.json", Path(config.output_dir) / "manifest.json"),
        ],
        config.s3_output,
    )
    checkpoint_dir = Path(config.work_dir) / "_s3_restore"
    downloaded = download_prefix(config.s3_output.rstrip("/") + "/" + checkpoint_prefix, checkpoint_dir)
    if not downloaded:
        return
    mappings = {
        checkpoint_dir / "progress.json": Path(config.progress_path),
        checkpoint_dir / "dataset_report.json": Path(config.report_path),
        checkpoint_dir / "metadata.partial.json": Path(config.output_dir) / "metadata.partial.json",
        checkpoint_dir / "manifest.json": Path(config.output_dir) / "manifest.json",
        checkpoint_dir / "dedup" / "exact_hashes.sqlite": Path(config.work_dir) / "dedup" / "exact_hashes.sqlite",
        checkpoint_dir / "dedup" / "exact_hashes.sqlite-wal": Path(config.work_dir) / "dedup" / "exact_hashes.sqlite-wal",
        checkpoint_dir / "dedup" / "exact_hashes.sqlite-shm": Path(config.work_dir) / "dedup" / "exact_hashes.sqlite-shm",
    }
    for source, destination in mappings.items():
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
    output_downloaded.extend(restore_shards_from_manifest(config))
    log_progress(
        f"s3_restore files={len(downloaded) + len(output_downloaded)} "
        f"checkpoint={ready.get('checkpoint_id')} source={config.s3_output}"
    )


def restore_shards_from_manifest(config: PipelineConfig) -> list[str]:
    import json

    manifest_path = Path(config.output_dir) / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    downloads = [
        (shard["path"], Path(config.output_dir) / shard["path"])
        for shard in manifest.get("shards", [])
        if shard.get("path")
    ]
    return download_files(downloads, config.s3_output)


def update_output_file_state(config: PipelineConfig, state: ProgressState) -> None:
    state.shard_bytes = {
        shard.path: file_size(Path(config.output_dir) / shard.path)
        for shard in state.pack_stats.shards
        if shard.tokens > 0
    }
    state.train_bytes = sum(size for path, size in state.shard_bytes.items() if path.startswith("train-"))
    state.val_bytes = sum(size for path, size in state.shard_bytes.items() if path.startswith("val-"))


def update_sft_file_state(config: PipelineConfig, state: ProgressState) -> None:
    state.shard_bytes = {
        shard.path: file_size(Path(config.output_dir) / shard.path)
        for shard in state.sft_stats.shards
        if shard.samples > 0
    }
    state.train_bytes = sum(size for path, size in state.shard_bytes.items() if path.startswith("train-"))
    state.val_bytes = sum(size for path, size in state.shard_bytes.items() if path.startswith("val-"))


def dataclasses_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)
