from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Aina code training dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--target-tokens", type=int, default=None, help="Override target token count.")
    parser.add_argument("--output-dir", default=None, help="Override packed output directory.")
    parser.add_argument("--work-dir", default=None, help="Override preprocessing work directory.")
    parser.add_argument("--progress-path", default=None, help="Override progress checkpoint path.")
    parser.add_argument("--report-path", default=None, help="Override dataset report path.")
    parser.add_argument("--sequence-length", type=int, default=None, help="Override packed sequence length.")
    parser.add_argument("--num-workers", type=int, default=None, help="Parallel workers for pretrain transform/tokenize.")
    parser.add_argument("--worker-batch-size", type=int, default=None, help="Rows per worker task for pretrain.")
    parser.add_argument("--log-interval-seconds", type=int, default=None, help="Seconds between progress log lines.")
    parser.add_argument("--skip-upload", action="store_true", help="Do not upload final output to S3.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from progress file.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start a fresh run.")
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=None,
        help="Limit samples per source for smoke tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config).with_overrides(
        target_tokens=args.target_tokens,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        work_dir=args.work_dir,
        progress_path=args.progress_path,
        report_path=args.report_path,
        num_workers=args.num_workers,
        worker_batch_size=args.worker_batch_size,
        log_interval_seconds=args.log_interval_seconds,
    )
    run_pipeline(
        config,
        resume=args.resume,
        skip_upload=args.skip_upload,
        max_samples_per_source=args.max_samples_per_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
