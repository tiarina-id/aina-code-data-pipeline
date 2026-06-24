#!/usr/bin/env python3
import _bootstrap  # noqa: F401

import argparse
import json
import sys

from aina_preproc.pack import PackedDatasetWriter


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack token JSON arrays from stdin into sharded train/val bin files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--dtype", choices=["uint16", "uint32"], required=True)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-sequences", type=int, default=100_000)
    args = parser.parse_args()
    with PackedDatasetWriter(
        args.output_dir,
        sequence_length=args.sequence_length,
        dtype=args.dtype,
        val_ratio=args.val_ratio,
        seed=args.seed,
        shard_sequences=args.shard_sequences,
    ) as writer:
        for line in sys.stdin:
            writer.add_tokens(json.loads(line))
        writer.flush_remainder()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
