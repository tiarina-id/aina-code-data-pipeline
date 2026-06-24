#!/usr/bin/env python3
import _bootstrap  # noqa: F401

import argparse
import json
import sys

from aina_preproc.dedup import ExactDeduper


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact-deduplicate normalized JSONL from stdin to stdout.")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    with ExactDeduper(args.db) as deduper:
        for line in sys.stdin:
            row = json.loads(line)
            if not deduper.is_duplicate(row):
                print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
