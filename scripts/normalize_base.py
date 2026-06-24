#!/usr/bin/env python3
import _bootstrap  # noqa: F401

import argparse
import json
import sys

from aina_preproc.config import SourceConfig
from aina_preproc.normalize import normalize_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize base JSONL from stdin to stdout.")
    parser.add_argument("--source", default="custom_base")
    parser.add_argument("--language", default=None)
    args = parser.parse_args()
    source = SourceConfig(name=args.source, type="base", hf_id="stdin", target_tokens=0, language=args.language)
    for line in sys.stdin:
        row = normalize_base(source, json.loads(line))
        if row is not None:
            print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
