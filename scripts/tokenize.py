#!/usr/bin/env python3
import _bootstrap  # noqa: F401

import argparse
import json
import sys

from aina_preproc.normalize import render_training_text
from aina_preproc.tokenize import load_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize normalized JSONL from stdin to stdout.")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--fallback-tokenizer", default=None)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path, args.fallback_tokenizer)
    for line in sys.stdin:
        row = json.loads(line)
        print(json.dumps(tokenizer.encode_with_eos(render_training_text(row))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
