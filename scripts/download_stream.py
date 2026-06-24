#!/usr/bin/env python3
import _bootstrap  # noqa: F401

import argparse

from aina_preproc.config import load_config
from aina_preproc.loaders import load_hf_stream


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test streaming access for one configured source.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    source = next(item for item in config.sources if item.name == args.source)
    for index, row in enumerate(load_hf_stream(source)):
        print(row)
        if index + 1 >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
