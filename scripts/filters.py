#!/usr/bin/env python3
import _bootstrap  # noqa: F401

import json
import sys

from aina_preproc.filters import should_keep


def main() -> int:
    for line in sys.stdin:
        row = json.loads(line)
        result = should_keep(row)
        if result.keep:
            print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
