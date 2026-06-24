#!/usr/bin/env python3
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print dataset report JSON.")
    parser.add_argument("--report", default="reports/dataset_report.json")
    args = parser.parse_args()
    print(json.dumps(json.loads(Path(args.report).read_text()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
