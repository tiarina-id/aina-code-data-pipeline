#!/usr/bin/env python3
import _bootstrap  # noqa: F401

import argparse

from aina_preproc.upload_s3 import upload_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a directory recursively to S3.")
    parser.add_argument("local_dir")
    parser.add_argument("s3_uri")
    args = parser.parse_args()
    for uri in upload_directory(args.local_dir, args.s3_uri):
        print(uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
