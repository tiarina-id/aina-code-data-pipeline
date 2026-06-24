from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from typing import Any


def upload_directory(local_dir: str | Path, s3_uri: str, *, include: set[str] | None = None) -> list[str]:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'boto3'. Install with `python3 -m pip install -e .`.") from exc

    bucket, prefix = parse_s3_uri(s3_uri)
    local_path = Path(local_dir)
    client = boto3.client("s3")
    uploaded: list[str] = []
    for file_path in local_path.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(local_path).as_posix()
        if include is not None and relative not in include and not any(
            relative.startswith(prefix.rstrip("/") + "/") for prefix in include if prefix.endswith("/")
        ):
            continue
        key = prefix + relative
        client.upload_file(str(file_path), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def upload_files(files: list[tuple[str | Path, str]], s3_uri: str) -> list[str]:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'boto3'. Install with `python3 -m pip install -e .`.") from exc

    bucket, prefix = parse_s3_uri(s3_uri)
    client = boto3.client("s3")
    uploaded: list[str] = []
    for local_file, relative_key in files:
        file_path = Path(local_file)
        if not file_path.is_file():
            continue
        key = prefix + relative_key.lstrip("/")
        client.upload_file(str(file_path), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def download_prefix(s3_uri: str, local_dir: str | Path) -> list[str]:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'boto3'. Install with `python3 -m pip install -e .`.") from exc

    bucket, prefix = parse_s3_uri(s3_uri)
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3")
    downloaded: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("/"):
                    continue
                relative = key[len(prefix) :] if key.startswith(prefix) else key
                destination = local_path / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(destination))
                downloaded.append(str(destination))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchBucket", "AccessDenied"}:
            raise
        return downloaded
    return downloaded


def download_files(files: list[tuple[str, str | Path]], s3_uri: str) -> list[str]:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'boto3'. Install with `python3 -m pip install -e .`.") from exc

    bucket, prefix = parse_s3_uri(s3_uri)
    client = boto3.client("s3")
    downloaded: list[str] = []
    for relative_key, local_file in files:
        destination = Path(local_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        key = prefix + relative_key.lstrip("/")
        try:
            client.download_file(bucket, key, str(destination))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                continue
            raise
        downloaded.append(str(destination))
    return downloaded


def put_json(relative_key: str, payload: dict[str, Any], s3_uri: str) -> str:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'boto3'. Install with `python3 -m pip install -e .`.") from exc
    import json

    bucket, prefix = parse_s3_uri(s3_uri)
    key = prefix + relative_key.lstrip("/")
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def get_json(relative_key: str, s3_uri: str) -> dict[str, Any] | None:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'boto3'. Install with `python3 -m pip install -e .`.") from exc
    import json

    bucket, prefix = parse_s3_uri(s3_uri)
    key = prefix + relative_key.lstrip("/")
    try:
        response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return None
        raise
    return json.loads(response["Body"].read().decode("utf-8"))


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix
