from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .normalize import render_training_text

MIN_CHARS = 50
MAX_CHARS = 200_000

SECRET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"AWS_SECRET_ACCESS_KEY",
        r"PRIVATE KEY",
        r"api_key\s*=",
        r"password\s*=",
        r"BEGIN RSA PRIVATE KEY",
    ]
]

LOCK_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}


@dataclass(frozen=True)
class FilterResult:
    keep: bool
    reason: str | None = None


def should_keep(record: dict[str, Any]) -> FilterResult:
    text = render_training_text(record)
    if not text.strip():
        return FilterResult(False, "empty")
    if len(text) < MIN_CHARS:
        return FilterResult(False, "too_short")
    if len(text) > MAX_CHARS:
        return FilterResult(False, "too_long")

    path = str(record.get("path") or "").split("/")[-1]
    if path in LOCK_FILES:
        return FilterResult(False, "lock_file")
    if is_minified(record, text):
        return FilterResult(False, "minified")
    if has_too_many_weird_chars(text):
        return FilterResult(False, "weird_chars")
    if is_binary_or_base64_like(text):
        return FilterResult(False, "binary_or_base64")
    if contains_secret(text):
        return FilterResult(False, "secret")
    return FilterResult(True)


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def has_too_many_weird_chars(text: str) -> bool:
    if not text:
        return True
    weird = 0
    for char in text:
        if char in "\n\r\t":
            continue
        category = ord(char)
        if category < 32 or category == 127:
            weird += 1
        elif category > 0xFFFF:
            weird += 1
    return weird / len(text) > 0.02


def is_minified(record: dict[str, Any], text: str) -> bool:
    language = (record.get("language") or "").lower()
    path = str(record.get("path") or "").lower()
    if language not in {"javascript", "typescript", "css"} and not path.endswith((".js", ".css")):
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    avg_line_len = sum(len(line) for line in lines) / len(lines)
    long_line_ratio = sum(1 for line in lines if len(line) > 500) / len(lines)
    return avg_line_len > 300 or long_line_ratio > 0.2


def is_binary_or_base64_like(text: str) -> bool:
    sample = text[:8192]
    if "\x00" in sample:
        return True
    compact = re.sub(r"\s+", "", sample)
    if len(compact) < 256:
        return False
    base64_chars = sum(1 for char in compact if char.isalnum() or char in "+/=")
    entropy = shannon_entropy(compact)
    return base64_chars / len(compact) > 0.97 and entropy > 4.5


def shannon_entropy(text: str) -> float:
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

