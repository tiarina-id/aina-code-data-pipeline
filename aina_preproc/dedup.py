from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from .normalize import render_training_text


class ExactDeduper:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY)")
        self.conn.commit()

    def is_duplicate(self, record: dict) -> bool:
        digest = canonical_hash(record)
        try:
            self.conn.execute("INSERT INTO seen(hash) VALUES (?)", (digest,))
            return False
        except sqlite3.IntegrityError:
            return True

    def checkpoint(self) -> None:
        self.conn.commit()

    def close(self, *, commit: bool = True) -> None:
        if commit:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()

    def __enter__(self) -> "ExactDeduper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(commit=exc_type is None)


def canonical_hash(record: dict) -> str:
    text = render_training_text(record)
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
