from __future__ import annotations

import json
from typing import Any

from .config import SourceConfig

SYSTEM_PROMPT = "You are Aina, a helpful coding assistant."


def normalize_row(source: SourceConfig, row: dict[str, Any]) -> dict[str, Any] | None:
    if source.type == "base":
        return normalize_base(source, row)
    if source.type == "instruct":
        return normalize_instruct(source, row)
    raise ValueError(f"Unsupported source type for {source.name}: {source.type}")


def normalize_base(source: SourceConfig, row: dict[str, Any]) -> dict[str, Any] | None:
    text = first_text(row, ["content", "code", "func_code_string", "whole_func_string", "text"])
    if text is None:
        return None
    language = source.language or row.get("language") or infer_language(source.name, row)
    return {
        "type": "base",
        "text": text,
        "source": source.name,
        "language": language,
        "path": row.get("path") or row.get("file_name") or row.get("repo_name"),
    }


def normalize_instruct(source: SourceConfig, row: dict[str, Any]) -> dict[str, Any] | None:
    existing_messages = normalize_existing_messages(row)
    if existing_messages:
        return {"type": "instruct", "messages": existing_messages, "source": source.name}

    if source.name.lower() == "mbpp":
        prompt = first_text(row, ["text", "prompt", "task"])
        answer = first_text(row, ["code", "canonical_solution", "test_list"])
        if isinstance(row.get("test_list"), list):
            tests = "\n".join(str(item) for item in row["test_list"])
            answer = f"{answer or ''}\n\nTests:\n{tests}".strip()
        return messages_record(source.name, prompt, answer)

    if source.name.lower() == "taco":
        prompt = first_text(row, ["question", "prompt", "statement"])
        answer = first_solution(row)
        return messages_record(source.name, prompt, answer)

    prompt = first_text(
        row,
        [
            "instruction",
            "prompt",
            "question",
            "input",
            "problem",
            "query",
            "user",
            "task",
        ],
    )
    answer = first_text(
        row,
        [
            "output",
            "response",
            "answer",
            "completion",
            "solution",
            "assistant",
            "accepted_solution",
        ],
    )
    return messages_record(source.name, prompt, answer)


def messages_record(source_name: str, prompt: str | None, answer: str | None) -> dict[str, Any] | None:
    if not prompt or not answer:
        return None
    return {
        "type": "instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "source": source_name,
    }


def first_text(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            joined = "\n".join(str(item) for item in value if item is not None)
            if joined.strip():
                return joined
    return None


def first_solution(row: dict[str, Any]) -> str | None:
    for key in ["solutions", "solution", "answer", "output"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
            return value
        if isinstance(value, list) and value:
            return str(value[0])
    return None


def normalize_existing_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    value = row.get("messages") or row.get("conversations")
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from")
        content = item.get("content") or item.get("value")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            continue
        role = normalize_role(role)
        if role:
            messages.append({"role": role, "content": content})
    roles = {message["role"] for message in messages}
    if "user" not in roles or "assistant" not in roles:
        return None
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def normalize_role(role: str) -> str | None:
    normalized = role.strip().lower()
    if normalized in {"system"}:
        return "system"
    if normalized in {"human", "user", "instruction", "prompt"}:
        return "user"
    if normalized in {"gpt", "assistant", "model", "response", "output"}:
        return "assistant"
    return None


def infer_language(source_name: str, row: dict[str, Any]) -> str | None:
    path = str(row.get("path") or row.get("file_name") or "").lower()
    if source_name.endswith("_python") or path.endswith(".py"):
        return "python"
    if source_name.endswith("_javascript") or path.endswith((".js", ".jsx")):
        return "javascript"
    if source_name.endswith("_typescript") or path.endswith((".ts", ".tsx")):
        return "typescript"
    return None


def render_training_text(record: dict[str, Any]) -> str:
    if record["type"] == "base":
        return record["text"]
    parts = []
    for message in record["messages"]:
        parts.append(f"<|{message['role']}|>\n{message['content']}")
    return "\n".join(parts)
