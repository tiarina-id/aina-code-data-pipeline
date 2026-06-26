from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


AINA_CHAT_TEMPLATE = """{% for message in messages %}<|{{ message['role'] }}|>
{{ message['content'] }}
{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %}"""


class TokenizerLike(Protocol):
    eos_token_id: int | None
    vocab_size: int

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...

    def save_pretrained(self, save_directory: str | Path) -> None:
        ...


@dataclass
class TokenizerBundle:
    tokenizer: TokenizerLike
    source: str

    @property
    def eos_token_id(self) -> int:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer has no eos_token_id; set one before running preprocessing.")
        return int(eos)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    @property
    def dtype(self) -> str:
        return "uint16" if self.vocab_size < 65_536 else "uint32"

    def encode_with_eos(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False)) + [self.eos_token_id]

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output)


def load_tokenizer(tokenizer_path: str, fallback_tokenizer: str | None = None) -> TokenizerBundle:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers'. Install with `python3 -m pip install -e .`."
        ) from exc

    path = Path(tokenizer_path)
    if path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        source = str(path)
    elif fallback_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer, use_fast=True)
        source = fallback_tokenizer
    else:
        raise FileNotFoundError(
            f"Tokenizer path does not exist: {tokenizer_path}. Set fallback_tokenizer in config."
        )

    if tokenizer.eos_token_id is None:
        if tokenizer.sep_token is not None:
            tokenizer.eos_token = tokenizer.sep_token
        elif tokenizer.pad_token is not None:
            tokenizer.eos_token = tokenizer.pad_token
        else:
            raise ValueError("Tokenizer must define eos_token_id, sep_token, or pad_token.")
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = AINA_CHAT_TEMPLATE
    if getattr(tokenizer, "model_max_length", 0) < 8192:
        tokenizer.model_max_length = 8192
    return TokenizerBundle(tokenizer=tokenizer, source=source)


def copy_tokenizer_artifacts(tokenizer_path: str, output_dir: str | Path, bundle: TokenizerBundle) -> None:
    destination = Path(output_dir) / "tokenizer"
    source = Path(tokenizer_path)
    if source.exists() and source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        bundle.save(destination)
