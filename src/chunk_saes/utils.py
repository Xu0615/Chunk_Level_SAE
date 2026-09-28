from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PILE_SOURCES = (
    "Pile-CC",
    "PubMed Central",
    "ArXiv",
    "Github",
    "FreeLaw",
    "StackExchange",
    "USPTO Backgrounds",
    "PubMed Abstracts",
    "Gutenberg (PG-19)",
    "Wikipedia (en)",
    "DM Mathematics",
    "Ubuntu IRC",
    "EuroParl",
    "HackerNews",
    "PhilPapers",
    "NIH ExPorter",
    "Enron Emails",
)

_SPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_RE = re.compile(r"\n{4,}")


def stable_u64(value: str, seed: int = 0) -> int:
    payload = f"{seed}\0{value}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def content_hash(text: str) -> str:
    normalized = _SPACE_RE.sub(" ", text.strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def clean_document(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_SPACE_RE.sub(" ", line).strip() for line in text.splitlines())
    return _BLANK_RE.sub("\n\n\n", text).strip()


def looks_like_boilerplate(text: str) -> bool:
    if not text:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    unique_ratio = len(set(lines)) / len(lines)
    longest = max(map(len, lines))
    alpha = sum(ch.isalpha() for ch in text)
    return unique_ratio < 0.2 or longest > 200_000 or alpha < min(32, len(text) // 20)


def document_id(record: dict[str, Any], ordinal: int, text: str) -> str:
    meta = record.get("meta") or {}
    for key in ("id", "document_id", "doc_id", "url", "sha1", "file"):
        value = meta.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    # The source shard has no explicit document IDs. A raw-content digest is
    # fast, stable, and assigns byte-identical duplicates to the same split.
    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()
    return f"content:{digest}"


def pile_source(record: dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    return str(meta.get("pile_set_name") or meta.get("source") or "unknown")


def document_split(doc_id: str, seed: int) -> str:
    bucket = stable_u64(doc_id, seed) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def parse_csv(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def parse_int_csv(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(part) for part in value]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def distributed_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def tokenizer_fingerprint(model_path: str | Path) -> str:
    root = Path(model_path)
    digest = hashlib.sha256()
    found = False
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        path = root / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("ascii"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    if not found:
        raise FileNotFoundError(f"No tokenizer files found under {root}")
    return digest.hexdigest()


def model_fingerprint(model_path: str | Path) -> str:
    """Hash model configuration, index, and every local safetensors payload."""

    root = Path(model_path)
    paths = [
        path
        for path in (
            root / "config.json",
            root / "model.safetensors.index.json",
            *sorted(root.glob("*.safetensors")),
        )
        if path.is_file()
    ]
    if not paths or not any(path.suffix == ".safetensors" for path in paths):
        raise FileNotFoundError(f"No local model safetensors found under {root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def log(message: str, *, rank: int = 0, main_only: bool = False) -> None:
    if not main_only or rank == 0:
        print(f"[rank {rank}] {message}", flush=True)
