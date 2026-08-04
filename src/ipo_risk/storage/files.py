"""原始文件哈希、路径和产物读写。

职责：
  提供文件级别的底层工具，被解析器、报告生成器、证据渲染器等模块调用。

主要功能：
  - sha256_file: 计算文件的 SHA-256 哈希，用于 Evidence.source_hash 字段。
  - ensure_dir: 确保目标目录存在。
  - write_json / read_json: 高速 JSON 读写（使用 orjson）。
  - write_jsonl / read_jsonl: 逐行 JSON（适合大批量解析结果）。

设计原则：
  - 原始数据只读：这个模块不会修改 data/raw/ 下的任何文件。
  - 产物可重建：artifacts/ 和 data/interim/ 下的文件都可以重新生成。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson


# ---------------------------------------------------------------------------
# 文件哈希
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """计算文件的 SHA-256 哈希值。

    参数:
        path: 文件路径。
        chunk_size: 分块读取大小，默认 1MB，避免大文件吃满内存。
                    我们的招股书 PDF 最大约 28MB，分块读取很必要。

    返回:
        64 位小写十六进制字符串。

    为什么需要这个：
        Evidence 契约要求 source_hash 字段，确保每条证据都能追溯到
        具体的 PDF 文件版本。如果 PDF 被修改过，哈希会变，就能发现。
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 目录工具
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """确保目录存在，不存在则创建。

    参数:
        path: 目标目录路径。

    返回:
        同一个路径（方便链式调用）。
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# JSON 读写
# ---------------------------------------------------------------------------

def write_json(data: Any, path: Path, pretty: bool = True) -> Path:
    """将数据写入 JSON 文件。

    参数:
        data: 要序列化的数据（字典、列表等）。
        path: 目标文件路径。
        pretty: 是否格式化输出（开发阶段 True，生产可设 False 节省空间）。

    返回:
        写入的文件路径。
    """
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        if pretty:
            # 用标准库 json 做格式化输出（可读性更好）
            text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            temporary.write_text(text, encoding="utf-8")
        else:
            # 用 orjson 做紧凑输出（速度更快）
            temporary.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def read_json(path: Path) -> Any:
    """读取 JSON 文件。

    参数:
        path: JSON 文件路径。

    返回:
        解析后的 Python 对象。
    """
    with open(path, "rb") as f:
        return orjson.loads(f.read())


# ---------------------------------------------------------------------------
# JSONL 读写（适合大批量解析结果，每行一个 JSON 对象）
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    """将记录列表写入 JSONL 文件（每行一个 JSON）。

    为什么用 JSONL：
        672 页招股书的解析结果如果放一个大 JSON，读起来不方便。
        JSONL 每行一个对象，可以逐行读取、流式处理。
    """
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec, option=orjson.OPT_SERIALIZE_NUMPY))
            f.write(b"\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，返回记录列表。"""
    results: list[dict[str, Any]] = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(orjson.loads(line))
    return results
