"""JSONL file utilities for loading and appending records."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

from loguru import logger


T = TypeVar("T")


def iter_jsonl(
    path: Path,
    parser: Callable[[dict], T] | None = None,
    skip_errors: bool = True,
) -> Iterator[T | dict]:
    """
    Iterate over JSONL file records.

    Args:
        path: Path to JSONL file
        parser: Optional callable to transform dict records (e.g., dataclass constructor)
        skip_errors: If True, skip malformed lines with a warning; if False, raise

    Yields:
        Parsed records (dict if no parser, T if parser provided)

    Example:
        >>> for record in iter_jsonl(Path("data.jsonl")):
        ...     print(record["id"])

        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Example:
        ...     id: str
        ...     value: int
        >>> for ex in iter_jsonl(Path("data.jsonl"), parser=lambda d: Example(**d)):
        ...     print(ex.id)
    """
    if not path.exists():
        logger.warning("JSONL file does not exist", path=str(path))
        return

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                if skip_errors:
                    logger.warning("Skipping malformed JSONL line", line=line_num, error=str(e))
                    continue
                raise

            if parser is not None:
                try:
                    yield parser(data)
                except (TypeError, KeyError) as e:
                    if skip_errors:
                        logger.warning("Skipping unparseable record", line=line_num, error=str(e))
                        continue
                    raise
            else:
                yield data


def load_jsonl(
    path: Path,
    parser: Callable[[dict], T] | None = None,
    skip_errors: bool = True,
) -> list[T | dict]:
    """
    Load all records from a JSONL file into a list.

    Args:
        path: Path to JSONL file
        parser: Optional callable to transform dict records
        skip_errors: If True, skip malformed lines; if False, raise

    Returns:
        List of records

    Example:
        >>> records = load_jsonl(Path("data.jsonl"))
        >>> len(records)
        42
    """
    return list(iter_jsonl(path, parser=parser, skip_errors=skip_errors))


def append_jsonl(path: Path, data: dict) -> None:
    """
    Append a single record to a JSONL file.

    Creates parent directories and file if they don't exist.

    Args:
        path: Path to JSONL file
        data: Dictionary to append as JSON line

    Example:
        >>> append_jsonl(Path("log.jsonl"), {"event": "start", "ts": 1234567890})
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")
