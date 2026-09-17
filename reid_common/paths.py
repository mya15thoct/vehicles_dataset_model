"""Path resolution helpers for locating images/annotations under one of several roots."""

from __future__ import annotations

from pathlib import Path


def resolve_with_fallback(root: Path, name: str, fallback_root: Path) -> Path:
    """Look for `name` under `root`, then under `fallback_root`.

    Returns the path under `root` even if neither exists, so callers get a
    predictable path to report in an error message.
    """
    path = root / name
    if path.exists():
        return path
    fallback = fallback_root / name
    if fallback.exists():
        return fallback
    return path


def resolve_or_bare(root: Path, name: str) -> Path:
    """Look for `name` under `root`; fall back to `name` interpreted on its own."""
    path = root / name
    return path if path.exists() else Path(name)
