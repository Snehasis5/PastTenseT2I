"""
Utility functions: logging, config loading, image I/O, result serialization.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_env(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise EnvironmentError(
            f"Environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in your API keys."
        )
    return val


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def save_image(image: Image.Image, path: str | Path) -> Path:
    """Save a PIL image to disk, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    image.save(p)
    return p


def image_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base-64 string."""
    import io
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_image(b64_str: str) -> Image.Image:
    """Decode a base-64 string back to a PIL image."""
    import io
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Load benchmark prompts from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("harmful_prompts.json must be a JSON array.")
    required = {"id", "original_prompt"}
    for i, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Prompt #{i} missing fields: {missing}")
    return data


def append_result(result: dict[str, Any], path: str | Path) -> None:
    """Append a single result dict as a JSONL line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load all JSONL results from a file."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def retry_with_backoff(fn, retries: int = 5, base_delay: float = 2.0):
    """Call fn() with exponential backoff on exception."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            logging.getLogger(__name__).warning(
                f"Attempt {attempt + 1} failed: {e}. Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def sanitize_filename(s: str, max_len: int = 80) -> str:
    """Convert an arbitrary string to a safe filename component."""
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"[\s]+", "_", s)
    return s[:max_len]
