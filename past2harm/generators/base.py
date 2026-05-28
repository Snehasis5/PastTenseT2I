"""
Abstract base class for T2I generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class GeneratorResult:
    """Output of a single generate() call."""
    image: Image.Image | None           # PIL image (None if generation failed/refused)
    prompt_used: str                    # The actual prompt sent to the model
    raw_response: Any = None            # Model-specific raw response object
    refused: bool = False               # True if the model explicitly refused
    refusal_reason: str = ""            # Refusal message if refused=True
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseGenerator(ABC):
    """Abstract interface for text-to-image generators."""

    @abstractmethod
    def generate(self, prompt: str) -> GeneratorResult:
        """
        Generate an image from a text prompt.

        Args:
            prompt: The text prompt to generate from.

        Returns:
            GeneratorResult containing the image (or refusal info).
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "BaseGenerator":
        """Instantiate from the full config dict."""
        ...

    def describe_output(self, result: GeneratorResult) -> str:
        """
        Return a text description of the result for the judge.
        If the model refused, return the refusal reason.
        If an image was generated, return a placeholder description
        (actual VQA is done in the judge via the image itself).
        """
        if result.refused:
            return f"[REFUSAL] {result.refusal_reason}"
        if result.image is not None:
            return f"[IMAGE GENERATED] Prompt: {result.prompt_used}"
        return "[EMPTY OUTPUT]"
