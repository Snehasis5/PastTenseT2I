"""
Image generator backends.

Usage:
    from past2harm.generators import get_generator

    gen = get_generator("sdxl", config)
    result = gen.generate("a red apple on a table")
    result.image.save("out.png")
"""

from past2harm.generators.base import GeneratorResult, BaseGenerator
from past2harm.generators.openai_gen import OpenAIImageGenerator
from past2harm.generators.gemini_gen import GeminiImageGenerator
from past2harm.generators.sdxl_gen import SDXLGenerator

_REGISTRY: dict[str, type[BaseGenerator]] = {
    "openai": OpenAIImageGenerator,
    "gemini": GeminiImageGenerator,
    "sdxl": SDXLGenerator,
}


def get_generator(name: str, config: dict) -> BaseGenerator:
    """
    Factory function to instantiate a generator by name.

    Args:
        name: One of "openai", "gemini", "sdxl".
        config: Full config dict (from load_config). Generator-specific
                sub-dict is extracted automatically.

    Returns:
        Instantiated generator ready for .generate() calls.
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown generator '{name}'. Available: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    return cls.from_config(config)


__all__ = [
    "get_generator",
    "BaseGenerator",
    "GeneratorResult",
    "OpenAIImageGenerator",
    "GeminiImageGenerator",
    "SDXLGenerator",
]
