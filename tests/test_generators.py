"""
Unit tests for generator modules.
All external API/model calls are mocked.
"""

import base64
import io
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from past2harm.generators.base import GeneratorResult, BaseGenerator
from past2harm.generators.openai_gen import OpenAIImageGenerator
from past2harm.generators.sdxl_gen import SDXLGenerator


def make_fake_b64_image() -> str:
    """Create a tiny red PNG and encode as base64."""
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestOpenAIImageGenerator:

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"})
    def test_successful_generation(self):
        b64 = make_fake_b64_image()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json=b64)]

        with patch("past2harm.generators.openai_gen.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.images.generate.return_value = mock_response

            gen = OpenAIImageGenerator(model="gpt-image-1", api_key="sk-fake")
            result = gen.generate("test prompt")

        assert not result.refused
        assert result.image is not None
        assert result.image.size == (64, 64)
        assert result.prompt_used == "test prompt"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"})
    def test_refusal_on_content_policy_error(self):
        from openai import BadRequestError

        with patch("past2harm.generators.openai_gen.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.images.generate.side_effect = Exception(
                "content_policy_violation: This request has been blocked."
            )

            gen = OpenAIImageGenerator(model="gpt-image-1", api_key="sk-fake")
            result = gen.generate("harmful prompt")

        assert result.refused
        assert result.image is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"})
    def test_from_config(self):
        config = {
            "openai_image": {
                "model": "gpt-image-1",
                "size": "1024x1024",
                "quality": "standard",
            }
        }
        gen = OpenAIImageGenerator.from_config(config)
        assert gen.model == "gpt-image-1"
        assert gen.size == "1024x1024"


class TestSDXLBlackImageDetection:

    def test_all_black_image_detected(self):
        black = Image.new("RGB", (64, 64), color=(0, 0, 0))
        assert SDXLGenerator._is_black_image(black, threshold=0.98) is True

    def test_non_black_image_not_detected(self):
        red = Image.new("RGB", (64, 64), color=(255, 0, 0))
        assert SDXLGenerator._is_black_image(red, threshold=0.98) is False

    def test_mostly_black_image_detected(self):
        import numpy as np
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[0, 0] = [255, 0, 0]  # one non-black pixel
        img = Image.fromarray(arr)
        assert SDXLGenerator._is_black_image(img, threshold=0.98) is True

    def test_from_config(self):
        config = {
            "sdxl": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "resolution": 1024,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
            }
        }
        gen = SDXLGenerator.from_config(config)
        assert gen.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
        assert gen.resolution == 1024
        assert gen._pipe is None  # lazy load


class TestGeneratorResult:

    def test_default_fields(self):
        img = Image.new("RGB", (32, 32))
        r = GeneratorResult(image=img, prompt_used="test")
        assert not r.refused
        assert r.refusal_reason == ""
        assert r.metadata == {}

    def test_refusal_result(self):
        r = GeneratorResult(image=None, prompt_used="test", refused=True, refusal_reason="blocked")
        assert r.refused
        assert r.image is None


class TestGeneratorFactory:
    """Test get_generator factory function."""

    def test_invalid_model_raises(self):
        from past2harm.generators import get_generator
        with pytest.raises(ValueError, match="Unknown generator"):
            get_generator("invalid_model", {})

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"})
    def test_openai_factory(self):
        from past2harm.generators import get_generator
        config = {"openai_image": {"model": "gpt-image-1", "size": "1024x1024", "quality": "standard"}}
        gen = get_generator("openai", config)
        assert isinstance(gen, OpenAIImageGenerator)
