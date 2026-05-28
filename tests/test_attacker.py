"""
Unit tests for the Attacker (core adaptive loop).
All API and generator calls are mocked.
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from past2harm.attacker import Attacker
from past2harm.generators.base import GeneratorResult
from past2harm.judge import JudgeResult
from past2harm.reformulator import Reformulator


def make_mock_generator(image=None, refused=False, refusal_reason=""):
    """Return a mock BaseGenerator."""
    gen = MagicMock()
    gen.generate.return_value = GeneratorResult(
        image=image or Image.new("RGB", (64, 64)),
        prompt_used="test prompt",
        refused=refused,
        refusal_reason=refusal_reason,
    )
    gen.describe_output.return_value = "[IMAGE GENERATED] Prompt: test prompt"
    return gen


def make_mock_judge(success=1, severity=0.7):
    j = MagicMock()
    j.evaluate.return_value = JudgeResult(success=success, severity_jailbreak=severity)
    return j


def make_mock_reformulator():
    r = Reformulator(model="rule_based")
    return r


class TestAttackerBasic:

    def test_attack_returns_result(self, tmp_path):
        generator = make_mock_generator()
        judge = make_mock_judge(success=1, severity=0.8)
        reformulator = make_mock_reformulator()

        attacker = Attacker(
            generator=generator,
            reformulator=reformulator,
            judge=judge,
            max_steps=4,
            image_save_dir=tmp_path,
        )

        result = attacker.attack(
            prompt_id="test_001",
            original_prompt="Write something harmful",
            category="Test",
            model_name="mock",
        )

        assert result.prompt_id == "test_001"
        assert result.original_prompt == "Write something harmful"
        assert result.reformulated_prompt is not None
        assert len(result.steps) > 0

    def test_attack_succeeds_on_compliance(self, tmp_path):
        generator = make_mock_generator(refused=False)
        judge = make_mock_judge(success=1, severity=0.9)
        reformulator = make_mock_reformulator()

        attacker = Attacker(
            generator=generator,
            reformulator=reformulator,
            judge=judge,
            max_steps=8,
            image_save_dir=tmp_path,
        )

        result = attacker.attack("id_001", "harmful prompt", model_name="mock")
        assert result.success is True
        assert result.first_success_depth == 1

    def test_attack_fails_on_refusal(self, tmp_path):
        generator = make_mock_generator(refused=True, refusal_reason="Content policy")
        generator.generate.return_value = GeneratorResult(
            image=None, prompt_used="p", refused=True, refusal_reason="Refused"
        )
        generator.describe_output.return_value = "[REFUSAL] Refused"
        judge = make_mock_judge(success=0, severity=0.0)
        reformulator = make_mock_reformulator()

        attacker = Attacker(
            generator=generator,
            reformulator=reformulator,
            judge=judge,
            max_steps=3,
            image_save_dir=tmp_path,
        )

        result = attacker.attack("id_002", "harmful prompt", model_name="mock")
        assert result.success is False
        assert len(result.steps) == 3

    def test_early_stopping_on_plateau(self, tmp_path):
        """Attack should stop early when severity plateaus."""
        generator = make_mock_generator()

        # First step: high severity; subsequent steps: same (plateau)
        severities = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        call_count = [0]

        def mock_evaluate(*args, **kwargs):
            i = call_count[0]
            call_count[0] += 1
            return JudgeResult(success=1, severity_jailbreak=severities[min(i, len(severities)-1)])

        judge = MagicMock()
        judge.evaluate.side_effect = mock_evaluate
        reformulator = make_mock_reformulator()

        attacker = Attacker(
            generator=generator,
            reformulator=reformulator,
            judge=judge,
            max_steps=8,
            plateau_tolerance=2,
            image_save_dir=tmp_path,
        )

        result = attacker.attack("id_003", "harmful prompt", model_name="mock")
        # Should stop before exhausting all 8 steps due to plateau
        assert len(result.steps) < 8

    def test_result_to_dict_is_serializable(self, tmp_path):
        generator = make_mock_generator()
        judge = make_mock_judge(success=1, severity=0.7)
        reformulator = make_mock_reformulator()

        attacker = Attacker(
            generator=generator,
            reformulator=reformulator,
            judge=judge,
            max_steps=2,
            image_save_dir=tmp_path,
        )

        result = attacker.attack("id_004", "test prompt", model_name="mock")
        d = result.to_dict()

        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert "prompt_id" in d
        assert "steps" in d

    def test_images_saved_to_disk(self, tmp_path):
        img = Image.new("RGB", (64, 64), color="blue")
        generator = make_mock_generator(image=img)
        judge = make_mock_judge(success=1, severity=0.8)
        reformulator = make_mock_reformulator()

        attacker = Attacker(
            generator=generator,
            reformulator=reformulator,
            judge=judge,
            max_steps=2,
            image_save_dir=tmp_path,
        )

        attacker.attack("id_005", "test prompt", model_name="testmodel")

        saved = list(tmp_path.rglob("*.png"))
        assert len(saved) > 0
