"""
PAST2HARM: Adaptive Past-Tense Attack for Jailbreaking Multimodal AI

Public API surface:
    from past2harm import Attacker, Reformulator, Judge
    from past2harm.generators import get_generator
"""

from past2harm.reformulator import Reformulator
from past2harm.judge import Judge
from past2harm.attacker import Attacker
from past2harm.metrics import compute_asr, compute_severity_stats

__version__ = "1.0.0"
__all__ = ["Reformulator", "Judge", "Attacker", "compute_asr", "compute_severity_stats"]
