from setuptools import setup, find_packages

setup(
    name="past2harm",
    version="1.0.0",
    author="Snehasis Mukhopadhyay",
    description="Adaptive Past-Tense Jailbreak Framework for Multimodal AI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.30.0",
        "google-generativeai>=0.8.0",
        "diffusers>=0.29.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "torch>=2.2.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "tenacity>=8.2.0",
        "jsonlines>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "past2harm-attack=scripts.run_attack:main",
            "past2harm-eval=scripts.evaluate_asr:main",
        ]
    },
)
