"""Leakage-safe LightGBM experiment package."""

from .config import ExperimentConfig


# Increment this whenever the files that a completed run must export change.
# The Colab notebook checks it after clearing cached imports, so mixing a new
# notebook with an older Drive package fails before a long training run starts.
OUTPUT_CONTRACT_VERSION = 2


def run_experiment(config: ExperimentConfig):
    """Lazy import keeps schema/unit tests usable before Colab dependencies install."""
    from .experiment import run_experiment as _run

    return _run(config)

__all__ = ["ExperimentConfig", "OUTPUT_CONTRACT_VERSION", "run_experiment"]
