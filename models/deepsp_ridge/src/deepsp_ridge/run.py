"""CLI entry point for DeepSP Ridge model."""

from abdev_core import create_cli_app
from .model import DeepSPRidgeModel

app = create_cli_app(DeepSPRidgeModel, "DeepSP Ridge")

if __name__ == "__main__":
    app()

