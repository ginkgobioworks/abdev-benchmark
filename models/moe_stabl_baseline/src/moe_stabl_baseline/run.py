"""CLI entry point for MOE Stabl baseline model."""

from abdev_core import create_cli_app
from .model import MoeStablBaselineModel

app = create_cli_app(MoeStablBaselineModel, "MOE Stabl Baseline")

if __name__ == "__main__":
    app()