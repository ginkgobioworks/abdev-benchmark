"""CLI entry point for MOE baseline model."""

from abdev_core import create_cli_app
from .model import MoeBaselineModel

app = create_cli_app(MoeBaselineModel, "MOE Baseline")

if __name__ == "__main__":
    app()
