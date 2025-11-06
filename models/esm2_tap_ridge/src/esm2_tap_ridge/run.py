"""CLI interface for ESM2 + TAP Ridge regression model."""

from abdev_core import create_cli_app
from .model import ESM2TapRidgeModel


app = create_cli_app(ESM2TapRidgeModel, "ESM2 + TAP Ridge")


if __name__ == "__main__":
    app()
