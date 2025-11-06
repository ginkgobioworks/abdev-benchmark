"""CLI interface for ESM2 + TAP XGBoost model."""

from abdev_core import create_cli_app
from .model import ESM2TapXGBModel


app = create_cli_app(ESM2TapXGBModel, "ESM2 + TAP XGBoost")


if __name__ == "__main__":
    app()
