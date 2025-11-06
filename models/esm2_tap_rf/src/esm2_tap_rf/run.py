"""CLI interface for ESM2 + TAP Random Forest model."""

from abdev_core import create_cli_app
from .model import ESM2TapRFModel


app = create_cli_app(ESM2TapRFModel, "ESM2 + TAP Random Forest")


if __name__ == "__main__":
    app()
