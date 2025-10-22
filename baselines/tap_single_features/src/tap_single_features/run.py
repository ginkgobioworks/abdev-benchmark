"""CLI interface for TAP Single Features baseline."""

from abdev_core import create_cli_app
from .model import TapSingleFeaturesModel


app = create_cli_app(TapSingleFeaturesModel, "TAP Single Features")


if __name__ == "__main__":
    app()

