"""CLI interface for TAP Linear baseline."""

from abdev_core import create_cli_app
from .model import TapLinearModel


app = create_cli_app(TapLinearModel, "TAP Linear")


if __name__ == "__main__":
    app()

