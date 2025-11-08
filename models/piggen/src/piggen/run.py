"""CLI interface for p-IgGen baseline."""

from abdev_core import create_cli_app
from .model import PiGGenModel


app = create_cli_app(PiGGenModel, "p-IgGen")


if __name__ == "__main__":
    app()
