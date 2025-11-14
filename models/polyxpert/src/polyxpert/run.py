"""CLI interface for PolyXpert model."""

from abdev_core import create_cli_app
from .model import PolyXpertModel


app = create_cli_app(PolyXpertModel, "polyxpert")


if __name__ == "__main__":
    app()

