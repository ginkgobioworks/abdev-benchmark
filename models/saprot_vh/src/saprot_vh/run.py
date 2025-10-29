"""CLI interface for Saprot_VH baseline."""

from abdev_core import create_cli_app
from .model import SaprotVhModel


app = create_cli_app(SaprotVhModel, "Saprot_VH")


if __name__ == "__main__":
    app()

