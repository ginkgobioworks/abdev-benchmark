"""CLI interface for ESM2 Ridge baseline."""

from abdev_core import create_cli_app
from .model import OneHotRidgeModel


app = create_cli_app(OneHotRidgeModel, "OneHotRidgeModel")


if __name__ == "__main__":
    app()

