"""CLI interface for ESM2 Ridge baseline."""

from abdev_core import create_cli_app
from .model import ESM2RidgeModel


app = create_cli_app(ESM2RidgeModel, "ESM2 Ridge")


if __name__ == "__main__":
    app()
