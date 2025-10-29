"""CLI interface for DeepViscosity baseline."""

from abdev_core import create_cli_app
from .model import DeepViscosityModel


app = create_cli_app(DeepViscosityModel, "DeepViscosity")


if __name__ == "__main__":
    app()
