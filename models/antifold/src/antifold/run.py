"""CLI interface for AntiFold baseline."""

from abdev_core import create_cli_app
from .model import AntifoldModel


app = create_cli_app(AntifoldModel, "AntiFold")


if __name__ == "__main__":
    app()

