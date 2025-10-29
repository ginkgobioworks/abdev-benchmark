"""CLI interface for Aggrescan3D baseline."""

from abdev_core import create_cli_app
from .model import Aggrescan3dModel


app = create_cli_app(Aggrescan3dModel, "Aggrescan3D")


if __name__ == "__main__":
    app()
