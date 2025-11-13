"""CLI interface for Saprot_VH baseline."""

from abdev_core import create_cli_app
from .model import Saprot_VH_VL_Model


app = create_cli_app(Saprot_VH_VL_Model, "Saprot_VH_VL")


if __name__ == "__main__":
    app()

