"""CLI interface for SSH2 model."""

from abdev_core import create_cli_app
from .model import SSH2Model


app = create_cli_app(SSH2Model, "ssh2")


if __name__ == "__main__":
    app()

