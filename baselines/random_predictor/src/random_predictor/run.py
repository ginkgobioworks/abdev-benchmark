"""CLI interface for Random Predictor baseline."""

from abdev_core import create_cli_app
from .model import RandomPredictorModel


app = create_cli_app(RandomPredictorModel, "Random Predictor")


if __name__ == "__main__":
    app()
