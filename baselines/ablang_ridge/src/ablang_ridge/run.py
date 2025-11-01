from abdev_core import create_cli_app
from .model import AblangRidgeModel

app = create_cli_app(AblangRidgeModel, "AbLang2 + Ridge Baseline")

if __name__ == "__main__":
    app()