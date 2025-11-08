from abdev_core import create_cli_app
from .model import Ablang2ElasticNetModel

app = create_cli_app(Ablang2ElasticNetModel, "AbLang2 + ElasticNet")

if __name__ == "__main__":
    app()