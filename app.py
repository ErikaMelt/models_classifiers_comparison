from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP
from components.layout import create_layout


def main() -> None:
    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "Sentiment Classifier Movies Review"
    app.layout = create_layout(app)
    app.run()


if __name__ == "__main__":
    main()


