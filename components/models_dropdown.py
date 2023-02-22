from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from models.select_model import select_model
from . import ids, display_model_results


def render(app: Dash) -> html.Div:
    models = [ids.NAIVE_BAYES, ids.LOGISTIC_REGRESSION, ids.RANDOM_FOREST]

    @app.callback(
        Output(ids.MODEL_RESULTS, "children"),
        Input(ids.MODELS_DROPDOWN, "value"),
        State("submit-button", "n_clicks"),
    )
    def update_selected_model(selected_model, n_clicks):
        if n_clicks:
            results = select_model(selected_model)
            return display_model_results.render(results, selected_model)

    return html.Div(
        children=[
            html.H6("Select a Model"),
            dcc.Dropdown(
                id=ids.MODELS_DROPDOWN,
                options=[{"label": model, "value": model} for model in models],
                value=models[0],
                multi=False,
            ),
            html.Button("Submit", id="submit-button", n_clicks=0),
            html.Div(id=ids.MODEL_RESULTS),
        ]
    )
