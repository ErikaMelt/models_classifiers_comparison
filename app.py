import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc

import ids
from callbacks.app_callback import get_model_metrics

external_stylesheets = [dbc.themes.BOOTSTRAP, "/assets/style.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(
        [html.H1("Model Classifier Comparison for Movies Reviews", className='display-4 mb-4'),
         html.P(
             'Summary of the metrics of each model selected for the classification of movies reviews positives or '
             'negatives', className='lead'),
         html.Label("Select Model", className='dropdown-labels'),
         dcc.Dropdown(
             id='class-dropdown',
             className='dropdown',
             options=[{"label": model, "value": model} for model in ids.models],
             value=ids.models[0],
             multi=False,
         ),
         html.Button(id='update-button', children="Submit", className='btn btn-primary mt-3')]
        , id='left-container', className='col-md-4'),
    html.Div([
        html.Div([
            html.Div([
                dcc.Loading(
                    id="loading",
                    children=[
                        dcc.Store(id='results-store'),
                        html.Div(id='output-div')
                    ]
                )
            ], className='row'),
        ]),
    ], id='right-container', className='col-md-8')
], id='container', className='container-fluid')

app.callback(
    Output(component_id='results-store', component_property='data'),
    Input(component_id='update-button', component_property='n_clicks'),
    State(component_id='class-dropdown', component_property='value')
)(get_model_metrics)


@app.callback(
    Output('output-div', 'children'),
    [Input('results-store', 'data')]
)
def display_results(results):
    if results is not None:
        cards = []
        for key, value in results.items():
            if key in ["accuracy", "precision", "recall"]:
                card = dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5(key.capitalize(), className="card-title"),
                                html.P(f"{value:.2f}", className="card-value"),
                            ],
                            className="card-body"
                        )
                    ],
                    className="m-3",
                )
                cards.append(card)
        return html.Div([
            html.Div([
                html.Div(cards, className='d-flex flex-wrap justify-content-center align-items-center mb-4')
            ], className='row mb-4'),
            html.Div([
                html.Div(
                    dcc.Graph(id="cm", figure={'data': results['cm']['data'], 'layout': results['cm']['layout']}),
                    className="graph-container")
            ], className='row mb-4'),
            html.Div([
                html.Div(
                    dcc.Graph(id="roc", figure={'data': results['roc']['data'], 'layout': results['roc']['layout']}),
                    className="graph-container")
            ], className='row mb-4')
        ])
    else:
        return html.Div("No results to display, please select a model to start")


if __name__ == '__main__':
    app.run_server(debug=True)
