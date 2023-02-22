import dash
from dash import html, dcc

import ids

app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div({
        html.H1("Model Classifier Comparison for Movies Reviews"),
        html.P(
            'Summary of the metrics of each model selected for the classification of movies reviews positives or '
            'negatives'),
        # html.Img(),
        html.Label("Select Model", className='dropdown-labels'),
        dcc.Dropdown(
            id='class-dropdown',
            className='dropdown',
            options=[{"label": model, "value": model} for model in ids.models],
            value=ids.models[0],
            multi=False,
        ),
        html.Button(id='update-button', children="Submit")
    }, id='left-container'),
    html.Div([
        html.Div([
            html.Label("Metrics", className='other-labels'),
            dcc.Graph(id="histogram"),
            dcc.Graph(id="barplot")
        ], id='visualisation'),
        html.Div([
            dcc.Graph(id='table'),
        ], id='data-extract')
    ], id='right-container')
], id='container')


if __name__ == '__main__':
    app.run_server(debug=True)


