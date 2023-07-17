import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc

import ids
from callbacks.app_callback import get_model_metrics
from models.data import load_data
from models.plots import plot_graph

external_stylesheets = [dbc.themes.BOOTSTRAP, "/assets/style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Load Movies Reviews
X_train, y_train, X_test, y_test = load_data(num_words=2000)

# Calculate the number of positive and negative reviews
num_positive = sum(y_train)
num_negative = len(y_train) - num_positive

# Create the graph
graph = dcc.Graph(figure=plot_graph(num_positive, num_negative))

# Image source URL
img_url = "assets/movies_review.jpg"

app.layout = html.Div([
    dcc.Store(id='data-store', data={'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}),
    html.Div([
         html.Img(id="my-image", src=img_url),
         html.P(
             'Select a model from the list to assess its performance using key metrics including accuracy, recall, confusion matrix, and ROC Curve', className='first-paragraph'),
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
            html.H1("Movie Review Classification with Machine Learning", className='display-4 mb-4'),
            html.P(
                'This project involves a comparative analysis of three machine learning classification algorithms: logistic regression, random forest, and Naive Bayes. The analysis is performed on the IMDB Movies review dataset obtained from Keras, with the goal of accurately classifying reviews as positive or negative. By thoroughly examining the performance of each algorithm, the study aims to determine the most effective approach for sentiment classification in the context of movie reviews.'         , className='second-paragraph'),
            html.Div(children=[graph], className="graph-model", style={'padding-bottom': '400px'}),
            html.Div([
                dcc.Loading(
                    id="loading",
                    children=[
                        dcc.Store(id='results-store'),
                        html.Div(id='output-div')
                    ]
                )
            ], className='row', style={'padding-top': '150px'}),
        ]),
    ], id='right-container', className='col-md-8')
], id='container', className='container-fluid')


app.callback(
    Output(component_id='results-store', component_property='data'),
    Input(component_id='update-button', component_property='n_clicks'),
    State(component_id='class-dropdown', component_property='value'),
    State('data-store', 'data')
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
                                    html.H5(key.capitalize(), className="card-title", style={"text-align": "center"}),
                                    html.P(f"{value:.2f}", className="card-value", style={"text-align": "center"})

                            ],
                            className="card-body"
                        )
                    ],
                    className="m-3",
                )
                cards.append(card)
        return html.Div([
            html.Div([
                html.Div(cards, className='d-flex flex-wrap justify-content-center align-items-center mb-4'),
                html.Div([
                    dcc.Graph(id="roc", figure={'data': results['roc']['data'], 'layout': results['roc']['layout']}),
                ], className='col-md-6'),
                html.Div([
                    dcc.Graph(id="cm", figure={'data': results['cm']['data'], 'layout': results['cm']['layout']}),
                ], className='col-md-6'),
            ], className='row mb-4'),
        ])
    else:
        return html.Div("")


if __name__ == '__main__':
    app.run_server(debug=True)
