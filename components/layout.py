from dash import Dash, html
from dash_bootstrap_components import Row, Col, Tabs, Tab
from components import models_dropdown, header


def create_layout(app: Dash) -> html.Div:
    html.Link(rel='stylesheet', href='/assets/style.css'),
    return html.Div(
        className="app-div",
        children=[
            header.render(app),
            html.Hr(),
            Row([
                Col(
                    children=[
                        html.Div([
                            html.Label('Precision:'),
                            html.Div('{:.3f}'.format(545))
                        ]),
                        models_dropdown.render(app),
                    ],
                    width=4,
                ),
                Col(
                    children=[
                        Tabs([
                            Tab(label='Tab 1', children=[
                                html.H1('Content of Tab 1')
                            ]),
                            Tab(label='Tab 2', children=[
                                html.H1('Content of Tab 2')
                            ])
                        ])
                    ],
                    width=8,
                ),
            ])
        ],
    )
