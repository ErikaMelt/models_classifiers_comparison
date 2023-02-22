from dash import Dash, html

def render(app: Dash) -> html.Div:
    return html.Div(
        html.Div(
            [
                html.H1("Classificator Model Comparison", className="display-3"),
                html.P(
                    "We are going to get the IMBD Movie reviews and classify if the review is positive or negative "
                    "We are going to use a Logistic Regression, Random Forest, Bayes Naives",
                    className="lead",
                ),
                html.Hr(className="my-2"),
            ],
            className="py-3"
        ),
        className="p-3 bg-light rounded-3",
    )
