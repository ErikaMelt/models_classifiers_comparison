from dash import html

def render(results, model) -> html.Div:
    return html.Div(f"Results for {model}: {results}")
