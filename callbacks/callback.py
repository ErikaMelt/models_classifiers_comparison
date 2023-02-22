from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from app import app


def update_output(input1, input2):
    def callback_function(input1_value, input2_value):
        # Perform some operation on the inputs
        result = input1_value + input2_value
        # Return the result
        return result
    # Return the callback function to be registered with the app
    return callback_function


# Define a function to create a callback that triggers an alert
def trigger_alert(input):
    def callback_function(input_value):
        # Raise a PreventUpdate exception if the input is empty
        if not input_value:
            raise PreventUpdate
        # Otherwise, trigger an alert with the input value
        alert_text = f"You entered: {input_value}"
        return f"alert('{alert_text}')"
    # Return the callback function to be registered with the app
    return callback_function

'''
# Define a function to create a callback that updates a graph
def update_graph(input):
    def callback_function(input_value):
        # Get the data and preprocess it
        data = get_data()
        processed_data = preprocess_data(data, input_value)
        # Train the model
        model = train_model(processed_data)
        # Create a figure based on the model output
        fig = create_figure(model)
        # Return the figure to update the graph
        return dcc.Graph(figure=fig)
    # Return the callback function to be registered with the app
    return callback_function
'''

# Register the callbacks with the app
app.callback(
    Output('output-component', 'children'),
    [Input('input1', 'value'),
     Input('input2', 'value')]
)(update_output('input1', 'input2'))

app.callback(
    Output('alert-component', 'children'),
    [Input('input3', 'value')]
)(trigger_alert('input3'))

app.callback(
    Output('graph-component', 'figure'),
    [Input('input4', 'value')]
)(update_graph('input4'))
