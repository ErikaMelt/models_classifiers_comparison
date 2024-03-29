import plotly.graph_objects as go
import plotly.graph_objs as go
from dash import dcc
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_heat_map(cm):
    # Create heatmap figure with annotations
    heatmap = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues',
        text=cm,
        colorbar=dict(title='Count')
    ))

    heatmap.update_layout(
        title='Confusion Matrix',
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label')
    )

    heatmap.update_traces(textfont=dict(color='black', size=12), hoverinfo='text')

    # create dictionary with plotly figure and layout
    plot_dict = {'data': heatmap.data, 'layout': heatmap.layout}

    return plot_dict


def plot_ROC_Curve(y_test, y_pred_prob):
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Create trace for ROC curve
    trace = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC={:.2f})'.format(roc_auc))

    # Create layout for plot
    layout = go.Layout(title='ROC Curve', xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'), hovermode='closest')

    # Create dictionary with plotly figure and layout
    plot_dict = {'data': [trace], 'layout': layout}

    return plot_dict


def plot_graph(num_positive, num_negative):
    graph = go.Figure(
        data=[go.Bar(x=['Positive', 'Negative'], y=[num_positive, num_negative], marker=dict(color='#4170A6'))],
        layout=go.Layout(title='Number of Positive and Negative Reviews'))

    return graph
