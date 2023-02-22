import plotly.graph_objects as go
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_heap_map(cm):
    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr = cm[0, 1] / np.sum(cm[0, :])
    tpr = cm[1, 1] / np.sum(cm[1, :])

    # Create heatmap figure with annotations
    heatmap = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues',
        colorbar=dict(title='Count')
    ))

    # Add annotations for FPR and TPR
    annotations = [
        go.layout.Annotation(
            x=0.5,
            y=-0.15,
            text=f'FPRate: {fpr:.2f}',
            showarrow=False,
            font=dict(size=14)
        ),
        go.layout.Annotation(
            x=-0.15,
            y=0.5,
            text=f'TPR: {tpr:.2f}',
            showarrow=False,
            font=dict(size=14),
            textangle=-90
        )
    ]

    heatmap.update_layout(
        title='Confusion Matrix',
        annotations=annotations,
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label')
    )

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
