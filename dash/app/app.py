import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import flask
from dash.dependencies import (Input, Output)
import requests
import time
import numpy as np
import plotly.graph_objs as go
import psycopg2
from psycopg2.extras import execute_values
import time
from functions import *


server = flask.Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(children='Profiles Stream Analysis'),
    
    # BLOCK 1
    html.Div(style={'backgroundColor': colors['background'], 'display':'inline-block'}, children=[
    html.H1(children='WS-1',style={'display': 'inline-block'}),
    dcc.Graph(id='demo-live-0',style={'display': 'inline-block'}),
    dcc.Graph(id='demo-metrics-0',style={'display': 'inline-block'}),
    dcc.Graph(id='demo-p-0',style={'display': 'inline-block'})]),

    # BLOCK 2
    html.Div(style={'backgroundColor': colors['background'], 'display':'inline-block'}, children=[
    html.H1(children='WS-2',style={'display': 'inline-block'}),
    dcc.Graph(id='demo-live-1',style={'display': 'inline-block'}),
    dcc.Graph(id='demo-metrics-1',style={'display': 'inline-block'}),
    dcc.Graph(id='demo-p-1',style={'display': 'inline-block'})]),

    # BLOCK 3
    html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(children='WS-3',style={'display': 'inline-block'}),
    dcc.Graph(id='demo-live-2',style={'display': 'inline-block'}),
    dcc.Graph(id='demo-metrics-2',style={'display': 'inline-block'}),
    dcc.Graph(id='demo-p-2',style={'display': 'inline-block'})]),

    ## for every 2 secs the layout updates
    dcc.Interval(id='output-update', interval=10*1000)
])

# BLOCK 1
@app.callback(
    Output('demo-live-0','figure'),
    [Input(component_id='output-update', component_property='n_intervals')]
)
def get_updates_0(n_intervals):
    fig = get_live_updates(0)    
    return fig

@app.callback(
    Output('demo-metrics-0','figure'),
    [Input(component_id='output-update', component_property='n_intervals')]
)
def get_updates_metrics_0(n_intervals):
    fig = get_metric_updates(0)
    return fig

@app.callback(
    Output('demo-p-0','figure'),
    [Input(component_id='output-update', component_property='n_intervals')]
)
def get_updates_p_0(n_intervals):
    fig = get_p_updates(0)
    return fig


# BLOCK 2
@app.callback(
        Output('demo-live-1','figure'),
        [Input(component_id='output-update', component_property='n_intervals')]
)
def get_updates_1(n_intervals):
    fig = get_live_updates(1)
    return fig

@app.callback(
    Output('demo-metrics-1','figure'),
    [Input(component_id='output-update', component_property='n_intervals')]
)
def get_updates_metrics_1(n_intervals):
    fig = get_metric_updates(1)
    return fig

@app.callback(
    Output('demo-p-1','figure'),
    [Input(component_id='output-update', component_property='n_intervals')]
)
def get_updates_p_1(n_intervals):
    fig = get_p_updates(1)
    return fig

# BLOCK 3
@app.callback(
        Output('demo-live-2','figure'),
        [Input(component_id='output-update', component_property='n_intervals')]
)
def get_updates_2(n_intervals):
    fig = get_live_updates(2)
    return fig

@app.callback(
    Output('demo-metrics-2','figure'),
    [Input(component_id='output-update', component_property='n_intervals')]
)
def get_updates_metrics_2(n_intervals):
    fig = get_metric_updates(2)
    return fig

@app.callback(
    Output('demo-p-2','figure'),
    [Input(component_id='output-update', component_property='n_intervals')]
)
def get_updates_p_2(n_intervals):
    fig = get_p_updates(2)
    return fig

# FUNCTIONS

def get_live_updates(machine):
    values = get_last_machine(machine) 
    x = list(np.arange(len(values)))
    time.sleep(1)
    data = go.Scatter(
        x=x, y=values, mode='lines+markers'
    )
    layout = go.Layout(
        title="Last Process",
        xaxis_title="Time",
        yaxis_title="Temperature",
        autosize=False,
        width=600,
        height=270,
        margin=dict(l=50, r=30, t=30, b=30),
        paper_bgcolor="gray",
        plot_bgcolor='lightgray'
    )
    fig= {'data' : [data], 'layout' : layout}
    return fig


def get_metric_updates(machine):
    mean,max,min = get_metrics(machine)
    x = list(np.arange(len(mean)))
    time.sleep(1)
    data = [
    go.Scatter(
        x=x, y=mean, mode='lines+markers', name='mean'
    ),
    go.Scatter(
        x=x, y=max, mode='lines+markers', name='max'
    ),
    go.Scatter(
        x=x, y=min, mode='lines+markers', name='min'
    )
    ]
    layout = go.Layout(
        title="Metrics over time",
        xaxis_title="Time",
        yaxis_title="Temperature",
        autosize=False,
        width=600,
        height=270,
        margin=dict(l=50, r=30, t=30, b=30),
        paper_bgcolor="gray",
        plot_bgcolor='lightblue'
    )
    fig= {'data' : data, 'layout' : layout}
    return fig

def get_p_updates(machine):
    values = get_p(machine)
    x = list(np.arange(len(values)))
    time.sleep(1)
    data = [
    go.Scatter(
        x=x, y=values, mode='lines+markers', marker=dict(color=values) 
    )
    ]
    layout = go.Layout(
        title="Concept drift evaluation",
        xaxis_title="Time",
        yaxis_title="p-value",
        yaxis_type="log",
        autosize=False,
        width=400,
        height=270,
        margin=dict(l=50, r=30, t=30, b=30),
        paper_bgcolor="gray",
        plot_bgcolor='lightblue'
    )
    fig= {'data' : data, 'layout' : layout}
    return fig


if __name__ == "__main__":
    import os

    debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True
    app.run_server(host="0.0.0.0", port=8050, debug=debug)
