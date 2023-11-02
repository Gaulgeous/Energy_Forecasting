import os
import pandas as pd
import numpy as np
import datetime as dt
import dash
import plotly.express as px
from dash import dcc
from dash import html, callback

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh
from bayes_opt_scripts.ui_datafinder import Forecaster

from plotly.subplots import make_subplots
import plotly.graph_objects as go

forecaster = Forecaster()

dash.register_page(__name__, name='Predictor')

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE", "drop_text": "white", 
             "drop_bg": "#082255", "drop_out": "#007ACE", "pred_colour": 'aqua', 
             "actual_colour": 'white', "base_colour": "#42C4F7"}

# DryBulb	DewPnt	WetBulb	Humidity	ElecPrice	SYSLoad	Holiday


layout = html.Div(
[
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Energy Consumption Forecaster", className="app__header__title"),
                        html.P(
                            "This home page predicts the future consumption of energy for individual households using a variety of ML models.",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                html.Div([
                    html.Label('Dry Bulb', style={"color": "white"}),
                    dcc.Input(id='input-box-db', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Dew Point', style={"color": "white"}),
                    dcc.Input(id='input-box-dp', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Wet Bulb', style={"color": "white"}),
                    dcc.Input(id='input-box-wb', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Humidity', style={"color": "white"}),
                    dcc.Input(id='input-box-hum', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),
            ],
        ),
        html.Div(
            [
                html.Div([
                    html.Label('Electrical Price', style={"color": "white"}),
                    dcc.Input(id='input-box-e', type='text', value='',style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Holiday', style={"color": "white"}),
                    dcc.Input(id='input-box-hol', type='text', value='', style={
                            'color': app_color['drop_text'],            
                            'background-color': app_color['drop_bg'], 
                            'border-color': app_color['drop_out'],    
                        },),
                ], style={'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.Div('', id='empty_label', style={'opacity':0})
                ], style={'width': '24%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Button('Predict', id='button'),
                ], style={'width': '24%', 'display': 'inline-block'}),
            ],
        ),
        html.Div(
            [
                # data distributions
                html.Div(
                    [
                        html.Div(
                            [html.H6("Parameter Distributions", className="graph__title")]
                        ),
                        dcc.Graph(
                            id="parameter_plot",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),
                    ],
                    className="two-thirds column energy__forecast__container",
                ),
                html.Div(
                    [
                        # wind direction
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Forecasted Consumption:", className="graph__title"
                                        )
                                    ]
                                ),
                                html.Div(id="forecasted_amount", children="0.000", 
                                         style={"color": "white", "fontSize": "20px"}),
                            ],
                            className="graph__container second",
                        ),
                        # consumption distribution
                        html.Div(
                            [
                                dcc.Graph(
                                    id="consumption-distribution",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container first",
                        ),
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)

@callback(
    Output('parameter_plot', 'figure'),
    Input('button', 'n_clicks'),
)
def gen_energy_forecast(_):
    """
    Generate the wind speed graph.
    :params interval: update the graph based on an interval
    """
    data = forecaster.return_data()

    # Define the figure
    fig = go.Figure()
    i = 0
    # Add the box traces to the figure
    for column in data.columns:
        fig.add_trace(
            go.Box(y=data[column], name=column, boxpoints='all')
        )

    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        height=700,
    )

    return dict(data=fig.data, layout=layout)


@callback(Output('forecasted_amount', 'children'),
              Input('button', 'n_clicks'),
              State('input-box-hol', 'value'),
              State('input-box-hum', 'value'),
              State('input-box-db', 'value'),
              State('input-box-dp', 'value'),
              State('input-box-wb', 'value'),)
def update_consumption_label(_, holiday, humidity, drybulb, dewpoint, wetbulb):
    if holiday != '' and humidity != '' and drybulb != '' \
            and dewpoint != '' and wetbulb != '':
        
        targets = forecaster.return_targets()
        
        humidity = int(humidity)
        holiday = int(holiday)
        drybulb = int(drybulb)
        dewpoint = int(dewpoint)
        wetbulb = int(wetbulb)

        input = forecaster.create_input(humidity, holiday, drybulb, dewpoint, wetbulb, 10)
        prediction = forecaster.predict(input)

        prediction = np.median(targets)

        return '{:.2f}'.format(prediction)
    else:
        return "0.000"


@callback(Output('consumption-distribution', 'figure'),
              Input('button', 'n_clicks'),
              State('input-box-hol', 'value'),
              State('input-box-hum', 'value'),
              State('input-box-db', 'value'),
              State('input-box-dp', 'value'),
              State('input-box-wb', 'value'),)
def update_consumption_distribution(_, holiday, humidity, drybulb, dewpoint, wetbulb):


    targets = forecaster.return_targets()

    try:
        bin_val = np.histogram(
            targets,
            bins=20,
        )
    
    except Exception as error:
        raise PreventUpdate

    bin_val_max = max(bin_val[0])

    median_val = np.median(targets)

    trace = dict(
        type="bar",
        x=bin_val[1],
        y=bin_val[0],
        marker={"color": app_color["graph_line"]},
        showlegend=False,
        hoverinfo="x+y",
    )

    layout = dict(
        height=350,
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        xaxis={
            "title": "Consumption",
            "showgrid": False,
            "showline": False,
            "fixedrange": True,
        },
        yaxis={
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "title": "Number of Samples",
            "fixedrange": True,
        },
        autosize=True,
        bargap=0.01,
        bargroupgap=0,
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "xanchor": "center",
            "y": 1,
            "x": 0.5,
        },
    )

    data = [trace]

    if holiday != '' and humidity != '' and drybulb != '' \
        and dewpoint != '' and wetbulb != '':

        traces_scatter = [
            {"line_dash": "dash", "line_color": "#2E5266", "name": "Prediction"},
        ]

        scatter_data = [
            dict(
                type="scatter",
                x=[bin_val[int(len(bin_val) / 2)]],
                y=[0],
                mode="lines",
                line={"dash": traces["line_dash"], "color": traces["line_color"]},
                marker={"opacity": 0},
                visible=True,
                name=traces["name"],
            )
            for traces in traces_scatter
        ]

        layout['shapes']=[
            {
                "xref": "x",
                "yref": "y",
                "y1": int(bin_val_max) + 0.5,
                "y0": 0,
                "x0": median_val,
                "x1": median_val,
                "type": "line",
                "line": {"dash": "dot", "color": "#BD9391", "width": 5},
            },
        ]
    
        data = [trace, scatter_data[0]]

    
    return dict(data=data, layout=layout)




