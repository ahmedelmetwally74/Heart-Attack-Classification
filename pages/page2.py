import dash
from dash import html, dcc, Dash
import dash_bootstrap_components as dbc

import dash
from dash import html, dcc, Dash, Input, Output, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import pathlib
from app import app
import plotly.graph_objects as go


loupe_icon = html.Img(src=app.get_asset_url("loupe.png"),
                      style={'height': '32px', 'margin-right': 10})

layout = dbc.Container(
    [
        dbc.Row(
            [
                html.H2("Mental Trends Around The World", className='mb-5',
                        style={'color': '#144a51', 'font-weight': 'bold',
                               'padding-top': '50px', 'text-align': 'center', 'font-size': '30px'}),
                html.Hr(),
                html.H6([loupe_icon, "Explore how geographic and demographic factors influence the prevalence of various mental health disorders."],
                        style={'text-align': 'left', 'margin-top': '10px', 'font-size': 20, 'color': '#333131'},
                        className='mb-4')
            ]
        )
    ]
)
