# import dash
# from dash import html
# from dash import dcc
# from dash.dependencies import Input, Output
# import pandas as pd
# import pathlib
# from app import app
# import dash_bootstrap_components as dbc
# import plotly.express as px

# # Define the path to the data
# PATH = pathlib.Path(__file__).parent
# DATA_PATH = PATH.joinpath("../data").resolve()

# # Load the heart dataset
# df = pd.read_csv(DATA_PATH.joinpath("heart_preprocessed.csv"))

# # Map target values to descriptive labels
# df['Heart Attack Risk'] = df['target'].map({0: 'Less chance', 1: 'More chance'})

# # Create the loupe icon
# search_icon = html.Img(
#     src=app.get_asset_url("loupe.png"),
#     style={'height': '29px', 'margin-right': 10}
# )

# # Define the layout of the app
# layout = dbc.Container(children=[
#     dbc.Row([
#         html.Div([
#             html.H2(
#                 "Heart Attack Risk based on Cholesterol and Age",
#                 className='mb-4',
#                 style={
#                     'color': '#144a51',
#                     'font-weight': 'bold',
#                     'padding-top': '80px',
#                     'text-align': 'left'
#                 }
#             )
#         ], style={'display': 'inline-flex'}),

#         html.H6(
#             [search_icon, "Explore how cholesterol levels vary with age and their relationship to heart attack risk."],
#             style={
#                 'text-align': 'left',
#                 'margin-top': '10px',
#                 'font-size': 20,
#                 'color': '#333131'
#             },
#             className='mb-4'
#         ),
#         html.Hr(style={'border-color': '#367d85'}),
#         html.Br(),

#         html.P(
#             "Filter by Heart Attack Risk",
#             style={'color': '#367d85', 'padding-left': '10px'}
#         ),

#         # Dropdown for filtering by Heart Attack Risk
#         dcc.Dropdown(
#             id='my-dropdown',
#             options=[
#                 {'label': 'Less chance', 'value': 'Less chance'},
#                 {'label': 'More chance', 'value': 'More chance'}
#             ],
#             multi=True,
#             value=None,
#             # style={'width': '50%'}
#         ),

#         dbc.Row([
#             dbc.Col([
#                 # Placeholder for the graph
#                 dcc.Graph(id='chol_vs_age_graph', style={"width": "100%", "height": "70vh"})
#             ], width=12, style={"marginTop": "20px", "paddingLeft": "200px"})
#         ], justify='center')
#     ])
# ])


# # Callback to update the plot based on dropdown selection
# @app.callback(
#     Output('chol_vs_age_graph', 'figure'),
#     [Input('my-dropdown', 'value')]
# )
# def update_graph(selected_risk):
#     # Filter the dataframe based on dropdown selection
#     filtered_df = df if not selected_risk else df[df['Heart Attack Risk'].isin(selected_risk)]
    
#     # Create the scatter plot
#     fig = px.scatter(
#         filtered_df,
#         x="age",
#         y="chol",
#         color="Heart Attack Risk",
#         size="chol",
#         title="Cholesterol Level by Age and Risk",
#         labels={"chol": "Cholesterol Level", "age": "Age"}
#     )

#     # Update the layout for the plot
#     fig.update_layout(
#         height=470,  
#         width=900,
#         margin=dict(l=40, r=40, t=40, b=40),
#         plot_bgcolor="white"
#     )
    
#     return fig

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import pathlib
from app import app
import dash_bootstrap_components as dbc
import plotly.express as px

# Define the path to the data
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

# Load the heart dataset
df = pd.read_csv(DATA_PATH.joinpath("heart_preprocessed.csv"))

# Map target values to descriptive labels
df['Heart Attack Risk'] = df['target'].map({0: 'Less chance', 1: 'More chance'})

# Create the loupe icon
search_icon = html.Img(
    src=app.get_asset_url("loupe.png"),
    style={'height': '29px', 'margin-right': 10}
)

# Define the layout of the app
layout = dbc.Container(children=[
    dbc.Row([
        html.Div([
            html.H2(
                "Heart Attack Risk based on Cholesterol and Age",
                className='mb-4',
                style={
                    'color': '#144a51',
                    'font-weight': 'bold',
                    'padding-top': '80px',
                    'text-align': 'left'
                }
            )
        ], style={'display': 'inline-flex'}),

        html.H6(
            [search_icon, "Explore how cholesterol levels vary with age and their relationship to heart attack risk."],
            style={
                'text-align': 'left',
                'margin-top': '10px',
                'font-size': 20,
                'color': '#333131'
            },
            className='mb-4'
        ),
        html.Hr(style={'border-color': '#367d85'}),
        html.Br(),

        html.P(
            "Filter by Heart Attack Risk",
            style={'color': '#367d85', 'padding-left': '10px'}
        ),

        # Dropdown for filtering by Heart Attack Risk
        dcc.Dropdown(
            id='my-dropdown',
            options=[
                {'label': 'Less chance', 'value': 'Less chance'},
                {'label': 'More chance', 'value': 'More chance'}
            ],
            multi=True,
            value=None,
            # style={'width': '50%'}
        ),

        dbc.Row([
            dbc.Col([
                # Placeholder for the graph
                dcc.Graph(id='chol_vs_age_graph', style={"width": "100%", "height": "70vh"})
            ], width=12, style={"marginTop": "20px", "paddingLeft": "250px"})
        ], justify='center')
    ])
])


# Callback to update the plot based on dropdown selection
@app.callback(
    Output('chol_vs_age_graph', 'figure'),
    [Input('my-dropdown', 'value')]
)
def update_graph(selected_risk):
    # Filter the dataframe based on dropdown selection
    filtered_df = df if not selected_risk else df[df['Heart Attack Risk'].isin(selected_risk)]
    
    # Create the scatter plot with marginal histograms
    fig = px.scatter(
        filtered_df,
        x="age",
        y="chol",
        color="Heart Attack Risk",
        size="chol",
        marginal_x="histogram",  
        marginal_y="histogram",  
        title="Cholesterol Level by Age and Risk",
        labels={"chol": "Cholesterol Level", "age": "Age"}
    )

    # Update the layout for the plot
    fig.update_layout(
        height=470,  
        width=900,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="white"
    )
    
    return fig


