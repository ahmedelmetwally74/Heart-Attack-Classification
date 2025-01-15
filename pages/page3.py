# # Import necessary libraries
# import dash
# from dash import html
# from dash import dcc
# # import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.express as px
# import pandas as pd
# import pathlib
# from app import app
# import dash_bootstrap_components as dbc

# PATH = pathlib.Path(__file__).parent
# DATA_PATH = PATH.joinpath("../data").resolve()

# anxiety = pd.read_csv(DATA_PATH.joinpath("anxiety.csv"))
# depressive = pd.read_csv(DATA_PATH.joinpath("depressive.csv"))

# anxiety = anxiety.sort_values(by='Year')
# depressive = depressive.sort_values(by='Year')

# loupe_icon = html.Img(src=app.get_asset_url("loupe.png"),
#                       style={'height': '34px', 'margin-right': 10})
# # Define the layout of the app
# layout = dbc.Container(children=[
#         dbc.Row([
#             html.Div([
#                 html.H2("The Correlation between GDP and",
#                         className='mb-4',
#                         style={'color': '#144a51', 'font-weight': 'bold', 'padding-top': '80px', 'text-align': 'left'}),
#                 dcc.Dropdown(
#                     id='disorder-dropdown',
#                     options=[

#                         {"label": html.Span(['Anxiety'], style={'color': '#144a51', 'font-size': 30,'font-weight':'bold'}),
#                          'value': 'anxiety'},
#                         {"label": html.Span(['Depression'], style={'color': '#144a51', 'font-size': 30,'font-weight':'bold'}),
#                          'value': 'depressive'}
#                     ],
#                     value='anxiety',
#                     style={'width': '13rem','margin-top':37,
#                            'fontSize': 50, 'color': '#2b6269', 'outline': 'none', 'border': 'none'}
#                 )
#             ], style={'display': 'inline-flex'}),

#             html.H6([loupe_icon,
#                      "Explore the interplay between economic development, as measured by Gross Domestic Product (GDP), and the prevalence of selected mental health disorders"],
#                     style={'text-align': 'left', 'margin-top': '10px', 'font-size': 20, 'color': '#333131'},
#                     className='mb-4'
#                     ),
#             html.Hr(style={'border-color': '#367d85'}),
#             html.Br(),
#             dbc.Row([
#             dbc.Col([
#             dcc.Graph(id='myGraph')],width=10)],
#                 justify='center'),

#             html.P("Filter by Continent",
#                    style={'color': '#367d85', 'padding-left': '10px'}),

#             dcc.Dropdown(
#                 id='my-dropdown',
#                 options=[{'label': region, 'value': region} for region in ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']],
#                 multi=True,
#                 value=None
#             )

#             ])
#         ])


# # Callback to update single graph based on slider and dropdown values
# @app.callback(
#     Output('myGraph', 'figure'),
#     [Input('my-dropdown', 'value'), Input('disorder-dropdown', 'value')]
# )
# def update_graph(dropdownvalue, selected_disorder):

#     if selected_disorder:
#         if selected_disorder == 'anxiety':
#             filtered_df = anxiety.copy()
#         elif selected_disorder == 'depressive':
#             filtered_df = depressive.copy()

#         if dropdownvalue is not None:
#             filtered_df = filtered_df[filtered_df['Continent'].isin(dropdownvalue)]

#         fig = px.scatter(filtered_df, x="GDP", y="Prevalence", size="Population (historical estimates)",
#                          color="Continent", hover_data=['Entity'], animation_frame='Year', log_x=True, size_max=60)

#         fig.update_layout(plot_bgcolor='white',
#                           xaxis=dict(
#                               linecolor='#5e216f',
#                               linewidth=2
#                           ),
#                           yaxis=dict(
#                               linecolor='#5e216f',
#                               linewidth=2
#                           ),
#                           height=470)

#         return fig
# Import necessary libraries
# Import necessary libraries
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
loupe_icon = html.Img(
    src=app.get_asset_url("loupe.png"),
    style={'height': '34px', 'margin-right': 10}
)

# Define the layout of the app
layout = dbc.Container(children=[
    dbc.Row([
        html.Div([
            html.H2(
                "The Correlation between Cholesterol and Age Based on Heart Attack Risk",
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
            [loupe_icon, "Explore how cholesterol levels vary with age and their relationship to heart attack risk."],
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
            ], width=12, style={"marginTop": "20px", "paddingLeft": "200px"})
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
    
    # Create the scatter plot
    fig = px.scatter(
        filtered_df,
        x="age",
        y="chol",
        color="Heart Attack Risk",
        size="chol",
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
