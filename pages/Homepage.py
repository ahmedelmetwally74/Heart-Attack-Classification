import pathlib
import dash
from dash import html, dcc, Dash, Input, Output, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
from app import app

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df1 = pd.read_csv(DATA_PATH.joinpath("heart_attack_factors.csv"))

indices = df1[df1['Year']<1999].index
df1 = df1.drop(indices, axis=0).reset_index(drop = True)

download_icon = DashIconify(icon="lets-icons:arrow-drop-down-big", style={'margin-right': 5})
chest_icon = html.Img(src=app.get_asset_url('chest-pain.png'),
                      style={'height': '48px', 'margin-right': 10})
blood_pressure_icon = html.Img(src=app.get_asset_url('blood-pressure.png'),
                           style={'height': '48px', 'margin-right': 10})
cholesterol_icon = html.Img(src=app.get_asset_url('cholesterol.png'),
                           style={'height': '48px', 'margin-right': 10})
exercise_icon = html.Img(src=app.get_asset_url('exercise.png'),
                           style={'height': '48px', 'margin-right': 10})
heart_rate_icon = html.Img(src=app.get_asset_url('heart-rate.png'),
                           style={'height': '48px', 'margin-right': 10})
submit_icon = DashIconify(icon="game-icons:click", style={'margin-right': 5,'height':40})
people_icon = DashIconify(icon="akar-icons:people-group", style={'height': '48px'})


layout = dbc.Container([
    dbc.Row([
    html.Br(),

    dbc.Col([
        dbc.Row([
            html.H2("The Influence of Demographic and Health Factors on Heart Disease",
                    className='mb-5',
                    style={'color': '#144a51', 'font-weight': 'bold', 'padding-top': '80px', 'text-align': 'justify'}),
            dbc.Col([
            dbc.Button([download_icon, "Show Description"], outline=True, color='info',
                       id="proj_des_button",className='mb-2', size=3)],width={'size': 4}),
            html.Div(id="proj_des")
        ],
            className='float-left mb-4'),
        dbc.Row([
        dbc.Col([
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            html.P(
                                "Chest pain can result from various conditions, such as heart problems like angina or heart attacks, or non-cardiac causes like muscle strain or acid reflux. The pain varies in intensity and location, ranging from sharp to pressure-like. While some types of pain are not dangerous, others can be life-threatening and require immediate medical attention. Chest pain can also impact quality of life due to stress, anxiety, and difficulty performing daily activities."
                                , style={'text-align': 'justify', 'color':'#272727'}),
                        ],
                        title=html.Div([

                            html.Span(chest_icon),
                            html.Div([
                                html.P("Chest Pain",
                                       style={'color':'#272727', 'font-size':16}),

                                html.P("Types and Impact",
                                       style={'font-size': 14, 'color': '#999999', 'margin-bottom': 0,
                                              })])

                        ], style={'display': 'inline-flex', 'align-items': 'center'})
                    ),

                    dbc.AccordionItem(
                        [
                            html.P(
                                "Blood pressure plays a significant role in heart disease. High blood pressure, or hypertension, can damage the heart and blood vessels over time. This damage makes it easier for plaque to build up in the arteries, a condition known as atherosclerosis. As plaque accumulates, it narrows the blood vessels, restricting blood flow to the heart. If the blood flow is severely restricted or blocked, it can lead to a heart attack. Managing blood pressure is crucial for reducing the risk of heart disease."
                                , style={'text-align': 'justify', 'color':'#272727'}),
                        ],
                         title = html.Div([

                            html.Span(blood_pressure_icon),
                            html.Div([
                                html.P("High Blood Pressure",
                                       style={'color': '#272727', 'font-size': 16}),

                                html.P("More about it",
                                       style={'font-size': 14, 'color': '#999999', 'margin-bottom': 0,
                                              })])

                        ], style={'display': 'inline-flex', 'align-items': 'center'})
                    ),
                    dbc.AccordionItem(
                        [
                            html.P(
                                "Cholesterol levels are another important risk factor for heart disease. High levels of LDL (low-density lipoprotein), often referred to as 'bad' cholesterol, contribute to the buildup of plaque in the arteries. This plaque narrows and hardens the arteries, restricting blood flow. On the other hand, low levels of HDL (high-density lipoprotein), or 'good' cholesterol, make it harder for the body to remove this plaque. The combination of high LDL and low HDL increases the risk of heart attacks and other cardiovascular problems. Managing cholesterol levels is essential for heart health"
                                , style={'text-align': 'justify', 'color':'#272727'}),
                        ],
                        title = html.Div([

                        html.Span(cholesterol_icon),
                        html.Div([
                            html.P("cholesterol",
                                   style={'color': '#272727', 'font-size': 16}),

                            html.P("How to manage?",
                                   style={'font-size': 14, 'color': '#999999', 'margin-bottom': 0,
                                          })])

                    ], style={'display': 'inline-flex', 'align-items': 'center'})

                    ),
                    dbc.AccordionItem(
                        [
                            html.P(
                                "Exercise is crucial for maintaining heart health. Regular physical activity strengthens the heart muscle, allowing it to pump blood more efficiently. It also helps lower blood pressure by improving the flexibility of blood vessels. Additionally, exercise improves circulation, ensuring that oxygen and nutrients are effectively delivered throughout the body. These benefits significantly reduce the risk of heart attacks and other cardiovascular conditions. Incorporating regular exercise into daily life is an important step toward protecting and improving heart health."
                                , style={'text-align': 'justify', 'color':'#272727'}),
                        ],
                        title = html.Div([

                        html.Span(exercise_icon),
                        html.Div([
                            html.P("Exercise and Heart Attack Prevention",
                                   style={'color': '#272727', 'font-size': 16}),

                            html.P("Keeping Your Heart Healthy",
                                   style={'font-size': 14, 'color': '#999999', 'margin-bottom': 0,
                                          })])

                         ], style={'display': 'inline-flex', 'align-items': 'center'})
                    ),
                    dbc.AccordionItem(
                        [
                            html.P(
                                "Heart rate is a critical indicator of cardiovascular health, reflecting how many times the heart beats per minute. Factors such as physical activity, stress, age, and overall fitness can influence heart rate. Monitoring heart rate helps in understanding one's health and detecting potential conditions like arrhythmia or tachycardia.",
                                style={'text-align': 'justify', 'color':'#272727'}),
                        ],
                        title = html.Div([

                        html.Span(heart_rate_icon),
                        html.Div([
                            html.P("Heart Rate",
                                   style={'color': '#272727', 'font-size': 16}),

                            html.P("Indicator of Cardiovascular Health",
                                   style={'font-size': 14, 'color': '#999999', 'margin-bottom': 0,
                                          })])

                        ], style={'display': 'inline-flex', 'align-items': 'center'})
                    ),

                ],style={'margin-top': '20px'}
            )

            ],className='float-left')
            ])
            ], width={'size': 7}),

        dbc.Col([
            dbc.Row([
                    html.Div([
                                    dbc.Button(id='live-card',
                                            className="text-center m-4 bg-white border-white ",
                                               ),
                                    dbc.Popover(id='popover',
                                                target="live-card",
                                                trigger="hover",
                                                placement="bottom")

                    ],style={"height": "80px", "padding-top": 60, "margin-left": "200px", "margin-bottom": "20px",'width': '70%'}),

                    dcc.Interval(
                            id='interval-component',
                            interval=6000,  # in milliseconds
                            n_intervals=0
                    )
                    ]),

            html.Div(children=[html.H6('Average Prevalence Rate of Heart Attack Reasons',
                                       style={'text-align':'center','color':'#6c857e'}),
                    dcc.Graph(id='bar_plot'),
                    dcc.Slider(df1['Year'].min(),
                               df1['Year'].max(),
                               marks={str(year): year for year in df1['Year'].unique() if (year%2 != 0)},
                               step=None,
                               id="year_slider",
                               value=df1['Year'].max(),className='mb-3'),
                    html.Div([
                                html.Div([
                                    html.P('select by country or worldwide :',
                                    style={'font-size' : 12}),
                                    dcc.Dropdown(df1['Entity'].unique(),
                                                 id='entity_dropdown',
                                                 value='World',
                                                 className='mb-4 me-5',
                                                 style={'width': '250px',
                                                        'border-color': '#a3dfe8'
                                                        }
                                                 )
                                    ],style={'margin-right': '150px'}),

                    dbc.Button([submit_icon, "submit"], outline=True, color='info',
                               id="submit_button",className='custom-button text-end')
                        ],style={'display': 'flex', 'align-items': 'center'})


                    ],style={'margin-top': 160})

                    ], width={'size': 5},align='top')

            ], justify='start')
    ])

@app.callback(
    Output("proj_des", "children"),
    Input("proj_des_button", "n_clicks")
)
def show_text(n_clicks):
    if n_clicks is not None and n_clicks % 2 != 0:
        dp = html.P('This dashboard is designed to explore the factors influencing heart attack risk and recovery. By analyzing key attributes such as age, gender, cholesterol levels, blood pressure, and exercise habits, we aim to uncover patterns and trends that can guide prevention and treatment efforts. Using reliable data sources, the dashboard provides insights into how various physiological and lifestyle factors contribute to heart health, highlighting areas where targeted interventions can make a difference.',
                    className="text-monospace float-none mt-3",
                    style={'color': '#272727', 'text-align': 'justify'})
        return dp
    else:
        return None


@app.callback(
    Output("bar_plot", "figure"),
    Input("year_slider", "value"),
    Input("entity_dropdown", "value"),
    Input("submit_button", "n_clicks")
)
def update_figure(selected_year, selected_entity, n):

    colors = {'High Blood Pressure': '#B4A7D6', 'High LDL Cholesterol': '#9FC5E8', 'Smoking': '#B7ECFE', 'Obesity': '#98dab8',
              'Diabetes': '#FFE599'}
    filtered_df = df1[(df1['Year'] == selected_year) & (df1['Entity'] == selected_entity)]
    fig=px.bar(filtered_df, x='Top 5 Risk Factors', y='Prevalence', color='Top 5 Risk Factors',color_discrete_map=colors, text='Prevalence')
    fig.update_layout(
    plot_bgcolor='white',
    font_color="#647585",
    height=550,
    showlegend=False,
    yaxis=dict(showticklabels=False),
    xaxis=dict(tickangle=0)
    )

    fig.update_traces(marker=dict(cornerradius=5),textfont_size=12,textposition="outside",texttemplate='%{text:.2s%}%')

    fig.add_annotation(
        x='High Blood Pressure',
        y=max(filtered_df['Prevalence']),
        text='The Highest Globally',
        showarrow=True,
        arrowhead=1,
        arrowcolor='orange',
        ay=-30,
        yref='y',
        yshift=30
    )
    fig.update_layout(
    xaxis_title='Health Factor',
    yaxis_title='Prevalence Rate (%)',
    xaxis=dict(
        tickmode='array',
        tickvals=filtered_df['Top 5 Risk Factors'],
        ticktext=[factor.replace(" ", "<br>") for factor in filtered_df['Top 5 Risk Factors']],
        # tickangle=45  
    ), showlegend=False)
    return fig

@app.callback(
    [Output('live-card', 'children'),
     Output('popover', 'children')],
    [Input('interval-component', 'n_intervals')]
)
# def update_display_value(n):
#     # Determine the index of the value to display based on the number of intervals passed
#     live_text = ['319.9', '280M', '70 M', '40M', '24M', '15M']
#     heart_factors = ['High Blood Pressure', 'High LDL Cholesterol', 'eating disorder', 'Smoking', 'Diabetes', 'Diabetes']
#     index = n % len(live_text)

#     # Create the contents of the card
#     button_contents = [html.H1(children=[html.I(className="bi bi-people-fill m-2"), live_text[index]],
#                               style={'color':'#144a51','font-weight':'bold'})]
#     popover = dbc.PopoverBody("Estimated affected people by {} in millions".format(heart_factors[index]),style={'color': '#272727'})

#     return button_contents, popover
def update_display_value(n):
    # Determine the index of the value to display based on the number of intervals passed
    live_text = ['1.13B', '28%', '650M', '1B', '400M']
    heart_factors = ['High Blood Pressure', 'High LDL Cholesterol', 'Obesity', 'Smoking', 'Diabetes']
    index = n % len(live_text)

    # Create the contents of the card
    button_contents = [html.H1(children=[html.I(className="bi bi-people-fill m-2"), live_text[index]],
                              style={'color':'#144a51','font-weight':'bold'})]
    popover = dbc.PopoverBody("Estimated affected people by {} in millions, which can lead to heart attacks in {} people.".format(heart_factors[index], live_text[index]), 
                              style={'color': '#272727'})

    return button_contents, popover