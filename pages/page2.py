import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Dash, Input, Output, State
import pandas as pd
import plotly.express as px
import pathlib
from app import app
import plotly.graph_objects as go
import json



# Loupe icon for visual enhancement
loupe_icon = html.Img(
    src=app.get_asset_url("loupe.png"),
    style={'height': '32px', 'margin-right': 10}
)

# Define the list of questions for heart attack classification
questions = [
    "How old are you? (Please enter your age in numbers)",
    "What is your gender? (Male or Female)",
    "What is your Chest Pain Type? (0: Typical Angina 'Most serious', 1: Atypical Angina, 2: Non-anginal pain, 3: Asymptomatic 'Least serious')",
    "What is your resting blood pressure? (in mm Hg)",
    "What is your serum cholesterol level? (in mg/dl)",
    "What is your fasting blood sugar level? (1: Yes if greater than 120 mg/dl, 0: No)",
    "What is your Resting Electrocardiographic Result? (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy)",
    "What is the maximum heart rate achieved during exercise?",
    "Do you experience Exercise-Induced Angina? (Yes or No)",
    "What is the ST depression induced by exercise relative to rest? (Enter a number from 0 to 5)",
    "What is the slope of the ST segment during peak exercise? (0: Upsloping, 1: Flat, 2: Downsloping)",
    "What is the number of major vessels colored by fluoroscopy? (Enter a number from 0 to 3)",
    "Are there any blood flow problems? (0: Normal, 1: Fixed defect, 2: Reversible defect)"
]
column_info = {
    "age": "Age of the individual (5–95 years).",
    "sex": "Gender of the individual (Male or Female).",
    "cp": "Chest Pain Type: 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal pain, 3 = Asymptomatic.",
    "trestbps": "Resting blood pressure (in mm Hg, typically 50–200).",
    "chol": "Serum Cholesterol: Cholesterol level in mg/dl (100–600).",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No).",
    "restecg": "Resting Electrocardiographic Results: 0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy.",
    "thalach": "Maximum heart rate achieved during exercise (60–220 bpm).",
    "exang": "Exercise-Induced Angina (Yes or No).",
    "oldpeak": "ST depression induced by exercise relative to rest (0.0–5.0).",
    "slope": "Slope of the ST segment during peak exercise: 0 = Upsloping, 1 = Flat, 2 = Downsloping.",
    "ca": "Number of major vessels colored by fluoroscopy (0–3).",
    "thal": "Thalassemia: 0 = Normal, 1 = Fixed defect, 2 = Reversible defect."
}
# Mapping questions to keys
question_keys = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

# Chatbot Design
chatbot_widget = html.Div(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        "Hello! I am your AI Doctor. Let's gather some information to assess your heart health.",
                        id="chatbot-greeting",
                        style={"font-weight": "bold", "margin-bottom": "10px"}
                    ),
                    html.Div(id="chatbot-question", style={"margin-bottom": "10px"}),
                    dcc.Textarea(
                        id="chatbot-input",
                        placeholder="Type your response here...",
                        style={
                            'width': '100%',
                            'height': '60px',
                            'margin-bottom': '10px'
                        }
                    ),
                    dbc.Button(
                        "Submit",
                        id="send-chatbot-message",
                        color="success",
                        style={"width": "100%"}
                    ),
                    html.Div(id="chatbot-response", style={
                        'margin-top': '20px',
                        'max-height': '200px',
                        'overflow-y': 'auto',
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'border-radius': '5px',
                        'background-color': '#f9f9f9'
                    })
                ]
            ),
            style={
                'width': '350px',
                'position': 'fixed',
                'top': '100px',
                'right': '20px',
                'z-index': 1000,
                'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'
            }
        )
    ]
)

# Main Layout
layout = dbc.Container(
    [
        dbc.Row(
            [
                html.H2(
                    "Heart Attack Analytics",
                    className='mb-5',
                    style={
                        'color': '#144a51',
                        'font-weight': 'bold',
                        'padding-top': '50px',
                        'text-align': 'center',
                        'font-size': '30px'
                    }
                ),
                html.Hr(),
                html.H6(
                    ["Explore how geographic and demographic factors influence the prevalence of various mental health disorders."],
                    style={
                        'text-align': 'left',
                        'margin-top': '10px',
                        'font-size': 20,
                        'color': '#333131'
                    },
                    className='mb-4'
                )
            ]
        ),
        chatbot_widget
    ]
)

# JSON Storage
user_responses = {}

# Callbacks to handle chatbot interactions
@app.callback(
    [
        Output("chatbot-question", "children"),
        Output("chatbot-input", "value"),
        Output("chatbot-response", "children")
    ],
    [Input("send-chatbot-message", "n_clicks")],
    [State("chatbot-input", "value"), State("chatbot-response", "children")]
)
def chatbot_interaction(send_clicks, user_input, current_responses):
    ctx = dash.callback_context

    if not ctx.triggered:
        # Start the conversation
        if not user_responses:
            first_question = questions[0]
            return first_question, "", ""

    if user_input:
        
        question_index = len(user_responses)

        current_key = question_keys[question_index]

        # Define validation rules
        validation_errors = {
            "age": lambda x: not x.isdigit() or int(x) < 5 or int(x) > 95,
            "sex": lambda x: x.lower() not in ["man", "male", "0", "woman", "female", "1"],
            "cp": lambda x: x not in ["0", "1", "2", "3"],
            "trestbps": lambda x: not x.isdigit() or int(x) < 50 or int(x) > 200,
            "chol": lambda x: not x.isdigit() or int(x) < 100 or int(x) > 400,
            "fbs": lambda x: not x.isdigit() or int(x) < 50 or int(x) > 400,
            "restecg": lambda x: x not in ["0", "1", "2"],
            "thalach": lambda x: not x.isdigit() or int(x) < 60 or int(x) > 220,
            "exang": lambda x: x.lower() not in ["yes", "no", "0", "1"],
            "oldpeak": lambda x: not x.replace('.', '', 1).isdigit() or float(x) < 0 or float(x) > 5,
            "slope": lambda x: x not in ["0", "1", "2"],
            "ca": lambda x: not x.isdigit() or int(x) < 0 or int(x) > 3,
            "thal": lambda x: x not in ["0", "1", "2"]
        }
        # Adjust values based on rules
        def adjust_value(key, value):
            if key == "sex":
                return "0" if value.lower() in ["man", "male", "0"] else "1"
            elif key == "fbs":
                return "1" if int(value) > 120 else "0"
            elif key == "exang":
                return "1" if value.lower() in ["yes", "1"] else "0"
            elif key == "oldpeak":
                return float(value)  # Ensure it's saved as float
            else:
                return value

        # Validate input
        if validation_errors[current_key](user_input):
            error_message = f"Invalid input for {current_key}. Please try again."
            chatbot_message = f"{questions[question_index]} (Hint: {column_info[current_key]})"
            return chatbot_message, "", f"{current_responses}\n{error_message}"

        # Store valid response
        user_responses[current_key] = adjust_value(current_key, user_input)

        # Print current state of JSON
        print(json.dumps(user_responses, indent=2))

        # Check if there are more questions
        if question_index + 1 < len(questions):
            next_question = questions[question_index + 1]
            return next_question, "", current_responses
        else:
            # End of questionnaire
            with open('user_responses.json', 'w') as file:
                json.dump(user_responses, file)
            return "Thank you! Your responses have been recorded.", "", current_responses

    return dash.no_update