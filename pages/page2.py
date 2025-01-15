import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, Dash, Input, Output, State
import pandas as pd
import plotly.express as px
import pathlib
from app import app
import plotly.graph_objects as go
import json
import joblib
import os
import requests

model = joblib.load("best_model.joblib")

# Define the list of questions for heart attack classification
questions = [
    "How old are you? (Please enter your age in numbers)",
    "What is your gender? (Male or Female)",
    "What is your Chest Pain Type? (0: Typical Angina 'Most serious', 1: Atypical Angina, 2: Non-anginal pain, 3: Asymptomatic 'Least serious')",
    "What is your resting blood pressure? (in mm Hg)",
    "What is your serum cholesterol level? (in mg/dl)",
    "What is your fasting blood sugar level?",
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

# Chatbot Design with Improved Visuals
chatbot_widget = html.Div(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        "🤖 Hello! I am your AI Doctor. Let's gather some information to assess your heart health.",
                        id="chatbot-greeting",
                        style={"font-weight": "bold", "margin-bottom": "10px", "font-size": "18px"}
                    ),
                    html.Div(id="chatbot-question", style={"margin-bottom": "10px", "font-size": "16px"}),
                    dcc.Textarea(
                        id="chatbot-input",
                        placeholder="Type your response here...",
                        style={
                            'width': '100%',
                            'height': '60px',
                            'margin-bottom': '10px',
                            'font-size': '16px'
                        }
                    ),
                    dbc.Button(
                        "💬 Submit",
                        id="send-chatbot-message",
                        color="success",
                        style={"width": "100%", "font-size": "16px"}
                    ),
                    html.Div(id="chatbot-response", style={
                        'margin-top': '20px',
                        'max-height': '200px',
                        'overflow-y': 'auto',
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'border-radius': '5px',
                        'background-color': '#f9f9f9',
                        'font-size': '14px'
                    })
                ]
            ),
            style={
                'width': '400px',
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
                    "AI Doctor - Heart Health Predictor",
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
                    [
                        "🧠 Explore how the AI Doctor utilizes advanced algorithms to predict and analyze the likelihood of heart-related conditions based on your unique health profile."
                    ],
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


def explain_prediction(df, column_info, prediction):
    """
    Generate an explanation for the heart attack prediction using Groq API.

    Args:
        df (pd.DataFrame): DataFrame containing patient data (single row).
        column_info (dict): Dictionary with descriptions of each feature/column.
        prediction (int): Model prediction (0 for No Heart Attack, 1 for Heart Attack).

    Returns:
        str: Explanation generated by the Groq API.
    """
    # Set the API key environment variable globally (if not already set)
    os.environ["GROQ_API_KEY"] = "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"

    # Set API URL and headers
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }

    # Convert DataFrame row to a readable string
    customer_data_str = df.to_string(index=False, header=True)

    # Format the prediction string
    prediction_str = "Heart Attack" if prediction == 1 else "No Heart Attack"

    # Prepare the prompt for the Groq API
    prompt = f"""
    Please explain why the individual with the following data is predicted to have a heart attack or not (Prediction: {prediction_str}):
    
    Collected Patient Data:
    {customer_data_str}

    Column Information:
    {column_info}

    Act like a doctor and your explanation should be based on the data provided and how it affects the heart attack prediction and give your patient kindly some advices based on the data.
    """

    # Prepare the data for the request
    payload = {
        "model": "llama3-8b-8192",  # Adjust the model name as needed
        "messages": [{"role": "user", "content": prompt}]
    }

    # Send the request to the Groq API
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for HTTP codes 4xx or 5xx
        explanation = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return explanation.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Groq API: {e}")
        return "There was an error generating the explanation. Please try again later."
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

        # Ensure the index is within the range of questions and keys
        if question_index >= len(question_keys):
            # End of questionnaire reached
            df = pd.DataFrame([user_responses])

            # Convert columns to appropriate types
            for col in df.columns:
                if col == "oldpeak":
                    df[col] = df[col].astype(float)
                else:
                    try:
                        df[col] = df[col].astype(float).astype(int)
                    except ValueError:
                        pass

            print(df)
            print(df.info())

            # Predict the outcome
            prediction = model.predict(df)[0]

            # Define the response message
            if prediction == 1:
                result_message = (
                    "Our analysis predicts that you are at risk of a heart attack soon. "
                    "Please consult your doctor immediately!"
                )
            else:
                result_message = (
                    "Our analysis indicates that you are not at immediate risk of a heart attack. "
                    "Keep up the healthy lifestyle!"
                )

            explanation = explain_prediction(df, column_info, prediction)
            with open('user_responses.json', 'w') as file:
                json.dump(user_responses, file)

            return explanation + "\n" + result_message, "", current_responses

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
                return "1" if value.lower() in ["man", "male", "1"] else "0"
            elif key == "fbs":
                return "1" if int(value) > 120 else "0"
            elif key == "exang":
                return "1" if value.lower() in ["yes", "1"] else "0"
            elif key == "oldpeak":
                return float(value)
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

    return dash.no_update
