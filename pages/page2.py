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


# loupe_icon = html.Img(src=app.get_asset_url("loupe.png"),
#                       style={'height': '32px', 'margin-right': 10})

# layout = dbc.Container(
#     [
#         dbc.Row(
#             [
#                 html.H2("Mental Trends Around The World", className='mb-5',
#                         style={'color': '#144a51', 'font-weight': 'bold',
#                                'padding-top': '50px', 'text-align': 'center', 'font-size': '30px'}),
#                 html.Hr(),
#                 html.H6([loupe_icon, "Explore how geographic and demographic factors influence the prevalence of various mental health disorders."],
#                         style={'text-align': 'left', 'margin-top': '10px', 'font-size': 20, 'color': '#333131'},
#                         className='mb-4')
#             ]
#         )
#     ]
# )

import os
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import requests

# Initialize the Dash app
from app import app

# Set API key
os.environ["GROQ_API_KEY"] = "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"

# Paths and model
model = joblib.load('best_model.joblib')
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
    "age": "Age of the individual.",
    "sex": "Gender of the individual (1 = male, 0 = female).",
    "cp": "Chest Pain Type: 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal pain, 3 = Asymptomatic.",
    "trestbps": "Resting blood pressure (in mm Hg).",
    "chol": "Serum Cholesterol: Cholesterol level in mg/dl.",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = True, 0 = False).",
    "restecg": "Resting Electrocardiographic Results: 0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy.",
    "thalach": "Maximum heart rate achieved during exercise.",
    "exang": "Exercise Induced Angina: 1 = Yes, 0 = No.",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "slope": "Slope of the ST segment during peak exercise: 0 = Upsloping, 1 = Flat, 2 = Downsloping.",
    "ca": "Number of Major vessels (0-3) colored by fluoroscopy.",
    "thal": "0 = normal; 1 = fixed defect; 2 = reversible defect."
}

# Layout
layout = dbc.Container(
    [
        dbc.Row(
            [
                html.H2("Heart Attack Risk Assessment Chatbot", className='mb-5',
                        style={'color': '#144a51', 'font-weight': 'bold', 'text-align': 'center'}),
                html.Hr(),
                html.Div(id='chatbot-output', style={
                    'width': '100%', 'height': '300px', 'overflow-y': 'scroll', 'border': '1px solid #ccc', 'padding': '10px',
                    'background-color': '#f8f9fa'
                }),
                dbc.Input(id='chatbot-input', placeholder="Type your message here...", style={'width': '80%', 'margin-top': '10px'}),
                dbc.Button("Send", id='send-button', color="primary", style={'margin-left': '10px', 'margin-top': '10px'}),
            ]
        )
    ],
    fluid=True
)

# Functions
def prepare_data_for_model(patient_data):
    expected_dtypes = {
        "age": int,
        "sex": int,
        "cp": int,
        "trestbps": int,
        "chol": int,
        "fbs": int,
        "restecg": int,
        "thalach": int,
        "exang": int,
        "oldpeak": float,
        "slope": int,
        "ca": int,
        "thal": int
    }
    for key, expected_type in expected_dtypes.items():
        if key in patient_data:
            patient_data[key] = expected_type(patient_data[key])
    return pd.DataFrame([patient_data])

def get_response(df, prediction, column_info):
    customer_data_str = "\n".join([f"{col}: {df.iloc[0][col]}" for col in df.columns])
    prompt = f"""
    Please explain why the individual with the following data is predicted to have a heart attack or not 
    (Prediction: {'Heart Attack' if prediction == 1 else 'No Heart Attack'}):
    {customer_data_str}
    Column Information:
    {column_info}
    Act like a doctor and give kind advice based on the data.
    """
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    return f"Error: {response.status_code}, {response.text}"

# Callback
@app.callback(
    Output('chatbot-output', 'children'),
    Input('send-button', 'n_clicks'),
    State('chatbot-input', 'value'),
    State('chatbot-output', 'children'),
    prevent_initial_call=True
)
def update_chat(n_clicks, user_input, chat_history):
    if not chat_history:
        chat_history = []

    if user_input:
        # Mock chat behavior
        chat_history.append(html.Div(f"You: {user_input}", style={'text-align': 'right', 'padding': '5px'}))

        try:
            # Collect and validate patient data step by step
            patient_data = {}
            question_index = len(chat_history) // 2
            if question_index < len(questions):
                question = questions[question_index]
                chat_history.append(html.Div(f"Bot: {question}", style={'text-align': 'left', 'padding': '5px'}))
                # Simulate collecting patient data
                if question_index == 0:
                    try:
                        age = int(user_input)
                        if age < 0 or age > 95:  # Validation for realistic age
                            return "Please enter a valid age between 0 and 95."
                        patient_data["age"] = age
                    except ValueError:
                        return "Please enter a valid age (a number)."
                elif question_index == 1:
                    if "male" in user_input.lower() or "man" in user_input.lower():
                        patient_data["sex"] = 1
                    elif "female" in user_input.lower() or "woman" in user_input.lower():
                        patient_data["sex"] = 0
                    else:
                        return "I didn't quite catch that. Could you please specify if the patient is Male or Female?"

                elif question_index == 2:
                    try:
                        cp = int(user_input)
                        if cp not in [0, 1, 2, 3]:
                            return "Please enter a valid chest pain type (0, 1, 2, 3)."
                        patient_data["cp"] = cp
                    except ValueError:
                        return "Please enter a valid chest pain type (0, 1, 2, 3)."

                elif question_index == 3:
                    try:
                        trestbps = int(user_input)
                        if trestbps < 50 or trestbps > 200:  # Validation for plausible range
                            return "Please enter a valid resting blood pressure (50-200 mm Hg)."
                        patient_data["trestbps"] = trestbps
                    except ValueError:
                        return "Please enter a valid resting blood pressure (a number)."

                elif question_index == 4:
                    try:
                        chol = int(user_input)
                        if chol < 100 or chol > 400:  # Validation for plausible range
                            return "Please enter a valid cholesterol level (100-400 mg/dl)."
                        patient_data["chol"] = chol
                    except ValueError:
                        return "Please enter a valid cholesterol level (a number)."

                elif question_index == 5:
                    try:
                        fbs_value = int(user_input)
                        if fbs_value > 120:
                            patient_data["fbs"] = 1
                        elif fbs_value <= 120:
                            patient_data["fbs"] = 0
                        else:
                            return "Please enter a valid number for your fasting blood sugar level."
                    except ValueError:
                        return "Please enter a valid number for your fasting blood sugar level."

                elif question_index == 6:
                    try:
                        restecg = int(user_input)
                        if restecg not in [0, 1, 2]:
                            return "Please enter a valid ECG result (0, 1, or 2)."
                        patient_data["restecg"] = restecg
                    except ValueError:
                        return "Please enter a valid ECG result (0, 1, or 2)."

                elif question_index == 7:
                    try:
                        thalach = int(user_input)
                        if thalach < 60 or thalach > 220:  # Validation for plausible range
                            return "Please enter a valid maximum heart rate (60-220)."
                        patient_data["thalach"] = thalach
                    except ValueError:
                        return "Please enter a valid maximum heart rate (a number)."

                elif question_index == 8:
                    if user_input.strip().lower() in ["yes", "1"]:
                        patient_data["exang"] = 1
                    elif user_input.strip().lower() in ["no", "0"]:
                        patient_data["exang"] = 0
                    else:
                        return "Please answer with 'Yes', 'No', or 1 for Yes and 0 for No."

                elif question_index == 9:
                    try:
                        oldpeak = float(user_input)  # Convert input to float (handles int or float)
                        if oldpeak < 0 or oldpeak > 5:  # Validate that it's within the range
                            return "Please enter a valid ST depression value (0-5)."
                        patient_data["oldpeak"] = oldpeak
                    except ValueError:
                        return "Please enter a valid ST depression value (a number between 0 and 5)."

                elif question_index == 10:
                    try:
                        slope = int(user_input)
                        if slope not in [0, 1, 2]:
                            return "Please enter a valid slope (0, 1, or 2)."
                        patient_data["slope"] = slope
                    except ValueError:
                        return "Please enter a valid slope (0, 1, or 2)."

                elif question_index == 11:
                    try:
                        ca = int(user_input)
                        if ca < 0 or ca > 3:  # Validation for plausible range
                            return "Please enter a valid number of major vessels (0-3)."
                        patient_data["ca"] = ca
                    except ValueError:
                        return "Please enter a valid number of major vessels (0-3)."

                elif question_index == 12:
                    try:
                        thal = int(user_input)
                        if thal not in [0, 1, 2]:
                            return "Please enter a valid blood flow problem (0, 1, or 2)."
                        patient_data["thal"] = thal
                    except ValueError:
                        return "Please enter a valid blood flow problem (0, 1, or 2)."

                # Once data is complete, run prediction
                if question_index == len(questions) - 1:
                    df = prepare_data_for_model(patient_data)
                    prediction = model.predict(df)[0]
                    explanation = get_response(df, prediction, column_info)
                    chat_history.append(html.Div(f"Bot: {explanation}", style={'text-align': 'left', 'padding': '5px'}))
            else:
                chat_history.append(html.Div("Bot: Thank you! Your data has been collected.", style={'text-align': 'left', 'padding': '5px'}))

        except Exception as e:
            chat_history.append(html.Div(f"Bot: Error processing your input. {str(e)}", style={'text-align': 'left', 'padding': '5px'}))

    return chat_history




################################################

# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_groq import ChatGroq
# import pandas as pd
# from groq import Groq
# import joblib
# import numpy as np
# import requests

# # Load environment variables
# load_dotenv()

# # Set the API key environment variable globally
# os.environ["GROQ_API_KEY"] = "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"

# # Paths
# working_dir = os.path.dirname(os.path.abspath(__file__))
# # Load the heart attack classification model
# model = joblib.load('best_model.joblib')

# # Define the list of questions for heart attack classification
# questions = [
#     "How old are you? (Please enter your age in numbers)",
#     "What is your gender? (Male or Female)",
#     "What is your Chest Pain Type? (0: Typical Angina 'Most serious', 1: Atypical Angina, 2: Non-anginal pain, 3: Asymptomatic 'Least serious')",
#     "What is the ST depression induced by exercise relative to rest? (Enter a number from 0 to 5)",
# ]

# # Update column information for heart attack classification
# column_info = {
#     "age": "Age of the individual.",
#     "sex": "Gender of the individual (1 = male, 0 = female).",
#     "cp": "Chest Pain Type: 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal pain, 3 = Asymptomatic.",
#     "oldpeak": "ST depression induced by exercise relative to rest."
# }

# # Function to prepare the patient data for the model
# def prepare_data_for_model(patient_data):

#     # Expected data types
#     expected_dtypes = {"age": int,"sex": int,"cp": int, "oldpeak": float}

#     # Validate and convert types
#     for key, expected_type in expected_dtypes.items():
#         if key in patient_data:
#             if not isinstance(patient_data[key], expected_type):
#                 try:
#                     patient_data[key] = expected_type(patient_data[key])
#                 except (ValueError, TypeError):
#                     raise ValueError(f"Invalid value for '{key}': Expected {expected_type.__name__}, got {type(patient_data[key]).__name__} with value {patient_data[key]}.")
#     processed_data = pd.DataFrame([patient_data])
#     return processed_data
# def get_response(df, prediction, column_info):
#     if df.empty:
#         return "No data available"
#     # Format the customer data as specified
#     def format_customer_data(df):
#         # Assuming the first row contains the necessary data
#         customer_data = df.iloc[0]
#         return "\n".join([f"{col.replace('_', ' ').title()}: {customer_data[col]}" for col in df.columns])

#     # Create formatted string from the DataFrame
#     customer_data_str = format_customer_data(df)

#     # Create the prompt using the formatted data string
#     prompt = f"""
#     Please explain why the individual with the following data is predicted to have a heart attack or not (Prediction: {'Heart Attack' if prediction == 1 else 'No Heart Attack'}):
#     Collected Patient Data:
#     {customer_data_str}
#     Column Information:
#     {column_info}
#     Act like a doctor and your explanation should be based on the data provided and how it affects the heart attack prediction and give your patient kindly some advices based on the data.
#     """
#     # Set API URL and headers for Groq
#     api_url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
#         "Content-Type": "application/json"
#     }
#     # Prepare the data for the request
#     payload = {
#         "model": "llama3-8b-8192",  # or whatever model you're using
#         "messages": [{"role": "user", "content": prompt}]
#     }
#     # Make the POST request to the API
#     response = requests.post(api_url, json=payload, headers=headers)

#     if response.status_code == 200:
#         return response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()
#     else:
#         return f"Error: {response.status_code}, {response.text}"

# # Function to create the conversational chain
# def create_chain():
#     llm = ChatGroq(
#         model_name="llama3-70b-8192",
#         temperature=0
#     )

#     # Define the prompt template for interacting with the medical assistant
#     interaction_prompt = PromptTemplate(
#         input_variables=["chat_history", "question"],
#         template=(
#             "Welcome to the medical assistant! I will be helping you collect information "
#             "to classify the heart attack risk for a patient.\n\n"
#             "Let's start by gathering some details about the patient.\n\n"
#             "Chat History:\n{chat_history}\n\n"
#             "Please answer the following question:\n{question}\n\n"
#             "I will verify the answer and make sure it is correct. After that, we will proceed to the next question.\n\n"
#         )
#     )
#     # Create the conversational chain with the assistant
#     doc_chain = LLMChain(
#         llm=llm,
#         prompt=interaction_prompt
#     )
#     return doc_chain

# # Streamlit App Configuration
# st.set_page_config(
#     page_title="Heart Attack Risk Assessment 🤖",
#     page_icon="❤",
#     layout="centered"
# )
# st.title("Medical Assistant for Heart Attack Classification")

# # Display default introductory message
# st.markdown(""" 
#     Welcome to the Medical Assistant for Heart Attack Classification!  
#     This chatbot is designed to help you gather essential information about a patient to predict their risk of heart attack.  
#     I will ask you a few questions to collect details about the patient, such as their age, gender, cholesterol levels, and other factors that may influence the prediction.  
#     Please provide accurate answers to each question to help improve the heart attack classification model.

#     Let's begin by gathering some basic information.
# """)

# # Create conversational chain
# if "conversation_chain" not in st.session_state:
#     st.session_state.conversation_chain = create_chain()

# # Initialize chat history in Streamlit session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Initialize the question index to keep track of where we are in the list
# if "question_index" not in st.session_state:
#     st.session_state.question_index = 0

# # Initialize patient info in session state (only once)
# if "patient_data" not in st.session_state:
#     st.session_state.patient_data = {}

# # Function to export patient_data to a DataFrame
# def export_patient_data_to_dataframe():
#     patient_data = st.session_state.patient_data
#     df = pd.DataFrame([patient_data])
#     return df

# # Function to handle the input and store the answer
# def handle_user_input(user_input):
#     if not user_input:  # Check if the input is empty or None
#         return "I didn't receive any input. Could you please provide an answer?"

#     current_question = questions[st.session_state.question_index]
#     patient_data = st.session_state.patient_data  # Accessing the session state's patient_data

#     # Normalize and verify the user's input for each question
#     if current_question == questions[0]:  # Age question
#         try:
#             age = int(user_input)
#             if age < 0 or age > 95:  # Validation for realistic age
#                 return "Please enter a valid age between 0 and 95."
#             patient_data["age"] = age
#         except ValueError:
#             return "Please enter a valid age (a number)."
        
#     elif current_question == questions[1]:  # Gender question
#         if "male" in user_input.lower() or "man" in user_input.lower():
#             patient_data["sex"] = 1
#         elif "female" in user_input.lower() or "woman" in user_input.lower():
#             patient_data["sex"] = 0
#         else:
#             return "I didn't quite catch that. Could you please specify if the patient is Male or Female?"

#     elif current_question == questions[2]:  # Chest pain type
#         try:
#             cp = int(user_input)
#             if cp not in [0, 1, 2, 3]:
#                 return "Please enter a valid chest pain type (0, 1, 2, 3)."
#             patient_data["cp"] = cp
#         except ValueError:
#             return "Please enter a valid chest pain type (0, 1, 2, 3)."

#     elif current_question == questions[3]:  # ST depression
#         try:
#             oldpeak = float(user_input)  # Convert input to float (handles int or float)
#             if oldpeak < 0 or oldpeak > 5:  # Validate that it's within the range
#                 return "Please enter a valid ST depression value (0-5)."
#             patient_data["oldpeak"] = oldpeak
#         except ValueError:
#             return "Please enter a valid ST depression value (a number between 0 and 5)."
#     # Proceed to the next question after handling the current answer
#     if st.session_state.question_index < len(questions) - 1:
#         st.session_state.question_index += 1
#         next_question = questions[st.session_state.question_index]
#         return next_question
#     else:
#         df = export_patient_data_to_dataframe()
#         df.to_csv('patient_data.csv', index=False)
#         patient_data_dict = df.iloc[0].to_dict()
#         # Call your prepare_data_for_model function
#         processed_data = prepare_data_for_model(patient_data_dict)
#         # Display the processed data
#         st.write("Thank you! We have collected all the necessary information.")
#         st.write("Here is the information you provided:")
#         st.dataframe(df)
#         prediction = model.predict(processed_data)
#         # Step 3: Display the prediction
#         st.write(f"The model predicts the patient is {'at high risk of heart attack' if prediction == 1 else 'at low risk of heart attack'}.")
#         # step 4: LLM Generate Explanation
#         explanation = get_response(df, prediction, column_info)
#         st.write("Explanation of the prediction:")
#         st.write(explanation)
#         st.write("If You want a help again please refresh the page.")
#         return "Have a great day!"
# # Automatically prompt the first question after intro
# if len(st.session_state.chat_history) == 0:
#     first_question = questions[st.session_state.question_index]
#     st.session_state.chat_history.append({"role": "assistant", "content": first_question})

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input and response
# user_input = st.chat_input("Please provide the information to start.")
# if user_input:
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)
#     with st.chat_message("assistant"):
#         try:
#             # Process user input and determine the next step
#             response = handle_user_input(user_input)
#             st.markdown(response)
#             st.session_state.chat_history.append({"role": "assistant", "content": response})
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
