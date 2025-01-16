# Heart Attack Classification Project

This repository contains a comprehensive project aimed at classifying heart attack risks and providing insightful tools for understanding and preventing heart attacks. The project includes both a Jupyter Notebook for data analysis and a Plotly Dashboard application with interactive features.

## Project Overview

### 1. Jupyter Notebook: `heart_attack.ipynb`

The notebook focuses on the classification problem and includes the following key steps:

#### Data Preprocessing:
- Cleaning and preparing the dataset for analysis.
- Handling missing values and ensuring data consistency.

#### Exploratory Data Analysis (EDA):
- Visualizing the relationships between features and their impact on heart attack risks.
- Identifying key factors that contribute to heart attacks using visualizations.

#### Model Training and Evaluation:
- Training machine learning models to predict heart attack risks.
- Evaluating model performance using metrics such as accuracy, precision, recall, and F1 score.

### 2. Plotly Dashboard Application

The dashboard is a user-friendly, interactive web application designed to:

#### Main Features:

- **Educational Insights:**
  - Pages dedicated to explaining the dangers of heart attacks.
  - Highlighting the factors that increase the risk of heart attacks.

- **AI Doctor Chatbot:**
  - Provides personalized advice and recommendations based on user inputs, test results, and other health metrics.
  - Collects user reports to assess the likelihood of a heart attack.

- **Global Insights:**
  - Displays worldwide statistics and trends related to heart attack factors.
  - Helps users understand the global burden of heart disease.

## Dataset

The project uses a publicly available dataset on heart disease, which includes features such as:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Levels
- Fasting Blood Sugar
- Resting Electrocardiographic Results
- Maximum Heart Rate Achieved
- Exercise-Induced Angina
- ST Depression Induced by Exercise
- The Slope of the Peak Exercise ST Segment
- Number of Major Vessels Colored by Fluoroscopy
- Thalassemia

This dataset is crucial for understanding and analyzing heart attack risks, and it has been processed extensively to improve the quality of insights.

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/ahmedelmetwally74/heart-attack-classification.git
    ```

2. Open the Jupyter Notebook:
    ```bash
    jupyter notebook heart_attack.ipynb
    ```

3. Launch the dashboard application:
    ```bash
    python index.py
    ```

4. Interact with the dashboard and explore its features.
