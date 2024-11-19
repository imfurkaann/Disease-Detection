import gradio as gr
import pickle
import pandas as pd
import numpy as np
import sys
import os
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time
from sklearn.preprocessing import StandardScaler

class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

seafoam = Seafoam()

def predict_heart_disease(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, 
                         RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    
    with open('data_processing/models/model_heart.pkl', 'rb') as f:
        heart_model = pickle.load(f)
        
    data = {
        'Age': [float(Age)],
        'Sex': [1 if Sex == 'Male' else 0],
        'RestingBP': [float(RestingBP)],
        'Cholesterol': [float(Cholesterol)],
        'FastingBS': [1 if FastingBS == 'Yes' else 0],
        'MaxHR': [float(MaxHR)],
        'ExerciseAngina': [1 if ExerciseAngina == 'Yes' else 0],
        'Oldpeak': [float(Oldpeak)],
        'ChestPainType_ASY': [1 if ChestPainType == 'ASY' else 0],
        'ChestPainType_ATA': [1 if ChestPainType == 'ATA' else 0],
        'ChestPainType_NAP': [1 if ChestPainType == 'NAP' else 0],
        'ChestPainType_TA': [1 if ChestPainType == 'TA' else 0],
        'ST_Slope_Down': [1 if ST_Slope == 'Down' else 0],
        'ST_Slope_Flat': [1 if ST_Slope == 'Flat' else 0],
        'ST_Slope_Up': [1 if ST_Slope == 'Up' else 0],
        'RestingECG_LVH': [1 if RestingECG == 'LVH' else 0],
        'RestingECG_Normal': [1 if RestingECG == 'Normal' else 0],
        'RestingECG_ST': [1 if RestingECG == 'ST' else 0]
    }
    df_input = pd.DataFrame(data)
    prediction = heart_model.predict(df_input)
    result = "Hasta" if prediction[0] == 1 else "Sağlıklı"
    html_result = f"<div class='result-box {result.lower()}'>{result}</div>"
    return html_result

def predict_bodyfat(Age,Weight,Height,Neck,Chest,Abdomen,Hip,Thigh,Knee,Ankle,Biceps,Forearm,Wrist):
    
    with open('data_processing/models/model_bodyfat.pkl', 'rb') as f:
        bodyfat_model = pickle.load(f)
        
    scaler = StandardScaler()

    data =  {
        "Age" :Age,
        "Weight":Weight,
        "Height":Height,
        "Neck":Neck,
        "Chest":Chest,
        "Abdomen":Abdomen,
        "Hip":Hip,
        "Thigh":Thigh,
        "Knee":Knee,
        "Ankle":Ankle,
        "Biceps":Biceps,
        "Forearm":Forearm,
        "Wrist":Wrist       
    }
    
    df_input = pd.DataFrame(data, index=[0])
    df_input["BMI"] = df_input["Weight"] / (df_input["Height"])**2
    df_input['BF_BMI'] = df_input['BMI'] * 1.39 + df_input['Age'] * 0.16 - 19.34    
    df_input["Obesite"] = np.where(df_input["BMI"] > 30, 1, 0)
    
    prediction = bodyfat_model.predict(df_input)
      
    result = prediction[0] 
    html_result = f"<div class='result-box {result}'>{result}</div>"
    return html_result

def predict_ashtma(Tiredness,Dry_Cough,Difficulty_in_Breathing,Sore_Throat,None_Sympton,Pains,Nasal_Congestion,Runny_Nose,None_Experiencing,Age,Gender):
    
    with open('data_processing/models/model_ashtma.pkl', 'rb') as f:
        ashtma_model = pickle.load(f)
    
    data = {
        
        "Tiredness" : 1 if Tiredness == "Yes" else 0,
        "Dry-Cough" :1 if Dry_Cough == "Yes" else 0,
        "Difficulty-in-Breathing" : 1 if Difficulty_in_Breathing == "Yes" else 0,
        "Sore-Throat" : 1 if Sore_Throat == "Yes" else 0,
        "None_Sympton" : 1 if None_Sympton == "Yes" else 0,
        "Pains" :  1 if Pains == "Yes" else 0,
        "Nasal-Congestion" :  1 if Nasal_Congestion == "Yes" else 0,
        "Runny-Nose" : 1 if Runny_Nose == "Yes" else 0,
        "None_Experiencing" : 1 if None_Experiencing == "Yes" else 0,
        "Age_0-9" :  1 if 0 <= Age <= 9 else 0,
        "Age_10-19" : 1 if 10 <= Age <= 19 else 0,
        "Age_20-24" : 1 if 20 <= Age <= 24 else 0,
        "Age_25-59" : 1 if 25 <= Age <= 59 else 0,
        "Age_60+" : 1 if 60 <= Age else 0,
        "Gender_Female" : 1 if Gender == "Female" else 0,
        "Gender_Male" : 1 if Gender == "Male" else 0
    }
    
    df_input = pd.DataFrame(data, index=[0])
    print(df_input)

    prediction = ashtma_model.predict(df_input)
    result = "Hasta" if prediction[0] == 1 else "Sağlıklı"
    html_result = f"<div class='result-box {result.lower()}'>{result}</div>"
    return html_result

with gr.Blocks(theme=seafoam) as demo:
    gr.HTML(
        """
        <style>
        body {
            margin: 0;
            padding: 0;
            background-color: #1a1c1f;
        }
        .gradio-container {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            max-width: none !important;
            background-color: #1a1c1f;
        }
        #main-container {
            border: none;
            border-radius: 0;
            box-shadow: none;
            background-color: #1a1c1f;
            padding: 0;
            margin: 0;
            min-height: 100vh;
        }
        .tabs {
            background-color: #1a1c1f;
            border: none;
            box-shadow: none;
        }
        .tab-nav {
            background-color: #1a1c1f;
            border: none;
            padding: 1rem 0;
        }
        #centered-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1a1c1f;
            border: none;
        }
        .form-container {
            background-color: #2d3339;
            padding: 2rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .result-box {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            margin: 20px auto;
            border-radius: 10px;
            max-width: 300px;
            color: white;
        }
        .hasta {
            background-color: #dc3545;
        }
        .sağlıklı {
            background-color: #28a745;
        }
        .gr-button {
            min-width: 200px;
            margin: 10px auto;
            display: block;
            background-color: #3d4752;
            border: none;
        }
        .gr-button:hover {
            background-color: #4a5562;
        }
        .gr-form {
            border: none;
            background-color: transparent;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        /* Input styling */
        .gr-input, .gr-dropdown, .gr-slider {
            background-color: #3d4752 !important;
            border: none !important;
            color: white !important;
        }
        .gr-input:focus, .gr-dropdown:focus {
            border: 1px solid #4a5562 !important;
        }
        /* Label styling */
        label {
            color: #e0e0e0 !important;
        }
        /* Markdown styling */
        .markdown-text {
            color: #e0e0e0 !important;
            text-align: center;
        }
        </style>
        """
    )

    with gr.Tabs() as tabs:
        # Heart Disease Tab
        with gr.Tab("Heart Disease Prediction"):
            with gr.Column(elem_id="centered-container"):
                gr.Markdown("# Heart Disease Prediction", elem_classes="markdown-text")
                
                with gr.Group(elem_classes="form-container"):
                    heart_age = gr.Number(label="Age")
                    heart_sex = gr.Dropdown(choices=["Male", "Female"], label="Sex")
                    heart_chest_pain = gr.Dropdown(choices=["ATA", "ASY", "NAP", "TA"], label="Chest Pain Type")
                    heart_resting_bp = gr.Number(label="Resting Blood Pressure")
                    heart_cholesterol = gr.Number(label="Cholesterol")
                    heart_fasting_bs = gr.Dropdown(choices=["Yes", "No"], label="Fasting Blood Sugar")
                    heart_resting_ecg = gr.Dropdown(choices=["LVH", "Normal", "ST"], label="Resting ECG")
                    heart_max_hr = gr.Number(label="Max Heart Rate")
                    heart_exercise_angina = gr.Dropdown(choices=["Yes", "No"], label="Exercise Angina")
                    heart_oldpeak = gr.Number(label="ST Depression")
                    heart_st_slope = gr.Dropdown(choices=["Up", "Flat", "Down"], label="ST Slope")
                    
                    heart_submit = gr.Button("Predict", size="lg")
                    heart_output = gr.HTML(label="Result")

        # BodyFat Tab
        with gr.Tab("BodyFat Prediction"):
            with gr.Column(elem_id="centered-container"):
                gr.Markdown("# BodyFat Prediction", elem_classes="markdown-text")
                with gr.Group(elem_classes="form-container"):
                    
                    bodyfat_Age = gr.Number(label="Age")
                    bodyfat_Weight = gr.Number(label="Weight")
                    bodyfat_Height = gr.Number(label="Height")
                    bodyfat_Neck = gr.Number(label="Neck")
                    bodyfat_Chest = gr.Number(label="Chest")
                    bodyfat_Abdomen = gr.Number(label="Abdomen")
                    bodyfat_Hip = gr.Number(label="Hip")
                    bodyfat_Thigh = gr.Number(label="Thigh")
                    bodyfat_Knee = gr.Number(label="Knee")
                    bodyfat_Ankle = gr.Number(label="Ankle")
                    bodyfat_Biceps = gr.Number(label="Biceps")
                    bodyfat_Forearm = gr.Number(label="Forearm")
                    bodyfat_Wrist = gr.Number(label="Wrist")

                    
                    bodyfat_submit = gr.Button("Predict", size="lg")
                    bodyfat_output = gr.HTML(label="Result")

        # Ashtma Tab
        with gr.Tab("Ashtma Prediction"):
            with gr.Column(elem_id="centered-container"):
                gr.Markdown("# Ashtma Prediction", elem_classes="markdown-text")
                
                with gr.Group(elem_classes="form-container"):
                    
                    ashtma_Tiredness = gr.Dropdown(choices=["Yes", "No"], label="Tiredness")
                    ashtma_Dry_Cough = gr.Dropdown(choices=["Yes", "No"], label="Dry_Cough")
                    ashtma_Difficulty_in_Breathing = gr.Dropdown(choices=["Yes", "No"], label="Difficulty_in_Breathing")
                    ashtma_Sore_Throat = gr.Dropdown(choices=["Yes", "No"], label="Sore_Throat")
                    ashtma_None_Sympton = gr.Dropdown(choices=["Yes", "No"], label="None_Sympton")
                    ashtma_Pains = gr.Dropdown(choices=["Yes", "No"], label="Pains")
                    ashtma_Nasal_Congestion = gr.Dropdown(choices=["Yes", "No"], label="Nasal_Congestion")
                    ashtma_Runny_Nose = gr.Dropdown(choices=["Yes", "No"], label="Runny_Nose")
                    ashtma_None_Experiencing = gr.Dropdown(choices=["Yes", "No"], label="None_Experiencing")
                    ashtma_Age = gr.Number(label="Age")
                    ashtma_Gender = gr.Dropdown(choices=["Male", "Female"], label="Gender")

                    ashtma_submit = gr.Button("Predict", size="lg")
                    ashtma_output = gr.HTML(label="Result")

    # Event handlers
    heart_submit.click(
        fn=predict_heart_disease,
        inputs=[
            heart_age, heart_sex, heart_chest_pain, heart_resting_bp,
            heart_cholesterol, heart_fasting_bs, heart_resting_ecg,
            heart_max_hr, heart_exercise_angina, heart_oldpeak, heart_st_slope
        ],
        outputs=heart_output
    )
    
    bodyfat_submit.click(
        fn=predict_bodyfat,
        inputs=[
            bodyfat_Age, bodyfat_Weight,
            bodyfat_Height, bodyfat_Neck,bodyfat_Chest,  bodyfat_Abdomen, bodyfat_Hip, bodyfat_Thigh,
            bodyfat_Knee, bodyfat_Ankle, bodyfat_Biceps, bodyfat_Forearm, bodyfat_Wrist
        ],
        outputs=bodyfat_output
    )
    
    ashtma_submit.click(
        fn=predict_ashtma,
        inputs=[
            ashtma_Tiredness, ashtma_Dry_Cough, ashtma_Difficulty_in_Breathing,ashtma_Sore_Throat,
            ashtma_None_Sympton, ashtma_Pains, ashtma_Nasal_Congestion, ashtma_Runny_Nose, ashtma_None_Experiencing,
            ashtma_Age, ashtma_Gender, 
        ],
        outputs=ashtma_output
    )

demo.launch()