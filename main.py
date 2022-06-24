from concurrent.futures import process
import os
import uvicorn
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

from src.utils import inference, process_data


class User(BaseModel):
    age: int = Field(example=35)
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay'
    ] = Field(example='State-gov')
    fnlgt: int = Field(example=20000)
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm',
        '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school', '5th-6th', '10th'
        'Preschool', '12th', '1st-4th'
    ] = Field(example='Bachelors')
    education_num: int = Field(example=9)
    marital_status: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Separated', 'Married-AF-spouse', 'Widowed'
    ] = Field(example='Widowed')
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty',
        'Other-service', 'Sales', 'Transport-moving', 'Farming-fishing',
        'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv',
        'Armed-Forces', 'Priv-house-serv'
    ] = Field(example='Adm-clerical')
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'
    ] = Field(example='Husband')
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
    ] = Field(example='Black')
    sex: Literal[
        'Male', 'Female'
    ] = Field(example='Male')
    capital_gain: int = Field(example=0)
    capital_loss: int = Field(example=20)
    hours_per_week: int = Field(example=40)
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto-Rico',
        'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Poland',
        'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti',
        'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece',
        'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands'
    ] = Field(example='United-States')


# Set up DVC in Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -r awsremote") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Create the application
app = FastAPI()


# Define GET function
@app.get("/")
async def get_items():
    return {"message": "Hello User! This is an app to predict if someone's income will exceed $50,000/year."}


# Define POST function
@app.post("/predict")
async def predict(input: User):
    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    # Load model, encoder and lb
    classifier = joblib.load("model/model.pkl")
    encoder = joblib.load("model/encoder.pkl")
    lb = joblib.load("model/lb.pkl")

    # Compiile input data
    input = np.array([[
        0,
        input.age,
        input.workclass,
        input.fnlgt,
        input.education,
        input.education_num,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.capital_gain,
        input.capital_loss,
        input.hours_per_week,
        input.native_country
    ]])

    # Convert to pandas dataframe
    df = pd.DataFrame(data=input, columns=[
        "",
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
    ])

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False
    )

    y_pred = inference(classifier, X)
    y = lb.inverse_transform(y_pred)[0]
    return {"prediction": y}