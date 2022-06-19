"""
Unit testing for ML model
"""
import pandas as pd
import pytest
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data import process_data

DATA_PATH = 'data/clean_census.csv'
MODEL_PATH = 'model/model.pkl'

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

@pytest.fixture()
def data():
    return pd.read_csv(DATA_PATH)

def test_data(data):
    assert data.shape[0] > 0
    assert data.shape[1] > 0

def test_process_data(data):
    train, test = train_test_split(data, test_size=0.3)
    X, y, _, _ = process_data(
        train, cat_features, label='salary'
    )
    assert len(X) == len(y)
    
def test_model():
    model = joblib.load("model/model.pkl")
    assert isinstance(model, RandomForestClassifier)