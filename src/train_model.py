"""
Script to train a machine learning model
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import joblib

from utils import train_model, compute_model_metrics, inference, process_data


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Get data
logging.info("Getting data")
data = pd.read_csv('data/clean_census.csv')

# Preprocess data
train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
logging.info("Training model")
model = train_model(X_train, y_train)

# Scoring
logging.info("Scoring on test set")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

# Save artifacts
logging.info("Saving artifacts")
joblib.dump(model, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')