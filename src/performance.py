"""
Function for outputing the performance of the model on slices of data    
"""
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from utils import process_data, inference, compute_model_metrics


DATA_PATH = "data/clean_census.csv"
MODEL_PATH = "model/model.pkl"
ENCODER_PATH = "model/encoder.pkl"
LB_PATH = "model/lb.pkl"

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

def get_sliced_performance_metrics():
    # Load data
    data = pd.read_csv(DATA_PATH)

    # Load model, encoder and lb
    model = joblib.load(MODEL_PATH)
    
    encoder = joblib.load(ENCODER_PATH)

    lb = joblib.load(LB_PATH)

    # Get test set
    _, test = train_test_split(data, test_size=0.2)

    # Slice data and get performance metrics
    for feature in cat_features:
        for entry in test[feature].unique():
            temp_df = test[test[feature] == entry]
            X_test, y_test, _, _ = process_data(
                temp_df, cat_features, label="salary", training=False,
                encoder=encoder, lb=lb
            )
            y_pred = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            print(f"{feature}: {entry}; Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}\n")
            with open("sliced_metrics/slice_output.txt", 'a') as file:
                file.write(f"{feature} = {entry}; Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}\n")


if __name__ == "__main__":
    get_sliced_performance_metrics()