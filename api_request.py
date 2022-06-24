import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

data = {
    "age": 29,
    "workclass": "Private",
    "fnlgt": 410924,
    "education": "Bachelors",
    "education_num": 9,
    "marital_status": "Married-civ-spouse",
    "occupation": "Tech-support",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
}

app_url = "https://predict-yo-income.herokuapp.com/predict"

result = requests.post(app_url, json=data)
assert result.status_code == 200

logging.info("Testing Heroku app")
logging.info(f"Status code: {result.status_code}")
logging.info(f"Response body: {result.json()}")
