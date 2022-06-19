import requests


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

app_url = "https://.herokuapp.com/predict"

result = requests.post(app_url, json=data)
assert result.status_code == 200
