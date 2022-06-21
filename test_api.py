from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_function():
    result = client.get("/")
    try:
        assert result.status_code == 200
        assert result.json() == {
            "message": "Hello User! This is an app to predict if someone's income will exceed $50,000/year."
        }
    except AssertionError as err:
        print("Error with GET function")
        raise err


def test_post_predict_above_50k():
    result = client.post("/predict", json={
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    })

    assert result.status_code == 200
    assert result.json() == {"prediction": ">50K"}


def test_post_predict_below_50k():
    result = client.post("/predict", json={
        "age": 37,
        "workclass": "Private",
        "fnlgt": 284582,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert result.status_code == 200
    assert result.json() == {"prediction": "<=50K"}
    