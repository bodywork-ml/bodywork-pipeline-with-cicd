"""Simple tests for the web service hosting the prediction endpoint."""
from pipeline.serve_model import app


api_test_client = app.test_client()


def test_service_returns_prediction_for_valid_feature_data():
    prediction_request_data = {
        'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2
    }
    resposne = api_test_client.post('/iris/v1/score', json=prediction_request_data)
    assert resposne.status_code == 200
    assert resposne.json["species_prediction"] == "setosa"
