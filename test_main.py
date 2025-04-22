import api
from fastapi.testclient import TestClient

client = TestClient(api.app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_sample_analyze():
    response = client.get("/analyze", params={"q": "Estos tíos de Netflix van a quemar una idea"})
    assert response.status_code == 200
    assert response.json()["probas"]["NEG"] > 0.90

