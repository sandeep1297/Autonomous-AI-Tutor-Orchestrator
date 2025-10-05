from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "YoLearn" in response.json()["message"]

def test_orchestrate_notes():
    message = "Generate structured notes on protein synthesis and include examples."
    response = client.post(f"/api/orchestrate?message={message}")
    data = response.json()
    assert response.status_code == 200
    assert data["status"] in ["SUCCESS", "FOUND_TOOL"]
    assert "note_maker" in data["tool_name"]
