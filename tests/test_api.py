import pytest
from fastapi.testclient import TestClient

from polycast.server import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYCAST_DB", str(tmp_path / "polycast.db"))
    monkeypatch.setenv("POLYCAST_AUDIT_DIR", str(tmp_path / "runs"))
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_forecast_endpoint_returns_a_probability(client):
    resp = client.post(
        "/forecast",
        json={
            "text": "Will the central bank cut the policy rate at the next scheduled meeting?",
            "resolution_criteria": "Resolved yes if the published decision confirms a reduction.",
            "resolve_by": "2026-12-18T00:00:00Z",
            "horizon": "months",
            "reference_class": "scheduled policy meetings",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 < body["probability"] < 1.0
    assert body["lower"] <= body["probability"] <= body["upper"]
    assert len(body["views"]) == 3


def test_forecast_endpoint_rejects_unresolvable_questions(client):
    resp = client.post(
        "/forecast",
        json={
            "text": "will it",
            "resolution_criteria": "yes or no",
            "resolve_by": "2026-12-18T00:00:00Z",
        },
    )
    assert resp.status_code == 422


def test_forecast_can_be_fetched_back(client):
    created = client.post(
        "/forecast",
        json={
            "text": "Will unemployment claims decline for a fifth consecutive week?",
            "resolution_criteria": "Resolved yes if the published count is below the prior week.",
            "resolve_by": "2026-12-31T00:00:00Z",
        },
    ).json()
    fetched = client.get(f"/forecast/{created['forecast_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["forecast_id"] == created["forecast_id"]


def test_unknown_forecast_returns_404(client):
    assert client.get("/forecast/does-not-exist").status_code == 404


def test_metrics_reports_provider_mode(client):
    body = client.get("/metrics").json()
    assert body["provider"] in {"hosted", "offline-stub"}
    assert "forecasts_stored" in body


def test_simulate_endpoint_runs(client):
    resp = client.post("/simulate", json={"n_agents": 20, "rounds": 30, "seed": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["mean_series"]) == 30
    assert body["truth"] == 0.7
