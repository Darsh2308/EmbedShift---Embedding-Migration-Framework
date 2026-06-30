"""Tests for the health endpoint."""

from app import __version__


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert "app" in body
    assert "environment" in body
