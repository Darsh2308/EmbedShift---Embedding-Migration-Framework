"""API tests: the full migration runnable purely over HTTP."""

import time

import numpy as np

from app.embedders import register_embedder
from app.stores import load_vectors, save_vectors
from app.stores.synthetic import make_related_spaces


def _setup_corpus(tmp_path, n=1500, d_old=48, d_new=64, noise=0.02, nonlinearity=0.0, seed=0):
    """Write an old-vectors file + register a 'new model' embedder that maps id->new vec."""
    old, new = make_related_spaces(n, d_old, d_new, noise=noise, nonlinearity=nonlinearity, seed=seed)
    ids = [f"doc-{i}" for i in range(n)]
    src = tmp_path / "old.npz"
    save_vectors(src, ids, old)

    id_to_new = {ids[i]: new[i] for i in range(n)}
    name = f"lookup-{seed}-{nonlinearity}"
    register_embedder(name, lambda texts: np.array([id_to_new[t] for t in texts], dtype=np.float32))
    return src, name, ids, old


def _poll(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def _connect(client, tmp_path, src, threshold_collection="corpus_v2"):
    return client.post(
        "/connect",
        json={
            "source_path": str(src),
            "output_dir": str(tmp_path / "out"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
    )


# --------------------------------------------------------------------------- #
# Happy path: connect -> sample -> train -> evaluate -> transform
# --------------------------------------------------------------------------- #
def test_full_migration_over_http(client, tmp_path):
    src, embedder, ids, old = _setup_corpus(tmp_path, n=1500, noise=0.02)

    # connect
    r = _connect(client, tmp_path, src)
    assert r.status_code == 200
    body = r.json()
    sid = body["session_id"]
    assert body["count"] == 1500 and body["dim"] == 48

    # sample
    r = client.post("/sample", json={
        "session_id": sid, "embedder": embedder,
        "sample_fraction": 0.3, "validation_fraction": 0.3,
    })
    assert r.status_code == 200
    s = r.json()
    assert s["sample_size"] == 450 and s["val_size"] == 135
    assert s["d_old"] == 48 and s["d_new"] == 64

    # train
    r = client.post("/train", json={"session_id": sid})
    assert r.status_code == 200
    t = r.json()
    assert t["mapper"]["d_old"] == 48 and t["mapper"]["d_new"] == 64
    assert t["cv_results"]  # cross-validation ran

    # evaluate
    r = client.post("/evaluate", json={"session_id": sid, "k": 10, "confidence_threshold": 0.5})
    assert r.status_code == 200
    e = r.json()
    assert e["passed"] is True
    assert e["report"]["evaluation"]["recall_at_k_mapped"] > e["report"]["evaluation"]["recall_at_k_old"]

    # transform (background job)
    r = client.post("/transform", json={"session_id": sid})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    final = _poll(client, job_id)
    assert final["status"] == "completed"
    assert final["result"]["n_transformed"] == 1500

    out_ids, out_vecs = load_vectors(final["result"]["output_path"])
    assert len(out_ids) == 1500 and out_vecs.shape[1] == 64


# --------------------------------------------------------------------------- #
# Gate failure path
# --------------------------------------------------------------------------- #
def test_transform_blocked_when_gate_fails(client, tmp_path):
    src, embedder, ids, old = _setup_corpus(tmp_path, n=1500, noise=0.05, nonlinearity=0.9, seed=1)
    sid = _connect(client, tmp_path, src).json()["session_id"]
    client.post("/sample", json={"session_id": sid, "embedder": embedder,
                                 "sample_fraction": 0.3, "validation_fraction": 0.3})
    client.post("/train", json={"session_id": sid, "mapper_kind": "linear"})
    r = client.post("/evaluate", json={"session_id": sid, "k": 10, "confidence_threshold": 0.95})
    assert r.json()["passed"] is False

    # transform refused (409) unless forced
    r = client.post("/transform", json={"session_id": sid})
    assert r.status_code == 409

    r = client.post("/transform", json={"session_id": sid, "force": True})
    assert r.status_code == 200
    final = _poll(client, r.json()["job_id"])
    assert final["status"] == "completed"
    assert final["result"]["n_transformed"] == 1500


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def test_connect_missing_file_404(client):
    r = client.post("/connect", json={"source_path": "/no/such/file.npz"})
    assert r.status_code == 404


def test_unknown_session_404(client):
    r = client.post("/train", json={"session_id": "deadbeef"})
    assert r.status_code == 404


def test_unknown_embedder_400(client, tmp_path):
    src, _, _, _ = _setup_corpus(tmp_path, n=200, seed=2)
    sid = _connect(client, tmp_path, src).json()["session_id"]
    r = client.post("/sample", json={"session_id": sid, "embedder": "does-not-exist"})
    assert r.status_code == 400


def test_train_before_sample_conflict(client, tmp_path):
    src, _, _, _ = _setup_corpus(tmp_path, n=200, seed=3)
    sid = _connect(client, tmp_path, src).json()["session_id"]
    r = client.post("/train", json={"session_id": sid})
    assert r.status_code == 409


def test_transform_before_evaluate_conflict(client, tmp_path):
    src, embedder, _, _ = _setup_corpus(tmp_path, n=1500, seed=4)
    sid = _connect(client, tmp_path, src).json()["session_id"]
    client.post("/sample", json={"session_id": sid, "embedder": embedder,
                                 "sample_fraction": 0.3, "validation_fraction": 0.3})
    client.post("/train", json={"session_id": sid})
    r = client.post("/transform", json={"session_id": sid})
    assert r.status_code == 409  # must evaluate first


def test_embedders_endpoint_lists_builtin(client):
    r = client.get("/embedders")
    assert r.status_code == 200
    assert "hashing" in r.json()["embedders"]
