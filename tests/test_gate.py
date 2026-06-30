"""Tests for the confidence gate and report."""


from app.core.gate import build_report, evaluate_and_gate, run_gate
from app.core.mapper import LinearMapper
from app.models.evaluation import ConfidenceReport, EvaluationResult, MapperInfo
from app.stores.synthetic import make_related_spaces


def _result(mapped, old, max_=1.0, quality=None):
    return EvaluationResult(
        k=10,
        n_queries=100,
        n_corpus=500,
        recall_at_k_max=max_,
        recall_at_k_mapped=mapped,
        recall_at_k_old=old,
        quality_retained=quality if quality is not None else mapped / max_,
        mean_cosine_mapped_vs_true=0.95,
        mapper=MapperInfo(kind="linear", d_old=48, d_new=64, lambda_=1.0, normalize_output=True),
    )


# --------------------------------------------------------------------------- #
# Gate logic
# --------------------------------------------------------------------------- #
def test_gate_passes_when_good():
    v = run_gate(_result(mapped=0.92, old=0.10), threshold=0.90)
    assert v.passed is True
    assert v.beats_do_nothing is True
    assert "PROCEED" in v.recommendation


def test_gate_fails_below_threshold():
    v = run_gate(_result(mapped=0.80, old=0.10), threshold=0.90)
    assert v.passed is False
    assert v.beats_do_nothing is True
    assert "below" in v.recommendation.lower()


def test_gate_fails_when_not_beating_do_nothing():
    v = run_gate(_result(mapped=0.30, old=0.35), threshold=0.10)
    assert v.passed is False
    assert v.beats_do_nothing is False
    assert "old model" in v.recommendation.lower() or "dissimilar" in v.recommendation.lower()


def test_gate_uses_recall_not_cosine():
    """High cosine must NOT rescue a low-recall mapper."""
    r = _result(mapped=0.40, old=0.10)
    r.mean_cosine_mapped_vs_true = 0.999  # looks great on cosine
    v = run_gate(r, threshold=0.90)
    assert v.passed is False  # recall-based gate still fails


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def test_build_report_and_text():
    report = build_report(_result(mapped=0.93, old=0.08), threshold=0.90)
    assert isinstance(report, ConfidenceReport)
    text = report.to_text()
    assert "CONFIDENCE REPORT" in text
    assert "QUALITY RETAINED" in text
    assert "PASS" in text


def test_report_save_load_roundtrip(tmp_path):
    report = build_report(_result(mapped=0.93, old=0.08), threshold=0.90)
    path = report.save(tmp_path / "report")  # .json appended
    assert path.exists()
    loaded = ConfidenceReport.load(path)
    assert loaded.verdict.passed == report.verdict.passed
    assert loaded.evaluation.quality_retained == report.evaluation.quality_retained


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def test_evaluate_and_gate_end_to_end():
    old, new = make_related_spaces(n=2500, d_old=48, d_new=64, noise=0.02, seed=0)
    tr, te = slice(0, 2000), slice(2000, 2500)
    mapper = LinearMapper(normalize_output=True).fit_cv(old[tr], new[tr], metric="cosine")

    report = evaluate_and_gate(old[te], new[te], mapper, threshold=0.5, k=10, max_queries=None)
    assert report.verdict.passed is True
    assert report.evaluation.recall_at_k_mapped > report.evaluation.recall_at_k_old
