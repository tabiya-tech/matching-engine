"""ZQF annotation invariants (v5 opportunities)."""

from app.routes import _zqf_annotation


class TestZqfAnnotation:
    def test_eligible_when_user_meets_minimum(self):
        eligible, gap = _zqf_annotation(user_zqf=5, job_zqf_min=3)
        assert eligible is True
        assert gap == 2

    def test_not_eligible_when_user_below_minimum(self):
        eligible, gap = _zqf_annotation(user_zqf=2, job_zqf_min=5)
        assert eligible is False
        assert gap == 3

    def test_missing_user_zqf_returns_none(self):
        assert _zqf_annotation(user_zqf=None, job_zqf_min=3) == (None, None)

    def test_missing_job_zqf_returns_none(self):
        assert _zqf_annotation(user_zqf=5, job_zqf_min=None) == (None, None)

    def test_accepts_float_job_zqf_min(self):
        eligible, gap = _zqf_annotation(user_zqf=4, job_zqf_min=4.0)
        assert eligible is True
        assert gap == 0
