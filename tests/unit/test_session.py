"""
Tests for session classification.
"""
import pytest
from datetime import datetime, timezone

from app.monitoring.session import SessionClassifier, SessionInfo


class TestSessionClassifier:
    def setup_method(self):
        self.clf = SessionClassifier()

    def _make_dt(self, hour: int) -> datetime:
        return datetime(2025, 1, 13, hour, 0, 0, tzinfo=timezone.utc)

    def test_ny_london_overlap(self):
        dt = self._make_dt(14)  # 14:00 UTC = NY/London overlap
        info = self.clf.classify(dt)
        assert info.name == "overlap"
        assert info.quality_score > 0.8

    def test_ny_session(self):
        dt = self._make_dt(18)  # 18:00 UTC = NY session
        info = self.clf.classify(dt)
        assert info.name == "new_york"

    def test_london_session(self):
        dt = self._make_dt(10)  # 10:00 UTC = London
        info = self.clf.classify(dt)
        assert info.name == "london"

    def test_asia_session(self):
        dt = self._make_dt(3)  # 03:00 UTC = Asia
        info = self.clf.classify(dt)
        assert info.name == "asia"

    def test_off_hours(self):
        dt = self._make_dt(23)  # 23:00 UTC = off hours
        info = self.clf.classify(dt)
        assert info.name == "off_hours"

    def test_quality_scores_ordered(self):
        """Overlap should have highest quality score."""
        overlap = self.clf.classify(self._make_dt(14))
        ny = self.clf.classify(self._make_dt(18))
        asia = self.clf.classify(self._make_dt(3))
        off = self.clf.classify(self._make_dt(23))
        assert overlap.quality_score >= ny.quality_score
        assert ny.quality_score >= asia.quality_score

    def test_get_session_score_returns_float(self):
        score = self.clf.get_session_score()
        assert 0.0 <= score <= 1.5  # weights can exceed 1.0
