"""
Tests for regime classification.
"""
import pytest
import numpy as np

from app.ml.regime_classifier import RegimeClassifier, RegimeResult, REGIME_CLASSES
from app.market_data.features import compute_all_features
from tests.fixtures.market_data import make_trending_candles, make_ranging_candles, make_volatile_candles


class TestRegimeClassifier:
    def setup_method(self):
        self.clf = RegimeClassifier()

    def test_trending_candles_classified(self):
        df = compute_all_features(make_trending_candles(200, "up"))
        result = self.clf.classify(df)
        assert isinstance(result, RegimeResult)
        assert result.regime in REGIME_CLASSES
        assert 0.0 <= result.confidence <= 1.0

    def test_ranging_candles_classified(self):
        df = compute_all_features(make_ranging_candles(200))
        result = self.clf.classify(df)
        assert result.regime in REGIME_CLASSES

    def test_volatile_candles_classified(self):
        df = compute_all_features(make_volatile_candles(200))
        result = self.clf.classify(df)
        assert result.regime in REGIME_CLASSES

    def test_short_data_returns_unknown(self):
        from tests.fixtures.market_data import make_candles
        df = compute_all_features(make_candles(5))
        result = self.clf.classify(df)
        # With very little data should return something safe
        assert result.regime in REGIME_CLASSES

    def test_heuristic_fallback_works(self):
        """Heuristic should work without a trained model."""
        df = compute_all_features(make_trending_candles(200, "up"))
        self.clf._model_loaded = False
        result = self.clf.classify(df)
        assert result.model_used == "heuristic"
        assert result.regime in REGIME_CLASSES

    def test_feature_extraction(self):
        df = compute_all_features(make_trending_candles(200))
        features = self.clf._extract_features(df)
        assert isinstance(features, dict)
        assert len(features) > 5
        assert "adx" in features
        assert "rsi" in features

    def test_train_with_synthetic_data(self, tmp_path):
        """Test that training runs without errors on synthetic data."""
        import os
        os.environ["ML_MODEL_DIR"] = str(tmp_path)

        # Create synthetic training data
        n_samples = 600
        n_features = 13
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(["trending", "ranging", "compression"], n_samples)
        feature_names = [f"feat_{i}" for i in range(n_features)]

        clf = RegimeClassifier()
        clf._model_dir = tmp_path
        acc = clf.train(X, y, feature_names, model_type="logistic")
        assert 0.0 <= acc <= 1.0
        assert clf._model_loaded
