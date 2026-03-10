"""
Trade Quality Scorer

Scores candidate trades beyond raw strategy triggers.
Output: confidence 0–1 for trade quality.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import structlog
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from app.config.settings import get_settings
from app.config.trading_config import get_trading_config

logger = structlog.get_logger(__name__)


class TradeQualityScorer:
    """
    Scores a trade candidate based on multi-factor input features.
    Trained on historical trade outcomes.
    Falls back to 0.5 (neutral) if no model is available.
    """

    def __init__(self) -> None:
        self._cfg = get_trading_config().ml
        settings = get_settings()
        self._model_dir = Path(settings.ml_model_dir)
        self._model: Optional[object] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._loaded = False

    def load_model(self, model_name: str = "quality_scorer") -> bool:
        model_path = self._model_dir / f"{model_name}.joblib"
        scaler_path = self._model_dir / f"{model_name}_scaler.joblib"
        features_path = self._model_dir / f"{model_name}_features.json"

        if not model_path.exists():
            return False
        try:
            self._model = joblib.load(model_path)
            if scaler_path.exists():
                self._scaler = joblib.load(scaler_path)
            if features_path.exists():
                with open(features_path) as f:
                    self._feature_names = json.load(f)
            self._loaded = True
            logger.info("quality_scorer_loaded")
            return True
        except Exception as e:
            logger.error("quality_scorer_load_failed", error=str(e))
            return False

    def score(self, features: Dict[str, float]) -> float:
        """
        Score a trade candidate. Returns 0–1.
        0.5 means neutral/unknown quality.
        """
        if not self._loaded or self._model is None:
            return 0.5  # neutral fallback

        try:
            feat_names = self._feature_names or sorted(features.keys())
            X = np.array([[features.get(f, 0.0) for f in feat_names]])
            if self._scaler:
                X = self._scaler.transform(X)
            probas = self._model.predict_proba(X)[0]
            # Assume binary: [prob_loss, prob_win]
            return float(probas[1]) if len(probas) >= 2 else 0.5
        except Exception as e:
            logger.warning("quality_score_error", error=str(e))
            return 0.5

    def build_feature_vector(
        self,
        trend_alignment: float,
        momentum: float,
        volatility_state: str,
        spread_pct: float,
        session_score: float,
        signal_score: float,
        adx: float,
        rsi: float,
        volume_ratio: float,
        regime: str,
        strategy_family: str,
        recent_win_rate: float = 0.5,
    ) -> Dict[str, float]:
        """Helper to build a consistent feature dict for scoring."""
        regime_encoded = {
            "trending": 0, "ranging": 1,
            "volatility_expansion": 2, "compression": 3, "unknown": 4,
        }.get(regime, 4)
        strategy_encoded = {
            "trend_pullback": 0, "breakout_retest": 1,
            "liquidity_sweep_reversal": 2, "volatility_expansion": 3,
        }.get(strategy_family, -1)
        volatility_encoded = {"low": 0, "medium": 1, "high": 2, "extreme": 3}.get(volatility_state, 1)

        return {
            "trend_alignment": trend_alignment,
            "momentum": momentum,
            "volatility_encoded": float(volatility_encoded),
            "spread_pct": spread_pct,
            "session_score": session_score,
            "signal_score": signal_score / 100.0,
            "adx": adx,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "regime_encoded": float(regime_encoded),
            "strategy_encoded": float(strategy_encoded),
            "recent_win_rate": recent_win_rate,
        }

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> float:
        """Train on historical trade data. y = 1 if trade was profitable."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        if len(X) < self._cfg.min_training_samples:
            logger.warning("quality_scorer_insufficient_data", samples=len(X))
            return 0.0

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))

        model_name = "quality_scorer"
        joblib.dump(model, self._model_dir / f"{model_name}.joblib")
        joblib.dump(scaler, self._model_dir / f"{model_name}_scaler.joblib")
        import json
        with open(self._model_dir / f"{model_name}_features.json", "w") as f:
            json.dump(feature_names, f)

        self._model = model
        self._scaler = scaler
        self._feature_names = feature_names
        self._loaded = True

        logger.info("quality_scorer_trained", accuracy=round(acc, 3), samples=len(X_train))
        return acc
