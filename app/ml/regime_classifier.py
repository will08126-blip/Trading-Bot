"""
Market Regime Classifier

Classifies market into: trending | ranging | volatility_expansion | compression | unknown

Uses:
- Statistical heuristics as baseline (always available)
- Trained ML model (loaded from disk if available)

The heuristic fallback is used when no model is trained yet.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from app.config.settings import get_settings
from app.config.trading_config import get_trading_config

logger = structlog.get_logger(__name__)

REGIME_CLASSES = ["trending", "ranging", "volatility_expansion", "compression", "unknown"]


@dataclass
class RegimeResult:
    regime: str
    confidence: float
    features: Dict[str, float]
    model_used: str  # "heuristic" | "ml"


class RegimeClassifier:
    """
    Classifies market regime from a candle DataFrame.
    Falls back to heuristics if no ML model is available.
    """

    def __init__(self) -> None:
        self._cfg = get_trading_config().ml
        settings = get_settings()
        self._model_dir = Path(settings.ml_model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model: Optional[object] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._model_loaded = False

    def load_model(self, model_name: str = "regime_classifier") -> bool:
        """Attempt to load a trained model from disk."""
        model_path = self._model_dir / f"{model_name}.joblib"
        scaler_path = self._model_dir / f"{model_name}_scaler.joblib"
        features_path = self._model_dir / f"{model_name}_features.json"

        if not model_path.exists():
            logger.info("regime_model_not_found", path=str(model_path))
            return False

        try:
            self._model = joblib.load(model_path)
            if scaler_path.exists():
                self._scaler = joblib.load(scaler_path)
            if features_path.exists():
                with open(features_path) as f:
                    self._feature_names = json.load(f)
            self._model_loaded = True
            logger.info("regime_model_loaded", path=str(model_path))
            return True
        except Exception as e:
            logger.error("regime_model_load_failed", error=str(e))
            self._model_loaded = False
            return False

    def save_model(self, model, scaler, feature_names: List[str], model_name: str = "regime_classifier") -> None:
        """Save trained model to disk."""
        joblib.dump(model, self._model_dir / f"{model_name}.joblib")
        if scaler:
            joblib.dump(scaler, self._model_dir / f"{model_name}_scaler.joblib")
        with open(self._model_dir / f"{model_name}_features.json", "w") as f:
            json.dump(feature_names, f)
        self._model = model
        self._scaler = scaler
        self._feature_names = feature_names
        self._model_loaded = True
        logger.info("regime_model_saved", model_name=model_name)

    def classify(self, df: pd.DataFrame) -> RegimeResult:
        """
        Classify the regime from an enriched candle DataFrame.
        Falls back to heuristics if no model available.
        """
        features = self._extract_features(df)

        if self._model_loaded and self._model is not None and self._cfg.fallback_to_heuristics is False:
            return self._ml_classify(features)

        if self._model_loaded and self._model is not None:
            try:
                return self._ml_classify(features)
            except Exception as e:
                logger.warning("ml_classify_fallback", error=str(e))

        return self._heuristic_classify(df, features)

    def _extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract numeric features from enriched DataFrame."""
        if df is None or len(df) < 20:
            return {}

        last = df.iloc[-1]

        def safe_get(col, default=0.0):
            val = last.get(col, default)
            return float(val) if not pd.isna(val) else default

        features = {
            "adx": safe_get("adx"),
            "adx_trend": 1.0 if safe_get("plus_di") > safe_get("minus_di") else -1.0,
            "rsi": safe_get("rsi_14"),
            "bb_width": safe_get("bb_width"),
            "bb_pct": safe_get("bb_pct"),
            "volatility_ratio": safe_get("volatility_ratio", 1.0),
            "volume_ratio": safe_get("volume_ratio", 1.0),
            "roc_5": safe_get("roc_5"),
            "roc_10": safe_get("roc_10"),
            "roc_20": safe_get("roc_20"),
            "atr_pct": safe_get("atr_pct"),
            "macd_hist": safe_get("macd_hist"),
            "structure_range_pct": safe_get("structure_range_pct"),
        }
        # EMA alignment
        ema20 = safe_get("ema_20", 1.0)
        ema50 = safe_get("ema_50", 1.0)
        close = safe_get("close", 1.0)
        features["price_vs_ema20"] = (close - ema20) / (ema20 + 1e-8)
        features["ema20_vs_ema50"] = (ema20 - ema50) / (ema50 + 1e-8)
        return features

    def _heuristic_classify(self, df: pd.DataFrame, features: Dict[str, float]) -> RegimeResult:
        """
        Statistical heuristic regime classification.
        Used as baseline and fallback.
        """
        adx = features.get("adx", 0.0)
        bb_width = features.get("bb_width", 1.0)
        volatility_ratio = features.get("volatility_ratio", 1.0)
        volume_ratio = features.get("volume_ratio", 1.0)

        # Historical BB width for comparison
        if df is not None and "bb_width" in df.columns and len(df) >= 20:
            bb_width_hist = df["bb_width"].tail(50)
            bb_width_pct = float((bb_width - bb_width_hist.min()) /
                                  (bb_width_hist.max() - bb_width_hist.min() + 1e-8))
        else:
            bb_width_pct = 0.5

        # Regime scoring
        scores = {
            "trending": 0.0,
            "ranging": 0.0,
            "volatility_expansion": 0.0,
            "compression": 0.0,
        }

        # Trending signals
        if adx > 30:
            scores["trending"] += 0.5
        if adx > 20:
            scores["trending"] += 0.2
        if abs(features.get("ema20_vs_ema50", 0.0)) > 0.005:
            scores["trending"] += 0.2

        # Ranging signals
        if adx < 25:
            scores["ranging"] += 0.3
        if 0.3 < bb_width_pct < 0.7:
            scores["ranging"] += 0.2
        if 40 <= features.get("rsi", 50) <= 60:
            scores["ranging"] += 0.1

        # Volatility expansion
        if volatility_ratio > 1.3:
            scores["volatility_expansion"] += 0.4
        if volume_ratio > 1.5:
            scores["volatility_expansion"] += 0.2
        if bb_width_pct > 0.8:
            scores["volatility_expansion"] += 0.2

        # Compression
        if volatility_ratio < 0.7:
            scores["compression"] += 0.4
        if bb_width_pct < 0.2:
            scores["compression"] += 0.3
        if volume_ratio < 0.7:
            scores["compression"] += 0.1

        best_regime = max(scores, key=lambda k: scores[k])
        best_score = scores[best_regime]
        confidence = min(best_score, 1.0)

        if confidence < 0.25:
            return RegimeResult("unknown", 0.3, features, "heuristic")

        return RegimeResult(best_regime, confidence, features, "heuristic")

    def _ml_classify(self, features: Dict[str, float]) -> RegimeResult:
        """Classify using trained ML model."""
        if not self._feature_names:
            feature_names = sorted(features.keys())
        else:
            feature_names = self._feature_names

        X = np.array([[features.get(f, 0.0) for f in feature_names]])

        if self._scaler:
            X = self._scaler.transform(X)

        pred_class = self._model.predict(X)[0]
        probas = self._model.predict_proba(X)[0]
        confidence = float(max(probas))

        return RegimeResult(str(pred_class), confidence, features, "ml")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_type: Optional[str] = None,
    ) -> float:
        """
        Train a regime classifier on labeled data.
        Returns validation accuracy.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        model_type = model_type or self._cfg.regime_model_type

        if len(X) < self._cfg.min_training_samples:
            logger.warning("insufficient_training_data",
                          samples=len(X), min_required=self._cfg.min_training_samples)
            return 0.0

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)

        logger.info("regime_model_trained", accuracy=round(accuracy, 3),
                   model_type=model_type, samples=len(X_train))

        self.save_model(model, scaler, feature_names)
        return accuracy
