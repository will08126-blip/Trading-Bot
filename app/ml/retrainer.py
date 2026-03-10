"""
ML Retraining Workflow

Generates training datasets from historical trades and retrains models.
Run via: python -m app.main --mode retrain
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.config.trading_config import get_trading_config
from app.ml.regime_classifier import RegimeClassifier
from app.ml.quality_scorer import TradeQualityScorer
from app.models.trade import Trade
from app.models.regime_state import RegimeState
from app.models.ml_model_meta import MLModelMeta

logger = structlog.get_logger(__name__)


class MLRetrainer:
    """
    Orchestrates ML model retraining.
    Extracts features from DB, trains models, saves to disk.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._cfg = get_trading_config().ml
        settings = get_settings()
        self._model_dir = Path(settings.ml_model_dir)

    async def retrain_all(self, mode: str = "paper") -> dict:
        """Retrain all ML models. Returns summary of results."""
        results = {}

        logger.info("retraining_started")

        # Retrain quality scorer
        quality_result = await self._retrain_quality_scorer(mode)
        results["quality_scorer"] = quality_result

        # Retrain regime classifier
        regime_result = await self._retrain_regime_classifier()
        results["regime_classifier"] = regime_result

        logger.info("retraining_complete", results=results)
        return results

    async def _retrain_quality_scorer(self, mode: str) -> dict:
        """Build dataset from trade history and retrain quality scorer."""
        result = await self._db.execute(
            select(Trade).where(Trade.mode == mode, Trade.status == "closed")
        )
        trades = result.scalars().all()

        if len(trades) < self._cfg.min_training_samples:
            return {"status": "skipped", "reason": "insufficient_data", "count": len(trades)}

        X_list = []
        y_list = []

        for trade in trades:
            if not trade.score_breakdown_json:
                continue
            try:
                # Parse stored score breakdown
                breakdown = json.loads(trade.score_breakdown_json) if isinstance(trade.score_breakdown_json, str) else {}
                features = {
                    "signal_score": trade.signal_score / 100.0,
                    "stop_distance_pct": trade.stop_distance_pct or 0.0,
                    "leverage": trade.leverage or 1.0,
                    "hold_duration_minutes": trade.hold_duration_minutes or 0.0,
                    "slippage_pct": trade.slippage_pct or 0.0,
                }
                X_list.append(list(features.values()))
                y_list.append(1 if (trade.realized_pnl or 0.0) > 0 else 0)
            except Exception:
                continue

        if len(X_list) < self._cfg.min_training_samples:
            return {"status": "skipped", "reason": "insufficient_parseable_data", "count": len(X_list)}

        X = np.array(X_list)
        y = np.array(y_list)

        scorer = TradeQualityScorer()
        feature_names = ["signal_score", "stop_distance_pct", "leverage",
                        "hold_duration_minutes", "slippage_pct"]
        acc = scorer.train(X, y, feature_names)

        await self._save_model_meta("quality_scorer", "gradient_boosting", acc, len(X_list))
        return {"status": "trained", "accuracy": acc, "samples": len(X_list)}

    async def _retrain_regime_classifier(self) -> dict:
        """Build regime dataset from saved regime states."""
        result = await self._db.execute(
            select(RegimeState).where(RegimeState.features_json != None)
        )
        regime_records = result.scalars().all()

        if len(regime_records) < self._cfg.min_training_samples:
            return {"status": "skipped", "reason": "insufficient_data",
                   "count": len(regime_records)}

        X_list = []
        y_list = []
        all_feature_names = set()

        for record in regime_records:
            if not record.features_json:
                continue
            try:
                features = json.loads(record.features_json)
                all_feature_names.update(features.keys())
                X_list.append(features)
                y_list.append(record.regime)
            except Exception:
                continue

        if len(X_list) < self._cfg.min_training_samples:
            return {"status": "skipped", "reason": "parse_errors", "count": len(X_list)}

        feature_names = sorted(all_feature_names)
        X = np.array([[r.get(f, 0.0) for f in feature_names] for r in X_list])
        y = np.array(y_list)

        classifier = RegimeClassifier()
        acc = classifier.train(X, y, feature_names, model_type=self._cfg.regime_model_type)

        await self._save_model_meta("regime_classifier", self._cfg.regime_model_type, acc, len(X_list))
        return {"status": "trained", "accuracy": acc, "samples": len(X_list)}

    async def _save_model_meta(
        self, name: str, model_type: str, val_score: float, samples: int
    ) -> None:
        self._db.add(MLModelMeta(
            model_name=name,
            model_type=model_type,
            version=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"),
            file_path=str(self._model_dir / f"{name}.joblib"),
            training_samples=samples,
            validation_score=val_score,
            is_active=True,
            trained_at=datetime.now(timezone.utc),
        ))
        try:
            await self._db.commit()
        except Exception:
            await self._db.rollback()
