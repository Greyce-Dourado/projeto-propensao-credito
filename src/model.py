"""Etapa 2b — pipelines de modelagem (pré-processamento + estimador).

Cada modelo é um Pipeline: o pré-processamento (one-hot das nominais) é ajustado DENTRO da
validação cruzada e do fit, só com dados de treino — sem vazamento. Os três modelos tratam
o desbalanceamento de ~22% (class_weight / scale_pos_weight), em vez de ignorá-lo.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data import NOMINAL, NUMERIC


def _preprocess(scale_numeric: bool) -> ColumnTransformer:
    # handle_unknown="ignore": se uma categoria não aparecer num fold de treino, não quebra.
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("nominal", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
            ("numeric", numeric, NUMERIC),
        ]
    )


def build_pipeline(estimator, scale_numeric: bool = False) -> Pipeline:
    return Pipeline([("pre", _preprocess(scale_numeric)), ("model", estimator)])


def get_models(pos_weight: float) -> dict[str, Pipeline]:
    """pos_weight = nº de negativos / nº de positivos no treino (~3,5), para o XGBoost."""
    return {
        # Baseline: regressão logística (linear, interpretável). Numéricas escaladas.
        "logistica": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            scale_numeric=True,
        ),
        # Random Forest: profundidade limitada para não decorar.
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
        ),
        # XGBoost: gradient boosting, hiperparâmetros modestos (projeto cirúrgico, sem grid grande).
        "xgboost": build_pipeline(
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9, scale_pos_weight=pos_weight,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            ),
        ),
    }
