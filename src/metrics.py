"""Etapa 2b — métricas de avaliação do modelo.

- AUC: separação geral (robusta ao desbalanceamento).
- KS (Kolmogorov-Smirnov): métrica canônica de crédito = máxima distância entre as distribuições
  acumuladas de score de inadimplentes vs. adimplentes. Equivale a max(TPR - FPR) na curva ROC.
- precisão/recall/F1 num limiar: acurácia engana com 22% de positivos, então olhamos isto.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    yhat = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    return {
        "auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "precision": round(float(precision_score(y_true, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, yhat, zero_division=0)), 4),
        "conf": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
