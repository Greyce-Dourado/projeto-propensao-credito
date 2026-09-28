"""Etapa 2b — treina baseline, Random Forest e XGBoost; mede AUC/KS/precisão-recall,
validação cruzada (estabilidade) e o gap treino-teste (overfitting).

Uso:  ./.venv/bin/python -m src.train
Salva results/model_metrics.json e results/roc_curves.png.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # salva figura sem abrir janela
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import ROOT, load_and_clean, split
from src.metrics import evaluate
from src.model import get_models


def main() -> int:
    df = load_and_clean()
    Xtr, Xte, ytr, yte = split(df)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())
    print(f"treino {len(ytr)} | teste {len(yte)} | scale_pos_weight={pos_weight:.3f}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados: dict[str, dict] = {}

    plt.figure(figsize=(6, 5))
    for nome, pipe in get_models(pos_weight).items():
        # validação cruzada no TREINO (AUC): estabilidade, não um número único de sorte
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        p_tr = pipe.predict_proba(Xtr)[:, 1]
        p_te = pipe.predict_proba(Xte)[:, 1]

        m = evaluate(yte, p_te)
        m["auc_treino"] = round(float(roc_auc_score(ytr, p_tr)), 4)
        m["auc_gap"] = round(m["auc_treino"] - m["auc"], 4)  # gap alto => overfitting
        m["cv_auc_mean"] = round(float(cv_auc.mean()), 4)
        m["cv_auc_std"] = round(float(cv_auc.std()), 4)
        resultados[nome] = m

        fpr, tpr, _ = roc_curve(yte, p_te)
        plt.plot(fpr, tpr, label=f"{nome}  (AUC={m['auc']:.3f}, KS={m['ks']:.3f})")

    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("Falso positivo (FPR)")
    plt.ylabel("Verdadeiro positivo (TPR)")
    plt.title("Curva ROC — conjunto de teste")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    (ROOT / "results").mkdir(exist_ok=True)
    plt.savefig(ROOT / "results" / "roc_curves.png", dpi=150)

    # Tabela comparativa
    print(f"{'modelo':<14}{'AUC':>7}{'KS':>7}{'prec':>7}{'recall':>8}{'F1':>7}"
          f"{'CV AUC (treino)':>18}{'gap tr-te':>11}")
    for nome, m in resultados.items():
        print(f"{nome:<14}{m['auc']:>7.3f}{m['ks']:>7.3f}{m['precision']:>7.3f}"
              f"{m['recall']:>8.3f}{m['f1']:>7.3f}"
              f"{m['cv_auc_mean']:>11.3f}±{m['cv_auc_std']:.3f}{m['auc_gap']:>11.3f}")

    (ROOT / "results" / "model_metrics.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nsalvo: results/model_metrics.json e results/roc_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
