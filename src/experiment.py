"""Etapa 3a — desenho do A/B: grupo de alto risco, power analysis e aleatorização.

Simulação transparente: o dataset não tem experimento real. Aqui só DESENHAMOS o teste
(quem entra, qual o tamanho de amostra necessário, quem é controle vs. tratamento). A execução
com o efeito assumido fica na Etapa 4.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src.data import ROOT, load_and_clean, split
from src.model import get_models

# --- parâmetros do desenho (explícitos e ajustáveis) ---
QUANTILE = 0.80   # grupo de alto risco = top 20% de score
REL_EFFECT = 0.20  # efeito ASSUMIDO: intervenção reduz a inadimplência em 20% relativo
ALPHA = 0.05
POWER = 0.80
SEED = 42

COHORT_PATH = ROOT / "data" / "processed" / "ab_cohort.csv"


def main() -> int:
    df = load_and_clean()
    Xtr, Xte, ytr, yte = split(df)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())

    # 1) Modelo final (Random Forest) treinado no treino, score no teste (população limpa).
    rf = get_models(pos_weight)["random_forest"]
    rf.fit(Xtr, ytr)
    score = rf.predict_proba(Xte)[:, 1]

    # 2) Grupo de alto risco = top 20% por score.
    limiar = np.quantile(score, QUANTILE)
    alto_risco = score >= limiar
    n_cohort = int(alto_risco.sum())

    # 3) Baseline REAL: taxa de inadimplência dentro do grupo (vem do target real, não inventada).
    y_cohort = yte.to_numpy()[alto_risco]
    p1 = float(y_cohort.mean())
    p2 = p1 * (1 - REL_EFFECT)  # taxa esperada no tratamento sob o efeito assumido

    # 4) Power analysis: n mínimo por grupo para detectar (p1 -> p2) com 80% de poder.
    h = abs(proportion_effectsize(p1, p2))  # Cohen's h (tamanho de efeito p/ proporções)
    analise = NormalIndPower()
    n_req = analise.solve_power(effect_size=h, alpha=ALPHA, power=POWER, ratio=1.0,
                                alternative="two-sided")

    # 5) Aleatorização em controle vs. tratamento (50/50).
    rng = np.random.default_rng(SEED)
    grupo = rng.choice(["controle", "tratamento"], size=n_cohort)
    n_por_braco = int(min((grupo == "controle").sum(), (grupo == "tratamento").sum()))

    # Poder efetivo com o n que de fato temos por braço.
    poder_efetivo = analise.solve_power(effect_size=h, nobs1=n_por_braco, alpha=ALPHA,
                                        ratio=1.0, alternative="two-sided")

    # --- relatório ---
    print(f"grupo de alto risco: {n_cohort} clientes (top {round((1-QUANTILE)*100)}% de score, "
          f"limiar de score = {limiar:.3f})")
    print(f"baseline (inadimplência real no grupo): {p1:.3f}")
    print(f"efeito assumido: -{int(REL_EFFECT*100)}% relativo  =>  {p1:.3f} -> {p2:.3f} "
          f"({(p1-p2)*100:.1f} p.p.)")
    print(f"tamanho de efeito (Cohen's h): {h:.3f}\n")
    print(f"n MÍNIMO por grupo p/ 80% de poder (alpha={ALPHA}): {int(np.ceil(n_req))}")
    print(f"n DISPONÍVEL por grupo (controle/tratamento): {n_por_braco}")
    print(f"poder efetivo com esse n: {poder_efetivo:.3f}  "
          f"-> {'suficiente' if poder_efetivo >= POWER else 'INSUFICIENTE'}")

    # 6) Salvar a coorte com o grupo e o desfecho real (para a Etapa 4).
    import pandas as pd
    coorte = pd.DataFrame({"score": score[alto_risco], "y_real": y_cohort, "grupo": grupo})
    COHORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coorte.to_csv(COHORT_PATH, index=False)
    print(f"\ncoorte salva em {COHORT_PATH.relative_to(ROOT)} ({len(coorte)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
