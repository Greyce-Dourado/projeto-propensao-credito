"""Etapa 4 — executa o A/B simulado e mede o resultado (lift, IC 95%, p-valor).

Premissa explícita: a intervenção evita REL_EFFECT (20%) das inadimplências que ocorreriam no
grupo tratado. O controle mantém o desfecho real. Inclui uma checagem A/A (sem intervenção),
que NÃO deve ser significativa — prova de que o desenho não fabrica falso positivo.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from src.data import ROOT
from src.experiment import COHORT_PATH, REL_EFFECT, SEED


def _teste(d_t, n_t, d_c, n_c):
    """Duas proporções (z) + IC 95% da diferença (tratamento - controle)."""
    stat, pval = proportions_ztest([d_t, d_c], [n_t, n_c], alternative="two-sided")
    low, high = confint_proportions_2indep(d_t, n_t, d_c, n_c, method="wald")
    return float(pval), float(low), float(high)


def main() -> int:
    coorte = pd.read_csv(COHORT_PATH)
    controle = coorte[coorte["grupo"] == "controle"]
    tratamento = coorte[coorte["grupo"] == "tratamento"]

    y_ctrl = controle["y_real"].to_numpy()
    y_trat_real = tratamento["y_real"].to_numpy()

    # Simular o efeito: "salvar" REL_EFFECT dos inadimplentes reais do tratamento.
    rng = np.random.default_rng(SEED)
    y_trat = y_trat_real.copy()
    idx_default = np.where(y_trat == 1)[0]
    salvos = rng.random(len(idx_default)) < REL_EFFECT
    y_trat[idx_default[salvos]] = 0

    n_c, n_t = len(y_ctrl), len(y_trat)
    d_c, d_t = int(y_ctrl.sum()), int(y_trat.sum())
    p_c, p_t = d_c / n_c, d_t / n_t

    pval, low, high = _teste(d_t, n_t, d_c, n_c)
    lift_abs = p_t - p_c
    lift_rel = lift_abs / p_c

    # Checagem A/A: sem intervenção (desfecho real nos dois) -> não deve ser significativo.
    d_t0 = int(y_trat_real.sum())
    pval_aa, _, _ = _teste(d_t0, n_t, d_c, n_c)

    print("=== A/B (com intervenção simulada) ===")
    print(f"controle:   n={n_c}  inadimplência={p_c:.3f}  ({d_c} de {n_c})")
    print(f"tratamento: n={n_t}  inadimplência={p_t:.3f}  ({d_t} de {n_t})")
    print(f"\nlift: {lift_abs * 100:+.1f} p.p.  (relativo {lift_rel * 100:+.1f}%)")
    print(f"IC 95% da diferença: [{low * 100:+.1f}, {high * 100:+.1f}] p.p.")
    print(f"p-valor (z, bicaudal): {pval:.3g}")
    print(f"significativo a 5%? {'SIM — rejeita H0' if pval < 0.05 else 'não'}")

    print("\n=== Checagem A/A (sem intervenção) ===")
    print(f"p-valor: {pval_aa:.3g}  -> {'ok, não significativo' if pval_aa >= 0.05 else 'ATENÇÃO: significativo'}")

    resultado = {
        "controle": {"n": n_c, "inadimplencia": round(p_c, 4), "defaults": d_c},
        "tratamento": {"n": n_t, "inadimplencia": round(p_t, 4), "defaults": d_t},
        "lift_abs_pp": round(lift_abs * 100, 2),
        "lift_rel_pct": round(lift_rel * 100, 2),
        "ic95_diff_pp": [round(low * 100, 2), round(high * 100, 2)],
        "p_valor": pval,
        "significativo_5pct": bool(pval < 0.05),
        "aa_p_valor": pval_aa,
        "efeito_assumido_rel": REL_EFFECT,
    }
    out = ROOT / "results" / "ab_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo: results/ab_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
