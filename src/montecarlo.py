"""Etapa 4 (fechamento) — valida o DESENHO do A/B por Monte Carlo.

Repete N vezes sobre a mesma coorte de alto risco: re-aleatoriza controle/tratamento, aplica o
efeito assumido no tratamento e roda o teste. Reporta a DISTRIBUIÇÃO, não um sorteio único:
- poder empírico  = fração de rodadas A/B com p<0,05 (deve bater com o poder analítico ~0,97);
- lift médio      = o desbalanceamento de acaso se cancela na média (aproxima o efeito real);
- falso positivo do A/A = fração de rodadas A/A com p<0,05 (deve ficar ~0,05 = alpha).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

from src.data import ROOT
from src.experiment import COHORT_PATH, REL_EFFECT

N_RUNS = 1000
SEED = 0


def _test(y_t: np.ndarray, y_c: np.ndarray) -> tuple[float, float, float]:
    d_t, n_t = int(y_t.sum()), len(y_t)
    d_c, n_c = int(y_c.sum()), len(y_c)
    _, p = proportions_ztest([d_t, d_c], [n_t, n_c], alternative="two-sided")
    p_t, p_c = d_t / n_t, d_c / n_c
    return float(p), p_t - p_c, (p_t - p_c) / p_c


def main() -> int:
    y = pd.read_csv(COHORT_PATH)["y_real"].to_numpy()
    n = len(y)
    rng = np.random.default_rng(SEED)

    ab_sig = aa_sig = 0
    lifts_abs, lifts_rel = [], []
    for _ in range(N_RUNS):
        trat = rng.random(n) < 0.5
        y_c = y[~trat]
        y_t_real = y[trat]

        # A/A: sem intervenção
        p_aa, _, _ = _test(y_t_real, y_c)
        aa_sig += p_aa < 0.05

        # A/B: aplica o efeito (salva REL_EFFECT dos inadimplentes do tratamento)
        y_t = y_t_real.copy()
        idx = np.where(y_t == 1)[0]
        y_t[idx[rng.random(len(idx)) < REL_EFFECT]] = 0
        p_ab, la, lr = _test(y_t, y_c)
        ab_sig += p_ab < 0.05
        lifts_abs.append(la)
        lifts_rel.append(lr)

    la = np.array(lifts_abs)
    lr = np.array(lifts_rel)
    poder = ab_sig / N_RUNS
    fp_aa = aa_sig / N_RUNS

    print(f"rodadas: {N_RUNS}  |  coorte: {n}  |  efeito injetado: -{int(REL_EFFECT*100)}% relativo\n")
    print(f"poder empírico (A/B p<0,05):     {poder:.3f}   (analítico ~0,969)")
    print(f"lift médio:                      {la.mean()*100:+.1f} p.p.  "
          f"(relativo {lr.mean()*100:+.1f}%)")
    print(f"dispersão do lift (desvio):      {la.std()*100:.1f} p.p.")
    print(f"falso positivo A/A (p<0,05):     {fp_aa:.3f}   (esperado ~0,05 = alpha)")

    resultado = {
        "n_runs": N_RUNS,
        "poder_empirico": round(poder, 4),
        "lift_abs_medio_pp": round(float(la.mean()) * 100, 2),
        "lift_rel_medio_pct": round(float(lr.mean()) * 100, 2),
        "lift_abs_desvio_pp": round(float(la.std()) * 100, 2),
        "falso_positivo_aa": round(fp_aa, 4),
        "efeito_injetado_rel": REL_EFFECT,
    }
    out = ROOT / "results" / "ab_montecarlo.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo: results/ab_montecarlo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
