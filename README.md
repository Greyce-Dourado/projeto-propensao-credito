# Propensão à inadimplência + validação por teste A/B

Modelo de propensão a **inadimplência** de crédito com validação estatística, e uma
**intervenção simulada validada por teste A/B controlado** — no mesmo padrão de um experimento:
hipótese clara, dimensionamento de amostra, métrica antes/depois e prova honesta de que o desenho
funciona.

## Resultados

**Modelagem** (30.000 clientes, alvo de inadimplência ~22%, teste com 5.993):

| modelo | AUC | KS | CV AUC (treino) | gap treino−teste |
|---|---:|---:|---:|---:|
| regressão logística | 0,719 | 0,363 | 0,726 | 0,009 |
| **random forest (final)** | **0,769** | **0,418** | **0,782** | **0,056** |
| xgboost | 0,772 | 0,420 | 0,778 | 0,106 |

Escolhi o **Random Forest**, não o XGBoost: empate técnico no teste (AUC 0,769 vs 0,772), mas
**generaliza melhor** (gap 0,056 vs 0,106) e é mais estável na validação cruzada. Priorizei
generalização, não o maior número por 0,003.

**Teste A/B** — intervenção simulada nos 20% de maior risco (1.199 clientes, inadimplência real
55,7%), com efeito assumido de −20% relativo. Desenho validado por **Monte Carlo (1.000 rodadas)**:

| métrica | resultado | esperado |
|---|---:|---:|
| poder empírico | 0,970 | ~0,969 (analítico) |
| lift médio | −11,2 p.p. (−19,9%) | −20% (efeito injetado) |
| falso positivo (A/A) | 0,046 | ~0,05 (α) |

O desenho **recupera o efeito injetado**, tem o **poder previsto** e **não gera falso positivo** —
a validação estatística completa, não um único resultado de sorte.

## Nota de honestidade

O A/B é uma **simulação transparente**: o dataset não contém um experimento real. O projeto
demonstra o *desenho e a análise* corretos de um teste (targeting pelo modelo, power analysis,
aleatorização, teste de hipótese, lift com IC, Monte Carlo) — **não** um resultado de negócio
real. O único número assumido (a intervenção evita 20% das inadimplências) está sempre rotulado.

## Problema

Prever quais clientes têm alta probabilidade de inadimplir e **simular** uma ação de mitigação
(contato proativo / plano de pagamento) direcionada ao grupo de alto risco, medindo o efeito com
um teste A/B controlado e estatisticamente dimensionado.

## Dados

[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
(UCI, Taiwan, 2005): 30.000 clientes, 23 features, alvo binário (~22% de inadimplência). Não
versionado no repo — baixar e colocar em `data/raw/UCI_Credit_Card.csv`.

Tratamentos de qualidade (sem vazamento): categorias não documentadas de `EDUCATION`/`MARRIAGE`
agrupadas em "outros"; `PAY_*` mantido como ordinal (magnitude = meses de atraso, o preditor mais
forte); duplicatas exatas removidas antes do split para não vazar entre treino e teste.

## Abordagem

1. **Dados** — limpeza determinística + split estratificado (preserva os ~22% nos dois lados).
2. **Modelagem** — baseline (logística) → Random Forest → XGBoost, todos tratando o
   desbalanceamento; métricas AUC, KS, precisão/recall; validação cruzada e checagem de
   overfitting (gap treino−teste). Codificação one-hot **dentro do Pipeline** (ajustada só no
   treino).
3. **A/B — desenho** — grupo de alto risco pelo score do modelo; *power analysis* (n mínimo por
   grupo para 80% de poder); aleatorização controle/tratamento.
4. **A/B — medição** — teste de duas proporções (z), lift com IC 95%, checagem A/A, e validação do
   desenho por Monte Carlo.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data          # limpeza + split (EDA)
python -m src.train         # modelos + métricas + curva ROC
python -m src.experiment    # grupo de alto risco + power analysis
python -m src.measure       # teste A/B (lift, IC, p-valor) + checagem A/A
python -m src.montecarlo    # validação do desenho (poder, viés, falso positivo)
```

## Estrutura

```
src/
├── data.py         # carregar, limpar, split estratificado
├── model.py        # pipelines (pré-processamento + estimador)
├── metrics.py      # AUC, KS, precisão/recall
├── train.py        # treina e compara os modelos
├── experiment.py   # grupo de alto risco + power analysis
├── measure.py      # teste A/B + checagem A/A
└── montecarlo.py   # validação do desenho por repetição
results/            # métricas (JSON) e curva ROC
```

## Status

- [x] Dados + modelagem preditiva (AUC/KS, CV, overfitting)
- [x] Desenho do A/B (power analysis, aleatorização)
- [x] Medição (lift + IC) e validação por Monte Carlo
- [ ] Publicação e material de portfólio

---

Desenvolvido por **Greyce Dourado** — Analista de Dados Sênior.
[LinkedIn](https://www.linkedin.com/in/greyce-dourado) · [GitHub](https://github.com/Greyce-Dourado)
