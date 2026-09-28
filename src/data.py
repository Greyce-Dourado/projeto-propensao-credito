"""Etapa 2a — carregar, limpar e dividir os dados de inadimplência de crédito.

Decisões de tratamento (todas determinísticas e sem vazamento — não dependem do alvo):
- EDUCATION: categorias não documentadas {0, 5, 6} agrupadas em 4 ("outros").
- MARRIAGE: categoria não documentada {0} agrupada em 3 ("outros").
- PAY_0..PAY_6 mantidos como numéricos ORDINAIS (a magnitude = meses de atraso tem significado).
- A codificação one-hot das nominais NÃO é feita aqui: vai dentro do Pipeline do modelo (2b),
  ajustada só no treino, para evitar vazamento.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
CLEAN = ROOT / "data" / "processed" / "credit_clean.csv"

TARGET = "target"
NOMINAL = ["SEX", "EDUCATION", "MARRIAGE"]  # categóricas nominais -> one-hot na etapa 2b
NUMERIC = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
FEATURES = NOMINAL + NUMERIC


def load_and_clean() -> pd.DataFrame:
    """Lê o CSV bruto, remove o ID e agrupa as categorias fantasma."""
    df = pd.read_csv(RAW).drop(columns=["ID"])
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    # Remove duplicatas EXATAS (todas as colunas iguais) antes do split, para não vazar
    # a mesma linha entre treino e teste. São ~35 linhas (0,1%); efeito prático mínimo,
    # mas é a escolha estatisticamente correta.
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split ESTRATIFICADO: preserva a taxa de ~22% de inadimplência em treino e teste.
    Sem estratificar, um alvo desbalanceado pode ficar com proporções diferentes nos splits."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def _eda(df: pd.DataFrame) -> None:
    print(f"linhas x colunas: {df.shape}")
    print(f"valores nulos (total): {int(df.isna().sum().sum())}")
    print(f"linhas duplicadas: {int(df.duplicated().sum())}")
    print(f"taxa de inadimplência (base rate): {df[TARGET].mean():.4f}\n")
    # Sinal do preditor mais forte: inadimplência cresce com os meses de atraso do último mês.
    por_atraso = df.groupby("PAY_0")[TARGET].agg(["mean", "size"])
    por_atraso.columns = ["taxa_inadimplencia", "n_clientes"]
    print("inadimplência por status de atraso (PAY_0):")
    print(por_atraso.round(3).to_string())


if __name__ == "__main__":
    df = load_and_clean()
    _eda(df)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    Xtr, Xte, ytr, yte = split(df)
    print(f"\nsplit -> treino {Xtr.shape[0]} | teste {Xte.shape[0]}")
    print(f"taxa no treino {ytr.mean():.4f} | taxa no teste {yte.mean():.4f}  (devem ser ~iguais)")
    print(f"dados limpos salvos em {CLEAN.relative_to(ROOT)}")
