"""Simulação de aprendizado de atalhos sob mudança de distribuição."""
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ESTAVEL, ATALHO = 0, 1
SEMENTE = 2026
MODELOS = {
    "Treino familiar": (ESTAVEL, ATALHO),
    "Treino diversificado": (ESTAVEL, ATALHO),
    "Só sinal estável": (ESTAVEL,),
}


def gerar_dados(n: int, ambiente: float, rng: np.random.Generator):
    """Amostra rótulos equiprováveis e duas características com ruído normal."""
    if n < 1 or not -1 <= ambiente <= 1:
        raise ValueError("Use n positivo e um ambiente entre -1 e 1.")
    y = rng.choice([-1, 1], size=n)
    estavel = 1.25 * y + rng.normal(size=n)
    atalho = 2 * ambiente * y + rng.normal(size=n)
    return np.column_stack([estavel, atalho]), y


def gerar_treinos(n: int, rng: np.random.Generator):
    """Mantém n fixo e compartilha a primeira metade entre os treinos."""
    if n < 2 or n % 2:
        raise ValueError("O total de treino deve ser par e maior ou igual a 2.")
    base = gerar_dados(n // 2, 1, rng)
    extras = (gerar_dados(n // 2, ambiente, rng) for ambiente in (1, -1))
    return tuple(
        (np.vstack([base[0], X]), np.concatenate([base[1], y]))
        for X, y in extras
    )


def ajustar_modelos(n: int, rng: np.random.Generator):
    familiar, diversificado = gerar_treinos(n, rng)
    treinos = (familiar, diversificado, familiar)
    return {
        nome: LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        .fit(X[:, colunas], y)
        for (nome, colunas), (X, y) in zip(MODELOS.items(), treinos, strict=True)
    }


def simular(
    ambientes: Iterable[float],
    *,
    repeticoes: int = 80,
    n_treino: int = 2_000,
    n_teste: int = 2_000,
    semente: int = SEMENTE,
) -> pd.DataFrame:
    """Ajusta uma vez por repetição; todos os modelos recebem o mesmo teste."""
    ambientes = tuple(ambientes)
    if repeticoes < 2 or not ambientes or len(set(ambientes)) != len(ambientes):
        raise ValueError("Use pelo menos duas repetições e ambientes distintos.")
    registros = []
    for repeticao, semente_repeticao in enumerate(
        np.random.SeedSequence(semente).spawn(repeticoes)
    ):
        treino_seed, teste_seed = semente_repeticao.spawn(2)
        modelos = ajustar_modelos(n_treino, np.random.default_rng(treino_seed))
        rng_teste = np.random.default_rng(teste_seed)
        for ambiente in ambientes:
            X, y = gerar_dados(n_teste, ambiente, rng_teste)
            for nome, modelo in modelos.items():
                registros.append(
                    (repeticao, ambiente, nome, modelo.score(X[:, MODELOS[nome]], y))
                )
    return pd.DataFrame(registros, columns=["repeticao", "ambiente", "modelo", "acuracia"])


def resumir(resultados: pd.DataFrame, grupos: list[str], coluna="acuracia"):
    """Resume a média e seu IC de 95% aproximado entre repetições."""
    resumo = resultados.groupby(grupos)[coluna].agg(
        media="mean", desvio="std", n="count"
    )
    resumo["margem"] = 1.96 * resumo["desvio"] / np.sqrt(resumo["n"])
    resumo["ic_inf"] = resumo["media"] - resumo["margem"]
    resumo["ic_sup"] = resumo["media"] + resumo["margem"]
    return resumo.reset_index()
