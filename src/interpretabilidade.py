"""Funções de interpretabilidade SHAP do modelo SIMAVE."""

from typing import Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.modelagem import VARIAVEIS_MODELO


ROTULOS_VARIAVEIS: dict[str, str] = {
    "Ano": "Ano da avaliação",
    "rede": "Rede de ensino",
    "etapa": "Etapa escolar",
    "disciplina": "Disciplina",
    "descricao_padronizada": "Conteúdo avaliado",
}


def _identificar_variavel_original(nome_codificado: str) -> str:
    """Relaciona uma coluna do One-Hot Encoding à variável original"""
    if nome_codificado.startswith("remainder__Ano"):
        return "Ano"
    for variavel in ("rede", "etapa", "disciplina", "descricao_padronizada"):
        if nome_codificado.startswith(f"categoricas__{variavel}_"):
            return variavel
    raise ValueError(f"Coluna codificada não reconhecida: {nome_codificado}")


def calcular_shap_agregado(
    pipeline: Pipeline,
    dados: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula SHAP da classe abaixo da mediana e agrega o One-Hot Encoding"""
    preprocessador = pipeline.named_steps["preprocessamento"]
    modelo = pipeline.named_steps["modelo"]
    matriz = preprocessador.transform(dados.loc[:, VARIAVEIS_MODELO])
    if hasattr(matriz, "toarray"):
        matriz = matriz.toarray()

    explicador = shap.TreeExplainer(modelo)
    explicacao = explicador(matriz, check_additivity=False)
    valores = np.asarray(explicacao.values)
    valores_base = np.asarray(explicacao.base_values)

    if valores.ndim == 3:
        valores = valores[:, :, 1]
    if valores_base.ndim == 2:
        valores_base = valores_base[:, 1]
    elif valores_base.ndim == 0:
        valores_base = np.repeat(float(valores_base), len(dados))

    nomes_codificados = preprocessador.get_feature_names_out()
    indices_por_variavel: dict[str, list[int]] = {variavel: [] for variavel in VARIAVEIS_MODELO}
    for indice, nome in enumerate(nomes_codificados):
        indices_por_variavel[_identificar_variavel_original(nome)].append(indice)

    valores_agregados = np.column_stack(
        [valores[:, indices_por_variavel[variavel]].sum(axis=1) for variavel in VARIAVEIS_MODELO]
    )
    return valores_agregados, valores_base.astype(float)


def criar_explicacao_local(
    pipeline: Pipeline,
    registro: pd.DataFrame,
) -> shap.Explanation:
    """Cria uma explicação SHAP local com cinco rótulos compreensíveis"""
    valores, valores_base = calcular_shap_agregado(pipeline, registro)
    dados_exibicao = [
        str(registro.iloc[0][variavel]) for variavel in VARIAVEIS_MODELO
    ]
    return shap.Explanation(
        values=valores[0],
        base_values=valores_base[0],
        data=dados_exibicao,
        feature_names=[ROTULOS_VARIAVEIS[variavel] for variavel in VARIAVEIS_MODELO],
    )


def resumir_importancia_global(
    valores_shap: np.ndarray,
    variaveis: Sequence[str] = VARIAVEIS_MODELO,
) -> pd.DataFrame:
    """Resume a importância global pela média do valor SHAP absoluto"""
    return (
        pd.DataFrame(
            {
                "variavel": [ROTULOS_VARIAVEIS[variavel] for variavel in variaveis],
                "importancia": np.abs(valores_shap).mean(axis=0),
            }
        )
        .sort_values("importancia", ascending=True)
        .reset_index(drop=True)
    )

