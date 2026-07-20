"""Treina e salva os artefatos usados pelo dashboard Streamlit"""

import argparse
from pathlib import Path
import sys

import pandas as pd


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from src.modelagem import (
    avaliar_modelo_final,
    carregar_bases_descritores,
    preparar_dados_modelagem,
    salvar_json,
    salvar_modelo,
    separar_treino_teste,
    treinar_modelo_final,
)


def criar_artefatos_dashboard(
    dados_modelagem: pd.DataFrame,
    limite_acerto: float,
    diretorio_artefatos: str | Path,
) -> dict[str, object]:
    """Treina o pipeline e grava modelo, métricas e agregados do dashboard"""
    diretorio = Path(diretorio_artefatos)
    diretorio.mkdir(parents=True, exist_ok=True)

    x_treino, x_teste, y_treino, y_teste = separar_treino_teste(dados_modelagem)
    modelo = treinar_modelo_final(x_treino, y_treino)
    metricas = avaliar_modelo_final(modelo, x_teste, y_teste)
    metricas.update(
        {
            "limite_acerto": limite_acerto,
            "registros_modelagem": int(len(dados_modelagem)),
            "registros_treino": int(len(x_treino)),
            "registros_teste": int(len(x_teste)),
        }
    )

    panorama = (
        dados_modelagem.groupby(["Ano", "rede", "etapa", "disciplina"], as_index=False)
        .agg(registros=("alvo", "size"), abaixo_mediana=("alvo", "sum"))
        .sort_values(["Ano", "rede", "etapa", "disciplina"])
    )
    amostra_shap = x_teste.sample(n=min(400, len(x_teste)), random_state=42)

    opcoes = {
        "anos": sorted(int(valor) for valor in dados_modelagem["Ano"].unique()),
        "redes": sorted(dados_modelagem["rede"].unique().tolist()),
        "etapas": sorted(dados_modelagem["etapa"].unique().tolist()),
        "disciplinas": sorted(dados_modelagem["disciplina"].unique().tolist()),
        "conteudos_por_disciplina": {
            disciplina: sorted(
                dados_modelagem.loc[
                    dados_modelagem["disciplina"].eq(disciplina), "descritor_descricao"
                ]
                .dropna()
                .drop_duplicates()
                .astype(str)
                .tolist()
            )
            for disciplina in sorted(dados_modelagem["disciplina"].unique())
        },
    }

    salvar_modelo(modelo, diretorio / "modelo_final.joblib")
    salvar_json(metricas, diretorio / "metricas.json")
    salvar_json(opcoes, diretorio / "opcoes_modelo.json")
    panorama.to_csv(diretorio / "panorama.csv", index=False)
    amostra_shap.to_csv(diretorio / "amostra_shap.csv", index=False)
    return metricas


def analisar_argumentos() -> argparse.Namespace:
    """Lê os caminhos informados na linha de comando"""
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument(
        "--dados",
        type=Path,
        default=RAIZ_PROJETO / "data" / "raw",
        help="Diretório dos CSVs DESCRITORES-<ANO>-ESCOLA-REDE-<REDE>.csv.",
    )
    analisador.add_argument(
        "--artefatos",
        type=Path,
        default=RAIZ_PROJETO / "artefatos",
        help="Diretório de saída do modelo e dos agregados.",
    )
    return analisador.parse_args()


def main() -> None:
    """Executa o treinamento completo"""
    argumentos = analisar_argumentos()
    print(f"Carregando bases de {argumentos.dados.resolve()}...")
    dados = carregar_bases_descritores(argumentos.dados)
    dados_modelagem, limite = preparar_dados_modelagem(dados)
    print(f"Treinando com {len(dados_modelagem):,} registros; limite = {limite:.1f}%...")
    metricas = criar_artefatos_dashboard(dados_modelagem, limite, argumentos.artefatos)
    print(
        "Artefatos concluídos. Acurácia balanceada: "
        f"{metricas['acuracia_balanceada'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()

