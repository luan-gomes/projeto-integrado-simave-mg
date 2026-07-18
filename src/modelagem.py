from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


ANOS_MODELAGEM: tuple[int, ...] = (2015, 2016, 2017, 2019, 2021, 2022, 2023, 2024)
PADRAO_ARQUIVOS = "DESCRITORES-*-ESCOLA-REDE-*.csv"

COLUNAS_14: tuple[str, ...] = (
    "tipo", "codigo_regional", "regional", "codigo_municipio", "municipio",
    "codigo_escola", "escola", "edicao", "rede", "etapa", "disciplina",
    "descritor_nome", "descritor_descricao", "acerto",
)
COLUNAS_15: tuple[str, ...] = (
    "tipo", "codigo_regional", "regional", "codigo_municipio", "municipio",
    "codigo_escola", "escola", "edicao", "rede", "codigo_etapa", "etapa",
    "disciplina", "descritor_nome", "descritor_descricao", "acerto",
)
COLUNAS_MODELAGEM: tuple[str, ...] = (
    "Ano", "rede", "etapa", "disciplina", "descritor_nome",
    "descritor_descricao", "acerto",
)
VARIAVEIS_MODELO: tuple[str, ...] = (
    "Ano", "rede", "etapa", "disciplina", "descricao_padronizada",
)
VARIAVEIS_CATEGORICAS: tuple[str, ...] = (
    "rede", "etapa", "disciplina", "descricao_padronizada",
)

CONTEUDOS_PORTUGUES = (
    "LOCALIZAR INFORMACOES EXPLICITAS|INFERIR INFORMACOES|INFERIR O SENTIDO|"
    "INFORMACAO IMPLICITA|IDENTIFICAR O TEMA|SENTIDO GLOBAL|FATO.*OPINIAO|"
    "OPINIAO.*FATO|FINALIDADE DE UM TEXTO"
)
CONTEUDOS_PORTUGUES_AMPLO = (
    "LOCALIZ|INFORMAC|INFER|TEMA|ASSUNTO|FATO|OPINIAO|INTERPRET|FINALIDADE"
)
CONTEUDOS_MATEMATICA_AMPLO = (
    "OPERAC|NUMER|RACIONAL|FRAC|PORCENT|PROPORC|TABEL|GRAF"
)
CONTEUDOS_MATEMATICA = (
    r"ADICAO|SUBTRACAO|MULTIPLICACAO|DIVISAO|OPERACOES|NUMERO RACIONAL|"
    r"NUMEROS RACIONAIS|FRACAO|FRACOES|PORCENTAGEM|PERCENTUAL|"
    r"PROPORCIONALIDADE|PROPORCAO|\bTABELA\b|\bTABELAS\b|\bGRAFICO\b|\bGRAFICOS\b"
)


@dataclass(frozen=True)
class ResultadoValidacao:
    """Resultados de uma validação cruzada."""

    pontuacoes: np.ndarray
    media: float
    desvio_padrao: float


def validar_nome_arquivo(caminho: str | Path) -> None:
    """Valida o padrão de nome usado pelo carregamento automático"""
    padrao = r"^DESCRITORES-(\d{4})-ESCOLA-REDE-(ESTADUAL|MUNICIPAL)\.csv$"
    if re.fullmatch(padrao, Path(caminho).name, flags=re.IGNORECASE) is None:
        raise ValueError(
            f"Nome fora do padrão. Renomeie o arquivo {Path(caminho).name} para "
            "DESCRITORES-<ANO>-ESCOLA-REDE-<ESTADUAL|MUNICIPAL>.csv."
        )


def extrair_ano(nome_arquivo: str, anos: Sequence[int] = ANOS_MODELAGEM) -> int:
    """Extrai do nome do arquivo um dos anos considerados na modelagem"""
    anos_texto = "|".join(str(ano) for ano in anos)
    resultado = re.search(anos_texto, Path(nome_arquivo).name)
    if resultado is None:
        raise ValueError(f"Não foi possível identificar o ano no arquivo: {nome_arquivo}")
    return int(resultado.group())


def ler_csv_descritores(caminho_csv: str | Path) -> pd.DataFrame:
    """Lê um CSV, pulando as três linhas anteriores ao cabeçalho"""
    return pd.read_csv(caminho_csv, skiprows=3, low_memory=False)


def organizar_colunas_base(dados: pd.DataFrame, ano: int) -> pd.DataFrame:
    """Aplica os nomes das colunas conforme a quantidade de colunas presente na base"""
    dados = dados.copy()
    if dados.shape[1] == 14:
        dados.columns = COLUNAS_14
    elif dados.shape[1] == 15:
        dados.columns = COLUNAS_15
    else:
        raise ValueError(
            f"A base de {ano} possui {dados.shape[1]} colunas. "
            "Eram esperadas 14 ou 15 colunas."
        )
    dados["Ano"] = ano
    return dados.loc[:, COLUNAS_MODELAGEM]


def localizar_arquivos_descritores(
    diretorio: str | Path,
    anos: Sequence[int] = ANOS_MODELAGEM,
    padrao: str = PADRAO_ARQUIVOS,
) -> list[Path]:
    """Localiza somente CSVs padronizados e pertencentes aos anos informados."""
    arquivos = sorted(Path(diretorio).glob(padrao))
    return [arquivo for arquivo in arquivos if any(str(ano) in arquivo.name for ano in anos)]


def carregar_bases_descritores(
    diretorio: str | Path,
    anos: Sequence[int] = ANOS_MODELAGEM,
) -> pd.DataFrame:
    """Carrega e concatena as bases anuais"""
    arquivos = localizar_arquivos_descritores(diretorio, anos)
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {Path(diretorio).resolve()} com o padrão "
            f"{PADRAO_ARQUIVOS}."
        )

    bases: list[pd.DataFrame] = []
    for arquivo in arquivos:
        validar_nome_arquivo(arquivo)
        ano = extrair_ano(arquivo.name, anos)
        base = organizar_colunas_base(ler_csv_descritores(arquivo), ano)
        bases.append(base)
    return pd.concat(bases, ignore_index=True)


def padronizar_texto(valor: object) -> object:
    """Remove espaços, acentos e diferenças de caixa de um texto"""
    if pd.isna(valor):
        return valor
    texto = str(valor).strip().upper()
    return "".join(
        caractere
        for caractere in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caractere) != "Mn"
    )


def padronizar_rede(valor: object) -> object:
    """Padroniza as variações de escrita das redes municipal e estadual presentes na variável rede"""
    texto = padronizar_texto(valor)
    if pd.isna(texto):
        return texto
    if "MUNICIP" in str(texto):
        return "MUNICIPAL"
    if "ESTAD" in str(texto):
        return "ESTADUAL"
    return texto


def filtrar_publico_modelagem(dados: pd.DataFrame) -> pd.DataFrame:
    """Padroniza os dados das colunas rede, etapa e disciplina, e filtra a base dedados
    por: rede MUNICIPAL ou ESTADUAL; etapa 5º ANO EF ou 9º ANO EF; e disciplina MATEMATICA
    ou LINGUA PORTUGUESA
    """
    dados = dados.copy()
    dados["rede"] = dados["rede"].apply(padronizar_rede)
    dados["etapa"] = dados["etapa"].apply(padronizar_texto)
    dados["disciplina"] = dados["disciplina"].apply(padronizar_texto)
    return dados.loc[
        dados["rede"].isin(["MUNICIPAL", "ESTADUAL"])
        & dados["etapa"].isin(["5º ANO EF", "9º ANO EF"])
        & dados["disciplina"].isin(["MATEMATICA", "LINGUA PORTUGUESA"])
    ].copy()


def selecionar_conteudos(dados: pd.DataFrame) -> pd.DataFrame:
    """Seleciona os conteúdos de Português e Matemática"""
    dados = dados.copy()
    dados["descricao_padronizada"] = dados["descritor_descricao"].apply(padronizar_texto)
    filtro_portugues = (
        dados["disciplina"].eq("LINGUA PORTUGUESA")
        & dados["descricao_padronizada"].str.contains(
            CONTEUDOS_PORTUGUES, na=False, regex=True
        )
    )
    filtro_matematica = (
        dados["disciplina"].eq("MATEMATICA")
        & dados["descricao_padronizada"].str.contains(
            CONTEUDOS_MATEMATICA, na=False, regex=True
        )
    )
    return dados.loc[filtro_portugues | filtro_matematica].copy()


def selecionar_conteudos_amplo(dados: pd.DataFrame) -> pd.DataFrame:
    """Aplica a primeira seleção exploratória de conteúdos feita no notebook"""
    dados = dados.copy()
    dados["descricao_padronizada"] = dados["descritor_descricao"].apply(padronizar_texto)
    filtro_portugues = (
        dados["disciplina"].eq("LINGUA PORTUGUESA")
        & dados["descricao_padronizada"].str.contains(
            CONTEUDOS_PORTUGUES_AMPLO, na=False, regex=True
        )
    )
    filtro_matematica = (
        dados["disciplina"].eq("MATEMATICA")
        & dados["descricao_padronizada"].str.contains(
            CONTEUDOS_MATEMATICA_AMPLO, na=False, regex=True
        )
    )
    return dados.loc[filtro_portugues | filtro_matematica].copy()


def limpar_percentual_acerto(dados: pd.DataFrame) -> pd.DataFrame:
    """Converte os valores da coluna acerto e remove valores BP, NP, NP/BP e ausências"""
    dados = dados.copy()
    dados["acerto"] = pd.to_numeric(
        dados["acerto"].astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )
    dados = dados.dropna(subset=["acerto"]).copy()
    if not dados["acerto"].between(0, 100).all():
        raise ValueError("Foram encontrados percentuais de acerto fora do intervalo de 0 a 100.")
    return dados


def criar_variavel_alvo(
    dados: pd.DataFrame,
    limite_acerto: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Cria o alvo binário abaixo da mediana e devolve o limite utilizado"""
    dados = dados.copy()
    limite = float(dados["acerto"].median() if limite_acerto is None else limite_acerto)
    dados["alvo"] = (dados["acerto"] < limite).astype(int)
    return dados, limite


def preparar_dados_modelagem(
    dados: pd.DataFrame,
    limite_acerto: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Executa os filtros, a limpeza e a criação do alvo"""
    dados = filtrar_publico_modelagem(dados)
    dados = selecionar_conteudos(dados)
    dados = limpar_percentual_acerto(dados)
    return criar_variavel_alvo(dados, limite_acerto)


def separar_treino_teste(
    dados: pd.DataFrame,
    tamanho_teste: float = 0.20,
    semente: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divisão do conjunto de dados em treino e teste"""
    caracteristicas = dados.loc[:, VARIAVEIS_MODELO].copy()
    alvo = dados["alvo"].copy()
    return train_test_split(
        caracteristicas,
        alvo,
        test_size=tamanho_teste,
        random_state=semente,
        stratify=alvo,
    )


def criar_preprocessador() -> ColumnTransformer:
    """Cria o One-Hot Encoding das variáveis categóricas do modelo."""
    return ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                list(VARIAVEIS_CATEGORICAS),
            )
        ],
        remainder="passthrough",
    )


def criar_pipeline_modelo_final() -> Pipeline:
    """Cria o pipeline Random Forest"""
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[("preprocessamento", criar_preprocessador()), ("modelo", modelo)]
    )


def criar_pipeline_regressao_logistica() -> Pipeline:
    """Cria a Regressão Logística"""
    return Pipeline(
        steps=[
            ("preprocessamento", criar_preprocessador()),
            ("modelo", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def criar_pipeline_arvore_decisao() -> Pipeline:
    """Cria a Árvore de Decisão"""
    return Pipeline(
        steps=[
            ("preprocessamento", criar_preprocessador()),
            (
                "modelo",
                DecisionTreeClassifier(
                    max_depth=15,
                    min_samples_leaf=50,
                    random_state=42,
                ),
            ),
        ]
    )


def avaliar_validacao_cruzada(
    pipeline: Pipeline,
    caracteristicas: pd.DataFrame,
    alvo: pd.Series,
    numero_divisoes: int = 3,
) -> ResultadoValidacao:
    """Avalia um pipeline por acurácia balanceada"""
    validacao = StratifiedKFold(n_splits=numero_divisoes, shuffle=True, random_state=42)
    pontuacoes = cross_val_score(
        pipeline,
        caracteristicas,
        alvo,
        cv=validacao,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    return ResultadoValidacao(pontuacoes, float(pontuacoes.mean()), float(pontuacoes.std()))


def treinar_modelo_final(
    caracteristicas_treino: pd.DataFrame,
    alvo_treino: pd.Series,
) -> Pipeline:
    """Cria e ajusta o modelo final com o conjunto de treinamento"""
    pipeline = criar_pipeline_modelo_final()
    pipeline.fit(caracteristicas_treino, alvo_treino)
    return pipeline


def avaliar_modelo_final(
    pipeline: Pipeline,
    caracteristicas_teste: pd.DataFrame,
    alvo_teste: pd.Series,
) -> dict[str, object]:
    """Calcula as métricas completas utilizadas no relatório e no dashboard"""
    previsoes = pipeline.predict(caracteristicas_teste)
    probabilidades = pipeline.predict_proba(caracteristicas_teste)[:, 1]
    return {
        "acuracia": float(accuracy_score(alvo_teste, previsoes)),
        "acuracia_balanceada": float(balanced_accuracy_score(alvo_teste, previsoes)),
        "auc_roc": float(roc_auc_score(alvo_teste, probabilidades)),
        "matriz_confusao": confusion_matrix(alvo_teste, previsoes, labels=[0, 1]).tolist(),
        "relatorio_classificacao": classification_report(
            alvo_teste,
            previsoes,
            labels=[0, 1],
            target_names=["Igual ou acima da mediana", "Abaixo da mediana"],
            output_dict=True,
        ),
    }


def preparar_registro_predicao(
    ano: int,
    rede: str,
    etapa: str,
    disciplina: str,
    descricao: str,
) -> pd.DataFrame:
    """Monta uma observação no formato esperado pelo pipeline salvo"""
    return pd.DataFrame(
        [
            {
                "Ano": int(ano),
                "rede": padronizar_rede(rede),
                "etapa": padronizar_texto(etapa),
                "disciplina": padronizar_texto(disciplina),
                "descricao_padronizada": padronizar_texto(descricao),
            }
        ],
        columns=VARIAVEIS_MODELO,
    )


def salvar_modelo(pipeline: Pipeline, caminho: str | Path, compressao: int = 3) -> None:
    """Salva o pipeline completo, incluindo o pré-processamento ajustado"""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, caminho, compress=compressao)


def carregar_modelo(caminho: str | Path) -> Pipeline:
    """Carrega um pipeline salvo por :func:`salvar_modelo`"""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {caminho}")
    return joblib.load(caminho)


def salvar_json(conteudo: dict[str, object], caminho: str | Path) -> None:
    """Salva metadados em JSON, convertendo tipos numéricos do NumPy"""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(conteudo, ensure_ascii=False, indent=2, default=lambda valor: valor.item()),
        encoding="utf-8",
    )
