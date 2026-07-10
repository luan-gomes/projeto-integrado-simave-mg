"""Funcoes de limpeza e preparacao dos dados de descritores do SIMAVE."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence
import glob
import unicodedata

import numpy as np
import pandas as pd


COLUMN_RENAMES: Mapping[str, str] = {
    "% acerto": "acerto",
    "%.acerto": "acerto",
    "descritores_nome": "descritor_nome",
    "descritores_descricao": "descritor_descricao",
    "cd_etapa": "codigo_etapa",
    "cd_escola": "codigo_escola",
}

CODE_COLUMNS: tuple[str, ...] = (
    "codigo_regional",
    "codigo_municipio",
    "codigo_escola",
    "codigo_etapa",
)

DEFAULT_DESCRIPTOR_PATTERN = "DESCRITORES-*-ESCOLA-REDE-*.csv"


def strip_accents(value: object) -> str:
    """Remove acentos e normaliza texto para comparacoes."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.upper()


def normalize_category(series: pd.Series) -> pd.Series:
    """Padroniza categorias preservando uma Serie do pandas."""
    return series.astype("string").map(strip_accents)


def read_descriptor_csv(path: str | Path, skiprows: int = 3) -> pd.DataFrame:
    """Le uma planilha CSV exportada do portal e aplica nomes padronizados."""
    df = pd.read_csv(path, skiprows=skiprows)
    df = df.rename(columns=COLUMN_RENAMES)
    return remove_footer_rows(df)


def remove_footer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas de fonte/rodape que aparecem no fim de algumas planilhas."""
    if "tipo" not in df.columns:
        return df.copy()

    tipo_norm = normalize_category(df["tipo"])
    return df.loc[tipo_norm.eq("ESCOLA")].copy()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza categorias, codigos e variaveis auxiliares usadas nos modelos."""
    out = df.rename(columns=COLUMN_RENAMES).copy()

    for column in CODE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    if "edicao" in out.columns:
        out["edicao"] = pd.to_numeric(out["edicao"], errors="coerce").astype("Int64")

    for column in ["tipo", "rede", "etapa", "disciplina", "regional", "municipio"]:
        if column in out.columns:
            out[column] = out[column].astype("string").str.strip()
            out[f"{column}_norm"] = normalize_category(out[column])

    if "disciplina_norm" in out.columns:
        out["is_matematica"] = out["disciplina_norm"].str.contains("MATEMATICA", na=False).astype(int)

    if "rede_norm" in out.columns:
        out["is_estadual"] = out["rede_norm"].str.contains("ESTADUAL", na=False).astype(int)

    return out


def convert_acerto(df: pd.DataFrame, column: str = "acerto") -> pd.DataFrame:
    """Converte o percentual de acerto para numerico e registra o status original."""
    if column not in df.columns:
        raise KeyError(f"Coluna obrigatoria ausente: {column}")

    out = df.copy()
    raw = out[column].astype("string").str.strip()
    raw_upper = raw.str.upper()
    numeric = (
        raw.str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    out[f"{column}_original"] = raw
    out[column] = numeric
    out[f"{column}_status"] = np.select(
        [
            raw_upper.eq("BP").fillna(False).to_numpy(bool),
            raw_upper.eq("-").fillna(False).to_numpy(bool),
            (raw_upper.isna() | raw_upper.eq("")).fillna(False).to_numpy(bool),
            numeric.isna().fillna(False).to_numpy(bool),
        ],
        [
            "baixa_participacao",
            "sem_valor",
            "vazio",
            "invalido",
        ],
        default="ok",
    )
    return out


def filter_etapas(df: pd.DataFrame, etapas: Sequence[int] = (5, 9)) -> pd.DataFrame:
    """Mantem apenas as etapas informadas pelo codigo da etapa."""
    if "codigo_etapa" not in df.columns:
        raise KeyError("Coluna obrigatoria ausente: codigo_etapa")

    return df.loc[df["codigo_etapa"].isin(etapas)].copy()


def filter_valid_target(df: pd.DataFrame, target: str = "acerto") -> pd.DataFrame:
    """Remove registros sem alvo numerico valido."""
    if target not in df.columns:
        raise KeyError(f"Coluna obrigatoria ausente: {target}")

    return df.loc[df[target].notna()].copy()


def summarize_target_status(df: pd.DataFrame, status_column: str = "acerto_status") -> pd.DataFrame:
    """Resume a qualidade do alvo por ano, etapa, disciplina e rede."""
    if status_column not in df.columns:
        raise KeyError(f"Coluna obrigatoria ausente: {status_column}")

    group_columns = [
        column
        for column in ["edicao", "rede", "etapa", "disciplina"]
        if column in df.columns
    ]
    summary = (
        df.groupby(group_columns + [status_column], dropna=False)
        .size()
        .rename("registros")
        .reset_index()
    )
    total = summary.groupby(group_columns, dropna=False)["registros"].transform("sum")
    summary["percentual"] = (summary["registros"] / total * 100).round(2)
    return summary.sort_values(group_columns + [status_column]).reset_index(drop=True)


def make_binary_target(
    df: pd.DataFrame,
    threshold: float,
    target: str = "acerto",
    output: str = "baixo_desempenho",
) -> pd.DataFrame:
    """Cria alvo binario. A escolha do limiar deve ser justificada no notebook."""
    if target not in df.columns:
        raise KeyError(f"Coluna obrigatoria ausente: {target}")

    out = df.copy()
    out[output] = (out[target] < threshold).astype(int)
    return out


def make_group_quantile_target(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    quantile: float = 0.25,
    target: str = "acerto",
    output: str = "baixo_desempenho",
    threshold_output: str = "limiar_grupo",
) -> pd.DataFrame:
    """Cria alvo binario usando um limiar por grupo, como etapa e disciplina."""
    missing = [column for column in [*group_columns, target] if column not in df.columns]
    if missing:
        raise KeyError(f"Colunas obrigatorias ausentes: {missing}")

    out = df.copy()
    thresholds = out.groupby(list(group_columns))[target].transform(lambda value: value.quantile(quantile))
    out[threshold_output] = thresholds
    out[output] = (out[target] <= thresholds).astype(int)
    return out


def find_descriptor_files(
    data_dir: str | Path,
    pattern: str = DEFAULT_DESCRIPTOR_PATTERN,
) -> list[Path]:
    """Localiza arquivos CSV de descritores em um diretorio."""
    data_dir = Path(data_dir)
    return sorted(data_dir.glob(pattern))


def load_descriptor_files(pattern: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Carrega varios CSVs de descritores em um unico DataFrame padronizado."""
    if isinstance(pattern, (str, Path)):
        files = [Path(file) for file in sorted(glob.glob(str(pattern)))]
    else:
        files = [Path(file) for file in pattern]

    frames = [read_descriptor_csv(file) for file in files]
    if not frames:
        raise FileNotFoundError(f"Nenhum arquivo encontrado para: {pattern}")

    df = pd.concat(frames, ignore_index=True)
    df = standardize_columns(df)
    df = convert_acerto(df)
    return df
