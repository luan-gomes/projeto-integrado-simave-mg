"""Dashboard final do projeto de modelagem dos descritores do SIMAVE-MG."""

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shap
import streamlit as st


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from src.interpretabilidade import (
    ROTULOS_VARIAVEIS,
    calcular_shap_agregado,
    criar_explicacao_local,
    resumir_importancia_global,
)
from src.modelagem import (
    VARIAVEIS_MODELO,
    carregar_modelo,
    preparar_registro_predicao,
)


CAMINHO_ARTEFATOS = RAIZ_PROJETO / "artefatos"
COR_AZUL = "#185FA5"
COR_LARANJA = "#F28E2B"

st.set_page_config(
    page_title="Desempenho escolar nas disciplinas de Ling. Portuguesa e Matemática do 5º e 9º ano do EF | SIMAVE-MG",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {border: 1px solid #e4e9f0;
      padding: .85rem 1rem; border-radius: .75rem;}
    .ato {color: #185FA5; font-size: .78rem; font-weight: 700;
      letter-spacing: .08em; text-transform: uppercase;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def carregar_recursos():
    """Carrega uma única vez o pipeline e os metadados do produto."""
    modelo = carregar_modelo(CAMINHO_ARTEFATOS / "modelo_final.joblib")
    metricas = json.loads((CAMINHO_ARTEFATOS / "metricas.json").read_text(encoding="utf-8"))
    opcoes = json.loads((CAMINHO_ARTEFATOS / "opcoes_modelo.json").read_text(encoding="utf-8"))
    return modelo, metricas, opcoes


@st.cache_data
def carregar_dados_dashboard() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega somente agregados e uma amostra anônima do conjunto de teste."""
    panorama = pd.read_csv(CAMINHO_ARTEFATOS / "panorama.csv")
    amostra = pd.read_csv(CAMINHO_ARTEFATOS / "amostra_shap.csv")
    return panorama, amostra


try:
    modelo, metricas, opcoes = carregar_recursos()
    panorama, amostra_shap = carregar_dados_dashboard()
except FileNotFoundError as erro:
    st.error(f"Não foi possível iniciar o dashboard: {erro}")
    st.code("python scripts/treinar_modelo.py --dados data/raw")
    st.stop()


st.title("Desempenho escolar no SIMAVE-MG")
st.markdown(
    "<span class='ato'>Ato 1 · O problema</span>",
    unsafe_allow_html=True,
)
st.write(
    "Quais combinações de ano, rede, etapa, disciplina e conteúdo merecem atenção "
    "por apresentarem maior chance de desempenho abaixo da mediana histórica?"
)
st.warning(
    "**Limitações — leia antes de usar:**"
    "O corte de 57% é estatístico, não pedagógico; "
    "O modelo utilizou apenas as informações disponíveis nas bases analisadas. Outros fatores que podem influenciar o desempenho dos estudantes, como frequência escolar, " 
    "condições socioeconômicas, infraestrutura da escola, formação dos professores e participação dos alunos, não estavam disponíveis;"
    "Foram removidos registros sem percentual de acerto válido, incluindo os códigos BP, NP e NP/BP. Além disso, após a aplicação dos filtros definidos para o estudo, "
    "as bases de 2015 e 2017 não permaneceram na etapa de modelagem;"
    "O modelo apresentou desempenho satisfatório, mas ainda realizou classificações incorretas em parte dos registros."
    "Dessa forma, seus resultados devem ser utilizados como apoio à análise e à tomada de decisão, não substituindo a avaliação realizada por profissionais da área educacional."
)

aba_panorama, aba_predicao, aba_global = st.tabs(
    ["1 · Panorama geral", "2 · Predição individual + SHAP", "3 · Interpretabilidade global"]
)

with aba_panorama:
    st.markdown("<span class='ato'>Ato 2 · Os dados</span>", unsafe_allow_html=True)
    st.subheader("Qual é a situação geral?")

    coluna_filtro, coluna_resumo = st.columns([1, 3], gap="large")
    with coluna_filtro:
        anos = sorted(panorama["Ano"].unique().tolist())
        ano_selecionado = st.select_slider(
            "Ano da avaliação", options=["Todos", *anos], value="Todos"
        )
        rede_selecionada = st.multiselect(
            "Rede de ensino",
            sorted(panorama["rede"].unique()),
            default=sorted(panorama["rede"].unique()),
        )
        disciplina_selecionada = st.multiselect(
            "Disciplina",
            sorted(panorama["disciplina"].unique()),
            default=sorted(panorama["disciplina"].unique()),
        )

    dados_filtrados = panorama.loc[
        panorama["rede"].isin(rede_selecionada)
        & panorama["disciplina"].isin(disciplina_selecionada)
    ].copy()
    if ano_selecionado != "Todos":
        dados_filtrados = dados_filtrados.loc[dados_filtrados["Ano"].eq(int(ano_selecionado))]

    total = int(dados_filtrados["registros"].sum())
    abaixo = int(dados_filtrados["abaixo_mediana"].sum())
    percentual = abaixo / total * 100 if total else 0

    with coluna_resumo:
        metrica_1, metrica_2, metrica_3 = st.columns(3)
        metrica_1.metric("Registros analisados", f"{total:,.0f}".replace(",", "."))
        metrica_2.metric("Abaixo da mediana", f"{percentual:.1f}%".replace(".", ","))
        metrica_3.metric("Ponto de referência", "57%", help="Mediana histórica do M3")

        grafico = (
            dados_filtrados.groupby(["Ano", "disciplina"], as_index=False)
            .agg(registros=("registros", "sum"), abaixo_mediana=("abaixo_mediana", "sum"))
        )
        grafico["percentual"] = grafico["abaixo_mediana"] / grafico["registros"] * 100
        figura = px.line(
            grafico,
            x="Ano",
            y="percentual",
            color="disciplina",
            markers=True,
            labels={"percentual": "Abaixo da mediana (%)", "disciplina": "Disciplina"},
            color_discrete_sequence=[COR_AZUL, COR_LARANJA],
        )
        figura.update_layout(yaxis_range=[0, 100], legend_title_text="")
        st.plotly_chart(figura, width="stretch")
    st.caption(
        "A visualização resume os 697.617 registros válidos usados na modelagem."
    )


with aba_predicao:
    st.markdown("<span class='ato'>Ato 3 · O modelo</span>", unsafe_allow_html=True)
    st.subheader("Por que este caso recebeu essa classificação?")
    entrada, resultado = st.columns([1, 1.35], gap="large")

    with entrada:
        ano = st.selectbox("Ano da avaliação", opcoes["anos"], index=len(opcoes["anos"]) - 1)
        rede = st.selectbox("Rede de ensino", opcoes["redes"])
        etapa = st.selectbox("Etapa escolar", opcoes["etapas"])
        disciplina = st.selectbox("Disciplina", opcoes["disciplinas"])
        descricoes = opcoes["conteudos_por_disciplina"][disciplina]
        descricao = st.selectbox("Conteúdo avaliado", descricoes)
        registro = preparar_registro_predicao(ano, rede, etapa, disciplina, descricao)

    probabilidade = float(modelo.predict_proba(registro)[0, 1])
    classe = int(probabilidade >= 0.5)
    rotulo = "Prioridade de atenção" if classe else "Igual ou acima da mediana"

    with resultado:
        st.metric(
            "Resultado do apoio à triagem",
            rotulo,
            f"{probabilidade * 100:.1f}% de chance de ficar abaixo da mediana".replace(".", ","),
            delta_color="inverse",
        )
        st.progress(probabilidade, text="Probabilidade estimada de desempenho abaixo de 57%")
        st.caption(
            "Probabilidade não é certeza. O resultado compara o caso aos padrões históricos do modelo."
        )

        explicacao = criar_explicacao_local(modelo, registro)
        figura_local, eixo = plt.subplots(figsize=(8, 4.3))
        shap.plots.waterfall(explicacao, max_display=5, show=False)
        eixo.set_title("Como cada informação alterou a previsão", loc="left", fontsize=12)
        st.pyplot(figura_local, clear_figure=True, width="stretch")
        st.caption("Vermelho aumenta e azul reduz a chance de desempenho abaixo da mediana.")


with aba_global:
    st.markdown("<span class='ato'>Ato 3 · O modelo</span>", unsafe_allow_html=True)
    st.subheader("Quais fatores mais influenciam o modelo no geral?")
    valores_shap, _ = calcular_shap_agregado(modelo, amostra_shap)
    importancia = resumir_importancia_global(valores_shap)

    modo = st.radio(
        "Forma de leitura",
        ["Importância média", "Distribuição dos efeitos"],
        horizontal=True,
    )
    if modo == "Importância média":
        figura_global = px.bar(
            importancia,
            x="importancia",
            y="variavel",
            orientation="h",
            labels={"importancia": "Impacto SHAP médio (valor absoluto)", "variavel": ""},
            color_discrete_sequence=[COR_AZUL],
        )
        st.plotly_chart(figura_global, width="stretch")
    else:
        distribuicao = pd.DataFrame(
            valores_shap,
            columns=[ROTULOS_VARIAVEIS[variavel] for variavel in VARIAVEIS_MODELO],
        ).melt(var_name="Informação", value_name="Efeito SHAP")
        figura_distribuicao = px.strip(
            distribuicao,
            x="Efeito SHAP",
            y="Informação",
            color="Efeito SHAP",
        )
        figura_distribuicao.add_vline(x=0, line_dash="dash", line_color="#6b7280")
        figura_distribuicao.update_coloraxes(showscale=False)
        st.plotly_chart(figura_distribuicao, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("Classificações corretas", f"{metricas['acuracia'] * 100:.1f}%".replace(".", ","))
    m2.metric(
        "Equilíbrio entre as classes",
        f"{metricas['acuracia_balanceada'] * 100:.1f}%".replace(".", ","),
    )
    m3.metric("Capacidade de separação (AUC)", f"{metricas['auc_roc']:.3f}".replace(".", ","))

    st.markdown("<span class='ato'>Ato 4 · Limitações</span>", unsafe_allow_html=True)
    st.info(
        "Os efeitos SHAP explicam o comportamento do modelo, não relações de causa e efeito. "
        "A amostra global vem do conjunto de teste e os nomes foram agrupados em cinco conceitos "
        "para evitar expor centenas de colunas técnicas do One-Hot Encoding."
    )

st.divider()
st.caption(
    "Projeto Integrado em Ciência de Dados · UNIMONTES · Dados públicos do SIMAVE-MG (2016–2024)"
)
