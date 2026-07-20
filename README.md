# Desempenho escolar nas disciplinas de Língua Portuguesa e Matemática do 5º e 9º anos do Ensino Fundamental | SIMAVE-MG

## Sobre este repositório

Este repositório é utilizado na disciplina de Projeto Integrado em Ciência de Dados, no semestre 2026.2, da Especialização em Ciência de Dados da Universidade Estadual de Montes Claros (UNIMONTES).

## Sobre o projeto

O objetivo do deste projeto é predizer o desempenho em Língua Portuguesa e Matemática, do 5º e 9º ano do Ensino Fundamental, das escolas Estaduais e Municipais do estado de Minas Gerais, utilizando a série histórica de 2015-2025 disponível no Portal das Avaliações Educacionais de Minas Gerais, para que Instituições e pessoas interessadas possam tomar medidas e criar políticas com maior eficiência a fim de melhorar esses indicadores.

## Sobre os dados

**Fonte:** Portal das Avaliações Educacionais de Minas Gerais (SIMAVE)  
**Link:** Arquivos das Redes Estadual e Municipal presentes na seção Descritores no Portal das Avaliações do Estado de MG [https://avaliacoes.educacao.mg.gov.br/dados-abertos](https://avaliacoes.educacao.mg.gov.br/dados-abertos)  
**Data de acesso e coleta de dados:** 05/06/2026

**Procedimento de obtenção dos dados:**
- Ao acessar a página supracitada, os dados foram baixados utilizando os links de cada ano das Redes Estadual e Municipal presentes na seção Descritores.
- Cada link direcionava para uma planilha `.xlsx` no Planilhas (Google Drive) e cada planilha foi transformada no formato `.csv` utilizando a função `Arquivo > Baixar > Valores separados por vírgula (.csv)` disponível na aplicação Planilhas do Google.
- Após o download, os arquivos `.csv` foram renomeados para seguir o padrão `DESCRITORES-<ANO>-ESCOLA-REDE-<MUNICIPAL|ESTADUAL>.csv`.

**Observações sobre disponibilidade dos dados:**
- Ainda que disponíveis na página, não foi possível acessar o link das seguintes planilhas da Rede Estadual referentes aos anos de 2017 e 2018 e do ano de 2017 da Rede Municipal
- Além disso, os anos de 2020 e 2025 não possuem dados disponíveis na página

**Observações sobre características dos arquivos de dados:**
- As planilhas exportadas do portal têm três linhas introdutórias antes do cabeçalho real. A função de carga já pula essas linhas, padroniza nomes de colunas e remove linhas de rodapé como fonte/observação.

## Executar o dashboard

Recomenda-se Python 3.11, 3.12 ou 3.13 em um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Para facilitar o processo de execução do dashboard, o app usa os artefatos versionados em `artefatos/`. Sendo assim, não é necessário baixar as bases brutas para visualizar a entrega. Mas é possível executar o processo compelto seguindo a seção abaixo.

## Reproduzir o treinamento

Os dados brutos não são versionados por causa do tamanho. Baixe as planilhas da seção **Descritores** do [Portal das Avaliações de Minas Gerais](https://avaliacoes.educacao.mg.gov.br/dados-abertos), exporte-as como CSV e coloque os arquivos em `data/raw/`.

Cada nome deve seguir exatamente um destes formatos:

```text
DESCRITORES-2022-ESCOLA-REDE-ESTADUAL.csv
DESCRITORES-2022-ESCOLA-REDE-MUNICIPAL.csv
```

Por exemplo, o arquivo fornecido como `Descritores_2022.xlsx - ESCOLA.csv` contém a rede estadual e deve ser renomeado para `DESCRITORES-2022-ESCOLA-REDE-ESTADUAL.csv`. Se uma exportação contiver mais de uma rede, separe-a antes de aplicar os nomes padronizados. A leitura pula as três linhas introdutórias do arquivo e aceita as estruturas históricas com 14 ou 15 colunas.

Para reconstruir o modelo e os agregados:

```bash
python scripts/treinar_modelo.py --dados data/raw
```

O script executa as mesmas etapas do notebook M3: padronização textual, recorte de público e conteúdos, remoção de percentuais inválidos, criação do alvo pela mediana, divisão estratificada 80/20, One-Hot Encoding e treinamento da Random Forest.

## Estrutura

```text
projeto-integrado-simave-mg/
├── app/
│   └── app.py                         # Entrega M4 - Dashboard Streamlit
├── artefatos/
│   ├── modelo_final.joblib            # Pipeline ajustado
│   ├── metricas.json                  # Avaliação no teste
│   ├── panorama.csv                   # Dados agregados do panorama
│   ├── amostra_shap.csv               # Amostra anônima para SHAP global
│   └── opcoes_modelo.json             # Valores válidos dos controles
├── data/
│   ├── raw/                           # Dados obtidos conforme seção "Sobre os dados"
├── docs/
|   └── escopo.pdf                     # Entrega M1 - definição inicial do escopo do projeto
├── notebooks/
│   ├── exploration/analise-exploratoria-de-dados.ipynb # Primeira versão entrega M2
│   └── final/
|       ├── analise-exploratoria-de-dados.ipynb     # Entrega M2 - EDA
|       └── relatorio-modelagem.ipynb               # Entrega M3 - Modelagem
├── scripts/
│   └── treinar_modelo.py              # Reprodução dos artefatos
├── src/
│   ├── modelagem.py                   # Pré-processamento e treinamento
│   ├── interpretabilidade.py          # Funções SHAP
│   └── simave_preprocessing.py        # Funções da análise exploratória
├── .gitignore
├── requirements.txt
└── README.md
```

## Limitações

- O limiar de 57% é uma referência estatística da amostra, não uma classificação pedagógica oficial.
- Frequência, nível socioeconômico, infraestrutura, formação docente e participação não fazem parte das entradas.
- Registros com `BP`, `NP`, `NP/BP`, hífen ou sem percentual foram removidos.
- Após os filtros, 2015 e 2017 não permaneceram na modelagem; o produto representa dados de 2016, 2019 e 2021–2024.
- SHAP explica como o modelo usa as informações; não demonstra causalidade.
- O resultado serve como apoio à análise e não substitui o julgamento de profissionais da educação.