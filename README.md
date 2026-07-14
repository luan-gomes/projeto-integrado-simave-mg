# projeto-integrado-simave-mg

## Sobre este repositório

Este repositório é utilizado na disciplina de Projeto Integrado em Ciência de Dados, no semestre 2026.2, da Especialização em Ciência de Dados da Universidade Estadual de Montes Claros (UNIMONTES).

## Sobre o projeto

O objetivo do deste projeto é predizer o desempenho em Língua Portuguesa e Matemática, do 5º e 9º ano do Ensino Fundamental, das escolas Estaduais e Municipais do estado de Minas Gerais, utilizando a série histórica de 2015-2025 disponível no Portal das Avaliações Educacionais de Minas Gerais, para que Instituições e pessoas interessadas possam tomar medidas e criar políticas com maior eficiência a fim de melhorar esses indicadores.

## Sobre os dados

**Fonte:** Portal das Avaliações Educacionais de Minas Gerais (SIMAVE)
**Link:** Arquivos das Redes Estadual e Municipal presentes na seção Descritores no Portal das Avaliações do Estado de MG [https://avaliacoes.educacao.mg.gov.br/dados-abertos](https://avaliacoes.educacao.mg.gov.br/dados-abertos)
**Data de acesso da coleta de dados:** 05/06/2026

**Procedimento de obtenção dos dados:**
- Ao acessar a página supracitada, os dados foram baixados utilizando os links de cada ano das Redes Estadual e Municipal presentes na seção Descritores.
- Cada link direcionava para uma planilha `.xlsx` no Planilhas (Google Drive) e cada planilha foi transformada no formato `.csv` utilizando a função `Arquivo > Baixar > Valores separados por vírgula (.csv)` disponível na aplicação Planilhas do Google.
- Após o download, os arquivos `.csv` foram renomeados para seguir o padrão `DESCRITORES-<ANO>-ESCOLA-REDE-<MUNICIPAL|ESTADUAL>.csv`.


**Observações sobre disponibilidade dos dados:**
- Ainda que disponíveis na página, não foi possível acessar o link das seguintes planilhas da Rede Estadual referentes aos anos de 2017 e 2018 e do ano de 2017 da Rede Municipal
- Além disso, os anos de 2020 e 2025 não possuem dados disponíveis na página

**Observações sobre características dos arquivos de dados:**
- As planilhas exportadas do portal têm três linhas introdutórias antes do cabeçalho real. A função de carga já pula essas linhas, padroniza nomes de colunas e remove linhas de rodapé como fonte/observação.

## Como executar
Antes de instalar as dependências é recomendado criar uma venv (confira este [link](https://docs.python.org/pt-br/3/library/venv.html)) para mais informações sobre a criação de ambientes virtuais em python

```
git clone
pip install -r requirements.txt
```

## Estrutura

```
projeto-integrado-simave-mg/
├── data/
│   ├── raw/ # Dados obtidos conforme seção "Sobre os dados"
├── docs/
|   └── escopo.pdf # Entrega M1 - definição inicial do escopo do projeto
├── notebooks/
│   ├── exploration/
|       └── analise-exploratoria-de-dados.ipynb # Rascunho entrega M2
│   └── final/
|       ├── analise-exploratoria-de-dados.ipynb # Entrega M2 - EDA
|       └── relatorio-modelagem.ipynb # Entrega M3 - Modelagem
├── src/
│   └── simave_preprocessing.py # Funções reutilizáveis de pré-processamento de dados
├── .gitignore
├── requirements.txt
└── README.md
```
OBS: Os dados no diretório data/raw não estão versionados por conta do tamanho dos arquivos.
