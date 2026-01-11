<h1 id="inicio" align="center">
  ChurnInsight — Data Science<br>
  <img src="https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow" alt="Status" width="180" height="30" />
  <img src="https://img.shields.io/badge/Versão-1.2.1-blue" alt="Versão" width="89" height="30" />
    <img src="https://img.shields.io/badge/Tests-5%2F5%20Passed-brightgreen" alt="Tests" width="111" height="30" />
</h1>

<h2 align="center">🔗 Repositórios Relacionados</h2>

O **ChurnInsight** é um projeto distribuído em múltiplos repositórios, cada um responsável por uma parte específica da solução.

Este repositório contém a parte de **Data Science**, desenvolvido em **Python**, responsável pela análise de dados, pré-processamento, treinamento do modelo preditivo e disponibilização das previsões por meio de uma API.

Além dele, o projeto conta com os seguintes repositórios complementares:

* **ChurnInsight — Backend**: desenvolvido em **Java com Spring Boot**, responsável pela orquestração da solução, regras de negócio, integrações e consumo das previsões do modelo.
* **ChurnInsight — Frontend**: responsável pela interface visual da aplicação e pelo consumo das APIs expostas pelo backend.


*   👉 [**ChurnInsight — Backend**](https://github.com/renancvitor/churninsight-backend-h12-25b) 
*   👉 [**ChurnInsight — Frontend**](https://github.com/lucasns06/churninsight-frontend) 

---

### 🚀 API em Produção (Swagger UI)
🔗 **[https://churn-hackathon.onrender.com/docs](https://churn-hackathon.onrender.com/docs)**

⚠️ **Nota para o Squad:** A documentação interativa em `/docs` é a **Single Source of Truth** para o contrato da API. Verifique sempre os schemas antes de integrar.

---

<h2 align="center">📑 Sumário</h2>

*   [Visão Geral do Projeto](#visao-geral)
*   [Fonte dos Dados](#fonte-dados)
*   [Problema de Negócio](#problema)
*   [Abordagem de Data Science](#abordagem)
*   [Tecnologias e Ferramentas](#tecnologias)
*   [Estrutura do Repositório](#estrutura)
*   [Dicionário de Dados](#dicionario)
*   [Integração com o Backend](#integracao)
*   [Métricas e Resultados](#metricas)
*   [Primeiros Entregáveis](#entregaveis)
*   [Decisões Técnicas](#decisoes)
*   [Como Executar a API](#como-executar)
*   [Validação Técnica e Garantia de Qualidade](#validacao)
*   [Deploy com Docker](#deploy)
*   [Contribuições](#contribuicoes)
*   [Licença](#licenca)
  
---

<h2 id="visao-geral" align="center">Visão Geral do Projeto</h2>

O **ChurnInsight** é uma solução desenvolvida durante o **Hackathon da Alura** com o objetivo de prever o risco de **cancelamento de clientes (churn)** em serviços recorrentes, como bancos digitais, plataformas de assinatura e soluções SaaS.

A plataforma integra **Data Science** e **Backend** para transformar dados de clientes em **insights acionáveis**, permitindo que empresas antecipem riscos de evasão e tomem decisões baseadas em dados.

O projeto foi concebido como um **MVP funcional**, com arquitetura simples, clara e preparada para evolução.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="fonte-dados" align="center">Fonte dos Dados</h2>

Dataset público via Kaggle: **[Willian Oliveira](https://www.kaggle.com/datasets/willianoliveiragibin/customer-churn/data/code)** 

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="problema" align="center">Problema de Negócio</h2>

A perda de clientes impacta diretamente a receita de negócios recorrentes.  
Identificar clientes com maior probabilidade de churn permite ações preventivas mais eficazes, reduzindo custos de aquisição e aumentando a retenção.

O ChurnInsight atua exatamente nesse ponto, oferecendo previsões claras e interpretáveis a partir de dados reais de clientes.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="abordagem" align="center">Abordagem de Data Science</h2>

A abordagem do squad para o MVP inclui:

### 🔹 1. Pré-processamento
* Limpeza de metadados (`RowNumber`, `CustomerId`, `Surname`).  
* **One-Hot Encoding** para variáveis geográficas e de gênero.  
* Normalização com `StandardScaler`, aplicada apenas ao treino (evita *data leakage*).

### 🔹 2. Engenharia de Features
Criação de indicadores de comportamento:  
* `Age_Tenure`: interação entre idade e tempo de relacionamento.  
* `Balance_Salary_Ratio`: proporção entre saldo bancário e salário estimado.  
* `High_Value_Customer`: flag para clientes acima da mediana financeira.

### 🔹 3. Modelagem e Explicabilidade
* **Modelo:** `RandomForestClassifier` (`n_estimators=200`)  
* **Estratégia:** pesos balanceados (`class_weight={0:1, 1:3}`) para focar no churn  
* **Explicabilidade Local:** a API indica as variáveis mais relevantes para o risco de churn de cada cliente

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="tecnologias" align="center">Tecnologias e Ferramentas</h2>

As tecnologias utilizadas no projeto incluem:

### Linguagens e Bibliotecas
- **🐍 Python 3** — linguagem base da solução
- **📊 pandas 2.3.3** — manipulação e análise de dados
- **📊 numpy 2.4.0** — manipulação e cálculo numérico
- **🤖 scikit-learn 1.6.1** — modelagem, pré-processamento e métricas
- **💾 joblib 1.5.3** — serialização do pipeline de Machine Learning
- **🌐 FastAPI 0.127.0** — API REST para inferência do modelo
- **🔧 Uvicorn 0.40.0** — servidor ASGI para execução da API
- **📦 pyarrow 22.0.0** — leitura e escrita de dados em formato Parquet
- **📌 pydantic >=2.0,<3.0** — validação de dados e schemas da API
- **📌 python-multipart** — upload de arquivos via API
- **📌 requests 2.31.0** — chamadas HTTP externas (quando necessário)
- **📌 httpx** — chamadas HTTP assíncronas (teste ou integração)
- **📌 pytest** — execução de testes automatizados

### Ferramentas de Apoio
- **🧪 Jupyter Notebook / Google Colab** — EDA, experimentação e prototipação
- **🔗 Git & GitHub** — versionamento de código e colaboração
- **🐳 Docker & Docker Compose** — padronização de ambiente e deploy
- **☁️ Render** — hospedagem e execução da API em produção

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="estrutura" align="center">Estrutura do Repositório</h2>

```plaintext
app/                      
├── models/
|   ├── __init__.py        
│   ├── model.joblib        # Modelo serializado        
└── main.py                 # API FastAPI

data/                       
├── Churn.csv               # Dados brutos
└── dataset.parquet         # Dados tratados

docs/                       
└── Documentação Técnica de Visualizações.md  # Gráficos e análises

notebooks/                  
└── Churn_Hackathon.ipynb   # EDA e modelagem

tests/                      
├── integration/            
│   ├── __init__.py
│   ├── test_integration_health.py
│   ├── test_integration_previsao.py
│   └── test_integration_root.py
└── unit/                   
    ├── __init__.py
    ├── test_unit_payload.py
    ├── test_unit_previsao_lote.py
    └── teste_unit_explicabilidade.py

__init__.py
.gitignore
Dockerfile
LICENSE
README.md
check_all.sh                # Script de validação total
conftest.py
docker-compose.yml
requirements.txt                   
stress_test.py                     
```

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---
<h2 id="dicionario" align="center">Dicionário de Dados</h2>

|       Coluna        |  Tipo   |         Descrição           |   Faixa Esperada (Validação)   |
| :-------------------| :-----: | :---------------------------| :------------------------------|
| **CreditScore**     | `int`   | Score financeiro do cliente | 350 – 850                      |
| **Geography**       | `str`   | País de origem              | `France`, `Germany` ou `Spain` |
| **Gender**          | `str`   | Gênero do cliente           | `Male` ou `Female`             |
| **Age**             | `int`   | Idade do cliente            | 18 – 92 anos                   |
| **Tenure**          | `int`   | Anos de relacionamento      | 0 – 10 anos                    |
| **Balance**         | `float` | Saldo em conta              | R$ 0 – 500.000                 |
| **EstimatedSalary** | `float` | Salário anual estimado      | Decimal Positivo               |

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="integracao" align="center">Integração com o Backend</h2>

O serviço de **Data Science (FastAPI)** fornece previsões de churn para o **Backend**.  

### 🛠 Artefatos de Integração
- **model.joblib** — pipeline de ML serializado (modelo + pré-processamento).  
- **API FastAPI** — endpoint `/previsao` exposto em produção via **Render**.  
- **Dockerfile & Docker Compose** — garantem consistência do ambiente e facilitam execução local ou em nuvem.  

### 🔁 Fluxo de Comunicação
1. Backend envia JSON com dados do cliente para a API Python.
2. A API executa a inferência usando `model.joblib`.
3. Retorna `previsao`, `probabilidade`, `nivel_risco` e `explicabilidade`.

📥 **Entrada**

POST https://churn-hackathon.onrender.com/previsao

Content-Type: application/json

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 5,
  "Balance": 60000,
  "EstimatedSalary": 80000
}
```
📤 **Saída**

````json
{
  "previsao": "Vai continuar",
  "probabilidade": 0.24,
  "nivel_risco": "BAIXO",
  "explicabilidade": [
    "Age",
    "Balance",
    "Germany"
  ]
}
````

---

<h2 id="metricas" align="center">Métricas e Resultados do Modelo</h2>

O modelo final foi avaliado em uma base de teste (dados inéditos) para garantir sua capacidade de generalização. Os resultados refletem uma escolha estratégica para maximizar a retenção de clientes.

### 🎯 Estratégia de Negócio (Threshold 0.35)
Adotamos um **limite de decisão de 0.35** (em vez do padrão 0.50). Esta decisão técnica visa priorizar o **Recall** (Sensibilidade). Em problemas de Churn, o custo de perder um cliente é geralmente superior ao custo de uma ação de marketing preventiva. Portanto, ajustamos o modelo para ser mais sensível e identificar o maior número possível de clientes em risco.

| Métrica              | Valor      | Impacto de Negócio                                                               |
| :--------------------| :----------| :--------------------------------------------------------------------------------|
| **ROC-AUC**          | **0.7669** | Indica uma boa capacidade do modelo em distinguir entre quem sai e quem fica.    |
| **Acurácia**         | **79.00%** | Proporção global de acertos do modelo em ambas as classes.                       |
| **Recall (Churn)**   | **47.91%** | Capacidade de identificar quase metade de todos os churns reais para intervenção.|
| **Precisão (Churn)** | **48.39%** | Quase metade dos alertas gerados resultam em churns confirmados.                 |

### 🔍 Principais Insights do Modelo
Mesmo sem visualizações nesta página, a análise de importância das variáveis revelou que:
1. **Idade (`Age`)**: É o fator de maior peso; clientes mais velhos tendem a apresentar maior taxa de cancelamento.
2. **Engenharia de Features**: Variáveis criadas pelo squad, como a relação saldo/salário, figuram entre os 10 principais preditores.
3. **Geografia**: Clientes da Alemanha apresentam um comportamento de churn distinto das demais regiões.

* 👉 [**Acesse aqui a Documentação Técnica de Visualizações**](https://github.com/LeticiaPaesano/Churn_Hackathon/blob/main/docs/Documenta%C3%A7%C3%A3o%20T%C3%A9cnica%20de%20Visualiza%C3%A7%C3%B5es.md)

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="entregaveis" align="center">Primeiros Entregáveis do Squad</h2>

Rascunho dos principais entregáveis iniciais:

✅ **Concluídos:**

✅ Notebook EDA + Modelagem Final.

✅ API FastAPI v1.2.1 com Explicabilidade.

✅ Pipeline Serializado.

✅ Suite de Testes Automatizados.

✅ Dockerização Concluída.

⏳ Apresentação Final do Squad.

**Esses itens serão refinados com o decorrer do hackathon.**

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="decisoes" align="center">Decisões Técnicas</h2>

| Decisão            | Motivo                                      | Impacto                                         |
|--------------------|---------------------------------------------|-------------------------------------------------|
| Random Forest      | Melhor tratamento de relações não lineares  | Maior robustez e estabilidade do modelo         |
| Threshold 0.35     | Priorização da captura de clientes em risco | Aumento do Recall e redução de falsos negativos |
| Explicabilidade    | Necessidade de transparência no CRM         | Adoção de princípios de IA responsável          |


<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="validacao" align="center">🧪 Validação Técnica e Garantia de Qualidade</h2>

Para garantir a confiabilidade das previsões e a estabilidade da API em produção, este repositório passou por um rigoroso processo de validação técnica via Bash, garantindo que o modelo treinado no notebook se comporte exatamente da mesma forma na API.

### ✅ Checklist de Integridade
- [x] **Sintaxe**: Validada em 100% dos arquivos do repositório com `compileall`.
- [x] **Carga de Artefatos**: Modelo de 39MB lido com sucesso e mapeamento de chaves (Notebook vs API) corrigido.
- [x] **Fidelidade de Negócio**: Threshold estratégico de **0.35** sincronizado e validado.
- [x] **Testes Automatizados**: Suite completa com **5/5 aprovados** (Integração e Unidade).
- [x] **Processamento em Lote**: Validação de upload e processamento assíncrono de 10k linhas (`Churn.csv`).
- [x] **Docker**: Build e execução de testes internos validados via `docker-compose`.

### ⚡ Automação de Testes

Criamos um script de automação que executa toda a esteira de testes (Sintaxe -> Pytest -> Stress Test Local) em um único comando:

```bash
chmod +x check_all.sh
./check_all.sh
```

### 📊 Monitoramento de Performance
Durante os testes de estresse local, a API manteve um consumo médio de **277MB de RAM**, operando com segurança dentro dos limites de **512MB** do plano gratuito do Render.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="como-executar" align="center">Como Executar a API de Modelo</h2>

1️⃣ Via Docker (Recomendado):

```docker-compose up --build```

- Executa primeiro os testes automatizados (`pytest -v`) antes de iniciar a API.

- API disponível em: `http://localhost:8000`

- Swagger UI (documentação interativa) em: `http://localhost:8000/docs`

2️⃣ Via Python Local (Desenvolvimento)

```
# 1. Criar ambiente virtual
python -m venv venv
# 2. Ativar (Linux/Mac) ou venv\Scripts\activate (Windows)
source venv/bin/activate 
# 3. Instalar dependências
pip install -r requirements.txt
# 4. Executar
uvicorn app.main:app --reload
```

- Certifique-se que `app/models/model.joblib` existe antes de iniciar a API.

- O parâmetro `--reload` reinicia automaticamente a API ao alterar código (apenas para dev).
- 
3️⃣ Rodar Testes Automatizados

```
pytest -v
```

---

<h2 id="deploy" align="center">Deploy com Docker e Render</h2>

A API é empacotada via Docker e publicada automaticamente no Render Cloud.

**Endpoints Importantes**

Health Check: 

```GET /health```

Documentação (Swagger): 

```/docs```

**Produção**

```https://churn-hackathon.onrender.com/docs```

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

---

<h2 id="contribuicoes" align="center">Contribuições</h2>

Contribuições do squad - Para colaborar:
1. Crie uma branch (git checkout -b feature/nome-da-feature)
2. Faça suas alterações
3. Envie um Pull Request descrevendo o que foi modificado

Durante o hackathon, manteremos comunicação constante para evitar conflitos ou trabalho duplicado.

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>

<h2 id="licenca" align="center">Licença</h2>

📌 Este projeto está licenciado sob a **Licença MIT**, o que significa que você pode utilizá-lo, modificar, compartilhar e distribuir livremente, desde que mantenha o aviso de copyright e inclua uma cópia da licença original.

Para mais detalhes, consulte o arquivo [LICENSE](LICENSE) ou a [licença MIT oficial](https://opensource.org/licenses/MIT).

<p align="right"><a href="#inicio">⬆️ Voltar ao início</a></p>
