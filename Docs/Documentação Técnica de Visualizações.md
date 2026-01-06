# 📑 Documentação Técnica de Visualizações (Churn Analysis)

Este documento contém os artefatos visuais gerados durante o desenvolvimento do projeto **Churn — hackathon Alura**. A ordem das visualizações abaixo segue o fluxo lógico de análise exploratória, validação de hipóteses e avaliação do modelo final.

---

### 1. Análise Exploratória de Dados (EDA)

<img width="1485" height="990" alt="image" src="https://github.com/user-attachments/assets/2381d739-a2b5-4ab1-83f3-79ee8811855d" />

Esta visualização inicial apresenta a distribuição de frequência das variáveis numéricas do dataset.

* **Distribuição do Target:** Confirma o desbalanceamento das classes, com ~80% de retenção e ~20% de churn (Exited).
* **Comportamento Financeiro:** Revela um volume significativo de clientes com saldo bancário (`Balance`) zerado, o que fundamentou a criação de novas features.
* **Perfil Demográfico:** A variável `Age` apresenta uma concentração predominante entre 30 e 45 anos, com o pico de frequência (moda) estabelecido entre 35 e 37 anos, indicando uma base majoritariamente de adultos jovens.

---

### 2. Mapa de Correlação 

<img width="865" height="782" alt="Mapa de Correlação" src="https://github.com/user-attachments/assets/82dae504-32b7-43a3-9209-d82de91127cd" />

**Principal Preditor:** A variável ``Age`` possui a maior correlação positiva `**(0.29)** com Exited, sendo o fator de maior influência no churn.

**Fator Financeiro:** O ``Balance`` apresenta correlação de **0.12**, indicando que o volume de saldo também impacta moderadamente na saída do cliente.

**Baixa Colinearidade:** Não há correlações fortes entre as variáveis preditoras, o que evita redundância de dados e garante a estabilidade do modelo.

**Variáveis Irrelevantes:** ``Tenure`` e ``EstimatedSalary`` possuem correlações próximas a zero, sugerindo baixo poder preditivo isolado.

---

### 3. Matriz de Confusão - RF Balanceado (Validação)

<img width="671" height="547" alt="Matriz de Confusão - RF Balanceado" src="https://github.com/user-attachments/assets/7809f6ff-e286-4f36-a6f2-dbae128dc210" />

**Objetivo:** Ajuste inicial de hiperparâmetros e threshold (0.35) para priorizar a sensibilidade ao churn.

**Desempenho:** Identificou corretamente 185 casos de evasão, apresentando um equilíbrio entre a captura de clientes em risco e o volume de falsos positivos.

---

### 4. Matriz de Confusão - RF FINAL (Teste)

<img width="658" height="547" alt="Matriz de Confusão - RF Final" src="https://github.com/user-attachments/assets/53f324f6-0498-4902-ac0e-5c51e7498443" />

**Objetivo:** Validação da capacidade de generalização do modelo com dados inéditos, mantendo a estratégia de negócio.

**Diferencial Técnico:** O modelo demonstrou alta estabilidade, aumentando a captura de churn real para 194 casos e reduzindo os erros de omissão (Falsos Negativos) de 222 para 213.

**Conclusão:** A manutenção do número de Falsos Positivos (208) em ambos os cenários confirma que o modelo é robusto e não sofreu overfitting, garantindo previsões confiáveis para a tomada de decisão.

---

### 5. Importância das Variáveis (Interpretabilidade)

<img width="1189" height="790" alt="Top 10 Features Mais Importantes" src="https://github.com/user-attachments/assets/9371dfe0-1a32-4696-9634-af5a67e04d07" />

Análise do ranking dos atributos que mais influenciaram as decisões do algoritmo **Random Forest**:

**Liderança Absoluta:** A Idade (``Age``) consolida-se como o fator preditivo mais robusto, apresentando a maior importância relativa no modelo.

**Sucesso da Engenharia de Dados**: As variáveis criadas ``Age_Tenure`` e ``Balance_Salary_Ratio`` figuram no Top 10, confirmando que a combinação de idade com tempo de casa e a relação saldo/salário agregaram valor preditivo inédito ao modelo.

`**Preditores Secundários**: O comportamento financeiro, representado por ``EstimatedSalary``, ``CreditScore`` e ``Balance``, mantém uma relevância alta e equilibrada na tomada de decisão.

`**Influência Geográfica**: A localização (especialmente ``Geography_Germany``) surge como um fator demográfico relevante, superando variáveis como gênero na predição de churn.

 <p align="right"><a href="../README.md">🔄 Voltar para a documentação completa</a></p>
