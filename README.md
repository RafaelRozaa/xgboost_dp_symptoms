# Análise Preditiva de Depressão em Adolescentes por Reforço de Gradiente (XGBoost)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-yellowgreen.svg)](https://scikit-learn.org/)

Modelo de triagem preditiva para identificação de **depressão moderada a grave** em universitários, utilizando **XGBoost** sobre características demográficas e sintomas de estresse (PSS) e ansiedade (GAD).

---

## Sumário

- [Motivação/Objetivo](#motivaçãoobjetivo)
- [Descrição do Dataset](#descrição-do-dataset)
  - [Descrição dos dados](#descrição-dos-dados)
  - [Preparação e pré-processamento](#preparação-e-pré-processamento)
  - [Link para o dataset](#link-para-o-dataset)
- [Modelo de Aprendizado de Máquina](#modelo-de-aprendizado-de-máquina)
  - [Formalização matemática](#formalização-matemática)
  - [Método de validação](#método-de-validação)
  - [Medidas de desempenho](#medidas-de-desempenho)
- [Avaliação](#avaliação)
  - [Amostras de treino, validação e teste](#amostras-de-treino-validação-e-teste)
- [Conclusão](#conclusão)
- [Como reproduzir](#como-reproduzir)
- [Estrutura do repositório](#estrutura-do-repositório)

---

## Motivação/Objetivo

A depressão é um dos transtornos mentais mais prevalentes entre adolescentes e jovens universitários, mas frequentemente subdiagnosticada por barreiras de acesso, estigma e sobrecarga dos serviços de saúde.

Este projeto tem por objetivo construir e avaliar um **modelo de triagem preditiva** capaz de identificar, a partir de características demográficas e de sintomas de **estresse (PSS)** e **ansiedade (GAD)**, indivíduos com **alta probabilidade de depressão moderada a grave**.

**Aplicação prática:** fornecer uma ferramenta de triagem operacional com **threshold calibrado por Coeficiente de Correlação de Matthews (MCC)**, que possa desonerar a capacidade de atendimento clínico.

**Definição do alvo (binário, derivado do PHQ-9):**

| Classe | Rótulos originais | `dep_y` |
|:------:|:------------------|:-------:|
| Não Depressivo | `No Depression`, `Minimal Depression`, `Mild Depression` | `0` |
| Depressivo | `Moderate Depression`, `Moderately Severe Depression`, `Severe Depression` | `1` |

---

## Descrição do Dataset

### Descrição dos dados

O dataset é a **fusão de três planilhas** `Stress.csv`, `Anxiety.csv` e `Depression.csv` do *MHP Anxiety Stress Depression Dataset of University Students*. Cada planilha compartilha um bloco de 7 colunas demográficas e contém os itens dos respectivos instrumentos psicométricos.

| Bloco | Colunas | Conteúdo |
|:------|:-------:|:---------|
| **Demografia** (compartilhada) | 7 | `Age`, `Gender`, `University`, `Department`, `Academic Year`, `Current CGPA`, `Waiver` |
| **PSS** (Perceived Stress Scale) | 10 + 1 | `PSS1`–`PSS10` (Likert 0–4) + `Stress Value` |
| **GAD** (Generalized Anxiety Disorder) | 7 + 1 | `GAD1`–`GAD7` (Likert 0–3) + `Anxiety Value` |
| **PHQ** (Patient Health Questionnaire) | 9 + 1 | `PHQ1`–`PHQ9` + `Depression Value` (apenas para derivar o alvo) |
| **Rótulos multiclasse** | 3 | `Stress Label`, `Anxiety Label`, `Depression Label` |

**Tamanhos:**

| Conjunto | Linhas | Colunas |
|:---------|-------:|--------:|
| Treinamento (fusão das 3 planilhas) | 2.028 | 39 |
| Validação externa (*Bangladesh Mental Health Dataset*) | 9.984* | 82 |

*Dataset de validação externa possui 734 linhas utilizáveis (PHQ completo)

**Distribuição do alvo no dataset de treinamento:**

| Classe PHQ-9 | % | Mapeamento binário |
|:-------------|--:|:------------------:|
| `No Depression` | 2,2% | 0 |
| `Minimal Depression` | 4,8% | 0 |
| `Mild Depression` | 20,4% | 0 |
| `Moderate Depression` | 22,5% | **1** |
| `Moderately Severe Depression` | 25,2% | **1** |
| `Severe Depression` | 24,9% | **1** |

Alvo binário resultante: **1.473 positivos (72,6%)** e **555 negativos (27,4%)** (desbalanceamento moderado).

### Preparação e pré-processamento

1. **Fusão das três planilhas** Alinhamento por `N = min(len)` e reunião dos itens PSS/GAD/PHQ após renomeação padronizada.
2. **Renomeação por regex** `1. ...` → `PSS1`, `GAD1`, `PHQ1`, etc.
3. **Construção do alvo binário** (`dep_y`) a partir de `Depression Label`.
4. **Codificação ordinal** `Age_ord`, `Year_ord`, `CGPA_ord`.
5. **Feature engineering:**
   - `Waiver_bin` (Yes/No → 1/0);
   - Interações: `CGPA_x_Year`, `CGPA_x_Age`, `Year_x_Age`;
   - Indicadores: `CGPA_baixo`, `CGPA_alto`, `Calouro`, `Formando`.
6. **Agregados Likert** para PSS e GAD (`_likert_aggregates`): `mean`, `std`, `max`, `min`, `range`, `n_high` (≥3), `n_zero`, `skew`.
7. **Remoção de vazamento (leakage):**
   - Colunas `PHQ*` e `Depression Label` **nunca** entram em `FEAT_ALL` (verificado por `assert`);
   - Auditoria de Informação Mútua: `MI máxima = 0,234` < 0,9.
8. **Split por grupos** via `GroupShuffleSplit` usando chave `group_key` (SHA-1 de demografia + PSS + GAD), garante que **dois perfis idênticos não caiam em splits diferentes** (overlap train∩val = train∩test = val∩test = **0**).
9. **Encoding** `StandardScaler` (numéricas) + `OneHotEncoder(handle_unknown="ignore")` (categóricas: `Gender`, `University`, `Department`, `Waiver`). Resultado: **76 features** (46 numéricas + 4 categóricas após OHE).
10. **Feature set final** (`FEAT_ALL`, 50 features brutas): demografia + PSS (10 itens) + GAD (7 itens) + `Stress Value` + `Anxiety Value` + 16 agregados + 8 engenheiradas.
11. **Harmonização do domínio externo** (em seção de validação externa):
    - Parsing de Likert em inglês e bengali (`parse_likert`);
    - Filtro por estudantes/graduados (`is_student`) → 5.502/9.984 (55,1%);
    - Mapeamento PSS/GAD/PHQ, com **inversão dos itens positivos** da PSS (PSS4, PSS5, PSS7, PSS8);
    - Filtro por PHQ-9 completo → 964 linhas; alvo proxy `dep_y_proxy = (PHQ_total ≥ 10)`;
    - Imputação de PSS/GAD faltantes com **medianas do treino interno**;
    - Filtro demográfico final → **734 linhas utilizáveis**;
    - Verificação de ausência de PHQ em `FEAT_ALL` (`assert`).

### Link para o dataset

- **MHP Anxiety Stress Depression Dataset of University Students** (Figshare)
   https://figshare.com/articles/dataset/MHP_Anxiety_Stress_Depression_Dataset_of_University_Students/25771164
- **Bangladesh Mental Health Dataset** (Zenodo - validação externa)
   https://zenodo.org/records/17066559

**Arquivos esperados em `/content/`:**

```text
/content/
├── Stress.csv
├── Anxiety.csv
├── Depression.csv
└── Combined Dataset final.csv
```

---

## Modelo de Aprendizado de Máquina

### Formalização matemática

Seja $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ com $\mathbf{x}_i \in \mathbb{R}^{50}$ (features brutas) e $y_i \in \{0, 1\}$ (depressão binária).

**Modelo XGBoost (gradient boosting aditivo):**

$$
\hat{y}_i = \sigma\big(F(\mathbf{x}_i)\big),
\qquad
F(\mathbf{x}_i) = \sum_{k=1}^{K} f_k(\mathbf{x}_i),
\qquad
f_k \in \mathcal{F}
$$

em que $\mathcal{F}$ é o espaço de árvores de decisão de profundidade limitada e $\sigma(z) = \frac{1}{1 + e^{-z}}$ é a sigmoide (a probabilidade calibrada é $\hat{p}_i = \sigma(F(\mathbf{x}_i))$).

**Função objetivo regularizada:**

$$
\mathcal{L} = \sum_{i=1}^{N} \ell(y_i, \hat{p}_i) + \sum_{k=1}^{K} \Omega(f_k),
\qquad
\ell(y, p) = -\big[y \log p + (1-y) \log(1-p)\big]
$$

$$
\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \|w\|^2 + \alpha \|w\|_1
$$

com $T$ = número de folhas, $w$ = pesos das folhas e $(\gamma, \lambda, \alpha)$ = hiperparâmetros de regularização. Neste projeto: $\gamma = 0$, $\lambda = 1$, $\alpha = 0$.

**Hiperparâmetros utilizados:**

| Parâmetro | Valor |
|:----------|:-----:|
| `n_estimators` (máx.) | 200 |
| `early_stopping_rounds` | 40 |
| `max_depth` | 3 |
| `learning_rate` | 0,05 |
| `subsample` | 0,8 |
| `colsample_bytree` | 0,8 |
| `reg_lambda` ($\lambda$) | 1,0 |
| `reg_alpha` ($\alpha$) | 0,0 |
| `min_child_weight` | 1 |
| `tree_method` | `hist` |
| `eval_metric` | `logloss`, `aucpr` |

**Regra de decisão operacional:**

$$
\hat{y}^{\text{bin}}_i = \mathbb{1}\big[\hat{p}_i \geq \tau\big]
$$

com **threshold $\tau$ escolhido por MCC na validação interna** e capacidade-alvo restrita a $\{0{,}01, \dots, 0{,}90\}$:

$$
\tau^\star = Q_{1-c}\big(\{\hat{p}_j\}_{j \in \text{val}}\big),
\qquad
c^\star = \arg\max_{c \in \mathcal{C}} \text{MCC}_{\text{val}}(c)
$$

em que $Q$ é o quantil empírico.

**Correção de prior shift de Saerens** (versão iterativa EM, usada na validação externa):

$$
\hat{p}^{\text{corr}}(x) =
\frac{
\hat{p}(x) \cdot (\pi_{\text{ext}} / \pi_{\text{trn}})
}{
\hat{p}(x) \cdot (\pi_{\text{ext}} / \pi_{\text{trn}})
+ (1-\hat{p}(x)) \cdot ((1-\pi_{\text{ext}}) / (1-\pi_{\text{trn}}))
}
$$

com $\pi_{\text{ext}}$ estimado iterativamente como a média de $\hat{p}^{\text{corr}}$.

### Método de validação

- **Split por grupos** com `GroupShuffleSplit` estratificado por `group_key` (hash SHA-1 de demografia + PSS + GAD), evitando vazamento entre perfis idênticos.
- **Proporção 70/15/15:** treino (1.420), validação (304), teste (304).
- **Early stopping** monitorado no conjunto de validação `best_iteration = 39`, `n_boosted_rounds = 79`.
- **Threshold operacional** escolhido **apenas na validação interna** por MCC (capacidade restrita a $\{0{,}01, \dots, 0{,}90\}$) e então **congelado e herdado** para o teste e para a validação externa.
- **Curva operacional descritiva:** para cada capacidade $c$, threshold = quantil $1-c$ de $\hat{p}_{\text{val}}$, avaliado em $\hat{p}_{\text{test}}$, produzindo `recall`, `precision`, `F1`, `MCC`, `NNT`, `lift`.
- **Validação externa** (Bangladesh): o mesmo `preprocess` e o mesmo `model` são aplicados a `ext_eng[FEAT_ALL]`; o **threshold externo é fixo a priori** ($\tau_{\text{ext}} = 0{,}50$, análogo ao PHQ-9 ≥ 10) e **não depende de nenhum dado do domínio externo** — decisão motivada pelo diagnóstico de *covariate shift* severo (ver abaixo).

### Medidas de desempenho

| Categoria | Métrica | Definição / Observação |
|:----------|:--------|:-----------------------|
| Discriminação | **ROC-AUC** | $\mathbb{P}(\hat{p}^{+} > \hat{p}^{-})$ |
| Discriminação (foco na classe positiva) | **PR-AUC** (`average_precision_score`) | Área sob a curva precisão-recall |
| Calibração | **Brier score** | $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$ |
| Calibração | **ECE** (Expected Calibration Error) | Média ponderada do gap por bin |
| Classificação (threshold) | **Acurácia, Precisão, Recall, F1** | Matriz de confusão |
| Classificação (desequilíbrio) | **MCC**, **Balanced Accuracy** | Coeficientes robustos |
| Utilidade clínica | **NNT** (Number Needed to Treat) | $(TP+FP)/\max(TP,1)$ |
| Ganho sobre a prevalência | **Lift** | precisão / prevalência |
| Auditoria de leakage | **Informação Mútua** | `mutual_info_classif` máx. < 0,9 |

---

## Avaliação

### Amostras de treino, validação e teste

| Split | Linhas | Positivos (`dep_y=1`) | Negativos | Proporção |
|:------|-------:|----------------------:|----------:|----------:|
| **Treino** | 1.420 | 1.009 | 411 | 70% |
| **Validação** | 304 | 238 | 66 | 15% |
| **Teste** | 304 | 226 | 78 | 15% |
| **Validação externa** | 734 | 55  | 679 | — |

O overlap de perfis (`group_key`) entre splits é **zero** em todas as combinações, confirmando que não há vazamento por perfil demográfico + sintomático.

### Medidas de desempenho :

A Capacidade operacional escolhida na validação (por MCC) foi **60,0%** → threshold herdado para o teste: $\tau^\star = 0{,}8023$.

#### **Diagnostico de Treinamento**

| Métrica | Valor |
|:--------|------:|
| `best_iteration` | 39 |
| `best_score` | 0,9748 |
| `n_boosted_rounds` | 79 |
| Brier (validação) | 0,1025 |
| ECE (validação) | 0,0669 |


#### **Relatório de classificação: conjunto de teste interno (threshold $0{,}8023$):**

| Classe | Precisão | Recall | F1-score | Suporte |
|:-------|---------:|-------:|---------:|--------:|
| Não Depressivo | 0,54 | 0,90 | 0,67 | 78 |
| Depressivo | 0,95 | 0,73 | 0,83 | 226 |
| **Acurácia** | | | **0,78** | 304 |
| *Macro avg* | 0,75 | 0,82 | 0,75 | 304 |
| *Weighted avg* | 0,85 | 0,78 | 0,79 | 304 |

#### **Métricas globais no teste:**

| Métrica | Valor |
|:--------|------:|
| **PR-AUC** | 0,9587 |
| **ROC-AUC** | 0,8950 |
| **Brier** | 0,1115 |
| **ECE** | 0,0742 |
| **MCC** | 0,5579 |
| Positivos preditos | 174 / 304 (57,2%) |
| True Positives (TP) | 166 |
| **NNT** | 1,05 |

####  **Curva operacional (recall @ threshold herdado vs. auto-referenciado no teste):**

| Capacidade | Threshold (`p_val`) | Recall (herdado) | Recall (self-ref) | Gap |
|-----------:|--------------------:|-----------------:|------------------:|----:|
| 1% | 0,9477 | 0,0885 | 0,0885 | 0,0000 |
| 20% | 0,9363 | 0,2788 | 0,2743 | +0,0044 |
| 40% | 0,9110 | 0,4248 | 0,5265 | −0,1018 |
| **60%** | **0,8023** | **0,7345** | 0,7566 | −0,0221 |
| 80% | 0,5118 | 0,9071 | 0,9469 | −0,0398 |
| 90% | 0,3172 | 0,9735 | 0,9779 | −0,0044 |

A capacidade de **60%** produz o melhor equilíbrio MCC na validação, com **recall de 0,73** no teste interno, capturando ~73% dos casos de depressão moderada a grave com **precisão de 0,95** na classe positiva. O NNT ≈ 1,05 significa que, em média, **cada ~1 paciente sinalizado como positivo corresponde a 1 caso verdadeiro**.

####  **Ranking de Informação Mútua (top-5, treino):**

| Feature | MI com `dep_y` |
|:--------|---------------:|
| `GAD_mean` | 0,2219 |
| `Anxiety Value` | 0,2154 |
| `GAD7` | 0,1762 |
| `GAD4` | 0,1568 |
| `GAD5` | 0,1380 |

As features mais informativas são **agregados de ansiedade (GAD)**, seguidos por itens individuais do GAD.

#### **Métricas no domínio externo (threshold fixo a priori = 0,50):**

| Métrica | Valor |
|:--------|------:|
| ROC-AUC | **0,9881** |
| PR-AUC | **0,9651** |
| Spearman($\hat{p}$, PHQ_total) | 0,2529 |
| Brier | 0,1181 |
| ECE | **0,3201** |
| MCC (threshold 0,50) | **0,7755** |
| Recall | 0,9818 |
| Precision | 0,6429 |
| NNT | 1,56 |

#### **Diagnóstico de shift no domínio externo:**

| Diagnóstico | Resultado | Interpretação |
|:------------|:----------|:--------------|
| **EM de Saerens** | $\pi_{\text{ext}} = 0{,}0000$ (colapsou em 12 iter) | **Premissa de label shift FALSA** |
| **KLIEP** (AUC do classificador de domínio) | **0,9999** | Covariate shift **severo** |
| **KLIEP** ($w(x)$) | mediana = 1.604; máx = 151.794 | Distribuições de $X$ **muito** diferentes |

> **Decisão metodológica:** sob *covariate shift* severo, métodos que assumem $P(x)$ estável (Saerens, quantil herdado) são **inadequados**. O notebook adota **threshold fixo a priori** ($0{,}50$), que **não depende de nenhum dado do domínio externo**.

---

## Conclusão

O modelo XGBoost sobre o conjunto completo (demografia + PSS + GAD) alcança desempenho consistente para triagem de depressão moderada a grave:

- **Discriminação forte:** PR-AUC = 0,959; ROC-AUC = 0,895.
- **Boa calibração:** Brier = 0,112; ECE = 0,074 (probabilidades utilizáveis para priorização).
- **Sem vazamento:** auditoria de MI (máx. 0,234 < 0,9) e `assert` que bloqueia qualquer feature PHQ.
- **Split honesto:** `GroupShuffleSplit` por perfil demográfico + sintomático, com **zero overlap** entre treino/validação/teste.

**Principais achados:**

1. **Ansiedade (GAD) é o preditor mais informativo**, seguido por estresse percebido (PSS), e seus valores de Informação Multual e ganho por feature também justificam a manutenção de PSS e GAD como features preditivas dos sintomas de depressão.
2. **Threshold por MCC com restrição de capacidade** evita soluções degeneradas (marcar todos como positivos) e respeita a restrição operacional de um programa de triagem.
3. **Threshold fixo a priori (0,50)** se demosntrou uma escolha metodologicamente honesta, por não depender de nenhum dado do domínio externo.

**Limitações e trabalhos futuros:**

- Não foi feita uma extensa análise da importância relativa de cada fator em especifico, focando no uso dos dados para a predição dos sintomas de depressão, futuros trabalhos poderiam refinar essa análise para um melhor feature engineering e eliminação de redundâncias.
- Os datasets utilizados não captam todos os fatores relevantes transversalmente, como suporte social e neuroticismo
- Não foi possivel adquirir o dataset de referência para fatores de relevância em sintomas de depressão, a aquisisão dele habilita a replicação de seus resultados e comparar a precisão do modelo quando utilizando um dataset de fatores previamente analisados em seu aprendizado
- A calibração externa é ruim (ECE = 0,3201), o que é esperado sob covariate shift severo; trabalhos futuros poderiam explorar **recalibração no domínio externo**

**Reprodutibilidade:** o notebook é totalmente determinístico (`SEED = 42`), com auditoria de *leakage*, split agrupado e escolha de threshold explicitamente isolada da validação externa.

---

## Como reproduzir

### Pré-requisitos

```bash
Python >= 3.10
xgboost >= 3.0
scikit-learn >= 1.5
pandas >= 2.0
numpy >= 1.26
scipy >= 1.11
matplotlib >= 3.8
seaborn >= 0.13
```

### Instalação

```bash
git clone https://github.com/<usuario>/<repo>.git
```

### Execução

1. Abra `LLM-XGBoost.ipynb` no Jupyter ou no Google Colab.
2. Baixe os datasets na pasta data e coloque-os em `/content/`.
3. Para primeira execução, descomente o comando de instalação
```bash
!pip install -q xgboost scikit-learn pandas numpy scipy matplotlib seaborn
```
4. Execute todas as células em ordem.

```bash
jupyter notebook LLM-XGBoost.ipynb
```

---

## Estrutura do repositório

```text
.
├── README.md
├── LLM-XGBoost.ipynb        # Notebook principal
└── data/
    ├── Stress.csv
    ├── Anxiety.csv
    ├── Depression.csv
    └── Combined Dataset final.csv
```

## Referências

- Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System**. *KDD '16*.
- Saerens, M., Latinne, P., & Decaestecker, C. (2002). **Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure**. *Neural Computation*, 14(1), 21–41.
- Cohen, S., Kamarck, T., & Mermelstein, R. (1983). **A Global Measure of Perceived Stress**. *Journal of Health and Social Behavior*, 24(4), 385–396.
- Spitzer, R. L., Kroenke, K., Williams, J. B. W., & Löwe, B. (2006). **A Brief Measure for Assessing Generalized Anxiety Disorder: The GAD-7**. *Archives of Internal Medicine*, 166(10), 1092–1097.
- Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). **The PHQ-9: Validity of a Brief Depression Severity Measure**. *Journal of General Internal Medicine*, 16(9), 606–613.
