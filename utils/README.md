# 🧰 utils_data.py — Toolkit de Funções Utilitárias para Projetos de Dados

Este módulo centraliza funções reutilizáveis para **ingestão, limpeza, transformação, engenharia de atributos, tratamento de nulos, exportação e controle de artefatos**.
É o núcleo do *Data Project Template* e garante **reprodutibilidade, modularidade e clareza** em todos os notebooks (N1, N2 e N3).

---

## 📦 Estrutura Geral

| Categoria | Funções Principais |
|------------|--------------------|
| 🔧 Configuração e Caminhos | `get_project_root`, `load_config`, `ensure_dirs`, `discover_processed_path` |
| 🧹 Qualidade e Tipagem | `run_quality_and_typing`, `render_quality_and_typing` |
| ⚠️ Tratamento de Nulos | `null_fill_from_config`, `render_null_fill_report` |
| 📈 Engenharia de Atributos | `_safe_div`, `_signed_log1p`, `recompute_charge_gap_features`, `recompute_avg_charge_safe` |
| 📊 Métricas e Avaliação | `compute_metrics`, `try_plot_roc` |
| 🧾 Persistência e Exportação | `persist_artifacts`, `save_report_df`, `ensure_project_root` |

---

## ⚙️ 1. Configuração e Caminhos

### `get_project_root()`  
Localiza automaticamente a raiz do projeto com base no arquivo `config/defaults.json`.  
Utilizado por todos os notebooks para referência de diretórios.

### `load_config(defaults_path, local_path=None)`  
Carrega o arquivo `defaults.json` e o opcional `local.json`, realizando *merge* com prioridade para o local.  
Retorna um dicionário de configurações consolidadas.

### `ensure_dirs(config)`  
Garante a existência dos diretórios principais: `artifacts`, `reports` e `models`.  
Retorna suas referências como `Path`.

### `discover_processed_path(config)`  
Retorna o caminho completo do arquivo processado (gerado no N1) com base nas chaves do `config`.

---

## 🧹 2. Qualidade e Tipagem

### `run_quality_and_typing(df, config)`  
Executa padronização e coerção de tipos, incluindo:
- Conversão numérica e categórica;
- Normalização de capitalização e espaços;
- Deduplicação condicional;
- Registro de estatísticas de memória e tipos.

### `render_quality_and_typing(result)`  
Renderiza o resumo visual da etapa de qualidade: dimensões, memória e conversões aplicadas.

---

## ⚠️ 3. Tratamento de Nulos

### `null_fill_from_config(df, config, root=None)`  
Preenche valores nulos com base nas opções do bloco `null_fill_with_flag` do `config`.  
Principais parâmetros:
- `enabled`: ativa/desativa o preenchimento;
- `numeric_fill`: valor de substituição para colunas numéricas;
- `categorical_fill`: valor de substituição para colunas categóricas;
- `cols_numeric_zero`: lista explícita de colunas a preencher com zero;
- `flag_suffix`: sufixo de flag para indicar valores substituídos;
- `report_relpath`: caminho relativo do relatório de comparação.

Retorna:  
`(df_preenchido, metadados)` com informações sobre flags, colunas tratadas e caminhos de relatório.

### `render_null_fill_report(meta)`  
Exibe um relatório claro e colorido com:
- Colunas preenchidas;
- Flags criadas;
- Caminho de relatório salvo;
- Tabelas “antes” e “depois” do preenchimento.

---

## 📈 4. Engenharia de Atributos (Feature Engineering)

### `_safe_div(num, den, fallback=0.0)`  
Divisão protegida contra divisão por zero e `NaN`.  
Usada internamente em razões como `TotalCharges / tenure`.

### `_signed_log1p(x)`  
Cálculo de log1p assinado (`sign(x) * log1p(|x|)`) — evita NaN de domínio para valores ≤ -1.  
Empregado em `charge_gap_log1p` e outras transformações logarítmicas.

### `recompute_charge_gap_features(df)`  
Recalcula as colunas derivadas:
- `charge_gap = TotalCharges - (MonthlyCharges * tenure)`  
- `charge_gap_log1p = sign(charge_gap) * log1p(|charge_gap|)`  

Evita valores nulos e mantém consistência com colunas base após preenchimentos.

### `recompute_avg_charge_safe(df)`  
Recalcula `avg_charge_per_month` com segurança:  
- Se `tenure > 0`: calcula `TotalCharges / tenure`;  
- Se `tenure == 0`: retorna `0` e opcionalmente marca flag `_was_missing`.

---

## 📊 5. Métricas e Avaliação

### `compute_metrics(y_true, y_pred)`  
Gera dicionário de métricas: acurácia, F1-score, precisão, recall e matriz de confusão.

### `try_plot_roc(model, X_test, y_test)`  
Tenta exibir a curva ROC (Receiver Operating Characteristic) de forma segura e padronizada.

---

## 🧾 6. Persistência e Exportação

### `persist_artifacts(df, config)`  
Exporta o dataframe processado, artefatos e metadados para os diretórios definidos em `config`.

### `save_report_df(df, relpath, root=None)`  
Salva qualquer dataframe de relatório (como comparativo de nulos ou metadados) no diretório `reports/`.

### `ensure_project_root()`  
Valida a estrutura de diretórios do projeto e cria o `__init__.py` em `utils/` se ausente.

---

## 🧠 7. Integração entre Etapas (N1 → N2 → N3)

O módulo `utils_data.py` foi projetado para conectar cada fase do projeto de dados:

| Fase | Funções-Chave |
|------|----------------|
| N1 - Preparação | `run_quality_and_typing`, `null_fill_from_config`, `recompute_charge_gap_features`, `recompute_avg_charge_safe` |
| N2 - Modelagem | `discover_processed_path`, `compute_metrics`, `try_plot_roc` |
| N3 - Análise | `load_config`, `persist_artifacts`, `save_report_df` |

---

## 🧩 Versão e Autoria

**Versão:** 1.3.0  
**Autor:** Fábio Emmanuel de Andrade Aguiar (Fabyuu)  
**Descrição:** Toolkit unificado e resiliente para pipelines de dados, desenvolvido para o projeto *Telco Customer Churn Analysis* e o *Data Project Template* genérico.
