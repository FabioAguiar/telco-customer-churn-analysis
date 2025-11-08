# ⚙️ `config/` — Guia completo de parâmetros

Este diretório centraliza as **configurações** usadas pelos notebooks do template (N1/N2/N3).  
O arquivo principal é **`defaults.json`** (parâmetros padrão). Opcionalmente, você pode criar um **`local.json`** para **sobrescrever** valores **sem alterar** o template.

> O carregamento faz *merge* de `defaults.json` com `local.json` (prioridade para `local.json`).

---

## 📂 Arquivos
- **`defaults.json`** — Configurações padrão do projeto.
- **`local.json`** — (Opcional) Sobreposições locais por projeto/ambiente.

---

## 🔧 Parâmetros globais (nível raiz)

| Chave | Tipo | Padrão | Descrição |
|---|---|---|---|
| `infer_types` | bool | `True` | Otimiza tipos (ex.: *downcast* numérico) para reduzir memória. |
| `cast_numeric_like` | bool | `True` | Converte textos que “parecem numéricos” em números. |
| `strip_whitespace` | bool | `True` | Remove espaços em branco excedentes em colunas textuais. |
| `handle_missing` | bool | `True` | Ativa tratamento de valores ausentes (N1). |
| `missing_strategy` | str | `"simple"` | Estratégia de imputação (ex.: `"simple"`). |
| `detect_outliers` | bool | `True` | Ativa detecção de outliers (N1). |
| `outlier_method` | str | `"iqr"` | Método de detecção (`"iqr"` ou `"zscore"`). |
| `encode_categoricals` | bool | `True` | Ativa codificação de variáveis categóricas. |
| `encoding_type` | str | `"onehot"` | Tipo de codificação (`"onehot"`/`"ordinal"`). |
| `scale_numeric` | bool | `False` | Ativa escalonamento numérico (N1/N2). |
| `scaler` | str | `"standard"` | Escolha do *scaler* (`"standard"`/`"minmax"`). |
| `date_features` | bool | `True` | Geração de *features* de data (N1). |
| `text_features` | bool | `True` | *Features* simples de texto (N1). |
| `export_interim` | bool | `True` | Exporta dataset intermediário (`data/interim`). |
| `normalize_categories` | bool | `True` | Normaliza rótulos de categorias equivalentes. |
| `export_processed` | bool | `True` | Exporta dataset final (`data/processed`). |
| `artifacts_dir` | str | `"artifacts"` | Pasta base de artefatos (modelos, etc.). |
| `data_raw_dir` | str | `"data/raw"` | Pasta dos dados brutos. |
| `data_processed_dir` | str | `"data/processed"` | Pasta dos dados processados. |
| `data_processed_file` | str | `"processed.parquet"` | Nome do arquivo processado alvo (N2). |
| `target_column` | str | `"Churn"` | Nome padrão da *target* no dataset processado (N2). |
| `test_size` | float | `0.2` | Proporção de teste para `train_test_split` (N2). |
| `random_state` | int | `42` | Semente aleatória para reprodutibilidade (N2). |

---

## 📦 Seção: `outliers`

```json
{
  "cols": null,
  "exclude_cols": [
    "customerID"
  ],
  "exclude_binaries": true,
  "iqr_factor": 1.5,
  "z_threshold": 3.0,
  "persist_summary": true,
  "persist_relpath": "outliers/summary.csv"
}
```

| Chave | Tipo | Descrição |
|---|---|---|
| `cols` | list\|null | Colunas específicas (ou `null` para todas numéricas). |
| `exclude_cols` | list | Colunas a ignorar (ex.: IDs). |
| `exclude_binaries` | bool | Ignora colunas 0/1. |
| `iqr_factor` | float | Multiplicador do IQR (ex.: 1.5). |
| `z_threshold` | float | Limite de Z-score (ex.: 3.0). |
| `persist_summary` | bool | Salva CSV com resumo. |
| `persist_relpath` | str | Caminho relativo dentro de `reports/`. |

---

## 🔁 Seção: `deduplicate`

```json
{
  "subset": null,
  "keep": "first",
  "log_enabled": true,
  "log_relpath": "duplicates.csv"
}
```

| Chave | Tipo | Descrição |
|---|---|---|
| `subset` | list\|null | Colunas que definem duplicidade (`null` = linha inteira). |
| `keep` | str\|bool | `"first"`, `"last"` ou `false` (remove todas). |
| `log_enabled` | bool | Gera log de duplicatas removidas. |
| `log_relpath` | str | Caminho do log (dentro de `reports/`). |

> **Observação (legado):** Existem chaves de nível raiz relacionadas a deduplicação — `deduplicate_subset`, `deduplicate_keep`, `deduplicate_log`, `deduplicate_log_filename`.  
> **Recomendação:** use **apenas** o bloco `deduplicate` para evitar configurações conflitantes. As chaves legadas serão descontinuadas.

---

## 🧠 Seção: `feature_engineering`

```json
{
  "enable_default_rules": true,
  "log1p_cols": [],
  "ratios": [],
  "binaries": [],
  "date_parts": []
}
```

| Chave | Tipo | Descrição |
|---|---|---|
| `enable_default_rules` | bool | Regras básicas automáticas. |
| `log1p_cols` | list | Colunas para `log1p`. |
| `ratios` | list | Proporções entre colunas. |
| `binaries` | list | Colunas binárias a partir de condições. |
| `date_parts` | list | Partes de data customizadas. |

---

## 🗓️ Seção: `dates`

```json
{
  "detect_regex": "(date|data|dt_|_dt$|_date$|_at$|time|timestamp|created|updated)",
  "explicit_cols": [],
  "dayfirst": false,
  "utc": false,
  "formats": [],
  "min_ratio": 0.8,
  "report_path": "date_parse_report.csv"
}
```

| Chave | Tipo | Descrição |
|---|---|---|
| `detect_regex` | str | Regex para detectar colunas de data. |
| `explicit_cols` | list | Colunas forçadas como datetime. |
| `dayfirst` | bool | Usa formato D/M/Y. |
| `utc` | bool | Converte para UTC. |
| `formats` | list | Formatos aceitos. |
| `min_ratio` | float | Mínimo de sucesso no parsing. |
| `report_path` | str | Relatório salvo em `reports/`. |

---

## 🎯 Seção: `target`

```json
{
  "name": "Churn",
  "source": "Churn",
  "positive": "Yes",
  "negative": "No"
}
```

| Chave | Tipo | Descrição |
|---|---|---|
| `name` | str | Nome final da *target* (após N1). |
| `source` | str | Coluna de origem no bruto. |
| `positive` | str | Classe positiva. |
| `negative` | str | Classe negativa. |

### `class_map`
```json
{
  "Yes": 1,
  "No": 0
}
```
Mapeamento opcional de rótulos para inteiros (ex.: `"Yes" → 1`, `"No" → 0`).

---

## 🧾 Seção: `reporting`

```json
{
  "manifest_enabled": true
}
```

| Chave | Tipo | Descrição |
|---|---|---|
| `manifest_enabled` | bool | Gera/atualiza `reports/manifest.jsonl` com os artefatos. |

---

## 🧪 Parâmetros de treino (N2)

- `target_column`: `Churn`  
- `test_size`: `0.2`  
- `random_state`: `42`  

> Usados em `train_test_split` e validação.

---

## 📁 Pastas de trabalho

- `artifacts_dir`: `artifacts` → onde ficam modelos (`.joblib`) e params/metrics json.  
- `data_raw_dir`: `data/raw`  
- `data_processed_dir`: `data/processed` / `data_processed_file`: `processed.parquet`

---

## ✅ Recomendações

1. **Evite** duplicidade entre chaves legadas e blocos estruturados (ex.: prefira `deduplicate.*`).  
2. Use **`local.json`** para ajustes por projeto/ambiente.  
3. Versione `config/` e confira `reports/manifest.jsonl` após execuções.  
4. Ajuste `class_map` e `target` conforme seu problema (binário vs. multiclasse).

---

## ✳️ Exemplo de `local.json` mínimo

```json
{
  "detect_outliers": false,
  "scale_numeric": true,
  "scaler": "minmax",
  "target": {
    "name": "Churn",
    "source": "Churn",
    "positive": "Yes",
    "negative": "No"
  }
}
```

---

**Em suma:** este README cobre **todas as chaves** presentes em `defaults.json`, incluindo **pastas**, **parâmetros de treino**, **módulos de N1** (outliers, imputação, datas, texto, *feature engineering*) e **N2** (split, *target*, `class_map`). 
