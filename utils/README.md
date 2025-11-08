# 🧰 `utils/` — Utility Toolkit for Data Projects (v1.2.2-merged)

Coleção de utilitários usada pelos notebooks (N1→N3) para **ingestão**, **limpeza**, **engenharia de atributos**, **datas**, **texto**, **codificação/escala**, **catálogo de DataFrames**, **artefatos** e **manifest**.  
Módulo principal: **`utils/utils_data.py`** (versão `UTILS_DATA_VERSION = "1.2.2"`).

> Import típico no notebook:
> ```python
> import importlib, utils.utils_data as ud
> importlib.reload(ud)
> from utils.utils_data import TableStore
> ```

---

## 🧭 Descoberta de raiz, config e manifest

### `ensure_project_root() -> Path`
- Sobe a árvore até encontrar `config/defaults.json` e fixa a **raiz do projeto**.
- Injeta `utils/` no `sys.path` (para imports estáveis nos notebooks em qualquer subpasta).
- Emite log: `PROJECT_ROOT: <path>`.

### `load_config(base_abs=None, local_abs=None) -> dict`
- Carrega `config/defaults.json` e faz *merge* profundo com `config/local.json` (se existir).

### Manifest helpers
- `load_manifest()`, `save_manifest()`, `update_manifest()`
- `record_step(name, details=None)` e *context manager* `with_step(name, details=None)` para auditar etapas no tempo.

### Artefatos e relatórios
- `get_artifacts_dir(subdir=None)` → **`reports/artifacts[/<subdir>]`** (garante diretório). **Use isto no N1**.
- `save_artifact(obj, name)` / `load_artifact(name)` → `.joblib` em `artifacts/` (modelos, encoders, etc.).
- `save_report_df(df, rel_path)` e `save_text(text, rel_path)` gravam em `reports/<rel_path>`.

---

## 📥 Ingestão & 📤 Exportação

- `infer_format_from_suffix(path) -> "csv"|"parquet"`
- `load_csv(path, **kwargs)` → wrapper do `pd.read_csv`
- `load_table_simple(path, fmt=None, *args, **kwargs)`  
  Compatível com: `fmt` **ou** dicionário de `read_opts` posicional.
- `save_table(df, path, fmt=None, **kwargs)` → respeita a extensão (`.csv`/`.parquet`), cria pastas e loga.
- `list_directory_files(dir)` e `suggest_source_path(dir, pattern="*.csv")` → inventário rápido de fontes.
- `save_named_interims({name: df}, base_dir, fmt="parquet")` → salva múltiplos *interims* nomeados.

---

## 🔎 Perfil, tipagem & qualidade

- `basic_overview(df) -> dict` → shape, dtypes, memória (MB).
- `strip_whitespace(df, cols=None)` → *trim* + colapso de espaços para textos.
- `infer_numeric_like(df, cols=None, decimal=".", thousands=None, report_path="cast_report.csv") -> (df, report)`  
  Converte “strings numéricas” para números e **persiste relatório** em `reports/` (via `save_report_df`).
- `n1_quality_typing(df, config)` / `n1_quality_typing_dict(df, config)` → *pipeline* compacto com logs.

### Faltantes, duplicatas e outliers
- `missing_report(df)` → tabela com `missing_count`/`missing_pct`.
- `simple_impute_with_flags(df, strategy="median") -> (df, meta)` → flags `was_missing` por coluna (rastreável).
- `deduplicate_rows(df, subset=None, keep="first", config=None) -> df`  
  **Nova** assinatura lê `config["deduplicate"]` (subset/keep) se passado.
- `apply_outlier_flags(df, config=None, method=None, iqr_factor=None, z_threshold=None, ...) -> (df, info)`  
  **Nova** API que cria colunas `<col>_is_outlier` por **IQR** ou **Z-score**, respeitando `config["outliers"]`  
  (cols, exclude_cols, exclude_binaries, iqr_factor, z_threshold) e pode **persistir** resumo em `reports/outliers/summary.csv`.

---

## 🔤 Categóricas & 🔢 Numéricas

- `encode_categories(df, cols=None, drop_first=False, high_cardinality_threshold=20, top_k=None, other_label="__OTHER__") -> (df, meta)`
- `encode_categories_safe(df, exclude_cols=None, **kwargs)` → ignora alvo/IDs e protege contra alta cardinalidade.
- `scale_numeric(df, method="standard"|"minmax", cols=None) -> (df, meta)`
- `scale_numeric_safe(df, exclude_cols=None, only_continuous=True, **kwargs)` → evita dummies/booleanas.
- `apply_encoding_and_scaling(df, config) -> (df, meta)` → orquestra encode→scale lendo sub-`config` (`encoding`/`scaling`).

---

## 📅 Datas

- `detect_date_candidates(df, regex_list=None)`
- `parse_dates_with_report(df, cols=None, dayfirst=False, utc=False, errors="coerce", min_ratio=0.6, report_path="date_parse_report.csv") -> (df, report)`
- **Nova:** `parse_dates_with_report_cfg(df, cfg) -> (df, report, parsed_cols)`  
  Lê um dicionário `cfg` com: `detect_regex`, `explicit_cols`, `dayfirst`, `utc`, `formats`, `min_ratio`, `report_path`.
- `expand_date_features(df, cols)` → `*_year`, `*_month`, `*_day`, `*_dow`, `*_week`, `*_quarter`.
- **Nova:** `expand_date_features_plus(df, date_cols, features=("year","month","day","dayofweek","quarter","week","is_month_start","is_month_end"), prefix_mode="auto") -> list[str]`
- `build_calendar_from(df, col, freq="D") -> dim_date`

---

## 📝 Texto

- **Nova (ampliada):** `extract_text_features(df, *, lower=True, strip_collapse_ws=True, keywords=None, blacklist=None, export_summary=True, summary_dir=None) -> (df, summary_df)`  
  - Limpeza leve (minúsculas/opcional e espaços).  
  - Métricas: `<col>_len`, `<col>_word_count`.  
  - Flags por *keywords*: `<col>_has_<kw>`.  
  - Exporta `text_features_summary.csv` quando configurado.

---

## 🎯 Target

- `build_target(df, config) -> (df, meta)` → regra simples com `col`/`op`/`value` (uso pontual).
- `ensure_target_from_config(df, config, verbose=False) -> (df, target_name, class_map, report_df)`  
  Lê `config["target"] = {name, source, positive, negative}`.  
  - Se `name` já existir no DF → **respeita**.  
  - Se `source` existir → cria `name` mapeando `positive`/`negative`.  
  - Caso contrário → cria `name` nulo e reporta **não criado**.  
  - `class_map` persistível via `globals()["class_map"] = class_map` (usado no N1 para alimentar `meta.json`).

---

## 📚 Catálogo: `TableStore`

Mini-catálogo para múltiplos DataFrames nomeados com *current*:
```python
T = TableStore(initial={"main": df}, current="main")
T.add("features_v1", df2, set_current=True)
df = T.get()         # pega o current
df_raw = T["main"]   # dict-like
display(T.list())    # inventário com memória
```

---

## 🧪 Exemplos (copiar-e-colar)

### 1) Datas com cfg + features
```python
df, rep, parsed = ud.parse_dates_with_report_cfg(
    df,
    {"detect_regex": r"(date|data|_at$|_date$)", "min_ratio": 0.8, "dayfirst": False}
)
created = ud.expand_date_features_plus(df, parsed, features=("year","month","week","is_month_end"))
```

### 2) Outliers com persistência de resumo
```python
df, out_info = ud.apply_outlier_flags(df, config)
# out_info["persisted"] → {'report_relpath': 'outliers/summary.csv', 'rows': ...} quando habilitado
```

### 3) Texto com keywords e blacklist
```python
df, txt_sum = ud.extract_text_features(
    df, keywords=["error","cancel","premium"], blacklist=["customerID"],
    export_summary=True, summary_dir=ud.get_artifacts_dir("text_features")
)
```

### 4) Encode & Scale seguras
```python
ENC = {"exclude_cols": ["Churn","customerID"], "high_cardinality_threshold": 50}
SCL = {"exclude_cols": ["Churn"], "method": "standard"}
df_enc, meta = ud.apply_encoding_and_scaling(df, {"encoding": ENC, "scaling": SCL})
```

### 5) Exportações com caminho relativo à raiz
```python
root = ud.ensure_project_root()
ud.save_report_df(df.head(10), "quick/preview.csv", root=root)  # → reports/quick/preview.csv
art_dir = ud.get_artifacts_dir("export")                       # → reports/artifacts/export
```

---

## 🔖 Convenções e Logs

- Sufixos de auditoria: `_is_outlier`, `was_missing`, `<col>_num`, `<col>_has_<kw>`.
- Logs via `logger` do módulo (`reports/data_preparation.log` quando configurado no notebook).

---

## ✅ Dependências

- `pandas`, `numpy`
- `scikit-learn` (para encode/scale e imputações avançadas)
- Python ≥ 3.10 recomendado
- (Opcional) `joblib` para artefatos; `weasyprint`/`pandoc` para `md_to_pdf`.

---

## 🔁 Compatibilidade Retroativa

Este módulo mantém **aliases e assinaturas compatíveis** com versões anteriores:
- `resolve_n1_paths()` aceita chamadas antigas (com/sem `config`).
- `TableStore` preserva métodos (`add/get/use/list`) e acesso `dict-like`.
- `load_table_simple` aceita `fmt` **ou** o `read_opts` via *args*.

---

## 📌 Dicas de uso no N1

- Use `ud.get_artifacts_dir("<subdir>")` para **todas** as saídas auxiliares do N1 (ex.: `export`, `text_features`, `calendar`, `outliers`).  
- Garanta a *seed* global cedo com `ud.set_random_seed(seed)` (ou defina `RANDOM_SEED` pelo `config`).  
- Ao criar o **target**, propague `class_map` para o `meta.json` e para o N2.

---

## 🧾 Exportações (API)

Principais nomes expostos via `__all__`:  
`ensure_project_root`, `load_config`, `load_manifest`, `save_manifest`, `update_manifest`, `record_step`, `with_step`,  
`save_artifact`, `load_artifact`, `save_report_df`, `save_text`,  
`N1Paths`, `resolve_n1_paths`, `path_of`,  
`list_directory_files`, `infer_format_from_suffix`, `load_csv`, `load_table_simple`, `save_table`, `suggest_source_path`,  
`strip_whitespace`, `infer_numeric_like`, `n1_quality_typing`, `n1_quality_typing_dict`,  
`simple_impute_with_flags`, `deduplicate_rows`, `detect_outliers_iqr`, `detect_outliers_zscore`, `apply_outlier_flags`,  
`normalize_categories`, `encode_categories`, `encode_categories_safe`, `scale_numeric`, `scale_numeric_safe`, `apply_encoding_and_scaling`,  
`detect_date_candidates`, `parse_dates_with_report`, `parse_dates_with_report_cfg`, `expand_date_features`, `expand_date_features_plus`, `build_calendar_from`,  
`extract_text_features`,  
`build_target`, `ensure_target_from_config`,  
`TableStore`, `basic_overview`, `missing_report`, `merge_chain`,  
`generate_human_report_md`, `md_to_pdf`,  
`set_random_seed`, `set_display`,  
`UTILS_DATA_VERSION`.
