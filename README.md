# 🧰 `utils/` — Utility Toolkit for Data Projects (v1.2.2-merged)

Coleção de utilitários usada pelos notebooks (N1→N3) para **ingestão**, **limpeza**, **engenharia de atributos**, **datas**, **texto**, **codificação/escala**, **catálogo de DataFrames**, **artefatos**, **manifest** e, no N2, **UI futurista com Grid/Random Search (Hyperdrive)**.  
Módulo principal: **`utils/utils_data.py`** (versão `UTILS_DATA_VERSION = "1.2.2-merged"`).

> Import típico no notebook:
> ```python
> import importlib, utils.utils_data as ud
> importlib.reload(ud)        # útil durante edição do módulo
> from utils.utils_data import TableStore
> ```

---

## 🧭 Descoberta de raiz, config e manifest

- `ensure_project_root() -> Path`  
  Sobe a árvore até `config/defaults.json`, devolve a **raiz do projeto** e injeta `utils/` em `sys.path`.

- `load_config(base_abs=None, local_abs=None) -> dict`  
  Lê `config/defaults.json` e faz *deep merge* com `config/local.json` se existir.

- Manifest helpers: `load_manifest()`, `save_manifest()`, `update_manifest()`, `record_step(name, details=None)`, e o *context manager* `with_step(name, details=None)` para auditar etapas com timestamps.

---

## 📦 Artefatos, relatórios e paths

- `save_artifact(obj, name)` / `load_artifact(name)`  
  Salva/carrega `.joblib` em `artifacts/`. Registra passo no manifest.

- `save_report_df(df, rel_path)` e `save_text(text, rel_path)`  
  Persistem em `reports/<rel_path>`, criando pastas conforme necessário.

- `get_artifacts_dir(subdir: str | None = None) -> Path`  
  Garante e retorna `reports/artifacts[/<subdir>]`. **Observação:** a função aparece duas vezes no arquivo (mesma assinatura/propósito) — comportamento idêntico.

- Paths N1 (dataclass): `N1Paths` + helpers `resolve_n1_paths(...)` (compatível com chamadas antigas) e `path_of(*parts)`.

---

## 📥 Ingestão & 📤 Exportação

- `list_directory_files(path) -> DataFrame`  
  Inventário recursivo de arquivos (tamanho, sufixo, mtime).

- `suggest_source_path(directory, pattern="*.csv", max_rows=50) -> DataFrame`  
  “Vitrine” rápida de possíveis fontes.

- `infer_format_from_suffix(path) -> "csv"|"parquet"`  
  Infere formato pelo sufixo.

- `load_csv(...)` e `load_table_simple(path, fmt=None, *args, **kwargs) -> DataFrame`  
  Compatível com chamadas antigas (dicionário posicional de opções) e autoformato.

- `save_table(df, path, fmt=None, **kwargs) -> Path`  
  Respeita extensão e loga linhas salvas.

---

## 🔎 Visões rápidas, merge e qualidade

- `basic_overview(df) -> dict`  
  Linhas, colunas, dtypes, memória MB.

- `missing_report(df) -> DataFrame`  
  `%` e contagem de nulos por coluna.

- `merge_chain(base, tables: dict, steps: list) -> DataFrame`  
  Orquestra merges encadeados declarativos (com `on`/`left_on`/`right_on`, `validate`, `drop_cols`).

- `strip_whitespace(df, cols=None)`  
  Trim/colapso de espaços para texto.

- `infer_numeric_like(df, cols=None, decimal=".", thousands=None, report_path="cast_report.csv") -> (df, report)`  
  Converte strings numéricas com relatório em `reports/`.

- `n1_quality_typing_dict(df, config)` e `n1_quality_typing(df, config)`  
  Pipeline compacto (strip → inferência numérica) com logs e relatório de cast.

---

## 🧩 Faltantes, duplicidade e outliers

- `simple_impute_with_flags(df, strategy="median") -> (df, meta)`  
  Imputa numéricas (média/mediana) e categoriza faltantes com `<col>_was_missing`.

- `handle_missing_step(df, config, save_reports=True, prefer="auto") -> dict`  
  Orquestra “faltantes” ponta-a-ponta (relatórios before/after + estratégias `simple`/`knn`/`iterative` com fallback).

- `deduplicate_rows(df, subset=None, keep="first") -> (df, log)`  
  Remove duplicadas e devolve log com removidas. **Obs.:** ao final do arquivo existe uma **segunda** definição compatível que aceita `config` e retorna apenas `df` (preferir a primeira assinatura; a segunda preserva compat retroativa).

- `detect_outliers_iqr(df, cols=None, k=1.5) -> DataFrame[bool]`  
- `detect_outliers_zscore(df, cols=None, z=3.0) -> DataFrame[bool]`  
  Máscaras booleanas por coluna.

- `apply_outlier_flags(df, config=None, *, method=None, iqr_factor=None, z_threshold=None, cols=None, exclude_cols=None, exclude_binaries=None, flag_suffix="_is_outlier", persist=None, persist_relpath=None) -> (df, info)`  
  Cria `<col>_is_outlier` por IQR/Z-score, com exclusões, persistência opcional de **resumo** em `reports/outliers/summary.csv`.

---

## 🔤 Categóricas & 🔢 Numéricas

- `normalize_categories(df, cols=None, case="lower", trim=True, strip_accents=True, cfg=None, report_path=None) -> (df, report)`  
  Normalização (case/acentos/espacos) com mapeamentos globais/por coluna e relatório opcional.

- `encode_categories(df, cols=None, drop_first=False, high_cardinality_threshold=20, top_k=None, other_label="__OTHER__") -> (df, meta)`  
  One-hot (com *top-k* p/ alta cardinalidade).  
  `encode_categories_safe(df, exclude_cols=None, **kwargs)`.

- `scale_numeric(df, method="standard"|"minmax", cols=None) -> (df, meta)`  
  `scale_numeric_safe(df, exclude_cols=None, **kwargs)`.

- `apply_encoding_and_scaling(df, config) -> (df, meta)`  
  Orquestra encode→scale lendo `config["encoding"]` e `config["scaling"]`.

---

## 📅 Datas

- `detect_date_candidates(df, regex_list=None) -> list[str]`  
  Heurística por nome.

- `parse_dates_with_report(df, cols=None, dayfirst=False, utc=False, errors="coerce", min_ratio=0.6, report_path="date_parse_report.csv") -> (df, report)`  
  Parsing com relatório de sucesso/erros.

- **Nova:** `parse_dates_with_report_cfg(df, cfg) -> (df, report, parsed_cols)`  
  Variante via dicionário (regex/explicit, formatos, `min_ratio`) e lista de colunas convertidas.

- `expand_date_features(df, cols) -> df`  
  `*_year`, `*_month`, `*_day`, `*_dow`, `*_week`, `*_quarter`.

- **Nova:** `expand_date_features_plus(df, date_cols, *, features=(...), prefix_mode="auto") -> list[str]`  
  Suporta `dayofweek`, `is_month_start`, `is_month_end`, etc., e retorna nomes criados.

- `build_calendar_from(df, col, freq="D") -> DataFrame`  
  Gera dimensão-calendário entre min/max da coluna.

---

## 📝 Texto

- **Nova (ampliada):** `extract_text_features(df, *, lower=True, strip_collapse_ws=True, keywords=None, blacklist=None, export_summary=True, summary_dir=None) -> (df, summary_df)`  
  Limpeza leve + métricas (`_len`, `_word_count`) e flags por *keywords*; resumo opcional em CSV.  
  (Há também uma versão mais simples com assinatura antiga.)

---

## 🎯 Target

- `build_target(df, config) -> (df, meta)`  
  Regra simples `col/op/value` para derivar alvo.

- `ensure_target_from_config(df, config, verbose=False) -> (df, target_name, class_map, report_df)`  
  Garante/deriva a coluna `target` a partir de `config["target"]` (mapeando `positive`/`negative` quando aplicável).

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

## 🧪 Métricas rápidas, plots e persistência de modelos (N2)

- `compute_metrics(y_true, y_pred) -> dict`  
  Acurácia e F1 com média adequada (binária vs. macro).

- `try_plot_roc(clf, X_test, y_test) -> bool`  
  Tenta plotar ROC (binário + `predict_proba`).

- `persist_artifacts(name, pipeline, metrics, params, models_dir: Path, reports_dir: Path)`  
  Salva `.joblib` + `*_metrics.json` + `*_params.json` e anexa entrada em `reports/manifest.jsonl`.

---

## 🧭 Helpers específicos para N2 / organização de pastas

- `get_project_root() -> Path`  
  Atalho p/ raiz (usa `ensure_project_root`).

- `ensure_artifact_dirs(cfg) -> (artifacts_dir, reports_dir, models_dir)`  
  Garante diretórios padrão e faz log.

- `resolve_processed_path(cfg) -> Path`  
  Encontra o arquivo final do N1 em `data/processed` com heurísticas e mensagens de diagnóstico.  
  (**Aliases compat:** `ensure_dirs(cfg)` e `discover_processed_path(cfg)` disponíveis no bloco de retrocompatibilidade.)

- `summarize_columns(df) -> (numeric_cols, categorical_cols, other_cols)`  
  Particiona colunas por tipo para o N2.

---

## 🚀 N2 — UI Futurista + Hyperdrive (Grid/Random Search)

Recursos que permitem montar, **no notebook**, um painel “painel interdimensional” com seleção de modelos, abas de hiperparâmetros com travas, treino direto e busca de hiperparâmetros (GridSearchCV / RandomizedSearchCV):

- `n2_inject_css_theme()`  
  Injeta o tema visual (CSS) usado pelo painel futurista.

- `n2_model_registry() -> dict`  
  Registro de modelos (Dummy, LogisticRegression, KNN, RandomForest) e widgets dos hiperparâmetros.

- `n2_build_models_ui(preprocess, X_train, y_train, X_test, y_test, models_dir, reports_dir)`  
  Monta toda a UI:  
  1) **Seleção de modelos** (checkbox) com **trava de abas**;  
  2) **Abas de hiperparâmetros** com widgets;  
  3) **Treino direto** (usa os hiperparâmetros atuais dos widgets);  
  4) **Hyperdrive** — gera automaticamente um `param_grid` a partir dos widgets e executa Grid/Random Search;  
  5) **Persistência opcional** do melhor pipeline/relatórios via `persist_artifacts`.

> **Uso típico no N2** (após definir `preprocess`, `X_train`, `y_train`, `X_test`, `y_test` e pastas):
> ```python
> artifacts_dir, reports_dir, models_dir = ud.ensure_artifact_dirs(cfg)
> ud.n2_inject_css_theme()
> ud.n2_build_models_ui(preprocess, X_train, y_train, X_test, y_test, models_dir, reports_dir)
> ```

---

## 🔖 Convenções e Logs

- Sufixos de auditoria: `_is_outlier`, `was_missing`, `<col>_len`, `<col>_word_count`, `<col>_has_<kw>`.
- Logs via `logger` do módulo (varia conforme funções chamadas).

---

## ✅ Dependências

- Python ≥ 3.10  
- `pandas`, `numpy`, `scikit-learn` (encode/scale/imputers)  
- (Opcional) `joblib` para artefatos; `weasyprint` ou `pandoc` para `md_to_pdf`.

---

## 🔁 Compatibilidade Retroativa

- Assinaturas preservadas para `resolve_n1_paths`, `load_table_simple`, `n1_quality_typing`, `TableStore`, etc.  
- Aliases auxiliares (`ensure_dirs`, `discover_processed_path`) mantidos para ambientes antigos.

---

## 📌 API pública (principais símbolos)

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
`get_project_root`, `ensure_artifact_dirs`, `resolve_processed_path`, `summarize_columns`,  
`compute_metrics`, `try_plot_roc`, `persist_artifacts`,  
`n2_inject_css_theme`, `n2_model_registry`, `n2_build_models_ui`,  
`UTILS_DATA_VERSION`.

---

### O que foi acrescentado vs. teu README anterior

- Seção **N2 — UI Futurista + Hyperdrive** com `n2_inject_css_theme`, `n2_model_registry`, `n2_build_models_ui`.  
- Helpers **N2**: `get_project_root`, `ensure_artifact_dirs`, `resolve_processed_path` (+ aliases `ensure_dirs`, `discover_processed_path`), `summarize_columns`.  
- Utilitários de **métricas/plots/persistência**: `compute_metrics`, `try_plot_roc`, `persist_artifacts`.  
- Observação sobre **duplicidade** de `get_artifacts_dir` e **duas** assinaturas de `deduplicate_rows` (mantidas por compatibilidade).
