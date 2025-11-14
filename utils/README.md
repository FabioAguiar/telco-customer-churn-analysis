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

## 🧾 Lista completa de funções do `utils_data.py`

Abaixo está a relação numerada de todas as funções de topo implementadas em `utils_data.py (versão atualizada)`, com suas assinaturas simplificadas e uma breve descrição.

1. `def _apply_tab_title_style(tab, idx, title, enabled):` — Função Apply tab title style
   Função utilitária `_apply_tab_title_style` utilizada em fluxos internos do template de dados.

2. `def _badge(text, kind):` — Função Badge
   Função utilitária `_badge` utilizada em fluxos internos do template de dados.

3. `def _bool_badge(flag):` — Função Bool badge
   Função utilitária `_bool_badge` utilizada em fluxos internos do template de dados.

4. `def _build_search_space(model_registry, model_name):` — Função Build search space
   Função utilitária `_build_search_space` utilizada em fluxos internos do template de dados.

5. `def _card(title, subtitle, accent):` — Função Card
   Função utilitária `_card` utilizada em fluxos internos do template de dados.

6. `def _card_html(title, subtitle, accent):` — Cria um 'card' simples para separar seções no notebook.
   Cria um 'card' simples para separar seções no notebook.

7. `def _deep_merge(a, b):` — Função Deep merge
   Função utilitária `_deep_merge` utilizada em fluxos internos do template de dados.

8. `def _dtypes_summary(df):` — Função Dtypes summary
   Função utilitária `_dtypes_summary` utilizada em fluxos internos do template de dados.

9. `def _ensure_datetime_with_ratio(s):` — Tenta converter uma série para datetime de forma tolerante.
   Tenta converter uma série para datetime de forma tolerante. Retorna (serie_convertida, parse_ratio).
Não levanta warning; cai em NaT quando não converte.

10. `def _find_target_case(df, cfg, fallback):` — Encontra a coluna alvo case-insensitive usando config (target.name/target_col...
   Encontra a coluna alvo case-insensitive usando config (target.name/target_column).

11. `def _find_up(relative_path, start):` — Função Find up
   Função utilitária `_find_up` utilizada em fluxos internos do template de dados.

12. `def _fmt_auto(x, decimals):` — Se for inteiro, sem casas; senão, até 'decimals' casas (trim).
   Se for inteiro, sem casas; senão, até 'decimals' casas (trim).

13. `def _fmt_compact(x):` — Formata números sem zeros inúteis.
   Formata números sem zeros inúteis. - int -> 123 - float -> até 3 casas, removendo zeros (ex.: 6.821,
0.5, 12)

14. `def _fmt_mem_mb(x_mb):` — Formata memória em MB com até 2 casas, sem zeros finais.
   Formata memória em MB com até 2 casas, sem zeros finais.

15. `def _manifest_path(root):` — Função Manifest path
   Função utilitária `_manifest_path` utilizada em fluxos internos do template de dados.

16. `def _missing_top_with_dtype(df, top):` — Função Missing top with dtype
   Função utilitária `_missing_top_with_dtype` utilizada em fluxos internos do template de dados.

17. `def _normalize_str(x):` — Função Normalize str
   Função utilitária `_normalize_str` utilizada em fluxos internos do template de dados.

18. `def _overview_table_neat(df):` — Função Overview table neat
   Função utilitária `_overview_table_neat` utilizada em fluxos internos do template de dados.

19. `def _params_vbox(spec_params):` — Função Params vbox
   Função utilitária `_params_vbox` utilizada em fluxos internos do template de dados.

20. `def _quiet_utils_data_logger():` — Função Quiet utils data logger
   Função utilitária `_quiet_utils_data_logger` utilizada em fluxos internos do template de dados.

21. `def _read_table_auto(_path):` — Leitura robusta para parquet/csv/xlsx.
   Leitura robusta para parquet/csv/xlsx.

22. `def _resolve_n1_paths_core(root):` — Função Resolve n1 paths core
   Função utilitária `_resolve_n1_paths_core` utilizada em fluxos internos do template de dados.

23. `def _set_disabled(box, disabled):` — Função Set disabled
   Função utilitária `_set_disabled` utilizada em fluxos internos do template de dados.

24. `def _shorten_path(p, keep_parts):` — Abrevia caminho para exibição (…/<tail>).
   Abrevia caminho para exibição (…/<tail>).

25. `def _show_block(title, subtitle, df_display, accent):` — Função Show block
   Função utilitária `_show_block` utilizada em fluxos internos do template de dados.

26. `def _styler_hide_index(sty):` — Função Styler hide index
   Função utilitária `_styler_hide_index` utilizada em fluxos internos do template de dados.

27. `def _to_html_table(df, caption):` — Função To html table
   Função utilitária `_to_html_table` utilizada em fluxos internos do template de dados.

28. `def _top_k_categories(s, k):` — Função Top k categories
   Função utilitária `_top_k_categories` utilizada em fluxos internos do template de dados.

29. `def _widget_to_candidates(w):` — Função Widget to candidates
   Função utilitária `_widget_to_candidates` utilizada em fluxos internos do template de dados.

30. `def apply_encoding_and_scaling(df, config):` — Função Apply encoding and scaling
   Função utilitária `apply_encoding_and_scaling` utilizada em fluxos internos do template de dados.

31. `def apply_outlier_flags(df, config):` — Cria colunas booleanas <col>_is_outlier para cada coluna indicada (ou numéric...
   Cria colunas booleanas <col>_is_outlier para cada coluna indicada (ou numéricas) a partir de um
*mask* de outliers calculado por IQR ou z-score. Lê preferências do `config` atual:   -
config["detect_outliers"] (bool)   - config["outlier_method"] ("iqr"|"zscore")   -
config["outliers"] dict com:       - cols (lista ou null)            -> restringe a colunas
específicas       - exclude_cols (lista)            -> ignora colunas       - exclude_binaries
(bool)         -> omite colunas {0,1} e {True,False}       - iqr_factor (float)       - z_threshold
(float)       - persist_summary (bool)          -> salva CSV de resumo       - persist_relpath (str)
-> ex: "outliers/summary.csv"  Retorna (df_modificado, info_dict). O df retorna *cópia* com flags
adicionadas.

32. `def basic_overview(df):` — Função Basic overview
   Função utilitária `basic_overview` utilizada em fluxos internos do template de dados.

33. `def build_calendar_from(df, col, freq):` — Função Build calendar from
   Função utilitária `build_calendar_from` utilizada em fluxos internos do template de dados.

34. `def build_target(df, config):` — Função Build target
   Função utilitária `build_target` utilizada em fluxos internos do template de dados.

35. `def coerce_df(obj):` — Garante um DataFrame. Se vier (df, meta), retorna o primeiro elemento.
   Garante um DataFrame. Se vier (df, meta), retorna o primeiro elemento.

36. `def deduplicate_rows(df, subset, keep):` — Função Deduplicate rows
   Função utilitária `deduplicate_rows` utilizada em fluxos internos do template de dados.

37. `def deduplicate_rows(df, subset, keep, config):` — Remove linhas duplicadas do DataFrame.
   Remove linhas duplicadas do DataFrame.  Parâmetros:   - subset: lista de colunas a considerar (None
= todas)   - keep: 'first' (mantém a 1ª), 'last' (mantém a última) ou False (remove todas as
duplicadas)   - config: dicionário de configuração (opcional) com chaves:       {
"deduplicate": {           "subset": ["col1", "col2"],  # colunas de referência           "keep":
"first"         }       }  Retorna:   df sem duplicadas.

38. `def detect_date_candidates(df, regex_list):` — Função Detect date candidates
   Função utilitária `detect_date_candidates` utilizada em fluxos internos do template de dados.

39. `def detect_outliers_iqr(df, cols, k):` — Função Detect outliers iqr
   Função utilitária `detect_outliers_iqr` utilizada em fluxos internos do template de dados.

40. `def detect_outliers_zscore(df, cols, z):` — Função Detect outliers zscore
   Função utilitária `detect_outliers_zscore` utilizada em fluxos internos do template de dados.

41. `def dtypes_summary(df):` — Contagem por dtype (string).
   Contagem por dtype (string).

42. `def encode_categories(df, cols, drop_first, high_cardinality_threshold, top_k, other_label):` — Função Encode categories
   Função utilitária `encode_categories` utilizada em fluxos internos do template de dados.

43. `def encode_categories_safe(df, exclude_cols, **kwargs):` — Função Encode categories safe
   Função utilitária `encode_categories_safe` utilizada em fluxos internos do template de dados.

44. `def ensure_artifact_dirs(cfg):` — Função Ensure artifact dirs
   Função utilitária `ensure_artifact_dirs` utilizada em fluxos internos do template de dados.

45. `def ensure_project_root():` — Função Ensure project root
   Função utilitária `ensure_project_root` utilizada em fluxos internos do template de dados.

46. `def ensure_target_from_config(df, config, verbose):` — Garante a existência/consistência do target conforme o config['target'].
   Garante a existência/consistência do target conforme o config['target'].  Retorna: df, target_name,
class_map, report_df - Nunca sobrescreve um target existente. - Compara valores de forma case-
insensitive e com strip().

47. `def ensure_utils_import():` — Garante que a raiz do projeto e o pacote utils/ estejam acessíveis no sys.path.
   Garante que a raiz do projeto e o pacote utils/ estejam acessíveis no sys.path. Retorna o
PROJECT_ROOT detectado.  Uso típico no notebook (N1/N2/N3):  >>> from utils.utils_data import
ensure_utils_import >>> PROJECT_ROOT = ensure_utils_import() >>> import utils.utils_data as ud  # já
deve funcionar sem erro de módulo  Esta função é não-intrusiva: não altera comportamentos
existentes, apenas ajusta o sys.path e cria utils/__init__.py se necessário.

48. `def expand_date_features(df, cols):` — Função Expand date features
   Função utilitária `expand_date_features` utilizada em fluxos internos do template de dados.

49. `def expand_date_features_plus(df, date_cols):` — Cria colunas derivadas a partir de colunas datetime.
   Cria colunas derivadas a partir de colunas datetime.  features suportados:   - year, month, day,
dayofweek, quarter, week, is_month_start, is_month_end  Retorna:   lista de nomes das colunas
criadas

50. `def extract_text_features(df, cols, report_path, root):` — Função Extract text features
   Função utilitária `extract_text_features` utilizada em fluxos internos do template de dados.

51. `def extract_text_features(df):` — Extrai métricas básicas de colunas textuais (string/object) e gera relatório...
   Extrai métricas básicas de colunas textuais (string/object) e gera relatório de texto.  Parâmetros:
- lower: converte para minúsculas   - strip_collapse_ws: remove espaços extras   - keywords: lista
de palavras-chave a serem contadas   - blacklist: colunas a ignorar   - export_summary: salva CSV de
resumo (True/False)   - summary_dir: caminho para salvar o relatório (Path ou string)  Retorna:
(DataFrame transformado, DataFrame resumo)

52. `def extract_text_features_fast(df):` — Versão otimizada: acumula novas colunas em um dict e concatena de uma vez,
   Versão otimizada: acumula novas colunas em um dict e concatena de uma vez, evitando alta
fragmentação de DataFrame.

53. `def fix_avg_charge_zero_tenure(df, avg_col, tenure_col, create_flag):` — Regra derivada: se tenure == 0 e avg_charge_per_month é NaN -> setar 0 e flagar.
   Regra derivada: se tenure == 0 e avg_charge_per_month é NaN -> setar 0 e flagar.

54. `def fix_target_then_summary(df, config, verbose):` — Envolve ensure_target_from_config e, se as classes não forem reconhecidas,
   Envolve ensure_target_from_config e, se as classes não forem reconhecidas, normaliza labels e tenta
novamente.

55. `def generate_human_report_md(df, title):` — Função Generate human report md
   Função utilitária `generate_human_report_md` utilizada em fluxos internos do template de dados.

56. `def get_artifacts_dir(subdir):` — Retorna o diretório de artefatos do projeto (`reports/artifacts`), garantindo...
   Retorna o diretório de artefatos do projeto (`reports/artifacts`), garantindo sua existência.
Parâmetros:   - subdir (opcional): nome de subpasta dentro de artifacts (ex.: "outliers" ou
"calendar")  Exemplo:   >>> path = get_artifacts_dir("calendar")   >>> print(path)
C:/Users/fabio/Projetos DEV/data projects/data-project-template/reports/artifacts/calendar

57. `def get_artifacts_dir(subdir):` — Retorna o diretório de artefatos do projeto (`reports/artifacts`), garantindo...
   Retorna o diretório de artefatos do projeto (`reports/artifacts`), garantindo sua existência.
Parâmetros:   - subdir (opcional): nome de subpasta dentro de artifacts (ex.: "outliers" ou
"calendar")  Exemplo:   >>> path = get_artifacts_dir("calendar")   >>> print(path)
C:/Users/fabio/Projetos DEV/data projects/data-project-template/reports/artifacts/calendar

58. `def get_project_root():` — Função Get project root
   Função utilitária `get_project_root` utilizada em fluxos internos do template de dados.

59. `def handle_missing_step(df, config, save_reports, prefer):` — Executa a etapa de 'faltantes' ponta-a-ponta:
   Executa a etapa de 'faltantes' ponta-a-ponta:   - Gera relatório 'antes'
(reports/missing/before.csv)   - Aplica estratégia (simple | knn | iterative). 'auto' lê do config
com fallbacks   - Gera relatório 'depois' (reports/missing/after.csv)  Compatibilidade: mantém
assinatura, contratos e caminhos do método original. Melhorias: não cria flags encadeadas (evita
*_was_missing_was_missing...) e            só flaggeia colunas que tinham NaN de fato. Retorna dict:
{'df','before','after','strategy','imputed_cols'}

60. `def human_size(num_bytes):` — Converte bytes em B/KB/MB/GB/TB com formatação amigável.
   Converte bytes em B/KB/MB/GB/TB com formatação amigável. - Para KB/MB/GB: 0 casas decimais se >=
100; 1 casa se < 100.

61. `def infer_format_from_suffix(path):` — Função Infer format from suffix
   Função utilitária `infer_format_from_suffix` utilizada em fluxos internos do template de dados.

62. `def infer_numeric_like(df, cols, decimal, thousands, report_path, root):` — Função Infer numeric like
   Função utilitária `infer_numeric_like` utilizada em fluxos internos do template de dados.

63. `def list_directory_files(path):` — Função List directory files
   Função utilitária `list_directory_files` utilizada em fluxos internos do template de dados.

64. `def list_raw_sources_safe(raw_dir, pattern, show_rel, rel_root):` — Lista arquivos em data/raw sem expor caminho absoluto.
   Lista arquivos em data/raw sem expor caminho absoluto. Retorna colunas: file, size, size_bytes,
modified, relpath (opcional).

65. `def load_artifact(name, root):` — Função Load artifact
   Função utilitária `load_artifact` utilizada em fluxos internos do template de dados.

66. `def load_config(base_abs, local_abs):` — Função Load config
   Função utilitária `load_config` utilizada em fluxos internos do template de dados.

67. `def load_csv(path, **kwargs):` — Função Load csv
   Função utilitária `load_csv` utilizada em fluxos internos do template de dados.

68. `def load_manifest(root):` — Função Load manifest
   Função utilitária `load_manifest` utilizada em fluxos internos do template de dados.

69. `def load_table_simple(path, fmt, *args, **kwargs):` — Compatível com:
   Compatível com: - load_table_simple(path, fmt=None, **read_opts) - load_table_simple(path, fmt,
read_opts_dict)

70. `def md_to_pdf(md_text, out_path, engine):` — Função Md to pdf
   Função utilitária `md_to_pdf` utilizada em fluxos internos do template de dados.

71. `def merge_chain(base, tables, steps):` — Função Merge chain
   Função utilitária `merge_chain` utilizada em fluxos internos do template de dados.

72. `def missing_report(df):` — Função Missing report
   Função utilitária `missing_report` utilizada em fluxos internos do template de dados.

73. `def missing_top(df, top):` — Top N colunas com mais faltantes + dtype, com formatação compacta.
   Top N colunas com mais faltantes + dtype, com formatação compacta.

74. `def n1_quality_typing(df, config, root):` — Compat: retorna (df, meta_dict).
   Compat: retorna (df, meta_dict).

75. `def n1_quality_typing_dict(df, config, root):` — Nova API: retorna dict com 'df', 'steps' e 'cast_report'.
   Nova API: retorna dict com 'df', 'steps' e 'cast_report'.

76. `def n2_bootstrap_and_load(project_root):` — Bootstrap compacto do N2:
   Bootstrap compacto do N2:   - resolve PROJECT_ROOT   - carrega config (defaults/local)   - garante
dirs (artifacts/reports/models)   - resolve processed_path e lê df   - encontra TARGET_COL (case-
insensitive)   - sumariza tipos via summarize_columns Retorna dict com chaves: project_root, config,
artifacts_dir, reports_dir, models_dir,                          processed_path, df, target_col,
num_cols, cat_cols, other_cols

77. `def n2_build_models_ui(preprocess, X_train, y_train, X_test, y_test, models_dir, reports_dir):` — Monta toda a UI de:
   Monta toda a UI de:   - seleção de modelos,   - abas de hiperparâmetros (com travas),   - treino
direto   - Hyperdrive (GridSearchCV/RandomizedSearchCV)

78. `def n2_inject_css_theme():` — Injeta o tema 'painel interdimensional' no notebook.
   Injeta o tema 'painel interdimensional' no notebook.

79. `def n2_model_registry():` — Define os modelos disponíveis e seus widgets de hiperparâmetros.
   Define os modelos disponíveis e seus widgets de hiperparâmetros.

80. `def normalize_categories(df, cols, case, trim, strip_accents, cfg, report_path, root):` — Modo compat + avançado.
   Modo compat + avançado. - Sem cfg: usa (case/trim/strip_accents) simples. - Com cfg: espera chaves
como exclude, collapse_ws, null_values, global_map, per_column_map, cast_to_category. Retorna (df,
report) e opcionalmente salva o CSV do report se report_path for informado.

81. `def normalize_target_labels_inplace(df, target_name, positive_aliases, negative_aliases):` — Normaliza in-place os rótulos do target para 'yes'/'no' a partir de aliases c...
   Normaliza in-place os rótulos do target para 'yes'/'no' a partir de aliases comuns.

82. `def null_fill_from_config(df, config, root):` — Lê config["null_fill_with_flag"] (se existir e enabled) e aplica null_fill_wi...
   Lê config["null_fill_with_flag"] (se existir e enabled) e aplica null_fill_with_flag. Exemplo de
config (defaults.json):   "null_fill_with_flag": {     "enabled": true,     "numeric_fill": 0,
"categorical_fill": "__MISSING__",     "cols_numeric_zero": ["avg_charge_per_month"],
"flag_suffix": "_was_missing",     "report_relpath": "nulls/fill_summary.csv"   }  Retorna: (df,
meta). Se o recurso estiver desabilitado/ausente, retorna df inalterado e meta vazio.

83. `def null_fill_from_config(df, config, root):` — Estende o comportamento: se 'cols_numeric_zero' existir, usa essa lista.
   Estende o comportamento: se 'cols_numeric_zero' existir, usa essa lista. Se não existir, varre
somente colunas com NaN.

84. `def null_fill_with_flag(df, cols, numeric_fill, categorical_fill, flag_suffix):` — Preenche nulos nas colunas indicadas e cria flags <col>_was_missing (0/1).
   Preenche nulos nas colunas indicadas e cria flags <col>_was_missing (0/1). - Colunas numéricas
recebem `numeric_fill`; - Colunas não-numéricas recebem `categorical_fill`.  Retorna: (df_novo,
meta)   meta = {     "filled_cols": [...],     "flags_created": N,     "before_summary": DataFrame,
"after_summary": DataFrame   }

85. `def overview_table(df):` — Resumo compacto de linhas/colunas/memória.
   Resumo compacto de linhas/colunas/memória.

86. `def parse_dates_with_report(df, cols, dayfirst, utc, errors, min_ratio, report_path, max_fail_samples, root):` — Função Parse dates with report
   Função utilitária `parse_dates_with_report` utilizada em fluxos internos do template de dados.

87. `def parse_dates_with_report_cfg(df, cfg):` — Variante que lê um dicionário de configuração (cfg) e retorna:
   Variante que lê um dicionário de configuração (cfg) e retorna:   (df_convertido, report_df,
parsed_cols)  cfg:   - detect_regex: str regex para auto-detecção (default:
r"(date|data|dt_|_dt$|_date$)")   - explicit_cols: list[str] colunas explícitas (prioridade sobre
regex)   - dayfirst: bool (default False)   - utc: bool (default False)   - formats: list[str]
formatos strftime (ex.: ["%d/%m/%Y", "%Y-%m-%d"]); se vazio, usa auto   - min_ratio: float entre 0 e
1 (default 0.80) -> taxa mínima de parsing aceitável   - report_path: str|Path opcional para
persistir o relatório em reports/  Observações:   - Não altera a função existente
parse_dates_with_report; é uma variante complementar.

88. `def path_of(*parts):` — Função Path of
   Função utilitária `path_of` utilizada em fluxos internos do template de dados.

89. `def recompute_charge_gap_features(df, total_col, monthly_col, tenure_col, gap_col, gap_log1p_col):` — Recalcula charge_gap = TotalCharges - (MonthlyCharges * tenure)
   Recalcula charge_gap = TotalCharges - (MonthlyCharges * tenure) e charge_gap_log1p usando log
assinado para evitar NaN de domínio.

90. `def record_step(name, details, root):` — Função Record step
   Função utilitária `record_step` utilizada em fluxos internos do template de dados.

91. `def render_calendar_step(info):` — Renderiza um resumo amigável da etapa calendário.
   Renderiza um resumo amigável da etapa calendário.

92. `def render_categorical_candidates(df, cand, max_unique_ratio, max_unique_count, include_numeric_small, base_dir, top_n, head_bin, head_service):` — Renderiza cards organizados para candidatos de padronização categórica.
   Renderiza cards organizados para candidatos de padronização categórica. - Se `cand` for None, chama
`suggest_categorical_candidates` com os limites fornecidos. - Se `base_dir` for um caminho válido,
salva CSVs em base_dir/'categorical_candidates'. - Não altera nenhuma função existente; apenas usa
utilitários já presentes.

93. `def render_categorical_normalization(result, report_head):` — Renderiza cartões HTML e o relatório gerado por `run_categorical_normalization`.
   Renderiza cartões HTML e o relatório gerado por `run_categorical_normalization`.

94. `def render_date_step(parsed_cols, parse_report, candidates, created_features):` — Renderiza cards e tabelas para a etapa de datas.
   Renderiza cards e tabelas para a etapa de datas.

95. `def render_encoding_and_scaling(info):` — Renderiza um painel compacto com:
   Renderiza um painel compacto com:   - Card de título   - Resumo (top 20 do 'summary', se existir)
- Totais de colunas codificadas/escaladas e diretório de artefatos (se houver)

96. `def render_missing_step(res, df):` — Renderiza um resumo visual e auditável do tratamento de valores faltantes.
   Renderiza um resumo visual e auditável do tratamento de valores faltantes.  Parameters ----------
res : dict     Resultado retornado por utils_data.handle_missing_step() df : pandas.DataFrame
DataFrame resultante após a imputação

97. `def render_n2_status_panel_light(project_root, processed_path, df, target_name, num_cols, cat_cols, other_cols, test_size, random_state, scale_numeric, target_counts, target_pct, keep_path_parts):` — Renderiza o painel limpo do N2 (paleta Aqua/Roxo, fonte maior, caminho abrevi...
   Renderiza o painel limpo do N2 (paleta Aqua/Roxo, fonte maior, caminho abreviado).

98. `def render_null_fill_report(meta):` — Renderiza um card simples com o que foi preenchido e flags criadas.
   Renderiza um card simples com o que foi preenchido e flags criadas. Usa os helpers de card já
existentes no utils.

99. `def render_outlier_flags(out_info, df, top_n, title):` — Exibe cards com resumo e ranking de flags de outlier criadas.
   Exibe cards com resumo e ranking de flags de outlier criadas. - out_info: dict retornado por
apply_outlier_flags(...)   chaves esperadas (tolerante a ausência): created_flags, method, counts,
summary_path - df: DataFrame (opcional) para calcular % de linhas afetadas - top_n: quantas flags
exibir no ranking

100. `def render_quality_and_typing(result):` — Exibe os cards organizados com base no retorno do run_quality_and_typing().
   Exibe os cards organizados com base no retorno do run_quality_and_typing().

101. `def render_target_summary(info):` — Renderiza um painel compacto e padronizado para a variável-alvo:
   Renderiza um painel compacto e padronizado para a variável-alvo: - Card com status e fonte - Tabela
de contagens e percentuais - Badge com taxa positiva e classes detectadas - Alerta de
desbalanceamento extremo (opcional)

102. `def render_text_features_summary(summary_df):` — Exibe um painel compacto e legível para o resumo de features de texto.
   Exibe um painel compacto e legível para o resumo de features de texto. Não altera dados; apenas
organiza a visualização em três blocos:   1) Card + métricas gerais   2) Colunas com maior avg_len /
avg_words   3) Totais de hits por keyword (se houver)

103. `def resolve_n1_paths(*args):` — Compatível com duas formas:
   Compatível com duas formas: - resolve_n1_paths() ou resolve_n1_paths(root) -
resolve_n1_paths(config, root)  # notebooks antigos

104. `def resolve_processed_path(cfg):` — Função Resolve processed path
   Função utilitária `resolve_processed_path` utilizada em fluxos internos do template de dados.

105. `def run_calendar_step(df):` — Orquestra a criação da dimensão calendário:
   Orquestra a criação da dimensão calendário:   - Resolve parâmetros a partir do `config["calendar"]`
(se presente)   - Descobre coluna de data automaticamente quando não for informada   - Converte para
datetime com verificação de 'parse_ratio'   - Constrói, salva e (opcional) registra no catálogo   -
Retorna dict com artefatos e mensagens  Retorno:   {     "status": "ok" | "skipped" | "error",
"reason": <mensagem se skipped/error>,     "date_col": <coluna usada ou None>,     "freq": <freq>,
"output": <caminho final>,     "dim_date": <DataFrame ou None>,     "period": (start, end) ou None
}

106. `def run_categorical_normalization(df, cfg, report_path, silence_logs):` — Executa a padronização categórica com 'normalize_categories' (suporta API ava...
   Executa a padronização categórica com 'normalize_categories' (suporta API avançada e fallback).
Retorna:   {     "df": df_norm,     "report": cat_norm_report (DataFrame),     "impacto": DataFrame
Linhas/Colunas/Memória (antes/depois/Δ),     "_details": {...}   }

107. `def run_encoding_and_scaling(df, config):` — Executa a etapa unificada de Codificação Categórica & Escalonamento Numérico,
   Executa a etapa unificada de Codificação Categórica & Escalonamento Numérico, delegando para a
função já existente `apply_encoding_and_scaling(df, config)`.  Retorna:     df_out: DataFrame
transformado (códigos + escalas aplicadas)     info:   dict com chaves usuais:             -
"summary": DataFrame resumo (se existir)             - "encoded_cols": list[str]             -
"scaled_cols": list[str]             - "artifacts_dir": str | Path (se existir)

108. `def run_quality_and_typing(df, config):` — Executa a etapa de Qualidade & Tipagem com logs silenciados.
   Executa a etapa de Qualidade & Tipagem com logs silenciados. Retorna dict com:   {     "df":
DataFrame final,     "impacto": DataFrame Linhas/Colunas/Memória (antes/depois),     "conversoes":
cast_report filtrado (apenas mudanças reais) ou None,     "dups": duplicatas (amostra) ou None,
"dups_summary": resumo de duplicatas ou None   }

109. `def run_target_creation_and_summary(df, config, verbose):` — Orquestra a criação/validação do target usando ensure_target_from_config,
   Orquestra a criação/validação do target usando ensure_target_from_config, e retorna um pacote de
infos pronto para renderização.  Retorna um dict com:   - df: DataFrame (possivelmente atualizado)
- target_name: str   - class_map: dict com 'positive' e 'negative'   - tgt_report: DataFrame
(status, source, positive, negative)   - counts: dict {classe: contagem}   - total: int   -
pos_rate: float em [0,1]   - status: str   - source: str

110. `def save_artifact(obj, name, root):` — Função Save artifact
   Função utilitária `save_artifact` utilizada em fluxos internos do template de dados.

111. `def save_manifest(manifest, root):` — Função Save manifest
   Função utilitária `save_manifest` utilizada em fluxos internos do template de dados.

112. `def save_report_df(df, rel_path, root):` — Função Save report df
   Função utilitária `save_report_df` utilizada em fluxos internos do template de dados.

113. `def save_table(df, path, fmt, **kwargs):` — Função Save table
   Função utilitária `save_table` utilizada em fluxos internos do template de dados.

114. `def save_text(text, rel_path, root):` — Função Save text
   Função utilitária `save_text` utilizada em fluxos internos do template de dados.

115. `def scale_numeric(df, method, cols):` — Função Scale numeric
   Função utilitária `scale_numeric` utilizada em fluxos internos do template de dados.

116. `def scale_numeric_safe(df, exclude_cols, **kwargs):` — Função Scale numeric safe
   Função utilitária `scale_numeric_safe` utilizada em fluxos internos do template de dados.

117. `def scan_date_candidates(df, cfg):` — Scanner silencioso de possíveis colunas de data entre colunas object/strings.
   Scanner silencioso de possíveis colunas de data entre colunas object/strings. - Evita spam de
UserWarning do pandas (fallback dateutil). - Tenta formatos explícitos do cfg antes do parsing
genérico. - Opcionalmente amostra as séries para acelerar a detecção.  Retorna DataFrame com:
column, dtype, parse_ratio, sample_examples (ordenado por parse_ratio desc).

118. `def set_display(max_rows, max_cols):` — Função Set display
   Função utilitária `set_display` utilizada em fluxos internos do template de dados.

119. `def set_random_seed(seed):` — Função Set random seed
   Função utilitária `set_random_seed` utilizada em fluxos internos do template de dados.

120. `def show_block(title, subtitle, df_display, accent):` — Mostra um card com título/subtítulo seguido de um DataFrame estilizado.
   Mostra um card com título/subtítulo seguido de um DataFrame estilizado.

121. `def show_df_summary(df, label, accent):` — Mostra overview, dtypes e faltantes para um DataFrame 'principal' do pipeline.
   Mostra overview, dtypes e faltantes para um DataFrame 'principal' do pipeline.

122. `def show_df_summary_neat(df, label):` — Função Show df summary neat
   Função utilitária `show_df_summary_neat` utilizada em fluxos internos do template de dados.

123. `def show_source_overview(name, path, df):` — Mostra três cards: overview, dtypes e faltantes para uma fonte específica.
   Mostra três cards: overview, dtypes e faltantes para uma fonte específica.

124. `def show_source_overview_neat(name, path, df):` — Função Show source overview neat
   Função utilitária `show_source_overview_neat` utilizada em fluxos internos do template de dados.

125. `def signed_log1p_series(s):` — Aplica log1p assinado: sign(x) * log1p(|x|) — não gera NaN para x <= -1.
   Aplica log1p assinado: sign(x) * log1p(|x|) — não gera NaN para x <= -1. Mantém NaN somente onde s é
NaN.

126. `def simple_impute_with_flags(df, strategy, numeric_cols, categorical_cols):` — Função Simple impute with flags
   Função utilitária `simple_impute_with_flags` utilizada em fluxos internos do template de dados.

127. `def strip_whitespace(df, cols):` — Função Strip whitespace
   Função utilitária `strip_whitespace` utilizada em fluxos internos do template de dados.

128. `def suggest_categorical_candidates(df, max_unique_ratio, max_unique_count, include_numeric_small):` — Sugere colunas candidatas à padronização categórica com base em heurísticas:
   Sugere colunas candidatas à padronização categórica com base em heurísticas: - dtypes
texto/categoria/bool sempre entram - numéricas com poucos valores únicos entram se
include_numeric_small=True - calcula cardinalidade, % único e exemplos  Retorna DataFrame com:
column, dtype, n_unique, pct_unique, sample_values, suspected, reasons

129. `def suggest_source_path(directory, pattern, max_rows):` — Função Suggest source path
   Função utilitária `suggest_source_path` utilizada em fluxos internos do template de dados.

130. `def summarize_columns(df):` — Função Summarize columns
   Função utilitária `summarize_columns` utilizada em fluxos internos do template de dados.

131. `def summarize_missing(df):` — Retorna um resumo de valores nulos por coluna:
   Retorna um resumo de valores nulos por coluna:   column | missing_count | missing_pct | dtype

132. `def update_manifest(update, root):` — Função Update manifest
   Função utilitária `update_manifest` utilizada em fluxos internos do template de dados.

133. `def with_step(name, details, root):` — Função With step
   Função utilitária `with_step` utilizada em fluxos internos do template de dados.

---

## 🧩 Versão e Autoria

**Versão:** 1.3.0  
**Autor:** Fábio Emmanuel de Andrade Aguiar
**Descrição:** Toolkit unificado e resiliente para pipelines de dados, desenvolvido para o projeto *Telco Customer Churn Analysis* e o *Data Project Template* genérico.

