# 📔 `notebooks/` — Guia das funções definidas nos templates

Este README lista **todas as funções** definidas diretamente nos notebooks de template, com suas assinaturas e a primeira linha da docstring (quando existente). 
Use esta referência para localizar rapidamente onde cada utilitário é declarado e decidir se deve ser promovido para `utils/utils_data.py`.

> **Observação**: funções já migradas para `utils/` podem continuar referenciadas nos notebooks para fins didáticos.

Total de funções detectadas: **14**.

> ⚠️ O README anterior não mencionava **14** funções. Este arquivo inclui todas elas.

## 01_data_preparation_template.ipynb

| Função | Assinatura | Descrição | Notebook |
|---|---|---|---|
| `_find_up` | `_find_up(relative_path, start)` | — | `01_data_preparation_template.ipynb` |
| `_log` | `_log(msg)` | — | `01_data_preparation_template.ipynb` |
| `_save_df` | `_save_df(df_, path_)` | — | `01_data_preparation_template.ipynb` |


## 02_model_training_template.ipynb

| Função | Assinatura | Descrição | Notebook |
|---|---|---|---|
| `_dist` | `_dist(s)` | — | `02_model_training_template.ipynb` |
| `_find_up` | `_find_up(relative_path, start)` | — | `02_model_training_template.ipynb` |
| `_fmt_mb` | `_fmt_mb(n_bytes)` | — | `02_model_training_template.ipynb` |
| `_mb` | `_mb(nbytes)` | — | `02_model_training_template.ipynb` |
| `_pct` | `_pct(n, d)` | — | `02_model_training_template.ipynb` |
| `build_preprocess` | `build_preprocess(numeric_cols, categorical_cols, scale_numeric)` | — | `02_model_training_template.ipynb` |
| `build_preprocess` | `build_preprocess(numeric_cols, categorical_cols, scale_numeric)` | Cria um ColumnTransformer com: | `02_model_training_template.ipynb` |
| `collect_params_from_tab` | `collect_params_from_tab()` | — | `02_model_training_template.ipynb` |
| `compute_and_plot` | `compute_and_plot(pipe, name, X_test, y_test)` | — | `02_model_training_template.ipynb` |
| `on_train_clicked` | `on_train_clicked(_)` | — | `02_model_training_template.ipynb` |
| `train_and_eval` | `train_and_eval(models_selected, params_by_model)` | — | `02_model_training_template.ipynb` |

