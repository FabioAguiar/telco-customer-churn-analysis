# 📄 Manifesto de Execução (`manifest.json`)

O **manifesto** registra um _snapshot_ técnico da execução do N1 (e, opcionalmente, de outras etapas), para auditoria e reprodutibilidade.

> **Local padrão:** `reports/artifacts/export/manifest.json`

---

## ✨ Objetivo

- Documentar **quando** e **como** o pipeline rodou.
- Listar **arquivos exportados** (interim/processed/meta).
- Registrar **parâmetros efetivos** usados na execução (trecho do `config`).
- Ajudar na **auditoria** e **debug** (histórico de passos e relatórios).

---

## 🧩 Estrutura (campos principais)

| Campo | Tipo | Descrição |
|---|---|---|
| `created_at` | string (ISO) | Timestamp da execução. |
| `random_seed` | int | Seed global aplicada na sessão. |
| `config` | objeto | _Snapshot_ dos parâmetros relevantes do `defaults.json` + `local.json`. |
| `memory_mb` | float | Uso aproximado de memória do `DataFrame` final. |
| `shape` | [int, int] | Linhas e colunas do `DataFrame` final do N1. |
| `outlier_flags` | string[] | Colunas `_is_outlier` criadas. |
| `imputed_flags` | string[] | Colunas `was_imputed_*` criadas. |
| `exported` | objeto | Caminhos exportados (interim, processed, meta). |
| `run_steps` | objeto[] | Linha do tempo de passos executados, com _status_ e erro (se houver). |
| `reports` | string[] | Caminhos de relatórios gerados. |

> Observação: o **`config`** incluído é propositalmente **resumido**. Se necessário, amplie/filtre no código para focar só no que importa.

---

## 🧪 Exemplo mínimo (ilustrativo)

```json
{
  "created_at": "2025-11-03T06:46:53",
  "random_seed": 42,
  "config": {
    "detect_outliers": true,
    "outlier_method": "iqr",
    "encode_categoricals": true,
    "encoding_type": "onehot",
    "scale_numeric": false,
    "date_features": true,
    "text_features": true,
    "target": { "name": "Churn" }
  },
  "memory_mb": 12.34,
  "shape": [7043, 28],
  "outlier_flags": ["MonthlyCharges_is_outlier"],
  "imputed_flags": ["was_imputed_TotalCharges"],
  "exported": {
    "interim": "data/interim/interim.parquet",
    "processed": "data/processed/processed.parquet",
    "meta_file": "artifacts/metadata/dataset_meta.json"
  },
  "run_steps": [
    {"name": "n1_quality_typing:start", "ts": "2025-11-03T06:45:10"},
    {"name": "n1_quality_typing:end",   "ts": "2025-11-03T06:45:12"}
  ],
  "reports": [
    "reports/cast_report.csv",
    "reports/overview_after_quality.json"
  ]
}
```

---

## 🔧 Como é gerado no N1

No final do N1, o manifesto é salvo por um _helper_ do notebook, usando o diretório resolvido por `ud.get_artifacts_dir("export")`:

```python
from datetime import datetime
import json
from pathlib import Path
import utils.utils_data as ud

ARTIFACTS_DIR = ud.get_artifacts_dir("export")
manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "random_seed": RANDOM_SEED,
    "config": {
        "detect_outliers": config.get("detect_outliers", True),
        "outlier_method": config.get("outlier_method", "iqr"),
        "encode_categoricals": config.get("encode_categoricals", True),
        "encoding_type": config.get("encoding_type", "onehot"),
        "scale_numeric": config.get("scale_numeric", False),
        "scaler": config.get("scaler", "standard"),
        "date_features": config.get("date_features", True),
        "text_features": config.get("text_features", True),
        "target": (config.get("target") or {}).get("name")
    },
    "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    "shape": list(df.shape),
    "outlier_flags": [c for c in df.columns if c.endswith("_is_outlier")],
    "imputed_flags": [c for c in df.columns if c.startswith("was_imputed_")],
    "exported": {
        "interim": str(OUTPUT_INTERIM) if config.get("export_interim", True) else None,
        "processed": str(OUTPUT_PROCESSED) if config.get("export_processed", True) else None,
        "meta_file": str(META_FILE)
    },
    "run_steps": [],
    "reports": []
}

(ARTIFACTS_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
```

---

## ✅ Boas práticas

- **Não versionar** `manifest.json` em produção (ele é derivado). Em projetos de estudo/portfólio, você **pode** comitar para transparência.  
- Se o manifesto ficar muito grande, **resuma** o bloco `config`.  
- Use `run_steps` para marcar pontos críticos do pipeline (ex.: _start_, _end_, _error_).  
- Sempre que mudar a estrutura, **atualize este README**.

---

## 📍 Onde encontrar

- Manifesto do N1: `reports/artifacts/export/manifest.json`  
- Metadados do dataset: `artifacts/metadata/dataset_meta.json`

> O manifesto complementa o `meta.json`: enquanto o `meta` descreve o **dataset**, o `manifest` descreve a **execução**.
