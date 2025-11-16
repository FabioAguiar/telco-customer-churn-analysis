# -*- coding: utf-8 -*-
"""utils_data.py — Utilitários centrais para projetos de dados (N1/N2/N3).

Versão "merge" que combina as funções antigas (compatibilidade retroativa)
com as melhorias da v1.1.3. Objetivos:
- Manter assinaturas antigas que seus notebooks já usam;
- Adicionar as funções novas e relatórios extras;
- Fornecer "wrappers" quando a assinatura/retorno mudou.

Principais compatibilidades:
- resolve_n1_paths aceita tanto (root) quanto (config, root);
- N1Paths tem aliases .raw_dir/.interim_dir/.processed_dir/.reports_dir/.artifacts_dir;
- load_table_simple aceita (path, fmt=None, **read_opts) e também (path, fmt, read_opts_dict);
- n1_quality_typing agora retorna (df, meta) para compatibilidade; a variante nova fica em n1_quality_typing_dict;
- normalize_categories aceita cfg= (modo avançado) e também parâmetros simples (case/trim/etc.).
- TableStore suporta __init__(initial=..., current=...) e mantém métodos get/use/list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import contextlib
import datetime as dt
import json
import logging
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
except Exception:  # pragma: no cover
    StandardScaler = None  # type: ignore
    MinMaxScaler = None  # type: ignore

__all__ = [
    # raiz/config/manifest
    "ensure_project_root", "load_config",
    "load_manifest", "save_manifest", "update_manifest",
    "record_step", "with_step",
    # I/O de artefatos e relatórios
    "save_artifact", "load_artifact",
    "save_report_df", "save_text",
    # paths
    "N1Paths", "resolve_n1_paths", "path_of",
    # I/O tabelas
    "list_directory_files", "infer_format_from_suffix", "load_csv", "load_table_simple", "save_table",
    "suggest_source_path",
    # limpeza e tipagem
    "strip_whitespace", "infer_numeric_like",
    "n1_quality_typing", "n1_quality_typing_dict",
    # missing/duplicatas/outliers
    "simple_impute_with_flags", "deduplicate_rows",
    "detect_outliers_iqr", "detect_outliers_zscore",
    # categóricas/encoding/scaling
    "normalize_categories", "encode_categories", "encode_categories_safe",
    "scale_numeric", "scale_numeric_safe",
    # datas
    "detect_date_candidates", "parse_dates_with_report", "expand_date_features", "build_calendar_from",
    # texto
    "extract_text_features",
    # target e pipeline compacto
    "build_target", "ensure_target_from_config",
    "apply_encoding_and_scaling",
    # util de catálogo
    "TableStore",
    # visões rápidas e merge
    "basic_overview", "missing_report", "merge_chain",
    # relatórios humanos
    "generate_human_report_md", "md_to_pdf",
    # conveniências
    "set_random_seed", "set_display",
    # versão
    "UTILS_DATA_VERSION", 
    "apply_outlier_flags"
, "parse_dates_with_report_cfg", "expand_date_features_plus"]

UTILS_DATA_VERSION = "1.2.2"

logger = logging.getLogger("utils_data")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Raiz do projeto, Config e Manifest
# -----------------------------------------------------------------------------
def _find_up(relative_path: str, start: Optional[Path] = None) -> Optional[Path]:
    start = start or Path.cwd()
    rel = Path(relative_path)
    for base in (start, *start.parents):
        cand = base / rel
        if cand.exists():
            return cand
    return None

def ensure_project_root() -> Path:
    cfg = _find_up("config/defaults.json")
    if cfg is None:
        raise FileNotFoundError("config/defaults.json não encontrado ao subir a árvore de diretórios.")
    root = cfg.parent.parent
    logger.info(f"PROJECT_ROOT: {root}")
    utils_dir = root / "utils"
    if utils_dir.exists() and str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
        logger.info(f"sys.path ok. utils: {utils_dir}")
    return root

def load_config(base_abs: Optional[Path] = None, local_abs: Optional[Path] = None) -> Dict[str, Any]:
    root = ensure_project_root()
    base = base_abs or (root / "config" / "defaults.json")
    local = local_abs or (root / "config" / "local.json")
    with base.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if local.exists():
        with local.open("r", encoding="utf-8") as f:
            local_cfg = json.load(f)
        cfg = _deep_merge(cfg, local_cfg)
    return cfg

def _deep_merge(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out

def _manifest_path(root: Optional[Path] = None) -> Path:
    root = root or ensure_project_root()
    return root / "reports" / "manifest.json"

def load_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    p = _manifest_path(root)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"runs": []}

def save_manifest(manifest: Mapping[str, Any], root: Optional[Path] = None) -> None:
    p = _manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def update_manifest(update: Mapping[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    m = load_manifest(root)
    m = _deep_merge(m, update)
    save_manifest(m, root)
    return m

def record_step(name: str, details: Optional[Mapping[str, Any]] = None, root: Optional[Path] = None) -> None:
    m = load_manifest(root)
    entry = {
        "step": name,
        "details": dict(details or {}),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    m.setdefault("runs", []).append(entry)
    save_manifest(m, root)
    logger.info(f"[manifest] step='{name}' registrado.")

@contextlib.contextmanager
def with_step(name: str, details: Optional[Mapping[str, Any]] = None, root: Optional[Path] = None):
    record_step(f"{name}:start", details, root)
    try:
        yield
        record_step(f"{name}:end", details, root)
    except Exception as e:
        record_step(f"{name}:error", {"error": str(e)}, root)
        raise

def save_artifact(obj: Any, name: str, root: Optional[Path] = None) -> Path:
    if joblib is None:
        raise RuntimeError("joblib não está disponível. Instale com `pip install joblib`.")
    root = root or ensure_project_root()
    path = root / "artifacts" / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)  # type: ignore
    record_step("save_artifact", {"name": name, "path": str(path)}, root)
    return path

def load_artifact(name: str, root: Optional[Path] = None) -> Any:
    if joblib is None:
        raise RuntimeError("joblib não está disponível. Instale com `pip install joblib`.")
    root = root or ensure_project_root()
    path = root / "artifacts" / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Artifact não encontrado: {path}")
    return joblib.load(path)  # type: ignore

def save_report_df(df: pd.DataFrame, rel_path: Union[str, Path], root: Optional[Path] = None) -> Path:
    root = root or ensure_project_root()
    path = root / "reports" / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    record_step("save_report_df", {"path": str(path), "rows": len(df)})
    logger.info(f"[report] salvo: {path} ({len(df)} linhas)")
    return path

# -----------------------------------------------------------------------------
# Diretório padrão de artefatos (relatórios, saídas intermediárias, etc.)
# -----------------------------------------------------------------------------
def get_artifacts_dir(subdir: str | None = None) -> Path:
    """
    Retorna o diretório de artefatos do projeto (`reports/artifacts`), garantindo sua existência.
    
    Parâmetros:
      - subdir (opcional): nome de subpasta dentro de artifacts (ex.: "outliers" ou "calendar")

    Exemplo:
      >>> path = get_artifacts_dir("calendar")
      >>> print(path)
      C:/Users/fabio/Projetos DEV/data projects/data-project-template/reports/artifacts/calendar
    """
    try:
        root = ensure_project_root()
    except Exception:
        root = Path.cwd()

    base = root / "reports" / "artifacts"
    ensure_dir(base)

    if subdir:
        base = base / subdir
        ensure_dir(base)

    try:
        logger.info(f"[path] artifacts_dir -> {base}")
    except Exception:
        pass

    return base

# -----------------------------------------------------------------------------
# Diretório padrão de artefatos (relatórios, saídas intermediárias, etc.)
# -----------------------------------------------------------------------------
def get_artifacts_dir(subdir: str | None = None) -> Path:
    """
    Retorna o diretório de artefatos do projeto (`reports/artifacts`), garantindo sua existência.

    Parâmetros:
      - subdir (opcional): nome de subpasta dentro de artifacts (ex.: "outliers" ou "calendar")

    Exemplo:
      >>> path = get_artifacts_dir("calendar")
      >>> print(path)
      C:/Users/fabio/Projetos DEV/data projects/data-project-template/reports/artifacts/calendar
    """
    # fallback local para ensure_dir caso ainda não esteja definido
    def _ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    try:
        root = ensure_project_root()
    except Exception:
        root = Path.cwd()

    base = root / "reports" / "artifacts"
    _ensure_dir(base)

    if subdir:
        base = base / subdir
        _ensure_dir(base)

    try:
        logger.info(f"[path] artifacts_dir -> {base}")
    except Exception:
        pass

    return base


def save_text(text: str, rel_path: Union[str, Path], root: Optional[Path] = None) -> Path:
    root = root or ensure_project_root()
    path = root / "reports" / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    record_step("save_text", {"path": str(path), "size": len(text)})
    logger.info(f"[report] texto salvo: {path} ({len(text)} chars)")
    return path

# -----------------------------------------------------------------------------
# Paths N1 (com aliases de compatibilidade)
# -----------------------------------------------------------------------------
@dataclass
class N1Paths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    reports: Path
    artifacts: Path

# aliases legados
setattr(N1Paths, "raw_dir", property(lambda self: self.data_raw))
setattr(N1Paths, "interim_dir", property(lambda self: self.data_interim))
setattr(N1Paths, "processed_dir", property(lambda self: self.data_processed))
setattr(N1Paths, "reports_dir", property(lambda self: self.reports))
setattr(N1Paths, "artifacts_dir", property(lambda self: self.artifacts))

def _resolve_n1_paths_core(root: Optional[Path] = None) -> N1Paths:
    root = root or ensure_project_root()
    return N1Paths(
        root=root,
        data_raw=root / "data" / "raw",
        data_interim=root / "data" / "interim",
        data_processed=root / "data" / "processed",
        reports=root / "reports",
        artifacts=root / "artifacts",
    )

def resolve_n1_paths(*args) -> N1Paths:
    """Compatível com duas formas:
    - resolve_n1_paths() ou resolve_n1_paths(root)
    - resolve_n1_paths(config, root)  # notebooks antigos
    """
    if len(args) == 0:
        return _resolve_n1_paths_core(None)
    if len(args) == 1:
        a0 = args[0]
        if isinstance(a0, (str, Path)):
            return _resolve_n1_paths_core(Path(a0))
        else:
            return _resolve_n1_paths_core(None)
    if len(args) >= 2:
        root = args[1]
        return _resolve_n1_paths_core(Path(root))
    return _resolve_n1_paths_core(None)

def path_of(*parts: str, root: Optional[Path] = None) -> Path:
    root = root or ensure_project_root()
    return root.joinpath(*parts)

# -----------------------------------------------------------------------------
# I/O e inspeção
# -----------------------------------------------------------------------------
def list_directory_files(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    rows = []
    for p in sorted(path.rglob("*")):
        if p.is_file():
            rows.append({
                "path": str(p.resolve()),
                "name": p.name,
                "suffix": p.suffix.lower(),
                "size_bytes": p.stat().st_size,
                "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            })
    return pd.DataFrame(rows)

def suggest_source_path(directory: Union[str, Path], pattern: str = "*.csv", max_rows: int = 50) -> pd.DataFrame:
    directory = Path(directory)
    rows = []
    for p in sorted(directory.glob(pattern)):
        if p.is_file():
            rows.append({
                "path": str(p.resolve()),
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            })
    df = pd.DataFrame(rows)
    if len(df) > max_rows:
        df = df.head(max_rows)
    logger.info(f"[suggest_source_path] {len(rows)} arquivo(s) encontrados; exibindo {len(df)}.")
    return df

def infer_format_from_suffix(path: Union[str, Path]) -> str:
    s = Path(path).suffix.lower()
    if s == ".csv":
        return "csv"
    if s in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(f"Não sei inferir formato a partir de '{s}'. Use csv/parquet.")

def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def load_table_simple(path: Union[str, Path], fmt: Optional[Union[str, dict]] = None, *args, **kwargs) -> pd.DataFrame:
    """Compatível com:
       - load_table_simple(path, fmt=None, **read_opts)
       - load_table_simple(path, fmt, read_opts_dict)
    """
    # caso antigo: terceiro arg é read_opts_dict posicional
    if args and isinstance(args[0], dict) and not kwargs:
        kwargs = args[0]
    # caso em que segundo arg veio como dict por engano
    if isinstance(fmt, dict):
        kwargs = {**fmt, **kwargs}
        fmt = None
    fmt = fmt or infer_format_from_suffix(path)
    if fmt == "csv":
        return pd.read_csv(path, **kwargs)
    if fmt == "parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Formato não suportado: {fmt}")

def save_table(df: pd.DataFrame, path: Union[str, Path], fmt: Optional[str] = None, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt or infer_format_from_suffix(path)
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8", **kwargs)
    elif fmt == "parquet":
        df.to_parquet(path, index=False, **kwargs)
    else:
        raise ValueError(f"Formato não suportado: {fmt}")
    logger.info(f"[save_table] {path} ({len(df)} linhas)")
    return path

# -----------------------------------------------------------------------------
# Visões rápidas e merge
# -----------------------------------------------------------------------------
def basic_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024**2)), 3),
    }

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss_cnt = df.isna().sum()
    miss_pct = (df.isna().mean() * 100).round(2)
    rep = (
        pd.DataFrame({
            "column": df.columns,
            "missing_count": miss_cnt.values,
            "missing_pct": miss_pct.values,
        })
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )
    return rep

def merge_chain(base: pd.DataFrame, tables: dict, steps: list) -> pd.DataFrame:
    df = base.copy()
    for step in steps:
        src_name = step.get("from")
        if src_name not in tables:
            raise KeyError(f"[merge_chain] Tabela '{src_name}' não encontrada. Disponíveis: {list(tables.keys())}")
        right = tables[src_name]
        how = step.get("how", "left")
        suffixes = tuple(step.get("suffixes", ("", "_r")))
        kwargs = {"how": how, "suffixes": suffixes}
        if "on" in step:
            kwargs["on"] = step["on"]
        if "left_on" in step or "right_on" in step:
            kwargs["left_on"] = step.get("left_on")
            kwargs["right_on"] = step.get("right_on")
        if "validate" in step:
            kwargs["validate"] = step["validate"]
        df = pd.merge(df, right, **kwargs)
        for c in step.get("drop_cols", []) or []:
            if c in df.columns:
                df = df.drop(columns=c)
    return df

# -----------------------------------------------------------------------------
# Limpeza e tipagem
# -----------------------------------------------------------------------------
def strip_whitespace(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = df.copy()
    cols = list(cols) if cols is not None else list(df.columns)
    for c in cols:
        if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df

def infer_numeric_like(df: pd.DataFrame, cols: Optional[Sequence[str]] = None,
                       decimal: str = ".", thousands: Optional[str] = None,
                       report_path: Optional[Union[str, Path]] = "cast_report.csv",
                       root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    cols = list(cols) if cols is not None else [c for c in df.columns if df[c].dtype == "object"]
    rep_rows = []
    for c in cols:
        s = df[c].astype(str)
        before_nulls = s.isna().sum() if not isinstance(df[c].dtype, np.dtype) else df[c].isna().sum()
        s2 = s
        if thousands:
            s2 = s2.str.replace(thousands, "", regex=False)
        if decimal != ".":
            s2 = s2.str.replace(decimal, ".", regex=False)
        new = pd.to_numeric(s2, errors="coerce")
        introduced_nans = new.isna().sum() - before_nulls
        df[c] = new if not new.isna().all() else df[c]
        rep_rows.append({
            "column": c,
            "converted_non_null": int(new.notna().sum()),
            "introduced_nans": int(introduced_nans),
            "dtype_after": str(df[c].dtype),
        })
    report = pd.DataFrame(rep_rows).sort_values("converted_non_null", ascending=False)
    if report_path:
        save_report_df(report, report_path, root=root)
    return df, report

def n1_quality_typing_dict(df: pd.DataFrame, config: Mapping[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    """Nova API: retorna dict com 'df', 'steps' e 'cast_report'."""
    out: Dict[str, Any] = {"steps": []}
    typing_cfg = config.get("typing", {})
    with with_step("n1_quality_typing", {"config": typing_cfg}, root):
        if typing_cfg.get("strip_whitespace", True):
            df = strip_whitespace(df)
            out["steps"].append("strip_whitespace")
        if typing_cfg.get("infer_numeric_like", True):
            df, rep = infer_numeric_like(
                df,
                decimal=typing_cfg.get("decimal", "."),
                thousands=typing_cfg.get("thousands"),
                report_path=typing_cfg.get("cast_report_path", "cast_report.csv"),
                root=root,
            )
            out["cast_report"] = rep
            out["steps"].append("infer_numeric_like")
    out["df"] = df
    return out

def n1_quality_typing(df: pd.DataFrame, config: Mapping[str, Any], root: Optional[Path] = None):
    """Compat: retorna (df, meta_dict)."""
    out = n1_quality_typing_dict(df, config, root=root)
    return out["df"], out

# -----------------------------------------------------------------------------
# Missing, duplicatas e outliers
# -----------------------------------------------------------------------------
def simple_impute_with_flags(df: pd.DataFrame, strategy: str = "median",
                             numeric_cols: Optional[Sequence[str]] = None,
                             categorical_cols: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    meta: Dict[str, Any] = {"strategy": strategy, "imputed": []}
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

    for c in numeric_cols:
        flag = f"{c}_was_missing"
        wasna = df[c].isna()
        df[flag] = wasna.astype(int)
        fill = df[c].median() if strategy == "median" else (df[c].mean() if strategy == "mean" else 0)
        df[c] = df[c].fillna(fill)
        meta["imputed"].append({"col": c, "fill": fill})

    for c in categorical_cols:
        flag = f"{c}_was_missing"
        wasna = df[c].isna()
        df[flag] = wasna.astype(int)
        df[c] = df[c].fillna("__MISSING__")
        meta["imputed"].append({"col": c, "fill": "__MISSING__"})

    return df, meta

# --- Helpers de robustez para N1 ------------------------------------------------
def coerce_df(obj) -> pd.DataFrame:
    """Garante um DataFrame. Se vier (df, meta), retorna o primeiro elemento."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, tuple) and len(obj) > 0 and isinstance(obj[0], pd.DataFrame):
        return obj[0]
    raise TypeError(f"[coerce_df] Esperado DataFrame/tupla(df,*), recebi: {type(obj)}")

def handle_missing_step(df: pd.DataFrame,
                        config: Mapping[str, Any],
                        save_reports: bool = True,
                        prefer: str = "auto") -> Dict[str, Any]:
    """
    Executa a etapa de 'faltantes' ponta-a-ponta:
      - Gera relatório 'antes' (reports/missing/before.csv)
      - Aplica estratégia (simple | knn | iterative). 'auto' lê do config com fallbacks
      - Gera relatório 'depois' (reports/missing/after.csv)

    Compatibilidade: mantém assinatura, contratos e caminhos do método original.
    Melhorias: não cria flags encadeadas (evita *_was_missing_was_missing...) e
               só flaggeia colunas que tinham NaN de fato.
    Retorna dict: {'df','before','after','strategy','imputed_cols'}
    """
    df = coerce_df(df)

    # --- helpers internos ----------------------------------------------------
    def _is_flag(col: str) -> bool:
        return isinstance(col, str) and col.endswith("_was_missing")

    # Lê config nos dois formatos aceitos
    missing_cfg = dict(config.get("missing", {}))
    handle = bool(config.get("handle_missing", missing_cfg.get("enabled", True)))
    strategy = (config.get("missing_strategy",
                           missing_cfg.get("strategy", "simple")) or "simple").lower()

    # Parâmetros extras (com defaults)
    knn_k  = int(missing_cfg.get("knn_k", 5))
    it_max = int(missing_cfg.get("iterative_max_iter", 10))
    it_seed = int(missing_cfg.get("iterative_random_state", 42))

    # Onde salvar relatórios (preserva seus caminhos relativos)
    before_rel = "missing/before.csv"
    after_rel  = "missing/after.csv"

    # Relatório antes
    rep_before = missing_report(df)
    if save_reports:
        save_report_df(rep_before, before_rel)

    out: Dict[str, Any] = {
        "before": rep_before,
        "strategy": strategy,
        "imputed_cols": []
    }

    if not handle:
        out["df"] = df
        out["after"] = rep_before.copy()
        return out

    # --------------------- Estratégias (com salvaguardas) --------------------
    def _simple(df_in: pd.DataFrame):
        """
        Usa simple_impute_with_flags apenas nas colunas não-flag e reintegra no df original,
        evitando criar flags encadeadas.
        """
        non_flag_cols = [c for c in df_in.columns if not _is_flag(c)]
        if not non_flag_cols:
            return df_in, [], "simple"

        # roda imputação/flags apenas nas colunas de interesse
        sub = df_in[non_flag_cols].copy()
        df_imp, meta = simple_impute_with_flags(sub)
        cols_imp = [m["col"] for m in meta.get("imputed", [])] if isinstance(meta, dict) else []

        df_out = df_in.copy()
        # substitui valores imputados nas colunas originais
        for c in non_flag_cols:
            if c in df_imp.columns:
                df_out[c] = df_imp[c]

        # adiciona flags criadas (apenas de primeira camada)
        for c in df_imp.columns:
            if _is_flag(c) and c not in df_out.columns:
                df_out[c] = df_imp[c]

        return df_out, cols_imp, "simple"

    def _knn(df_in: pd.DataFrame):
        """
        Imputa numéricos com KNNImputer; cria flags apenas para colunas numéricas
        que realmente tinham NaN (e evita flags encadeadas/duplicadas).
        """
        try:
            from sklearn.impute import KNNImputer  # type: ignore
        except Exception:
            return _simple(df_in)

        num_cols = [c for c in df_in.columns
                    if (not _is_flag(c)) and pd.api.types.is_numeric_dtype(df_in[c])]
        if not num_cols:
            return _simple(df_in)

        missing_num_cols = [c for c in num_cols if df_in[c].isna().any()]
        if not missing_num_cols:
            # nada a imputar -> mantém como está
            return df_in.copy(), [], "knn"

        df_out = df_in.copy()
        # flags apenas para colunas numéricas com NaN (e não duplicar se já existe)
        for c in missing_num_cols:
            flag = f"{c}_was_missing"
            if flag not in df_out.columns:
                df_out[flag] = df_out[c].isna().astype(int)

        imputer = KNNImputer(n_neighbors=knn_k)
        df_out[num_cols] = imputer.fit_transform(df_out[num_cols])

        return df_out, missing_num_cols, "knn"

    def _iterative(df_in: pd.DataFrame):
        """
        Imputa numéricos com IterativeImputer; cria flags apenas para colunas numéricas
        que tinham NaN (evita encadeamento e duplicatas).
        """
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer  # type: ignore
        except Exception:
            return _simple(df_in)

        num_cols = [c for c in df_in.columns
                    if (not _is_flag(c)) and pd.api.types.is_numeric_dtype(df_in[c])]
        if not num_cols:
            return _simple(df_in)

        missing_num_cols = [c for c in num_cols if df_in[c].isna().any()]
        if not missing_num_cols:
            return df_in.copy(), [], "iterative"

        df_out = df_in.copy()
        for c in missing_num_cols:
            flag = f"{c}_was_missing"
            if flag not in df_out.columns:
                df_out[flag] = df_out[c].isna().astype(int)

        imp = IterativeImputer(max_iter=it_max, random_state=it_seed, sample_posterior=False)
        df_out[num_cols] = imp.fit_transform(df_out[num_cols])

        return df_out, missing_num_cols, "iterative"

    # Seleção da estratégia com fallback (preserva sua lógica)
    chosen = strategy if prefer == "auto" else prefer
    try_chain = {
        "simple":    (_simple,    ["simple"]),
        "knn":       (_knn,       ["knn", "simple"]),
        "iterative": (_iterative, ["iterative", "simple"]),
        "mice":      (_iterative, ["iterative", "simple"]),
        "auto":      (None,       [strategy, "simple"]),
    }
    order = try_chain.get("auto" if prefer == "auto" else chosen, (None, ["simple"]))[1]

    df_work = df
    used = "simple"
    imputed_cols: List[str] = []

    for opt in order:
        if opt == "simple":
            df_work, imputed_cols, used = _simple(df_work)
            break
        elif opt == "knn":
            df_work, imputed_cols, used = _knn(df_work)
            if used == "knn":
                break
        elif opt in {"iterative", "mice"}:
            df_work, imputed_cols, used = _iterative(df_work)
            if used == "iterative":
                break

    out["df"] = df_work
    out["strategy"] = used
    out["imputed_cols"] = imputed_cols

    # Relatório depois
    rep_after = missing_report(df_work)
    out["after"] = rep_after
    if save_reports:
        save_report_df(rep_after, after_rel)

    return out


def deduplicate_rows(df: pd.DataFrame, subset: Optional[Sequence[str]] = None, keep: str = "first") -> Tuple[pd.DataFrame, pd.DataFrame]:
    before = len(df)
    dup_mask = df.duplicated(subset=subset, keep=keep)
    log = df.loc[dup_mask].copy()
    out = df.loc[~dup_mask].copy()
    logger.info(f"[deduplicate_rows] removidas {len(log)} duplicatas (de {before}).")
    return out, log

def detect_outliers_iqr(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, k: float = 1.5) -> pd.DataFrame:
    cols = list(cols) if cols is not None else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    mask = pd.DataFrame(False, index=df.index, columns=cols)
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask[c] = (df[c] < lo) | (df[c] > hi)
    return mask

def detect_outliers_zscore(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, z: float = 3.0) -> pd.DataFrame:
    cols = list(cols) if cols is not None else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    mask = pd.DataFrame(False, index=df.index, columns=cols)
    for c in cols:
        s = df[c]
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma == 0:
            mask[c] = False
        else:
            zscores = (s - mu) / sigma
            mask[c] = zscores.abs() > z
    return mask

# -----------------------------------------------------------------------------
# Categóricas, encoding e scaling
# -----------------------------------------------------------------------------
def _normalize_str(x: str) -> str:
    x = unicodedata.normalize("NFKD", x)
    x = "".join([ch for ch in x if not unicodedata.combining(ch)])
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x

def normalize_categories(df: pd.DataFrame,
                         cols: Optional[Sequence[str]] = None,
                         case: str = "lower",
                         trim: bool = True,
                         strip_accents: bool = True,
                         cfg: Optional[Mapping[str, Any]] = None,
                         report_path: Optional[Union[str, Path]] = None,
                         root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Modo compat + avançado.
       - Sem cfg: usa (case/trim/strip_accents) simples.
       - Com cfg: espera chaves como exclude, collapse_ws, null_values, global_map, per_column_map, cast_to_category.
       Retorna (df, report) e opcionalmente salva o CSV do report se report_path for informado.
    """
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]

    # defaults
    collapse_ws = True
    null_values = set()
    global_map = {}
    per_column_map = {}
    cast_to_category = False
    exclude = set()

    if cfg:
        case = cfg.get("case", case)
        trim = cfg.get("trim", trim)
        strip_accents = cfg.get("strip_accents", strip_accents)
        collapse_ws = cfg.get("collapse_ws", True)
        null_values = set(map(str, cfg.get("null_values", [])))
        global_map = {str(k): v for k, v in (cfg.get("global_map") or {}).items()}
        per_column_map = {k: {str(kk): vv for kk, vv in v.items()} for k, v in (cfg.get("per_column_map") or {}).items()}
        cast_to_category = bool(cfg.get("cast_to_category", False))
        exclude = set(cfg.get("exclude", []))
        cols = [c for c in cols if c not in exclude]

    changes_rows: List[Dict[str, Any]] = []

    for c in cols:
        s_orig = df[c].astype(str)
        s = s_orig
        if trim:
            s = s.str.strip()
        if collapse_ws:
            s = s.str.replace(r"\s+", " ", regex=True)
        if strip_accents:
            s = s.map(lambda v: _normalize_str(v))
        if case == "lower":
            s = s.str.lower()
        elif case == "upper":
            s = s.str.upper()
        elif case == "title":
            s = s.str.title()
        if global_map:
            s = s.map(lambda v: global_map.get(v, v))
        if c in per_column_map:
            cmap = per_column_map[c]
            s = s.map(lambda v: cmap.get(v, v))
        if null_values:
            s = s.map(lambda v: (np.nan if str(v) in null_values else v))

        changed = (s_orig.astype(str) != s.astype(str)).sum()
        changes_rows.append({"column": c, "changed": int(changed)})
        df[c] = s
        if cast_to_category:
            df[c] = df[c].astype("category")

    report = pd.DataFrame(changes_rows).sort_values("changed", ascending=False).reset_index(drop=True)
    if report_path is not None:
        if isinstance(report_path, Path):
            out_path = report_path
        else:
            root = root or ensure_project_root()
            out_path = (root / "reports" / str(report_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False, encoding="utf-8")
        record_step("save_report_df", {"path": str(out_path), "rows": int(len(report))}, root)
    return df, report

def _top_k_categories(s: pd.Series, k: Optional[int]) -> Tuple[pd.Series, List[str]]:
    if not k or k <= 0:
        return s, []
    top = list(s.value_counts(dropna=False).head(k).index)
    s2 = s.where(s.isin(top), other="__OTHER__")
    return s2, top

def encode_categories(df: pd.DataFrame,
                      cols: Optional[Sequence[str]] = None,
                      drop_first: bool = False,
                      high_cardinality_threshold: int = 20,
                      top_k: Optional[int] = None,
                      other_label: str = "__OTHER__") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    meta: Dict[str, Any] = {"encoded": [], "high_cardinality": []}
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])]
    for c in cols:
        s = df[c].astype(str)
        card = s.nunique(dropna=True)
        use_top_k = (top_k is not None) and (card > high_cardinality_threshold)
        if use_top_k:
            s2, kept = _top_k_categories(s, top_k)
            meta["high_cardinality"].append({"col": c, "cardinality": int(card), "kept_top_k": int(len(kept))})
            s = s2.replace({"__OTHER__": other_label})
        else:
            kept = list(s.dropna().unique())
        dummies = pd.get_dummies(s, prefix=c, drop_first=drop_first, dummy_na=False)
        df = df.drop(columns=[c]).join(dummies)
        meta["encoded"].append({"col": c, "created_cols": list(dummies.columns), "kept": kept})
    return df, meta

def encode_categories_safe(df: pd.DataFrame,
                           exclude_cols: Optional[Sequence[str]] = None,
                           **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    exclude = set(exclude_cols or [])
    cols = [c for c in df.columns if (df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])) and c not in exclude]
    return encode_categories(df, cols=cols, **kwargs)

def scale_numeric(df: pd.DataFrame, method: str = "standard",
                  cols: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if StandardScaler is None or MinMaxScaler is None:
        raise RuntimeError("scikit-learn não está disponível. Instale com `pip install scikit-learn`.")
    df = df.copy()
    meta: Dict[str, Any] = {"method": method, "scaled": []}
    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    scaler = StandardScaler() if method == "standard" else MinMaxScaler() if method == "minmax" else None
    if scaler is None:
        raise ValueError("method deve ser 'standard' ou 'minmax'.")

    df[cols] = scaler.fit_transform(df[cols])
    meta["scaled"] = list(cols)
    meta["scaler"] = scaler
    return df, meta

def scale_numeric_safe(df: pd.DataFrame, exclude_cols: Optional[Sequence[str]] = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    exclude = set(exclude_cols or [])
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    return scale_numeric(df, cols=cols, **kwargs)

# -----------------------------------------------------------------------------
# Datas
# -----------------------------------------------------------------------------
_DEFAULT_DATE_REGEX = [r"(?:^|_)(date|data|dt)(?:$|_)", r"(?:_at$)", r"(?:_date$)"]


def detect_date_candidates(df: pd.DataFrame, regex_list: Optional[Sequence[str]] = None) -> List[str]:
    regex_list = list(regex_list) if regex_list else _DEFAULT_DATE_REGEX
    out: List[str] = []
    for c in df.columns:
        name = c.lower()
        if any(re.search(rx, name) for rx in regex_list):
            out.append(c)
    return out

def parse_dates_with_report(df: pd.DataFrame,
                            cols: Optional[Sequence[str]] = None,
                            dayfirst: bool = False,
                            utc: bool = False,
                            errors: str = "coerce",
                            min_ratio: float = 0.6,
                            report_path: Optional[Union[str, Path]] = "date_parse_report.csv",
                            max_fail_samples: int = 10,
                            root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if cols is None:
        cols = detect_date_candidates(df)

    rep_rows = []
    for c in cols:
        raw = df[c]
        parsed = pd.to_datetime(raw, dayfirst=dayfirst, utc=utc, errors=errors)
        success = parsed.notna().mean() if len(parsed) else 0.0
        fail_samples = raw[parsed.isna()].astype(str).head(max_fail_samples).tolist()
        rep_rows.append({
            "column": c,
            "success_ratio": float(round(success, 4)),
            "applied": bool(success >= min_ratio),
            "fail_samples": "; ".join(fail_samples),
        })
        if success >= min_ratio:
            df[c] = parsed

    report = pd.DataFrame(rep_rows).sort_values("success_ratio", ascending=False)
    if report_path:
        save_report_df(report, report_path, root=root)
    return df, report

def expand_date_features(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        s = df[c].dt
        df[f"{c}_year"] = s.year
        df[f"{c}_month"] = s.month
        df[f"{c}_day"] = s.day
        df[f"{c}_dow"] = s.dayofweek
        df[f"{c}_week"] = s.isocalendar().week.astype(int)
        df[f"{c}_quarter"] = s.quarter
    return df

def build_calendar_from(df: pd.DataFrame, col: str, freq: str = "D") -> pd.DataFrame:
    s = pd.to_datetime(df[col], errors="coerce")
    start, end = s.min(), s.max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"Coluna {col} não possui datas válidas.")
    idx = pd.date_range(start=start, end=end, freq=freq)
    cal = pd.DataFrame({"date": idx})
    cal["year"] = cal["date"].dt.year
    cal["month"] = cal["date"].dt.month
    cal["day"] = cal["date"].dt.day
    cal["dow"] = cal["date"].dt.dayofweek
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["quarter"] = cal["date"].dt.quarter
    return cal

# --- Calendário · etapa orquestrada + render --------------------------------
def _ensure_datetime_with_ratio(
    s: pd.Series,
    *,
    dayfirst: bool = False,
    as_utc: bool = False,
    min_ratio: float = 0.80
) -> tuple[pd.Series, float]:
    """
    Tenta converter uma série para datetime de forma tolerante.
    Retorna (serie_convertida, parse_ratio).
    Não levanta warning; cai em NaT quando não converte.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, utc=as_utc)
    ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
    return parsed, ratio


def run_calendar_step(
    df: pd.DataFrame,
    *,
    date_col: str | None = None,
    freq: str = "D",
    output: str | None = None,
    config: Mapping[str, Any] | None = None,
    paths: Any | None = None,
    catalog: Any | None = None,  # ex.: objeto T (catálogo), se existir
) -> dict[str, Any]:
    """
    Orquestra a criação da dimensão calendário:
      - Resolve parâmetros a partir do `config["calendar"]` (se presente)
      - Descobre coluna de data automaticamente quando não for informada
      - Converte para datetime com verificação de 'parse_ratio'
      - Constrói, salva e (opcional) registra no catálogo
      - Retorna dict com artefatos e mensagens

    Retorno:
      {
        "status": "ok" | "skipped" | "error",
        "reason": <mensagem se skipped/error>,
        "date_col": <coluna usada ou None>,
        "freq": <freq>,
        "output": <caminho final>,
        "dim_date": <DataFrame ou None>,
        "period": (start, end) ou None
      }
    """
    log = logger if "logger" in globals() else None
    cal_cfg = dict((config or {}).get("calendar", {}))

    # Prioridade: argumentos diretos > config > defaults
    date_col = date_col or cal_cfg.get("date_col")
    freq = freq or cal_cfg.get("freq", "D")
    output = output or cal_cfg.get("output")

    # Descoberta automática da coluna caso não informada
    if not date_col:
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        date_col = dt_cols[0] if dt_cols else None

    if not date_col:
        msg = "Nenhum campo de data disponível. Defina 'date_col' ou trate datas antes."
        if log: log.info(f"[calendar] {msg}")
        return {"status": "skipped", "reason": msg, "date_col": None, "freq": freq,
                "output": output, "dim_date": None, "period": None}

    s = df[date_col]
    if not pd.api.types.is_datetime64_any_dtype(s):
        # Puxa preferências globais de datas, se existirem
        dates_cfg = dict((config or {}).get("dates", {}))
        dayfirst = bool(dates_cfg.get("dayfirst", False))
        as_utc   = bool(dates_cfg.get("utc", False))
        parsed, ratio = _ensure_datetime_with_ratio(
            s, dayfirst=dayfirst, as_utc=as_utc, min_ratio=dates_cfg.get("min_ratio", 0.80)
        )
        if ratio >= float(dates_cfg.get("min_ratio", 0.80)):
            df[date_col] = parsed
        else:
            msg = (f"Coluna '{date_col}' não está em datetime (parse_ratio={ratio:.2f}). "
                   "Ajuste a etapa de Tratamento de Datas.")
            if log: log.warning(f"[calendar] {msg}")
            return {"status": "error", "reason": msg, "date_col": date_col, "freq": freq,
                    "output": output, "dim_date": None, "period": None}

    # Resolve saída padrão se não informada
    if not output:
        try:
            output = str(paths.processed_dir / "dim_date.parquet")  # convenção do template
        except Exception:
            output = "data/processed/dim_date.parquet"

    # Constrói calendário
    dim_date = build_calendar_from(df, date_col, freq=freq)

    # Persiste respeitando extensão
    save_table(dim_date, output)
    if log: log.info(f"[calendar] dim_date salvo em: {output}")

    # (Opcional) registrar no catálogo
    try:
        if catalog is not None:
            catalog.add("dim_date", dim_date)
            if log: log.info("[calendar] 'dim_date' registrado no catálogo.")
    except Exception:
        pass

    start_date = dim_date["date"].min()
    end_date   = dim_date["date"].max()

    return {
        "status": "ok",
        "reason": "",
        "date_col": date_col,
        "freq": freq,
        "output": output,
        "dim_date": dim_date,
        "period": (start_date, end_date),
    }


def render_calendar_step(info: Mapping[str, Any]) -> None:
    """Renderiza um resumo amigável da etapa calendário."""
    from IPython.display import display, HTML
    import pandas as pd

    def _card(title, subtitle=""):
        return HTML(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid #10b981;
                    border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
          <div style="font-weight:700;font-size:16px">{title}</div>
          <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
        </div>
        """)

    status = info.get("status")
    display(_card("📆 Dimensão Calendário", f"Status: {status}"))

    if status == "ok":
        start, end = info.get("period") or (None, None)
        date_col = info.get("date_col")
        freq = info.get("freq")
        output = info.get("output")
        display(HTML(
            f"<div style='font-size:12px;color:#374151'>"
            f"<b>Coluna de origem:</b> {date_col} &nbsp;|&nbsp; "
            f"<b>Freq:</b> {freq} &nbsp;|&nbsp; "
            f"<b>Saída:</b> <code>{output}</code><br>"
            f"<b>Período:</b> {getattr(start, 'date', lambda: start)()} → "
            f"{getattr(end, 'date', lambda: end)()} &nbsp;|&nbsp; "
            f"<b>Linhas:</b> {len(info.get('dim_date'))}"
            f"</div>"
        ))
        # Prévia
        display(info.get("dim_date").head(12))
    else:
        reason = info.get("reason", "")
        display(HTML(f"<div style='font-size:12px;color:#6b7280'>{reason}</div>"))


# -----------------------------------------------------------------------------
# Texto
# -----------------------------------------------------------------------------
def extract_text_features(df: pd.DataFrame, cols: Optional[Sequence[str]] = None,
                          report_path: Optional[Union[str, Path]] = "text_features/summary.csv",
                          root: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    rep = []
    for c in cols:
        s = df[c].astype(str)
        df[f"{c}_len"] = s.str.len()
        df[f"{c}_word_count"] = s.str.split().map(len)
        rep.append({"column": c, "len_col": f"{c}_len", "word_count_col": f"{c}_word_count"})
    report = pd.DataFrame(rep)
    if report_path:
        save_report_df(report, report_path, root=root)
    return df, report

def render_text_features_summary(
    summary_df: pd.DataFrame,
    *,
    title: str = "📝 Tratamento de Texto",
    subtitle: str = "Comprimento, contagem de palavras e hits de keywords",
    keywords: list[str] | None = None,
    top: int = 20
) -> None:
    """
    Exibe um painel compacto e legível para o resumo de features de texto.
    Não altera dados; apenas organiza a visualização em três blocos:
      1) Card + métricas gerais
      2) Colunas com maior avg_len / avg_words
      3) Totais de hits por keyword (se houver)
    """
    from IPython.display import display, HTML
    import pandas as pd
    import numpy as np

    if summary_df is None or summary_df.empty:
        display(HTML("<div style='color:#6b7280'>— Nenhum resumo de texto disponível —</div>"))
        return

    def _card(title, subtitle=""):
        return HTML(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid #6366f1;
                    border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
          <div style="font-weight:700;font-size:16px">{title}</div>
          <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
        </div>
        """)

    # 1) Cabeçalho + métricas gerais
    display(_card(title, subtitle))
    n_cols = int(summary_df.shape[0])
    total_non_null = int(summary_df.get("non_null", pd.Series(dtype=int)).sum() or 0)
    met = pd.DataFrame([
        {"Métrica": "Colunas textuais analisadas", "Valor": n_cols},
        {"Métrica": "Total de células não nulas (texto)", "Valor": total_non_null},
    ])
    display(met)

    # 2) Prioridades: maiores comprimentos médios e contagem média de palavras
    cols_base = [c for c in ["column", "non_null", "avg_len", "avg_words"] if c in summary_df.columns]
    if cols_base:
        view_len = (summary_df[cols_base]
                    .sort_values(["avg_len","avg_words"], ascending=False)
                    .head(top)
                    .reset_index(drop=True))
        display(_card("🔎 Top colunas por comprimento médio", "Ordenado por avg_len e avg_words"))
        display(view_len)

    # 3) Totais por keyword (se existir padrão kw_*_count)
    kw_cols = [c for c in summary_df.columns if c.startswith("kw_") and c.endswith("_count")]
    if kw_cols:
        totals = summary_df[kw_cols].sum(axis=0).sort_values(ascending=False)
        df_totals = (totals.rename("ocorrencias")
                           .to_frame()
                           .reset_index()
                           .rename(columns={"index": "keyword_metric"}))
        # Se o caller passou keywords, reorganiza na ordem fornecida
        if keywords:
            desired = [f"kw_{k}_count" for k in keywords]
            df_totals["order"] = df_totals["keyword_metric"].apply(lambda x: desired.index(x) if x in desired else 10_000)
            df_totals = df_totals.sort_values(["order","ocorrencias"], ascending=[True, False]).drop(columns=["order"])
        display(_card("🏷️ Ocorrências por keyword", "Soma das aparições por coluna analisada"))
        display(df_totals)

    # 4) Dica leve
    display(HTML("<div style='color:#6b7280;font-size:12px;margin-top:4px'>"
                 "Dica: ajuste TEXT_CFG['keywords'] para acompanhar termos de negócio relevantes.</div>"))

def extract_text_features_fast(
    df: pd.DataFrame,
    *,
    lower: bool = True,
    strip_collapse_ws: bool = True,
    keywords: list[str] | None = None,
    blacklist: list[str] | None = None,
    export_summary: bool = True,
    summary_dir: str | os.PathLike | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Versão otimizada: acumula novas colunas em um dict e concatena de uma vez,
    evitando alta fragmentação de DataFrame.
    """
    import re, os
    from pathlib import Path

    keywords = keywords or []
    blacklist = set(blacklist or [])
    df_out = df.copy()
    new_cols = {}  # <- acumula aqui

    text_cols = [c for c in df_out.columns
                 if c not in blacklist and (df_out[c].dtype == "object" or pd.api.types.is_string_dtype(df_out[c]))]

    rows = []
    for c in text_cols:
        s = df_out[c].astype("string")
        if lower: s = s.str.lower()
        if strip_collapse_ws:
            s = s.str.replace(r"\s+", " ", regex=True).str.strip()

        # métricas
        non_null = int(s.notna().sum())
        avg_len = float(s.dropna().str.len().mean() or 0.0)
        avg_words = float(s.dropna().str.split().str.len().mean() or 0.0)

        # flags por keyword
        kw_counts = {}
        for kw in keywords:
            kw_flag = f"kw_{kw}_flag"
            kw_count = f"kw_{kw}_count"
            patt = rf"\b{re.escape(kw)}\b"
            flag_series = s.str.contains(patt, regex=True, na=False)
            new_cols[kw_flag] = flag_series
            kw_counts[kw_count] = int(flag_series.sum())

        # linha do summary
        row = {"column": c, "non_null": non_null, "avg_len": round(avg_len, 2), "avg_words": round(avg_words, 2)}
        row.update(kw_counts)
        rows.append(row)

    # concatena todas as novas colunas de uma vez
    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    summary = pd.DataFrame(rows).sort_values(["avg_len","avg_words"], ascending=False).reset_index(drop=True)

    if export_summary and summary_dir:
        summary_dir = Path(summary_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        outp = summary_dir / "text_features_summary.csv"
        summary.to_csv(outp, index=False, encoding="utf-8")
        try:
            logger.info(f"[text] resumo salvo em: {outp} ({summary.shape[1]} colunas)")
        except Exception:
            pass

    return df_out, summary


# -----------------------------------------------------------------------------
# Target e pipeline compacto
# -----------------------------------------------------------------------------
def build_target(df: pd.DataFrame, config: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    tcfg = config.get("target", {})
    name = tcfg.get("name", "target")
    rule = tcfg.get("rule", {})
    col, op, value = rule.get("col"), rule.get("op"), rule.get("value")
    if col is None or op is None:
        raise ValueError("Config de target inválida: especifique 'col' e 'op'.")
    ops = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }
    if op not in ops:
        raise ValueError(f"Operador não suportado: {op}")
    df[name] = ops[op](df[col], value).astype(int)
    meta = {"name": name, "rule": rule}
    return df, meta

# -----------------------------------------------------------------------------
# Criação e verificação do target (variável alvo)
# -----------------------------------------------------------------------------
def ensure_target_from_config(df: pd.DataFrame, config: dict, verbose: bool = False):
    """
    Garante a existência/consistência do target conforme o config['target'].

    Retorna: df, target_name, class_map, report_df
    - Nunca sobrescreve um target existente.
    - Compara valores de forma case-insensitive e com strip().
    """
    import pandas as pd

    tgt_cfg   = dict(config.get("target", {}))
    tname     = tgt_cfg.get("name", "target")
    positive  = tgt_cfg.get("positive")
    negative  = tgt_cfg.get("negative")
    src_col   = tgt_cfg.get("source")

    def _norm(x):
        return None if x is None else str(x).strip().casefold()

    pos_n = _norm(positive)
    neg_n = _norm(negative)

    # 1) Se o target já existe, apenas reporta (idempotente)
    if tname in df.columns:
        if verbose:
            print(f"[target] Coluna '{tname}' já existe — nenhuma ação necessária.")
        # tenta inferir classes do próprio df, com casefold
        s_norm = df[tname].astype(str).str.strip().str.casefold()
        uniq   = sorted([u for u in s_norm.dropna().unique().tolist()])
        # class_map informativo: se houver pos/neg no config, mantenha
        class_map = {1: positive, 0: negative} if (positive is not None and negative is not None) else {u:i for i,u in enumerate(uniq)}
        report = pd.DataFrame({
            "target":   [tname],
            "status":   ["já existe"],
            "source":   [src_col or tname],
            "positive": [positive],
            "negative": [negative]
        })
        return df, tname, class_map, report

    # 2) Se não existe, mas há coluna fonte
    if src_col and src_col in df.columns:
        if verbose:
            print(f"[target] Criando '{tname}' a partir de '{src_col}'.")
        s = df[src_col]

        if positive is not None and negative is not None:
            s_norm = s.astype(str).str.strip().str.casefold()
            df[tname] = pd.NA
            df.loc[s_norm == pos_n, tname] = 1
            df.loc[s_norm == neg_n, tname] = 0
            class_map = {1: positive, 0: negative}
        elif s.dtype == "bool":
            df[tname] = s.astype(int)
            class_map = {1: True, 0: False}
        else:
            uniq = sorted(s.dropna().unique().tolist())
            mapping = {val: i for i, val in enumerate(uniq)}
            df[tname] = s.map(mapping)
            if verbose:
                print(f"[target] Mapeamento automático: {mapping}")
            class_map = mapping

        rep = pd.DataFrame({
            "target":   [tname],
            "status":   ["criado"],
            "source":   [src_col],
            "positive": [positive],
            "negative": [negative]
        })
        if verbose:
            print(f"[target] Conclusão: criado ({tname})")
        return df, tname, class_map, rep

    # 3) Sem target e sem fonte → não cria nada “vazio” (evita ruído)
    if verbose:
        print(f"[target] Nem target '{tname}' existe nem source configurada. Nenhuma ação tomada.")
    rep = pd.DataFrame({
        "target":   [tname],
        "status":   ["não criado"],
        "source":   [src_col],
        "positive": [positive],
        "negative": [negative]
    })
    return df, tname, {}, rep



def apply_encoding_and_scaling(df: pd.DataFrame,
                               config: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    meta: Dict[str, Any] = {}

    enc_cfg = config.get("encoding", {})
    df, enc_meta = encode_categories_safe(
        df,
        exclude_cols=enc_cfg.get("exclude_cols"),
        drop_first=enc_cfg.get("drop_first", False),
        high_cardinality_threshold=enc_cfg.get("high_cardinality_threshold", 20),
        top_k=enc_cfg.get("top_k", None),
    )
    meta["encoding"] = enc_meta

    sc_cfg = config.get("scaling", {})
    df, sc_meta = scale_numeric_safe(
        df,
        exclude_cols=sc_cfg.get("exclude_cols"),
        method=sc_cfg.get("method", "standard"),
    )
    meta["scaling"] = sc_meta

    return df, meta

# -----------------------------------------------------------------------------
# Catálogo simples de DataFrames
# -----------------------------------------------------------------------------
class TableStore:
    """
    Catálogo simples de DataFrames nomeados.
    Compatível com __init__(initial=..., current=...) e APIs legadas.
    """
    def __init__(self, initial: Optional[dict] = None, current: Optional[str] = None):
        self._tables: Dict[str, pd.DataFrame] = {}
        self.current: Optional[str] = None
        if initial:
            for name, df in initial.items():
                self.put(name, df)
        if current is not None:
            if current not in self._tables:
                raise KeyError(f"'{current}' não encontrada em initial: {list(self._tables.keys())}")
            self.current = current
        elif self._tables:
            self.current = sorted(self._tables.keys())[0]

    # aliases legados
    def add(self, name: str, df: pd.DataFrame, set_current: bool = False) -> None:
        self.put(name, df, set_current=set_current)

    def put(self, name: str, df: pd.DataFrame, set_current: bool = False) -> None:
        self._tables[name] = df.copy()
        if set_current or self.current is None:
            self.current = name

    def get(self, name: Optional[str] = None) -> pd.DataFrame:
        key = name or self.current
        if key is None or key not in self._tables:
            raise KeyError(f"Tabela '{key}' não encontrada. Disponíveis: {list(self._tables.keys())}")
        return self._tables[key].copy()

    def use(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            raise KeyError(f"Tabela '{name}' não encontrada. Disponíveis: {list(self._tables.keys())}")
        self.current = name
        return self._tables[name].copy()

    def names(self) -> List[str]:
        return sorted(self._tables.keys())

    def list(self) -> pd.DataFrame:
        rows = []
        for name, df in self._tables.items():
            rows.append({
                "name": name,
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024**2)), 3),
                "current": bool(name == self.current),
            })
        return pd.DataFrame(rows).sort_values(["current", "name"], ascending=[False, True]).reset_index(drop=True)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.get(name)

    def __setitem__(self, name: str, df: pd.DataFrame) -> None:
        self.put(name, df)

    def __len__(self) -> int:
        return len(self._tables)

# -----------------------------------------------------------------------------
# Helpers e relatórios
# -----------------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"[set_random_seed] seed={seed}")

def set_display(max_rows: int = 200, max_cols: int = 120) -> None:
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_cols)
    logger.info(f"[set_display] rows={max_rows} cols={max_cols}")

def generate_human_report_md(df: pd.DataFrame, title: str = "Relatório de Dados") -> str:
    lines = [f"# {title}", "", f"- Linhas: {len(df)}", f"- Colunas: {len(df.columns)}", ""]
    lines.append("## Dtypes")
    for c, t in df.dtypes.astype(str).to_dict().items():
        lines.append(f"- **{c}**: `{t}`")
    lines.append("")
    lines.append("## Missing (%)")
    miss = df.isna().mean().mul(100).round(2).sort_values(ascending=False)
    for c, v in miss.items():
        lines.append(f"- {c}: {v}%")
    return "\n".join(lines)

def md_to_pdf(md_text: str, out_path: Union[str, Path], engine: str = "weasyprint") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if engine == "weasyprint":
        try:
            import weasyprint  # type: ignore
        except Exception as e:
            raise RuntimeError("weasyprint não está instalado. Use `pip install weasyprint` ou engine='pandoc'.") from e
        html = f"<pre>{md_text}</pre>"
        weasyprint.HTML(string=html).write_pdf(str(out_path))
        return out_path
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc não encontrado no PATH. Instale o binário ou use engine='weasyprint'.")
    tmp_md = out_path.with_suffix(".tmp.md")
    tmp_md.write_text(md_text, encoding="utf-8")
    cp = subprocess.run(["pandoc", str(tmp_md), "-o", str(out_path)], capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"Falha no pandoc: {cp.stderr}")
    tmp_md.unlink(missing_ok=True)
    return out_path


def apply_outlier_flags(
    df: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
    *,
    method: Optional[str] = None,           # "iqr" | "zscore" | None -> lê do config
    iqr_factor: Optional[float] = None,     # multiplicador do IQR (ex.: 1.5 ou 3.0)
    z_threshold: Optional[float] = None,    # z-score threshold (ex.: 3.0)
    cols: Optional[Sequence[str]] = None,   # se None, usa numéricas
    exclude_cols: Optional[Sequence[str]] = None,
    exclude_binaries: Optional[bool] = None,
    flag_suffix: str = "_is_outlier",
    persist: Optional[bool] = None,
    persist_relpath: Optional[str] = None,  # caminho relativo em reports/
    root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cria colunas booleanas <col>_is_outlier para cada coluna indicada (ou numéricas) a partir de
    um *mask* de outliers calculado por IQR ou z-score. Lê preferências do `config` atual:
      - config["detect_outliers"] (bool)
      - config["outlier_method"] ("iqr"|"zscore")
      - config["outliers"] dict com:
          - cols (lista ou null)            -> restringe a colunas específicas
          - exclude_cols (lista)            -> ignora colunas
          - exclude_binaries (bool)         -> omite colunas {0,1} e {True,False}
          - iqr_factor (float)
          - z_threshold (float)
          - persist_summary (bool)          -> salva CSV de resumo
          - persist_relpath (str)           -> ex: "outliers/summary.csv"

    Retorna (df_modificado, info_dict). O df retorna *cópia* com flags adicionadas.
    """
    # --- ler config com retrocompatibilidade ---
    cfg = dict(config or {})
    out_cfg = dict(cfg.get("outliers", {}))

    enabled = bool(cfg.get("detect_outliers", True))
    if not enabled:
        return df.copy(), {
            "applied": False,
            "reason": "detect_outliers desabilitado no config",
            "flag_cols": [],
            "counts": {},
        }

    method = (method or cfg.get("outlier_method") or out_cfg.get("method") or "iqr").lower()
    iqr_k = float(iqr_factor if iqr_factor is not None else out_cfg.get("iqr_factor", 1.5))
    z_thr = float(z_threshold if z_threshold is not None else out_cfg.get("z_threshold", 3.0))
    user_cols = cols if cols is not None else out_cfg.get("cols")
    exc_cols = set((out_cfg.get("exclude_cols", []) or []) + list(exclude_cols or []))
    exc_bin = bool(out_cfg.get("exclude_binaries", True) if exclude_binaries is None else exclude_binaries)

    persist_flag = bool(out_cfg.get("persist_summary", False) if persist is None else persist)
    persist_rel = persist_relpath or out_cfg.get("persist_relpath", "outliers/summary.csv")

    # --- escolher colunas ---
    if user_cols is None:
        cols_work = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        cols_work = [c for c in user_cols if c in df.columns]

    # excluir colunas removidas e binárias se solicitado
    cols_work = [c for c in cols_work if c not in exc_cols]
    if exc_bin:
        def _is_binary(s: pd.Series) -> bool:
            vals = set(pd.Series(s.dropna().unique()).tolist())
            return vals.issubset({0,1}) or vals.issubset({True, False})
        cols_work = [c for c in cols_work if not _is_binary(df[c])]

    # --- calcula máscara de outliers ---
    if method == "iqr":
        mask = detect_outliers_iqr(df, cols=cols_work, k=iqr_k)
    elif method in {"z", "zscore", "z-score"}:
        mask = detect_outliers_zscore(df, cols=cols_work, z=z_thr)
        method = "zscore"
    else:
        # fallback para iqr
        mask = detect_outliers_iqr(df, cols=cols_work, k=iqr_k)
        method = "iqr"

    # --- adiciona colunas flag ---
    out = df.copy()
    flag_cols: list[str] = []
    counts: Dict[str, int] = {}
    for c in mask.columns:
        flag_name = f"{c}{flag_suffix}"
        out[flag_name] = mask[c].astype(bool)
        flag_cols.append(flag_name)
        counts[flag_name] = int(mask[c].sum())

    info: Dict[str, Any] = {
        "applied": True,
        "method": method,
        "iqr_factor": iqr_k,
        "z_threshold": z_thr,
        "scanned_cols": cols_work,
        "flag_cols": flag_cols,
        "created_flags": len(flag_cols),
        "counts": counts,
        "persisted": None,
    }

    # --- persistência opcional do resumo ---
    if persist_flag and len(flag_cols) > 0:
        # salva em reports/<persist_rel>
        if persist_rel is None or str(persist_rel).strip() == "":
            persist_rel = "outliers/summary.csv"
        report_df = (
            pd.Series(counts, name="outliers_count")
            .sort_values(ascending=False)
            .to_frame()
            .reset_index()
            .rename(columns={"index": "flag"})
        )
        try:
            save_report_df(report_df, persist_rel, root=root)
            info["persisted"] = {"report_relpath": persist_rel, "rows": int(len(report_df))}
        except Exception as e:
            logger.warning(f"[apply_outlier_flags] falha ao persistir '{persist_rel}': {e}")

    logger.info(f"[apply_outlier_flags] flags criadas: {len(flag_cols)} | método={method}")
    return out, info

# -----------------------------------------------------------------------------
# Remoção de duplicidades
# -----------------------------------------------------------------------------
def deduplicate_rows(
    df: pd.DataFrame,
    subset: Optional[Sequence[str]] = None,
    keep: str = "first",
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Remove linhas duplicadas do DataFrame.

    Parâmetros:
      - subset: lista de colunas a considerar (None = todas)
      - keep: 'first' (mantém a 1ª), 'last' (mantém a última) ou False (remove todas as duplicadas)
      - config: dicionário de configuração (opcional) com chaves:
          {
            "deduplicate": {
              "subset": ["col1", "col2"],  # colunas de referência
              "keep": "first"
            }
          }

    Retorna:
      df sem duplicadas.
    """
    df = df.copy()

    # Preferências do config
    dedup_cfg = (config or {}).get("deduplicate", {}) if isinstance(config, dict) else {}
    subset = dedup_cfg.get("subset", subset)
    keep = dedup_cfg.get("keep", keep)

    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    n_after = len(df)
    removed = n_before - n_after

    logger.info(
        f"[deduplicate_rows] Removidas {removed} duplicadas "
        f"({n_before} → {n_after}) | subset={subset or 'ALL'} | keep={keep}"
    )

    return df



# -----------------------------------------------------------------------------
# Tratamento de Datas (versão com config dict): parsing com relatório + colunas parseadas
# -----------------------------------------------------------------------------
def parse_dates_with_report_cfg(
    df: pd.DataFrame,
    cfg: Optional[Mapping[str, Any]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Variante que lê um dicionário de configuração (cfg) e retorna:
      (df_convertido, report_df, parsed_cols)

    cfg:
      - detect_regex: str regex para auto-detecção (default: r"(date|data|dt_|_dt$|_date$)")
      - explicit_cols: list[str] colunas explícitas (prioridade sobre regex)
      - dayfirst: bool (default False)
      - utc: bool (default False)
      - formats: list[str] formatos strftime (ex.: ["%d/%m/%Y", "%Y-%m-%d"]); se vazio, usa auto
      - min_ratio: float entre 0 e 1 (default 0.80) -> taxa mínima de parsing aceitável
      - report_path: str|Path opcional para persistir o relatório em reports/

    Observações:
      - Não altera a função existente parse_dates_with_report; é uma variante complementar.
    """
    cfg = dict(cfg or {})
    regex = cfg.get("detect_regex", r"(date|data|dt_|_dt$|_date$)")
    explicit = list(cfg.get("explicit_cols", []) or [])
    dayfirst = bool(cfg.get("dayfirst", False))
    utc = bool(cfg.get("utc", False))
    fmts = list(cfg.get("formats", []) or [])
    min_ratio = float(cfg.get("min_ratio", 0.80))
    report_path = cfg.get("report_path", "date_parse_report.csv")

    # Seleciona colunas candidatas
    candidates: list[str] = []
    for c in explicit:
        if c in df.columns:
            candidates.append(c)
    if not candidates:
        pattern = re.compile(regex, flags=re.IGNORECASE)
        candidates = [c for c in df.columns if pattern.search(str(c))]

    out = df.copy()
    results = []
    parsed_cols: list[str] = []

    for col in candidates:
        raw = out[col]
        n = int(raw.notna().sum())
        if n == 0:
            results.append({"column": col, "non_null": 0, "parsed": 0, "ratio": 0.0, "method": "skipped_empty", "format": None})
            continue

        chosen_series = None
        chosen_fmt = None
        chosen_method = "auto"

        if fmts:
            for fmt in fmts:
                try:
                    tmp = pd.to_datetime(raw, format=fmt, errors="coerce", dayfirst=dayfirst, utc=utc)
                    ok = int(tmp.notna().sum())
                    if ok / max(n, 1) >= min_ratio:
                        chosen_series = tmp
                        chosen_fmt = fmt
                        chosen_method = "format"
                        break
                except Exception:
                    continue

        if chosen_series is None:
            tmp = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst, utc=utc)
            chosen_series = tmp
            chosen_fmt = None
            chosen_method = "auto"

        ok = int(chosen_series.notna().sum())
        ratio = ok / max(n, 1)

        results.append({
            "column": col,
            "non_null": n,
            "parsed": ok,
            "ratio": round(ratio, 4),
            "method": chosen_method,
            "format": chosen_fmt
        })

        if ratio >= min_ratio:
            out[col] = chosen_series
            parsed_cols.append(col)

    report_df = pd.DataFrame(results, columns=["column","non_null","parsed","ratio","method","format"]).sort_values("column")

    # Persistência opcional usando helper do módulo (se existir)
    try:
        if report_path:
            save_report_df(report_df, report_path)
    except Exception:
        pass

    try:
        logger.info(f"[dates/cfg] parsed_cols={parsed_cols} (min_ratio={min_ratio})")
    except Exception:
        pass

    return out, report_df, parsed_cols


# --- Datas · utilitários de varredura e renderização ------------------------
def scan_date_candidates(
    df: pd.DataFrame,
    cfg: Mapping[str, Any] | None = None,
    *,
    min_ratio: float = 0.20,
    sample: int | None = 2000
) -> pd.DataFrame:
    """
    Scanner silencioso de possíveis colunas de data entre colunas object/strings.
    - Evita spam de UserWarning do pandas (fallback dateutil).
    - Tenta formatos explícitos do cfg antes do parsing genérico.
    - Opcionalmente amostra as séries para acelerar a detecção.

    Retorna DataFrame com: column, dtype, parse_ratio, sample_examples
    (ordenado por parse_ratio desc).
    """
    import warnings

    cfg = dict(cfg or {})
    dayfirst = bool(cfg.get("dayfirst", False))
    as_utc   = bool(cfg.get("utc", False))
    formats  = list(cfg.get("formats", []) or [])

    obj_cols = [c for c in df.columns
                if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    rows = []

    for c in obj_cols:
        s = df[c]
        if sample and len(s) > sample:
            s = s.sample(sample, random_state=42)

        # Tenta formatos declarados primeiro (sem barulho)
        parsed = None
        if formats:
            for fmt in formats:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        p = pd.to_datetime(s, format=fmt, errors="coerce",
                                           dayfirst=dayfirst, utc=as_utc)
                    if p.notna().mean() >= min_ratio:
                        parsed = p
                        break
                except Exception:
                    pass

        # Se não bateu nenhum formato, tenta parsing genérico (silencioso)
        if parsed is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                parsed = pd.to_datetime(s, errors="coerce",
                                        dayfirst=dayfirst, utc=as_utc)

        ratio = float(parsed.notna().mean())
        if ratio >= 0.01:  # só registra se houve algum parsing
            ex = s.dropna().astype(str).unique().tolist()[:5]
            rows.append({
                "column": c,
                "dtype": str(df[c].dtype),
                "parse_ratio": round(ratio, 3),
                "sample_examples": ex
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("parse_ratio", ascending=False).reset_index(drop=True)
    return out


def render_date_step(parsed_cols: list[str],
                     parse_report: pd.DataFrame | None = None,
                     candidates: pd.DataFrame | None = None,
                     created_features: list[str] | None = None) -> None:
    """Renderiza cards e tabelas para a etapa de datas."""
    from IPython.display import display, HTML
    import pandas as pd

    def _card(title, subtitle=""):
        return HTML(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid #0ea5e9;
                    border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
          <div style="font-weight:700;font-size:16px">{title}</div>
          <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
        </div>
        """)

    display(_card("📅 Tratamento de Datas", "Detecção, parsing e expansão de features"))

    if parsed_cols:
        msg = f"{len(parsed_cols)} coluna(s) convertida(s): {parsed_cols}"
        display(HTML(f"<div style='color:#16a34a;font-size:12px'>{msg}</div>"))
        if isinstance(parse_report, pd.DataFrame) and not parse_report.empty:
            display(HTML("<b>📑 Relatório de parsing (amostra):</b>"))
            display(parse_report.head(20))
        if created_features:
            display(HTML("<b>🧩 Features de data criadas (amostra):</b>"))
            display(pd.DataFrame({"feature": created_features}).head(20))
    else:
        display(HTML("<div style='color:#6b7280;font-size:12px'>Nenhuma coluna de data detectada ou convertida.</div>"))
        if isinstance(candidates, pd.DataFrame) and not candidates.empty:
            display(_card("🔎 Possíveis colunas de data", "Use dates.explicit_cols e/ou dates.formats"))
            display(candidates.head(20))
            display(HTML("<div style='color:#6b7280;font-size:12px'>"
                         "Dica: mova as colunas acima para <code>dates.explicit_cols</code> "
                         "e/ou informe <code>dates.formats</code> no config."
                         "</div>"))


# -----------------------------------------------------------------------------
# Expansão de features de data (versão estendida com prefix_mode e conjunto de features)
# -----------------------------------------------------------------------------
def expand_date_features_plus(
    df: pd.DataFrame,
    date_cols: Sequence[str],
    *,
    features: Sequence[str] = ("year","month","day","dayofweek","quarter","week","is_month_start","is_month_end"),
    prefix_mode: str = "auto"  # "auto" -> <col>_<feature>, "plain" -> <feature>
) -> list[str]:
    """
    Cria colunas derivadas a partir de colunas datetime.

    features suportados:
      - year, month, day, dayofweek, quarter, week, is_month_start, is_month_end

    Retorna:
      lista de nomes das colunas criadas
    """
    created: list[str] = []
    for col in date_cols:
        if col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        for f in features:
            out_name = f"{col}_{f}" if prefix_mode == "auto" else f
            try:
                if f == "year":
                    df[out_name] = df[col].dt.year
                elif f == "month":
                    df[out_name] = df[col].dt.month
                elif f == "day":
                    df[out_name] = df[col].dt.day
                elif f == "dayofweek":
                    df[out_name] = df[col].dt.dayofweek
                elif f == "quarter":
                    df[out_name] = df[col].dt.quarter
                elif f == "week":
                    try:
                        df[out_name] = df[col].dt.isocalendar().week.astype(int)
                    except Exception:
                        df[out_name] = df[col].dt.week
                elif f == "is_month_start":
                    df[out_name] = df[col].dt.is_month_start
                elif f == "is_month_end":
                    df[out_name] = df[col].dt.is_month_end
                else:
                    continue
                created.append(out_name)
            except Exception:
                # Ignora feature específica se a operação falhar
                continue

    try:
        logger.info(f"[dates/features+] criadas: {len(created)} -> {created[:8]}{'...' if len(created)>8 else ''}")
    except Exception:
        pass

    return created


# -----------------------------------------------------------------------------
# Tratamento de Texto — extração de features básicas
# -----------------------------------------------------------------------------

def extract_text_features(
    df: pd.DataFrame,
    *,
    lower: bool = True,
    strip_collapse_ws: bool = True,
    keywords: list[str] | None = None,
    blacklist: list[str] | None = None,
    export_summary: bool = True,
    summary_dir: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrai métricas básicas de colunas textuais (string/object) e gera relatório de texto.

    Parâmetros:
      - lower: converte para minúsculas
      - strip_collapse_ws: remove espaços extras
      - keywords: lista de palavras-chave a serem contadas
      - blacklist: colunas a ignorar
      - export_summary: salva CSV de resumo (True/False)
      - summary_dir: caminho para salvar o relatório (Path ou string)

    Retorna:
      (DataFrame transformado, DataFrame resumo)
    """
    df_out = df.copy()
    text_cols = df_out.select_dtypes(include="object").columns.tolist()
    blacklist = set(blacklist or [])
    keywords = list(keywords or [])
    cols_proc = [c for c in text_cols if c not in blacklist]

    summary = []

    for col in cols_proc:
        s = df_out[col].astype(str)

        if lower:
            s = s.str.lower()
        if strip_collapse_ws:
            s = s.str.replace(r"\s+", " ", regex=True).str.strip()

        df_out[col] = s

        # features básicas
        df_out[f"{col}_len"] = s.str.len()
        df_out[f"{col}_word_count"] = s.str.split().apply(len)

        col_summary = {
            "column": col,
            "non_null": int(s.notna().sum()),
            "avg_len": round(s.str.len().mean(), 2),
            "avg_words": round(s.str.split().apply(len).mean(), 2),
        }

        # contagem de keywords
        for kw in keywords:
            kw_flag = f"{col}_has_{kw}"
            df_out[kw_flag] = s.str.contains(rf"\b{kw}\b", regex=True, na=False)
            col_summary[f"kw_{kw}_count"] = int(df_out[kw_flag].sum())

        summary.append(col_summary)

    # construir DataFrame resumo
    text_summary = pd.DataFrame(summary)
    if not text_summary.empty:
        text_summary = text_summary.sort_values("column").reset_index(drop=True)

    # persistência opcional
    if export_summary and summary_dir is not None:
        summary_dir = Path(summary_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        out_path = summary_dir / "text_features_summary.csv"
        text_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
        try:
            logger.info(f"[text] resumo salvo em: {out_path} ({len(text_summary)} colunas)")
        except Exception:
            pass

    return df_out, text_summary

# -----------------------------------------------------------------------------
# Patch additions for N2 interoperability (non-breaking)
# -----------------------------------------------------------------------------

def get_project_root() -> Path:
    return ensure_project_root()

def ensure_artifact_dirs(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    root = ensure_project_root()
    artifacts_dir = (root / cfg.get("artifacts_dir", "artifacts")).resolve()
    reports_dir = (root / "reports").resolve()
    models_dir = (artifacts_dir / "models").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"[ensure_artifact_dirs] artifacts={artifacts_dir} | reports={reports_dir} | models={models_dir}")
    except Exception:
        pass
    return artifacts_dir, reports_dir, models_dir

def resolve_processed_path(cfg: Dict[str, Any]) -> Path:
    root = ensure_project_root()
    data_proc_dir = (root / cfg.get("data_processed_dir", "data/processed")).resolve()
    explicit = cfg.get("data_processed_file")

    if explicit:
        cand = data_proc_dir / explicit
        if cand.exists():
            return cand
        else:
            try:
                logger.warning(f"[resolve_processed_path] data_processed_file='{explicit}' não encontrado em {data_proc_dir}.")
            except Exception:
                print(f"[WARN] data_processed_file='{explicit}' não encontrado em {data_proc_dir}.")

    for ext in ("*.parquet", "*.pq", "*.csv", "*.xlsx"):
        matches = sorted(data_proc_dir.glob(ext))
        if matches:
            try:
                logger.info(f"[resolve_processed_path] usando '{matches[0].name}' em {data_proc_dir}.")
            except Exception:
                pass
            return matches[0]

    for ext in ("processed.parquet", "processed.csv", "final.parquet", "final.csv"):
        for f in data_proc_dir.rglob(ext):
            try:
                logger.info(f"[resolve_processed_path] fallback recursivo usando {f}")
            except Exception:
                pass
            return f

    if not data_proc_dir.exists():
        raise FileNotFoundError(f"Diretório {data_proc_dir} não existe. Garanta que o N1 criou a pasta/arquivo.")

    files = [f.name for f in data_proc_dir.glob("*")]
    hint = "\n".join(f" - {n}" for n in files) if files else " (vazio)"
    raise FileNotFoundError(
        "Nenhum arquivo processado encontrado em data/processed/."
        f" Conteúdo atual de {data_proc_dir}:\n{hint}\n"
        "-> Ajuste 'data_processed_file' no config ou rode o N1 para gerar o arquivo final."
    )

def summarize_columns(df: pd.DataFrame) -> Tuple[list, list, list]:
    import numpy as _np
    numeric_cols = df.select_dtypes(include=[_np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[_np.number]).columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols + categorical_cols]
    return numeric_cols, categorical_cols, other_cols

try:
    __all__.extend(["get_project_root", "ensure_artifact_dirs", "resolve_processed_path", "summarize_columns"])
    __all__ = sorted(set(__all__))
except Exception:
    pass


# ============================================================================
# Adições utilitárias para N2 — definidas apenas se ainda não existirem
# ============================================================================
from pathlib import Path
from typing import Any, Dict, Tuple
import json

# ---- Descoberta da raiz do projeto via config/defaults.json ----
try:
    _find_up  # type: ignore[name-defined]
except NameError:
    def _find_up(relative_path: str, start: Path | None = None) -> Path | None:
        start = start or Path.cwd()
        rel = Path(relative_path)
        for base in (start, *start.parents):
            cand = base / rel
            if cand.exists():
                return cand
        return None

try:
    get_project_root  # type: ignore[name-defined]
except NameError:
    def get_project_root() -> Path:
        cfg_path = _find_up("config/defaults.json")
        if cfg_path is None:
            raise FileNotFoundError("config/defaults.json não encontrado. Abra o notebook dentro do projeto.")
        return cfg_path.parent.parent  # .../config/defaults.json -> raiz do projeto

# ---- Leitura de config ----
try:
    load_config  # type: ignore[name-defined]
except NameError:
    def load_config(config_rel: str = "config/defaults.json") -> Dict[str, Any]:
        root = get_project_root()
        cfg_file = (root / config_rel).resolve()
        if not cfg_file.exists():
            raise FileNotFoundError(f"Arquivo de config não encontrado: {cfg_file}")
        with cfg_file.open("r", encoding="utf-8") as f:
            return json.load(f)

# ---- Diretórios de artefatos/reports/models na RAIZ do projeto ----
try:
    ensure_dirs  # type: ignore[name-defined]
except NameError:
    def ensure_dirs(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
        root = get_project_root()
        artifacts_dir = (root / cfg.get("artifacts_dir", "artifacts")).resolve()
        reports_dir = (root / "reports").resolve()
        models_dir = (artifacts_dir / "models").resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(f"[ensure_dirs] artifacts={artifacts_dir} | reports={reports_dir} | models={models_dir}")  # type: ignore[name-defined]
        except Exception:
            pass
        return artifacts_dir, reports_dir, models_dir

# ---- Resolução do arquivo processado (saída do N1) ----
try:
    discover_processed_path  # type: ignore[name-defined]
except NameError:
    def discover_processed_path(cfg: Dict[str, Any]) -> Path:
        root = get_project_root()
        data_proc_dir = (root / cfg.get("data_processed_dir", "data/processed")).resolve()
        explicit = cfg.get("data_processed_file")

        # 1) Nome explícito
        if explicit:
            cand = data_proc_dir / explicit
            if cand.exists():
                return cand
            else:
                try:
                    logger.warning(f"[discover_processed_path] data_processed_file='{explicit}' não encontrado em {data_proc_dir}.")  # type: ignore[name-defined]
                except Exception:
                    print(f"[WARN] data_processed_file='{explicit}' não encontrado em {data_proc_dir}.")

        # 2) Extensões comuns
        for ext in ("*.parquet", "*.pq", "*.csv", "*.xlsx"):
            matches = sorted(data_proc_dir.glob(ext))
            if matches:
                try:
                    logger.info(f"[discover_processed_path] usando '{matches[0].name}' em {data_proc_dir}.")  # type: ignore[name-defined]
                except Exception:
                    pass
                return matches[0]

        # 3) Fallback recursivo por nomes comuns
        for ext in ("processed.parquet", "processed.csv", "final.parquet", "final.csv"):
            for f in data_proc_dir.rglob(ext):
                try:
                    logger.info(f"[discover_processed_path] fallback recursivo usando {f}")  # type: ignore[name-defined]
                except Exception:
                    pass
                return f

        # 4) Diagnóstico
        if not data_proc_dir.exists():
            raise FileNotFoundError(f"Diretório {data_proc_dir} não existe. Garanta que o N1 criou a pasta/arquivo.")

        files = [f.name for f in data_proc_dir.glob("*")]
        hint = "\n".join(f" - {n}" for n in files) if files else " (vazio)"
        msg = (
            "Nenhum arquivo processado encontrado em data/processed/."
            f" Conteúdo atual de {data_proc_dir}:\n{hint}\n"
            "-> Ajuste 'data_processed_file' no config ou rode o N1 para gerar o arquivo final."
        )
        raise FileNotFoundError(msg)

# ---- Resumo de colunas (num/cat/outros) ----
try:
    summarize_columns  # type: ignore[name-defined]
except NameError:
    def summarize_columns(df):
        import numpy as _np
        numeric_cols = df.select_dtypes(include=[_np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[_np.number]).columns.tolist()
        other_cols = [c for c in df.columns if c not in numeric_cols + categorical_cols]
        return numeric_cols, categorical_cols, other_cols

# ---- Métricas e plots auxiliares para avaliação ----
try:
    compute_metrics  # type: ignore[name-defined]
except NameError:
    def compute_metrics(y_true, y_pred):
        import pandas as _pd
        from sklearn.metrics import accuracy_score, f1_score
        avg = "binary" if _pd.Series(y_true).nunique() == 2 else "macro"
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=avg)),
        }

try:
    try_plot_roc  # type: ignore[name-defined]
except NameError:
    def try_plot_roc(clf, X_test, y_test):
        import pandas as _pd
        from sklearn.metrics import RocCurveDisplay
        import matplotlib.pyplot as _plt

        # Plot ROC apenas para binário e se houver predict_proba
        if _pd.Series(y_test).nunique() != 2:
            return False
        if not hasattr(clf, "predict_proba"):
            return False
        try:
            RocCurveDisplay.from_estimator(clf, X_test, y_test)
            _plt.title("ROC Curve")
            _plt.show()
            return True
        except Exception as e:
            print(f"[AVISO] ROC não foi plotado: {e}")
            return False

try:
    persist_artifacts  # type: ignore[name-defined]
except NameError:
    def persist_artifacts(name, pipeline, metrics, params, models_dir: Path, reports_dir: Path):
        from datetime import datetime as _dt
        import json as _json
        import joblib as _joblib

        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        base = f"{name}_{ts}"

        _joblib.dump(pipeline, models_dir / f"{base}.joblib")
        (models_dir / f"{base}_metrics.json").write_text(_json.dumps(metrics, indent=2, ensure_ascii=False))
        (models_dir / f"{base}_params.json").write_text(_json.dumps(params, indent=2, ensure_ascii=False, default=str))

        manifest_rec = {"ts": ts, "model": name, "file": f"{base}.joblib", "metrics": metrics, "params": params}
        (reports_dir / "manifest.jsonl").open("a", encoding="utf-8").write(_json.dumps(manifest_rec, ensure_ascii=False) + "\n")
        print(f"[OK] Artefatos salvos em: {models_dir} e manifesto atualizado em reports/manifest.jsonl")


# -----------------------------------------------------------------------------
# UI Futurista + Hyperdrive (Grid/Random) para N2
# -----------------------------------------------------------------------------

import json, time
from typing import Dict, Any, Tuple
import ipywidgets as W
from IPython.display import display, HTML, clear_output

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

# Reusa utilidades já existentes (persist_artifacts, etc.)
# Certifique-se que persist_artifacts já está definido acima neste módulo.

def n2_inject_css_theme() -> None:
    """Injeta o tema 'painel interdimensional' no notebook."""
    display(HTML(r"""
    <style>
      :root{
        --lumen-bg: radial-gradient(1200px 600px at 20% -10%, rgba(0,255,209,0.20) 0%, transparent 50%),
                    radial-gradient(900px 600px at 110% 110%, rgba(98,0,255,0.18) 0%, transparent 40%),
                    linear-gradient(135deg, #0a0f16 0%, #0b0e14 100%);
        --lumen-panel: rgba(15, 22, 33, 0.55);
        --lumen-glass: rgba(11, 18, 29, 0.5);
        --lumen-border: rgba(0, 255, 209, 0.35);
        --lumen-glow: 0 0 0.5rem rgba(0,255,209,0.35), inset 0 0 0.5rem rgba(0,255,209,0.1);
        --lumen-warn: #ffb86b;
        --lumen-accent: #00ffd1;
        --lumen-accent-2: #9a7dff;
        --lumen-muted: #9aa4ad;
        --lumen-text: #e8f4ff;
      }
      .lumen-console {
        background: var(--lumen-bg);
        border-radius: 14px;
        padding: 14px 16px;
        border: 1px solid var(--lumen-border);
        box-shadow: var(--lumen-glow);
        backdrop-filter: blur(10px) saturate(110%);
        color: var(--lumen-text);
        font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
      }
      .lumen-header { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
      .lumen-title { font-weight:700; letter-spacing:.3px; font-size:16px; color:var(--lumen-accent);
        text-shadow: 0 0 10px rgba(0,255,209,0.4); }
      .lumen-chip { padding:2px 8px; border:1px solid var(--lumen-border); border-radius:999px;
        font-size:11px; color:var(--lumen-accent-2); background: rgba(154,125,255,0.07);}
      .lumen-row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
      .lumen-controls { display:flex; align-items:center; gap:8px; justify-content:space-between; margin-top:8px;}
      .lumen-warning { color: var(--lumen-warn); font-size: 12px; margin-left: 6px; opacity: .9;}
      .lumen-note { color: var(--lumen-muted); font-size: 12px; margin-left: 4px; }
      .widget-tab { border:1px solid var(--lumen-border)!important; border-radius:10px; overflow:hidden;
        box-shadow: inset 0 0 10px rgba(0,255,209,0.08); background: var(--lumen-glass)!important; }
      .widget-tab>div:nth-child(1) {
        background: linear-gradient(90deg, rgba(0,255,209,0.06), rgba(154,125,255,0.05)) !important;
        border-bottom: 1px solid rgba(0,255,209,0.25) !important; }
      .widget-tab .p-TabBar-tab { color:var(--lumen-text)!important; text-shadow:0 0 8px rgba(0,255,209,0.25);
        border-right: 1px solid rgba(0,255,209,0.10); background:transparent!important; }
      .widget-tab .p-TabBar-tab.p-mod-current {
        background: linear-gradient(180deg, rgba(0,255,209,0.1), rgba(0,0,0,0)) !important;
        box-shadow: inset 0 -3px 0 var(--lumen-accent); }
      .lumen-card { margin:8px; padding:8px 10px; border:1px dashed rgba(0,255,209,0.25);
        border-radius:10px; background: rgba(0, 12, 20, 0.35); }
      .lumen-label { min-width:160px; color:var(--lumen-muted); font-size:12px; }
      .lumen-play .widget-button { background: linear-gradient(135deg, rgba(0,255,209,0.25), rgba(154,125,255,0.25));
        color: var(--lumen-text); border:1px solid var(--lumen-border); box-shadow: var(--lumen-glow); }
    </style>
    """))

def n2_model_registry() -> Dict[str, Dict[str, Any]]:
    """Define os modelos disponíveis e seus widgets de hiperparâmetros."""
    return {
        "DummyClassifier": {
            "class": __import__("sklearn.dummy", fromlist=["DummyClassifier"]).DummyClassifier,
            "params": {
                "strategy": W.Dropdown(options=["most_frequent", "prior", "stratified", "uniform"], value="most_frequent")
            },
        },
        "LogisticRegression": {
            "class": __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression,
            "params": {
                "C": W.FloatLogSlider(base=10, min=-3, max=3, step=0.1, value=1.0),
                "max_iter": W.IntSlider(min=100, max=5000, step=100, value=1000),
                "solver": W.Dropdown(options=["lbfgs", "liblinear", "saga"], value="lbfgs"),
            },
        },
        "KNeighborsClassifier": {
            "class": __import__("sklearn.neighbors", fromlist=["KNeighborsClassifier"]).KNeighborsClassifier,
            "params": {
                "n_neighbors": W.IntSlider(min=1, max=50, step=1, value=5),
                "weights": W.Dropdown(options=["uniform", "distance"], value="uniform"),
                "p": W.IntSlider(min=1, max=2, step=1, value=2),
            },
        },
        "RandomForestClassifier": {
            "class": __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier,
            "params": {
                "n_estimators": W.IntSlider(min=50, max=1000, step=50, value=200),
                "max_depth": W.Dropdown(options=[None, 5, 10, 20, 50], value=None),
                "min_samples_split": W.IntSlider(min=2, max=20, step=1, value=2),
                "min_samples_leaf": W.IntSlider(min=1, max=20, step=1, value=1),
            },
        },
    }

# --------- helpers internos ----------
def _params_vbox(spec_params: Dict[str, Any]) -> W.VBox:
    rows = []
    for k, widget in spec_params.items():
        if hasattr(widget, "layout"):
            widget.layout.width = "340px"
        rows.append(W.HBox([W.HTML(f"<span class='lumen-label'>{k}</span>"), widget]))
    card = W.VBox(rows, layout=W.Layout(padding="6px"))
    return W.VBox([card], layout=W.Layout())

def _set_disabled(box: W.VBox, disabled: bool=True):
    for child in box.children:
        for row in child.children:
            if isinstance(row, W.HBox) and len(row.children) == 2:
                try:
                    row.children[1].disabled = disabled
                except Exception:
                    pass

def _apply_tab_title_style(tab: W.Tab, idx: int, title: str, enabled: bool):
    mark = "✦ " if enabled else "⛔ "
    tab.set_title(idx, f"{mark}{title}")

def _widget_to_candidates(w):
    # Transforma um widget em uma lista de candidatos discretos para Grid/Random
    if isinstance(w, W.IntSlider):
        lo, hi = int(w.min), int(w.max)
        if hi <= lo:
            return [int(w.value)]
        mid = int(w.value)
        vals = sorted({lo, mid, hi})
        if len(vals) < 3 and hi - lo > 5:
            vals = sorted({lo, (lo+hi)//2, hi})
        return list(vals)
    if isinstance(w, (W.FloatSlider, W.FloatLogSlider)):
        try:
            base = w.base if hasattr(w, "base") else 10
            lo_exp, hi_exp = float(w.min), float(w.max)
            exps = np.linspace(lo_exp, hi_exp, num=5)
            vals = [float(base**e) for e in exps]
            vals = sorted(set([round(v, 10) for v in vals] + [float(w.value)]))
            return vals
        except Exception:
            lo, hi = float(getattr(w, "min", 0.1)), float(getattr(w, "max", 10.0))
            vals = np.linspace(lo, hi, num=5).tolist()
            if float(w.value) not in vals:
                vals.append(float(w.value))
            return sorted(set([round(v, 10) for v in vals]))
    if isinstance(w, W.Dropdown):
        return [opt for opt in w.options]
    return [getattr(w, "value", None)]

def _build_search_space(model_registry: Dict[str, Dict[str, Any]], model_name: str) -> dict:
    spec = model_registry[model_name]
    grid = {}
    for pname, widget in spec["params"].items():
        cand = _widget_to_candidates(widget)
        grid[f"clf__{pname}"] = cand
    return grid

# --------- componentes públicos ----------
def n2_build_models_ui(preprocess, X_train, y_train, X_test, y_test, models_dir, reports_dir):
    """
    Monta toda a UI de:
      - seleção de modelos,
      - abas de hiperparâmetros (com travas),
      - treino direto
      - Hyperdrive (GridSearchCV/RandomizedSearchCV)
    """
    n2_inject_css_theme()
    MODEL_REGISTRY = n2_model_registry()

    # Abas de params
    model_checks = {
        name: W.Checkbox(value=(name in ["LogisticRegression", "RandomForestClassifier"]),
                         description=name, indent=False)
        for name in MODEL_REGISTRY
    }

    model_param_boxes = {}
    tab_children, tab_titles = [], []
    name_to_index, index_to_name = {}, {}

    for i, (name, spec) in enumerate(MODEL_REGISTRY.items()):
        box = _params_vbox(spec["params"])
        model_param_boxes[name] = box
        tab_children.append(W.VBox([W.HTML(f"<div class='lumen-chip'>Hyperparams</div>"), box],
                                   layout=W.Layout(padding="6px")))
        tab_titles.append(name)
        name_to_index[name] = i
        index_to_name[i] = name

    tab = W.Tab(children=tab_children)
    for i, t in enumerate(tab_titles):
        _apply_tab_title_style(tab, i, t, model_checks[t].value)

    enabled_models = {n for n, cb in model_checks.items() if cb.value}
    for name, box in model_param_boxes.items():
        _set_disabled(box, disabled=(name not in enabled_models))

    last_valid_index = next((name_to_index[n] for n in MODEL_REGISTRY if n in enabled_models), 0)
    tab.selected_index = last_valid_index

    def on_model_toggle(change, model_name: str):
        nonlocal last_valid_index, enabled_models
        is_on = change["new"]
        if is_on:
            enabled_models.add(model_name)
        else:
            enabled_models.discard(model_name)
        _set_disabled(model_param_boxes[model_name], disabled=(not is_on))
        _apply_tab_title_style(tab, name_to_index[model_name], model_name, is_on)
        current_idx = tab.selected_index
        current_name = index_to_name.get(current_idx)
        if current_name == model_name and (not is_on):
            next_idx = None
            for nm in MODEL_REGISTRY.keys():
                if nm in enabled_models:
                    next_idx = name_to_index[nm]
                    break
            if next_idx is not None:
                tab.selected_index = next_idx
                last_valid_index = next_idx

    for name, cb in model_checks.items():
        cb.observe(lambda ch, n=name: on_model_toggle(ch, n), names="value")

    def on_tab_change(change):
        nonlocal last_valid_index
        if change["name"] == "selected_index":
            new_idx = change["new"]
            if new_idx is None or new_idx == -1:
                return
            new_name = index_to_name.get(new_idx)
            if new_name not in enabled_models:
                tab.selected_index = last_valid_index
            else:
                last_valid_index = new_idx

    tab.observe(on_tab_change, names="selected_index")

    # Treino simples
    btn_train = W.Button(description="Treinar modelos selecionados", button_style="success", icon="play")
    cb_persist = W.Checkbox(value=True, description="Salvar artefatos (modelo, métricas, params)")
    out_simple = W.Output()

    @out_simple.capture(clear_output=True)
    def _on_train_clicked(_):
        selected = [n for n in MODEL_REGISTRY if n in enabled_models]
        if not selected:
            print("Nenhum modelo selecionado.")
            return
        print("Treinando:", ", ".join(selected))
        for name in selected:
            Model = MODEL_REGISTRY[name]["class"]
            params = {}
            for row in model_param_boxes[name].children[0].children:
                label_html, widget = row.children
                key = label_html.value.replace("<span class='lumen-label'>","").replace("</span>","")
                params[key] = getattr(widget, "value", None)

            pipe = SKPipeline(steps=[("prep", preprocess), ("clf", Model(**params))])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            avg = "binary" if pd.Series(y_test).nunique() == 2 else "macro"
            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average=avg)
            print(f"[{name}] test_accuracy={acc:.4f} | test_f1={f1:.4f}")

            if cb_persist.value:
                persist_artifacts(
                    name=f"{name}_Manual",
                    pipeline=pipe,
                    metrics={"test_accuracy": float(acc), "test_f1": float(f1)},
                    params=params,
                    models_dir=models_dir,
                    reports_dir=reports_dir
                )

    btn_train.on_click(_on_train_clicked)

    # Hyperdrive
    search_title = W.HTML("<div class='lumen-title'>Hyperdrive · Grid / Random Search</div>")
    search_note  = W.HTML("<span class='lumen-note'>Escolha um modelo, gere a grade e execute a busca.</span>")
    dd_model_search = W.Dropdown(options=list(MODEL_REGISTRY.keys()), value="LogisticRegression", description="Modelo:")
    dd_strategy = W.Dropdown(options=["GridSearchCV", "RandomizedSearchCV"], value="GridSearchCV", description="Estratégia:")
    dd_scoring = W.Dropdown(options=["accuracy", "f1", "f1_macro", "roc_auc", "roc_auc_ovr"], value="f1", description="Scoring:")
    sl_cv = W.IntSlider(min=3, max=10, step=1, value=5, description="CV folds:")
    sl_niter = W.IntSlider(min=5, max=200, step=5, value=30, description="n_iter (Random):")
    btn_generate = W.Button(description="Gerar grade", icon="cogs")
    btn_runsearch = W.Button(description="Executar Hyperdrive", icon="rocket", button_style="success")
    ta_space = W.Textarea(value="{}", description="Espaço de busca (JSON):", layout=W.Layout(width="100%", height="130px"))
    out_info = W.Output()
    out = W.Output()

    @out_info.capture(clear_output=True)
    def _on_generate_clicked(_):
        model_name = dd_model_search.value
        grid = _build_search_space(MODEL_REGISTRY, model_name)
        size = 1
        for _, v in grid.items():
            size *= max(1, len(v))
        print(f"[INFO] Espaço gerado para {model_name}: {size} combinações (aprox.)")
        print("[INFO] Param grid:")
        print(json.dumps(grid, indent=2, ensure_ascii=False))
        ta_space.value = json.dumps(grid, indent=2, ensure_ascii=False)

    btn_generate.on_click(_on_generate_clicked)

    @out.capture(clear_output=True)
    def _on_runsearch_clicked(_):
        try:
            param_grid = json.loads(ta_space.value)
            assert isinstance(param_grid, dict)
        except Exception as e:
            print(f"[ERRO] Espaço de busca inválido: {e}")
            return

        model_name = dd_model_search.value
        Model = MODEL_REGISTRY[model_name]["class"]
        clf = Model()
        pipe = SKPipeline(steps=[("prep", preprocess), ("clf", clf)])

        scoring = dd_scoring.value
        cv = int(sl_cv.value)
        strategy = dd_strategy.value
        start = time.time()

        print(f"[Hyperdrive] Estratégia: {strategy} | Modelo: {model_name} | scoring={scoring} | cv={cv}")
        try:
            if strategy == "GridSearchCV":
                search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1)
            else:
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=param_grid,
                    n_iter=int(sl_niter.value),
                    scoring=scoring,
                    cv=cv,
                    random_state=42,
                    n_jobs=-1
                )
            search.fit(X_train, y_train)  # X cru — preprocess no pipeline
            dur = time.time() - start

            print(f"\n[OK] Busca finalizada em {dur:.1f}s")
            print("[BEST] score (cv):", round(float(search.best_score_), 4))
            print("[BEST] params:")
            print(json.dumps(search.best_params_, indent=2, ensure_ascii=False))

            y_pred = search.best_estimator_.predict(X_test)
            avg = "binary" if pd.Series(y_test).nunique() == 2 else "macro"
            acc = accuracy_score(y_test, y_pred)
            f1v = f1_score(y_test, y_pred, average=avg)
            print("\n[TEST] accuracy:", round(float(acc), 4), "| f1:", round(float(f1v), 4))

            # Persistência opcional — reaproveita cb_persist do bloco simples
            if cb_persist.value:
                params_clean = {k: (v if isinstance(v, (int, float, str, type(None), bool)) else str(v))
                                for k, v in search.best_params_.items()}
                persist_artifacts(
                    name=f"{model_name}_Hyperdrive_{strategy}",
                    pipeline=search.best_estimator_,
                    metrics={"cv_best_score": float(search.best_score_),
                             "test_accuracy": float(acc),
                             "test_f1": float(f1v),
                             "scoring": scoring,
                             "cv": cv},
                    params=params_clean,
                    models_dir=models_dir,
                    reports_dir=reports_dir
                )
        except Exception as e:
            print(f"[FALHA] {type(e).__name__}: {e}")

    btn_runsearch.on_click(_on_runsearch_clicked)

    # Layout final
    header = W.HBox([
        W.HTML("<div class='lumen-title'>Seletor de modelos · Hyperdrive</div>"),
        W.HTML("<div class='lumen-chip'>v2 · warp-ready</div>"),
    ], layout=W.Layout(justify_content="space-between"))

    checks_row = W.HBox(list(model_checks.values()))
    controls_simple = W.HBox([btn_train, cb_persist], layout=W.Layout(gap="8px"))

    search_controls_top = W.HBox([dd_model_search, dd_strategy, dd_scoring, sl_cv, sl_niter],
                                 layout=W.Layout(gap="8px", flex_flow="row wrap"))
    search_controls_btns = W.HBox([btn_generate, btn_runsearch], layout=W.Layout(gap="8px"))
    hyper_box = W.VBox([
        W.HTML("<div class='lumen-chip'>Hyperdrive — Busca de hiperparâmetros</div>"),
        search_title,
        search_note,
        W.HTML("<div class='lumen-chip'>Espaço de busca (param_grid)</div>"),
        ta_space,
        search_controls_top,
        search_controls_btns,
        out_info,
        W.HTML("<hr style='border-color: rgba(0,255,209,0.2)'>"),
        out
    ], layout=W.Layout(padding="8px"))

    panel = W.VBox([
        header,
        checks_row,
        tab,
        W.HTML("<div class='lumen-chip'>Treino direto (usa os hiperparâmetros selecionados nas abas acima)</div>"),
        controls_simple,
        out_simple,
        W.HTML("<div style='height:10px;'></div>"),
        hyper_box
    ], layout=W.Layout(padding="8px"))

    display(W.Box([panel], layout=W.Layout(width="100%")),
            HTML("<div class='lumen-console' style='margin-top:8px;'></div>"))


# =========================================================
# UI helpers & fontes "safe" (não expõe caminhos absolutos)
# =========================================================
try:
    from IPython.display import display, HTML  # noqa
except Exception:  # ambiente sem Jupyter
    def display(*_args, **_kwargs):  # type: ignore
        pass
    class HTML:  # type: ignore
        def __init__(self, *_a, **_k): pass

from pathlib import Path
from datetime import datetime
import pandas as _pd

# bump de versão (opcional)
try:
    UTILS_DATA_VERSION
    UTILS_DATA_VERSION = str(UTILS_DATA_VERSION) + "+ui-helpers"
except NameError:
    UTILS_DATA_VERSION = "1.2.3-ui"

_UI_ACCENTS = {
    "info": "#0ea5e9",
    "ok": "#22c55e",
    "warn": "#f59e0b",
    "err": "#ef4444",
    "muted": "#64748b",
    "violet": "#8b5cf6",
    "teal": "#10b981",
}

def human_size(num_bytes: int) -> str:
    """Converte bytes em B/KB/MB/GB/TB com formatação amigável.
    - Para KB/MB/GB: 0 casas decimais se >= 100; 1 casa se < 100.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    u = 0
    while size >= 1024 and u < len(units) - 1:
        size /= 1024.0
        u += 1
    if u == 0:
        return f"{int(size)} {units[u]}"
    fmt = "{:.0f}" if size >= 100 else "{:.1f}"
    return f"{fmt.format(size)} {units[u]}"

def list_raw_sources_safe(raw_dir: Path, pattern: str = "*", show_rel: bool = True,
                          rel_root: str = "data/raw") -> _pd.DataFrame:
    """Lista arquivos em data/raw sem expor caminho absoluto.
    Retorna colunas: file, size, size_bytes, modified, relpath (opcional).
    """
    rows = []
    for p in sorted(raw_dir.glob(pattern)):
        if not p.is_file():
            continue
        st = p.stat()
        rows.append({
            "file": p.name,
            "size": human_size(st.st_size),
            "size_bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            **({"relpath": str(Path(rel_root) / p.name)} if show_rel else {})
        })
    cols = ["file", "size", "size_bytes", "modified"] + (["relpath"] if show_rel else [])
    return _pd.DataFrame(rows, columns=cols)

def _card_html(title: str, subtitle: str = "", accent: str = _UI_ACCENTS["teal"]) -> HTML:
    """Cria um 'card' simples para separar seções no notebook."""
    return HTML(f"""
    <div style="border:1px solid #e5e7eb;border-left:6px solid {accent};
                border-radius:10px;padding:12px 14px;margin:16px 0;background:#fafafa">
      <div style="font-weight:700;font-size:16px;line-height:1.2">{title}</div>
      <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
    </div>
    """)



def overview_table(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo compacto de linhas/colunas/memória."""
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    rows = [
        {"Métrica": "Linhas",        "Valor": _fmt_compact(df.shape[0])},
        {"Métrica": "Colunas",       "Valor": _fmt_compact(df.shape[1])},
        {"Métrica": "Memória (MB)",  "Valor": _fmt_compact(mem_mb)},
    ]
    return pd.DataFrame(rows, columns=["Métrica", "Valor"])

def dtypes_summary(df: _pd.DataFrame) -> _pd.DataFrame:
    """Contagem por dtype (string)."""
    return (df.dtypes.astype(str)
              .value_counts()
              .rename_axis("dtype")
              .reset_index(name="cols"))

def missing_top(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Top N colunas com mais faltantes + dtype, com formatação compacta."""
    try:
        mr = missing_report(df)  # usa a função já existente
        # adicionar dtype
        dtype_map = df.dtypes.astype(str).to_dict()
        mr["dtype"] = mr["column"].map(dtype_map)

        # renomear colunas para a saída final
        mr = mr.rename(columns={
            "column": "Coluna",
            "missing_count": "Faltantes",
            "missing_pct": "%Faltantes"
        })

        # ordenar e limitar
        mr = mr.sort_values("%Faltantes", ascending=False).head(top).reset_index(drop=True)

        # tipos adequados e formatação compacta
        mr["Faltantes"] = mr["Faltantes"].astype(int)
        # arredonda para 3 casas e remove zeros excedentes na hora de exibir
        mr["%Faltantes"] = mr["%Faltantes"].round(3).map(_fmt_compact)

        # reorganizar colunas para ficar mais intuitivo
        cols = ["Coluna", "dtype", "Faltantes", "%Faltantes"]
        mr = mr[cols]
        return mr
    except Exception:
        # fallback simples se algo der errado
        nuls = df.isna().sum().sort_values(ascending=False)
        out = pd.DataFrame({"Coluna": nuls.index, "Faltantes": nuls.values})
        out["%Faltantes"] = (out["Faltantes"] / len(df)) * 100
        out["dtype"] = out["Coluna"].map(df.dtypes.astype(str).to_dict())
        out["Faltantes"] = out["Faltantes"].astype(int)
        out["%Faltantes"] = out["%Faltantes"].round(3).map(_fmt_compact)
        cols = ["Coluna", "dtype", "Faltantes", "%Faltantes"]
        return out[cols].head(top).reset_index(drop=True)

def show_block(title: str, subtitle: str, df_display: _pd.DataFrame, accent: str = _UI_ACCENTS["info"]) -> None:
    """Mostra um card com título/subtítulo seguido de um DataFrame estilizado."""
    display(_card_html(title, subtitle, accent=accent))
    try:
        display(df_display.style.set_properties(**{"font-size": "12px"}))
    except Exception:
        display(df_display)

def show_source_overview(name: str, path: Path, df: _pd.DataFrame) -> None:
    """Mostra três cards: overview, dtypes e faltantes para uma fonte específica."""
    show_block(f"📥 Fonte: {name}", f"Arquivo: {path.name}", overview_table(df), accent=_UI_ACCENTS["info"])
    show_block("🧬 Tipos (resumo)", "Contagem por dtype", dtypes_summary(df), accent=_UI_ACCENTS["muted"])
    show_block("🩺 Faltantes (top 20)", "Colunas com mais ausentes", missing_top(df, top=20), accent=_UI_ACCENTS["warn"])

def show_df_summary(df: _pd.DataFrame, label: str = "DF base", accent: str = _UI_ACCENTS["ok"]) -> None:
    """Mostra overview, dtypes e faltantes para um DataFrame 'principal' do pipeline."""
    show_block(f"🧾 Visão geral — {label}", "", overview_table(df), accent=accent)
    show_block(f"🧬 Tipos (resumo) — {label}", "Contagem por dtype", dtypes_summary(df), accent=_UI_ACCENTS["muted"])
    show_block(f"🩺 Faltantes (top 20) — {label}", "Após merges (se houver)", missing_top(df, top=20), accent=_UI_ACCENTS["warn"])


def _fmt_compact(x):
    """Formata números sem zeros inúteis.
    - int -> 123
    - float -> até 3 casas, removendo zeros (ex.: 6.821, 0.5, 12)
    """
    import numpy as _np
    import pandas as _pd

    if x is None or (isinstance(x, float) and _np.isnan(x)):
        return ""
    # numpy types
    if isinstance(x, (_np.integer, )):
        return int(x)
    if isinstance(x, (_np.floating, )):
        xf = float(x)
        if xf.is_integer():
            return int(xf)
        s = f"{xf:.3f}".rstrip("0").rstrip(".")
        return s
    # pandas scalar int/float
    if isinstance(x, (int, )):
        return x
    if isinstance(x, float):
        if float(x).is_integer():
            return int(x)
        s = f"{x:.3f}".rstrip("0").rstrip(".")
        return s
    return x

# ========= PATCH: UI "neat" + runner de Qualidade & Tipagem =========
# (cole este bloco no final do utils_data.py)

from typing import Optional, Sequence, Mapping, Any, Tuple, Dict
import pandas as _pd
from IPython.display import display, HTML
import logging as _logging
from contextlib import contextmanager as _contextmanager
from pathlib import Path as _Path

# ---------- formatação ----------
def _fmt_auto(x: float, decimals: int = 2) -> str:
    """Se for inteiro, sem casas; senão, até 'decimals' casas (trim)."""
    try:
        if float(x).is_integer():
            return str(int(round(float(x))))
        s = f"{float(x):.{decimals}f}"
        return s.rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def _fmt_mem_mb(x_mb: float) -> str:
    """Formata memória em MB com até 2 casas, sem zeros finais."""
    return f"{_fmt_auto(x_mb, 2)} MB"

# ---------- helpers de exibição ----------
def _card(title: str, subtitle: str = "", accent: str = "#22c55e") -> HTML:
    return HTML(f"""
    <div style="border:1px solid #e5e7eb;border-left:6px solid {accent};
                border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
      <div style="font-weight:700;font-size:16px;letter-spacing:.2px">{title}</div>
      <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
    </div>
    """)

def _overview_table_neat(df: _pd.DataFrame) -> _pd.DataFrame:
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    return _pd.DataFrame([
        {"Métrica": "Linhas",       "Valor": int(df.shape[0])},
        {"Métrica": "Colunas",      "Valor": int(df.shape[1])},
        {"Métrica": "Memória (MB)", "Valor": _fmt_mem_mb(mem_mb)},
    ])

def _dtypes_summary(df: _pd.DataFrame) -> _pd.DataFrame:
    return (df.dtypes.astype(str)
              .value_counts()
              .rename_axis("dtype")
              .reset_index(name="cols"))

def _missing_top_with_dtype(df: _pd.DataFrame, top: int = 20) -> _pd.DataFrame:
    miss_cnt = df.isna().sum()
    miss_pct = df.isna().mean().mul(100)
    dtypes = df.dtypes.astype(str)
    out = (
        _pd.DataFrame({
            "Coluna": df.columns,
            "Tipo": [dtypes[c] for c in df.columns],
            "Faltantes": miss_cnt.values,
            "%Faltantes": miss_pct.values
        })
        .sort_values("%Faltantes", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
    # formatação bonita
    out["Faltantes"] = out["Faltantes"].map(lambda v: int(v))
    out["%Faltantes"] = out["%Faltantes"].map(lambda v: _fmt_auto(float(v), 2))
    return out

def _show_block(title: str, subtitle: str, df_display: _pd.DataFrame, accent: str = "#0ea5e9") -> None:
    display(_card(title, subtitle, accent=accent))
    try:
        display(df_display.style.set_properties(**{"font-size": "12px"}))
    except Exception:
        display(df_display)

# ---------- APIs "neat" para fontes e df final ----------
def show_source_overview_neat(name: str, path: _Path | str, df: _pd.DataFrame) -> None:
    path = _Path(path)
    _show_block(f"📥 Fonte: {name}", f"Arquivo: {path.name}", _overview_table_neat(df), accent="#0ea5e9")
    _show_block("🧬 Tipos (resumo)", "Contagem por dtype", _dtypes_summary(df), accent="#64748b")
    _show_block("🩺 Faltantes (top 20)", "Colunas com mais ausentes", _missing_top_with_dtype(df), accent="#f59e0b")

def show_df_summary_neat(df: _pd.DataFrame, label: str = "DF base") -> None:
    _show_block(f"🧾 Visão geral — {label}", "", _overview_table_neat(df), accent="#22c55e")
    _show_block(f"🧬 Tipos (resumo) — {label}", "Contagem por dtype", _dtypes_summary(df), accent="#64748b")
    _show_block(f"🩺 Faltantes (top 20) — {label}", "Após merges (se houver)", _missing_top_with_dtype(df), accent="#f59e0b")

# ---------- silenciador de logs do módulo ----------
@_contextmanager
def _quiet_utils_data_logger():
    try:
        logger  # usa o logger já definido no módulo
    except Exception:
        # fallback: cria logger compatível, se não existir
        import sys as _sys
        _lg = _logging.getLogger("utils_data")
        if not _lg.handlers:
            h = _logging.StreamHandler(_sys.stdout)
            h.setFormatter(_logging.Formatter("[%(levelname)s] %(message)s"))
            _lg.addHandler(h)
        globals()["logger"] = _lg  # injeta
    prev = logger.level
    try:
        logger.setLevel(_logging.WARNING)
        yield
    finally:
        logger.setLevel(prev)

# ---------- Runner + Render para Qualidade & Tipagem ----------
def run_quality_and_typing(df: _pd.DataFrame, config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Executa a etapa de Qualidade & Tipagem com logs silenciados.
    Retorna dict com:
      {
        "df": DataFrame final,
        "impacto": DataFrame Linhas/Colunas/Memória (antes/depois),
        "conversoes": cast_report filtrado (apenas mudanças reais) ou None,
        "dups": duplicatas (amostra) ou None,
        "dups_summary": resumo de duplicatas ou None
      }
    """
    import pandas as pd
    df_before_shape = df.shape
    mem_before = df.memory_usage(deep=True).sum() / (1024**2)

    with _quiet_utils_data_logger():
        if hasattr(globals().get("n1_quality_typing_dict", None), "__call__"):
            rep = n1_quality_typing_dict(df, config)
            df2 = rep["df"]
        else:
            df2, meta = n1_quality_typing(df, config)
            rep = meta if isinstance(meta, dict) else {"df": df2}

    mem_after  = df2.memory_usage(deep=True).sum() / (1024**2)
    delta_rows = df2.shape[0] - df_before_shape[0]
    delta_cols = df2.shape[1] - df_before_shape[1]
    delta_mem  = mem_after - mem_before

    impacto = pd.DataFrame([
        {"Métrica":"Linhas",  "Antes": int(df_before_shape[0]), "Depois": int(df2.shape[0]), "Δ": int(delta_rows)},
        {"Métrica":"Colunas", "Antes": int(df_before_shape[1]), "Depois": int(df2.shape[1]), "Δ": int(delta_cols)},
        {"Métrica":"Memória", "Antes": _fmt_mem_mb(mem_before), "Depois": _fmt_mem_mb(mem_after), "Δ": f"{delta_mem:+.2f} MB"},
    ])

    cast_report = rep.get("cast_report") if isinstance(rep, dict) else None
    conversoes = None
    if isinstance(cast_report, pd.DataFrame) and not cast_report.empty:
        conv = cast_report.copy()
        conv = conv[(conv["converted_non_null"] > 0) | (conv["dtype_after"] != "object")].copy()
        for c in ("converted_non_null","introduced_nans"):
            if c in conv.columns:
                conv[c] = conv[c].astype(int)
        if not conv.empty:
            conversoes = conv[["column","converted_non_null","introduced_nans","dtype_after"]]

    dups = rep.get("duplicates") if isinstance(rep, dict) else None
    dsum = rep.get("duplicates_summary") if isinstance(rep, dict) else None
    # normaliza vazios para None
    if isinstance(dups, pd.DataFrame) and dups.empty:
        dups = None
    if isinstance(dsum, pd.DataFrame) and dsum.empty:
        dsum = None

    return {
        "df": df2,
        "impacto": impacto,
        "conversoes": conversoes,
        "dups": dups,
        "dups_summary": dsum
    }

def render_quality_and_typing(result: Dict[str, Any]) -> None:
    """Exibe os cards organizados com base no retorno do run_quality_and_typing()."""
    display(_card("🧹 Qualidade & Tipagem", "Conversões, memória e checagens básicas"))
    display(result["impacto"])

    if result.get("conversoes") is not None:
        display(_card("🔢 Conversões aplicadas", "Somente o que realmente mudou"))
        display(result["conversoes"])

    if result.get("dups") is not None:
        display(_card("🔁 Duplicatas detectadas", "Amostra"))
        display(result["dups"].head(10))
        if result.get("dups_summary") is not None:
            display(result["dups_summary"].head(20))
    else:
        display(_card("✅ Sem duplicidades", "Nenhuma chave duplicada detectada"))

    nota = (
        "<b>Notas rápidas:</b><br>"
        "• Valores “introduced_nans” indicam entradas não parseáveis (ex.: strings vazias).<br>"
        "• Você pode imputar/filtrar esses casos nas próximas etapas."
    )
    display(HTML(f"<div style='color:#6b7280;font-size:12px'>{nota}</div>"))
# ======================= FIM DO PATCH =======================

def suggest_categorical_candidates(
    df,
    max_unique_ratio: float = 0.5,
    max_unique_count: int = 50,
    include_numeric_small: bool = True,
):
    """
    Sugere colunas candidatas à padronização categórica com base em heurísticas:
    - dtypes texto/categoria/bool sempre entram
    - numéricas com poucos valores únicos entram se include_numeric_small=True
    - calcula cardinalidade, % único e exemplos

    Retorna DataFrame com:
      column, dtype, n_unique, pct_unique, sample_values, suspected, reasons
    """
    import pandas as pd
    import numpy as np

    rows = []
    n_rows = max(1, len(df))
    lower_yes = {"yes","y","sim","s","true","t","1"}
    lower_no  = {"no","n","nao","não","false","f","0"}
    service_phrases = {"no internet service", "no phone service"}

    for c in df.columns:
        s = df[c]
        dt = str(s.dtype)
        try:
            nun = int(s.nunique(dropna=True))
        except Exception:
            nun = int(pd.Series(s).nunique(dropna=True))
        pct = float(nun) / float(n_rows) if n_rows else 0.0

        # dtype flags
        is_textual = (dt == "object") or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s)
        is_boolish = pd.api.types.is_bool_dtype(s)
        is_small_numeric = (pd.api.types.is_numeric_dtype(s) and nun <= 10) if include_numeric_small else False

        suspected = bool(is_textual or is_boolish or is_small_numeric)
        reasons = []
        if is_textual: reasons.append("texto/categoria")
        if is_boolish: reasons.append("booleano")
        if is_small_numeric: reasons.append("numérico baixa cardinalidade")

        # sinaliza padrões binários
        sample = pd.Series(s.dropna().astype(str).head(200)).str.strip().str.lower()
        uniq_sample = set(pd.Series(s.dropna().astype(str).str.strip().str.lower().unique()[:50]))
        if len(uniq_sample & lower_yes) > 0 or len(uniq_sample & lower_no) > 0:
            reasons.append("binário (yes/no)")
            suspected = True
        if len(uniq_sample & service_phrases) > 0:
            reasons.append("frases de serviço (mapear p/ 'No')")
            suspected = True

        # coleta exemplos
        top_vals = (
            s.astype(str)
             .str.slice(0, 60)
             .value_counts(dropna=True)
             .head(5)
             .index.tolist()
        )
        rows.append({
            "column": c,
            "dtype": dt,
            "n_unique": nun,
            "pct_unique": round(pct, 4),
            "sample_values": top_vals,
            "suspected": suspected,
            "reasons": ", ".join(reasons) if reasons else "",
        })

    out = pd.DataFrame(rows).sort_values(
        ["suspected", "n_unique"], ascending=[False, True]
    ).reset_index(drop=True)
    return out


def run_categorical_normalization(df, cfg, report_path=None, silence_logs=True):
    """
    Executa a padronização categórica com 'normalize_categories' (suporta API avançada e fallback).
    Retorna:
      {
        "df": df_norm,
        "report": cat_norm_report (DataFrame),
        "impacto": DataFrame Linhas/Colunas/Memória (antes/depois/Δ),
        "_details": {...}
      }
    """
    import logging
    import pandas as pd
    from pathlib import Path as _Path

    _logger = globals().get("logger") or logging.getLogger(__name__)
    prev = _logger.level
    try:
        if silence_logs:
            _logger.setLevel(logging.WARNING)

        before_shape = df.shape
        before_mem = df.memory_usage(deep=True).sum() / (1024**2)

        # tenta API avançada
        cat_norm_report = None
        used_fallback = False
        try:
            df_norm, cat_norm_report = normalize_categories(  # type: ignore
                df,
                cfg=cfg,
                report_path=report_path
            )
        except TypeError:
            # fallback para assinatura simples
            used_fallback = True
            text_cols = [c for c in df.columns if (df[c].dtype == "object") or pd.api.types.is_string_dtype(df[c])]
            exclude = set((cfg or {}).get("exclude", []))
            target_cols = [c for c in text_cols if c not in exclude]
            df_before = df.copy()

            df_norm = normalize_categories(  # type: ignore
                df,
                cols=target_cols,
                case=(cfg or {}).get("case", "lower"),
                trim=(cfg or {}).get("trim", True),
                strip_accents=(cfg or {}).get("strip_accents", True),
            )
            if isinstance(df_norm, tuple):
                df_norm = df_norm[0]

            # constrói relatório mínimo
            changes = []
            for c in target_cols:
                changed = (df_before[c].astype(str) != df_norm[c].astype(str)).sum()
                if changed > 0:
                    changes.append({"column": c, "changed": int(changed)})
            import pandas as pd
            cat_norm_report = pd.DataFrame(changes).sort_values("changed", ascending=False).reset_index(drop=True)

        # persistência opcional
        if report_path is not None and isinstance(cat_norm_report, pd.DataFrame):
            try:
                _p = _Path(report_path)
                _p.parent.mkdir(parents=True, exist_ok=True)
                cat_norm_report.to_csv(_p, index=False, encoding="utf-8")
                # registra em manifest, se o util existir
                if "save_report_df" in globals():
                    try:
                        save_report_df(cat_norm_report, _p)  # type: ignore
                    except Exception:
                        pass
            except Exception:
                pass

        # impacto
        after_mem = df_norm.memory_usage(deep=True).sum() / (1024**2)
        delta_rows = df_norm.shape[0] - before_shape[0]
        delta_cols = df_norm.shape[1] - before_shape[1]
        delta_mem  = after_mem - before_mem

        def _fmt_mem(x): return f"{x:.2f} MB"
        impacto = pd.DataFrame([
            {"Métrica":"Linhas",  "Antes": int(before_shape[0]), "Depois": int(df_norm.shape[0]), "Δ": int(delta_rows)},
            {"Métrica":"Colunas", "Antes": int(before_shape[1]), "Depois": int(df_norm.shape[1]), "Δ": int(delta_cols)},
            {"Métrica":"Memória", "Antes": _fmt_mem(before_mem), "Depois": _fmt_mem(after_mem), "Δ": f"{delta_mem:+.2f} MB"},
        ])

        return {
            "df": df_norm,
            "report": cat_norm_report,
            "impacto": impacto,
            "_details": {
                "used_fallback": used_fallback,
                "before_shape": before_shape,
                "after_shape": df_norm.shape,
                "mem_before": before_mem,
                "mem_after": after_mem,
            }
        }
    finally:
        if silence_logs:
            _logger.setLevel(prev)


def render_categorical_normalization(result, report_head: int = 20):
    """
    Renderiza cartões HTML e o relatório gerado por `run_categorical_normalization`.
    """
    from IPython.display import display, HTML
    import pandas as pd

    def _card(title, subtitle=""):
        return HTML(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid #7c3aed;
                    border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
          <div style="font-weight:700;font-size:16px">{title}</div>
          <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
        </div>
        """)

    display(_card("🏷️ Padronização Categórica", "Normalização de texto, mapeamentos globais e ajustes por coluna"))
    impacto = result.get("impacto")
    if isinstance(impacto, pd.DataFrame) and not impacto.empty:
        display(impacto)

    rep = result.get("report")
    if isinstance(rep, pd.DataFrame) and not rep.empty:
        display(_card("📑 Relatório (top mudanças)", "Primeiras linhas do relatório de normalização"))
        display(rep.head(report_head))
    else:
        display(_card("✅ Nenhuma mudança significativa", "Colunas categóricas já estavam padronizadas"))

    used_fb = result.get("_details", {}).get("used_fallback", False)
    rodape = "Modo: função avançada" if not used_fb else "Modo: fallback simples"
    display(HTML(f"<div style='color:#6b7280;font-size:12px'>"
                 f"{rodape}. Artefatos gravados quando 'report_path' é fornecido.</div>"))
# ============================================================================


def render_categorical_candidates(
    df,
    cand=None,
    max_unique_ratio: float = 0.5,
    max_unique_count: int = 50,
    include_numeric_small: bool = True,
    base_dir=None,
    top_n: int = 30,
    head_bin: int = 20,
    head_service: int = 20,
):
    """
    Renderiza cards organizados para candidatos de padronização categórica.
    - Se `cand` for None, chama `suggest_categorical_candidates` com os limites fornecidos.
    - Se `base_dir` for um caminho válido, salva CSVs em base_dir/'categorical_candidates'.
    - Não altera nenhuma função existente; apenas usa utilitários já presentes.
    """
    import pandas as pd
    from IPython.display import display, HTML
    from pathlib import Path as _Path

    # 0) Geração de candidatos (se não fornecido)
    if cand is None:
        if "suggest_categorical_candidates" in globals():
            cand = suggest_categorical_candidates(
                df,
                max_unique_ratio=max_unique_ratio,
                max_unique_count=max_unique_count,
                include_numeric_small=include_numeric_small,
            )
        else:
            raise RuntimeError("suggest_categorical_candidates não está disponível em utils_data.")

    # 1) Preparos
    def _card(title, subtitle=""):
        return HTML(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid #7c3aed;
                    border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
          <div style="font-weight:700;font-size:16px">{title}</div>
          <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
        </div>
        """)

    cand_view = (cand
        .sort_values(["suspected", "n_unique"], ascending=[False, True])
        .reset_index(drop=True))

    # Destaques
    reasons_series = cand.get("reasons")
    if reasons_series is not None:
        binarios = cand[reasons_series.str.contains("binário", case=False, na=False)].copy()
        servicos = cand[reasons_series.str.contains("serviço", case=False, na=False)].copy()
    else:
        binarios = cand.iloc[0:0].copy()
        servicos = cand.iloc[0:0].copy()

    summary = pd.DataFrame([
        {"Métrica": "Total de colunas", "Valor": int(len(df.columns))},
        {"Métrica": "Candidatas (suspected=True)", "Valor": int(cand["suspected"].sum()) if "suspected" in cand.columns else 0},
        {"Métrica": "Binárias (sugeridas Yes/No)", "Valor": int(len(binarios))},
        {"Métrica": "Com frases de serviço", "Valor": int(len(servicos))},
    ])

    cols_base = ["column", "dtype", "n_unique", "pct_unique", "sample_values", "reasons"]

    # 2) Exibição
    display(_card("🏷️ Descoberta de Candidatos à Padronização Categórica",
                  "Heurísticas de cardinalidade, tipos e padrões textuais"))
    display(summary)

    if not cand_view.empty:
        display(_card("🔎 Top candidatos (prioridade)", "Ordenado por suspeita e baixa cardinalidade"))
        display(cand_view.loc[:, [c for c in cols_base if c in cand_view.columns]].head(top_n))

    if not binarios.empty:
        display(_card("✅ Provavelmente binárias (Yes/No)", "Bom ponto para mapear Yes/No no CAT_NORM_CFG"))
        display(binarios.loc[:, [c for c in cols_base if c in binarios.columns]].head(head_bin))

    if not servicos.empty:
        display(_card("📡 Frases de serviço", "Ex.: 'No internet service' → 'No' (use global_map)"))
        display(servicos.loc[:, [c for c in cols_base if c in servicos.columns]].head(head_service))

    # 3) Persistência opcional
    out_dir = None
    try:
        if base_dir is not None:
            out_dir = _Path(base_dir)
        elif "paths" in globals() and hasattr(globals().get("paths"), "reports_dir"):
            out_dir = _Path(globals()["paths"].reports_dir)  # type: ignore

        if out_dir is not None:
            out_dir = out_dir / "categorical_candidates"
            out_dir.mkdir(parents=True, exist_ok=True)
            cand_view.loc[:, [c for c in cols_base if c in cand_view.columns]].to_csv(out_dir / "candidatos_priorizados.csv", index=False, encoding="utf-8")
            if not binarios.empty:
                binarios.loc[:, [c for c in cols_base if c in binarios.columns]].to_csv(out_dir / "candidatos_binarios.csv", index=False, encoding="utf-8")
            if not servicos.empty:
                servicos.loc[:, [c for c in cols_base if c in servicos.columns]].to_csv(out_dir / "candidatos_frases_servico.csv", index=False, encoding="utf-8")
    except Exception:
        # Persistência é best-effort; a interface visual é prioritária
        pass


# === Renderizador da etapa de Tratamento de Valores Faltantes ===============
def render_missing_step(res, df):
    """
    Renderiza um resumo visual e auditável do tratamento de valores faltantes.

    Parameters
    ----------
    res : dict
        Resultado retornado por utils_data.handle_missing_step()
    df : pandas.DataFrame
        DataFrame resultante após a imputação
    """
    import pandas as pd
    from IPython.display import display, HTML

    def _card(title, subtitle=""):
        return HTML(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid #0ea5e9;
                    border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
          <div style="font-weight:700;font-size:16px">{title}</div>
          <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
        </div>
        """)

    # --- Cabeçalho
    display(_card("🧩 Tratamento de Valores Faltantes", "Imputação + flags de auditoria (_was_missing)"))

    before = res.get("before", pd.DataFrame())
    after  = res.get("after", pd.DataFrame())

    total_missing_before = int(before["missing_count"].sum()) if "missing_count" in before else 0
    cols_with_missing_before = int((before["missing_count"] > 0).sum()) if "missing_count" in before else 0

    # Flags criadas (colunas *_was_missing)
    flag_rows = after[after["column"].str.endswith("_was_missing")] if "column" in after else pd.DataFrame()
    flags_created = int(flag_rows.shape[0])

    summary = pd.DataFrame([
        {"Métrica": "Estratégia", "Valor": res.get("strategy", "—")},
        {"Métrica": "Linhas no df", "Valor": int(df.shape[0])},
        {"Métrica": "Colunas no df", "Valor": int(df.shape[1])},
        {"Métrica": "Total de valores faltantes (antes)", "Valor": total_missing_before},
        {"Métrica": "Colunas com faltantes (antes)", "Valor": cols_with_missing_before},
        {"Métrica": "Flags criadas (_was_missing)", "Valor": flags_created},
    ])

    display(summary)

    # --- Antes: colunas com faltantes
    top_before = (before.sort_values("missing_count", ascending=False)
                         .query("missing_count > 0") if "missing_count" in before else pd.DataFrame())

    if not top_before.empty:
        display(_card("🔎 Antes: colunas com faltantes (topo)", "Ordenado por quantidade de faltantes"))
        display(top_before.head(20))
    else:
        display(_card("✅ Antes: sem faltantes", "Nenhuma coluna possuía valores ausentes"))

    # --- Depois da imputação
    display(_card("🧪 Depois da imputação", "Verificação de faltantes e flags"))
    still_missing = (
        after[~after["column"].str.endswith("_was_missing")].query("missing_count > 0")
        if "column" in after and "missing_count" in after else pd.DataFrame()
    )
    if not still_missing.empty:
        display(HTML("<div style='color:#b91c1c;font-size:12px'>⚠️ Ainda existem faltantes nas colunas abaixo:</div>"))
        display(still_missing.sort_values("missing_count", ascending=False).head(20))
    else:
        display(HTML("<div style='color:#16a34a;font-size:12px'>Tudo ok: nenhuma coluna permanece com faltantes.</div>"))

    # --- Flags criadas
    if flags_created > 0:
        display(_card("🏳️ Flags de auditoria criadas", "Colunas *_was_missing adicionadas ao DataFrame"))
        display(flag_rows.sort_values("column").head(30))
# ============================================================================

# === Renderizador da etapa de Detecção de Outliers ===========================
def render_outlier_flags(out_info: dict, df=None, top_n: int = 20, title: str = "🚨 Detecção de Outliers"):
    """
    Exibe cards com resumo e ranking de flags de outlier criadas.
    - out_info: dict retornado por apply_outlier_flags(...)
      chaves esperadas (tolerante a ausência): created_flags, method, counts, summary_path
    - df: DataFrame (opcional) para calcular % de linhas afetadas
    - top_n: quantas flags exibir no ranking
    """
    import pandas as pd
    from IPython.display import display, HTML

    def _card(title, subtitle=""):
        return HTML(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid #ef4444;
                    border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
          <div style="font-weight:700;font-size:16px">{title}</div>
          <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
        </div>
        """)

    method = out_info.get("method", "—")
    created = int(out_info.get("created_flags", 0))
    counts  = out_info.get("counts") or {}
    nrows   = int(getattr(df, "shape", (0, 0))[0]) if df is not None else None

    # Cabeçalho
    display(_card(title, f"Método: {method} · Flags criadas: {created}"))

    # Resumo
    total_outliers = int(sum(int(v) for v in counts.values())) if counts else 0
    impacted_cols  = int(sum(1 for v in counts.values() if v and int(v) > 0))
    summary_rows = [
        {"Métrica": "Linhas no df", "Valor": nrows if nrows is not None else "—"},
        {"Métrica": "Flags criadas", "Valor": created},
        {"Métrica": "Colunas com outliers (>0)", "Valor": impacted_cols},
        {"Métrica": "Total de marcações de outlier", "Valor": total_outliers},
    ]
    summary = pd.DataFrame(summary_rows)
    display(summary)

    # Ranking (top_n)
    if counts:
        rank = (pd.Series(counts).astype(int)
                .sort_values(ascending=False)
                .rename("outliers")
                .to_frame())
        rank.index.name = "flag"
        if nrows:
            rank["pct"] = (rank["outliers"] / nrows * 100).round(2)
        display(_card("📈 Top flags por incidência", f"Exibindo até {top_n}"))
        cols = ["outliers"] + (["pct"] if "pct" in rank.columns else [])
        display(rank.head(top_n)[cols])
    else:
        display(HTML("<div style='color:#16a34a;font-size:12px'>Nenhuma flag foi criada.</div>"))

    # Caminho do relatório (se existir)
    summary_path = out_info.get("summary_path") or out_info.get("report_path")
    if summary_path:
        display(HTML(f"<div style='color:#6b7280;font-size:12px;margin-top:8px'>Resumo salvo em: <code>{summary_path}</code></div>"))
# =============================================================================

# === Encoders & Scalers · Orquestração + Render ===
def run_encoding_and_scaling(df: "pd.DataFrame",
                             config: "Mapping[str, Any] | None" = None) -> "tuple[pd.DataFrame, dict]":
    """
    Executa a etapa unificada de Codificação Categórica & Escalonamento Numérico,
    delegando para a função já existente `apply_encoding_and_scaling(df, config)`.

    Retorna:
        df_out: DataFrame transformado (códigos + escalas aplicadas)
        info:   dict com chaves usuais:
                - "summary": DataFrame resumo (se existir)
                - "encoded_cols": list[str]
                - "scaled_cols": list[str]
                - "artifacts_dir": str | Path (se existir)
    """
    import pandas as pd  # local import para evitar dependências globais

    cfg = config or {}
    try:
        df_out, info = apply_encoding_and_scaling(df, cfg)
    except TypeError:
        # Em algumas versões antigas a assinatura podia ser (df) apenas.
        # Fallback defensivo para não quebrar:
        df_out, info = apply_encoding_and_scaling(df)  # type: ignore

    # Normaliza campos obrigatórios do info
    info = info or {}
    info.setdefault("encoded_cols", info.get("encoded_cols", []))
    info.setdefault("scaled_cols", info.get("scaled_cols", []))

    # Constrói um resumo se a função base não forneceu
    if "summary" not in info or info.get("summary") is None:
        try:
            import pandas as pd
            enc = list(info.get("encoded_cols") or [])
            scl = list(info.get("scaled_cols") or [])
            rows = []
            if enc:
                rows.append({"step": "encoding", "count": len(enc), "example": ", ".join(enc[:5])})
            if scl:
                rows.append({"step": "scaling", "count": len(scl), "example": ", ".join(scl[:5])})
            info["summary"] = pd.DataFrame(rows)
        except Exception:
            pass

    return df_out, info


def render_encoding_and_scaling(info: "Mapping[str, Any]") -> None:
    """
    Renderiza um painel compacto com:
      - Card de título
      - Resumo (top 20 do 'summary', se existir)
      - Totais de colunas codificadas/escaladas e diretório de artefatos (se houver)
    """
    from IPython.display import display, HTML
    import pandas as pd

    def _card(title: str, subtitle: str = "") -> "HTML":
        return HTML(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid #3b82f6;
                    border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
          <div style="font-weight:700;font-size:16px">{title}</div>
          <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
        </div>
        """)

    display(_card("🔢 Codificação & Escalonamento",
                  "Transforma categóricas em numéricas e normaliza variáveis contínuas"))

    summary = info.get("summary", None)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        display(HTML("<b>📊 Resumo (top 20):</b>"))
        display(summary.head(20))

    enc_cols = list(info.get("encoded_cols") or [])
    scl_cols = list(info.get("scaled_cols") or [])
    artifacts = info.get("artifacts_dir", None)

    # Mensagens finais compactas
    msg = f"✅ Codificação aplicada em {len(enc_cols)} colunas | Escalonamento aplicado em {len(scl_cols)} colunas."
    display(HTML(f"<div style='margin-top:6px;color:#111827'>{msg}</div>"))
    if artifacts:
        display(HTML(f"<div style='color:#6b7280;font-size:12px'>🗂️ Artefatos salvos em: {artifacts}</div>"))

# === Target · Orquestração + Renderização ================================

def run_target_creation_and_summary(df: "pd.DataFrame",
                                    config: "Mapping[str, Any]",
                                    verbose: bool = True) -> "dict[str, Any]":
    """
    Orquestra a criação/validação do target usando ensure_target_from_config,
    e retorna um pacote de infos pronto para renderização.

    Retorna um dict com:
      - df: DataFrame (possivelmente atualizado)
      - target_name: str
      - class_map: dict com 'positive' e 'negative'
      - tgt_report: DataFrame (status, source, positive, negative)
      - counts: dict {classe: contagem}
      - total: int
      - pos_rate: float em [0,1]
      - status: str
      - source: str
    """
    df_out, target_name, class_map, tgt_report = ensure_target_from_config(
        df, config, verbose=verbose
    )

    # Extrai metadados do relatório (se existir)
    status = None
    source = None
    try:
        if tgt_report is not None and len(tgt_report) > 0:
            status = str(tgt_report.loc[0, "status"])
            source = str(tgt_report.loc[0, "source"])
    except Exception:
        pass

    # Contagens e taxa positiva
    import pandas as pd
    vc = df_out[target_name].value_counts(dropna=False)
    total = int(vc.sum())
    pos_label = (class_map or {}).get("positive", "yes")
    neg_label = (class_map or {}).get("negative", "no")
    pos = int(vc.get(pos_label, 0))
    neg = int(vc.get(neg_label, 0))
    pos_rate = (pos / total) if total else 0.0

    return {
        "df": df_out,
        "target_name": target_name,
        "class_map": class_map,
        "tgt_report": tgt_report,
        "counts": {k: int(v) for k, v in vc.to_dict().items()},
        "total": total,
        "pos_rate": pos_rate,
        "status": status,
        "source": source,
        "pos_label": pos_label,
        "neg_label": neg_label,
    }


def render_target_summary(info: "Mapping[str, Any]") -> None:
    """
    Renderiza um painel compacto e padronizado para a variável-alvo:
    - Card com status e fonte
    - Tabela de contagens e percentuais
    - Badge com taxa positiva e classes detectadas
    - Alerta de desbalanceamento extremo (opcional)
    """
    from IPython.display import display, HTML
    import pandas as pd

    def _card(title: str, subtitle: str = "") -> "HTML":
        return HTML(f"""
        <div style="border:1px solid #e5e7eb;border-left:6px solid #7c3aed;
                    border-radius:10px;padding:12px 14px;margin:12px 0;background:#fafafa">
          <div style="font-weight:700;font-size:16px">{title}</div>
          <div style="color:#6b7280;font-size:12px;margin-top:2px">{subtitle}</div>
        </div>
        """)

    tname = info.get("target_name", "target")
    status = info.get("status") or "—"
    source = info.get("source") or "—"
    pos_label = info.get("pos_label", "yes")
    neg_label = info.get("neg_label", "no")
    total = int(info.get("total", 0))
    pos_rate = float(info.get("pos_rate", 0.0))

    # Card topo
    display(_card("🎯 Variável-Alvo (target)",
                  f"Nome: <b>{tname}</b> • Status: <b>{status}</b> • Fonte: <b>{source}</b>"))

    # Tabela de contagens
    counts = info.get("counts", {}) or {}
    neg = int(counts.get(neg_label, 0))
    pos = int(counts.get(pos_label, 0))

    tbl = pd.DataFrame([
        {"classe": neg_label, "contagem": neg, "percentual": f"{(neg/total):.2%}" if total else "—"},
        {"classe": pos_label, "contagem": pos, "percentual": f"{(pos/total):.2%}" if total else "—"},
        {"classe": "TOTAL",   "contagem": total, "percentual": "100.00%" if total else "—"},
    ])
    display(tbl)

    # Badge e classes detectadas
    classes_list = list(counts.keys())
    display(HTML(f"""
    <div style="margin-top:8px;font-size:13px;color:#374151">
      <b>Taxa de {pos_label}:</b> {pos_rate:.2%} • 
      <b>Classes detectadas:</b> {classes_list}
    </div>
    """))

    # Alerta de desbalanceamento extremo
    if total and (pos_rate < 0.10 or pos_rate > 0.90):
        display(HTML(
            "<div style='color:#b91c1c;font-size:12px;margin-top:6px'>"
            "⚠️ Alvo fortemente desbalanceado — considere reamostragem/ponderação no N2."
            "</div>"
        ))


def normalize_target_labels_inplace(df, target_name, positive_aliases=None, negative_aliases=None):
    """
    Normaliza in-place os rótulos do target para 'yes'/'no' a partir de aliases comuns.
    """
    import pandas as pd

    if target_name not in df.columns:
        return

    s = df[target_name]

    pos_alias = set(map(str, (positive_aliases or {
        "yes","y","true","1","sim","positivo","pos","churn","churned"
    })))
    neg_alias = set(map(str, (negative_aliases or {
        "no","n","false","0","nao","não","negativo","neg","retained","stay"
    })))

    s_norm = s.astype(str).str.strip().str.lower()

    # tenta mapear direto
    mapped = s_norm.where(~s_norm.isin(pos_alias | neg_alias))
    mapped = mapped.mask(s_norm.isin(pos_alias), "yes")
    mapped = mapped.mask(s_norm.isin(neg_alias), "no")

    # tenta mapear booleanos/numéricos residuais
    mapped = mapped.fillna(
        s_norm.map({
            "1": "yes", "0": "no",
            "true": "yes", "false": "no"
        })
    )

    df[target_name] = mapped


def fix_target_then_summary(df, config, verbose=True):
    """
    Envolve ensure_target_from_config e, se as classes não forem reconhecidas,
    normaliza labels e tenta novamente.
    """
    df1, tname, cmap, rep = ensure_target_from_config(df, config, verbose=verbose)

    vc = df1[tname].value_counts(dropna=False)
    # se counts vazias para 'yes' e 'no', tenta normalizar e refazer o resumo
    need_fix = (("yes" not in vc.index) and ("no" not in vc.index)) or vc.sum() == 0

    if need_fix:
        normalize_target_labels_inplace(df1, tname)
    return run_target_creation_and_summary(df1, config, verbose=verbose)

# =============================================================================
# 🔧 Null handling com flag (aditivo e retro-compatível)
# =============================================================================
from typing import Optional, Sequence, Mapping, Tuple, Dict, Any
import pandas as _pd
import numpy as _np

def summarize_missing(df: _pd.DataFrame) -> _pd.DataFrame:
    """
    Retorna um resumo de valores nulos por coluna:
      column | missing_count | missing_pct | dtype
    """
    total = len(df)
    if total == 0:
        return _pd.DataFrame(columns=["column","missing_count","missing_pct","dtype"])
    miss = df.isna().sum().rename("missing_count").to_frame()
    miss["missing_pct"] = (miss["missing_count"] / total * 100).round(2)
    miss["column"] = miss.index
    miss["dtype"] = [str(df[c].dtype) for c in miss["column"]]
    miss = miss.loc[miss["missing_count"] > 0, ["column","missing_count","missing_pct","dtype"]]
    return miss.reset_index(drop=True)


def null_fill_with_flag(
    df: _pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    numeric_fill: float | int = 0,
    categorical_fill: str = "__MISSING__",
    flag_suffix: str = "_was_missing",
) -> Tuple[_pd.DataFrame, Dict[str, Any]]:
    """
    Preenche nulos nas colunas indicadas e cria flags <col>_was_missing (0/1).
    - Colunas numéricas recebem `numeric_fill`;
    - Colunas não-numéricas recebem `categorical_fill`.

    Retorna: (df_novo, meta)
      meta = {
        "filled_cols": [...],
        "flags_created": N,
        "before_summary": DataFrame,
        "after_summary": DataFrame
      }
    """
    out = df.copy()
    before = summarize_missing(out)

    if cols is None:
        cols = [c for c in out.columns if out[c].isna().any()]

    flags_created = 0
    filled_cols: list[str] = []

    for c in cols:
        if c not in out.columns:
            continue
        s = out[c]
        mask = s.isna()
        if not mask.any():
            continue

        # cria flag
        flag_col = f"{c}{flag_suffix}"
        out[flag_col] = mask.astype(int)
        flags_created += 1

        # aplica preenchimento conforme dtype
        if _pd.api.types.is_numeric_dtype(s):
            out[c] = s.fillna(numeric_fill)
        else:
            out[c] = s.fillna(categorical_fill)

        filled_cols.append(c)

    after = summarize_missing(out)
    meta: Dict[str, Any] = {
        "filled_cols": filled_cols,
        "flags_created": flags_created,
        "before_summary": before,
        "after_summary": after,
    }
    return out, meta


def null_fill_from_config(
    df: _pd.DataFrame,
    config: Mapping[str, Any],
    root: Optional[Path] = None
) -> Tuple[_pd.DataFrame, Dict[str, Any]]:
    """
    Lê config["null_fill_with_flag"] (se existir e enabled) e aplica null_fill_with_flag.
    Exemplo de config (defaults.json):
      "null_fill_with_flag": {
        "enabled": true,
        "numeric_fill": 0,
        "categorical_fill": "__MISSING__",
        "cols_numeric_zero": ["avg_charge_per_month"],
        "flag_suffix": "_was_missing",
        "report_relpath": "nulls/fill_summary.csv"
      }

    Retorna: (df, meta). Se o recurso estiver desabilitado/ausente, retorna df inalterado e meta vazio.
    """
    spec = (config or {}).get("null_fill_with_flag")
    if not spec or not bool(spec.get("enabled", False)):
        return df, {"enabled": False}

    numeric_fill = spec.get("numeric_fill", 0)
    categorical_fill = spec.get("categorical_fill", "__MISSING__")
    flag_suffix = spec.get("flag_suffix", "_was_missing")
    report_relpath = spec.get("report_relpath", "nulls/fill_summary.csv")

    # Se cols não vieram, aplicamos somente nas que têm nulos
    cols = spec.get("cols_numeric_zero") or spec.get("cols")  # compat
    if cols is None:
        cols = [c for c in df.columns if df[c].isna().any()]

    df2, meta = null_fill_with_flag(
        df,
        cols=cols,
        numeric_fill=numeric_fill,
        categorical_fill=categorical_fill,
        flag_suffix=flag_suffix,
    )
    meta["enabled"] = True

    # Persiste um resumo (antes/depois) em reports, se disponível infra de reports
    try:
        before = meta.get("before_summary")
        after = meta.get("after_summary")
        # monta um comparativo simples
        if isinstance(before, _pd.DataFrame) and isinstance(after, _pd.DataFrame):
            comp = before.merge(after, on=["column","dtype"], how="outer", suffixes=("_before","_after"))
            comp = comp.fillna(0)
            save_report_df(comp, report_relpath, root=root)
            meta["report_path"] = str((ensure_project_root() / "reports" / report_relpath).resolve())
    except Exception as e:
        logger.warning(f"[null_fill_from_config] falha ao persistir relatório: {e}")

    return df2, meta


def render_null_fill_report(meta: Dict[str, Any]) -> None:
    """
    Renderiza um card simples com o que foi preenchido e flags criadas.
    Usa os helpers de card já existentes no utils.
    """
    if not isinstance(meta, dict) or not meta.get("enabled", False):
        display(_card("🟦 Nulos (fill+flag)", "Desabilitado por configuração"))
        return

    title = "🟦 Nulos (fill+flag)"
    subtitle = "Preenchimento de nulos com colunas de flag"
    display(_card(title, subtitle))

    filled_cols = meta.get("filled_cols") or []
    flags_created = int(meta.get("flags_created", 0))
    print(f"• Colunas preenchidas: {len(filled_cols)}")
    if filled_cols:
        print("  - " + ", ".join(filled_cols[:12]) + (" ..." if len(filled_cols) > 12 else ""))
    print(f"• Flags criadas      : {flags_created}")

    if meta.get("report_path"):
        print(f"• Relatório (comparativo antes/depois) salvo em: {meta['report_path']}")

    # Exibe tabelas de antes/depois (se existirem)
    if isinstance(meta.get("before_summary"), _pd.DataFrame) and not meta["before_summary"].empty:
        display(_card("Antes do preenchimento", "Resumo de nulos"))
        display(meta["before_summary"])
    if isinstance(meta.get("after_summary"), _pd.DataFrame) and not meta["after_summary"].empty:
        display(_card("Depois do preenchimento", "Resumo de nulos"))
        display(meta["after_summary"])
    elif isinstance(meta.get("after_summary"), _pd.DataFrame):
        display(_card("Depois do preenchimento", "Sem nulos restantes nas colunas tratadas"))

# =============================================================================
# 🔧 Patches de qualidade: regras derivadas e transformações seguras
# =============================================================================
import numpy as _np
import pandas as _pd

def fix_avg_charge_zero_tenure(
    df: _pd.DataFrame,
    avg_col: str = "avg_charge_per_month",
    tenure_col: str = "tenure",
    create_flag: bool = True,
) -> _pd.DataFrame:
    """
    Regra derivada: se tenure == 0 e avg_charge_per_month é NaN -> setar 0 e flagar.
    """
    if avg_col not in df.columns or tenure_col not in df.columns:
        return df
    mask = (df[tenure_col] == 0) & (df[avg_col].isna())
    if mask.any():
        if create_flag:
            flag = f"{avg_col}_was_missing"
            if flag not in df.columns:
                df[flag] = 0
            df.loc[mask, flag] = 1
        df.loc[mask, avg_col] = 0
    return df


def signed_log1p_series(s: _pd.Series) -> _pd.Series:
    """
    Aplica log1p assinado: sign(x) * log1p(|x|) — não gera NaN para x <= -1.
    Mantém NaN somente onde s é NaN.
    """
    a = _np.asarray(s, dtype="float64")
    out = _np.sign(a) * _np.log1p(_np.abs(a))
    return _pd.Series(out, index=s.index, name=s.name)


def recompute_charge_gap_features(
    df: _pd.DataFrame,
    total_col: str = "TotalCharges",
    monthly_col: str = "MonthlyCharges",
    tenure_col: str = "tenure",
    gap_col: str = "charge_gap",
    gap_log1p_col: str = "charge_gap_log1p",
) -> _pd.DataFrame:
    """
    Recalcula charge_gap = TotalCharges - (MonthlyCharges * tenure)
    e charge_gap_log1p usando log assinado para evitar NaN de domínio.
    """
    missing = [c for c in [total_col, monthly_col, tenure_col] if c not in df.columns]
    if missing:
        return df  # silencioso; não existe base para o cálculo

    df[gap_col] = df[total_col] - (df[monthly_col] * df[tenure_col])
    df[gap_log1p_col] = signed_log1p_series(df[gap_col])
    return df


# -----------------------------------------------------------------------------
# 🔧 Preenchimento com flag a partir de config — adiciona suporte a lista explícita
# -----------------------------------------------------------------------------
def null_fill_from_config(
    df: _pd.DataFrame,
    config: dict,
    root: Path | None = None
):
    """
    Estende o comportamento: se 'cols_numeric_zero' existir, usa essa lista.
    Se não existir, varre somente colunas com NaN.
    """
    spec = (config or {}).get("null_fill_with_flag")
    if not spec or not bool(spec.get("enabled", False)):
        return df, {"enabled": False}

    numeric_fill = spec.get("numeric_fill", 0)
    categorical_fill = spec.get("categorical_fill", "__MISSING__")
    flag_suffix = spec.get("flag_suffix", "_was_missing")
    report_relpath = spec.get("report_relpath", "nulls/fill_summary.csv")

    # se a lista vier no config, usamos exatamente ela
    cols = spec.get("cols_numeric_zero") or spec.get("cols")
    if cols is None:
        cols = [c for c in df.columns if df[c].isna().any()]

    df2 = df.copy()

    # --- resumo antes
    total = len(df2)
    before = (
        df2.isna().sum().to_frame("missing_count")
        .assign(missing_pct=lambda x: (x["missing_count"] / total * 100).round(2))
        .query("missing_count > 0")
        .reset_index(names="column")
    )

    filled_cols = []
    flags_created = 0

    for c in cols:
        if c not in df2.columns:
            continue
        mask = df2[c].isna()
        if not mask.any():
            continue
        flag_col = f"{c}{flag_suffix}"
        if flag_col not in df2.columns:
            df2[flag_col] = 0
            flags_created += 1
        df2.loc[mask, flag_col] = 1
        if _pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].fillna(numeric_fill)
        else:
            df2[c] = df2[c].fillna(categorical_fill)
        filled_cols.append(c)

    # --- resumo depois
    after = (
        df2.isna().sum().to_frame("missing_count")
        .assign(missing_pct=lambda x: (x["missing_count"] / total * 100).round(2))
        .query("missing_count > 0")
        .reset_index(names="column")
    )

    # tenta salvar relatório via helper do projeto (se existir)
    try:
        comp = before.merge(after, on="column", how="outer", suffixes=("_before", "_after")).fillna(0)
        save_report_df(comp, report_relpath, root=root)  # sua função já existente
        report_path = str((ensure_project_root() / "reports" / report_relpath).resolve())
    except Exception:
        report_path = None

    meta = {
        "enabled": True,
        "filled_cols": filled_cols,
        "flags_created": flags_created,
        "before_summary": before,
        "after_summary": after,
        "report_path": report_path,
    }
    return df2, meta


# =============================================================================
# N2 • Bootstrap compacto + leitura e painel estilizado (aditivo, safe)
# =============================================================================
from pathlib import Path as _Path
from typing import Any as _Any, Dict as _Dict, Tuple as _Tuple
import json as _json
import pandas as _pd
import numpy as _np
import html as _html

try:
    # só existe em ambiente notebook
    from IPython.display import HTML as _HTML, display as _display
except Exception:
    _HTML = None
    def _display(*args, **kwargs):  # fallback para execução não-notebook
        pass

# ---------- utilidades internas ----------
def _read_table_auto(_path: _Path) -> _pd.DataFrame:
    """Leitura robusta para parquet/csv/xlsx."""
    suf = _path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return _pd.read_parquet(_path)
    if suf == ".csv":
        return _pd.read_csv(_path)
    if suf in (".xlsx", ".xls"):
        return _pd.read_excel(_path)
    raise ValueError(f"Extensão não suportada: {suf} ({_path})")

def _find_target_case(df: _pd.DataFrame, cfg: dict, fallback: str = "target") -> str:
    """Encontra a coluna alvo case-insensitive usando config (target.name/target_column)."""
    name = (cfg.get("target", {}) or {}).get("name", cfg.get("target_column", fallback))
    name = name or fallback
    mask = df.columns.str.lower() == str(name).lower()
    return df.columns[mask][0] if mask.any() else name

def _shorten_path(p: _Path | str, keep_parts: int = 4) -> str:
    """Abrevia caminho para exibição (…/<tail>)."""
    p = _Path(p)
    parts = list(p.parts)
    if len(parts) <= keep_parts:
        return str(p)
    tail = _Path(*parts[-keep_parts:])
    anchor = p.anchor if p.anchor else ""
    tail_str = str(tail)
    if anchor and tail_str.startswith(anchor):
        tail_str = tail_str[len(anchor):]
    return f"...{anchor}{tail_str}".replace("\\\\", "\\")

def _styler_hide_index(sty: _pd.io.formats.style.Styler) -> _pd.io.formats.style.Styler:
    try:
        return sty.hide(axis="index")  # pandas >= 2
    except Exception:
        try:
            return sty.hide_index()    # pandas < 2
        except Exception:
            return sty

def _to_html_table(df: _pd.DataFrame, caption: str | None = None) -> str:
    sty = (
        df.style
        .set_table_attributes('class="tbl"')
        .set_properties(**{"text-align": "right", "padding": "6px 10px"})
    )
    sty = _styler_hide_index(sty)
    html_tbl = sty.to_html()
    if caption:
        html_tbl = f'<div class="tbl-caption">{_html.escape(caption)}</div>' + html_tbl
    return html_tbl

def _badge(text: str, kind: str = "info") -> str:
    cls = {"ok": "ok", "warn": "warn", "info": "info"}.get(kind, "info")
    return f'<span class="badge {cls}">{_html.escape(text)}</span>'

def _bool_badge(flag: bool) -> str:
    return _badge("ON" if flag else "OFF", "ok" if flag else "warn")

# ---------- API pública p/ N2 ----------
def n2_bootstrap_and_load(project_root: _Path | None = None) -> _Dict[str, _Any]:
    """
    Bootstrap compacto do N2:
      - resolve PROJECT_ROOT
      - carrega config (defaults/local)
      - garante dirs (artifacts/reports/models)
      - resolve processed_path e lê df
      - encontra TARGET_COL (case-insensitive)
      - sumariza tipos via summarize_columns
    Retorna dict com chaves: project_root, config, artifacts_dir, reports_dir, models_dir,
                             processed_path, df, target_col, num_cols, cat_cols, other_cols
    """
    # raiz
    try:
        root = project_root or ensure_project_root()
    except Exception:
        root = project_root or get_project_root()

    # config
    cfg = load_config(root / "config" / "defaults.json", root / "config" / "local.json")

    # dirs
    artifacts_dir, reports_dir, models_dir = ensure_dirs(cfg)

    # processed
    processed_path = discover_processed_path(cfg)
    if not processed_path.is_absolute():
        processed_path = (root / processed_path).resolve()
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed não encontrado: {processed_path}")

    # leitura
    df = _read_table_auto(processed_path)

    # target e tipos
    target_col = _find_target_case(df, cfg, fallback="target")
    if target_col not in df.columns:
        raise AssertionError(f"Target '{target_col}' não encontrada no dataset.")
    num_cols, cat_cols, other_cols = summarize_columns(df)

    return {
        "project_root": root,
        "config": cfg,
        "artifacts_dir": artifacts_dir,
        "reports_dir": reports_dir,
        "models_dir": models_dir,
        "processed_path": processed_path,
        "df": df,
        "target_col": target_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "other_cols": other_cols,
    }

def render_n2_status_panel_light(
    project_root: _Path,
    processed_path: _Path,
    df: _pd.DataFrame,
    target_name: str,
    num_cols: list[str],
    cat_cols: list[str],
    other_cols: list[str],
    test_size: float,
    random_state: int,
    scale_numeric: bool,
    target_counts: _pd.Series | None = None,
    target_pct: _pd.Series | None = None,
    keep_path_parts: int = 4,
):
    """Renderiza o painel limpo do N2 (paleta Aqua/Roxo, fonte maior, caminho abreviado)."""
    if _HTML is None:
        return  # ambiente sem notebook

    # métricas
    n_rows, n_cols = df.shape
    mem = f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
    null_total = int(df.isna().sum().sum())
    null_any_cols = int((df.isna().sum() > 0).sum())

    # caminhos abreviados
    fmt_root = _html.escape(_shorten_path(project_root, keep_parts=keep_path_parts))
    fmt_file = _html.escape(_shorten_path(processed_path, keep_parts=keep_path_parts))
    fmt_fmt  = _html.escape(processed_path.suffix.lower())

    # distribuição do target
    if target_counts is None or target_pct is None:
        _counts = df[target_name].value_counts(dropna=False)
        _pct = (_counts / len(df) * 100).round(2)
    else:
        _counts, _pct = target_counts, target_pct
    tgt_df = (
        _pd.DataFrame({"value": _counts.index.astype(str),
                       "count": _counts.values,
                       "pct": _pct.values})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    tgt_df["pct"] = tgt_df["pct"].map(lambda x: f"{x:.2f}%")
    tgt_html = _to_html_table(tgt_df, caption=f"Distribuição de '{target_name}'")

    params_df = _pd.DataFrame({
        "parâmetro": ["test_size", "random_state", "scale_numeric"],
        "valor": [test_size, random_state, "ON" if scale_numeric else "OFF"]
    })
    params_html = _to_html_table(params_df, caption="Parâmetros (pré-split)")

    css = """
    <style>
    .n2-wrap { 
      --bg:#f7f9fc; --fg:#0f172a; --muted:#6b7280; --card:#ffffff; --edge:#e5e7eb;
      --aqua:#22d3ee; --aqua-soft:rgba(34,211,238,.15);
      --purple:#8b5cf6; --purple-soft:rgba(139,92,246,.12);
      --ok:#10b981; --warn:#f59e0b; --info:#6366f1;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color: var(--fg);
      background: var(--bg);
      border: 1px solid var(--edge);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 25px rgba(2,6,23,.06);
      font-size: 15px;
    }
    .n2-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
    .n2-title { font-size: 22px; font-weight: 800; letter-spacing: .2px; }
    .n2-sub { color: var(--muted); font-size: 13px; }
    .grid { display:grid; grid-template-columns: repeat(12, 1fr); gap:14px; }
    .card { background: var(--card); border:1px solid var(--edge); border-radius:14px; padding:14px; box-shadow: 0 3px 8px rgba(2,6,23,.04); }
    .span-4 { grid-column: span 4; } .span-6 { grid-column: span 6; } .span-12 { grid-column: span 12; }
    .knum { font-weight:800; font-size: 24px; }
    .muted { color: var(--muted); font-size: 13px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 14px; }
    .tbl { border-collapse: collapse; width:100%; font-size: 14px; background: #ffffff; color: var(--fg); border:1px solid var(--edge); border-radius:12px; overflow:hidden; }
    .tbl th, .tbl td { border-bottom: 1px solid #eef2f7; padding: 10px 12px; }
    .tbl th { text-align:left; color:#374151; background: linear-gradient(0deg, #fafcff, #f3f6fb); }
    .tbl-caption { font-size:13px; color: var(--muted); margin: 6px 0 6px 2px; }
    .accent { box-shadow: inset 0 0 0 1px var(--edge), 0 0 0 2px var(--aqua-soft); }
    .accent-purple { box-shadow: inset 0 0 0 1px var(--edge), 0 0 0 2px var(--purple-soft); }
    .badge { border-radius: 999px; padding: 3px 10px; font-size: 12px; border:1px solid var(--edge); }
    .badge.ok { background: rgba(16,185,129,.14); color:#065f46; border-color: rgba(16,185,129,.35);}
    .badge.warn { background: rgba(245,158,11,.14); color:#7c4a03; border-color: rgba(245,158,11,.35);}
    .badge.info { background: var(--purple-soft); color:#4c1d95; border-color: rgba(139,92,246,.35);}
    </style>
    """

    html_panel = f"""
    <div class="n2-wrap accent">
      <div class="n2-head">
        <div class="n2-title">N2 — Resumo do Bootstrap e Leitura do Dataset</div>
        <div class="n2-sub mono">{fmt_fmt} · {n_rows}×{n_cols} · {mem}</div>
      </div>
      <div class="grid">
        <div class="card span-6 accent">
          <div class="muted">Projeto</div>
          <div class="mono">{fmt_root}</div>
          <div class="muted" style="margin-top:10px;">Arquivo lido</div>
          <div class="mono">{fmt_file}</div>
        </div>

        <div class="card span-6 accent">
          <div class="muted">Dimensão</div>
          <div class="knum">{n_rows} linhas <span class="muted">×</span> {n_cols} colunas</div>
          <div class="muted" style="margin-top:6px;">Memória estimada: <span class="pill aqua">{mem}</span></div>
          <div class="muted" style="margin-top:6px;">Nulos: <span class="pill aqua">{null_total} células · {null_any_cols} col(s)</span></div>
        </div>

        <div class="card span-4 accent">
          <div class="muted">Tipos de colunas</div>
          <div class="muted">num: <span class="pill aqua">{len(num_cols)}</span></div>
          <div class="muted">cat: <span class="pill aqua">{len(cat_cols)}</span></div>
          <div class="muted">other: <span class="pill aqua">{len(other_cols)}</span></div>
        </div>

        <div class="card span-4 accent-purple">
          <div class="muted">Target</div>
          <div class="knum">{_html.escape(target_name)}</div>
          <div class="muted">Valores únicos: {df[target_name].nunique()}</div>
        </div>

        <div class="card span-4 accent">
          <div class="muted">Parâmetros (pré-split)</div>
          <div class="muted">test_size · <span class="pill aqua">{test_size}</span></div>
          <div class="muted" style="margin-top:6px;">random_state · <span class="pill aqua">{random_state}</span></div>
          <div class="muted" style="margin-top:6px;">scale_numeric · <span class="pill aqua">{'ON' if scale_numeric else 'OFF'}</span></div>
        </div>

        <div class="card span-6 accent">{tgt_html}</div>
        <div class="card span-6 accent">{params_html}</div>

        <div class="card span-12 muted" style="text-align:right;">
          Gerado automaticamente • N2 • {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
      </div>
    </div>
    """
    _display(_HTML(css + html_panel))


def ensure_utils_import() -> "Path":
    """
    Garante que a raiz do projeto e o pacote utils/ estejam acessíveis no sys.path.
    Retorna o PROJECT_ROOT detectado.

    Uso típico no notebook (N1/N2/N3):

    >>> from utils.utils_data import ensure_utils_import
    >>> PROJECT_ROOT = ensure_utils_import()
    >>> import utils.utils_data as ud  # já deve funcionar sem erro de módulo

    Esta função é não-intrusiva: não altera comportamentos existentes, apenas
    ajusta o sys.path e cria utils/__init__.py se necessário.
    """
    import sys as _sys
    from pathlib import Path as _Path

    def _find_up(relative: str, start: "Path | None" = None) -> "Path | None":
        start = start or _Path.cwd()
        rel = _Path(relative)
        for base in (start, *start.parents):
            cand = base / rel
            if cand.exists():
                return cand
        return None

    _cfg = _find_up("config/defaults.json")
    if _cfg is None:
        raise FileNotFoundError("config/defaults.json não encontrado. Confirme a estrutura do projeto.")

    project_root = _cfg.parent.parent.resolve()
    utils_dir = project_root / "utils"
    if not utils_dir.exists():
        raise ModuleNotFoundError(f"Pasta 'utils' não encontrada em {project_root}")

    init_file = utils_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    root_str = str(project_root)
    if root_str not in _sys.path:
        _sys.path.insert(0, root_str)

    return project_root

# ============================================================================
# N2 — Bootstrap Reporting Helpers (aditivo, não intrusivo)
# ============================================================================
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as _np
import pandas as _pd
from pathlib import Path as _Path


@dataclass
class N2Params:
    """
    Parametrização básica usada no resumo do bootstrap do N2.

    Attributes
    ----------
    test_size : float
        Proporção do conjunto de teste (ex.: 0.2 = 20%).
    random_state : int
        Semente de aleatoriedade usada em train_test_split/modelos.
    scale_numeric : bool
        Indica se as variáveis numéricas serão escalonadas no pré-processamento.
    """
    test_size: float = 0.2
    random_state: int = 42
    scale_numeric: bool = True


def _n2_fmt_mem_mb(df: _pd.DataFrame) -> str:
    """Retorna o consumo de memória do DataFrame em MB (string formatada)."""
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    return f"{mb:.2f} MB"


def _n2_dtype_overview(df: _pd.DataFrame) -> Dict[str, int]:
    """
    Gera um resumo simples dos tipos:
    - num: colunas numéricas
    - cat: todas as demais (object, category, bool, datetime, etc.)
    - other: reservado (0 nesta heurística).
    """
    num = df.select_dtypes(include=[_np.number]).shape[1]
    cat = df.shape[1] - num
    return {"num": num, "cat": cat, "other": 0}


def _n2_null_overview(df: _pd.DataFrame) -> Tuple[int, int, int]:
    """
    Retorna (null_cells, total_cells, cols_with_nulls).
    """
    total_cells = df.size
    null_cells = int(df.isna().sum().sum())
    cols_with_nulls = int(df.isna().any().sum())
    return null_cells, total_cells, cols_with_nulls


def _n2_class_balance(y: _pd.Series) -> _pd.DataFrame:
    """
    DataFrame com contagem e percentual da variável alvo (y).
    """
    vc = y.value_counts(dropna=False)
    pct = (vc / vc.sum() * 100).round(2)
    return (
        _pd.DataFrame({"count": vc, "pct": pct})
        .rename_axis(y.name or "target")
    )


def _n2_params_table(params: N2Params) -> _pd.DataFrame:
    """
    Tabela simples com os parâmetros principais (test_size, random_state, scale_numeric).
    """
    return _pd.DataFrame(
        {
            "parâmetro": ["test_size", "random_state", "scale_numeric"],
            "valor": [
                float(params.test_size),
                int(params.random_state),
                "ON" if params.scale_numeric else "OFF",
            ],
        }
    )


def n2_bootstrap_log_and_report(
    df: _pd.DataFrame,
    X: _pd.DataFrame,
    y: _pd.Series,
    target_col: str,
    project_root: Optional[str] = None,
    processed_file_path: Optional[str] = None,
    params: Optional[N2Params] = None,
    print_log: bool = True,
) -> Dict[str, Any]:
    """
    Gera logs e um resumo programático do bootstrap do N2.

    Esta função NÃO altera o DataFrame. Ela apenas:
      - Imprime logs de diagnóstico (PROJECT_ROOT, arquivo processado, target, shapes, balanceamento).
      - Calcula estatísticas básicas sobre tipos, nulos, memória.
      - Retorna um dicionário com artefatos para uso no notebook (markdown + tabelas).

    Parameters
    ----------
    df : DataFrame
        Dataset completo (incluindo a coluna alvo).
    X : DataFrame
        Features (df sem a coluna alvo).
    y : Series
        Variável alvo.
    target_col : str
        Nome da coluna alvo em df.
    project_root : str, opcional
        Caminho da raiz do projeto (usado apenas para logging / contexto).
    processed_file_path : str, opcional
        Caminho do arquivo processado (data/processed/processed.parquet).
    params : N2Params, opcional
        Parâmetros principais (test_size, random_state, scale_numeric).
    print_log : bool, default True
        Se True, imprime mensagens no estilo [INFO]/[CHECK].

    Returns
    -------
    Dict[str, Any]
        {
          "markdown_header": str,
          "dtype_counts": dict,
          "class_balance": DataFrame,
          "params_table": DataFrame,
          "meta": dict
        }
    """
    params = params or N2Params()

    # Resumos básicos
    dtype_counts = _n2_dtype_overview(df)
    null_cells, total_cells, cols_with_nulls = _n2_null_overview(df)
    mem_est = _n2_fmt_mem_mb(df)
    rows, cols = df.shape

    # Logs estilo console
    if print_log:
        if project_root:
            print(f"[INFO] PROJECT_ROOT: {project_root}")
        if project_root and processed_file_path:
            print(f"[INFO] Projeto:        {project_root}")
            print(f"[INFO] Processed file: {processed_file_path}")
        print(f"[INFO] Target column:  {target_col}")
        print(f"[INFO] Shapes -> X: {X.shape} | y: {y.shape}")
        print(f"[CHECK] Target nulos={int(y.isna().sum())} | classes únicas={y.nunique()}")
        # Distribuição da target (sem quebrar em caso extremo)
        cb = _n2_class_balance(y)
        print(cb)

    # Artefatos para uso no notebook
    class_balance = _n2_class_balance(y)
    params_table = _n2_params_table(params)

    # Texto-resumo tipo "cabeçalho" (pode ser usado em Markdown/painel)
    fmt_root = f"...{str(project_root)[-60:]}" if project_root else ""
    fmt_file = f"...{str(processed_file_path)[-60:]}" if processed_file_path else ""
    header_lines = [
        "N2 — Resumo do Bootstrap e Leitura do Dataset",
        f".parquet · {rows}×{cols} · {mem_est}",
    ]
    if project_root:
        header_lines.append("Projeto")
        header_lines.append(fmt_root)
    if processed_file_path:
        header_lines.append("Arquivo lido")
        header_lines.append(fmt_file)
    header_lines.append(f"Dimensão\n{rows} linhas × {cols} colunas")
    header_lines.append(f"Memória estimada: {mem_est}")
    header_lines.append(f"Nulos: {null_cells} células · {cols_with_nulls} col(s)")
    header_lines.append(
        "Tipos de colunas\n"
        f"num: {dtype_counts['num']}\n"
        f"cat: {dtype_counts['cat']}\n"
        f"other: {dtype_counts['other']}"
    )
    header_lines.append("Target")
    header_lines.append(target_col)
    header_lines.append(f"Valores únicos: {y.nunique()}")
    header_lines.append(
        "Parâmetros (pré-split)\n"
        f"test_size · {params.test_size}\n"
        f"random_state · {params.random_state}\n"
        f"scale_numeric · {'ON' if params.scale_numeric else 'OFF'}"
    )
    markdown_header = "\n\n".join(header_lines)

    meta = {
        "rows": rows,
        "cols": cols,
        "mem_estimate": mem_est,
        "null_cells": null_cells,
        "null_columns": cols_with_nulls,
        "dtype_counts": dtype_counts,
        "total_cells": total_cells,
    }

    return {
        "markdown_header": markdown_header,
        "dtype_counts": dtype_counts,
        "class_balance": class_balance,
        "params_table": params_table,
        "meta": meta,
    }


# ============================================================================
# N2 — Bootstrap compacto e painel de status (para notebook 02)
# ============================================================================

from typing import Dict, Any


def n2_bootstrap_context(config: dict | None = None) -> Dict[str, Any]:
    """
    Bootstrap padrão do N2: resolve raiz, carrega config, garante diretórios,
    descobre o dataset processado, lê o DataFrame e retorna um contexto completo.

    Se `config` for None, `load_config()` é chamado internamente.

    Returns
    -------
    dict
        {
          "project_root": Path,
          "cfg": dict,
          "artifacts_dir": Path,
          "reports_dir": Path,
          "models_dir": Path,
          "processed_path": Path,
          "df": DataFrame,
          "X": DataFrame,
          "y": Series,
          "target_col": str,
          "num_cols": list[str],
          "cat_cols": list[str],
          "other_cols": list[str],
          "test_size": float,
          "random_state": int,
          "scale_numeric": bool,
        }
    """
    from pathlib import Path as _Path
    import pandas as _pd
    import numpy as _np

    # 1) Raiz, config e dirs
    project_root = get_project_root()
    if config is None:
        config = load_config()
    artifacts_dir, reports_dir, models_dir = ensure_dirs(config)

    # 2) Descobre caminho do processed
    processed_path = discover_processed_path(config)

    # 3) Leitura robusta do dataset
    suffix = processed_path.suffix.lower()
    try:
        if suffix in (".parquet", ".pq"):
            try:
                df = _pd.read_parquet(processed_path, engine="pyarrow")
            except Exception:
                df = _pd.read_parquet(processed_path, engine="fastparquet")
        elif suffix == ".csv":
            df = _pd.read_csv(processed_path, low_memory=False, encoding="utf-8")
        elif suffix in (".xlsx", ".xls"):
            df = _pd.read_excel(processed_path)
        else:
            raise ValueError(f"Extensão não suportada: {suffix}")
    except Exception as e:
        raise RuntimeError(
            f"Falha ao ler '{processed_path.name}': {type(e).__name__}: {e}"
        )

    # 4) Determinação tolerante da coluna alvo
    cfg_target = (
        (config.get("target_column"))
        or (config.get("target", {}) or {}).get("name")
        or "target"
    )

    cols_lower = {c.lower(): c for c in df.columns}
    key = str(cfg_target).lower()
    if key not in cols_lower:
        raise KeyError(
            f"Target '{cfg_target}' não encontrada no dataset. "
            f"Defina corretamente 'target_column' (ou target.name) em config/defaults.json."
        )
    target_col = cols_lower[key]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5) Tipos de coluna (tenta usar summarize_columns; se falhar, faz fallback)
    try:
        num_cols, cat_cols, other_cols = summarize_columns(df)
    except Exception:
        num_cols = df.select_dtypes(include=[_np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        other_cols = []

    # 6) Parâmetros de split
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    scale_numeric = bool(config.get("scale_numeric", True))

    return {
        "project_root": project_root,
        "cfg": config,
        "artifacts_dir": artifacts_dir,
        "reports_dir": reports_dir,
        "models_dir": models_dir,
        "processed_path": processed_path,
        "df": df,
        "X": X,
        "y": y,
        "target_col": target_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "other_cols": other_cols,
        "test_size": test_size,
        "random_state": random_state,
        "scale_numeric": scale_numeric,
    }


def n2_render_status_panel(ctx: Dict[str, Any], keep_path_parts: int = 4) -> None:
    """
    Renderiza um painel HTML com o resumo do dataset do N2, seguindo o tema
    escuro com acentos aqua/roxo (compatível com o formulário de hiperparâmetros).

    Parameters
    ----------
    ctx : dict
        Contexto retornado por `n2_bootstrap_context`.
    keep_path_parts : int, default 4
        Quantidade de partes finais do caminho a mostrar (para encurtar o path).
    """
    from IPython.display import HTML, display as _display
    import pandas as _pd
    import html as _html
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    import numpy as _np  # noqa: F401  (mantido caso queira usar depois)

    df = ctx["df"]
    target_name = ctx["target_col"]
    project_root = _Path(ctx["project_root"])
    processed_path = _Path(ctx["processed_path"])
    num_cols = ctx["num_cols"]
    cat_cols = ctx["cat_cols"]
    other_cols = ctx["other_cols"]
    test_size = ctx["test_size"]
    random_state = ctx["random_state"]
    scale_numeric = ctx["scale_numeric"]

    # ---------- helpers internos ----------
    def _fmt_bytes(n_bytes: int) -> str:
        return f"{n_bytes / (1024**2):.2f} MB"

    def _shorten_path(p: _Path | str, keep_parts: int = 4) -> str:
        p = _Path(p)
        parts = list(p.parts)
        if len(parts) <= keep_parts:
            return str(p)
        tail = _Path(*parts[-keep_parts:])
        root = p.anchor
        if root and not str(tail).startswith(root):
            return f"...{root}{tail}".replace("\\\\", "\\")
        return f"...{tail}".replace("\\\\", "\\")

    def _styler_hide_index(sty: _pd.io.formats.style.Styler) -> _pd.io.formats.style.Styler:
        try:
            return sty.hide(axis="index")  # pandas >= 2
        except Exception:
            try:
                return sty.hide_index()    # pandas < 2
            except Exception:
                return sty

    def _to_html_table(df_: _pd.DataFrame, caption: str | None = None) -> str:
        sty = (
            df_.style
            .set_table_attributes('class="tbl"')
            .set_properties(**{"text-align": "right", "padding": "6px 10px"})
        )
        sty = _styler_hide_index(sty)
        html_tbl = sty.to_html()
        if caption:
            html_tbl = f'<div class="tbl-caption">{_html.escape(caption)}</div>' + html_tbl
        return html_tbl

    def _badge(text: str, kind: str = "info") -> str:
        cls = {"ok": "ok", "warn": "warn", "info": "info"}.get(kind, "info")
        return f'<span class="badge {cls}">{_html.escape(text)}</span>'

    def _bool_badge(flag: bool) -> str:
        return _badge("ON" if flag else "OFF", "ok" if flag else "warn")

    # ---------- métricas básicas ----------
    n_rows, n_cols = df.shape
    mem = _fmt_bytes(df.memory_usage(deep=True).sum())
    null_total = int(df.isna().sum().sum())
    null_any_cols = int(df.isna().any().sum())

    fmt_root = _html.escape(_shorten_path(project_root, keep_parts=keep_path_parts))
    fmt_file = _html.escape(_shorten_path(processed_path, keep_parts=keep_path_parts))
    fmt_fmt = _html.escape(processed_path.suffix.lower())

    # Distribuição da target
    counts = df[target_name].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(2)
    tgt_df = (
        _pd.DataFrame(
            {
                "value": counts.index.astype(str),
                "count": counts.values,
                "pct": pct.values,
            }
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    tgt_df["pct"] = tgt_df["pct"].map(lambda x: f"{x:.2f}%")
    tgt_html = _to_html_table(tgt_df, caption=f"Distribuição de '{target_name}'")

    params_df = _pd.DataFrame(
        {
            "parâmetro": ["test_size", "random_state", "scale_numeric"],
            "valor": [test_size, random_state, "ON" if scale_numeric else "OFF"],
        }
    )
    params_html = _to_html_table(params_df, caption="Parâmetros (pré-split)")

    # ---------- CSS + HTML ----------
    css = """
    <style>
    .n2-wrap { 
      --bg:#f7f9fc; --fg:#0f172a; --muted:#6b7280; --card:#ffffff; --edge:#e5e7eb;
      --aqua:#22d3ee; --aqua-soft:rgba(34,211,238,.15);
      --purple:#8b5cf6; --purple-soft:rgba(139,92,246,.12);
      --ok:#10b981; --warn:#f59e0b; --info:#6366f1;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color: var(--fg);
      background: var(--bg);
      border: 1px solid var(--edge);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 25px rgba(2,6,23,.06);
      font-size: 15px;
    }
    .n2-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
    .n2-title { font-size: 22px; font-weight: 800; letter-spacing: .2px; }
    .n2-sub { color: var(--muted); font-size: 13px; }
    .grid { display:grid; grid-template-columns: repeat(12, 1fr); gap:14px; }
    .card { background: var(--card);
            border:1px solid var(--edge); border-radius:14px; padding:14px;
            box-shadow: 0 3px 8px rgba(2,6,23,.04); }
    .span-4 { grid-column: span 4; } .span-6 { grid-column: span 6; } .span-12 { grid-column: span 12; }
    .knum { font-weight:800; font-size: 24px; }
    .muted { color: var(--muted); font-size: 13px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 14px; }
    .kv { display:flex; align-items:center; gap:10px; margin:6px 0; }

    .pill {
      display:inline-block;
      border-radius:999px;
      border:1px solid var(--edge);
      padding:3px 10px;
      font-size:13px;
      background:#fbfdff;
      color:inherit;
    }
    .pill.aqua {
      border-color: var(--aqua);
      background: var(--aqua-soft);
    }

    .badge { border-radius: 999px; padding: 3px 10px; font-size: 12px; border:1px solid var(--edge); }
    .badge.ok { background: rgba(16,185,129,.14); color:#065f46; border-color: rgba(16,185,129,.35);}
    .badge.warn { background: rgba(245,158,11,.14); color:#7c4a03; border-color: rgba(245,158,11,.35);}
    .badge.info { background: var(--purple-soft); color:#4c1d95; border-color: rgba(139,92,246,.35);}
    .tbl { border-collapse: collapse; width:100%; font-size: 14px; background: #ffffff;
           color: var(--fg); border:1px solid var(--edge); border-radius:12px; overflow:hidden; }
    .tbl th, .tbl td { border-bottom: 1px solid #eef2f7; padding: 10px 12px; }
    .tbl th { text-align:left; color:#374151;
              background: linear-gradient(0deg, #fafcff, #f3f6fb); }
    .tbl-caption { font-size:13px; color: var(--muted); margin: 6px 0 6px 2px; }
    .accent { box-shadow: inset 0 0 0 1px var(--edge), 0 0 0 2px var(--aqua-soft); }
    .accent-purple { box-shadow: inset 0 0 0 1px var(--edge), 0 0 0 2px var(--purple-soft); }
    </style>
    """

    html_panel = f"""
    <div class="n2-wrap accent">
      <div class="n2-head">
        <div class="n2-title">N2 — Resumo do Bootstrap e Leitura do Dataset</div>
        <div class="n2-sub mono">{fmt_fmt} · {n_rows}×{n_cols} · {mem}</div>
      </div>
      <div class="grid">
        <div class="card span-6 accent">
          <div class="muted">Projeto</div>
          <div class="mono">{fmt_root}</div>
          <div class="muted" style="margin-top:10px;">Arquivo lido</div>
          <div class="mono">{fmt_file}</div>
        </div>

        <div class="card span-6 accent">
          <div class="muted">Dimensão</div>
          <div class="knum">{n_rows} linhas <span class="muted">×</span> {n_cols} colunas</div>
          <div class="kv"><span class="muted">Memória estimada:</span> <span class="pill aqua">{mem}</span></div>
          <div class="kv"><span class="muted">Nulos:</span> <span class="pill aqua">{null_total} células · {null_any_cols} col(s)</span></div>
        </div>

        <div class="card span-4 accent">
          <div class="muted">Tipos de colunas</div>
          <div class="kv"><span class="pill aqua">num: {len(num_cols)}</span></div>
          <div class="kv"><span class="pill aqua">cat: {len(cat_cols)}</span></div>
          <div class="kv"><span class="pill aqua">other: {len(other_cols)}</span></div>
        </div>

        <div class="card span-4 accent-purple">
          <div class="muted">Target</div>
          <div class="knum">{_html.escape(target_name)}</div>
          <div class="muted">Valores únicos: {df[target_name].nunique()}</div>
        </div>

        <div class="card span-4 accent">
          <div class="muted">Parâmetros (pré-split)</div>
          <div class="kv"><span class="muted">test_size</span><span class="pill aqua">{test_size}</span></div>
          <div class="kv"><span class="muted">random_state</span><span class="pill aqua">{random_state}</span></div>
          <div class="kv"><span class="muted">scale_numeric</span>{_bool_badge(scale_numeric)}</div>
        </div>

        <div class="card span-6 accent">
          {tgt_html}
        </div>

        <div class="card span-6 accent">
          {params_html}
        </div>

        <div class="card span-12 muted" style="text-align:right;">
          Gerado automaticamente • N2 • {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
      </div>
    </div>
    """
    _display(HTML(css + html_panel))


# =============================================================================
# N2 — Split treino/teste & resumo de colunas (display)
# =============================================================================

def _n2_make_target_distribution_table(y, target_name="target"):
    """
    Gera uma tabela com contagem e percentual da variável alvo.

    Parameters
    ----------
    y : pandas.Series
        Série da variável alvo.
    target_name : str, default "target"
        Nome da coluna alvo (para rótulo da tabela).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexado pelos valores da target,
        com colunas 'count' e 'pct'.
    """
    import pandas as pd

    if y is None:
        return pd.DataFrame(columns=["count", "pct"])

    vc = y.value_counts(dropna=False)
    total = vc.sum() if vc.sum() else 1
    pct = (vc / total * 100).round(2)

    df = pd.DataFrame({"count": vc, "pct": pct})
    df.index.name = target_name
    return df


def n2_display_split_and_column_summary(
    *,
    numeric_cols,
    categorical_cols,
    ignored_cols,
    X_train,
    X_test,
    y_train,
    y_test,
    target_name,
    split_params=None,
    rare_categories=None,
    rare_threshold=None,
    logger=None,
):
    """
    Exibe um painel resumido (HTML) do split treino/teste e da distribuição
    da variável alvo, em um layout mais organizado e coeso, alinhado ao painel N2.

    Esta função é apenas de APRESENTAÇÃO. A lógica de preparação
    (separar X/y, fazer o train_test_split, etc.) continua no notebook.
    """
    import pandas as pd
    from IPython.display import display, HTML

    split_params = split_params or {}
    rare_categories = rare_categories or {}
    ignored_cols = ignored_cols or []

    # ------------------------------------------------------------------
    # 1) Log textual (somente via logger; prints ficam no notebook)
    # ------------------------------------------------------------------
    msg_cols = (
        f"Colunas numéricas: {len(numeric_cols)} | "
        f"categóricas: {len(categorical_cols)} | "
        f"ignoradas: {len(ignored_cols)}"
    )
    msg_split = (
        "Split params -> "
        f"test_size={split_params.get('test_size')} | "
        f"random_state={split_params.get('random_state')} | "
        f"stratify={split_params.get('stratify')}"
    )
    msg_shapes = (
        f"X_train: {X_train.shape} | "
        f"X_test: {X_test.shape}"
    )

    if logger is not None:
        logger.info(msg_cols)
        logger.info(msg_split)
        logger.info(msg_shapes)

    # ------------------------------------------------------------------
    # 2) Distribuições da target (geral / train / test)
    # ------------------------------------------------------------------
    y_all = pd.concat([y_train, y_test], axis=0)

    df_all = _n2_make_target_distribution_table(y_all, target_name=target_name).reset_index()
    df_tr = _n2_make_target_distribution_table(y_train, target_name=target_name).reset_index()
    df_te = _n2_make_target_distribution_table(y_test, target_name=target_name).reset_index()

    def _build_table_html(df: pd.DataFrame) -> str:
        if df.empty:
            return "<em>Sem dados</em>"

        headers = "".join(
            f"<th>{col}</th>" for col in df.columns
        )
        body_rows = []
        for _, row in df.iterrows():
            cells = "".join(f"<td>{row[col]}</td>" for col in df.columns)
            body_rows.append(f"<tr>{cells}</tr>")
        body = "".join(body_rows)

        return f"""
        <table class="n2p-table">
          <thead><tr>{headers}</tr></thead>
          <tbody>{body}</tbody>
        </table>
        """

    card_all = f"""
      <div class="n2p-mini-card">
        <div class="n2p-mini-title">Geral</div>
        {_build_table_html(df_all)}
      </div>
    """

    card_tr = f"""
      <div class="n2p-mini-card">
        <div class="n2p-mini-title">Train</div>
        {_build_table_html(df_tr)}
      </div>
    """

    card_te = f"""
      <div class="n2p-mini-card">
        <div class="n2p-mini-title">Test</div>
        {_build_table_html(df_te)}
      </div>
    """

    dist_block_html = f"""
      <div class="n2p-section">
        <div class="n2p-section-title">
          Distribuição da target — geral / train / test
        </div>
        <div class="n2p-grid-3">
          {card_all}
          {card_tr}
          {card_te}
        </div>
      </div>
    """

    # ------------------------------------------------------------------
    # 3) Categorias raras
    # ------------------------------------------------------------------
    if rare_categories:
        th = rare_threshold if rare_threshold is not None else "limiar configurado"
        rare_rows = [
            {"column": col, "n_rare_categories": n_rare}
            for col, n_rare in rare_categories.items()
        ]
        df_rare = pd.DataFrame(rare_rows).sort_values(
            "n_rare_categories", ascending=False
        )

        rare_table = _build_table_html(df_rare)

        rare_block_html = f"""
          <div class="n2p-section">
            <div class="n2p-section-title">
              Colunas categóricas com categorias raras (&lt; {th} amostras) no train
            </div>
            <div class="n2p-mini-card n2p-mini-full">
              {rare_table}
            </div>
          </div>
        """
    else:
        rare_block_html = """
          <div class="n2p-section">
            <div class="n2p-section-title">
              Nenhuma coluna categórica com categorias raras no train com o limite atual.
            </div>
          </div>
        """

    # ------------------------------------------------------------------
    # 4) Card único com tudo dentro (estilo alinhado ao painel N2)
    # ------------------------------------------------------------------
    html = f"""

    <style>
      .n2p-card {{
        background: #f8fafc;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 16px 20px 18px 20px;
        margin: 10px 0 16px 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }}

      .n2p-title {{
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 6px;
      }}

      .n2p-header-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-size: 0.9rem;
        margin-bottom: 8px;
      }}

      .n2p-header-block {{
        min-width: 180px;
      }}

      .n2p-label {{
        opacity: 0.75;
        font-size: 0.8rem;
        margin-bottom: 2px;
        color: #64748b;
      }}

      .n2p-section {{
        margin-top: 14px;
        font-size: 0.9rem;
      }}

      .n2p-section-title {{
        font-weight: 600;
        margin-bottom: 8px;
        color: #0f172a;
      }}

      .n2p-grid-3 {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }}

      .n2p-mini-card {{
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 10px 12px;
        flex: 1 1 220px;
      }}

      .n2p-mini-full {{
        flex: 1 1 100%;
      }}

      .n2p-mini-title {{
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 4px;
        color: #0f172a;
      }}

      .n2p-table {{
        border-collapse: collapse;
        margin-top: 4px;
        font-size: 0.84rem;
        width: 100%;
      }}

      .n2p-table th {{
        background: #e2f3ff;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        padding: 4px 8px;
        text-align: center;
        font-weight: 600;
      }}

      .n2p-table td {{
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        padding: 4px 8px;
        text-align: center;
      }}

      .n2p-table tbody tr:nth-child(even) td {{
        background: #f1f5f9;
      }}
    </style>

    <div class="n2p-card">

      <div class="n2p-title">
        Split treino/teste &amp; resumo de colunas
      </div>

      <div class="n2p-header-row">

        <div class="n2p-header-block">
          <div class="n2p-label">Colunas</div>
          <div>
            numéricas: <strong>{len(numeric_cols)}</strong> ·
            categóricas: <strong>{len(categorical_cols)}</strong> ·
            ignoradas: <strong>{len(ignored_cols)}</strong>
          </div>
        </div>

        <div class="n2p-header-block">
          <div class="n2p-label">Parâmetros do split</div>
          <div>
            test_size=<strong>{split_params.get('test_size')}</strong> ·
            random_state=<strong>{split_params.get('random_state')}</strong> ·
            stratify=<strong>{split_params.get('stratify')}</strong>
          </div>
        </div>

        <div class="n2p-header-block">
          <div class="n2p-label">Shapes</div>
          <div>
            X_train: <strong>{X_train.shape}</strong> ·
            X_test: <strong>{X_test.shape}</strong>
          </div>
        </div>

      </div>

      {dist_block_html}
      {rare_block_html}

    </div>
    """

    display(HTML(html))


# =============================================================================
# N2 — Painel de pré-processamento (One-Hot + escala)
# =============================================================================

def n2_display_preprocess_summary(
    *,
    num_cols,
    cat_cols,
    scale_numeric,
    X_train_t,
    X_test_t,
    feat_names=None,
    mem_train=None,
    mem_test=None,
):
    """
    Exibe um painel resumindo o pré-processamento:
      - quantidade de colunas numéricas/categóricas de entrada
      - shapes transformados (X_train_t, X_test_t)
      - uso de escala em numéricas
      - memória estimada dos arrays transformados
      - preview dos nomes de features geradas (tabela 3 colunas)

    Esta função é apenas de apresentação.
    """
    import numpy as _np
    from IPython.display import HTML, display

    # ---------------------------------------------------------
    # Cálculos auxiliares
    # ---------------------------------------------------------
    n_in_num = len(num_cols)
    n_in_cat = len(cat_cols)

    n_train, n_features = X_train_t.shape
    n_test  = X_test_t.shape[0]

    def _mb(nbytes):
        return (float(nbytes) / (1024.0 ** 2)) if nbytes is not None else None

    if mem_train is None and isinstance(X_train_t, _np.ndarray):
        mem_train = X_train_t.nbytes
    if mem_test is None and isinstance(X_test_t, _np.ndarray):
        mem_test = X_test_t.nbytes

    mem_train_mb = _mb(mem_train)
    mem_test_mb  = _mb(mem_test)

    # Preview das features (até 24) organizado em tabela 3 colunas
    preview_feats = []
    if feat_names is not None:
        preview_feats = list(feat_names[:24])

    rows_html = ""
    if preview_feats:
        for i in range(0, len(preview_feats), 3):
            row = preview_feats[i:i+3]
            # completa com strings vazias se faltar coluna
            while len(row) < 3:
                row.append("")
            cells = "".join(f"<td>{name}</td>" for name in row)
            rows_html += f"<tr>{cells}</tr>"
        features_table_html = f"""
          <table class="n2pp-feature-table">
            <thead>
              <tr>
                <th>feature 1</th>
                <th>feature 2</th>
                <th>feature 3</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        """
    else:
        features_table_html = "<em>Preview indisponível (feat_names=None).</em>"

    # ---------------------------------------------------------
    # HTML + CSS (padrão N2, tema claro)
    # ---------------------------------------------------------
    html = f"""
    <style>
      .n2pp-card {{
        background: #f8fafc;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 14px 18px 18px 18px;
        margin: 10px 0 20px 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 0.9rem;
      }}
      .n2pp-title {{
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 8px;
      }}
      .n2pp-header-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-bottom: 10px;
      }}
      .n2pp-header-block {{
        min-width: 180px;
      }}
      .n2pp-label {{
        font-size: 0.8rem;
        opacity: 0.7;
        margin-bottom: 2px;
      }}
      .n2pp-value-strong {{
        font-weight: 600;
      }}
      .n2pp-mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
                     "Liberation Mono", "Courier New", monospace;
      }}
      .n2pp-section {{
        margin-top: 10px;
      }}
      .n2pp-section-title {{
        font-weight: 600;
        margin-bottom: 4px;
      }}
      .n2pp-feature-table {{
        border-collapse: collapse;
        margin-top: 4px;
        font-size: 0.82rem;
        width: 100%;
        max-width: 640px;
      }}
      .n2pp-feature-table th,
      .n2pp-feature-table td {{
        border: 1px solid #e2e8f0;
        padding: 4px 6px;
        text-align: left;
      }}
      .n2pp-feature-table thead th {{
        background: #e0f2fe;
        color: #075985;
        font-weight: 600;
      }}
      .n2pp-feature-table tbody tr:nth-child(odd) td {{
        background: #f1f5f9;
      }}
    </style>

    <div class="n2pp-card">
      <div class="n2pp-title">
        Pré-processamento — One-Hot denso + escala
      </div>

      <div class="n2pp-header-row">
        <div class="n2pp-header-block">
          <div class="n2pp-label">Colunas de entrada</div>
          <div>
            numéricas: <span class="n2pp-value-strong">{n_in_num}</span> ·
            categóricas: <span class="n2pp-value-strong">{n_in_cat}</span>
          </div>
        </div>

        <div class="n2pp-header-block">
          <div class="n2pp-label">Shapes transformados</div>
          <div class="n2pp-mono">
            X_train_t: <span class="n2pp-value-strong">({n_train}, {n_features})</span><br/>
            X_test_t: <span class="n2pp-value-strong">({n_test}, {n_features})</span>
          </div>
        </div>

        <div class="n2pp-header-block">
          <div class="n2pp-label">Parâmetros</div>
          <div>
            escala numérica: <span class="n2pp-value-strong">{'ON' if scale_numeric else 'OFF'}</span>
          </div>
        </div>

        <div class="n2pp-header-block">
          <div class="n2pp-label">Memória estimada</div>
          <div>
            train: <span class="n2pp-value-strong">{mem_train_mb:.2f} MB</span> ·
            test: <span class="n2pp-value-strong">{mem_test_mb:.2f} MB</span>
          </div>
        </div>
      </div>

      <div class="n2pp-section">
        <div class="n2pp-section-title">Preview de features transformadas</div>
        {features_table_html}
      </div>
    </div>
    """

    display(HTML(html))
