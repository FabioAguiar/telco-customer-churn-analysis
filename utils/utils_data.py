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

UTILS_DATA_VERSION = "1.2.2-merged"

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
    Retorna dict: {'df','before','after','strategy','imputed_cols'}
    """
    df = coerce_df(df)

    # Lê config em dois formatos possíveis
    missing_cfg = dict(config.get("missing", {}))
    handle = bool(config.get("handle_missing", missing_cfg.get("enabled", True)))
    strategy = (config.get("missing_strategy",
                           missing_cfg.get("strategy", "simple")) or "simple").lower()

    # Parâmetros extras (com defaults)
    knn_k = int(missing_cfg.get("knn_k", 5))
    it_max = int(missing_cfg.get("iterative_max_iter", 10))
    it_seed = int(missing_cfg.get("iterative_random_state", 42))

    # Onde salvar relatórios
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

    # Estratégias
    def _simple(df_in: pd.DataFrame):
        df_out, meta = simple_impute_with_flags(df_in)
        cols = [m["col"] for m in meta.get("imputed", [])]
        return df_out, cols, "simple"

    def _knn(df_in: pd.DataFrame):
        try:
            from sklearn.impute import KNNImputer  # type: ignore
        except Exception:
            # fallback
            return _simple(df_in)
        num_cols = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        if not num_cols:
            return _simple(df_in)
        df_out = df_in.copy()
        for c in num_cols:
            df_out[f"{c}_was_missing"] = df_out[c].isna().astype(int)
        imputer = KNNImputer(n_neighbors=knn_k)
        df_out[num_cols] = imputer.fit_transform(df_out[num_cols])
        # cols imputadas: as que tinham NaN
        cols = [c for c in num_cols if df_in[c].isna().any()]
        return df_out, cols, "knn"

    def _iterative(df_in: pd.DataFrame):
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer  # type: ignore
        except Exception:
            return _simple(df_in)
        num_cols = [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]
        if not num_cols:
            return _simple(df_in)
        df_out = df_in.copy()
        for c in num_cols:
            df_out[f"{c}_was_missing"] = df_out[c].isna().astype(int)
        imp = IterativeImputer(max_iter=it_max, random_state=it_seed, sample_posterior=False)
        df_out[num_cols] = imp.fit_transform(df_out[num_cols])
        cols = [c for c in num_cols if df_in[c].isna().any()]
        return df_out, cols, "iterative"

    # Seleção da estratégia com fallback
    chosen = strategy if prefer == "auto" else prefer
    try_chain = {
        "simple":    (_simple,   ["simple"]),
        "knn":       (_knn,      ["knn", "simple"]),
        "iterative": (_iterative,["iterative", "simple"]),
        "mice":      (_iterative,["iterative", "simple"]),
        "auto":      (None,      [strategy, "simple"]),
    }

    # resolve ordem de tentativa
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
    Garante a existência de uma coluna target conforme configuração.

    Retorna:
      df, target_name, class_map, report_df
    """
    import pandas as pd

    # Ler parâmetros do config
    tgt_cfg = config.get("target", {})
    target_name = tgt_cfg.get("name", "target")
    positive = tgt_cfg.get("positive", None)
    negative = tgt_cfg.get("negative", None)
    src_col = tgt_cfg.get("source", None)

    # Caso já exista no dataset
    if target_name in df.columns:
        if verbose:
            print(f"[target] Coluna '{target_name}' já existe — nenhuma ação necessária.")
        tgt_report = pd.DataFrame({
            "target": [target_name],
            "status": ["já existe"],
            "source": [src_col or target_name],
            "positive": [positive],
            "negative": [negative]
        })
        class_map = {1: positive, 0: negative}
        return df, target_name, class_map, tgt_report

    # Se precisar derivar a coluna a partir de outra
    if src_col and src_col in df.columns:
        if verbose:
            print(f"[target] Criando '{target_name}' a partir da coluna '{src_col}'.")

        series = df[src_col]

        if positive is not None and negative is not None:
            df[target_name] = series.map({positive: 1, negative: 0})
        elif series.dtype == "bool":
            df[target_name] = series.astype(int)
        else:
            # fallback genérico: mapeia valores únicos
            unique_vals = sorted(series.dropna().unique())
            mapping = {val: i for i, val in enumerate(unique_vals)}
            df[target_name] = series.map(mapping)
            if verbose:
                print(f"[target] Mapeamento automático: {mapping}")

        class_map = {1: positive, 0: negative} if positive and negative else mapping
        status = "criado"
    else:
        if verbose:
            print("[target] Nenhuma coluna de origem encontrada — criando target nulo.")
        df[target_name] = pd.NA
        class_map = {}
        status = "não criado"

    tgt_report = pd.DataFrame({
        "target": [target_name],
        "status": [status],
        "source": [src_col],
        "positive": [positive],
        "negative": [negative]
    })

    if verbose:
        print(f"[target] Conclusão: {status} ({target_name})")

    return df, target_name, class_map, tgt_report


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
