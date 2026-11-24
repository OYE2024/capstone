# -*- coding: utf-8 -*-
"""Дослідницький PU-пайплайн. Комірки Spyder запускаються окремо.

У Spyder збережіть файл, виберіть інтерпретатор із requirements.txt, відкрийте
комірку потрібного етапу та запустіть її клавішами Ctrl+Enter. У CONFIG задайте
run_dir для продовження наявного запуску; без нього створюється новий запуск.
Командний рядок: python pipeline.py --stage all [--run-dir ШЛЯХ].
"""
# %% 1. Параметри дослідження
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import platform
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "temp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "temp" / "cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

TAGS = ("suspicious", "value", "bonus", "niche", "sharp", "insider", "arber",
        "high_roller", "glitch", "late", "bot", "ghost")
CONFIG = {
    # Селектор етапу — перше налаштування.
    "research_stage": 1,            # 1 — етап 1; 2 — етап 2 з методом stage2.method.

    # Спільні дані, розбиття та запуск.
    "common": {
        "source": str(ROOT / "datasets" / "data.csv"),
        "output_root": str(ROOT / "outputs"),
        "audit_dir": str(ROOT / "audit"),
        "run_dir": None,
        "baseline_run": None,
        "seed": 42,
        "max_rows": None,
        "split": {"test": 0.15, "validation": 0.15, "candidates": 5},
        "eda_plot_rows": 100000,
    },

    # Етап 1: навчання за спостережуваними мітками.
    "stage1": {
        "model_type": "lightgbm",  # lightgbm або catboost.
        # class_balance_selector: 0 — без додаткового балансування (ваги 1);
        # 1 — увімкнути автоматичний метод, вибраний нижче для поточної моделі.
        # LightGBM: scale_pos_weight (N_U/N_P) або is_unbalance;
        # CatBoost: Balanced або SqrtBalanced.
        "class_balance_selector": 0,
        "imbalance_method": {"lightgbm": "scale_pos_weight", "catboost": "Balanced"},
        # feature_importance_selector: 0 — використовувати всі ознаки й зберегти їх важливість;
        # 1 — автоматично підібрати набір ознак окремо для кожного тегу на validation.
        "feature_importance_selector": 0,
        # threshold_strategy: fixed — fixed_threshold; f1 або f2 — підібрати поріг
        # на validation для найкращого відповідного показника.
        "threshold_strategy": "fixed",
        "fixed_threshold": 0.5,
        "models": {
            "lightgbm": {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31,
                         "n_jobs": 2, "verbosity": -1},
            "catboost": {"iterations": 150, "learning_rate": 0.05, "depth": 6,
                         "l2_leaf_reg": 3.0, "thread_count": 2, "verbose": False,
                         "allow_writing_files": False},
        },
        "optimization": {"enabled": False, "max_candidates": 4,
                          "lightgbm": {"num_leaves": [15, 31], "learning_rate": [0.03, 0.05]},
                          "catboost": {"depth": [4, 6], "learning_rate": [0.03, 0.05]}},
        # Окремий діагностичний блок запускається після навчання лише для етапу 1.
        "learning_curves": {"enabled": True, "eval_period": 5, "max_train_rows": 100000},
    },

    # Етап 2: обидві гілки використовують CatBoost; method вибирає PU-підхід.
    "stage2": {
        "method": "nnpu",          # nnpu або sar_pu
        # Умови фільтра налаштовуються окремо для кожного тегу.
        # Порожній список conditions означає невідому область, а не автоматичне A/B.
        "filter_rules": {tag: {"logic": "all", "conditions": [], "version": None} for tag in TAGS},
        "audit": {"confidence": 0.95, "margin": 0.05, "prior": 0.5,
                  "design_effect": 1.0, "unresolved_fraction": 0.1,
                  "max_invited": None, "bootstrap_iterations": 300},
        "catboost": {"iterations": 150, "learning_rate": 0.05, "depth": 6,
                     "l2_leaf_reg": 3.0, "thread_count": 2, "verbose": False,
                     "allow_writing_files": False},
        "threshold_strategy": "fixed",
        "fixed_threshold": 0.5,
        "optimization": {"enabled": False, "max_candidates": 4,
                         "catboost": {"depth": [4, 6], "learning_rate": [0.03, 0.05]}},
        "nnpu": {
            "prior_source": "random_audit",
            "positive_distribution": "observed_P_plus_ipw_audited_positive_U",
        },
        "sar_pu": {
            "propensity_model": "catboost",
            "propensity_target": "P_vs_true_positive_U",
            # Для кожного тегу задаються ознаки, що впливали на історичне присвоєння мітки.
            # Не вважаємо автоматично, що ознаки фільтра A/B пояснюють імовірність розмітки.
            "propensity_features": {tag: [] for tag in TAGS},
            "local_certainty_decay": 0.9,
            "max_em_iterations": 5,
            "convergence_tolerance": 0.005,
            "convergence_patience": 2,
        },
    },

    # Калібрування score до ймовірностей — окремий етап для обох гілок етапу 2.
    "calibration": {"min_per_class": 20, "method": "sigmoid", "audit_bins": 5,
                    "per_bin": 30},
}

# %% 2. Імпорт бібліотек і допоміжні функції
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from scipy.stats import norm
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.preprocessing import OrdinalEncoder

EXCLUDED = {"project", "platform_id", "customer_id", "customer_login", "report_date",
            "last_own_fraud_tags", "last_merged_fraud_tags", "last_ccf", "last_ltd",
            "row_id", "split", "validation_role", "y_no_observed_tag"} | {f"y_{tag}" for tag in TAGS} | {f"filter_{tag}" for tag in TAGS}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run(cfg):
    if cfg.get("run_dir"):
        path = Path(cfg["run_dir"]).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Папку запуску не знайдено: {path}")
        return path
    parent = Path(cfg["output_root"]) / date.today().isoformat()
    parent.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100000):
        path = parent / f"run_{number:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Не вдалося створити папку запуску")


def parse_tags(value):
    if pd.isna(value) or not str(value).strip():
        return ()
    raw = str(value).strip()
    if raw in ("[]", "{}"): return ()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, str): parsed = [parsed]
        if isinstance(parsed, (list, tuple, set)):
            return tuple(str(x).strip().lower() for x in parsed if str(x).strip())
    except (ValueError, SyntaxError):
        pass
    # У CSV можуть траплятися JSON-подібні значення або значення, розділені комами.
    return tuple(x.strip().strip("'\" ").lower() for x in raw.strip("[]").split(",") if x.strip())


def load_source(cfg):
    path = Path(cfg["source"])
    if not path.is_file(): raise FileNotFoundError(f"Вхідний CSV не знайдено: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=cfg.get("max_rows"),
                     dtype={"project": "string", "platform_id": "string", "report_date": "string"})
    required = {"project", "platform_id", "report_date", "last_own_fraud_tags"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Відсутні обов'язкові колонки: {sorted(missing)}")
    if df[["project", "platform_id"]].isna().any().any():
        raise ValueError("Ключ гравця project + platform_id містить пропуски")
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    return df


def apply_filter_rule(df, rule):
    """Обчислити правило з трьома станами: 1, 0 або невідомо."""
    conditions = rule.get("conditions", [])
    logic = rule.get("logic", "all")
    if logic not in ("all", "any"): raise ValueError("Логіка фільтра має бути all або any")
    if not conditions: return pd.Series(pd.NA, index=df.index, dtype="Int8")
    result = None
    for condition in conditions:
        feature, op = condition.get("feature"), condition.get("op")
        if feature not in df.columns: raise ValueError(f"Ознаку фільтра не знайдено: {feature}")
        if feature in {"last_own_fraud_tags", "last_merged_fraud_tags", "row_id", "split", "validation_role"} or str(feature).startswith(("y_", "filter_")):
            raise ValueError(f"Для фільтра потрібна вхідна ознака, а не ціль чи службове поле: {feature}")
        values = df[feature]
        value = condition.get("value")
        if op not in ("is_missing", "not_missing") and value is None:
            raise ValueError(f"Не задано значення фільтра для {feature}")
        if op in ("in", "not_in", "between") and not isinstance(value, (list, tuple)):
            raise ValueError(f"Оператор {op} потребує списку значень")
        if op == "between" and len(value) != 2: raise ValueError("between потребує двох меж")
        try:
            if op == "is_missing": mask = values.isna().astype("boolean")
            elif op == "not_missing": mask = values.notna().astype("boolean")
            elif op == "in": mask = values.isin(value).astype("boolean").mask(values.isna())
            elif op == "not_in": mask = (~values.isin(value)).astype("boolean").mask(values.isna())
            elif op == "between": mask = values.between(*value).astype("boolean").mask(values.isna())
            else:
                ops = {"==": lambda s, v: s.eq(v), "!=": lambda s, v: s.ne(v),
                       ">": lambda s, v: s.gt(v), ">=": lambda s, v: s.ge(v),
                       "<": lambda s, v: s.lt(v), "<=": lambda s, v: s.le(v)}
                if op not in ops: raise ValueError(f"Невідомий оператор фільтра: {op}")
                mask = ops[op](values, value).astype("boolean").mask(values.isna())
        except TypeError as exc:
            raise ValueError(f"Несумісні типи значення та ознаки {feature}") from exc
        result = mask if result is None else (result & mask if logic == "all" else result | mask)
    return result.astype("Int8")


def transform(df, cfg=None):
    cfg = CONFIG if cfg is None else cfg
    result = df.copy()
    parsed = result["last_own_fraud_tags"].fillna("[]").astype(str).map(parse_tags)
    result["is_vip"] = parsed.map(lambda tags: int("vip" in tags)).astype("int8")
    for tag in TAGS:
        # Нуль означає відсутність СПОСТЕРЕЖУВАНОЇ мітки, а не істинний негативний клас.
        result[f"y_{tag}"] = parsed.map(lambda tags, t=tag: int(t in tags)).astype("int8")
        result[f"filter_{tag}"] = (apply_filter_rule(result, cfg["filter_rules"].get(tag, {}))
                                    if cfg["research_stage"] == 2 else pd.Series(pd.NA, index=result.index, dtype="Int8"))
    result["y_no_observed_tag"] = (result[[f"y_{t}" for t in TAGS]].sum(axis=1) == 0).astype("int8")
    return result


def group_ids(df):
    return pd.MultiIndex.from_frame(df[["project", "platform_id"]]).factorize()[0]


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def write_status(run, name, message):
    with (run / "status.txt").open("a", encoding="utf-8") as out:
        out.write(f"{name}: {message}\n")
    print(f"{name}: {message}", flush=True)


def read_prepared(run):
    path = run / "prepared.csv"
    if not path.exists(): raise FileNotFoundError("Спочатку запустіть етап preparation.")
    return pd.read_csv(path, low_memory=False, dtype={"project": "string", "platform_id": "string",
                                                      "report_date": "string"})

# %% 3. EDA: структура, пропуски, розподіли та теги

def stage_eda(cfg, run):
    df = load_source(cfg)
    n = len(df)
    if not n: raise ValueError("Вхідний файл не містить рядків")
    folder = run / "eda"
    folder.mkdir(exist_ok=True)
    source_cols = [c for c in df if c != "row_id"]
    key = ["project", "platform_id", "report_date"]
    structure = {"rows": n, "columns": len(source_cols), "exact_duplicates": int(df.duplicated(source_cols).sum()),
                 "unique_slices": int(df[key].drop_duplicates().shape[0]),
                 "repeated_slices": int(df.duplicated(key).sum()), "source_sha256": file_hash(cfg["source"])}
    save_json(folder / "structure.json", structure)
    pd.DataFrame({"column": source_cols, "dtype": [str(df[c].dtype) for c in source_cols]}).to_csv(folder / "structure.csv", index=False)
    missing = df[source_cols].isna().sum()
    pd.DataFrame({"column": missing.index, "count": missing.values, "fraction": missing.values/n}).to_csv(folder / "missing.csv", index=False)
    numeric = df[source_cols].select_dtypes(include="number")
    numeric.describe(percentiles=np.arange(.01, 1, .01)).T.to_csv(folder / "numeric.csv")
    cats = []
    for col in df[source_cols].select_dtypes(exclude="number"):
        counts = df[col].fillna("<пропущено>").value_counts().head(10)
        for rank, (value, count) in enumerate(counts.items(), 1):
            cats.append({"column": col, "unique": df[col].nunique(dropna=True), "rank": rank,
                         "value": value, "count": int(count), "fraction": count/n})
    pd.DataFrame(cats).to_csv(folder / "categorical.csv", index=False)
    df.groupby("report_date", dropna=False).size().rename("rows").to_csv(folder / "rows_by_date.csv")
    counts = {}
    sets = df["last_own_fraud_tags"].fillna("[]").astype(str).map(parse_tags)
    for tags in sets:
        for tag in set(tags): counts[tag] = counts.get(tag, 0) + 1
    tag_counts = pd.DataFrame([{"tag": k, "rows": v, "fraction": v/n} for k, v in counts.items()]).sort_values("rows", ascending=False) if counts else pd.DataFrame(columns=["tag", "rows", "fraction"])
    tag_counts.to_csv(folder / "tags.csv", index=False)
    plt.figure(figsize=(12, 5))
    if len(tag_counts):
        plt.bar(tag_counts.tag, tag_counts.rows); plt.yscale("log")
        for i, row in enumerate(tag_counts.itertuples()): plt.text(i, row.rows, f"{row.rows}\n{row.fraction:.2%}", ha="center", fontsize=7)
    plt.xticks(rotation=60, ha="right"); plt.ylabel("Підтверджені рядки (логарифмічна шкала)")
    savefig(folder / "tags.png")
    labels = np.column_stack([sets.map(lambda x, t=t: t in x).to_numpy(dtype=np.int8) for t in TAGS])
    cardinality = labels.sum(axis=1)
    pd.Series(cardinality).value_counts().sort_index().rename("rows").to_csv(folder / "cardinality.csv")
    plt.figure(); plt.hist(cardinality, bins=np.arange(-.5, len(TAGS)+1.5)); plt.xlabel("Кількість спостережених міток"); plt.ylabel("Рядки"); savefig(folder / "cardinality.png")
    # int64 запобігає переповненню під час множення міток.
    co = labels.astype(np.int64).T @ labels.astype(np.int64)
    pd.DataFrame(co, index=TAGS, columns=TAGS).to_csv(folder / "cooccurrence_observed.csv")
    counts_by_tag = labels.sum(axis=0, dtype=np.int64)
    union = counts_by_tag[:, None] + counts_by_tag[None, :] - co
    observed_jaccard = np.divide(co, union, out=np.full(co.shape, np.nan), where=union > 0)
    pd.DataFrame(observed_jaccard, index=TAGS, columns=TAGS).to_csv(folder / "jaccard_observed.csv")
    plt.figure(figsize=(10, 8)); plt.imshow(np.log1p(co), cmap="viridis"); plt.colorbar(label="log(1 + кількість)")
    plt.xticks(range(len(TAGS)), TAGS, rotation=90); plt.yticks(range(len(TAGS)), TAGS); savefig(folder / "cooccurrence_observed.png")
    plt.figure(figsize=(10, 8)); plt.imshow(observed_jaccard, vmin=0, vmax=1, cmap="viridis"); plt.colorbar(label="Спостережений Jaccard")
    plt.xticks(range(len(TAGS)), TAGS, rotation=90); plt.yticks(range(len(TAGS)), TAGS); savefig(folder / "jaccard_observed.png")
    (folder / "summary.txt").write_text("Частоти та Jaccard відображають лише спостережені мітки. Повна маска перевірок у джерелі не задана; Jaccard для істинних статусів не обчислюється.\n" + json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status(run, "EDA", f"Оброблено рядків: {n}; результати: {folder}")

# %% 4. Перетворення, групове розділення та планування аудиту

def split_groups(df, cfg):
    groups = group_ids(df)
    y = df[[f"y_{tag}" for tag in TAGS]].to_numpy(dtype=np.int8)
    total = y.sum(axis=0)
    best, best_error = None, float("inf")
    # Серед кількох варіантів групового розбиття вибираємо той, де розподіл підтверджених міток найближчий.
    for seed in range(cfg["seed"], cfg["seed"] + cfg["split"]["candidates"]):
        first = GroupShuffleSplit(n_splits=1, test_size=cfg["split"]["test"], random_state=seed)
        trval, test = next(first.split(df, groups=groups))
        relative = cfg["split"]["validation"] / (1-cfg["split"]["test"])
        second = GroupShuffleSplit(n_splits=1, test_size=relative, random_state=seed)
        train_rel, val_rel = next(second.split(trval, groups=groups[trval]))
        train, val = trval[train_rel], trval[val_rel]
        error = 0.0
        for idx, target in [(train, 1-cfg["split"]["test"]-cfg["split"]["validation"]),
                            (val, cfg["split"]["validation"]), (test, cfg["split"]["test"])]:
            error += np.abs(y[idx].sum(axis=0) / np.maximum(total, 1) - target)[total > 0].sum()
        if error < best_error: best, best_error = (train, val, test), error
    result = np.full(len(df), "", dtype="<U10")
    for name, idx in zip(("train", "validation", "test"), best): result[idx] = name
    return result


def stage_preparation(cfg, run):
    if (run / "prepared.csv").exists(): raise FileExistsError("Підготовлені дані вже існують у цьому запуску")
    df = transform(load_source(cfg), cfg)
    df["split"] = split_groups(df, cfg)
    df["validation_role"] = ""
    valid_idx = np.flatnonzero(df["split"].to_numpy() == "validation")
    valid_groups = group_ids(df.iloc[valid_idx])
    tune_rel, calibration_rel = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=cfg["seed"]+917).split(valid_idx, groups=valid_groups))
    df.loc[valid_idx[tune_rel], "validation_role"] = "tuning"
    df.loc[valid_idx[calibration_rel], "validation_role"] = "calibration"
    overlap = df.groupby(["project", "platform_id"], dropna=False)["split"].nunique()
    if (overlap > 1).any(): raise RuntimeError("Гравці перетинаються між вибірками")
    df.to_csv(run / "prepared.csv", index=False)
    df[["row_id", "project", "platform_id", "report_date", "split", "validation_role"]].to_csv(run / "split.csv", index=False)
    filter_hash = hashlib.sha256(json.dumps(cfg["filter_rules"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    save_json(run / "data_identity.json", {"source_sha256": file_hash(cfg["source"]),
                                            "split_sha256": file_hash(run / "split.csv"),
                                            "filter_rules_sha256": filter_hash})
    summary = df.groupby("split")[[f"y_{t}" for t in TAGS]].agg(["size", "sum"])
    summary.to_csv(run / "split_summary.csv")
    counts = []
    for tag in TAGS:
        for area, mask in (("A", df[f"filter_{tag}"].eq(1).fillna(False)),
                           ("B", df[f"filter_{tag}"].eq(0).fillna(False)),
                           ("unknown", df[f"filter_{tag}"].isna())):
            counts.append({"tag": tag, "area": area, "rows": int(mask.sum()),
                           "observed_positive": int(df.loc[mask, f"y_{tag}"].sum())})
    pd.DataFrame(counts).to_csv(run / "filter_summary.csv", index=False)
    if cfg["research_stage"] == 2:
        plan_audit(df, cfg, run)
    write_status(run, "Підготовка", f"Записано {len(df)} рядків; групи не перетинаються")


def plan_audit(df, cfg, run):
    """Створити окремі випадкові вибірки U для A/B та незалежних частин даних."""
    folder = run / "audit_sample_size"; folder.mkdir(exist_ok=True)
    samples = run / "audit_samples"; samples.mkdir(exist_ok=True)
    settings = cfg["audit"]
    z = float(norm.ppf((1 + settings["confidence"])/2))
    p, e, r, deff = (settings["prior"], settings["margin"], settings["unresolved_fraction"], settings["design_effect"])
    if not (0 < e < 1 and 0 < p < 1 and 0 <= r < 1 and deff > 0):
        raise ValueError("Некоректні параметри аудиту")
    for pos, tag in enumerate(TAGS):
        selected, sizes = [], []
        for split_index, part in enumerate(("train", "validation", "test")):
            part_mask = df.split.eq(part)
            if part == "validation": part_mask &= df.validation_role.eq("tuning")
            for area_index, (area, flag) in enumerate((("A", 1), ("B", 0))):
                pool = df.loc[part_mask & df[f"y_{tag}"].eq(0) & df[f"filter_{tag}"].eq(flag).fillna(False)]
                n = len(pool)
                n_srs = math.ceil(n*z*z*p*(1-p)/((n-1)*e*e+z*z*p*(1-p))) if n else 0
                resolved = min(n, math.ceil(n_srs*deff)); invited = min(n, math.ceil(resolved/(1-r)))
                if settings["max_invited"] is not None: invited = min(invited, int(settings["max_invited"]))
                seed = cfg["seed"] + pos*10 + split_index*2 + area_index
                chosen = pool.sample(n=invited, random_state=seed) if invited else pool
                sample = chosen[["row_id", "project", "platform_id", "report_date"]].copy()
                sample["split"] = part; sample["area"] = area; sample["audit_result"] = ""
                sample["selection_probability"] = invited/n if n else 0
                sample["audit_run"] = str(run.resolve())
                selected.append(sample)
                sizes.append({"tag":tag, "split":part, "area":area, "population":f"U-{part}-{area}",
                              "N_U":n, "z":z, "p":p, "margin":e, "DEFF":deff,
                              "unresolved_fraction":r, "n_srs":n_srs, "n_resolved":resolved,
                              "n_invited":invited, "selection_probability":invited/n if n else None,
                              "unknown_filter_rows":int((part_mask & df[f"y_{tag}"].eq(0) & df[f"filter_{tag}"].isna()).sum()),
                              "seed":seed, "planned_responses_sufficient": invited*(1-r) >= resolved if n else None})
        pd.concat(selected, ignore_index=True).to_csv(samples / f"{tag}.csv", index=False)
        pd.DataFrame(sizes).to_csv(folder / f"{tag}.csv", index=False)
    (folder / "design.txt").write_text(
        "Випадковий відбір U окремо за тегом, A/B та частиною даних. "
        "Validation для порога не перетинається з калібрувальними групами. "
        "Невідомий фільтр не включається до A/B. Результати заповніть у шаблоні та "
        "збережіть у audit/<тег>.csv; зберігайте невирішені рядки.\n", encoding="utf-8")

# %% 5. Аналіз аудиту та перевірка налаштувань

class AuditNotReadyError(RuntimeError):
    """Аудит не готовий; подальше навчання й оцінювання зупиняються."""


def stage_audit(cfg, run):
    df = read_prepared(run).set_index("row_id", drop=False)
    cfg["hidden_positive_fraction"] = {tag: {"A": None, "B": None} for tag in TAGS}
    result, answers, problems = [], [], []
    if cfg["research_stage"] == 1:
        save_json(run / "configuration_effective.json", cfg)
        pd.DataFrame(columns=["tag", "split", "area", "fraction", "status"]).to_csv(run / "audit_analysis.csv", index=False)
        write_status(run, "Аудит", "Етап 1: аудит і PU-корекція вимкнені")
        return
    if cfg["research_stage"] != 2: raise ValueError("research_stage має дорівнювати 1 або 2")
    mapping = {"1":1, "1.0":1, "positive":1, "позитивний":1,
               "0":0, "0.0":0, "negative":0, "негативний":0}
    audit_dir = Path(cfg["audit_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    created_templates, missing_manifests = [], []
    # Створюємо користувацькі CSV для аудиту в цільовій папці. Це незаповнені
    # шаблони: наявність файлу не означає, що аудит завершено.
    for tag in TAGS:
        path = audit_dir / f"{tag}.csv"
        manifest_path = run / "audit_samples" / f"{tag}.csv"
        if path.exists():
            continue
        if not manifest_path.exists():
            missing_manifests.append(f"{tag}: отсутствует манифест выборки {manifest_path}")
            continue
        template = pd.read_csv(manifest_path)
        if "audit_result" not in template:
            template["audit_result"] = ""
        template.to_csv(path, index=False)
        created_templates.append(str(path))
    if created_templates:
        write_status(run, "Аудит", "Созданы незаполненные шаблоны: " + ", ".join(created_templates))
    problems.extend(missing_manifests)
    for tag in TAGS:
        path = audit_dir / f"{tag}.csv"
        manifest_path = run / "audit_samples" / f"{tag}.csv"
        unknown_count = int(df[f"filter_{tag}"].isna().sum())
        if unknown_count:
            problems.append(f"{tag}: фільтр не визначений для {unknown_count} рядків")
        if not path.exists():
            result.extend({"tag":tag,"split":"train","area":area,"fraction":None,"status":"файл відсутній"} for area in ("A", "B"))
            problems.append(f"{tag}: відсутній файл {path}")
            continue
        try:
            if not manifest_path.exists(): raise ValueError("немає вибірки аудиту цього запуску")
            audit = pd.read_csv(path)
            if not {"row_id", "audit_result", "audit_run"}.issubset(audit):
                raise ValueError("потрібні row_id, audit_result та audit_run із шаблону")
            if audit.empty: raise ValueError("файл порожній")
            origins = audit.audit_run.dropna().unique()
            if len(origins) != 1: raise ValueError("потрібне єдине джерело аудиту")
            origin_info, current_info = Path(origins[0]) / "data_identity.json", run / "data_identity.json"
            if not origin_info.exists() or not current_info.exists() or json.loads(origin_info.read_text()) != json.loads(current_info.read_text()):
                raise ValueError("дані, фільтри або розділення не відповідають джерелу аудиту")
            audit["row_id"] = pd.to_numeric(audit.row_id, errors="raise").astype("int64")
            if audit.row_id.duplicated().any(): raise ValueError("повторні row_id")
            sampled = pd.read_csv(manifest_path)
            if sampled.empty: raise ValueError("вибірка аудиту порожня; перевірте фільтр та розміри областей")
            if not audit.row_id.isin(sampled.row_id).all(): raise ValueError("рядки не належать вибірці цього запуску")
            if not audit.row_id.isin(df.index).all(): raise ValueError("невідомі row_id")
            merged = sampled.drop(columns="audit_result").merge(
                audit[["row_id", "audit_result"]], on="row_id", how="left", validate="one_to_one")
            observed = df.loc[merged.row_id]
            if not observed[f"y_{tag}"].eq(0).all(): raise ValueError("аудит містить не U")
            expected_area = observed[f"filter_{tag}"].map({1:"A", 0:"B"}).to_numpy()
            if not np.array_equal(expected_area, merged.area.to_numpy()): raise ValueError("область аудиту не відповідає фільтру")
            if not np.array_equal(observed.split.to_numpy(), merged.split.to_numpy()): raise ValueError("частина даних не відповідає розділенню")
            values = merged.audit_result.astype("string").str.strip().str.lower()
            merged["target"] = values.map(mapping).astype("float64")
            unresolved = int(merged.target.isna().sum())
            if unresolved:
                problems.append(f"{tag}: файл аудиту не обработан — отсутствуют допустимые значения audit_result для {unresolved} строк")
            if (merged.selection_probability <= 0).any():
                problems.append(f"{tag}: вибірка має недопустиму ймовірність відбору")
            sizes_path = run / "audit_sample_size" / f"{tag}.csv"
            if not sizes_path.exists(): raise ValueError("немає розрахунку розмірів вибірки")
            sizes = pd.read_csv(sizes_path)
            if ((sizes.N_U > 0) & (sizes.n_invited == 0)).any():
                problems.append(f"{tag}: не відібрано рядки в непорожній області аудиту")
            merged["tag"] = tag
            answers.append(merged)
            for (part, area), group in merged.groupby(["split", "area"], observed=True):
                known = group.dropna(subset=["target"])
                fraction = float(known.target.mean()) if len(known) else None
                if part == "train" and fraction is not None: cfg["hidden_positive_fraction"][tag][area] = fraction
                low = high = None
                if len(known) and 0 < fraction < 1:
                    players = known.groupby(["project", "platform_id"]).target.agg(["sum", "count"])
                    if len(players) > 1:
                        rng = np.random.default_rng(cfg["seed"])
                        sums, counts = players["sum"].to_numpy(), players["count"].to_numpy()
                        draws = []
                        for _ in range(cfg["audit"]["bootstrap_iterations"]):
                            idx = rng.integers(0, len(players), len(players))
                            draws.append(sums[idx].sum()/counts[idx].sum())
                        alpha = (1-cfg["audit"]["confidence"])/2
                        low, high = np.quantile(draws, [alpha, 1-alpha]).tolist()
                elif len(known) and fraction in (0, 1):
                    effective_n = len(known)/cfg["audit"]["design_effect"]
                    z = norm.ppf((1+cfg["audit"]["confidence"])/2)
                    denom = 1 + z*z/effective_n
                    centre = (fraction + z*z/(2*effective_n))/denom
                    radius = z * math.sqrt(fraction*(1-fraction)/effective_n + z*z/(4*effective_n**2))/denom
                    low, high = max(0., centre-radius), min(1., centre+radius)
                result.append({"tag":tag,"split":part,"area":area,"fraction":fraction,
                               "n_invited":len(group),"n_resolved":len(known),
                               "n_positive":int(known.target.sum()),"n_unresolved":len(group)-len(known),
                               "ci_low":low,"ci_high":high,"sha256":file_hash(path),
                               "status":"частка серед вирішених; невирішені можуть зміщувати оцінку"})
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
            result.append({"tag":tag,"status":f"пропущено: {exc}"})
            problems.append(f"{tag}: {exc}")
    pd.DataFrame(result).to_csv(run / "audit_analysis.csv", index=False)
    if answers:
        checked_answers = pd.concat(answers, ignore_index=True)
        checked_answers.to_csv(run / "audit_answers.csv", index=False)
        # Перевірити всі теги до навчання, щоб не створити часткову модель.
        for tag in TAGS:
            if any(item.startswith(f"{tag}:") for item in problems): continue
            try:
                for part in ("train", "validation", "test"):
                    part_df = df[df.split == part]
                    if part == "validation": part_df = part_df[part_df.validation_role == "tuning"]
                    reviewed = reviewed_data(part_df, tag, checked_answers, part)
                    if part == "train" and reviewed.target.nunique() < 2:
                        raise ValueError("для train потрібні перевірені позитивні та негативні результати")
                    if part == "validation" and reviewed.target.sum() == 0:
                        raise ValueError("для validation немає позитивних результатів")
            except (ValueError, KeyError) as exc:
                problems.append(f"{tag}: {exc}")
    elif (run / "audit_answers.csv").exists(): (run / "audit_answers.csv").unlink()
    save_json(run / "configuration_effective.json", cfg)
    save_json(run / "audit_readiness.json", {"ready": not problems, "problems": problems})
    if problems:
        message = (f"Аудит не готов: {len(problems)} проблем. Обучение CatBoost + nnPU и оценивание остановлены. "
                   f"Заполните audit_result в CSV: {audit_dir}; шаблоны выборок: {run / 'audit_samples'}; "
                   f"подробности: {run / 'audit_readiness.json'}")
        write_status(run, "Аудит", message)
        raise AuditNotReadyError(message)
    method = cfg.get("stage2_method", "nnpu")
    write_status(run, "Аудит", f"Все файлы и результаты аудита проверены для ветки {method}")

# %% 6. Навчання базових моделей та оптимізація

def model_factory(cfg, params=None):
    params = params or {}
    if cfg["model_type"] == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**{**cfg["lightgbm"], **params, "random_state": cfg["seed"]})
    if cfg["model_type"] == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**{**cfg["catboost"], **params, "random_seed": cfg["seed"]})
    raise ValueError("model_type має бути lightgbm або catboost")


def encode_train(train, others, features):
    # Кодувальник навчається лише на train і зберігається разом із моделлю.
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]
    medians = train[numeric].replace([np.inf, -np.inf], np.nan).median().fillna(0)
    encoder = None
    if categorical:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(train[categorical].fillna("<пропущено>").astype(str))
    def apply(frame):
        result = pd.DataFrame(index=frame.index)
        if numeric: result[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(medians).astype("float32")
        if categorical: result[categorical] = encoder.transform(frame[categorical].fillna("<пропущено>").astype(str)).astype("float32")
        return result[features]
    return apply(train), [apply(x) for x in others], {"numeric":numeric,"categorical":categorical,"medians":medians,"encoder":encoder,"features":features}


def model_feature_importance(model, model_type):
    if model_type == "lightgbm":
        values = np.asarray(model.feature_importances_, dtype=float)
        method = f"feature_importances_ ({model.importance_type})"
    elif model_type == "catboost":
        values = np.asarray(model.get_feature_importance(type="FeatureImportance"), dtype=float)
        method = "get_feature_importance (FeatureImportance)"
    else:
        raise ValueError(f"Важливість ознак не підтримується для {model_type}")
    return values, method


def importance_threshold_candidates(features, values):
    """Сформувати для кожного тегу набори ознак за квантилями частки важливості, включно з усіма ознаками."""
    values = np.asarray(values, dtype=float)
    if len(features) != len(values): raise ValueError("Кількість важливостей не відповідає ознакам")
    total = float(values.sum())
    fractions = values / total if total > 0 else np.zeros_like(values)
    cuts = np.unique(np.quantile(fractions, [0.0, 0.25, 0.5, 0.75, 1.0]))
    candidates, seen = [], set()
    for cutoff in cuts:
        chosen = [feature for feature, fraction in zip(features, fractions)
                  if fraction >= cutoff - 1e-15]
        key = tuple(chosen)
        if chosen and key not in seen:
            seen.add(key)
            candidates.append((float(cutoff), chosen))
    if not candidates: candidates = [(0.0, list(features))]
    return fractions, candidates


BALANCE_METHODS = {
    "lightgbm": {"is_unbalance", "scale_pos_weight"},
    "catboost": {"Balanced", "SqrtBalanced"},
}


def class_balance_options(cfg, n_positive, n_unlabeled):
    """Повернути вбудовані для моделі параметри автоматичного балансування класів і поля звіту."""
    if cfg["class_balance_selector"] == 0:
        return {}, "none", 1.0
    model_type = cfg["model_type"]
    method = cfg["imbalance_method"].get(model_type)
    if method not in BALANCE_METHODS[model_type]:
        choices = ", ".join(sorted(BALANCE_METHODS[model_type]))
        raise ValueError(f"Для {model_type} imbalance_method має бути одним із: {choices}")
    if model_type == "lightgbm":
        if method == "is_unbalance": return {"is_unbalance": True}, method, None
        return {"scale_pos_weight": n_unlabeled / n_positive}, method, n_unlabeled / n_positive
    return {"auto_class_weights": method}, method, None


def fit_baseline_binary(cfg, x, y, params, balance_params):
    model = model_factory(cfg, {**params, **balance_params})
    model.fit(x, y)
    return model


class CatBoostNNPUObjective:
    """Невід'ємний PU-ризик для власної CPU-функції CatBoost."""
    def calc_ders_range(self, approxes, targets, weights):
        score = np.asarray(approxes, dtype=float)
        target = np.asarray(targets, dtype=int)
        positive = target == 1
        if not positive.any():
            raise ValueError("nnPU objective received no positive examples")
        sample_weight = np.ones(len(score), dtype=float) if weights is None else np.asarray(weights, dtype=float)
        positive_weight = sample_weight[positive]
        weight_sum = positive_weight.sum()
        prior = weight_sum / len(score)
        if not np.isfinite(prior) or not 0 < prior < 1:
            raise ValueError(f"nnPU prior is outside (0, 1): {prior}")
        normalized = positive_weight / weight_sum
        pos_score = score[positive]
        positive_risk = prior * np.dot(normalized, np.logaddexp(0.0, -pos_score))
        negative_risk = (np.logaddexp(0.0, score).mean()
                         - prior * np.dot(normalized, np.logaddexp(0.0, pos_score)))
        probability = expit(score)
        gradient = np.zeros_like(score)
        hessian = np.zeros_like(score)
        if negative_risk > 0:
            # Позитивний ризик і його віднімання в негативному ризику скорочуються
            # на P, залишаючи невід'ємний внесок ризику суміші.
            gradient[:] = probability / len(score)
            gradient[positive] -= prior * normalized
            hessian[:] = probability * (1.0 - probability) / len(score)
        else:
            gradient[positive] = prior * normalized * (probability[positive] - 1.0)
            hessian[positive] = (prior * normalized * probability[positive]
                                 * (1.0 - probability[positive]))
        # Власна функція втрат CatBoost для Python очікує від'ємні першу та другу
        # похідні, як у прикладі Logloss в документації.
        return [(-float(g), -float(h)) for g, h in zip(gradient, hessian)]


def fit_nnpu(x, positive_positions, positive_weights, cfg, params=None):
    """Навчити CatBoost за nnPU-ризиком на P та позитивних U з аудиту, зважених оберненою ймовірністю відбору."""
    from catboost import CatBoostClassifier, Pool
    positive_positions = np.asarray(positive_positions, dtype=int)
    positive_weights = np.asarray(positive_weights, dtype=float)
    prior = positive_weights.sum() / len(x)
    if not 0 < prior < 1 or not np.all(np.isfinite(positive_weights)):
        raise ValueError("Неприпустима оцінка частки позитивних для nnPU")
    target = np.zeros(len(x), dtype=np.int8)
    target[positive_positions] = 1
    weights = np.ones(len(x), dtype=float)
    weights[positive_positions] = positive_weights
    model_params = {**cfg["catboost"], **(params or {})}
    for balance_key in ("auto_class_weights", "scale_pos_weight", "class_weights"):
        model_params.pop(balance_key, None)
    model_params.update({"random_seed":cfg["seed"],
                         "loss_function":CatBoostNNPUObjective(),
                         # Для власних Python-функцій CatBoost потребує явно задану метрику.
                         # Logloss використовується лише для діагностики, це не nnPU-ризик навчання.
                         "eval_metric":"Logloss",
                         "boosting_type":"Plain", "task_type":"CPU"})
    model = CatBoostClassifier(**model_params)
    model.fit(Pool(x, label=target, weight=weights))
    return model, prior


def _fit_weighted_catboost(x, target, weights, cfg, params=None, loss="Logloss"):
    """Навчити бінарний класифікатор CatBoost, відкинувши псевдорядки з нульовою вагою."""
    from catboost import CatBoostClassifier
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(weights) & (weights > 1e-10)
    y_all = np.asarray(target, dtype=float if loss == "CrossEntropy" else np.int8)
    y = y_all[keep]
    if loss == "CrossEntropy":
        if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
            raise ValueError("CrossEntropy для SAR-EM потребує цілі в [0, 1]")
        if not np.any(y > 0) or not np.any(y < 1):
            raise ValueError("CrossEntropy для SAR-EM потребує позитивної та негативної ваги")
    elif len(np.unique(y)) < 2:
        raise ValueError("Для CatBoost потрібні позитивні та негативні вагомі приклади")
    model_params = {**cfg["catboost"], **(params or {})}
    for key in ("auto_class_weights", "scale_pos_weight", "class_weights"):
        model_params.pop(key, None)
    model_params.update({"random_seed":cfg["seed"], "loss_function":loss, "task_type":"CPU"})
    model = CatBoostClassifier(**model_params)
    model.fit(x.iloc[keep], y, sample_weight=weights[keep])
    return model


def _sar_propensity_features(cfg, tag, all_features):
    """Визначити для тегу ознаки propensity, відомі на момент присвоєння мітки."""
    configured = cfg["sar_pu"].get("propensity_features", "from_filter_rules")
    if configured == "from_filter_rules":
        rule = cfg["filter_rules"].get(tag, {})
        names = [condition.get("feature") for condition in rule.get("conditions", [])]
        names = [name for name in names if name]
    elif isinstance(configured, dict):
        names = configured.get(tag, [])
    elif isinstance(configured, (list, tuple)):
        names = list(configured)
    else:
        raise ValueError("sar_pu.propensity_features має бути from_filter_rules, списком або словником за тегом")
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError(f"{tag}: SAR-PU потребує відомих propensity-ознак; задайте їх або умови фільтра")
    missing = set(names) - set(all_features)
    if missing:
        raise ValueError(f"{tag}: propensity-ознаки відсутні серед ознак моделі: {sorted(missing)}")
    return names


def fit_sar_pu(x, propensity_x, observed_s, known_y, audit_weights, cfg, params=None):
    """Навчити CatBoost за SAR-EM, використовуючи підтверджені аудитом мітки U як фіксовані приховані цілі.

    ``observed_s`` — індикатор історичної мітки. ``known_y`` дорівнює 1 для P,
    містить істинне значення з аудиту для відібраних U та NaN для неперевірених U.
    Ваги аудиту використовуються під час початкового навчання з учителем; наступні
    кроки EM використовують усю сукупність навчальних ознак і фіксують мітки
    перевірених рядків відповідно до результатів аудиту.
    """
    observed_s = np.asarray(observed_s, dtype=np.int8)
    known_y = np.asarray(known_y, dtype=float)
    audit_weights = np.asarray(audit_weights, dtype=float)
    if not (len(x) == len(propensity_x) == len(observed_s) == len(known_y) == len(audit_weights)):
        raise ValueError("SAR-PU: довжини ознак, міток і аудиту не збігаються")
    p_mask = observed_s == 1
    u_mask = ~p_mask
    reviewed_u = u_mask & np.isfinite(known_y)
    if not p_mask.any() or not reviewed_u.any():
        raise ValueError("SAR-PU: потрібні історичні P та перевірені випадки U")
    if not np.isin(known_y[reviewed_u], [0.0, 1.0]).all():
        raise ValueError("SAR-PU: audit_result має містити лише 0 або 1")
    known_positive_u = reviewed_u & (known_y == 1)
    if not known_positive_u.any():
        raise ValueError("SAR-PU: у train-аудиті немає позитивних випадків U для оцінювання propensity")
    if not np.isfinite(audit_weights[reviewed_u]).all() or (audit_weights[reviewed_u] <= 0).any():
        raise ValueError("SAR-PU: некоректні ваги відбору аудиту")

    decay = float(cfg["sar_pu"]["local_certainty_decay"])
    max_iter = int(cfg["sar_pu"]["max_em_iterations"])
    tolerance = float(cfg["sar_pu"]["convergence_tolerance"])
    patience = int(cfg["sar_pu"]["convergence_patience"])
    if not 0 < decay <= 1 or max_iter < 1 or tolerance <= 0 or patience < 1:
        raise ValueError("SAR-PU: некоректні параметри EM")

    # Початкове навчання з надійними мітками. Обернені ймовірності відбору аудиту
    # коригують стратифіковану вибірку U лише на цьому початковому кроці.
    init_mask = p_mask | reviewed_u
    init_target = np.where(p_mask[init_mask], 1, known_y[init_mask]).astype(np.int8)
    init_weight = np.where(p_mask[init_mask], 1.0, audit_weights[init_mask])
    f_model = _fit_weighted_catboost(x.iloc[np.flatnonzero(init_mask)], init_target,
                                     init_weight, cfg, params)

    # e(x_e)=P(s=1|y=1,x_e) навчається лише на істинно позитивних випадках:
    # історичні P мають s=1; приховані позитивні з аудиту — s=0 і ваги, обернені до ймовірності відбору.
    e_rows = p_mask | known_positive_u
    e_target = observed_s[e_rows]
    e_weights = np.where(p_mask[e_rows], 1.0, audit_weights[e_rows])
    e_model = _fit_weighted_catboost(propensity_x.iloc[np.flatnonzero(e_rows)], e_target,
                                     e_weights, cfg, params)

    q = np.zeros(len(x), dtype=float)
    q[p_mask] = 1.0
    q[reviewed_u] = known_y[reviewed_u]
    previous_q = q[u_mask & ~reviewed_u].copy()
    stable_rounds, converged, iterations = 0, False, 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        f_probability = np.clip(f_model.predict_proba(x.iloc[np.flatnonzero(u_mask)])[:, 1], 1e-7, 1-1e-7)
        e_probability = np.clip(e_model.predict_proba(propensity_x.iloc[np.flatnonzero(u_mask)])[:, 1], 1e-7, 1-1e-7)
        e_probability = np.minimum(decay * e_probability, 1-1e-7)
        posterior = (f_probability * (1-e_probability)) / (
            f_probability * (1-e_probability) + 1-f_probability)
        q[u_mask] = posterior
        q[reviewed_u] = known_y[reviewed_u]
        q[p_mask] = 1.0

        # Крок M для f: CrossEntropy приймає очікуване значення класу q як
        # м'яку ціль, що дає змогу не збільшувати матрицю навчання втричі.
        f_y = q.copy()
        f_y[p_mask] = 1.0
        f_model = _fit_weighted_catboost(x, f_y, np.ones(len(x)), cfg, params,
                                         loss="CrossEntropy")

        # Крок M для e: механізм історичного присвоєння мітки навчається лише
        # на очікуваних істинно позитивних випадках. Для P мітку було присвоєно, для U — ні.
        e_positions = np.flatnonzero(p_mask | (u_mask & (q > 1e-10)))
        e_y = observed_s[e_positions]
        e_w = np.where(p_mask[e_positions], 1.0, q[e_positions])
        e_model = _fit_weighted_catboost(propensity_x.iloc[e_positions], e_y, e_w, cfg, params)

        current_q = q[u_mask & ~reviewed_u]
        change = float(np.mean(np.abs(current_q - previous_q))) if len(current_q) else 0.0
        stable_rounds = stable_rounds + 1 if change <= tolerance else 0
        if stable_rounds >= patience:
            converged = True
            break
        previous_q = current_q.copy()

    return f_model, e_model, {"em_iterations":iterations, "em_converged":converged,
                              "mean_unreviewed_positive_posterior":float(np.mean(q[u_mask & ~reviewed_u]))
                              if np.any(u_mask & ~reviewed_u) else None}


def choose_threshold(target, score, weights=None, strategy="fixed", fixed_threshold=0.5):
    """Вибрати фіксований поріг передбачення або оптимізувати F1/F2 на валідації."""
    if strategy not in ("fixed", "f1", "f2"):
        raise ValueError("threshold_strategy має бути fixed, f1 або f2")
    if not 0 <= fixed_threshold <= 1:
        raise ValueError("fixed_threshold має бути у межах [0, 1]")
    if np.sum(target) == 0: return None, None
    if strategy == "fixed":
        threshold = float(fixed_threshold)
        predicted = np.asarray(score) >= threshold
        precision = precision_score(target, predicted, sample_weight=weights, zero_division=0)
        recall = recall_score(target, predicted, sample_weight=weights, zero_division=0)
        f1 = f1_score(target, predicted, sample_weight=weights, zero_division=0)
        f2 = (5 * precision * recall / (4 * precision + recall)
              if 4 * precision + recall else 0.0)
        return threshold, {"recall":float(recall), "precision":float(precision),
                           "f1":float(f1), "f2":float(f2),
                           "selection_score":float(f1), "threshold_strategy":strategy}
    precision, recall, thresholds = precision_recall_curve(target, score, sample_weight=weights)
    if not len(thresholds): return None, None
    precision, recall = precision[:-1], recall[:-1]
    if strategy in ("f1", "f2"):
        beta_squared = 1 if strategy == "f1" else 4
        scores = ((1 + beta_squared) * precision * recall /
                  np.maximum(beta_squared * precision + recall, 1e-15))
        best = np.flatnonzero(scores >= scores.max() - 1e-12)
        # За однакових значень F-міри перевагу віддаємо вищому порогу, щоб зменшити кількість спрацювань.
    i = int(best[-1])
    selected_precision, selected_recall = float(precision[i]), float(recall[i])
    f1 = (2 * selected_precision * selected_recall /
          max(selected_precision + selected_recall, 1e-15))
    f2 = (5 * selected_precision * selected_recall /
          max(4 * selected_precision + selected_recall, 1e-15))
    selection_score = f1 if strategy == "f1" else f2 if strategy == "f2" else selected_precision
    return float(thresholds[i]), {"recall":selected_recall, "precision":selected_precision,
                                  "f1":float(f1), "f2":float(f2),
                                  "selection_score":float(selection_score),
                                  "threshold_strategy":strategy}


def reviewed_data(part_df, tag, answers, part):
    """P та випадково перевірені U з вагами відбору; неперевірені U не використовуються."""
    positive = part_df.loc[part_df[f"y_{tag}"].eq(1), ["row_id"]].copy()
    positive["target"] = 1; positive["weight"] = 1.0
    audited = answers[(answers.tag == tag) & (answers.split == part)].copy()
    audited = audited[audited.row_id.isin(part_df.row_id)]
    if audited.empty or audited.target.isna().any():
        raise ValueError("немає повних відповідей незалежного аудиту")
    area_field = f"filter_{tag}"
    if part_df[area_field].isna().any(): raise ValueError("частина фільтрів невідома")
    for area, flag in (("A", 1), ("B", 0)):
        if ((part_df[area_field] == flag) & (part_df[f"y_{tag}"] == 0)).any() and not (audited.area == area).any():
            raise ValueError(f"не вистачає аудиту U-{part}-{area}")
    if (audited.selection_probability <= 0).any(): raise ValueError("невідома ймовірність відбору")
    checked = audited[["row_id", "target"]].copy()
    checked["weight"] = 1/audited.selection_probability.to_numpy()
    joined = pd.concat([positive, checked], ignore_index=True).set_index("row_id")
    if joined.index.duplicated().any(): raise ValueError("повторні рядки перевірки")
    return joined


def stage_train(cfg, run):
    if (run / "model" / "model.joblib").exists():
        raise FileExistsError("Модель цього запуску вже існує. Для нового навчання створіть новий run.")
    if cfg["research_stage"] not in (1, 2): raise ValueError("research_stage має бути 1 або 2")
    # На етапі 2 завжди використовується CatBoost; PU-метод вибирається окремо.
    if cfg["research_stage"] == 2:
        cfg["model_type"] = "catboost"
    if cfg.get("feature_importance_selector") not in (0, 1):
        raise ValueError("feature_importance_selector має бути 0 або 1")
    if cfg.get("class_balance_selector") not in (0, 1):
        raise ValueError("class_balance_selector має бути 0 або 1")
    if cfg.get("model_type") not in BALANCE_METHODS:
        raise ValueError("model_type має бути lightgbm або catboost")
    if cfg["research_stage"] == 1 and cfg["class_balance_selector"] == 1:
        # Зупинити виконання до аудиту й підготовки даних, якщо метод вибраної моделі некоректний.
        class_balance_options(cfg, 1, 1)
    stage_audit(cfg, run)
    stage2_method = cfg.get("stage2_method", "nnpu")
    if cfg["research_stage"] == 2 and stage2_method not in ("nnpu", "sar_pu"):
        raise ValueError("stage2.method має бути nnpu або sar_pu")
    if cfg["research_stage"] == 2:
        if cfg["feature_importance_selector"] != 0:
            raise ValueError("Автоматичний відбір ознак за importance доступний лише на етапі 1")
    df = read_prepared(run)
    train = df[df.split == "train"].copy()
    val = df[(df.split == "validation") & (df.validation_role == "tuning")].copy()
    calibration = df[(df.split == "validation") & (df.validation_role == "calibration")].copy()
    test = df[df.split == "test"].copy()
    features = [c for c in df.columns if c not in EXCLUDED and not c.startswith(("filter_", "y_"))]
    constants = [c for c in features if train[c].nunique(dropna=False) <= 1]
    features = [c for c in features if c not in constants]
    if not features: raise ValueError("Немає неконстантних ознак")
    pd.DataFrame({"feature":constants}).to_csv(run / "constant_features.csv", index=False)
    xtr, (xval, xcal, xtest), encoder = encode_train(train, [val, calibration, test], features)
    selected = list(features)
    selected_by_tag = {} if cfg["feature_importance_selector"] == 1 else None
    answers_path = run / "audit_answers.csv"
    answers = pd.read_csv(answers_path) if cfg["research_stage"] == 2 and answers_path.exists() else None
    train_by_id, val_by_id = train.set_index("row_id"), val.set_index("row_id")
    models, propensity_models, thresholds, results = {}, {}, {}, []
    importance_rows, selection_candidate_rows, selection_summary_rows = [], [], []
    sar_propensity_rows = []
    scores_val = calibration[["row_id", "project", "platform_id"]].copy()
    scores_test = test[["row_id", "project", "platform_id"]].copy()
    for tag in TAGS:
        tag_features = selected_by_tag.get(tag, selected) if selected_by_tag is not None else selected
        tag_xtr, tag_xval, tag_xcal, tag_xtest = xtr[tag_features], xval[tag_features], xcal[tag_features], xtest[tag_features]
        y = train[f"y_{tag}"].astype(int); yv = val[f"y_{tag}"].astype(int)
        np_, nu = int(y.sum()), int((1-y).sum())
        if cfg["research_stage"] == 1 and (not np_ or not nu):
            results.append({"tag":tag,"status":"пропущено: відсутній один із спостережуваних класів"}); continue
        fit_x, fit_y, valid_x, valid_y = tag_xtr, y.to_numpy(), tag_xval, yv.to_numpy()
        nnpu_prior = None
        chosen_propensity_model, chosen_sar_info = None, None
        fit_weights = np.ones(len(y)); val_weights = np.ones(len(yv))
        if cfg["research_stage"] == 2:
            try:
                if answers is None: raise ValueError("відсутні результати аудиту")
                train_review = reviewed_data(train, tag, answers, "train")
                val_review = reviewed_data(val, tag, answers, "validation")
                fit_positions = train_by_id.index.get_indexer(train_review.index)
                valid_positions = val_by_id.index.get_indexer(val_review.index)
                fit_y = train_review.target.to_numpy(dtype=int)
                fit_weights = train_review.weight.to_numpy(dtype=float)
                valid_x, valid_y = tag_xval.iloc[valid_positions], val_review.target.to_numpy(dtype=int)
                val_weights = val_review.weight.to_numpy(dtype=float)
                if len(np.unique(fit_y)) < 2 or not valid_y.sum():
                    raise ValueError("недостатньо перевірених прикладів для навчання чи validation")
                if stage2_method == "sar_pu":
                    known_y = np.full(len(train), np.nan, dtype=float)
                    known_y[y.to_numpy() == 1] = 1.0
                    audit_train = answers[(answers.tag == tag) & (answers.split == "train")].copy()
                    audit_positions = train_by_id.index.get_indexer(audit_train.row_id)
                    if (audit_positions < 0).any():
                        raise ValueError("SAR-PU: train-аудит містить невідомі row_id")
                    known_y[audit_positions] = audit_train.target.to_numpy(dtype=float)
                    audit_weights = np.ones(len(train), dtype=float)
                    audit_weights[audit_positions] = 1 / audit_train.selection_probability.to_numpy(dtype=float)
                    propensity_names = _sar_propensity_features(cfg, tag, features)
                    propensity_x = xtr[propensity_names]
            except ValueError as exc:
                results.append({"tag":tag,"status":f"пропущено етап 2: {exc}"}); continue
        if cfg["research_stage"] == 1:
            balance_params, effective_balance_method, positive_class_weight = class_balance_options(cfg, np_, nu)
        else:
            balance_params, effective_balance_method, positive_class_weight = {}, f"not_applicable_{stage2_method}", None
        grids = (list(ParameterGrid(cfg["optimization"][cfg["model_type"]]))[:cfg["optimization"]["max_candidates"]]
                 if cfg["optimization"]["enabled"] else [{}])
        chosen_model, chosen, best_metric = None, None, None
        selected_importance_threshold = None
        threshold_strategy = cfg.get("threshold_strategy", "fixed")
        fixed_threshold = cfg.get("fixed_threshold", 0.5)
        if cfg["research_stage"] == 1 and cfg["feature_importance_selector"] == 1:
            if not grids:
                raise ValueError("Для відбору ознак потрібен хоча б один варіант параметрів")
            pilot_params = grids[0]
            pilot_model = fit_baseline_binary(cfg, tag_xtr, y.to_numpy(), pilot_params, balance_params)
            values, importance_method = model_feature_importance(pilot_model, cfg["model_type"])
            fractions, feature_candidates = importance_threshold_candidates(tag_features, values)
            importance_rows.extend({"tag":tag,"feature":feature,"importance":float(value),
                                    "importance_fraction":float(fraction),
                                    "importance_method":importance_method,
                                    "importance_scope":"all_features_selection_model"}
                                   for feature, value, fraction in zip(tag_features, values, fractions))
            best_selection_key = None
            selected_candidate_info = None
            for candidate_id, (cutoff, candidate_features) in enumerate(feature_candidates, 1):
                candidate_best_key = None
                for params in grids:
                    candidate_model = fit_baseline_binary(
                        cfg, tag_xtr[candidate_features], y.to_numpy(), params, balance_params)
                    candidate_scores = candidate_model.predict_proba(tag_xval[candidate_features])[:, 1]
                    candidate_threshold, candidate_info = choose_threshold(
                        yv.to_numpy(), candidate_scores, val_weights,
                        strategy=threshold_strategy, fixed_threshold=fixed_threshold)
                    if candidate_info is None:
                        candidate_threshold = fixed_threshold
                        candidate_pred = candidate_scores >= candidate_threshold
                        candidate_precision = precision_score(yv, candidate_pred, zero_division=0)
                        candidate_recall = recall_score(yv, candidate_pred, zero_division=0)
                        candidate_f1 = f1_score(yv, candidate_pred, zero_division=0)
                        candidate_f2 = (5 * candidate_precision * candidate_recall /
                                        (4 * candidate_precision + candidate_recall)
                                        if 4 * candidate_precision + candidate_recall else 0.0)
                        candidate_selection_score = candidate_f2 if threshold_strategy == "f2" else candidate_f1
                    else:
                        candidate_pred = candidate_scores >= candidate_threshold
                        candidate_precision = candidate_info["precision"]
                        candidate_recall = candidate_info["recall"]
                        candidate_f1 = candidate_info["f1"]
                        candidate_f2 = candidate_info["f2"]
                        candidate_selection_score = candidate_info["selection_score"]
                    candidate_fp = int(np.sum((yv.to_numpy() == 0) & candidate_pred))
                    candidate_key = (float(candidate_selection_score), float(candidate_recall),
                                     -candidate_fp, -len(candidate_features))
                    row = {"tag":tag,"candidate_id":candidate_id,
                           "importance_fraction_cutoff":cutoff,
                           "feature_count":len(candidate_features),
                           "selected_features":"|".join(candidate_features),
                           "threshold_strategy":threshold_strategy,
                           "selection_metric":"f1" if threshold_strategy == "fixed" else threshold_strategy,
                           "prediction_threshold":candidate_threshold,
                           "validation_f1":candidate_f1,"validation_f2":candidate_f2,
                           "validation_selection_score":candidate_selection_score,
                           "validation_recall":candidate_recall,
                           "validation_precision":candidate_precision,"validation_FP":candidate_fp,
                           "class_balance_selector":cfg["class_balance_selector"],
                           "imbalance_method":effective_balance_method,
                           "positive_class_weight":positive_class_weight,
                           "parameters":json.dumps(params)}
                    selection_candidate_rows.append(row)
                    if candidate_best_key is None or candidate_key > candidate_best_key:
                        candidate_best_key = candidate_key
                        current_choice = (candidate_model, params, candidate_features, cutoff,
                                          candidate_threshold, candidate_info, candidate_scores,
                                          candidate_fp, candidate_precision, candidate_recall,
                                          candidate_f1, candidate_f2, candidate_selection_score)
                if candidate_best_key is not None and (best_selection_key is None or candidate_best_key > best_selection_key):
                    best_selection_key = candidate_best_key
                    selected_candidate_info = current_choice
            (chosen_model, chosen_params, tag_features, selected_importance_threshold,
             selection_prediction_threshold,
             selection_info, selection_scores, selection_fp, selection_precision,
             selection_recall, selection_f1, selection_f2,
             selection_selection_score) = selected_candidate_info
            chosen = (effective_balance_method, chosen_params)
            best_metric = selection_selection_score
            selected_by_tag[tag] = list(tag_features)
            tag_xtr, tag_xval = xtr[tag_features], xval[tag_features]
            tag_xcal, tag_xtest = xcal[tag_features], xtest[tag_features]
            fit_x, valid_x = tag_xtr, tag_xval
            selection_summary_rows.append({"tag":tag,"importance_fraction_cutoff":selected_importance_threshold,
                                           "feature_count":len(tag_features),
                                           "threshold_strategy":threshold_strategy,
                                           "selection_metric":"f1" if threshold_strategy == "fixed" else threshold_strategy,
                                           "prediction_threshold":selection_prediction_threshold,
                                           "validation_f1":selection_f1,"validation_recall":selection_recall,
                                           "validation_f2":selection_f2,
                                           "validation_selection_score":selection_selection_score,
                                           "validation_precision":selection_precision,
                                           "validation_FP":selection_fp,
                                           "class_balance_selector":cfg["class_balance_selector"],
                                           "imbalance_method":effective_balance_method,
                                           "positive_class_weight":positive_class_weight,
                                           "tie_break_rule":"selected metric, then recall, fewer FP, fewer features"})
        else:
            for params in grids:
                if cfg["research_stage"] == 2:
                    if stage2_method == "nnpu":
                        positive = fit_y == 1
                        model, nnpu_prior = fit_nnpu(xtr, fit_positions[positive], fit_weights[positive], cfg, params)
                        propensity_model, sar_info = None, None
                    else:
                        model, propensity_model, sar_info = fit_sar_pu(
                            xtr, propensity_x, y.to_numpy(), known_y, audit_weights, cfg, params)
                    chosen_balance_method = f"not_applicable_{stage2_method}"
                else:
                    model = fit_baseline_binary(cfg, fit_x, fit_y, params, balance_params)
                    propensity_model, sar_info = None, None
                    chosen_balance_method = effective_balance_method
                pred = model.predict_proba(valid_x)[:,1]
                threshold, info = choose_threshold(
                    valid_y, pred, val_weights,
                    strategy=threshold_strategy, fixed_threshold=fixed_threshold)
                metric = info["selection_score"] if info else -1.0
                if chosen_model is None or metric > best_metric:
                    best_metric, chosen, chosen_model = metric, (chosen_balance_method, params), model
                    chosen_propensity_model, chosen_sar_info = propensity_model, sar_info
            if cfg["feature_importance_selector"] == 0 and cfg["research_stage"] == 1:
                values, importance_method = model_feature_importance(chosen_model, cfg["model_type"])
                fractions, _ = importance_threshold_candidates(tag_features, values)
                importance_rows.extend({"tag":tag,"feature":feature,"importance":float(value),
                                        "importance_fraction":float(fraction),
                                        "importance_method":importance_method,
                                        "importance_scope":"final_all_features_model"}
                                       for feature, value, fraction in zip(tag_features, values, fractions))
        pred = chosen_model.predict_proba(valid_x)[:,1]
        threshold, info = choose_threshold(
            valid_y, pred, val_weights,
            strategy=threshold_strategy, fixed_threshold=fixed_threshold)
        if valid_y.sum():
            pr, rc, cuts = precision_recall_curve(valid_y, pred, sample_weight=val_weights)
            if len(cuts):
                idx = np.unique(np.linspace(0, len(cuts)-1, min(len(cuts), 101), dtype=int))
                sorted_scores = np.sort(pred)
                pd.DataFrame({"threshold":cuts[idx], "precision":pr[idx], "recall":rc[idx],
                              "selected_rows":len(pred)-np.searchsorted(sorted_scores, cuts[idx], side="left"),
                              "selected_fraction":(len(pred)-np.searchsorted(sorted_scores, cuts[idx], side="left"))/len(pred),
                              "weighted_selected_fraction":[float(np.average(pred >= cut, weights=val_weights)) for cut in cuts[idx]]}).to_csv(run / f"validation_thresholds_{tag}.csv", index=False)
        models[tag] = chosen_model; thresholds[tag] = threshold
        if chosen_propensity_model is not None:
            propensity_models[tag] = chosen_propensity_model
            sar_propensity_rows.append({"tag":tag, "propensity_features":"|".join(propensity_names),
                                        "em_iterations":chosen_sar_info["em_iterations"],
                                        "em_converged":chosen_sar_info["em_converged"],
                                        "mean_unreviewed_positive_posterior":chosen_sar_info["mean_unreviewed_positive_posterior"]})
        scores_val[tag] = chosen_model.predict_proba(tag_xcal)[:,1]; scores_test[tag] = chosen_model.predict_proba(tag_xtest)[:,1]
        results.append({"tag":tag,"status":"навчено" if info else "навчено; недостатньо позитивних validation для порога",
                        "feature_count":len(tag_features),
                        "feature_importance_fraction_cutoff":selected_importance_threshold,
                        "feature_selection_metric":(("f1" if threshold_strategy == "fixed" else threshold_strategy)
                                                     if cfg["feature_importance_selector"] == 1 else None),
                        "feature_selection_validation_f1":selection_f1 if cfg["feature_importance_selector"] == 1 and cfg["research_stage"] == 1 else None,
                        "feature_selection_validation_f2":selection_f2 if cfg["feature_importance_selector"] == 1 and cfg["research_stage"] == 1 else None,
                        "feature_selection_validation_score":selection_selection_score if cfg["feature_importance_selector"] == 1 and cfg["research_stage"] == 1 else None,
                        "class_balance_selector":cfg["class_balance_selector"] if cfg["research_stage"] == 1 else 0,
                        "imbalance_method":chosen[0],
                        "positive_class_weight":positive_class_weight if cfg["research_stage"] == 1 else None,
                        "n_P":np_,"n_U":nu,
                        "parameters":json.dumps(chosen[1]),"threshold":threshold,
                        "validation_recall":info["recall"] if info else None,
                        "validation_precision":info["precision"] if info else None,
                        "validation_f1":info["f1"] if info else None,
                        "validation_f2":info["f2"] if info else None,
                        "validation_selection_score":info["selection_score"] if info else None,
                        "threshold_strategy":threshold_strategy,
                        "research_stage":cfg["research_stage"], "nnpu_prior":nnpu_prior})
        write_status(run, "Навчання", f"{tag}: P={np_}, U={nu}, поріг={threshold}")
    if selected_by_tag is not None:
        selected = [feature for feature in features
                    if any(feature in tag_features for tag_features in selected_by_tag.values())]
    selected_report = ([{"tag":tag,"feature":feature} for tag in TAGS for feature in selected]
                       if selected_by_tag is None else
                       [{"tag":tag,"feature":feature}
                        for tag, tag_features in selected_by_tag.items() for feature in tag_features])
    pd.DataFrame(selected_report).to_csv(run / "selected_features.csv", index=False)
    corr = xtr[selected].corr(); corr.to_csv(run / "feature_correlation.csv")
    plt.figure(figsize=(max(8, len(selected)*.3), max(6, len(selected)*.3)))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm"); plt.colorbar()
    plt.xticks(range(len(selected)), selected, rotation=90, fontsize=6)
    plt.yticks(range(len(selected)), selected, fontsize=6)
    savefig(run / "feature_correlation.png")
    if selection_candidate_rows:
        pd.DataFrame(selection_candidate_rows).to_csv(run / "feature_selection_candidates.csv", index=False)
        pd.DataFrame(selection_summary_rows).to_csv(run / "feature_selection_summary.csv", index=False)
    pd.DataFrame(results).to_csv(run / "training_summary.csv", index=False)
    if sar_propensity_rows:
        pd.DataFrame(sar_propensity_rows).to_csv(run / "sar_propensity_summary.csv", index=False)
    if not models:
        raise RuntimeError("Жодну модель не навчено: перевірте навчальні дані або звіт аудиту та training_summary.csv")
    model_dir = run / "model"; model_dir.mkdir(exist_ok=True)
    package = {"models":models,"encoding":encoder,"selected_features":selected,
               "selected_features_by_tag":selected_by_tag,"thresholds":thresholds,
               "propensity_models":propensity_models,
               "tags":TAGS,"model_type":cfg["model_type"] if cfg["research_stage"] == 1 else f"catboost_{stage2_method}",
               "stage2_method":stage2_method if cfg["research_stage"] == 2 else None,
               "research_stage":cfg["research_stage"],
               "score_meaning":"некалібрований скор; не гарантована ймовірність істинного тегу"}
    joblib.dump(package, model_dir / "model.joblib")
    if cfg["research_stage"] == 1:
        pd.DataFrame(importance_rows).to_csv(run / "feature_importances.csv", index=False)
    save_json(model_dir / "configuration.json", cfg)
    scores_val.to_csv(run / "scores_validation.csv", index=False); scores_test.to_csv(run / "scores_test.csv", index=False)
    write_status(run, "Модель", f"Збережено {len(models)} класифікаторів у {model_dir}")

# %% 7. Діагностика навчання етапу 1: Logloss та Average Precision за ітераціями

def _encode_learning_curve_data(frame, encoding, features):
    """Кодувати діагностичні дані параметрами, навченими на train."""
    result = pd.DataFrame(index=frame.index)
    numeric = [name for name in encoding["numeric"] if name in features]
    categorical = [name for name in encoding["categorical"] if name in features]
    if numeric:
        result[numeric] = (frame[numeric].replace([np.inf, -np.inf], np.nan)
                           .fillna(encoding["medians"]).astype("float32"))
    if categorical:
        result[categorical] = encoding["encoder"].transform(
            frame[categorical].fillna("<пропущено>").astype(str)).astype("float32")
    return result[features]


def stage_learning_curves(cfg, run):
    """Зберегти криві навчання лише для базових моделей етапу 1."""
    if cfg["research_stage"] != 1:
        raise ValueError("Графіки Logloss та Average Precision доступні лише для етапу 1")
    settings = cfg.get("learning_curves", {})
    if not settings.get("enabled", True):
        write_status(run, "Графіки навчання", "Вимкнено в конфігурації")
        return
    model_path = run / "model" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Спочатку навчіть модель етапу 1: model/model.joblib не знайдено")
    period = int(settings.get("eval_period", 5))
    max_train_rows = settings.get("max_train_rows", 100000)
    max_train_rows = None if max_train_rows is None else int(max_train_rows)
    if period < 1 or (max_train_rows is not None and max_train_rows < 1):
        raise ValueError("learning_curves.eval_period і max_train_rows мають бути додатними")

    bundle = joblib.load(model_path)
    if bundle.get("research_stage") != 1:
        raise ValueError("Збережена модель не належить до етапу 1")
    df = read_prepared(run)
    train = df[df["split"].eq("train")]
    validation = df[df["split"].eq("validation") & df["validation_role"].eq("tuning")]
    if train.empty or validation.empty:
        raise ValueError("Для кривих навчання потрібні train і tuning-validation")
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=int(cfg["seed"]))
    encoding = bundle["encoding"]
    tags = bundle["tags"]
    all_features = bundle["selected_features"]
    x_train_all = _encode_learning_curve_data(train, encoding, all_features)
    x_validation_all = _encode_learning_curve_data(validation, encoding, all_features)

    from sklearn.metrics import average_precision_score, log_loss
    curves = []
    n_tags = len(bundle["models"])
    for tag_index, (tag, model) in enumerate(bundle["models"].items(), 1):
        tag_features = (bundle.get("selected_features_by_tag") or {}).get(tag, all_features)
        x_train = x_train_all[tag_features]
        x_validation = x_validation_all[tag_features]
        y_train = train[f"y_{tag}"].to_numpy(dtype=np.int8)
        y_validation = validation[f"y_{tag}"].to_numpy(dtype=np.int8)
        if hasattr(model, "tree_count_"):
            tree_count = int(model.tree_count_)
        else:
            tree_count = int(getattr(model, "n_estimators_", cfg["lightgbm"]["n_estimators"]))
        checkpoints = list(range(period, tree_count, period))
        checkpoints.append(tree_count)

        if hasattr(model, "staged_predict_proba"):
            train_predictions = model.staged_predict_proba(
                x_train, eval_period=period, thread_count=int(cfg.get("catboost", {}).get("thread_count", 2)))
            validation_predictions = model.staged_predict_proba(
                x_validation, eval_period=period, thread_count=int(cfg.get("catboost", {}).get("thread_count", 2)))
        else:
            train_predictions = (model.predict_proba(x_train, num_iteration=iteration)
                                 for iteration in checkpoints)
            validation_predictions = (model.predict_proba(x_validation, num_iteration=iteration)
                                      for iteration in checkpoints)

        for iteration, train_probability, validation_probability in zip(
                checkpoints, train_predictions, validation_predictions):
            train_score = train_probability[:, 1]
            validation_score = validation_probability[:, 1]
            curves.append({
                "tag": tag,
                "iteration": iteration,
                "train_rows": len(y_train),
                "validation_rows": len(y_validation),
                "validation_positive_rows": int(y_validation.sum()),
                "train_logloss": log_loss(y_train, train_score, labels=[0, 1]),
                "validation_logloss": log_loss(y_validation, validation_score, labels=[0, 1]),
                "validation_average_precision": (average_precision_score(y_validation, validation_score)
                                                 if y_validation.sum() else np.nan),
            })
        write_status(run, "Графіки навчання", f"{tag}: обчислено метрики до {tree_count} ітерацій ({tag_index}/{n_tags})")

    curves_df = pd.DataFrame(curves)
    curves_df.to_csv(run / "learning_curves.csv", index=False)
    tags_with_models = list(bundle["models"])
    columns = min(4, max(1, len(tags_with_models)))
    rows = math.ceil(len(tags_with_models) / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.1 * rows), squeeze=False)
    for ax, tag in zip(axes.flat, tags_with_models):
        values = curves_df[curves_df["tag"].eq(tag)]
        best = values.loc[values["validation_logloss"].idxmin()]
        ax.plot(values["iteration"], values["train_logloss"], label=f"Train (n={int(values.train_rows.iloc[0]):,})",
                color="#2563eb", linewidth=1.7)
        ax.plot(values["iteration"], values["validation_logloss"], label="Validation (tuning)",
                color="#e4572e", linewidth=1.7)
        ax.axvline(best["iteration"], color="#e4572e", linestyle=":", linewidth=1)
        ax.set_title(f"{tag} · мін. validation Logloss на {int(best['iteration'])}")
        ax.set_xlabel("Кількість ітерацій (дерев)")
        ax.set_ylabel("Logloss (менше — краще)")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(tags_with_models):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Криві навчання: Logloss на train і tuning-validation", y=1.035, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(run / "learning_curves_logloss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.1 * rows), squeeze=False)
    for ax, tag in zip(axes.flat, tags_with_models):
        values = curves_df[curves_df["tag"].eq(tag)]
        positives = int(values["validation_positive_rows"].iloc[0])
        prevalence = positives / int(values["validation_rows"].iloc[0])
        ax.plot(values["iteration"], values["validation_average_precision"],
                label="Average Precision на validation", color="#178f6b", linewidth=1.7)
        ax.axhline(prevalence, color="#64748b", linestyle="--", linewidth=1,
                   label="Випадкове ранжування (частка позитивних)")
        best_index = values["validation_average_precision"].idxmax()
        if pd.isna(best_index):
            ax.set_title(f"{tag} · немає позитивних на validation")
        else:
            best = curves_df.loc[best_index]
            ax.axvline(best["iteration"], color="#178f6b", linestyle=":", linewidth=1)
            ax.set_title(f"{tag} · макс. AP {best['validation_average_precision']:.3f} на {int(best['iteration'])}")
        ax.set_xlabel("Кількість ітерацій (дерев)")
        ax.set_ylabel("Average Precision (більше — краще)")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(tags_with_models):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Average Precision (PR-AUC/AP) на tuning-validation", y=1.035, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(run / "learning_curves_average_precision.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    (run / "learning_curves_readme.md").write_text(
        "# Графіки навчання — етап 1\n\n"
        "`learning_curves_logloss.png` показує Logloss для train і tuning-validation; нижче значення краще. "
        "`learning_curves_average_precision.png` показує Average Precision (AP) на tuning-validation без порога; "
        "вище значення краще. Пунктирна горизонтальна лінія на графіку AP — частка позитивних міток, "
        "орієнтир для випадкового ранжування.\n\n"
        f"Train-крива розрахована на вибірці до {max_train_rows or 'усіх'} рядків із train; "
        f"validation-криві — на всіх {len(validation):,} рядках tuning-validation. "
        f"Метрики обчислюються з кроком {period} ітерацій за префіксами вже навчених дерев. "
        "Частина calibration і test не використовується. Оцінюються лише спостережувані мітки. "
        "Цей блок не перенавчає модель і не змінює пороги.\n",
        encoding="utf-8")
    write_status(run, "Графіки навчання", "Збережено learning_curves_logloss.png і learning_curves_average_precision.png")


# %% 8. Оцінювання спостереженої розмітки та шаблони аудиту скорингу

def stage_evaluate(cfg, run):
    if (run / "metrics_observed_per_teg.csv").exists():
        raise FileExistsError("Оцінювання цього запуску вже існує; створіть новий run для повторного оцінювання")
    df = read_prepared(run).set_index("row_id")
    scores = pd.read_csv(run / "scores_test.csv")
    package = joblib.load(run / "model" / "model.joblib")
    stage1_metrics = cfg["research_stage"] == 1
    pr_auc_column = "PR_AUC_AP_observed" if stage1_metrics else "AP_observed"
    rows = []
    targets_all, predictions_all = [], []
    for tag in package["models"]:
        truth = df.loc[scores.row_id, f"y_{tag}"].astype(int).to_numpy()
        score = scores[tag].to_numpy(); threshold = package["thresholds"][tag]
        if threshold is None:
            row = {"tag":tag,"n":len(truth),"observed_positive":int(truth.sum()),
                   "TP_observed":0,"FP_observed":0,"FN_observed":int(truth.sum()),
                   "threshold":None,"selected_rows":0,"selected_fraction":0.0,
                   "precision_observed":0.0,"recall_observed":0.0,
                   "F1_observed":0.0,"F2_observed":0.0}
            row[pr_auc_column] = (average_precision_score(truth, score)
                                  if len(np.unique(truth)) == 2 else np.nan)
            rows.append(row)
            continue
        pred = score >= threshold
        tp = int(np.sum((truth == 1) & pred))
        fp = int(np.sum((truth == 0) & pred))
        fn = int(np.sum((truth == 1) & ~pred))
        row = {"tag":tag,"n":len(truth),"observed_positive":int(truth.sum()),"threshold":threshold,
               "TP_observed":tp,"FP_observed":fp,"FN_observed":fn,
               "selected_rows":int(pred.sum()),"selected_fraction":float(pred.mean()),
               "precision_observed":precision_score(truth, pred, zero_division=0),
               "recall_observed":recall_score(truth, pred, zero_division=0),
               "F1_observed":f1_score(truth, pred, zero_division=0),
               "F2_observed":(5 * tp / (5 * tp + fp + 4 * fn)
                              if 5 * tp + fp + 4 * fn else 0.0),
               "ROC_AUC_observed":roc_auc_score(truth, score) if len(np.unique(truth)) == 2 else np.nan}
        row[pr_auc_column] = (average_precision_score(truth, score)
                              if len(np.unique(truth)) == 2 else np.nan)
        rows.append(row)
        targets_all.append(truth); predictions_all.append(pred)
        if truth.sum():
            prec, rec, _ = precision_recall_curve(truth, score)
            plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall спостереженої розмітки")
            plt.ylabel("Precision спостереженої розмітки"); plt.title(tag); savefig(run / f"pr_observed_{tag}.png")
        cm = confusion_matrix(truth, pred, labels=[0,1]); plt.figure(); plt.imshow(cm, cmap="Blues"); plt.colorbar()
        for (i,j), value in np.ndenumerate(cm): plt.text(j,i,str(value),ha="center",va="center")
        plt.xlabel("Передбачена спостережена мітка"); plt.ylabel("Спостережена мітка"); plt.title(tag); savefig(run / f"confusion_observed_{tag}.png")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run / "metrics_observed_per_teg.csv", index=False)
    ap_values = metrics[pr_auc_column].dropna() if pr_auc_column in metrics else pd.Series(dtype=float)
    if stage1_metrics:
        aggregate = {
            "research_stage": cfg["research_stage"],
            "model_type": package["model_type"],
            "feature_importance_selector": cfg.get("feature_importance_selector"),
            "class_balance_selector": cfg.get("class_balance_selector"),
            "imbalance_method": (cfg.get("imbalance_method", {}).get(cfg.get("model_type"))
                                 if cfg.get("class_balance_selector") == 1 else "none"),
            "threshold_strategy": cfg.get("threshold_strategy", "fixed"),
            "fixed_threshold": cfg.get("fixed_threshold", 0.5),
            "n_tags": int(len(metrics)),
            "n_tags_with_both_classes_for_PR_AUC": int(len(ap_values)),
            "macro_F1_observed": float(metrics.F1_observed.mean()) if "F1_observed" in metrics and len(metrics) else None,
            "macro_F2_observed": float(metrics.F2_observed.mean()) if "F2_observed" in metrics and len(metrics) else None,
            "macro_PR_AUC_AP_observed": float(ap_values.mean()) if len(ap_values) else None,
            "micro_F1_observed": (f1_score(np.concatenate(targets_all), np.concatenate(predictions_all), zero_division=0)
                                   if targets_all else None),
        }
    else:
        aggregate = {
            "macro_F1_observed": float(metrics.F1_observed.mean()) if "F1_observed" in metrics and len(metrics) else None,
            "micro_F1_observed": (f1_score(np.concatenate(targets_all), np.concatenate(predictions_all), zero_division=0)
                                   if targets_all else None),
        }
    save_json(run / "metrics_observed_aggregate.json", aggregate)
    if stage1_metrics:
        pd.DataFrame([aggregate]).to_csv(run / "metrics_observed_aggregate.csv", index=False)
        report_note = (
            "PR_AUC_AP_observed — Average Precision (PR-AUC без інтерполяції) окремо для кожного тегу; "
            "macro_PR_AUC_AP_observed — просте середнє за тегами, де в test присутні обидва класи.\n"
            "Macro F1 і Macro F2 — просте середнє відповідних метрик по тегах.\n")
    else:
        report_note = ""
    (run / "metrics_readme.txt").write_text(
        "УВАГА: метрики порівнюють score зі СПОСТЕРЕЖЕНИМИ тегами, не з істинними класами. Статус U невідомий.\n"
        + report_note + metrics.to_string(index=False), encoding="utf-8")
    plt.figure(figsize=(12,5))
    if len(metrics): plt.bar(metrics.tag, metrics[pr_auc_column].fillna(0)); plt.xticks(rotation=60, ha="right")
    if stage1_metrics:
        plt.ylabel("PR-AUC (Average Precision), спостережені мітки")
        savefig(run / "metrics_observed_PR_AUC.png")
    else:
        plt.ylabel("AP щодо спостереженої розмітки")
        savefig(run / "metrics_observed_AP.png")
    plt.figure(figsize=(12,5))
    if "recall_observed" in metrics:
        plt.bar(metrics.tag, metrics.recall_observed.fillna(0)); plt.xticks(rotation=60, ha="right")
    plt.ylim(0, 1); plt.ylabel("Recall щодо спостереженої розмітки"); savefig(run / "metrics_observed_recall.png")
    metric_plot_columns = ("F1_observed", "F2_observed", "ROC_AUC_observed") if stage1_metrics else ("F1_observed", "ROC_AUC_observed")
    for column in metric_plot_columns:
        if column in metrics:
            plt.figure(figsize=(12, 5)); plt.bar(metrics.tag, metrics[column].fillna(0))
            plt.xticks(rotation=60, ha="right"); plt.ylim(0, 1); plt.ylabel(column)
            savefig(run / f"metrics_{column}.png")
    # Зберігаємо окрему статистику за областями, зокрема невідомими фільтрами.
    area_rows = []
    for tag in package["models"]:
        area = df.loc[scores.row_id, f"filter_{tag}"].to_numpy()
        truth = df.loc[scores.row_id, f"y_{tag}"].astype(int).to_numpy()
        for name, flag in (("A", 1), ("B", 0), ("unknown", None)):
            mask = pd.isna(area) if flag is None else pd.Series(area).eq(flag).fillna(False).to_numpy()
            threshold = package["thresholds"][tag]
            row = {"tag":tag,"area":name,"n":int(mask.sum()),"observed_positive":int(truth[mask].sum())}
            if mask.any() and threshold is not None:
                pred = scores[tag].to_numpy()[mask] >= threshold
                row.update(recall_observed=recall_score(truth[mask], pred, zero_division=0) if truth[mask].sum() else None,
                           precision_observed=precision_score(truth[mask], pred, zero_division=0), selected=int(pred.sum()))
            area_rows.append(row)
    pd.DataFrame(area_rows).to_csv(run / "metrics_observed_by_area.csv", index=False)
    if cfg["research_stage"] == 2:
        evaluate_reviewed_test(cfg, run, df, scores, package)
        create_score_audit(cfg, run, df)
    write_status(run, "Оцінювання", "Метрики стосуються спостережених тегів, не істинних класів")


def audited_intervals(y, pred, weights, groups, cfg):
    """Кластерний бутстреп за гравцями для метрик перевіреної вибірки."""
    table = pd.DataFrame({"group": groups, "tp": weights*((y == 1) & pred),
                          "fp": weights*((y == 0) & pred), "fn": weights*((y == 1) & ~pred)})
    totals = table.groupby("group")[["tp", "fp", "fn"]].sum().to_numpy()
    if len(totals) < 2: return {}
    rng = np.random.default_rng(cfg["seed"])
    draws = np.empty((cfg["audit"]["bootstrap_iterations"], 2), dtype=float)
    for i in range(len(draws)):
        tp, fp, fn = totals[rng.integers(0, len(totals), len(totals))].sum(axis=0)
        draws[i] = (tp/(tp+fn) if tp+fn else np.nan, tp/(tp+fp) if tp+fp else np.nan)
    alpha = (1-cfg["audit"]["confidence"])/2
    result = {"ci_method": "наближений bootstrap за гравцями з фіксованими вагами відбору"}
    for i, metric in enumerate(("recall", "precision")):
        finite = draws[:, i][np.isfinite(draws[:, i])]
        if len(finite):
            result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = np.quantile(finite, [alpha, 1-alpha])
    return result


def evaluate_reviewed_test(cfg, run, df, scores, package):
    answers_path = run / "audit_answers.csv"
    if not answers_path.exists():
        write_status(run, "Аудит test", "Немає перевірених відповідей для незалежної оцінки")
        return
    answers = pd.read_csv(answers_path)
    test = df[df.split == "test"].reset_index()
    evaluated = [("stage2", scores, package)]
    if cfg.get("baseline_run"):
        baseline = Path(cfg["baseline_run"])
        old_identity = json.loads((baseline / "data_identity.json").read_text())
        new_identity = json.loads((run / "data_identity.json").read_text())
        if any(old_identity.get(k) != new_identity.get(k) for k in ("source_sha256", "split_sha256")):
            raise ValueError("Базовий запуск має інше джерело або розділення")
        evaluated.insert(0, ("baseline", pd.read_csv(baseline / "scores_test.csv"), joblib.load(baseline / "model" / "model.joblib")))
    rows = []
    for name, model_scores, model in evaluated:
        indexed = model_scores.set_index("row_id")
        for tag in model["models"]:
            try:
                reviewed = reviewed_data(test, tag, answers, "test")
                if model["thresholds"].get(tag) is None: raise ValueError("поріг не визначено")
                meta = test.set_index("row_id").loc[reviewed.index]
                pred_score = indexed.loc[reviewed.index, tag].to_numpy()
                for area, flag in (("all", None), ("A", 1), ("B", 0)):
                    mask = np.ones(len(reviewed), dtype=bool) if flag is None else meta[f"filter_{tag}"].eq(flag).to_numpy()
                    y = reviewed.target.to_numpy(dtype=int)[mask]; w = reviewed.weight.to_numpy(dtype=float)[mask]
                    pred = pred_score[mask] >= model["thresholds"][tag]
                    row = {"model":name,"tag":tag,"area":area,"n_reviewed":int(mask.sum()),"n_positive":int(y.sum()),
                           "threshold":model["thresholds"][tag],"status":"оцінено за аудитом"}
                    if len(y):
                        row.update(recall=recall_score(y, pred, sample_weight=w, zero_division=0) if y.sum() else None,
                                   precision=precision_score(y, pred, sample_weight=w, zero_division=0),
                                   AP=average_precision_score(y, pred_score[mask], sample_weight=w) if y.sum() else None,
                                   F1=f1_score(y, pred, sample_weight=w, zero_division=0),
                                   ROC_AUC=roc_auc_score(y, pred_score[mask], sample_weight=w) if len(np.unique(y)) == 2 else None,
                                   weighted_selected_rows=float(np.sum(w*pred)),
                                   weighted_selected_fraction=float(np.average(pred, weights=w)))
                        row.update(audited_intervals(y, pred, w, group_ids(meta.iloc[np.flatnonzero(mask)]), cfg))
                    rows.append(row)
            except (ValueError, KeyError) as exc:
                rows.append({"model":name,"tag":tag,"status":f"недостатньо аудиту: {exc}"})
    report = pd.DataFrame(rows)
    report.to_csv(run / "metrics_audited_test.csv", index=False)
    if "recall" in report:
        plotted = report[(report.get("area") == "all") & report.recall.notna()]
        plt.figure(figsize=(12,5))
        for name, group in plotted.groupby("model"):
            plt.plot(group.tag, group.recall, marker="o", label=name)
        plt.xticks(rotation=60, ha="right"); plt.ylim(0, 1); plt.ylabel("Recall за незалежним аудитом")
        if len(plotted): plt.legend()
        savefig(run / "metrics_audited_recall.png")
    for column in ("AP", "F1", "ROC_AUC"):
        if column in report:
            plotted = report[(report.get("area") == "all") & report[column].notna()]
            plt.figure(figsize=(12, 5))
            for name, group in plotted.groupby("model"):
                plt.plot(group.tag, group[column], marker="o", label=name)
            plt.xticks(rotation=60, ha="right"); plt.ylim(0, 1); plt.ylabel(column)
            if len(plotted): plt.legend()
            savefig(run / f"metrics_audited_{column}.png")


def create_score_audit(cfg, run, df):
    folder = run / "score_audit_samples"; folder.mkdir(exist_ok=True)
    save_json(folder / "metadata.json", {"model_sha256": file_hash(run / "model" / "model.joblib"),
                                         "population": "U", "seed": cfg["seed"]})
    settings = cfg["calibration"]
    for part in ("validation", "test"):
        scores = pd.read_csv(run / f"scores_{part}.csv")
        for i, tag in enumerate(TAGS):
            if tag not in scores: continue
            pool = scores.loc[df.loc[scores.row_id, f"y_{tag}"].to_numpy() == 0].copy()
            if not len(pool): continue
            pool["area"] = df.loc[pool.row_id, f"filter_{tag}"].map({1:"A", 0:"B"}).to_numpy()
            pool = pool.dropna(subset=["area"])
            if not len(pool): continue
            pool["score_bin"] = pd.qcut(pool[tag], q=settings["audit_bins"], duplicates="drop")
            pieces = []
            for _, part_df in pool.groupby(["area", "score_bin"], observed=True):
                count = min(len(part_df), settings["per_bin"])
                selected = part_df.sample(n=count, random_state=cfg["seed"]+i).copy()
                selected["selection_probability"] = count/len(part_df)
                pieces.append(selected)
            if pieces:
                result = pd.concat(pieces); result["audit_result"] = ""; result["population"] = "U"
                result["audit_run"] = str(run.resolve())
                result["model_sha256"] = file_hash(run / "model" / "model.joblib")
                result.to_csv(folder / f"{part}_{tag}.csv", index=False)

# %% 8. Незалежна калібрувальна вибірка та перевірка на test

def read_score_audit(cfg, run, part, tag):
    """Зчитати лише відповіді з раніше згенерованої вибірки; ваги беремо з маніфесту."""
    source = Path(cfg["audit_dir"]) / "calibration" / f"{part}_{tag}.csv"
    manifest = run / "score_audit_samples" / f"{part}_{tag}.csv"
    if not source.exists() or not manifest.exists():
        return None
    answers = pd.read_csv(source)
    selected = pd.read_csv(manifest)
    if not {"row_id", "audit_result", "audit_run", "model_sha256"}.issubset(answers):
        raise ValueError(f"{source}: потрібні row_id, audit_result, audit_run та model_sha256 із шаблону")
    if not answers.audit_run.eq(str(run.resolve())).all() or not answers.model_sha256.eq(file_hash(run / "model" / "model.joblib")).all():
        raise ValueError(f"{source}: результати належать іншому запуску або іншій моделі")
    if answers.row_id.duplicated().any() or not answers.row_id.isin(selected.row_id).all():
        raise ValueError(f"{source}: невідомі або повторні рядки аудиту")
    merged = selected.drop(columns=["audit_result"], errors="ignore").merge(
        answers[["row_id", "audit_result"]], on="row_id", how="left", validate="one_to_one")
    answer_map = {"1": 1, "positive": 1, "позитивний": 1,
                  "0": 0, "negative": 0, "негативний": 0}
    merged["target"] = merged.audit_result.astype("string").str.strip().str.lower().map(answer_map)
    merged = merged.dropna(subset=["target"]).copy()
    if (merged.selection_probability <= 0).any(): raise ValueError("Нульова ймовірність відбору")
    merged["weight"] = 1 / merged.selection_probability
    return merged


def stage_calibrate(cfg, run):
    if cfg["research_stage"] == 1:
        write_status(run, "Калібрування", "Етап 1: аудит та калібрування за аудитом не виконуються")
        return
    package_path = run / "model" / "model.joblib"
    package = joblib.load(package_path)
    folder = run / "calibration"; folder.mkdir(exist_ok=True)
    model_hash = file_hash(package_path)
    metadata_path = run / "score_audit_samples" / "metadata.json"
    if not metadata_path.exists() or json.loads(metadata_path.read_text(encoding="utf-8"))["model_sha256"] != model_hash:
        raise ValueError("Вибірка аудиту не відповідає поточній моделі; повторіть етап оцінювання")
    calibrators, summary = {}, []
    for tag in package["models"]:
        row = {"tag": tag, "population": "U", "method": cfg["calibration"]["method"]}
        try:
            valid = read_score_audit(cfg, run, "validation", tag)
            if valid is None:
                row["status"] = "немає незалежного аудиту validation"; summary.append(row); continue
            counts = valid.target.value_counts()
            row.update(validation_positive=int(counts.get(1, 0)), validation_negative=int(counts.get(0, 0)))
            if min(counts.get(0, 0), counts.get(1, 0)) < cfg["calibration"]["min_per_class"]:
                row["status"] = "недостатньо перевірених випадків"; summary.append(row); continue
            score, target, weights = valid[tag].to_numpy(), valid.target.astype(int), valid.weight.to_numpy()
            if cfg["calibration"]["method"] == "sigmoid":
                calibrator = LogisticRegression(random_state=cfg["seed"], max_iter=1000)
                calibrator.fit(score.reshape(-1, 1), target, sample_weight=weights)
            elif cfg["calibration"]["method"] == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(score, target, sample_weight=weights)
            else: raise ValueError("Невідомий метод калібрування")
            calibrators[tag] = calibrator
            row["status"] = "навчено для U; незалежна перевірка не виконана"
            test = read_score_audit(cfg, run, "test", tag)
            if test is not None and test.target.nunique() == 2:
                raw = test[tag].to_numpy(); y = test.target.astype(int); weight = test.weight.to_numpy()
                prob = (calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
                        if cfg["calibration"]["method"] == "sigmoid" else calibrator.predict(raw))
                prob = np.clip(prob, 1e-7, 1-1e-7)
                row.update(status="навчено; перевірено на аудиті test", test_n=len(test),
                           brier=brier_score_loss(y, prob, sample_weight=weight),
                           log_loss=log_loss(y, prob, sample_weight=weight, labels=[0, 1]),
                           AP=average_precision_score(y, prob, sample_weight=weight))
                frame = pd.DataFrame({"probability":prob, "target":y, "weight":weight})
                frame["bin"] = pd.cut(frame.probability, np.linspace(0, 1, 11), include_lowest=True)
                points = []
                for _, group in frame.groupby("bin", observed=True):
                    points.append((np.average(group.probability, weights=group.weight),
                                   np.average(group.target, weights=group.weight)))
                plt.figure(); plt.plot([0, 1], [0, 1], "--", label="ідеальна калібровка")
                if points: plt.plot(*zip(*points), marker="o", label="аудит test")
                plt.xlabel("Калібрована ймовірність"); plt.ylabel("Частка позитивних"); plt.title(tag); plt.legend()
                savefig(folder / f"reliability_{tag}.png")
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            row["status"] = f"помилка аудиту: {exc}"
        summary.append(row)
    pd.DataFrame(summary).to_csv(folder / "summary.csv", index=False)
    if calibrators:
        joblib.dump({"calibrators": calibrators, "scope":"U", "method":cfg["calibration"]["method"],
                     "model_sha256":model_hash}, run / "model" / "calibrators.joblib")
    write_status(run, "Калібрування", f"Побудовано калібраторів: {len(calibrators)}; лише для U")


def predict_saved(model_path, data):
    """Повернути скор спостереженої розмітки для нових рядків у початковій схемі CSV."""
    bundle = joblib.load(model_path)
    data = data.copy()
    if "is_vip" not in data:
        if "last_own_fraud_tags" not in data: raise ValueError("Потрібна ознака is_vip або last_own_fraud_tags")
        data["is_vip"] = data.last_own_fraud_tags.map(lambda x: int("vip" in parse_tags(x)))
    enc = bundle["encoding"]
    missing = set(enc["features"]) - set(data.columns)
    if missing: raise ValueError(f"Бракує ознак: {sorted(missing)}")
    x = pd.DataFrame(index=data.index)
    if enc["numeric"]:
        x[enc["numeric"]] = data[enc["numeric"]].replace([np.inf, -np.inf], np.nan).fillna(enc["medians"]).astype("float32")
    if enc["categorical"]:
        x[enc["categorical"]] = enc["encoder"].transform(data[enc["categorical"]].fillna("<пропущено>").astype(str)).astype("float32")
    scores = {}
    selected_by_tag = bundle.get("selected_features_by_tag") or {}
    for tag, model in bundle["models"].items():
        features = selected_by_tag.get(tag, bundle["selected_features"])
        scores[tag] = model.predict_proba(x[features])[:, 1]
    return pd.DataFrame(scores, index=data.index)

# %% 9. Запуск етапів і відновлення збереженого запуску

def resolve_config(config):
    """Перетворити конфігурацію за етапами у формат для наявних кроків пайплайна."""
    raw = copy.deepcopy(CONFIG if config is None else config)
    if "common" not in raw:  # Підтримати старі скрипти, які передають пласку конфігурацію.
        raw.setdefault("stage2_method", "nnpu")
        if raw.get("research_stage") == 2:
            raw["model_type"] = "catboost"
        return raw
    stage1, stage2 = raw["stage1"], raw["stage2"]
    common = raw["common"]
    cfg = copy.deepcopy(common)
    cfg["research_stage"] = raw["research_stage"]
    cfg["stage2_method"] = raw.get("stage2_method", stage2["method"])
    cfg["filter_rules"] = stage2["filter_rules"]
    cfg["audit"] = stage2["audit"]
    cfg["calibration"] = raw["calibration"]
    cfg["hidden_positive_fraction"] = {tag: {"A": None, "B": None} for tag in TAGS}
    cfg["nnpu"] = stage2["nnpu"]
    cfg["sar_pu"] = stage2["sar_pu"]
    if cfg["research_stage"] == 1:
        cfg.update({key:value for key,value in stage1.items() if key != "models"})
        cfg["lightgbm"] = stage1["models"]["lightgbm"]
        cfg["catboost"] = stage1["models"]["catboost"]
    elif cfg["research_stage"] == 2:
        cfg.update({key:value for key,value in stage2.items()
                    if key not in ("method", "filter_rules", "audit", "catboost", "nnpu", "sar_pu")})
        cfg["model_type"] = "catboost"
        cfg["catboost"] = stage2["catboost"]
        cfg["class_balance_selector"] = 0
        cfg["imbalance_method"] = {"lightgbm":"scale_pos_weight", "catboost":"Balanced"}
        cfg["feature_importance_selector"] = 0
    else:
        raise ValueError("research_stage має дорівнювати 1 або 2")

    # Сумісність зі скриптами експериментів репозиторію та перевизначеннями командного рядка.
    legacy_keys = ("source", "output_root", "audit_dir", "run_dir", "baseline_run", "seed", "max_rows",
                   "split", "model_type", "class_balance_selector", "imbalance_method",
                   "feature_importance_selector", "threshold_strategy", "fixed_threshold",
                   "lightgbm", "catboost", "optimization", "learning_curves", "experiment_id")
    for key in legacy_keys:
        if key in raw:
            cfg[key] = raw[key]
    if cfg["research_stage"] == 2:
        cfg["model_type"] = "catboost"
        # Параметри CatBoost для етапу 2 зберігаються в секції stage2;
        # застарілі параметри верхнього рівня не повинні непомітно їх замінювати.
        cfg["catboost"] = stage2["catboost"]
    return cfg


def run_stage(stage, config=None, run_dir=None):
    raw_cfg = copy.deepcopy(CONFIG if config is None else config)
    if run_dir is not None:
        if "common" in raw_cfg: raw_cfg["common"]["run_dir"] = str(run_dir)
        else: raw_cfg["run_dir"] = str(run_dir)
    cfg = resolve_config(raw_cfg)
    run = make_run(cfg)
    cfg["run_dir"] = str(run)
    identity_path = run / "data_identity.json"
    requested_path = run / "configuration_requested.json"
    original_raw = json.loads(requested_path.read_text(encoding="utf-8")) if requested_path.exists() else None
    original = resolve_config(original_raw) if original_raw else None
    if original and original.get("research_stage") != cfg["research_stage"]:
        raise ValueError("Для іншого етапу дослідження створіть новий run")
    if stage in ("all", "audit", "train", "learning_curves", "evaluate", "calibrate") and identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        filter_hash = hashlib.sha256(json.dumps(cfg["filter_rules"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if identity.get("filter_rules_sha256") != filter_hash:
            raise ValueError("Правила фільтрів змінилися; створіть новий запуск підготовки")
        if file_hash(cfg["source"]) != identity.get("source_sha256"):
            raise ValueError("Джерело змінилося; створіть новий запуск підготовки")
        if original:
            keys = ["seed", "split", "max_rows"] + (["audit", "stage2_method"] if cfg["research_stage"] == 2 else [])
            if any(original.get(key) != cfg.get(key) for key in keys):
                raise ValueError("Параметри підготовки або вибірки аудиту змінилися; створіть новий run")
    (run / "model").mkdir(exist_ok=True)
    if not requested_path.exists(): save_json(requested_path, raw_cfg)
    versions = {"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}
    import sklearn
    versions["scikit-learn"] = sklearn.__version__
    for lib in ("lightgbm", "catboost", "scipy", "matplotlib", "joblib"):
        try:
            from importlib.metadata import version
            versions[lib] = version(lib)
        except Exception: versions[lib] = "не встановлено"
    save_json(run / "versions.json", versions)
    stages = {"eda":stage_eda,"preparation":stage_preparation,"audit":stage_audit,
              "train":stage_train,"learning_curves":stage_learning_curves,
              "evaluate":stage_evaluate,"calibrate":stage_calibrate}
    if stage == "all":
        if not (run / "eda" / "summary.txt").exists(): stages["eda"](cfg, run)
        if not (run / "prepared.csv").exists(): stages["preparation"](cfg, run)
        elif not identity_path.exists() or not (run / "split.csv").exists():
            raise RuntimeError("Підготовку перервано до збереження службових файлів; створіть новий run")
        stages["audit"](cfg, run)
        if not (run / "model" / "model.joblib").exists(): stages["train"](cfg, run)
        if (cfg["research_stage"] == 1 and
                not all((run / name).exists() for name in (
                    "learning_curves_logloss.png", "learning_curves_average_precision.png"))):
            stages["learning_curves"](cfg, run)
        if not (run / "metrics_observed_per_teg.csv").exists(): stages["evaluate"](cfg, run)
        stages["calibrate"](cfg, run)
    elif stage in stages:
        if stage in ("evaluate", "calibrate") and cfg["research_stage"] == 2: stages["audit"](cfg, run)
        stages[stage](cfg, run)
    else: raise ValueError(f"Невідомий етап: {stage}")
    print(f"Папка запуску: {run}", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description="Дослідження мультиміткових PU-даних")
    parser.add_argument("--stage", choices=["all", "eda", "preparation", "audit", "train", "learning_curves", "evaluate", "calibrate"], default="all")
    parser.add_argument("--run-dir", default=None, help="Папка наявного запуску для продовження")
    parser.add_argument("--source", default=None, help="Альтернативний CSV")
    parser.add_argument("--model", choices=["lightgbm", "catboost"], default=None)
    parser.add_argument("--class-balance-selector", type=int, choices=[0, 1], default=None,
                        help="0 — без балансування; 1 — автоматичний метод для вибраної моделі")
    parser.add_argument("--imbalance-method",
                        choices=["scale_pos_weight", "is_unbalance", "Balanced", "SqrtBalanced"],
                        default=None, help="Автоматичний метод балансування для вибраної моделі")
    parser.add_argument("--research-stage", type=int, choices=[1, 2], default=None,
                        help="1 — базове навчання; 2 — навчання після аудиту")
    parser.add_argument("--stage2-method", choices=["nnpu", "sar_pu"], default=None,
                        help="Метод етапу 2: CatBoost + nnPU або окрема гілка SAR-PU")
    parser.add_argument("--threshold-strategy", choices=["fixed", "f1", "f2"], default=None)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--feature-importance-selector", type=int, choices=[0, 1], default=None,
                        help="0 — навчання на всіх ознаках; 1 — автоматичний відбір importance на validation")
    parser.add_argument("--max-rows", type=int, default=None, help="Обмеження рядків для налагодження")
    args = parser.parse_args()
    cfg = copy.deepcopy(CONFIG)
    if args.source: cfg["common"]["source"] = args.source
    if args.model: cfg["stage1"]["model_type"] = args.model
    if args.class_balance_selector is not None:
        cfg["stage1"]["class_balance_selector"] = args.class_balance_selector
    if args.imbalance_method:
        selected_model = cfg["stage1"]["model_type"]
        cfg["stage1"]["imbalance_method"][selected_model] = args.imbalance_method
    if args.research_stage is not None: cfg["research_stage"] = args.research_stage
    if args.stage2_method: cfg["stage2"]["method"] = args.stage2_method
    if args.threshold_strategy:
        cfg["stage1"]["threshold_strategy"] = args.threshold_strategy
        cfg["stage2"]["threshold_strategy"] = args.threshold_strategy
    if args.fixed_threshold is not None:
        cfg["stage1"]["fixed_threshold"] = args.fixed_threshold
        cfg["stage2"]["fixed_threshold"] = args.fixed_threshold
    if args.feature_importance_selector is not None:
        cfg["stage1"]["feature_importance_selector"] = args.feature_importance_selector
    if args.max_rows: cfg["common"]["max_rows"] = args.max_rows
    try:
        run_stage(args.stage, cfg, args.run_dir)
    except AuditNotReadyError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(2) from None

# %% 10. Командний рядок: запустити весь пайплайн
if __name__ == "__main__":
    main()
    sys.exit(0)

# %% 11. Spyder: EDA (за потреби задайте run_dir у CONFIG)
if __name__ == "__main__":
    from importlib import reload
    import pipeline
    reload(pipeline).run_stage("eda")

# %% 12. Spyder: підготовка та формування вибірок аудиту
if __name__ == "__main__":
    from importlib import reload
    import pipeline
    reload(pipeline).run_stage("preparation")

# %% 13. Spyder: аналіз наявних результатів аудиту
if __name__ == "__main__":
    from importlib import reload
    import pipeline
    reload(pipeline).run_stage("audit")

# %% 14. Spyder: навчання та оптимізація
if __name__ == "__main__":
    from importlib import reload
    import pipeline
    reload(pipeline).run_stage("train")

# %% 15. Spyder: оцінювання та шаблони аудиту після скорингу
if __name__ == "__main__":
    from importlib import reload
    import pipeline
    reload(pipeline).run_stage("evaluate")

# %% 16. Spyder: калібрування за незалежними результатами аудиту
if __name__ == "__main__":
    from importlib import reload
    import pipeline
    reload(pipeline).run_stage("calibrate")
