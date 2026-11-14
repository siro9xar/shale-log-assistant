"""Lithofacies classifier: random forest, WELL-BASED train/test split.

Why well-based: neighbouring depth samples of one well are almost copies of each
other. A random row split puts near-duplicates in both train and test and the
score looks far better than it will be on a new well. Here whole wells are held
out, so the test score is an estimate for a well the model has never seen.

Training table (CSV), standard units:
    WELL, DEPTH, GR [API], RHOB [g/cc], NPHI [v/v], RT [ohm.m], DT [us/ft], FACIES [text label]
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


# --------------------------------------------------------------------------- features
def build_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Feature table. 'X_LOG10' means log10 of curve X (used for resistivity)."""
    out = {}
    for name in features:
        if name.endswith("_LOG10"):
            base = name[: -len("_LOG10")]
            if base not in df.columns:
                raise KeyError(f"feature {name} needs curve {base}, which is not available")
            x = df[base].to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[name] = np.where(x > 0, np.log10(x), np.nan)
        else:
            if name not in df.columns:
                raise KeyError(f"feature curve {name} is not available")
            out[name] = df[name].to_numpy(dtype=float)
    return pd.DataFrame(out, index=df.index)


def _make_model(fc: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=fc["n_estimators"], max_depth=fc["max_depth"],
        min_samples_leaf=fc["min_samples_leaf"], class_weight=fc["class_weight"],
        random_state=fc["random_state"], n_jobs=-1)


def _scores(y_true, y_pred) -> dict:
    with warnings.catch_warnings():          # a class can be absent from a held-out well; that is expected
        warnings.simplefilter("ignore", category=UserWarning)
        return _scores_inner(y_true, y_pred)


def _scores_inner(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 3),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 3),
    }


# --------------------------------------------------------------------------- training
def train_classifier(table: pd.DataFrame, cfg: dict, out_dir: str | Path,
                     compare_random_split: bool = False) -> dict:
    """Train, evaluate on held-out WELLS, save model + metrics + confusion matrix."""
    fc = cfg["facies"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in ("WELL", "FACIES"):
        if col not in table.columns:
            raise KeyError(f"training table needs a '{col}' column")
    X_all = build_features(table, fc["features"])
    keep = X_all.notna().all(axis=1) & table["FACIES"].notna()
    n_dropped_nan = int((~keep).sum())
    X_all, y_all = X_all[keep], table.loc[keep, "FACIES"].astype(str)
    wells_all = table.loc[keep, "WELL"].astype(str)

    counts = y_all.value_counts()
    rare = sorted(counts[counts < fc["min_class_samples"]].index.tolist())
    if rare:
        keep2 = ~y_all.isin(rare)
        X_all, y_all, wells_all = X_all[keep2], y_all[keep2], wells_all[keep2]

    unique_wells = sorted(wells_all.unique())
    if len(unique_wells) < 3:
        raise ValueError("need at least 3 labelled wells for a well-based split")

    # ---- well-based split
    if fc["test_wells"]:
        test_wells = sorted(set(map(str, fc["test_wells"])) & set(unique_wells))
        if not test_wells:
            raise ValueError("none of facies.test_wells is in the training table")
        split_method = "wells listed in config facies.test_wells"
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=fc["test_fraction"], random_state=fc["random_state"])
        _, test_idx = next(gss.split(X_all, y_all, groups=wells_all))
        test_wells = sorted(wells_all.iloc[test_idx].unique())
        split_method = (f"GroupShuffleSplit on WELL (test_fraction={fc['test_fraction']} of wells, "
                        f"random_state={fc['random_state']})")
    is_test = wells_all.isin(test_wells).to_numpy()
    train_wells = sorted(set(unique_wells) - set(test_wells))
    assert not set(train_wells) & set(test_wells), "a well is in both train and test"

    X_tr, y_tr, g_tr = X_all[~is_test], y_all[~is_test], wells_all[~is_test]
    X_te, y_te = X_all[is_test], y_all[is_test]

    # ---- well-grouped cross-validation on the training wells
    cv = None
    n_splits = min(int(fc["cv_folds"]), len(train_wells))
    if n_splits >= 2:
        folds = []
        for tr, va in GroupKFold(n_splits=n_splits).split(X_tr, y_tr, groups=g_tr):
            m = _make_model(fc).fit(X_tr.iloc[tr], y_tr.iloc[tr])
            folds.append(_scores(y_tr.iloc[va], m.predict(X_tr.iloc[va])))
        cv = {"n_splits": n_splits, "grouping": "WELL (GroupKFold)"}
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            vals = [f[key] for f in folds]
            cv[f"{key}_mean"] = round(float(np.mean(vals)), 3)
            cv[f"{key}_std"] = round(float(np.std(vals)), 3)

    # ---- final model = trained on the training wells only, so the test metrics describe it exactly
    model = _make_model(fc).fit(X_tr, y_tr)
    classes = [str(c) for c in model.classes_]
    proba = model.predict_proba(X_te)
    y_pred = np.asarray(classes)[proba.argmax(axis=1)]
    conf = proba.max(axis=1)

    labels = sorted(set(classes) | set(y_te.unique()))
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(y_te, y_pred, labels=labels, zero_division=0)
    correct = (y_pred == y_te.to_numpy())
    thr = fc["low_confidence_threshold"]
    confident = conf >= thr

    def _acc(mask):
        return round(float(correct[mask].mean()), 3) if mask.any() else None

    metrics = {
        "model": "RandomForestClassifier",
        "hyperparameters": {k: fc[k] for k in ("n_estimators", "max_depth", "min_samples_leaf",
                                               "class_weight", "random_state")},
        "features": fc["features"],
        "classes": classes,
        "split": {"method": split_method, "n_train_wells": len(train_wells), "n_test_wells": len(test_wells),
                  "train_wells": train_wells, "test_wells": test_wells,
                  "n_train_samples": int(len(X_tr)), "n_test_samples": int(len(X_te))},
        "rows_dropped_missing_features_or_label": n_dropped_nan,
        "classes_dropped_as_too_rare": rare,
        "test": {**_scores(y_te, y_pred),
                 "per_class": {lab: {"precision": round(float(p), 3), "recall": round(float(r), 3),
                                     "f1": round(float(f), 3), "support": int(s)}
                               for lab, p, r, f, s in zip(labels, prec, rec, f1, sup)}},
        "cross_validation": cv,
        "confidence": {
            "definition": "max class probability of the forest (share of tree votes); not a calibrated probability",
            "threshold": thr,
            "test_accuracy_when_confidence_at_or_above_threshold": _acc(confident),
            "test_accuracy_when_confidence_below_threshold": _acc(~confident),
            "share_of_test_samples_below_threshold": round(float((~confident).mean()), 3),
        },
        "confusion_matrix": {"rows": "true", "columns": "predicted", "labels": labels, "matrix": cm.tolist()},
        "feature_importance": {f: round(float(i), 3) for f, i in zip(fc["features"], model.feature_importances_)},
    }

    if compare_random_split:      # shows how much a random row split flatters the model (leakage)
        Xa, Xb, ya, yb = train_test_split(X_all, y_all, test_size=fc["test_fraction"],
                                          random_state=fc["random_state"], stratify=y_all)
        leak = _make_model(fc).fit(Xa, ya)
        metrics["random_row_split_for_comparison_only"] = {
            **_scores(yb, leak.predict(Xb)),
            "note": "random split of depth samples: neighbouring samples leak between train and test; do NOT quote this score"}

    joblib.dump({"model": model, "features": fc["features"], "classes": classes}, out_dir / "facies_rf.joblib", compress=3)
    (out_dir / "facies_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=[f"true: {l}" for l in labels], columns=[f"pred: {l}" for l in labels]).to_csv(
        out_dir / "confusion_matrix.csv")
    _plot_confusion(cm, labels, metrics, out_dir / "confusion_matrix.png")
    return metrics


def _plot_confusion(cm: np.ndarray, labels: list[str], metrics: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros(cm.shape, dtype=float), where=row_sum > 0)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.1 * n + 3, 1.0 * n + 2.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted facies")
    ax.set_ylabel("True facies")
    t = metrics["test"]
    ax.set_title(f"Confusion matrix on {metrics['split']['n_test_wells']} held-out wells (row-normalised)\n"
                 f"accuracy {t['accuracy']}, balanced accuracy {t['balanced_accuracy']}, macro F1 {t['macro_f1']}",
                 fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- prediction
def load_model(path: str | Path) -> dict:
    return joblib.load(path)


def predict_facies(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return FACIES (label or None) and FACIES_CONF (max class probability) per depth sample."""
    X = build_features(df, bundle["features"])
    ok = X.notna().all(axis=1).to_numpy()
    facies = np.full(len(df), None, dtype=object)
    conf = np.full(len(df), np.nan)
    if ok.any():
        proba = bundle["model"].predict_proba(X[ok])
        facies[ok] = np.asarray(bundle["classes"], dtype=object)[proba.argmax(axis=1)]
        conf[ok] = proba.max(axis=1)
    return pd.DataFrame({"FACIES": facies, "FACIES_CONF": conf}, index=df.index)
