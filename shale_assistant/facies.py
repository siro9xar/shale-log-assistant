"""Lithofacies classifier: random forest, WELL-BASED train/test split.

Why well-based: neighbouring depth samples of one well are almost copies of each
other. A random row split puts near-duplicates in both train and test and the
score looks far better than it will be on a new well. Here whole wells are held
out, so the test score is an estimate for a well the model has never seen.

Training table (CSV), standard units:
    WELL, DEPTH, GR [API], RHOB [g/cc], NPHI [v/v], RT [ohm.m], DT [us/ft], FACIES [text label]

Evaluation
    5 or more labelled wells : hold out facies.test_fraction of the WELLS (or facies.test_wells),
                               plus well-grouped cross-validation on the training wells.
    2 to 4 labelled wells    : leave-one-well-out. Each well is held out once and predicted by a
                               model trained on the other wells; the scores pool those predictions.
                               The saved model is then trained on all wells.
    Rows are never split at random: that would leak neighbouring samples between train and test.
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


MIN_WELLS_FOR_HOLDOUT = 5            # fewer labelled wells than this -> leave-one-well-out
REQUIRED_COLUMNS = ["WELL", "DEPTH", "GR", "RHOB", "NPHI", "RT", "DT", "FACIES"]


# --------------------------------------------------------------------------- training table
def load_training_table(path: str | Path) -> pd.DataFrame:
    """Read and check a labelled table. Column names are case-insensitive; ',' or ';' separated.

    Units must be the standard ones (the same the pipeline uses when it predicts):
    GR API, RHOB g/cc, NPHI v/v, RT ohm.m, DT us/ft. Obvious unit mistakes are refused, not converted.
    """
    path = Path(path)
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = [str(c).strip().upper() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}. Needed: {', '.join(REQUIRED_COLUMNS)} "
                         f"(found: {', '.join(df.columns)})")
    for c in ("DEPTH", "GR", "RHOB", "NPHI", "RT", "DT"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["WELL"].notna()].copy()
    df["WELL"] = df["WELL"].astype(str).str.strip()
    facies = df["FACIES"].astype("string").str.strip()
    df["FACIES"] = facies.mask(facies == "")

    problems = []
    med = {c: float(df[c].median()) for c in ("RHOB", "NPHI", "DT") if df[c].notna().any()}
    if med.get("RHOB", 2.5) > 100:
        problems.append(f"RHOB median {med['RHOB']:.0f} looks like kg/m3; the table must be in g/cc (divide by 1000)")
    elif not 1.0 <= med.get("RHOB", 2.5) <= 3.5:
        problems.append(f"RHOB median {med['RHOB']:.2f} is not a bulk density in g/cc")
    if med.get("NPHI", 0.2) > 1.0:
        problems.append(f"NPHI median {med['NPHI']:.1f} looks like percent; the table must be in v/v (divide by 100)")
    if med.get("DT", 80.0) > 200:
        problems.append(f"DT median {med['DT']:.0f} looks like us/m; the table must be in us/ft (multiply by 0.3048)")
    if problems:
        raise ValueError(f"{path.name}: wrong units. " + "; ".join(problems))
    return df


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
    if y_all.nunique() < 2:
        raise ValueError(f"fewer than 2 facies classes left after dropping rows with missing curves and classes "
                         f"with fewer than facies.min_class_samples = {fc['min_class_samples']} samples "
                         f"(dropped: {rare}). Lower facies.min_class_samples or add data.")
    if len(unique_wells) < 2:
        raise ValueError("need at least 2 labelled wells: with one well the only possible test is a random row "
                         "split, which leaks neighbouring samples and overstates accuracy")
    notes = []
    wells_per_class = pd.DataFrame({"w": wells_all.to_numpy(), "y": y_all.to_numpy()}).groupby("y")["w"].nunique()
    one_well = sorted(wells_per_class[wells_per_class < 2].index.tolist())
    if one_well:
        notes.append(f"class(es) {one_well} occur in only one well: when that well is held out no model has "
                     "seen them, so they cannot be predicted there")

    if len(unique_wells) < MIN_WELLS_FOR_HOLDOUT:
        if fc["test_wells"]:
            notes.append("facies.test_wells is ignored: fewer than "
                         f"{MIN_WELLS_FOR_HOLDOUT} wells, leave-one-well-out is used instead")
        return _finish(*_leave_one_well_out(X_all, y_all, wells_all, unique_wells, fc), X_all, y_all, fc,
                       n_dropped_nan, rare, notes, compare_random_split, out_dir)

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
    if not train_wells:
        raise ValueError("facies.test_wells holds every well in the table: nothing is left to train on")

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
    split = {"method": split_method, "evaluation": "holdout",
             "n_train_wells": len(train_wells), "n_test_wells": len(test_wells),
             "train_wells": train_wells, "test_wells": test_wells,
             "n_train_samples": int(len(X_tr)), "n_test_samples": int(len(X_te))}
    return _finish(model, y_te, y_pred, conf, split, cv, X_all, y_all, fc, n_dropped_nan, rare, notes,
                   compare_random_split, out_dir)


def _leave_one_well_out(X_all, y_all, wells_all, unique_wells, fc):
    """Each well held out once. Returns (final model on all wells, y_true, y_pred, confidence, split, cv=None)."""
    w = wells_all.to_numpy()
    y_pred = np.empty(len(y_all), dtype=object)
    conf = np.full(len(y_all), np.nan)
    per_well = []
    for well in unique_wells:
        held = w == well
        m = _make_model(fc).fit(X_all[~held], y_all[~held])
        proba = m.predict_proba(X_all[held])
        y_pred[held] = np.asarray([str(c) for c in m.classes_], dtype=object)[proba.argmax(axis=1)]
        conf[held] = proba.max(axis=1)
        per_well.append({"well": well, "n_samples": int(held.sum()), **_scores(y_all[held], y_pred[held])})
    model = _make_model(fc).fit(X_all, y_all)
    n = len(unique_wells)
    split = {"method": f"leave-one-well-out (fallback: {n} labelled wells, fewer than {MIN_WELLS_FOR_HOLDOUT})",
             "evaluation": "leave_one_well_out",
             "n_train_wells": n, "n_test_wells": n, "train_wells": unique_wells, "test_wells": unique_wells,
             "n_train_samples": int(len(X_all)), "n_test_samples": int(len(X_all)),
             "per_well": per_well,
             "note": (f"each of the {n} wells was held out once and predicted by a model trained on the other "
                      f"{n - 1}; the test scores pool those predictions. The saved model is trained on all {n} "
                      "wells, so the scores describe models trained on one well fewer.")}
    return model, y_all, y_pred.astype(str), conf, split, None


def _finish(model, y_te, y_pred, conf, split, cv, X_all, y_all, fc, n_dropped_nan, rare, notes,
            compare_random_split, out_dir) -> dict:
    """Metrics, files and plots; the same for the held-out-wells and leave-one-well-out evaluations."""
    classes = [str(c) for c in model.classes_]
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
        "split": split,
        "notes": notes,
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
    how = ("leave-one-well-out over" if metrics["split"].get("evaluation") == "leave_one_well_out" else "on")
    ax.set_title(f"Confusion matrix {how} {metrics['split']['n_test_wells']} held-out wells (row-normalised)\n"
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
