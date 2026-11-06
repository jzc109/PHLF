"""Core computation for the PHLF risk calculator (framework-agnostic)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"

FEATURES = ["Residual_to_std", "Remove_to_std", "TBil", "INR", "ALB", "PLT", "CSPH"]
CONT = FEATURES[:-1]
CAT = ["CSPH"]

SPEC = {
    "Residual_to_std": {
        "label": "Residual liver ratio",
        "unit": "fraction of standard liver volume",
        "lo": 0.10,
        "hi": 1.20,
        "step": 0.01,
        "default": 0.90,
    },
    "Remove_to_std": {
        "label": "Resection ratio",
        "unit": "fraction of standard liver volume",
        "lo": 0.00,
        "hi": 1.00,
        "step": 0.01,
        "default": 0.30,
    },
    "TBil": {
        "label": "Total bilirubin",
        "unit": "μmol/L",
        "lo": 2.0,
        "hi": 600.0,
        "step": 1.0,
        "default": 18.0,
    },
    "INR": {
        "label": "INR",
        "unit": "",
        "lo": 0.70,
        "hi": 3.00,
        "step": 0.01,
        "default": 1.05,
    },
    "ALB": {
        "label": "Albumin",
        "unit": "g/L",
        "lo": 10.0,
        "hi": 60.0,
        "step": 0.5,
        "default": 45.0,
    },
    "PLT": {
        "label": "Platelet count",
        "unit": "×10⁹/L",
        "lo": 10.0,
        "hi": 600.0,
        "step": 1.0,
        "default": 140.0,
    },
    "CSPH": {
        "label": "Clinically significant portal hypertension",
        "unit": "",
        "lo": 0,
        "hi": 1,
        "step": 1,
        "default": 0,
    },
}


def load_models(names=None) -> dict:
    """Load models by name.

    CatBoost is loaded from its native ``.cbm`` file when available: that format
    is portable across Python / scikit-learn versions, which matters for
    deployment (Streamlit Community Cloud may not match the training stack).
    """
    if names is None:
        names = sorted(p.stem for p in MODEL_DIR.glob("*.joblib"))
    if not names:
        raise FileNotFoundError("No models found under outputs/models/.")

    models = {}
    for name in names:
        cbm = MODEL_DIR / f"{name}.cbm"
        joblib_path = MODEL_DIR / f"{name}.joblib"
        if name == "CatBoost" and cbm.exists():
            from catboost import CatBoostClassifier

            models[name] = CatBoostClassifier().load_model(str(cbm))
        elif joblib_path.exists():
            models[name] = joblib.load(joblib_path)
        else:
            raise FileNotFoundError(f"Model file for {name!r} not found in {MODEL_DIR}.")
    return models


def load_scaler():
    path = OUT_DIR / "standard_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError("outputs/standard_scaler.pkl not found.")
    return joblib.load(path)


def load_background():
    path = MODEL_DIR / "shap_background.csv"
    return pd.read_csv(path) if path.exists() else None


def load_pred_test():
    path = OUT_DIR / "pred_test.csv"
    return pd.read_csv(path) if path.exists() else None


def has_boot_models() -> bool:
    return any((OUT_DIR / "cache").glob("boot_models_*.joblib"))


def load_boot_models(max_models: int | None = None):
    files = sorted((OUT_DIR / "cache").glob("boot_models_*.joblib"))
    if not files:
        return []
    models = joblib.load(files[-1])
    return models[:max_models] if max_models else models


def make_design(values: dict, scaler) -> pd.DataFrame:
    row = pd.DataFrame([values])[FEATURES]
    row[CONT] = scaler.transform(row[CONT])
    return row[FEATURES]


def predict_one(
    models: dict, scaler, values: dict, model_name: str = "CatBoost"
) -> float:
    X = make_design(values, scaler)
    return float(models[model_name].predict_proba(X)[0, 1])


def cohort_rank(prob: float, pred_test: pd.DataFrame | None) -> float | None:
    if pred_test is None or pred_test.empty:
        return None
    return float((pred_test["prob_ISGLS"] < prob).mean())


def shap_explain(model, background: pd.DataFrame, values: dict, scaler):
    """Return (SHAP values, base value, raw clinical row).

    SHAP is computed on the standardised design matrix (what the model sees),
    but the last element is the **raw clinical row** so that plots can label
    features with values a clinician can read (e.g. TBil 18, not z = 0.45).
    """
    import shap

    X = make_design(values, scaler)
    explainer = shap.TreeExplainer(model, data=background, model_output="probability")
    sv = np.asarray(explainer.shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    base = float(np.asarray(explainer.expected_value).ravel()[-1])
    raw_row = pd.DataFrame([values])[FEATURES].iloc[0].to_numpy(dtype=float)
    return sv[0], base, raw_row


def planning_grid(
    patient: dict,
    model,
    scaler,
    boot_models,
    res_range=(0.35, 0.90),
    rm_range=(0.10, 0.60),
    grid: int = 81,
    threshold: float = 0.30,
    conf: float = 0.90,
    sum_constraint: float = 1.2,
) -> dict:
    """Grid enumeration + point prediction + bootstrap uncertainty + Pareto front."""
    res_axis = np.linspace(res_range[0], res_range[1], grid)
    rm_axis = np.linspace(rm_range[0], rm_range[1], grid)
    R, M = np.meshgrid(res_axis, rm_axis)
    mask = (R + M) <= sum_constraint

    pts = pd.DataFrame(
        {
            "Residual_to_std": R[mask],
            "Remove_to_std": M[mask],
            "TBil": patient["TBil"],
            "INR": patient["INR"],
            "ALB": patient["ALB"],
            "PLT": patient["PLT"],
            "CSPH": patient["CSPH"],
        }
    )[FEATURES]
    pts_s = pts.copy()
    pts_s[CONT] = scaler.transform(pts[CONT])

    point = model.predict_proba(pts_s)[:, 1]

    if boot_models:
        boot = np.vstack([m.predict_proba(pts_s)[:, 1] for m in boot_models])
        p_safe = (boot <= threshold).mean(axis=0)
        q_conf = np.percentile(boot, 100 * conf, axis=0)
        q_lo = np.percentile(boot, 2.5, axis=0)
        q_hi = np.percentile(boot, 97.5, axis=0)
        n_boot = len(boot_models)
    else:
        p_safe = (point <= threshold).astype(float)
        q_conf = point.copy()
        q_lo = q_hi = point.copy()
        n_boot = 0

    def to_mat(v):
        out = np.full(R.shape, np.nan)
        out[mask] = v
        return out

    df = pd.DataFrame(
        {
            "Residual": R[mask],
            "Remove": M[mask],
            "Point_risk": point,
            "P_safe": p_safe,
            "Risk_conf": q_conf,
            "Risk_lo": q_lo,
            "Risk_hi": q_hi,
        }
    )

    def max_safe_res(frame, col):
        ok = frame.loc[frame[col] <= threshold, "Remove"]
        return float(ok.max()) if len(ok) else float("nan")

    best_point = df.groupby("Remove")["Point_risk"].min().reset_index()
    best_conf = df.groupby("Remove")["Risk_conf"].min().reset_index()

    ordered = df.sort_values(["Remove", "Risk_conf"], ascending=[False, True])
    keep_idx, best = [], np.inf
    for idx, v in ordered["Risk_conf"].items():
        if v < best:
            keep_idx.append(idx)
            best = v
    front = df.loc[keep_idx].sort_values("Remove").reset_index(drop=True)
    safe = (
        front.loc[front["Risk_conf"] <= threshold]
        .sort_values(["Remove", "Risk_conf"], ascending=[False, True])
        .reset_index(drop=True)
    )

    if len(safe):
        rec = safe.iloc[0].to_dict()
        status = "safe"
    else:
        j = int(np.argmin(df["Risk_conf"].to_numpy()))
        rec = df.iloc[j].to_dict()
        status = "infeasible"

    return {
        "res_axis": res_axis,
        "rm_axis": rm_axis,
        "point_m": to_mat(point),
        "psafe_m": to_mat(p_safe),
        "qconf_m": to_mat(q_conf),
        "df": df,
        "front": front,
        "safe": safe,
        "best_point": best_point,
        "best_conf": best_conf,
        "r_point": max_safe_res(best_point, "Point_risk"),
        "r_conf": max_safe_res(best_conf, "Risk_conf"),
        "rec": rec,
        "status": status,
        "n_boot": n_boot,
        "n_candidates": int(mask.sum()),
    }


def threshold_metrics(y, prob, threshold: float) -> dict:
    from sklearn.metrics import confusion_matrix, roc_auc_score

    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, prob)),
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def best_f1_threshold(y, prob):
    from sklearn.metrics import precision_recall_curve

    prec, rec, thrs = precision_recall_curve(y, prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thrs[int(np.argmax(f1[:-1]))])
