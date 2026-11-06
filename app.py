"""Posthepatectomy liver failure (PHLF) risk calculator — local Streamlit app.

Run:
    streamlit run app.py

Artifacts expected under outputs/ (produced by liver20261105.qmd):
    models/CatBoost.joblib, models/CatBoost.cbm, models/shap_background.csv,
    standard_scaler.pkl, pred_test.csv, cache/boot_models_*.joblib
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import app_core as core

st.set_page_config(page_title="PHLF Risk Calculator", page_icon="🩺", layout="wide")

plt.rcParams.update(
    {
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
    }
)

JAMA = ["#3B4992", "#EE0000", "#87850C", "#631879", "#008280", "#D9534F"]

DISCLAIMER = (
    "**Research prototype.** Predictions come from a retrospective cohort model and are "
    "not a substitute for clinical judgement. Any surgical decision must be made by a "
    "qualified clinician with the full clinical picture."
)


@st.cache_resource(show_spinner="Loading CatBoost…")
def get_models():
    return core.load_models(["CatBoost"])  # only the primary model is needed


@st.cache_resource(show_spinner="Loading scaler…")
def get_scaler():
    return core.load_scaler()


@st.cache_resource(show_spinner="Loading SHAP background…")
def get_background():
    return core.load_background()


@st.cache_data(show_spinner="Loading test-set predictions…")
def get_pred_test():
    return core.load_pred_test()


@st.cache_resource(show_spinner="Loading bootstrap models (about 4 s)…")
def get_boot_models(n):
    return core.load_boot_models(n)


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Patient inputs")
st.sidebar.caption(
    "Seven variables used by the model. Continuous variables are standardised internally."
)

values = {}
for key in core.FEATURES:
    spec = core.SPEC[key]
    if key == "CSPH":
        values[key] = int(
            st.sidebar.selectbox(
                "Clinically significant portal hypertension",
                [0, 1],
                format_func=lambda v: "Yes (1)" if v == 1 else "No (0)",
            )
        )
    else:
        unit = f" ({spec['unit']})" if spec["unit"] else ""
        values[key] = float(
            st.sidebar.number_input(
                f"{spec['label']}{unit}",
                min_value=float(spec["lo"]),
                max_value=float(spec["hi"]),
                value=float(spec["default"]),
                step=float(spec["step"]),
            )
        )

threshold = st.sidebar.slider(
    "Decision threshold",
    0.05,
    0.80,
    0.30,
    0.01,
    help="Predicted risk >= threshold is classified as high risk. The study uses 0.30.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: CatBoost (primary model). "
    "Surgical planning ranges are set in the third tab."
)

# ----------------------------------------------------------------- data
models = get_models()
scaler = get_scaler()
background = get_background()
pred_test = get_pred_test()

model = models["CatBoost"]

try:
    prob = core.predict_one(models, scaler, values, "CatBoost")
except Exception as exc:  # noqa: BLE001 - UI guard
    st.error(f"Prediction failed: {exc}")
    st.stop()

rank = core.cohort_rank(prob, pred_test)
high_risk = prob >= threshold

# ----------------------------------------------------------------- header
st.title("🩺 Postoperative Liver Failure Risk Calculator")
st.markdown(
    "Estimated risk of **posthepatectomy liver failure (PHLF, ISGLS grade B/C)** after "
    "liver resection for hepatocellular carcinoma, based on residual liver volume, "
    "resection ratio, bilirubin, INR, albumin, platelet count and portal hypertension. "
    "The app also provides individual SHAP explanations and surgical-range planning."
)
st.info(DISCLAIMER)

c1, c2, c3 = st.columns(3)
c1.metric("Predicted risk of liver failure", f"{prob * 100:.1f}%")
c2.metric(
    f"Decision at threshold {threshold:.2f}",
    "High risk" if high_risk else "Low risk",
    delta=f"{prob - threshold:+.3f}",
    delta_color="inverse",
)
c3.metric(
    "Percentile in test cohort",
    "—" if rank is None else f"{rank * 100:.0f}%",
    help="Share of test-set patients with a lower predicted risk.",
)

tabs = st.tabs(
    [
        "1. Prediction result",
        "2. Individual explanation (SHAP)",
        "3. Surgical planning",
    ]
)

# ================================================================= tab 1
with tabs[0]:
    try:
        left, right = st.columns([1.2, 1])
        with left:
            rows = [
                {
                    "Variable": core.SPEC[k]["label"],
                    "Value": values[k],
                    "Unit": core.SPEC[k]["unit"],
                }
                for k in core.FEATURES
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            if high_risk:
                st.error(
                    f"Predicted risk **{prob:.3f}** is above the threshold of {threshold:.2f}: "
                    "high risk of postoperative liver failure. Consider downstaging strategies "
                    "(PVE/ALPPS), a smaller resection, or intensified perioperative care."
                )
            else:
                st.success(
                    f"Predicted risk **{prob:.3f}** is below the threshold of {threshold:.2f}: "
                    "low risk under the current plan. Re-assess with intraoperative findings."
                )

            st.download_button(
                "Download this result (CSV)",
                pd.DataFrame(
                    [
                        {
                            **values,
                            "model": "CatBoost",
                            "prob_PHLF": prob,
                            "threshold": threshold,
                            "high_risk": int(high_risk),
                        }
                    ]
                )
                .to_csv(index=False)
                .encode("utf-8-sig"),
                file_name="phlf_risk.csv",
                mime="text/csv",
            )

        with right:
            st.subheader("What the number means")
            if pred_test is not None and not pred_test.empty:
                m = core.threshold_metrics(
                    pred_test["y_true"], pred_test["prob_ISGLS"], threshold
                )
                st.markdown(
                    f"The model estimates a **{prob * 100:.1f}%** probability that this patient "
                    f"develops ISGLS grade B/C liver failure. At a decision threshold of "
                    f"**{threshold:.2f}**, the case is classified as "
                    f"**{'high' if high_risk else 'low'} risk**. "
                    f"At this threshold the model's sensitivity is **{m['sensitivity']:.2f}** "
                    f"and specificity is **{m['specificity']:.2f}** in the test cohort."
                )
            else:
                st.markdown(
                    f"The model estimates a **{prob * 100:.1f}%** probability of ISGLS grade B/C "
                    f"liver failure; at a threshold of **{threshold:.2f}** this is "
                    f"**{'high' if high_risk else 'low'} risk**."
                )
            st.caption(
                "Risk is a probability, not a certainty. The threshold reflects the "
                "relative harm of a missed event versus unnecessary intervention."
            )
    except Exception as exc:  # noqa: BLE001 - UI guard
        st.error(f"Prediction section error: {exc}")

# ================================================================= tab 2
with tabs[1]:
    try:
        if background is None:
            st.warning("models/shap_background.csv not found; SHAP is unavailable.")
        else:
            import shap

            sv, base, x_clinical = core.shap_explain(model, background, values, scaler)
            labels = [core.SPEC[f]["label"] for f in core.FEATURES]

            # data 用临床尺度：瀑布图右侧显示的是 TBil 18 而不是标准化后的 0.45
            exp = shap.Explanation(
                values=sv, base_values=base, data=x_clinical, feature_names=labels
            )
            fig = plt.figure(figsize=(9.2, 4.6))
            shap.plots.waterfall(exp, max_display=len(labels), show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            contrib = pd.DataFrame(
                {
                    "Variable": labels,
                    "Value": x_clinical,
                    "Unit": [core.SPEC[f]["unit"] for f in core.FEATURES],
                    "SHAP contribution (probability scale)": np.round(sv, 4),
                }
            ).sort_values(
                "SHAP contribution (probability scale)", key=np.abs, ascending=False
            )
            st.dataframe(contrib, hide_index=True, use_container_width=True)

            top = contrib.head(3)
            up = top[top["SHAP contribution (probability scale)"] > 0][
                "Variable"
            ].tolist()
            down = top[top["SHAP contribution (probability scale)"] < 0][
                "Variable"
            ].tolist()
            text = (
                f"Baseline risk (background mean) is **{base:.3f}**; this patient is "
                f"predicted at **{prob:.3f}**."
            )
            if up:
                text += f" Main factors **increasing** risk: **{', '.join(up)}**."
            if down:
                text += f" Main factors **decreasing** risk: **{', '.join(down)}**."
            st.markdown(text)
            st.caption(
                "Feature values are shown on the clinical scale (the model itself uses "
                "standardised inputs). SHAP is additive on the probability scale: "
                "base value + sum of SHAP values ≈ predicted risk."
            )
    except Exception as exc:  # noqa: BLE001 - UI guard
        st.error(f"SHAP section error: {exc}")

# ================================================================= tab 3
with tabs[2]:
    try:
        st.caption(
            "Fixing the other clinical variables, the feasible residual/resection region "
            "is scanned on a grid. Uncertainty comes from 1,000 CatBoost models refitted "
            "on bootstrap resamples of the training set."
        )
        if not core.has_boot_models():
            st.warning(
                "cache/boot_models_*.joblib not found: planning will show point estimates only."
            )

        with st.form("planning_form"):
            c1, c2, c3, c4 = st.columns(4)
            res_lo = c1.number_input("Residual ratio – min", 0.05, 1.20, 0.35, 0.01)
            res_hi = c2.number_input("Residual ratio – max", 0.05, 1.20, 0.90, 0.01)
            rm_lo = c3.number_input("Resection ratio – min", 0.00, 1.00, 0.10, 0.01)
            rm_hi = c4.number_input("Resection ratio – max", 0.00, 1.00, 0.60, 0.01)
            c5, c6, c7 = st.columns(3)
            grid = c5.slider(
                "Grid resolution",
                81,
                151,
                121,
                10,
                help="Number of points per axis (N x N grid). The manuscript uses N = 121.",
            )
            use_boot = c6.checkbox("Use bootstrap uncertainty", True)
            fast = c7.checkbox("Fast mode (200 models)", False)
            st.caption(
                f"Cell size ≈ {(res_hi - res_lo) / (grid - 1) * 100:.2f} percentage points "
                f"(residual) × {(rm_hi - rm_lo) / (grid - 1) * 100:.2f} percentage points "
                f"(resection). N = 121 matches the manuscript (≈0.46 × 0.42 pp); "
                "finer grids add no information because the bootstrap SD of the predicted "
                "risk is about 3–4 pp."
            )
            submitted = st.form_submit_button("Run planning", type="primary")

        if "planning" not in st.session_state:
            st.session_state.planning = None

        if submitted:
            if res_hi <= res_lo or rm_hi <= rm_lo:
                st.error("Upper bounds must exceed lower bounds.")
            else:
                boots = get_boot_models(200 if fast else None) if use_boot else []
                with st.spinner(f"Computing with {len(boots)} bootstrap models…"):
                    st.session_state.planning = core.planning_grid(
                        {**values},
                        model,
                        scaler,
                        boots,
                        res_range=(res_lo, res_hi),
                        rm_range=(rm_lo, rm_hi),
                        grid=int(grid),
                        threshold=threshold,
                        conf=0.90,
                    )

        plan = st.session_state.planning
        if plan is None:
            st.info("Set the ranges and press **Run planning**.")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
            res_axis, rm_axis = plan["res_axis"], plan["rm_axis"]

            ax = axes[0]
            vmax = float(np.nanmax(plan["point_m"]))
            mesh = ax.pcolormesh(
                res_axis,
                rm_axis,
                plan["point_m"],
                shading="auto",
                cmap="RdYlGn_r",
                vmin=0,
                vmax=vmax,
            )
            if np.nanmin(plan["point_m"]) < threshold < np.nanmax(plan["point_m"]):
                cs = ax.contour(
                    res_axis,
                    rm_axis,
                    plan["point_m"],
                    levels=[threshold],
                    colors="black",
                    linewidths=1.6,
                )
                ax.clabel(cs, fmt={threshold: f"risk={threshold:.2f}"}, fontsize=9)
            fig.colorbar(mesh, ax=ax, label="Predicted risk")
            ax.set_xlabel("Residual liver ratio")
            ax.set_ylabel("Resection ratio")
            ax.set_title("(a) Decision surface and iso-risk contour")

            ax = axes[1]
            mesh = ax.pcolormesh(
                res_axis,
                rm_axis,
                plan["psafe_m"],
                shading="auto",
                cmap="Blues",
                vmin=0,
                vmax=1,
            )
            if np.nanmin(plan["psafe_m"]) < 0.90 < np.nanmax(plan["psafe_m"]):
                cs = ax.contour(
                    res_axis,
                    rm_axis,
                    plan["psafe_m"],
                    levels=[0.90],
                    colors="black",
                    linewidths=1.6,
                )
                ax.clabel(cs, fmt={0.90: "P=0.90"}, fontsize=9)
            fig.colorbar(
                mesh, ax=ax, label="Bootstrap proportion with risk ≤ threshold"
            )
            ax.set_xlabel("Residual liver ratio")
            ax.set_ylabel("Resection ratio")
            ax.set_title("(b) Probability of being safe")

            ax = axes[2]
            bp, bc = plan["best_point"], plan["best_conf"]
            ax.plot(
                bp["Remove"],
                bp["Point_risk"],
                color=JAMA[2],
                lw=2,
                ls="--",
                label="Point estimate (lower envelope)",
            )
            ax.plot(
                bc["Remove"],
                bc["Risk_conf"],
                color=JAMA[3],
                lw=2,
                label="90% quantile (lower envelope)",
            )
            ax.axhline(
                threshold,
                color=JAMA[1],
                ls="--",
                lw=1.2,
                label=f"Threshold {threshold:.2f}",
            )
            for r, color in [(plan["r_point"], JAMA[2]), (plan["r_conf"], JAMA[3])]:
                if np.isfinite(r):
                    ax.axvline(r, color=color, lw=1.1, ls=":")
            if np.isfinite(plan["r_point"]) and np.isfinite(plan["r_conf"]):
                ax.annotate(
                    "",
                    xy=(plan["r_point"], threshold * 1.25),
                    xytext=(plan["r_conf"], threshold * 1.25),
                    arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.1},
                )
            ax.set_xlabel("Resection ratio")
            ax.set_ylabel("Predicted risk")
            ax.set_title("(c) Maximum feasible resection under two criteria")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            rec = plan["rec"]
            st.subheader("Recommended plan")
            if plan["status"] == "safe":
                st.success(
                    f"Residual **{rec['Residual'] * 100:.1f}%**, resection "
                    f"**{rec['Remove'] * 100:.1f}%**; point-estimate risk "
                    f"{rec['Point_risk']:.3f}, bootstrap proportion safe "
                    f"{rec['P_safe']:.2f}, 90% quantile {rec['Risk_conf']:.3f}."
                )
            else:
                st.warning(
                    "No plan reaches the 90% bootstrap criterion in this range. "
                    f"Closest plan: residual {rec['Residual'] * 100:.1f}%, resection "
                    f"{rec['Remove'] * 100:.1f}% (bootstrap proportion safe "
                    f"{rec['P_safe']:.2f}, 90% quantile {rec['Risk_conf']:.3f})."
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("Grid candidates", f"{plan['n_candidates']:,}")
            m2.metric(
                "Max resection (point estimate)",
                "—"
                if not np.isfinite(plan["r_point"])
                else f"{plan['r_point'] * 100:.1f}%",
            )
            m3.metric(
                "Max resection (90% confidence)",
                "—"
                if not np.isfinite(plan["r_conf"])
                else f"{plan['r_conf'] * 100:.1f}%",
            )

            if np.isfinite(plan["r_point"]) and np.isfinite(plan["r_conf"]):
                st.caption(
                    f"Reaching 90% confidence costs "
                    f"**{(plan['r_point'] - plan['r_conf']) * 100:.1f} percentage points** of "
                    "resection compared with the point estimate, i.e. the price of model uncertainty."
                )
            st.caption(
                f"Bootstrap models used: {plan['n_boot']}. Point estimates come from CatBoost."
            )
    except Exception as exc:  # noqa: BLE001 - UI guard
        st.error(f"Planning section error: {exc}")

st.divider()

with st.expander("Model information"):
    try:
        manifest = pd.read_csv(core.MODEL_DIR / "manifest.csv")
        row = manifest.loc[manifest["model"] == "CatBoost"].iloc[0]
        st.write(
            {
                "Model": row["class"],
                "Features": row["features"].split("|"),
                "Decision threshold": float(row["threshold"]),
                "Random seed": int(row["seed"]),
                "scikit-learn": row["sklearn"],
                "CatBoost": row["catboost"],
            }
        )
        fp = core.MODEL_DIR / "training_fingerprint.txt"
        if fp.exists():
            st.code(fp.read_text(encoding="utf-8"), language="text")
    except Exception as exc:  # noqa: BLE001 - UI guard
        st.warning(f"Model information unavailable: {exc}")

st.caption(
    "Development cohort: 1,349 patients (two centres); external validation: 199 patients. "
    "Model and dependency versions are recorded in outputs/models/manifest.csv."
)
