# PHLF Risk Calculator (Streamlit)

Web calculator for the risk of **posthepatectomy liver failure (PHLF, ISGLS grade B/C)**
after liver resection for hepatocellular carcinoma, based on the CatBoost model in
`liver20261105.qmd`.

Live sections:

1. **Prediction result** — risk, decision at the threshold, cohort percentile.
2. **Individual explanation (SHAP)** — waterfall plot with clinical-scale feature values.
3. **Surgical planning** — grid enumeration, bootstrap safety proportion, two frontiers,
   recommended plan.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new **public** (or private) repository on GitHub, e.g. `phlf-calculator`.
2. Upload the contents of this folder (keep the `outputs/` directory).
3. Go to <https://share.streamlit.io> → **New app** → choose the repository, branch `main`,
   main file `app.py`.
4. In **Advanced settings**, set Python to **3.12** (matches `runtime.txt`).
5. Deploy. First build takes a few minutes (CatBoost / shap wheels).

## Files

| Path | Purpose |
|:--|:--|
| `app.py` | Streamlit UI |
| `app_core.py` | Prediction / SHAP / planning / metrics |
| `outputs/models/CatBoost.cbm` | Trained CatBoost model (native format) |
| `outputs/models/shap_background.csv` | 100-row background (standardised scale) |
| `outputs/models/manifest.csv` | Model metadata (features, threshold, versions) |
| `outputs/standard_scaler.pkl` | Standardiser for the six continuous features |
| `outputs/pred_test.csv` | Test-set per-case probabilities (cohort percentile) |
| `outputs/cache/boot_models_deploy.joblib` | 300 bootstrap-refitted CatBoost models (uncertainty) |

## Notes

- **Memory**: the bundled bootstrap cache contains 300 models (~23 MB on disk, ~120 MB RAM)
  to stay well below the 1 GB limit of Community Cloud. The local analysis uses 1,000
  models; to use the full set, replace `outputs/cache/boot_models_deploy.joblib` with the
  `boot_models_*.joblib` file from `outputs/cache/` in the analysis repository (~78 MB,
  ~400 MB RAM — may exceed the free-tier limit).
- **SHAP**: interventional definition with the 100-row background; the plot labels feature
  values on the clinical scale while SHAP values are computed on standardised inputs.
- **Uncertainty**: reflects training-sample resampling only, not model misspecification or
  patient-level outcome randomness.
- **Threshold 0.30** was derived in the development cohort.
- Research prototype — **not medical advice**.
