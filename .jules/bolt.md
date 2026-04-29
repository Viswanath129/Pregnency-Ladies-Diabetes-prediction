## 2026-04-29 - Scikit-learn Single Row Inference
**Learning:** Instantiating pandas DataFrames for single-row inference in scikit-learn introduces significant overhead compared to 2D numpy arrays.
**Action:** Use 2D `numpy.array` instead of `pd.DataFrame` for single-row inference, and wrap in `warnings.catch_warnings()` to suppress UserWarning for missing feature names.
