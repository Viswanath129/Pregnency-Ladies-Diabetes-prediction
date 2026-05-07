## 2024-05-19 - Pandas overhead in single-row model inference
**Learning:** Instantiating `pd.DataFrame` inside the FastAPI endpoint for single-row scikit-learn model inference introduces significant overhead (approx 3ms per request).
**Action:** Bypass `pd.DataFrame` and use 2D `np.array` instead. Wrap the prediction logic in `warnings.catch_warnings()` to cleanly suppress `UserWarning` regarding missing feature names, and cast numpy scalar outputs to Python `float` before building the response.
