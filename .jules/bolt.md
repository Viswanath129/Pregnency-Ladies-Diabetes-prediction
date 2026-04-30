
## 2024-05-18 - Pandas vs Numpy Instantiation for Single-Row Inference Overhead
**Learning:** In the FastAPI backend `predict_risk` endpoint, initializing a `pandas.DataFrame` for single-row inference added significant overhead. Raw `numpy.array` was measured to be ~20x faster.
**Action:** Use 2D `numpy.array` instead of `pandas.DataFrame` for single-row scikit-learn model inference predictions in API endpoints to improve latency. Suppress `UserWarning` about feature names gracefully and ensure outputs are converted to native Python `float`.
