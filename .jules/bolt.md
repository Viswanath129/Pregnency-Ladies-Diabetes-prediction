## 2024-05-28 - Optimizing single-row ML inference in FastAPI
**Learning:** In a FastAPI backend serving classical ML models, passing data directly as a 2D `numpy.array` instead of a `pandas.DataFrame` avoids significant overhead from DataFrame instantiation and input validation during single-row inference.
**Action:** Always prefer `numpy` arrays over `pandas` DataFrames for high-frequency, single-record model inference, wrapping the call in `warnings.catch_warnings()` if missing feature names generate warnings.
