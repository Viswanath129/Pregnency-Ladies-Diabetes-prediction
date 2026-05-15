## 2024-05-24 - Bypass pandas DataFrames for single-row inference
**Learning:** Initializing pandas DataFrames and using DataFrame features for single-row inference within tight loops or web API endpoints introduces measurable and non-trivial overhead compared to native 2D numpy arrays.
**Action:** Always prefer 2D numpy arrays over DataFrames when performing single-row ML inference in real-time APIs. Remember to catch `UserWarning` if models were fit with feature names.
