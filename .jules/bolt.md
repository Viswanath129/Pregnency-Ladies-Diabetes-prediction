## 2024-05-18 - Async Endpoint Threadpool Bottleneck
**Learning:** In FastAPI, CPU-bound tasks like ML inference (scikit-learn `predict_proba`) should not be defined within `async def` endpoints, because this blocks the single threaded asyncio event loop and prevents other requests from being handled concurrently.
**Action:** Always use standard `def` for synchronous, CPU-bound inference endpoints so that FastAPI safely delegates them to a background threadpool.

## 2024-05-18 - DataFrame Allocation Overhead
**Learning:** Instantiating `pandas.DataFrame` from arrays for single-row inference carries significant overhead compared to executing predictions directly on `numpy.array`.
**Action:** Bypass DataFrame creation during live inference endpoints, utilizing 2D NumPy arrays instead, and safely suppress the subsequent `UserWarning` from scikit-learn regarding missing feature names using `warnings.catch_warnings()`. Ensure native types are outputted by wrapping numpy arrays in `float()`.
