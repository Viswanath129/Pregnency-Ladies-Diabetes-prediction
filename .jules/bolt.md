## 2026-05-02 - Optimize inference endpoint latency
**Learning:** For CPU-bound endpoints (like ML inference) in FastAPI, using `async def` blocks the event loop. Furthermore, instantiating `pandas.DataFrame` for single-row inference adds unnecessary latency overhead compared to using `numpy.array`.
**Action:** Use standard `def` for CPU-bound routes so FastAPI can run them in a thread pool. Use 2D `numpy.array` inputs for single-row scikit-learn model inference and wrap calls in `warnings.catch_warnings()` to suppress UserWarnings regarding missing feature names.
