## 2024-05-23 - Optimize single-row ML inference in FastAPI
**Learning:** Instantiating `pandas.DataFrame` inside high-throughput API endpoints for single-row inference adds considerable overhead. Furthermore, executing CPU-bound tasks (like scikit-learn models) inside `async def` endpoints will block FastAPI's async event loop.
**Action:** Replace `pandas.DataFrame` with `numpy.array` wrapped in `warnings.catch_warnings()` for scikit-learn inference, and declare CPU-bound FastAPI endpoints with standard `def` instead of `async def` to ensure they run in a separate thread pool.
