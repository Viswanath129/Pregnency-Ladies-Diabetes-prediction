## 2024-05-24 - Bypass DataFrame Instantiation & Externalize CPU Tasks
**Learning:** Instantiating `pandas.DataFrame` for single-row inference carries significant overhead in scikit-learn pipelines. Additionally, running CPU-bound ML tasks using `async def` in FastAPI blocks the event loop.
**Action:** When performing single-row predictions, use 2D `numpy.array` wrapped in `warnings.catch_warnings()` to suppress feature name warnings. Always use standard `def` for CPU-heavy endpoint handlers in FastAPI so it runs in an external thread pool.
