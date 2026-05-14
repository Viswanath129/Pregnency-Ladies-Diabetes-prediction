## 2024-05-18 - Avoid Pandas overhead and async for CPU-bound tasks
**Learning:** Instantiating `pandas.DataFrame` for single-row inference adds measurable overhead compared to raw 2D `numpy.array`. Additionally, CPU-bound ML inferences defined with `async def` in FastAPI block the main event loop.
**Action:** Always prefer 2D `numpy.array` and `warnings.catch_warnings()` (to handle scikit-learn feature name warnings) for single-row predictions. Define these endpoints with a standard `def` to allow FastAPI to offload execution to a thread pool.
