## 2024-05-18 - Fast API and Scikit-learn Optimization

**Learning:** When using scikit-learn for single-row inference in a fast API endpoint, passing `pandas.DataFrame` and using numpy for simple scalar math introduces a huge overhead. Furthermore, declaring CPU bound endpoints with `async def` will block the FastAPI event loop, severely degrading concurrent performance.

**Action:** For ML inference endpoints, use synchronous `def` instead of `async def` to allow FastAPI to schedule the blocking workload onto a thread pool. Optimize single row prediction by swapping `pandas.DataFrame` for `numpy.array` and use the built-in python `random` and `math` libraries for scalar math rather than numpy. Wrap numpy array inputs to scikit-learn models in `warnings.catch_warnings()` to suppress missing feature name warnings.
