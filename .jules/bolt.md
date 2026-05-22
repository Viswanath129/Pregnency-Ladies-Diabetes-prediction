
## 2024-05-22 - FastAPI & Scikit-Learn Inference Bottlenecks
**Learning:** For CPU-bound operations like Scikit-Learn `.predict()` and `.transform()` calls, defining a FastAPI route with `async def` incorrectly forces the execution into the main event loop, severely degrading concurrent request handling. Additionally, creating single-row `pandas.DataFrame` objects just for model inference adds unnecessary instantiation overhead.
**Action:** Use standard `def` for FastAPI endpoints performing CPU-bound ML inference so FastAPI delegates them to a threadpool. When doing single-row inferences, bypass `pandas.DataFrame` by passing raw 2D `numpy.array`s instead. Suppress the missing feature names `UserWarning` locally with `warnings.catch_warnings` if the models were fitted with feature names.
