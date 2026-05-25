## 2024-05-24 - ML Inference on FastAPI Threading & Scikit-Learn Overhead
**Learning:**
In FastAPI, marking an endpoint with `async def` runs it on the main asyncio event loop. When executing CPU-bound ML inferences (like scikit-learn model loading and prediction), doing this inside an `async def` blocks the event loop, causing severe latency for all other concurrent requests. Additionally, constructing a `pandas.DataFrame` for single-row inference introduces massive overhead compared to using a native `numpy.array`.

**Action:**
1. Always define endpoints performing heavy CPU tasks (like ML inference) with standard `def` (instead of `async def`), which allows FastAPI to offload execution to an external thread pool.
2. For single-row scikit-learn inference, bypass `pandas.DataFrame` entirely by passing a 2D `numpy.array` and suppressing the `UserWarning` (for missing feature names) via `warnings.catch_warnings()`.
