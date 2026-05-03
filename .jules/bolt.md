
## 2024-05-03 - FastAPI ML Inference Performance Bottleneck & Pandas Overhead
**Learning:** For ML inference in FastAPI, synchronous operations block the event loop if using `async def`, whereas `def` offloads CPU-bound work (like scikit-learn models) to an external thread pool. Furthermore, for single-row inference, instantiating `pandas.DataFrame` adds significant overhead; using `numpy.array` is measurably faster.
**Action:** When creating prediction endpoints, prefer synchronous `def` if the work is primarily CPU bound, and pass 2D `numpy.array` explicitly into scikit-learn model's `predict` or `predict_proba` rather than dataframes for single-inference performance optimizations.
