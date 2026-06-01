## 2024-06-13 - [FastAPI ML Inference Bottlenecks]
**Learning:** Instantiating `pandas.DataFrame` structures for every incoming request inside a FastAPI prediction route adds a significant (often 10%+) latency overhead for small payload inference.
**Action:** Use 2D `numpy.array` primitives instead, wrapping calls in `warnings.catch_warnings()` to suppress expected `sklearn` missing feature warnings gracefully. Additionally, primitive math modules (`math`, `random`) operate faster on scalar types compared to their numpy equivalents (`np.std`, `np.random.normal`).
