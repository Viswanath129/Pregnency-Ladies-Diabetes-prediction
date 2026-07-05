## 2025-02-18 - [FastAPI CPU Bound ML]
**Learning:** [In FastAPI, `async def` endpoints that run synchronous, CPU-bound machine learning inferences (like scikit-learn models) will block the main event loop, severely degrading concurrent performance.]
**Action:** [Always use standard `def` for FastAPI endpoints performing heavy synchronous computations to allow execution in an external thread pool.]

## 2025-02-18 - [Pandas vs NumPy in Single-Row Inference]
**Learning:** [Instantiating `pandas.DataFrame` for single-row inference in high-throughput endpoints introduces massive overhead compared to using native `numpy` 2D arrays.]
**Action:** [Bypass pandas entirely and use `numpy.array([[...]])` for processing single data points through scikit-learn pipelines. Suppress 'missing feature names' warnings globally with `warnings.filterwarnings` to avoid thread-safety issues.]

## 2025-02-18 - [NumPy Overhead for Scalars]
**Learning:** [Using NumPy functions (`np.clip`, `np.std`, `np.random.normal`) for scalar values or tiny arrays is much slower than using pure Python built-ins (`min/max`, `math.sqrt`, `random.gauss`) due to NumPy's dispatching overhead.]
**Action:** [Replace scalar NumPy operations with built-in `math` and `random` equivalents in latency-sensitive paths.]
