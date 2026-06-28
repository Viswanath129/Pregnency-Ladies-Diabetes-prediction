
## 2024-05-18 - [FastAPI Threading & Math Overhead]
**Learning:** In FastAPI, CPU-bound endpoints (like scikit-learn model serving) shouldn't be `async def`. It blocks the main event loop. Also, for tiny datasets or single-row inferences, `pandas.DataFrame` initialization and numpy scalar functions (`np.clip`, `np.std`) add huge measurable latency compared to pure python implementations and basic numpy arrays.
**Action:** When writing ML inference APIs, use sync `def` to utilize FastAPI's external threadpool. Always strip out `pandas` in favor of `np.array` for single rows, and use pure Python (`math`/`random`/`min`/`max`) for scalar operations in high-throughput hot paths.
