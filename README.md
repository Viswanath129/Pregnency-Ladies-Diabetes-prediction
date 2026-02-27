# Pregnancy Ladies Diabetes Prediction

## Project Overview
This project focuses on predicting diabetes risk in pregnant women using a combination of trained machine learning models, artificial neural networks, and simulated quantum variance. It features a FastAPI backend and a web-based user interface for real-time predictions.

## Research Question
How can we effectively combine different predictive modeling techniques (Classical ML, ANN, and simulated Quantum variance) to provide robust diabetes risk assessments during pregnancy?

## Methodology
The pipeline takes patient vitals (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age) and scales the input. The scaled data is passed through multiple predictive streams:
1. **Classical Machine Learning** (Scikit-Learn)
2. **Artificial Neural Network (ANN)**
3. **Simulated Quantum Variance** (adding gaussian noise to simulate quantum uncertainty)

The results are then aggregated using a **Meta-AI** model to output a final probability and diabetes risk label.

## Architecture
- **Backend:** FastAPI, Pandas, NumPy, Scikit-Learn
- **Storage/Models:** `.joblib` serialized Scikit-learn/ANN pipelines
- **Frontend:** HTML/JS integrated via FastAPI `FileResponse`

## Experiments
- Testing different weighting mechanisms for Meta-AI.
- Evaluating latency and response time using a real-time FastAPI local server.

## Preliminary Results
| Stream | Function | Output Type |
|--------|----------|-------------|
| Classical | Baseline probability | Float (0 to 1) |
| ANN | High-dimensional pattern extraction | Float (0 to 1) |
| Quantum | Uncertainty simulation | Float (0 to 1) |
| **Meta-AI** | Final Aggregation | **Risk % & Label** |

## Observations
- The ensemble Meta-AI approach provides smoother probability surfaces compared to isolated streams.
- Model loading is lightweight, enabling rapid real-time inference on local servers.

## Limitations
- Simulated Quantum stream rather than real quantum hardware.
- Local inference only; not yet tested on edge or cloud concurrent loads.

## Future Work
- Integration with true cloud-based Quantum APIs (e.g., Qiskit).
- Expanding the frontend dashboard with historical patient tracking.
- Optimizing model size for edge deployment.
