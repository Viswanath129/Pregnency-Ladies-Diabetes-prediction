import re

with open("backend/app.py", "r") as f:
    content = f.read()

# Add imports
content = content.replace("import joblib", "import joblib\nimport warnings\nimport random\nimport math")

# Make predict sync
content = content.replace("async def predict_risk", "def predict_risk")

# Optimize predict_risk body
old_body = """        if MODELS_LOADED:
            df = pd.DataFrame([vitals], columns=cols)
            scaled_data = MODELS["scaler"].transform(df)

            # Get probabilities from individual streams
            p_ml = MODELS["ml"].predict_proba(scaled_data)[:, 1][0]

            # ANN prediction (Handling potential different formats)
            try:
                p_ann = MODELS["ann"].predict_proba(scaled_data)[:, 1][0]
            except:
                pred = MODELS["ann"].predict(scaled_data)
                p_ann = pred[0][0] if len(pred.shape) > 1 else pred[0]

            # Simulated Quantum variance
            p_q = np.clip(p_ml + np.random.normal(0, 0.02), 0, 1)

            # Final Meta-AI decision
            meta_input = pd.DataFrame([[p_ml, p_ann, p_q]], columns=['Classical_Prob', 'ANN_Prob', 'Quantum_Prob'])
            final_prob = MODELS["meta"].predict_proba(meta_input)[:, 1][0]"""

new_body = """        if MODELS_LOADED:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                arr = np.array([vitals])
                scaled_data = MODELS["scaler"].transform(arr)

                # Get probabilities from individual streams
                p_ml = float(MODELS["ml"].predict_proba(scaled_data)[:, 1][0])

                # ANN prediction (Handling potential different formats)
                try:
                    p_ann = float(MODELS["ann"].predict_proba(scaled_data)[:, 1][0])
                except:
                    pred = MODELS["ann"].predict(scaled_data)
                    p_ann = float(pred[0][0] if len(pred.shape) > 1 else pred[0])

                # Simulated Quantum variance (Pure Python)
                p_q = max(0.0, min(1.0, p_ml + random.gauss(0, 0.02)))

                # Final Meta-AI decision
                meta_input = np.array([[p_ml, p_ann, p_q]])
                final_prob = float(MODELS["meta"].predict_proba(meta_input)[:, 1][0])"""

content = content.replace(old_body, new_body)

# Remove cols
content = content.replace('        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]\n', '')

# Optimize uncertainty calculation
old_std = '"uncertainty": round(float(np.std([p_ml, p_ann, p_q])), 4),'
new_std = '''# Pure Python std
    mean = (p_ml + p_ann + p_q) / 3.0
    var = ((p_ml - mean)**2 + (p_ann - mean)**2 + (p_q - mean)**2) / 3.0

    return {
        "risk_percent": risk_pct,
        "risk_label": label,
        "uncertainty": round(float(math.sqrt(var)), 4),'''

content = content.replace('''    return {
        "risk_percent": risk_pct,
        "risk_label": label,
        "uncertainty": round(float(np.std([p_ml, p_ann, p_q])), 4),''', new_std)

with open("backend/app.py", "w") as f:
    f.write(content)
