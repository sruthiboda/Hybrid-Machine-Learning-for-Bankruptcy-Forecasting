from flask import Flask, request, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# ✅ Load models and scaler safely
try:
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    ann_model = load_model("ann_model.keras")
except FileNotFoundError as e:
    raise RuntimeError(f"Model file missing: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = list(request.form.values())

        # Validate input length
        if len(input_values) != 64:
            return render_template(
                "index.html",
                prediction_text=f"Error: Expected 64 values, got {len(input_values)}."
            )

        # Convert inputs to floats
        try:
            input_features = [float(val) for val in input_values]
        except ValueError:
            return render_template(
                "index.html",
                prediction_text="Error: All inputs must be numeric."
            )

        # Scale features
        scaled = scaler.transform([input_features])

        # Predictions
        xgb_pred = float(xgb_model.predict_proba(scaled)[:, 1][0])
        ann_pred = float(ann_model.predict(scaled).flatten()[0])

        # Combine predictions
        combined = (xgb_pred + ann_pred) / 2
        result = "Bankrupt" if combined > 0.5 else "Not Bankrupt"
        confidence = round(combined * 100, 2)

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result} ({confidence}% confidence)"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == '__main__':
    if os.environ.get("RENDER") == "true":
        # Production mode on Render
        from waitress import serve
        port = int(os.environ.get("PORT", 8080))
        serve(app, host='0.0.0.0', port=port)
    else:
        # Local development mode
        import webbrowser
        from threading import Timer

        def open_browser():
            webbrowser.open_new("http://127.0.0.1:5000")

        Timer(1, open_browser).start()
        app.run(debug=True)



