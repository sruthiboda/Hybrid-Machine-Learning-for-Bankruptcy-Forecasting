<<<<<<< HEAD
from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import joblib

# ✅ Initialize the Flask app
app = Flask(__name__)

# ✅ Load models and scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
ann_model = load_model("ann_model.keras")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = list(request.form.values())

        # ✅ Check input count
        if len(input_values) != 64:
            return render_template("index.html", prediction_text=f"Error: Expected 64 values, got {len(input_values)}.")

        # ✅ Convert to float
        input_features = [float(val) for val in input_values]

        # ✅ Scale and predict
        scaled = scaler.transform([input_features])
        xgb_pred = xgb_model.predict_proba(scaled)[:, 1]
        ann_pred = ann_model.predict(scaled).flatten()
        combined = (xgb_pred + ann_pred) / 2
        result = "Bankrupt" if combined[0] > 0.5 else "Not Bankrupt"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# ✅ Only for local testing (don’t use webbrowser or debug=True on Render)
if __name__ == '__main__':
    app.run()
=======
from flask import Flask, request, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ✅ Load models and scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
ann_model = load_model("ann_model.keras")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = list(request.form.values())

        if len(input_values) != 64:
            return render_template("index.html", prediction_text=f"Error: Expected 64 values, got {len(input_values)}.")

        input_features = [float(val) for val in input_values]
        scaled = scaler.transform([input_features])

        xgb_pred = xgb_model.predict_proba(scaled)[:, 1]
        ann_pred = ann_model.predict(scaled).flatten()

        combined = (xgb_pred + ann_pred) / 2
        result = "Bankrupt" if combined[0] > 0.5 else "Not Bankrupt"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# ✅ Production-ready server
if __name__ == '__main__':
    import os
    if os.environ.get("RENDER") == "true":
        from waitress import serve
        serve(app, host='0.0.0.0', port=8080)
    else:
        # Local run
        import webbrowser
        from threading import Timer

        def open_browser():
            webbrowser.open_new("http://127.0.0.1:5000")

        Timer(1, open_browser).start()
        app.run(debug=True)

>>>>>>> e6660292faee72ed70a9d298713fcca0118d5b2e
