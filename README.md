# Hybrid-Machine-Learning-for-Bankruptcy-Forecasting
📌 Project Overview
This project focuses on advancing bankruptcy prediction using hybrid machine learning models (XGBoost + ANN) and ensemble learning (Voting Classifier) applied to an unbalanced Polish bankruptcy dataset. The goal is to improve financial risk assessment by addressing class imbalance and optimizing model performance.

🔗 Deployment: Flask-based web interface with SQLite backend.

🛠️ Key Features
✅ Hybrid Model: Combines XGBoost + Artificial Neural Network (ANN) for robust predictions.
✅ Optimization: Uses Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) for hyperparameter tuning.
✅ Ensemble Learning: Implements a Voting Classifier (Random Forest + Decision Trees) to boost accuracy.
✅ Data Preprocessing: Handles class imbalance, feature selection, and label encoding.
✅ Scalable Deployment: Built with Flask, SQLite, and Bootstrap for a user-friendly interface.

📊 Methodology
Data Exploration: Analyzed the Polish bankruptcy dataset.

Preprocessing:

Dropped irrelevant features.

Applied Keras transformers and label encoding.

Used Select Mutual for feature selection.

Model Development:

Trained SVC, XGBoost, ANN, Random Forest.

Optimized with GA/PSO.

Evaluation: Metrics include accuracy, precision, recall, F1-score.

Deployment: Flask app for real-time predictions.

💻 Tech Stack
Languages: Python
Frameworks: Flask, Jupyter Notebook
ML Libraries: Scikit-learn, XGBoost, TensorFlow/Keras
Database: SQLite3
Frontend: HTML, CSS, JavaScript, Bootstrap
Tools: Anaconda, Pandas, NumPy

🚀 Installation & Usage
Clone the repo:

bash
git clone https://github.com/your-repo/bankruptcy-forecasting.git  
Install dependencies:

bash
pip install -r requirements.txt  
Run the Flask app:

bash
python app.py  
Access the web interface:
Open http://localhost:5000 in your browser.

📈 Results
Hybrid XGBoost+ANN achieved ~95% accuracy after GA/PSO tuning.

Voting Classifier outperformed standalone models.

Flask app provides real-time bankruptcy risk scores.
