# 💳 Credit Card Fraud Detection (ML Project)

This is an end-to-end machine learning project to detect fraudulent credit card transactions. It uses a real-world imbalanced dataset and tracks experiments using MLflow.

## 📁 Project Structure

credit-card-fraud-detection/
├── data/ # Raw data (ignored in .gitignore)
├── models/ # Trained model files
├── src/ # Source code: training, prediction
├── .gitignore
├── requirements.txt
└── README.md
## 📊 Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions, only 492 frauds (highly imbalanced)

## 🧠 Techniques Used

- Data Cleaning and Preprocessing
- Train-Test Split
- Random Forest Classifier
- MLflow for experiment tracking
- Model evaluation (Accuracy, Precision, Recall, F1-Score)
- Joblib for model saving
- Git for version control

## 🚀 How to Run

1. How to Install Dependencies
   pip install -r requirements.txt

2. Run Training Script
   python src/train_model.py

3. (Optional) Launch MLflow UI
   mlflow ui
# Then visit: http://127.0.0.1:5000

✅ Output Sample
Accuracy: 99.9%
Precision: 0.87
Recall: 0.83
Confusion Matrix:
[[56852    12]
 [   17    81]]