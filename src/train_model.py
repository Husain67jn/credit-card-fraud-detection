import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import mlflow

from data_loader import load_data

# Set experiment name
mlflow.set_experiment("Credit Card Fraud Detection")

# Start MLflow run
with mlflow.start_run():
    print("✅ Dataset loaded successfully.")
    df = load_data("c:/Users/Family PC/Desktop/credit_card_fraud_detection/data/creditcard.csv")

    # Preprocessing
    X = df.drop(columns=['Time', 'Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    print("📚 Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("✅ Model training complete.")

    # Evaluation
    y_pred = clf.predict(X_test)
    print("\n🧾 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Logging with MLflow
    mlflow.sklearn.log_model(clf, "random_forest_model")
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # 💾 Save model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_model.joblib")
    joblib.dump(clf, model_path)
    print(f"\n💾 Model saved at: {model_path}")
