import joblib
import numpy as np
import os

# 🧠 Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "rf_model.joblib")
model = joblib.load(model_path)

# 📦 Example transaction (random values for all 29 features)
example = np.array([[-1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396, 0.0987,
                     0.3637, 0.0908, -0.5516, -0.6178, -0.9914, -0.3111, 1.4681, -0.4704,
                     0.2079, 0.0257, 0.4032, 0.2514, -0.0183, 0.2779, -0.1105, -0.2957,
                     0.0678, 0.0546, -0.4997, 0.0023, 149.62]])

# 🧾 Make prediction
prediction = model.predict(example)[0]

# 🧮 Output result
if prediction == 1:
    print("🚨 FRAUD DETECTED in this transaction!")
else:
    print("✅ Transaction looks NORMAL.")
