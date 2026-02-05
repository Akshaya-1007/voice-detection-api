import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------- CONFIG --------
CSV_PATH = r"C:\Users\aksha\Downloads\Audio (Buildathon)\features.csv"
MODEL_PATH = r"C:\Users\aksha\Downloads\Audio (Buildathon)\voice_detector.pkl"
SCALER_PATH = r"C:\Users\aksha\Downloads\Audio (Buildathon)\scaler.pkl"
# ------------------------

# Load data
df = pd.read_csv(CSV_PATH)

X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (VERY IMPORTANT)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model Training Completed")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["HUMAN", "AI_GENERATED"]))

# Save model and scaler
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n💾 Model saved as:", MODEL_PATH)
print("💾 Scaler saved as:", SCALER_PATH)
