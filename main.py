import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/student_data.csv")

print("First 5 rows:\n")
print(df.head())

# -----------------------------
# 2. Feature Selection
# -----------------------------
X = df[["study_hours", "attendance", "previous_score"]]
y = df["pass"]

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 4. Model Training
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Save trained model
joblib.dump(model, "model.pkl")

print("Model saved successfully.")