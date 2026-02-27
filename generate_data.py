import pandas as pd
import numpy as np

np.random.seed(42)

n = 200

study_hours = np.random.randint(0, 10, n)
attendance = np.random.randint(50, 100, n)
previous_score = np.random.randint(30, 100, n)

# Define passing logic
pass_status = (
    (study_hours * 5 + attendance * 0.3 + previous_score * 0.5) > 100
).astype(int)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "pass": pass_status
})

df.to_csv("data/student_data.csv", index=False)

print("Dataset generated successfully.")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print("\nFeature Importance:\n")
print(coefficients)