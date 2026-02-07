import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1️⃣ Load data
df = pd.read_csv("StudentsPerformance.csv").dropna()

y = df["math score"]
X = df.drop("math score", axis=1)

# 2️⃣ Column types (FIX warning)
cat_cols = X.select_dtypes(include=["object", "string"]).columns
num_cols = X.select_dtypes(exclude=["object", "string"]).columns

# 3️⃣ Preprocessing (CORRECT)
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

# 4️⃣ Model
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42
)

# 5️⃣ Pipeline
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# 6️⃣ Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Train
pipeline.fit(X_train, y_train)

# 8️⃣ Evaluate
pred = pipeline.predict(X_test)
print("R2:", r2_score(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

import matplotlib.pyplot as plt

plt.scatter(y_test, pred)
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual vs Predicted")
plt.show()




