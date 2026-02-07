🚀 Project Overview
Problem type: Regression

Target variable: math score

Model used: Random Forest Regressor

Goal: Predict student performance accurately using proper preprocessing and no data leakage

📂 Dataset
Source: Kaggle – Students Performance Dataset
Features include:

Gender

Race/ethnicity

Parental level of education

Lunch type

Test preparation course

Reading & writing scores

⚙️ Technologies Used
Python 3

Pandas, NumPy

Scikit‑learn

Matplotlib

🧠 ML Pipeline
Data cleaning (null removal)

Feature separation (categorical & numerical)

One‑Hot Encoding (categorical features)

Standard Scaling (numerical features)

Train‑test split (80/20)

Model training using Random Forest

Evaluation using R² and RMSE

Visualization (Actual vs Predicted)

📊 Model Performance
R² Score: 0.85

RMSE: ~6

✅ Strong predictive performance
✅ No overfitting
✅ Industry‑standard preprocessing

📈 Visualization
The Actual vs Predicted plot shows predictions closely aligned with real values, confirming strong generalization.

▶️ How to Run
pip install pandas numpy scikit-learn matplotlib
python model.py
📌 Key Learnings
Proper preprocessing improves performance significantly

Avoiding data leakage is critical

Random Forest works well for tabular regression problems

🔮 Future Improvements
Compare with Linear Regression & XGBoost

Hyperparameter tuning with GridSearchCV

Convert to classification (Pass/Fail) and analyze ROC‑AUC
