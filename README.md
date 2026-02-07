Student Performance Prediction using Machine Learning
Overview
This project builds a machine learning model to predict students’ math scores based on demographic and academic features.
The goal is to demonstrate a complete end‑to‑end ML workflow including preprocessing, model training, evaluation, and visualization using industry‑standard practices.

Problem Statement
Given student background information and previous exam scores, predict the math score as accurately as possible using supervised learning.

Problem Type: Regression

Target Variable: math score

Dataset
Source: Kaggle – Students Performance Dataset

Features
Gender

Race/Ethnicity

Parental level of education

Lunch type

Test preparation course

Reading score

Writing score

Tools and Technologies
Python

Pandas, NumPy

Scikit‑learn

Matplotlib

Methodology
Load and clean the dataset

Separate numerical and categorical features

Apply One‑Hot Encoding to categorical features

Apply Standard Scaling to numerical features

Split data into training and testing sets (80/20)

Train a Random Forest Regressor

Evaluate model performance using regression metrics

Visualize results using an Actual vs Predicted plot

Model
Algorithm: Random Forest Regressor
Reason: Strong performance on tabular data and ability to capture non‑linear relationships.

Results
R² Score: 0.85

RMSE: ~6

These results indicate strong predictive performance with good generalization and no overfitting.

Visualization
An Actual vs Predicted scatter plot is used to validate model performance.
Predictions closely align with real values, confirming model reliability.

How to Run the Project
pip install pandas numpy scikit-learn matplotlib
python model.py
Project Structure
ML_Project/
│
├── model.py
├── StudentsPerformance.csv
├── README.md
Key Learnings
Proper preprocessing significantly improves model performance

Using pipelines prevents data leakage

Random Forest performs well for structured regression problems

Future Work
Hyperparameter tuning with GridSearchCV

Model comparison with Linear Regression and Gradient Boosting

Convert problem to classification (Pass/Fail analysis)

Author
Minhaz Alam Jisan
Machine Learning and Python Enthusiast
