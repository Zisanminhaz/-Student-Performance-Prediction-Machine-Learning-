

# Student Performance Prediction (Machine Learning)

This project predicts students' math exam scores using other performance and demographic features with a Random Forest Regression model.

---

## Project Overview

The dataset used is `StudentsPerformance.csv`, and the goal is to predict a student’s **math score** based on other exam results and demographic data.

### Features:

* Data preprocessing and feature scaling
* Random Forest model for regression
* Evaluation using R² and RMSE
* Visualization of Actual vs Predicted math scores

---

## Technologies Used

* Python
* Pandas
* NumPy
* scikit-learn
* Matplotlib

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Zisanminhaz/-Student-Performance-Prediction-Machine-Learning-.git
cd Student-Performance-Prediction-Machine-Learning-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 3. Run the Model

```bash
python model.py
```

Make sure the dataset (`StudentsPerformance.csv`) is in the project folder.

---

## Evaluation Metrics

* **R² Score**: Measures how well the model explains the variance in the data.
* **RMSE (Root Mean Squared Error)**: Measures the average magnitude of error in predictions.

---

## Visualization

The model's performance is visualized using a scatter plot comparing **actual** vs **predicted** math scores.

---

## License

This project is distributed under the MIT License.


