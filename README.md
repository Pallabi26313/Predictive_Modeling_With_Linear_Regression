NAME:Pallabi Ghosh COMPANY:CODTECH IT SOLUTIONS ID:CT08DS8204 DOMAIN:Data Analytics DURATION:September-October 2024

# Predictive Modeling with Linear Regression

### Overview
This project focuses on building a **simple linear regression model** using the **Boston Housing Dataset** to predict the median value of owner-occupied homes (MEDV). The model uses features like the average number of rooms (RM) to make predictions. The project demonstrates the entire process, from data loading and preprocessing to training, evaluating, and visualizing the model's results.

### Table of Contents
1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
3. [Data](#data)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Visualizations](#visualizations)
7. [Technologies Used](#technologies-used)
8. [Future Improvements](#future-improvements)

---

### Project Structure
```bash
Predictive_Modeling_With_Linear_Regression/
│
├── data/                   # Data folder (optional if dataset stored locally)
│   └── boston_housing.csv   # Dataset (Boston Housing from OpenML)
│
├── notebooks/               # Jupyter notebooks for model training and evaluation
│   └── linear_regression.ipynb
│
├── src/                     # Source code folder (optional)
│   └── linear_regression.py  # Python script for model implementation
│
├── README.md                # Project description and instructions
└── requirements.txt         # Python dependencies
```

---

### Getting Started

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Predictive_Modeling_With_Linear_Regression.git
cd Predictive_Modeling_With_Linear_Regression
```

#### 2. Install dependencies
To install the required libraries, run:
```bash
pip install -r requirements.txt
```

#### 3. Run the Jupyter notebook or Python script
You can run the project either using a Jupyter notebook or a Python script:
- Open `notebooks/linear_regression.ipynb` for the full code in a notebook environment.
- Alternatively, you can run the model using the Python script `src/linear_regression.py`.

---

### Data
The **Boston Housing Dataset** contains various features about houses in Boston suburbs, including the number of rooms, crime rate, property tax rate, etc.

- **Features**:
  - `CRIM`: Per capita crime rate by town.
  - `ZN`: Proportion of residential land zoned for lots over 25,000 sq. ft.
  - `RM`: Average number of rooms per dwelling (used as a predictor in this project).
  - `AGE`: Proportion of owner-occupied units built prior to 1940.
  - **Target Variable**: `MEDV` (Median value of homes in $1000s).

---

### Modeling

#### Steps:
1. **Data Loading**: The dataset is fetched from OpenML using `fetch_openml()` from the **scikit-learn** library.
2. **Preprocessing**: Basic preprocessing steps include checking for missing values and selecting features.
3. **Feature Selection**: The feature `RM` (average number of rooms per dwelling) is selected as the independent variable, and `MEDV` (median value of homes) is used as the target.
4. **Model Training**: A **simple linear regression** model is trained using the `LinearRegression` class from **scikit-learn**.
5. **Prediction**: Predictions are made on the test set.

---

### Evaluation
The model's performance is evaluated using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **R-squared (R²)**: Measures how well the independent variable explains the variance in the dependent variable.

---

### Visualizations
1. **Regression Line**: Plotted against the training data to show the fit of the model.
2. **Actual vs Predicted Values**: Visualized using a scatter plot to assess the model's accuracy.

```python
# Plot the regression line
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, lr.predict(X_train), color='red', linewidth=2)
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median Value of Homes (MEDV)')
plt.title('Simple Linear Regression: RM vs MEDV')
plt.show()

# Actual vs Predicted Values
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted MEDV')
plt.show()
```

---

### Technologies Used
- **Python**: Programming language used for data manipulation and modeling.
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical operations.
- **Scikit-learn**: Machine learning library used for model training and evaluation.
- **Matplotlib & Seaborn**: Libraries for visualizing data and model results.

---

### Future Improvements
- **Feature Engineering**: Adding more features from the dataset (e.g., `LSTAT`, `AGE`) to create a multivariate regression model.
- **Regularization**: Implementing techniques such as **Ridge** or **Lasso** regression to prevent overfitting and improve model generalization.
- **Cross-Validation**: Using **k-fold cross-validation** to tune hyperparameters and get more robust results.

---

