# Atmospheric Ozone (O3) Concentration Prediction Using Machine Learning

## Overview

This project focuses on predicting **atmospheric ozone (O3) concentration** in the city of Athens using machine learning techniques. The dataset contains hourly measurements of multiple atmospheric components collected over a period of **five years**.

The objective of this project was to develop a regression model capable of predicting ozone concentration based on the relationship between O3 levels and other environmental measurements.

The project follows a complete machine learning workflow, including data cleaning, outlier detection, feature analysis, data preprocessing, model training, and performance evaluation.

## Dataset

The dataset consists of hourly atmospheric measurements collected in Athens over five years.

The target variable is:

- **O3 concentration** - Atmospheric ozone concentration to be predicted

The remaining atmospheric components were used as input features to train the regression model.

## Data Preprocessing

Several preprocessing steps were applied to improve the quality of the dataset and prepare it for machine learning:

### Outlier Detection and Removal

Outliers were identified using the **Z-score method**.

Measurements with a Z-score greater than 3 were considered abnormal and removed from the dataset to reduce the impact of extreme values on model performance.

### Feature Analysis

The relationship between the atmospheric components and ozone concentration was analyzed using **Pearson correlation**.

Based on the correlation results, three components with low contribution to ozone prediction were excluded from the final dataset.

### Feature Scaling

The remaining features were standardized using **StandardScaler** to ensure that all variables were on a comparable scale before model training.

## Machine Learning Model

For this regression problem, a **Bagging Regressor** from the Scikit-learn library was implemented.

The model uses an ensemble learning approach by combining multiple estimators to improve prediction stability and reduce overfitting.

Model configuration:

- Number of estimators: 100
- Maximum samples per estimator: 80%
- Out-of-bag evaluation enabled

## Model Evaluation

The dataset was divided into training and testing sets, and the model performance was evaluated using regression metrics.

Evaluation metrics:

- **R-squared (R²)**
- **Root Mean Squared Error (RMSE)**

The final model achieved:

| Metric | Score |
|--------|-------|
| R² Score | 0.79 |
| RMSE | 0.21 |

The results indicate that the model was able to capture a significant portion of the relationship between atmospheric variables and ozone concentration.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib (for visualization, if applicable)

## Machine Learning Workflow

```
Raw Atmospheric Data
          |
          ↓
Data Cleaning
          |
          ↓
Outlier Detection (Z-score)
          |
          ↓
Feature Selection (Pearson Correlation)
          |
          ↓
Feature Scaling
          |
          ↓
Train/Test Split
          |
          ↓
Bagging Regression Model
          |
          ↓
Performance Evaluation
```

## Key Skills Demonstrated

- Data preprocessing and cleaning
- Statistical analysis for feature selection
- Handling outliers in real-world datasets
- Regression model development
- Ensemble learning techniques
- Model evaluation and interpretation

## Conclusion

This project demonstrates the application of machine learning techniques to an environmental prediction problem using real-world atmospheric data. By combining statistical analysis, preprocessing techniques, and ensemble regression, the model was able to achieve reliable ozone concentration predictions.

The project provided practical experience in building a complete machine learning pipeline, from raw data processing to model evaluation.
