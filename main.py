import pandas as pd
import numpy as np

from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error


# =========================
# Load dataset
# =========================

df = pd.read_excel(
    "cleaned2.xlsx",
    sheet_name="Sheet2"
)

pd.set_option('display.max_columns', None)


# =========================
# Outlier detection
# =========================

def find_outliers(col):
    z = np.abs(stats.zscore(col, nan_policy='omit'))
    return pd.Series(z > 3, index=col.index)


numeric_cols = df.select_dtypes(include=np.number).columns

df_outliers = pd.DataFrame(index=df.index)

for col in numeric_cols:
    df_outliers[col] = find_outliers(df[col])


# Remove outliers except O3

outlier_check = df_outliers.drop(columns=["O3"], errors="ignore")

rows_with_outliers = outlier_check.any(axis=1)

df_clean = df.loc[~rows_with_outliers].copy()




# =========================
# Scaling
# =========================

target = "O3"

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    df_clean[numeric_cols]
)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=numeric_cols,
    index=df_clean.index
)


# =========================
# Features and target
# =========================

X = scaled_df.drop(columns=[target])
Y = scaled_df[target]


# =========================
# Train-test split
# =========================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=10
)


# =========================
# Bagging Regressor
# =========================

bag_model = BaggingRegressor(
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)


# Train model

bag_model.fit(X_train, Y_train)


# =========================
# Test set prediction
# =========================

predictions_scaled = bag_model.predict(X_test)


# =========================
# Inverse transformation
# =========================

o3_index = list(numeric_cols).index(target)


# Empty arrays for inverse scaling
pred_scaled_full = np.zeros(
    (len(predictions_scaled), len(numeric_cols))
)

y_test_scaled_full = np.zeros(
    (len(Y_test), len(numeric_cols))
)


# Insert O3 values
pred_scaled_full[:, o3_index] = predictions_scaled
y_test_scaled_full[:, o3_index] = Y_test


# Convert back to original O3 values
predictions_original = scaler.inverse_transform(
    pred_scaled_full
)[:, o3_index]


Y_test_original = scaler.inverse_transform(
    y_test_scaled_full
)[:, o3_index]


# =========================
# Evaluation
# =========================

test_R2 = bag_model.score(X_test, Y_test)

RMSE_original = np.sqrt(
    mean_squared_error(
        Y_test_original,
        predictions_original
    )
)


print("\nTest Set Performance:")
print(f"R-squared: {test_R2:.3f}")
print(f"RMSE (original O3 units): {RMSE_original:.3f}")


