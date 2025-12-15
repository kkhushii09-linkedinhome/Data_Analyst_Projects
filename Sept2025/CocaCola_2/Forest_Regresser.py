import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ======================
# DATA FETCH
# ======================
ticker = "KO"
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")

# 🔑 Fix MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.ffill()

# ======================
# FEATURE ENGINEERING
# ======================
data["MA_20"] = data["Close"].rolling(20).mean()
data["MA_50"] = data["Close"].rolling(50).mean()
data["Daily_Return"] = data["Close"].pct_change()
data["Volatility"] = data["Daily_Return"].rolling(20).std()
data.dropna(inplace=True)

features = ["Open", "High", "Low", "Volume",
            "MA_20", "MA_50", "Daily_Return", "Volatility"]

X = data[features]
y = data["Close"]

# ======================
# TRAIN / TEST SPLIT (IMPORTANT)
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ======================
# RANDOM FOREST TUNING
# ======================
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    'n_estimators': [300, 500, 800],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=30,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# ======================
# EVALUATION
# ======================
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Best Parameters:", random_search.best_params_)
print("MAE:", mae)
print("RMSE:", rmse)