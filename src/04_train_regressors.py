# 04_train_regressors.py  (train regressors using crop one-hot feature)
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

MODELS_DIR = Path('../models')
X = np.load(MODELS_DIR / 'X_reg.npy')    # now includes crop OHE
y_reg = np.load(MODELS_DIR / 'y_reg.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.20, random_state=42)

# XGBoost reg
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {"objective":"reg:squarederror","eval_metric":"rmse"}
bst_reg = xgb.train(params, dtrain, num_boost_round=300)
joblib.dump(bst_reg, MODELS_DIR / 'xgb_yield_reg.model')
pred_xgb = bst_reg.predict(dtest)
rmse_xgb = root_mean_squared_error(y_test, pred_xgb)
r2_xgb = r2_score(y_test, pred_xgb)
print(f"XGB RMSE: {rmse_xgb:.3f} | R2: {r2_xgb:.3f}")

# RandomForest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, MODELS_DIR / 'rf_yield_reg.pkl')
pred_rf = rf.predict(X_test)
rmse_rf = root_mean_squared_error(y_test, pred_rf)
r2_rf = r2_score(y_test, pred_rf)
print(f"RF RMSE: {rmse_rf:.3f} | R2: {r2_rf:.3f}")

print("Saved regressors to", MODELS_DIR)
