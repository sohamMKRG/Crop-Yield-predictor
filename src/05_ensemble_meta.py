# 05_ensemble_meta.py
import numpy as np, joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

MODELS_DIR = Path('../models')
X = np.load(MODELS_DIR/'X.npy')
y = np.load(MODELS_DIR/'y_cls.npy')
le = joblib.load(MODELS_DIR/'labelencoder_crop.pkl')
n_classes = len(le.classes_)

# OOF stacking with XGB and RF
oof = np.zeros((X.shape[0], n_classes))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, val_idx in skf.split(X, y):
    Xt, Xv = X[tr_idx], X[val_idx]
    yt = y[tr_idx]
    dtr = xgb.DMatrix(Xt, label=yt)
    dval = xgb.DMatrix(Xv)
    bst_fold = xgb.train({"objective":"multi:softprob","num_class":n_classes,"eval_metric":"mlogloss"}, dtr, num_boost_round=200)
    oof[val_idx] += bst_fold.predict(dval)
    rf_fold = RandomForestClassifier(n_estimators=150, random_state=42)
    rf_fold.fit(Xt, yt)
    oof[val_idx] += rf_fold.predict_proba(Xv)
# average probs
oof = oof / 2.0

meta = LogisticRegression(max_iter=1000)
meta.fit(oof, y)
joblib.dump(meta, MODELS_DIR/'meta_classifier.pkl')
print("Saved meta classifier to", MODELS_DIR)
