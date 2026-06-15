# 02_feature_engineer.py  (updated to create crop one-hot for regressors)
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

IN = Path('../data/cleaned_dataset.csv')
OUT_DIR = Path('../models')
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

# Derived features
df['NPK_sum'] = df['Nitrogen'] + df['Phosphorus'] + df['Potassium']
df['N_ratio'] = df['Nitrogen'] / (df['NPK_sum'] + 1e-6)
df['P_ratio'] = df['Phosphorus'] / (df['NPK_sum'] + 1e-6)
df['K_ratio'] = df['Potassium'] / (df['NPK_sum'] + 1e-6)
df['cost_to_price'] = df['input_cost'] / (df['market_price'] + 1e-6)

# Numerical and categorical features lists (same as before)
num_features = [
    'rainfall_mm','humidity','Temperature','pH_Value',
    'Nitrogen','Phosphorus','Potassium','NPK_sum',
    'N_ratio','P_ratio','K_ratio','input_cost','market_price','cost_to_price'
]
cat_features = ['soil_type','state']

for c in num_features:
    if c not in df.columns:
        df[c] = 0.0

X_num = df[num_features].fillna(0).values

# OHE for soil,state (existing)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = ohe.fit_transform(df[cat_features].astype(str))

# --- NEW: OHE for Crop (only for regressors) ---
ohe_crop = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_crop_ohe = ohe_crop.fit_transform(df[['Crop']].astype(str))

# Build X for classifiers (no crop)
X_cls = np.hstack([X_num, X_cat])
# Build X for regressors (include crop one-hot)
X_reg = np.hstack([X_num, X_cat, X_crop_ohe])

# scale (fit scaler on combined columns used by models; we create separate scalers if desired)
scaler = StandardScaler().fit(X_cls)    # keep old scaler behavior for classification
scaler_reg = StandardScaler().fit(X_reg)  # scaler used for regressors

Xs = scaler.transform(X_cls)
Xs_reg = scaler_reg.transform(X_reg)

# labels
le = LabelEncoder().fit(df['Crop'].astype(str))
y_cls = le.transform(df['Crop'].astype(str))
y_reg = df.get('yield_kg_per_ha_final')
if y_reg is None:
    raise ValueError("Expected column 'yield_kg_per_ha_final' in cleaned dataset")
y_reg = y_reg.values

# Save artifacts for classifiers (unchanged)
np.save(OUT_DIR / 'X.npy', Xs)
np.save(OUT_DIR / 'y_cls.npy', y_cls)
joblib.dump(ohe, OUT_DIR / 'ohe.pkl')
joblib.dump(scaler, OUT_DIR / 'scaler.pkl')
joblib.dump(le, OUT_DIR / 'labelencoder_crop.pkl')

# Save artifacts for regressors (NEW)
np.save(OUT_DIR / 'X_reg.npy', Xs_reg)
np.save(OUT_DIR / 'y_reg.npy', y_reg)
joblib.dump(ohe_crop, OUT_DIR / 'ohe_crop.pkl')
joblib.dump(scaler_reg, OUT_DIR / 'scaler_reg.pkl')

# Also save simple crop mean yields for quick fallback
crop_means = df.groupby('Crop')['yield_kg_per_ha_final'].mean().to_dict()
import json
Path(OUT_DIR / 'crop_means.json').write_text(json.dumps(crop_means))

print("Saved classifier and regressor artifacts to", OUT_DIR)
print("Classes:", list(le.classes_))
