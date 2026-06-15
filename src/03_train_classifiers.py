# 03_train_classifiers.py
import numpy as np, joblib, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

MODELS_DIR = Path('../models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(MODELS_DIR/'X.npy')
y = np.load(MODELS_DIR/'y_cls.npy')
le = joblib.load(MODELS_DIR/'labelencoder_crop.pkl')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# XGBoost classifier
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {"objective":"multi:softprob","num_class":len(le.classes_),"eval_metric":"mlogloss","verbosity":0}
bst = xgb.train(params, dtrain, num_boost_round=300)
joblib.dump(bst, MODELS_DIR/'xgb_classifier.model')
preds_xgb = np.argmax(bst.predict(dtest), axis=1)
print("XGB acc:", accuracy_score(y_test, preds_xgb))

# Random Forest classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, MODELS_DIR/'rf_classifier.pkl')
print("RF acc:", accuracy_score(y_test, rf.predict(X_test)))

# Attempt AE (Adaptive-Lemuria approx) if torch installed
USE_TORCH = True
try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    USE_TORCH = False
    print("PyTorch unavailable. Skipping AE-based model.", e)

if USE_TORCH:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    class SimpleAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(input_dim,256), nn.ReLU(),
                                     nn.Linear(256,128), nn.ReLU(),
                                     nn.Linear(128,64), nn.ReLU())
            self.dec = nn.Sequential(nn.Linear(64,128), nn.ReLU(),
                                     nn.Linear(128,256), nn.ReLU(),
                                     nn.Linear(256,input_dim))
        def encode(self,x): return self.enc(x)
        def forward(self,x): return self.dec(self.enc(x))
    ae = SimpleAE().to(device)
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    Xtensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(Xtensor, Xtensor), batch_size=64, shuffle=True)
    for epoch in range(20):
        ae.train()
        for xb, yb in loader:
            opt.zero_grad()
            out = ae(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    # classifier on top
    class AEClassifier(nn.Module):
        def __init__(self, encoder, encoded_dim, n_cls):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(nn.Linear(encoded_dim,64), nn.ReLU(), nn.Linear(64,n_cls))
        def forward(self,x): return self.head(self.encoder(x))
    encoder = ae.enc
    clf = AEClassifier(encoder, 64, len(le.classes_)).to(device)
    optc = optim.Adam(clf.parameters(), lr=1e-3)
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    Ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    for epoch in range(30):
        clf.train()
        optc.zero_grad()
        logits = clf(Xtr)
        loss = nn.CrossEntropyLoss()(logits, Ytr)
        loss.backward()
        optc.step()
    # eval
    clf.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        out = clf(Xt)
        preds_ae = out.softmax(1).argmax(1).cpu().numpy()
    print("AE-based acc:", accuracy_score(y_test, preds_ae))
    torch.save(ae.state_dict(), MODELS_DIR/'ae_encoder.pt')
    torch.save(clf.state_dict(), MODELS_DIR/'ae_classifier.pt')
else:
    print("AE classifier skipped.")
print("Saved classifiers to", MODELS_DIR)
