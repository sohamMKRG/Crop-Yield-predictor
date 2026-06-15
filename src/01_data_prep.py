# 01_data_prep.py
import pandas as pd
import numpy as np
from pathlib import Path

# Correct local path
DATA_IN = Path('../data/final_yield_complete.csv')
OUT = Path('../data/cleaned_dataset.csv')

df = pd.read_csv(DATA_IN)

df.columns = [c.strip() for c in df.columns]
df['Crop'] = df['Crop'].astype(str).str.strip().str.title()
df['state'] = df['state'].astype(str).str.strip().str.title()
df['soil_type'] = df['soil_type'].astype(str).str.strip().str.title()

df = df.drop_duplicates()

num_cols = [
    'rainfall_mm','humidity','input_cost','market_price',
    'Nitrogen','Phosphorus','Potassium','Temperature',
    'pH_Value','yield_kg_per_ha_final'
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df = df[df['pH_Value'].between(3,10, inclusive='both')]

df = df.dropna(subset=['rainfall_mm','Nitrogen','Phosphorus','Potassium',
                       'Temperature','yield_kg_per_ha_final'])

OUT.parent.mkdir(exist_ok=True)
df.to_csv(OUT, index=False)

print("Saved cleaned dataset to", OUT)
print("Rows:", len(df))
