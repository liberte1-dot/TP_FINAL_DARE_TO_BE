# =============================================================
# train_and_save.py — Sauvegarde UNIQUEMENT l'instance du modèle
# Aucun pipeline, aucun scaler séparé
# =============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (roc_curve, f1_score,
                                     classification_report,
                                     roc_auc_score)
from imblearn.over_sampling  import SMOTE


# ── ÉTAPE 1 : Chargement et nettoyage ─────────────────────────
df = pd.read_csv('framingham.csv')
df = df.drop(columns=['diaBP', 'currentSmoker'], errors='ignore')

for col in ['totChol', 'BMI', 'heartRate', 'glucose', 'cigsPerDay', 'sysBP']:
    df[col] = df[col].fillna(df[col].median())
for col in ['education', 'BPMeds']:
    df[col] = df[col].fillna(df[col].mode()[0])


# ── ÉTAPE 2 : Prétraitement complet sur TOUT le dataset ───────
# Sans pipeline, on applique toutes les transformations
# directement sur le DataFrame avant le split

# Transformation log1p
df['glucose_log']    = np.log1p(df['glucose'])
df['cigsPerDay_log'] = np.log1p(df['cigsPerDay'])
df = df.drop(columns=['glucose', 'cigsPerDay'])

# Winsorisation
for col in ['sysBP', 'totChol', 'BMI']:
    lo = np.percentile(df[col], 1)
    hi = np.percentile(df[col], 99)
    df[col] = np.clip(df[col], lo, hi)

# Split
X = df.drop(columns=['TenYearCHD']).values
y = df['TenYearCHD'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# StandardScaler
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


# ── ÉTAPE 3 : Entraînement ────────────────────────────────────
model = LogisticRegression(
    class_weight='balanced',
    C=0.1,
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)


# ── ÉTAPE 4 : Optimisation du seuil ───────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

_, _, thresholds = roc_curve(y_test, y_proba)
f1_scores = [
    f1_score(y_test, (y_proba >= t).astype(int), zero_division=0)
    for t in thresholds
]
best_threshold = float(thresholds[int(np.argmax(f1_scores))])

print(classification_report(
    y_test, (y_proba >= best_threshold).astype(int),
    target_names=['Sain', 'Malade']
))
print(f"AUC-ROC : {auc:.4f}")
print(f"Seuil optimal : {best_threshold:.4f}")


# ── ÉTAPE 5 : Sauvegarde — UNIQUEMENT l'instance du modèle ────
#
# ⚠️  IMPORTANT : si tu ne sauvegardes que le modèle,
#     le prétraitement (log1p, winsorisation, StandardScaler)
#     DOIT être refait manuellement dans Streamlit
#     avec EXACTEMENT les mêmes paramètres.
#
#     Le modèle a appris sur des données transformées et
#     standardisées — il ne peut pas recevoir des valeurs brutes.

joblib.dump(model,          'model_only.pkl',     compress=3)
joblib.dump(best_threshold, 'best_threshold.pkl', compress=3)

print("\n✅ model_only.pkl     → instance LogisticRegression")
print(f"✅ best_threshold.pkl → seuil = {best_threshold:.4f}")
print("\n⚠️  Rappel : le prétraitement doit être reproduit")
print("   manuellement dans app.py avant predict_proba()")
