"""
app.py  —  Outil d'aide a la decision · Risque coronarien a 10 ans
Framingham Heart Study | Regression Logistique + SMOTE + Seuil optimal

Lancement :  streamlit run app.py

Dependances :
    pip install streamlit scikit-learn imbalanced-learn joblib numpy pandas matplotlib
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Risque Coronarien — Framingham",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# FEUILLE DE STYLE
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    /* Palette institutionnelle */
    --bg:           #f7f6f2;
    --surface:      #ffffff;
    --rule:         #d8d4cc;
    --rule-light:   #ede9e3;

    --navy:         #162032;
    --navy-mid:     #2c3e55;
    --navy-light:   #4a6070;

    --red:          #b02a1f;
    --red-bg:       #fdf1f0;
    --red-border:   #e8a89f;

    --amber:        #7a4b00;
    --amber-bg:     #fdf8ec;
    --amber-border: #f0d08a;

    --green:        #1b5e35;
    --green-bg:     #edf6f1;
    --green-border: #9ed0b4;

    --blue:         #1a4780;
    --blue-bg:      #eef3fb;
    --blue-border:  #aac4e8;

    --text:         #1c1c1c;
    --text-mid:     #3d3d3d;
    --muted:        #7a7470;

    --sidebar-bg:   #162032;

    --serif:  'Libre Baskerville', Georgia, serif;
    --mono:   'IBM Plex Mono', 'Courier New', monospace;
    --sans:   'IBM Plex Sans', system-ui, sans-serif;
}

/* ── Base ── */
html, body, [class*="css"], .stApp {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
}
[data-testid="stSidebar"] * {
    color: #b8c4d0 !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--mono) !important;
    font-size: 0.58rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #4a6880 !important;
    border-top: 1px solid #223045 !important;
    padding-top: 1rem !important;
    margin-top: 1rem !important;
    margin-bottom: 0.55rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e2d40 !important;
    border: 1px solid #2e4060 !important;
    color: #d0dae6 !important;
    border-radius: 2px !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    background: #1e2d40 !important;
    border: 1px solid #2e4060 !important;
    color: #d0dae6 !important;
    border-radius: 2px !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #4a8fc0 !important;
}
[data-testid="stSidebar"] p {
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    color: #8a9aac !important;
}

/* ── Bouton d'action ── */
.stButton > button {
    background: var(--navy) !important;
    color: #ffffff !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    border: 2px solid transparent !important;
    border-radius: 2px !important;
    padding: 0.8rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #ffffff !important;
    color: var(--navy) !important;
    border: 2px solid var(--navy) !important;
}

/* ── Onglets ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid var(--rule) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 0.7rem 1.6rem !important;
    border-radius: 0 !important;
    background: transparent !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--navy) !important;
    border-bottom: 3px solid var(--navy) !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* ── Metriques ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 2px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    font-family: var(--mono) !important;
    font-size: 0.58rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--rule) !important;
    border-radius: 2px !important;
    background: var(--surface) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--navy-light) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--rule) !important;
    border-radius: 2px !important;
}

/* ── Titres ── */
h1, h2 {
    font-family: var(--serif) !important;
    color: var(--navy) !important;
    font-weight: 400 !important;
}

hr { border-color: var(--rule); margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES ARTEFACTS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model_path     = "model_only.pkl"
    threshold_path = "best_threshold.pkl"
    scaler_path    = "scaler.pkl"

    if os.path.exists(model_path) and os.path.exists(threshold_path):
        mdl = joblib.load(model_path)
        thr = joblib.load(threshold_path)
        scl = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        return mdl, float(thr), scl

    # Reconstruction depuis les donnees brutes si les .pkl sont absents
    st.warning("Fichiers .pkl introuvables — reconstruction du pipeline en cours...")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve
    from imblearn.over_sampling import SMOTE

    csv_candidates = [
        "data_preprocessing.csv",
        "data/data_preprocessing.csv",
        "../data/data_preprocessing.csv",
    ]
    df = None
    for p in csv_candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    if df is None:
        st.error("Impossible de trouver data_preprocessing.csv. Placez les fichiers .pkl a cote de app.py.")
        st.stop()

    # Preprocessing identique au notebook
    df['education'] = df['education'].replace({'level 1':1,'level 2':2,'level 3':3,'level 4':4})
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace({"No":0,"Yes":1})
    df["gender"] = df["gender"].replace({"male":1,"female":0})
    for col in ['ratio_pression','currentSmoker']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df['pression_arterielle_moyenne'] = (df['sysBP'] + 2*df['diaBP']) / 3
    df.drop(['sysBP','diaBP'], axis=1, inplace=True)

    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42, test_size=0.2
    )

    scl = StandardScaler()
    Xtr = pd.DataFrame(scl.fit_transform(X_train), columns=scl.get_feature_names_out())
    Xte = pd.DataFrame(scl.transform(X_test),      columns=scl.get_feature_names_out())

    Xsm, ysm = SMOTE(random_state=42, sampling_strategy=0.3/0.7).fit_resample(Xtr, y_train)

    mdl = LogisticRegression(
        C=0.05, penalty='l2', solver='saga',
        class_weight='balanced', tol=1e-4,
        max_iter=1000, random_state=42
    )
    mdl.fit(Xsm, ysm)

    yp = mdl.predict_proba(Xte)[:, 1]
    precs, recs, thrs = precision_recall_curve(y_test, yp)
    valid = np.where(recs[:-1] >= 0.75)[0]
    thr = float(thrs[valid[np.argmax(precs[valid])]])

    joblib.dump(mdl, model_path,     compress=3)
    joblib.dump(thr, threshold_path, compress=3)
    joblib.dump(scl, scaler_path,    compress=3)
    return mdl, thr, scl


model, BEST_THRESHOLD, scaler = load_artifacts()


# ══════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════
FEATURES = [
    'gender','age','education','cigsPerDay',
    'BPMeds','prevalentStroke','prevalentHyp','diabetes',
    'totChol','BMI','heartRate','glucose',
    'pression_arterielle_moyenne',
]

LABELS = {
    'gender':                     'Sexe',
    'age':                        'Age (annees)',
    'education':                  "Niveau d'education",
    'cigsPerDay':                 'Cigarettes par jour',
    'BPMeds':                     'Medicaments tension',
    'prevalentStroke':            'Antecedent AVC',
    'prevalentHyp':               'Hypertension',
    'diabetes':                   'Diabete',
    'totChol':                    'Cholesterol total (mg/dL)',
    'BMI':                        'IMC (kg/m2)',
    'heartRate':                  'Frequence cardiaque (bpm)',
    'glucose':                    'Glycemie a jeun (mg/dL)',
    'pression_arterielle_moyenne': 'Pression arterielle moyenne (mmHg)',
}

# Niveau de risque : (label, couleur_fg, couleur_bg, couleur_bordure)
RISK = {
    "FAIBLE":  ("#1b5e35", "#edf6f1", "#9ed0b4"),
    "MODERE":  ("#7a4b00", "#fdf8ec", "#f0d08a"),
    "ELEVE":   ("#b02a1f", "#fdf1f0", "#e8a89f"),
}

def risk_tier(p: float):
    if p < 0.30: return "FAIBLE"
    if p < 0.55: return "MODERE"
    return "ELEVE"

def preprocess(vals: dict) -> np.ndarray:
    row = pd.DataFrame([vals], columns=FEATURES)
    return scaler.transform(row) if scaler is not None else row.values

def predict(vals: dict):
    p = model.predict_proba(preprocess(vals))[0, 1]
    return float(p), int(p >= BEST_THRESHOLD)


# ══════════════════════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════════════════════

def fig_risk_report(proba: float, patient: dict) -> plt.Figure:
    """
    Figure principale : barre de risque + jauge circulaire + tableau de bord.
    Orientee decideurs : lisible en un coup d'oeil.
    """
    tier = risk_tier(proba)
    fg, bg_c, border = RISK[tier]

    fig = plt.figure(figsize=(12, 4.2), facecolor='#ffffff')
    gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 2.8, 1.4],
                            wspace=0.06, left=0.02, right=0.98,
                            top=0.88, bottom=0.18)

    # ── Panneau 1 : Jauge circulaire ──────────────────────────────
    ax_g = fig.add_subplot(gs[0])
    ax_g.set_facecolor('#ffffff')
    ax_g.set_aspect('equal')

    theta = np.linspace(np.pi, 0, 400)
    r_o, r_i = 1.0, 0.60

    # Arc de fond segmente
    seg_colors = [
        ("#9ed0b4", 0,   120),   # vert   0-30%
        ("#f0d08a", 120, 220),   # ambre 30-55%
        ("#e8a89f", 220, 400),   # rouge 55-100%
    ]
    for col, t0, t1 in seg_colors:
        th = theta[t0:t1+1]
        x = np.concatenate([r_i*np.cos(th), r_o*np.cos(th[::-1])])
        y = np.concatenate([r_i*np.sin(th), r_o*np.sin(th[::-1])])
        ax_g.fill(x, y, color=col, zorder=1, alpha=0.6)

    # Arc rempli selon proba
    cut = int(proba * 400)
    th  = theta[:cut]
    if len(th) > 1:
        x = np.concatenate([r_i*np.cos(th), r_o*np.cos(th[::-1])])
        y = np.concatenate([r_i*np.sin(th), r_o*np.sin(th[::-1])])
        ax_g.fill(x, y, color=fg, zorder=2, alpha=0.82)

    # Aiguille
    ang = np.pi * (1 - proba)
    ax_g.plot([0, 0.75*np.cos(ang)], [0, 0.75*np.sin(ang)],
              color=fg, linewidth=3.0, zorder=5, solid_capstyle='round')
    ax_g.add_patch(plt.Circle((0, 0), 0.09, color=fg, zorder=6))
    ax_g.add_patch(plt.Circle((0, 0), 0.05, color='white', zorder=7))

    # Texte central
    ax_g.text(0, 0.26, f"{proba*100:.1f}%", ha='center', va='center',
              fontsize=19, fontweight='bold', color=fg, fontfamily='monospace')
    ax_g.text(0, -0.08, tier, ha='center', va='center',
              fontsize=8, fontweight='bold', color=fg, fontfamily='monospace')

    # Labels de zone
    for ang_lbl, label in [(0.93*np.pi, "0%"), (0.5*np.pi, "50%"), (0.07*np.pi, "100%")]:
        ax_g.text(1.22*np.cos(ang_lbl), 1.22*np.sin(ang_lbl),
                  label, ha='center', va='center',
                  fontsize=6.5, color='#9a9490', fontfamily='monospace')

    ax_g.set_xlim(-1.4, 1.4)
    ax_g.set_ylim(-0.5, 1.4)
    ax_g.axis('off')
    ax_g.set_title("Probabilite CHD", fontsize=7.5, color='#7a7470',
                   fontfamily='monospace', pad=4)

    # ── Panneau 2 : Barre de risque horizontale ───────────────────
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_facecolor('#fafaf8')

    zones = [
        (0.00, 0.30, "#c8e8d4", "#1b5e35", "RISQUE FAIBLE\n< 30 %"),
        (0.30, 0.55, "#fde9b2", "#7a4b00", "RISQUE MODERE\n30 – 55 %"),
        (0.55, 1.00, "#f8c8c2", "#b02a1f", "RISQUE ELEVE\n> 55 %"),
    ]
    for x0, x1, zfill, zcol, ztxt in zones:
        ax_b.barh(0, x1-x0, left=x0, height=0.55, color=zfill, alpha=0.65, zorder=1)
        ax_b.text((x0+x1)/2, 0.42, ztxt, ha='center', va='bottom',
                  fontsize=6.8, color=zcol, fontfamily='monospace',
                  fontweight='bold', linespacing=1.4)

    # Remplissage actif
    fill_c = "#1b5e35" if proba < 0.30 else ("#c08000" if proba < 0.55 else "#b02a1f")
    ax_b.barh(0, proba, height=0.55, color=fill_c, alpha=0.78, zorder=2)

    # Marqueur de position
    ax_b.plot([proba, proba], [-0.32, 0.32], color=fg, linewidth=3,
              zorder=6, solid_capstyle='round')

    # Lignes de seuils
    for xv, xlab in [(0.30, "Seuil 30 %"), (0.55, "Seuil 55 %"), (BEST_THRESHOLD, f"Seuil modele\n{BEST_THRESHOLD:.2f}")]:
        lw  = 2.0 if xv == BEST_THRESHOLD else 1.0
        ls  = "-" if xv == BEST_THRESHOLD else ":"
        col = "#1a3060" if xv == BEST_THRESHOLD else "#9a9490"
        ax_b.axvline(xv, color=col, linewidth=lw, linestyle=ls,
                     ymin=0.04, ymax=0.96, zorder=4)
        ax_b.text(xv, -0.48, xlab, ha='center', fontsize=6.2,
                  color=col, fontfamily='monospace', linespacing=1.3)

    # Valeur
    off = 0.055 if proba < 0.85 else -0.16
    ax_b.text(proba + off, 0.09, f"{proba*100:.1f} %",
              ha='center', va='bottom', fontsize=14, fontweight='bold',
              color=fg, fontfamily='monospace', zorder=7)

    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(-0.72, 0.82)
    ax_b.set_xticks(np.arange(0, 1.01, 0.10))
    ax_b.set_xticklabels([f"{int(x*100)}%" for x in np.arange(0, 1.01, 0.10)],
                          fontsize=7, color="#8a8278", fontfamily='monospace')
    ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_visible(False)
    ax_b.set_title("Positionnement sur l'echelle de risque", fontsize=7.5,
                   color='#7a7470', fontfamily='monospace', pad=6)

    # ── Panneau 3 : Tableau de bord patient ───────────────────────
    ax_t = fig.add_subplot(gs[2])
    ax_t.set_facecolor('#ffffff')
    ax_t.axis('off')

    rows = [
        ("Age",          f"{int(patient['age'])} ans"),
        ("Sexe",         "Homme" if patient['gender']==1 else "Femme"),
        ("Hypertension", "Oui"   if patient['prevalentHyp']==1 else "Non"),
        ("Diabete",      "Oui"   if patient['diabetes']==1 else "Non"),
        ("Tabac",        f"{int(patient['cigsPerDay'])} cig/j"),
        ("Cholesterol",  f"{int(patient['totChol'])} mg/dL"),
        ("Glycemie",     f"{int(patient['glucose'])} mg/dL"),
        ("IMC",          f"{patient['BMI']:.1f}"),
        ("PAM",          f"{patient['pression_arterielle_moyenne']:.0f} mmHg"),
    ]
    y_pos = 0.96
    ax_t.text(0.5, 1.02, "Profil patient", ha='center', va='top',
              fontsize=7.5, color='#7a7470', fontfamily='monospace',
              transform=ax_t.transAxes)

    for i, (k, v) in enumerate(rows):
        bg_row = '#f7f6f2' if i % 2 == 0 else '#ffffff'
        ax_t.add_patch(plt.Rectangle((0, y_pos-0.105), 1, 0.10,
                                      facecolor=bg_row, edgecolor='none',
                                      transform=ax_t.transAxes, zorder=0))
        ax_t.text(0.05, y_pos-0.05, k, ha='left', va='center',
                  fontsize=7.2, color='#7a7470', fontfamily='monospace',
                  transform=ax_t.transAxes)
        ax_t.text(0.98, y_pos-0.05, v, ha='right', va='center',
                  fontsize=7.8, color='#1c1c1c', fontweight='bold',
                  fontfamily='monospace', transform=ax_t.transAxes)
        y_pos -= 0.106

    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)

    return fig


def fig_coefs() -> plt.Figure:
    df = pd.DataFrame({
        'var': [LABELS.get(f, f) for f in FEATURES],
        'coef': model.coef_[0],
    }).sort_values('coef')

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafaf8')

    colors = ['#1a6b8a' if c < 0 else '#b02a1f' for c in df['coef']]
    ax.barh(df['var'], df['coef'], color=colors, height=0.60, edgecolor='none')
    ax.axvline(0, color='#9a9490', linewidth=1.2)
    ax.set_xlabel('Coefficient (log-odds)', fontsize=8, color='#7a7470', fontfamily='monospace')
    ax.tick_params(labelsize=8.5, colors='#3d3d3d', length=0)
    for sp in ['top','right','left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#d8d4cc')
    ax.grid(axis='x', color='#ede9e3', linewidth=0.6, linestyle='--')

    handles = [
        mpatches.Patch(color='#b02a1f', label='Augmente le risque'),
        mpatches.Patch(color='#1a6b8a', label='Reduit le risque'),
    ]
    ax.legend(handles=handles, framealpha=0, fontsize=8.5, labelcolor='#3d3d3d')
    fig.tight_layout(pad=0.6)
    return fig


def fig_odds() -> plt.Figure:
    df = pd.DataFrame({
        'var': [LABELS.get(f, f) for f in FEATURES],
        'OR':  np.exp(model.coef_[0]),
    }).sort_values('OR')

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafaf8')

    colors = ['#b02a1f' if v > 1 else '#1a6b8a' for v in df['OR']]
    ax.barh(df['var'], df['OR'], color=colors, height=0.60, edgecolor='none')
    ax.axvline(1.0, color='#162032', linewidth=1.6)
    ax.text(1.015, 0, 'Neutre', fontsize=7, color='#7a7470',
            va='center', fontfamily='monospace')
    ax.set_xlabel('Odds Ratio', fontsize=8, color='#7a7470', fontfamily='monospace')
    ax.tick_params(labelsize=8.5, colors='#3d3d3d', length=0)
    for sp in ['top','right','left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#d8d4cc')
    ax.grid(axis='x', color='#ede9e3', linewidth=0.6, linestyle='--')

    handles = [
        mpatches.Patch(color='#b02a1f', label='Facteur de risque (OR > 1)'),
        mpatches.Patch(color='#1a6b8a', label='Facteur protecteur (OR < 1)'),
    ]
    ax.legend(handles=handles, framealpha=0, fontsize=8.5, labelcolor='#3d3d3d')
    fig.tight_layout(pad=0.6)
    return fig


# ══════════════════════════════════════════════════════════════
# EN-TETE INSTITUTIONNEL
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:#ffffff;border-bottom:3px solid #162032;
            padding:1.6rem 2.4rem 1.2rem;margin-bottom:1.8rem">
    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
              letter-spacing:0.2em;text-transform:uppercase;color:#7a7470;margin:0 0 0.4rem">
        Outil d'aide a la decision  /  Cardiologie Preventive
    </p>
    <h1 style="font-family:'Libre Baskerville',Georgia,serif;font-weight:400;
               font-size:1.95rem;color:#162032;margin:0 0 0.35rem;letter-spacing:-0.01em;
               line-height:1.15">
        Evaluation du Risque Coronarien a 10 ans
    </h1>
    <p style="font-family:'IBM Plex Sans',sans-serif;font-size:0.83rem;
              color:#6b6560;margin:0;line-height:1.6">
        Framingham Heart Study&ensp;&middot;&ensp;Regression Logistique (SMOTE + class_weight = balanced)
        &ensp;&middot;&ensp;
        Seuil de decision&nbsp;:&ensp;
        <span style="font-family:'IBM Plex Mono',monospace;color:#162032;font-weight:600">
            {BEST_THRESHOLD:.3f}
        </span>
        &ensp;(recall &ge; 75 %)
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR — FORMULAIRE PATIENT
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 0.2rem 0.6rem">
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                  letter-spacing:0.2em;text-transform:uppercase;color:#3a5870;margin:0">
            Framingham CHD
        </p>
        <p style="font-family:'Libre Baskerville',Georgia,serif;font-size:1.05rem;
                  font-weight:400;color:#c8d8e8;margin:0.25rem 0 0">
            Profil du Patient
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Demographie")
    gender    = st.selectbox("Sexe",               [1, 0], format_func=lambda x: "Homme" if x==1 else "Femme")
    age       = st.slider("Age (annees)",           20, 90, 52)
    education = st.selectbox("Niveau d'education", [1,2,3,4], format_func=lambda x: f"Niveau {x}")
    bmi       = st.number_input("IMC (kg/m2)",      10.0, 60.0, 26.5, step=0.1, format="%.1f")

    st.markdown("### Antecedents")
    prev_stroke = st.selectbox("Antecedent AVC",         [0,1], format_func=lambda x: "Oui" if x else "Non")
    prev_hyp    = st.selectbox("Hypertension",           [0,1], format_func=lambda x: "Oui" if x else "Non")
    diabetes    = st.selectbox("Diabete",                [0,1], format_func=lambda x: "Oui" if x else "Non")
    bp_meds     = st.selectbox("Medicaments tension",    [0,1], format_func=lambda x: "Oui" if x else "Non")

    st.markdown("### Mesures cliniques")
    tot_chol   = st.number_input("Cholesterol total (mg/dL)", 100, 700, 230, step=1)
    heart_rate = st.number_input("Frequence cardiaque (bpm)",  30, 200,  75, step=1)
    glucose    = st.number_input("Glycemie a jeun (mg/dL)",    40, 500,  80, step=1)
    cigs       = st.slider("Cigarettes par jour",              0,  70,   0)
    pam        = st.number_input("Pression art. moy. (mmHg)", 50.0, 200.0, 93.0, step=0.5, format="%.1f")

    st.markdown("""
    <div style="border-top:1px solid #223045;margin:1.2rem 0 0.8rem;padding-top:0.8rem">
        <p style="color:#3a5870;font-family:'IBM Plex Mono',monospace;font-size:0.6rem;line-height:1.9">
            PAM = (SYS + 2 x DIA) / 3<br>
            Normalisation : StandardScaler<br>
            Reequilibrage : SMOTE 30/70<br>
            Seuil calibre : precision-recall
        </p>
    </div>
    """, unsafe_allow_html=True)

    run = st.button("Analyser le risque")


# ══════════════════════════════════════════════════════════════
# ONGLETS PRINCIPAUX
# ══════════════════════════════════════════════════════════════
tab_pred, tab_model, tab_about = st.tabs([
    "  Prediction  ",
    "  Modele et Interpretation  ",
    "  A propos  ",
])


# ──────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ──────────────────────────────────────────────────────────────
with tab_pred:

    if not run:
        # Etat initial : guide de lecture des niveaux de risque
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #d8d4cc;border-left:4px solid #162032;
                    border-radius:2px;padding:2rem 2.4rem;margin-top:0.4rem">
            <p style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                      letter-spacing:0.16em;text-transform:uppercase;color:#7a7470;margin:0 0 0.6rem">
                Mode d'emploi
            </p>
            <p style="font-family:'Libre Baskerville',Georgia,serif;font-size:1.05rem;
                      font-weight:400;color:#162032;line-height:1.7;margin:0 0 1.6rem">
                Renseignez le profil clinique du patient dans le panneau de gauche,
                puis cliquez sur <strong>Analyser le risque</strong>.
                Un rapport structure apparaitra ici avec le niveau de risque,
                la probabilite estimee et la recommandation associee.
            </p>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3, gap="medium")
        levels = [
            (col_a, "#edf6f1", "#9ed0b4", "#1b5e35", "Risque Faible",  "P < 30 %",
             "Profil favorable. Pas de signe d'alerte identifie. Suivi standard recommande."),
            (col_b, "#fdf8ec", "#f0d08a", "#7a4b00", "Risque Modere",  "30 % — 55 %",
             "Facteurs de risque presents. Bilan approfondi et surveillance renforcee conseillee."),
            (col_c, "#fdf1f0", "#e8a89f", "#b02a1f", "Risque Eleve",   "P > 55 %",
             "Probabilite elevee de CHD. Prise en charge cardiologique prioritaire indiquee."),
        ]
        for col, bg, brd, fg, title, rng, desc in levels:
            with col:
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {brd};border-radius:2px;
                            padding:1.2rem 1.5rem;height:100%">
                    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                              letter-spacing:0.12em;text-transform:uppercase;
                              color:{fg};margin:0 0 0.2rem">{title}</p>
                    <p style="font-family:'IBM Plex Mono',monospace;font-size:1.15rem;
                              font-weight:600;color:{fg};margin:0 0 0.5rem">{rng}</p>
                    <p style="font-size:0.82rem;color:#3d3d3d;margin:0;line-height:1.55">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        patient = {
            'gender': gender, 'age': age, 'education': education,
            'cigsPerDay': cigs, 'BPMeds': bp_meds,
            'prevalentStroke': prev_stroke, 'prevalentHyp': prev_hyp,
            'diabetes': diabetes, 'totChol': tot_chol, 'BMI': bmi,
            'heartRate': heart_rate, 'glucose': glucose,
            'pression_arterielle_moyenne': pam,
        }

        proba, label = predict(patient)
        tier = risk_tier(proba)
        fg, bg_c, brd_c = RISK[tier]

        # ── Bandeau verdict ──────────────────────────────────────
        verdict_titre = "RISQUE ELEVE — Consultation cardiologique recommandee" \
                        if label == 1 else \
                        "RISQUE FAIBLE — Profil cardiovasculaire favorable"
        verdict_detail = (
            "Le modele identifie une probabilite significative de maladie coronarienne "
            "dans les dix prochaines annees. Une prise en charge preventive s'impose."
            if label == 1 else
            "Aucun signal clinique majeur detecte. Un suivi de routine reste conseille "
            "compte tenu des facteurs de mode de vie."
        )

        st.markdown(f"""
        <div style="background:{bg_c};border-left:5px solid {fg};border-radius:2px;
                    padding:1.4rem 2rem;margin-bottom:1.4rem">
            <p style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                      letter-spacing:0.16em;text-transform:uppercase;
                      color:{fg};opacity:0.8;margin:0 0 0.4rem">
                Verdict du modele
            </p>
            <p style="font-family:'Libre Baskerville',Georgia,serif;font-size:1.2rem;
                      font-weight:700;color:{fg};margin:0 0 0.3rem;line-height:1.3">
                {verdict_titre}
            </p>
            <p style="font-family:'IBM Plex Sans',sans-serif;font-size:0.86rem;
                      color:{fg};opacity:0.85;margin:0;line-height:1.6">
                {verdict_detail}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Figure principale ────────────────────────────────────
        fig_main = fig_risk_report(proba, patient)
        st.pyplot(fig_main, use_container_width=True)
        plt.close()

        # ── Metriques clés ───────────────────────────────────────
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4, gap="small")
        m1.metric("Probabilite estimee",  f"{proba*100:.1f} %")
        m2.metric("Seuil de decision",    f"{BEST_THRESHOLD*100:.1f} %")
        m3.metric("Classification",       "CHD Positif" if label==1 else "CHD Negatif")
        m4.metric("Niveau de risque",     tier)

        # ── Interpretation et recommandation ────────────────────
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        col_interp, col_reco = st.columns(2, gap="large")

        with col_interp:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #d8d4cc;border-radius:2px;
                        padding:1.3rem 1.6rem">
                <p style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                          letter-spacing:0.14em;text-transform:uppercase;
                          color:#7a7470;margin:0 0 0.7rem">Interpretation statistique</p>
                <p style="font-size:0.87rem;color:#3d3d3d;line-height:1.75;margin:0">
                    Le modele attribue a ce profil une probabilite de
                    <strong style="color:{fg};font-family:'IBM Plex Mono',monospace;font-size:1rem">
                        {proba*100:.1f} %
                    </strong>
                    de developper une coronaropathie dans les dix ans.
                    Le seuil de classification est fixe a
                    <strong>{BEST_THRESHOLD*100:.1f} %</strong>,
                    optimise pour capter au minimum 75 % des cas reels
                    (recall &ge; 0.75 sur la courbe Precision-Rappel).
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_reco:
            if label == 1:
                reco_txt = (
                    "Un bilan cardiologique complet est recommande en priorite : "
                    "ECG de repos, echocardiographie, bilan lipidique detaille "
                    "et evaluation du mode de vie (tabac, alimentation, activite physique). "
                    "Une consultation specialisee doit etre planifiee dans un delai court."
                )
            else:
                reco_txt = (
                    "Le profil presente un risque faible. Les mesures de prevention primaire "
                    "standard sont suffisantes : alimentation equilibree, activite physique "
                    "reguliere, arret du tabac si applicable. "
                    "Un bilan de controle dans 2 ans est recommande."
                )
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #d8d4cc;border-left:4px solid {fg};
                        border-radius:2px;padding:1.3rem 1.6rem">
                <p style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                          letter-spacing:0.14em;text-transform:uppercase;
                          color:#7a7470;margin:0 0 0.7rem">Recommandation clinique</p>
                <p style="font-size:0.87rem;color:#3d3d3d;line-height:1.75;margin:0">
                    {reco_txt}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Detail profil ────────────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        with st.expander("Detail complet du profil saisi"):
            df_pat = pd.DataFrame([patient]).T.rename(columns={0: "Valeur"})
            df_pat.index = [LABELS.get(k, k) for k in df_pat.index]
            df_pat["Valeur"] = df_pat["Valeur"].apply(
                lambda x: f"{x:.2f}" if isinstance(x, float) else str(int(x))
            )
            st.dataframe(df_pat, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 2 — MODELE ET INTERPRETATION
# ──────────────────────────────────────────────────────────────
with tab_model:

    col_coef, col_or = st.columns(2, gap="large")

    with col_coef:
        st.markdown("""
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                  letter-spacing:0.14em;text-transform:uppercase;color:#7a7470;margin:0 0 0.3rem">
            Coefficients du modele</p>
        <p style="font-size:0.8rem;color:#6b6560;margin:0 0 0.8rem;line-height:1.55">
            Barres rouges : facteurs augmentant le risque de CHD.
            Barres bleues : facteurs associes a un risque reduit.</p>
        """, unsafe_allow_html=True)
        st.pyplot(fig_coefs(), use_container_width=True)
        plt.close()

    with col_or:
        st.markdown("""
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                  letter-spacing:0.14em;text-transform:uppercase;color:#7a7470;margin:0 0 0.3rem">
            Odds Ratios</p>
        <p style="font-size:0.8rem;color:#6b6560;margin:0 0 0.8rem;line-height:1.55">
            OR > 1 : facteur de risque independant.
            OR < 1 : facteur protecteur.
            La ligne verticale represente la neutralite (OR = 1).</p>
        """, unsafe_allow_html=True)
        st.pyplot(fig_odds(), use_container_width=True)
        plt.close()

    st.markdown("---")

    # Tableau complet
    st.markdown("""
    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
              letter-spacing:0.14em;text-transform:uppercase;color:#7a7470;margin:0 0 0.6rem">
        Tableau — Coefficients, Odds Ratios et interpretation</p>
    """, unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        'Variable':    [LABELS.get(f, f) for f in FEATURES],
        'Coefficient': model.coef_[0].round(4),
    })
    coef_df['Odds Ratio']          = np.exp(coef_df['Coefficient']).round(4)
    coef_df['Effet sur le risque'] = coef_df['Odds Ratio'].apply(
        lambda x: f"+ {(x-1)*100:.1f} %" if x > 1 else f"- {(1-x)*100:.1f} %"
    )
    coef_df['Direction'] = coef_df['Odds Ratio'].apply(
        lambda x: "Augmente" if x > 1 else "Reduit"
    )
    coef_df = coef_df.sort_values('Odds Ratio', ascending=False).reset_index(drop=True)
    st.dataframe(coef_df, use_container_width=True, height=370)

    st.markdown("---")

    col_params, col_pipeline = st.columns(2, gap="large")

    with col_params:
        st.markdown("""
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                  letter-spacing:0.14em;text-transform:uppercase;color:#7a7470;margin:0 0 0.6rem">
            Hyperparametres retenus</p>
        """, unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(model.get_params().items(), columns=["Parametre","Valeur"]),
            use_container_width=True, height=280
        )

    with col_pipeline:
        st.markdown("""
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                  letter-spacing:0.14em;text-transform:uppercase;color:#7a7470;margin:0 0 0.6rem">
            Pipeline de traitement des donnees</p>
        <div style="background:#f7f6f2;border:1px solid #d8d4cc;border-radius:2px;
                    padding:1.2rem 1.6rem;font-family:'IBM Plex Mono',monospace;
                    font-size:0.79rem;line-height:2.1;color:#3d3d3d">
            1. Encodage categoriel (binaire / ordinal)<br>
            2. PAM = (sysBP + 2 x diaBP) / 3<br>
            3. Suppression variables redondantes (analyse VIF)<br>
            4. StandardScaler (mu = 0, sigma = 1)<br>
            5. SMOTE — reequilibrage ratio 30 / 70<br>
            6. LogisticRegression (class_weight = balanced)<br>
            7. RandomizedSearchCV — optimisation precision<br>
            8. Seuil optimal via courbe Precision-Rappel
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#eef3fb;border:1px solid #aac4e8;border-radius:2px;
                    padding:1rem 1.6rem;margin-top:0.7rem">
            <p style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                      letter-spacing:0.14em;text-transform:uppercase;
                      color:#1a4780;margin:0 0 0.2rem">Seuil de decision retenu</p>
            <p style="font-family:'IBM Plex Mono',monospace;font-size:1.8rem;
                      font-weight:600;color:#162032;margin:0">{BEST_THRESHOLD:.4f}</p>
            <p style="font-size:0.79rem;color:#4a5568;margin:0.25rem 0 0;line-height:1.5">
                Maximise la precision sous contrainte recall &ge; 0.75<br>
                Calcule sur l'ensemble de test (stratifie, 20 %)
            </p>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 3 — A PROPOS
# ──────────────────────────────────────────────────────────────
with tab_about:
    col_g, col_d = st.columns(2, gap="large")

    with col_g:
        for bg, border_l, fg, title, content in [
            ("#ffffff", "#162032", "#162032",
             "L'etude Framingham",
             "Lancee en 1948 a Framingham (Massachusetts), cette etude de cohorte longitudinale "
             "a suivi des milliers de residents sur plusieurs decennies pour identifier les facteurs "
             "de risque des maladies cardiovasculaires. Le jeu de donnees utilise contient des "
             "variables cliniques, biologiques et comportementales ; la variable cible est la "
             "survenue d'une <strong>maladie coronarienne dans les 10 annees suivantes</strong>."),
            ("#fdf1f0", "#b02a1f", "#b02a1f",
             "Avertissement medical",
             "Cet outil est developpe <strong>exclusivement a des fins pedagogiques et de "
             "recherche</strong>. Il ne constitue en aucun cas un diagnostic ni un avis "
             "clinique. Toute decision relative a la sante d'un patient doit etre prise "
             "par un professionnel de sante qualifie sur la base d'un examen clinique complet."),
        ]:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid #d8d4cc;border-left:4px solid {fg};
                        border-radius:2px;padding:1.4rem 1.8rem;margin-bottom:1rem">
                <p style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                          letter-spacing:0.14em;text-transform:uppercase;
                          color:{fg};opacity:0.75;margin:0 0 0.55rem">{title}</p>
                <p style="font-size:0.87rem;color:#3d3d3d;line-height:1.72;margin:0">{content}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_d:
        st.markdown("""
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                  letter-spacing:0.14em;text-transform:uppercase;color:#7a7470;margin:0 0 0.6rem">
            Limites connues du modele</p>
        <div style="background:#ffffff;border:1px solid #d8d4cc;border-radius:2px">
        """, unsafe_allow_html=True)

        limits = [
            ("Multicolinearite residuelle",
             "Des VIF eleves persistent entre variables hemodynamiques. "
             "Certains coefficients restent partiellement instables."),
            ("Desequilibre de classes",
             "Meme apres SMOTE, le rappel reste difficile a stabiliser selon le seuil. "
             "Des cas positifs peuvent etre manques."),
            ("Regularisation L2",
             "La penalisation Ridge compresse les coefficients, biaisant les Odds Ratios "
             "vers la neutralite (OR proche de 1)."),
            ("Donnees manquantes",
             "L'imputation n'est pas prise en charge dans ce pipeline. "
             "Des valeurs extremes peuvent affecter la prediction."),
            ("Validite externe limitee",
             "Le modele a ete entraine sur une cohorte americaine homogene. "
             "Ses performances peuvent se degrader sur d'autres populations."),
        ]

        for i, (titre, desc) in enumerate(limits):
            border_b = "border-bottom:1px solid #ede9e3;" if i < len(limits)-1 else ""
            st.markdown(f"""
            <div style="{border_b}padding:0.9rem 1.5rem">
                <p style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                          font-weight:600;color:#162032;margin:0 0 0.2rem">{titre}</p>
                <p style="font-size:0.82rem;color:#6b6560;margin:0;line-height:1.55">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
