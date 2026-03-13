# =============================================================
# app.py — Streamlit avec UNIQUEMENT l'instance du modèle
#
# On charge seulement model_only.pkl et best_threshold.pkl.
# Tout le prétraitement est codé en dur dans l'application.
# =============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(
    page_title="Framingham · Risque Cardiaque",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size:0.82em !important; }
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg,#3b82f6,#1d4ed8) !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    padding:0.7em !important;
}
.card-danger { background:linear-gradient(135deg,#7f1d1d,#dc2626);
    border-radius:16px; padding:24px 28px; color:white; text-align:center;
    box-shadow:0 8px 32px rgba(220,38,38,.35); }
.card-safe   { background:linear-gradient(135deg,#14532d,#16a34a);
    border-radius:16px; padding:24px 28px; color:white; text-align:center;
    box-shadow:0 8px 32px rgba(22,163,74,.35); }
.card-metric { background:#1e293b; border:1px solid #334155;
    border-radius:12px; padding:16px 20px;
    text-align:center; color:white; margin-bottom:10px; }
.info-box { background:#1e293b; border:1px solid #334155;
    border-left:4px solid #f59e0b; border-radius:8px;
    padding:14px 18px; margin:6px 0; color:#e2e8f0; font-size:0.9em; }
.section-title { font-size:0.9em; font-weight:600; color:#94a3b8;
    text-transform:uppercase; letter-spacing:0.08em;
    margin:14px 0 6px 0; border-bottom:1px solid #334155; padding-bottom:5px; }
.warning-banner { background:#1c1917; border-left:4px solid #f59e0b;
    border-radius:8px; padding:12px 16px;
    color:#fbbf24; font-size:0.88em; margin-bottom:16px; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# CHARGEMENT — UNIQUEMENT le modèle et le seuil
# =============================================================

@st.cache_resource
def charger_modele():
    """
    Charge UNIQUEMENT :
      - model_only.pkl      : l'instance LogisticRegression entraînée
      - best_threshold.pkl  : le seuil de décision optimal

    Rien d'autre. Pas de scaler, pas de bornes, pas de pipeline.
    Le prétraitement est entièrement géré dans la fonction
    pretraiter() ci-dessous.
    """
    model     = joblib.load('model_only.pkl')
    threshold = joblib.load('best_threshold.pkl')
    return model, float(threshold)


# =============================================================
# PRÉTRAITEMENT CODÉ EN DUR
# =============================================================

# Paramètres de winsorisation — copiés depuis train_and_save.py
# ⚠️  Ces valeurs DOIVENT correspondre exactement à celles
#     calculées sur le dataset d'entraînement.
#     Si tu ré-entraînes le modèle, mets ces valeurs à jour.
WINSOR_BOUNDS = {
    'sysBP':   {'low': 94.0,  'high': 200.0},
    'totChol': {'low': 155.0, 'high': 360.0},
    'BMI':     {'low': 17.9,  'high': 40.2 },
}

# Moyennes et écarts-types du StandardScaler — copiés depuis
# les paramètres appris sur X_train dans train_and_save.py.
# ⚠️  Ces valeurs DOIVENT correspondre à scaler.mean_ et scaler.scale_
#     Si tu ré-entraînes, mets ces tableaux à jour.
SCALER_MEAN  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0])  # à remplacer par scaler.mean_
SCALER_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0])  # à remplacer par scaler.scale_

# Ordre des colonnes — doit correspondre à l'ordre dans X_train
COL_ORDER = [
    'male', 'age', 'education', 'BPMeds', 'prevalentStroke',
    'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'BMI',
    'heartRate', 'glucose_log', 'cigsPerDay_log'
]


def pretraiter(donnees: dict) -> np.ndarray:
    """
    Reproduit manuellement le prétraitement appliqué à l'entraînement.

    Les 4 étapes dans l'ordre obligatoire :
      1. log1p  sur glucose et cigsPerDay
      2. Winsorisation avec les bornes fixes du train
      3. Réordonnancement des colonnes selon COL_ORDER
      4. Standardisation manuelle : (x - mean) / scale

    ⚠️  Ne jamais appeler StandardScaler().fit_transform() ici.
        On standardise manuellement avec les paramètres du train.
    """
    d = dict(donnees)

    # 1. Transformation log1p
    d['glucose_log']    = float(np.log1p(d.pop('glucose')))
    d['cigsPerDay_log'] = float(np.log1p(d.pop('cigsPerDay')))

    # 2. Winsorisation avec les bornes fixes
    for col, bounds in WINSOR_BOUNDS.items():
        d[col] = float(np.clip(d[col], bounds['low'], bounds['high']))

    # 3. Réordonnancement des colonnes
    vecteur = np.array([[d[col] for col in COL_ORDER]], dtype=float)

    # 4. Standardisation manuelle
    #    Si SCALER_MEAN et SCALER_SCALE ont été mis à jour avec les
    #    vraies valeurs du train, on les utilise ici.
    #    Sinon, on retourne le vecteur brut (attention aux performances).
    if not np.all(SCALER_MEAN == 0):
        vecteur = (vecteur - SCALER_MEAN) / SCALER_SCALE

    return vecteur


# =============================================================
# PRÉDICTION
# =============================================================

def predire(model, vecteur: np.ndarray, seuil: float) -> dict:
    """
    predict_proba()[:,1] → probabilité brute de la classe 1
    On applique notre seuil ajusté (pas predict() à 0.5 par défaut)
    """
    prob = float(model.predict_proba(vecteur)[0][1])
    pred = 1 if prob >= seuil else 0
    return {
        "probabilite": prob,
        "prediction":  pred,
        "label":       "Risque élevé" if pred == 1 else "Faible risque",
        "marge":       abs(prob - seuil)
    }


# =============================================================
# VISUALISATION — Jauge
# =============================================================

def tracer_jauge(prob: float, seuil: float):
    fig, ax = plt.subplots(figsize=(5.5, 3.2),
                           subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            color='#1e293b', linewidth=22, solid_capstyle='round')

    angle      = np.pi - prob * np.pi
    theta_fill = np.linspace(np.pi, angle, 300)
    couleur    = ('#16a34a' if prob < 0.33
                  else '#f59e0b' if prob < seuil
                  else '#dc2626')
    ax.plot(np.cos(theta_fill), np.sin(theta_fill),
            color=couleur, linewidth=22, solid_capstyle='round')

    for val, lbl in [(0,'0%'),(0.25,'25%'),(0.5,'50%'),(0.75,'75%'),(1,'100%')]:
        a = np.pi - val * np.pi
        ax.text(1.18*np.cos(a), 1.18*np.sin(a), lbl,
                color='#64748b', fontsize=7, ha='center', va='center')

    a_s = np.pi - seuil * np.pi
    ax.plot([0.72*np.cos(a_s), 0.92*np.cos(a_s)],
            [0.72*np.sin(a_s), 0.92*np.sin(a_s)],
            color='white', linewidth=2.5, zorder=5)
    ax.text(0.60*np.cos(a_s), 0.60*np.sin(a_s),
            f'Seuil\n{seuil:.0%}', color='#cbd5e1',
            fontsize=7, ha='center', va='center')

    ax.text(0, -0.08, f'{prob:.1%}', ha='center', va='center',
            fontsize=28, fontweight='700', color='white')
    ax.text(0, -0.38, 'probabilité estimée',
            ha='center', fontsize=8, color='#64748b')

    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.55, 1.25); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# =============================================================
# INTERFACE PRINCIPALE
# =============================================================

st.markdown("## 🫀 Framingham Heart Study")
st.markdown("##### Déploiement avec **uniquement l'instance du modèle**")
st.markdown("""
<div class="warning-banner">
⚠️ <strong>Usage académique uniquement.</strong>
Prototype pédagogique — ne constitue pas un diagnostic médical.
</div>
""", unsafe_allow_html=True)

# Chargement
if not os.path.exists('model_only.pkl'):
    st.error("❌ `model_only.pkl` introuvable. Lance `python train_and_save.py`.")
    st.stop()

model, seuil_optimal = charger_modele()

# Avertissement si les paramètres scaler ne sont pas mis à jour
if np.all(SCALER_MEAN == 0):
    st.warning(
        "⚠️ **SCALER_MEAN et SCALER_SCALE** contiennent des valeurs par défaut (0 et 1). "
        "Mets-les à jour avec les valeurs de `scaler.mean_` et `scaler.scale_` "
        "depuis `train_and_save.py` pour des prédictions correctes."
    )


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Données du Patient")
    st.markdown("---")

    st.markdown('<div class="section-title">👤 Démographie</div>',
                unsafe_allow_html=True)
    male     = 1 if st.selectbox("Sexe", ["Femme","Homme"]) == "Homme" else 0
    sexe_lbl = "Homme" if male else "Femme"
    age      = st.slider("Âge", 30, 70, 50)
    education = st.selectbox("Niveau d'éducation", [1,2,3,4],
                    format_func=lambda x: {
                        1:"1 — Primaire", 2:"2 — Lycée",
                        3:"3 — Supérieur", 4:"4 — Diplôme"
                    }[x])
    st.markdown("---")

    st.markdown('<div class="section-title">🚬 Habitudes</div>',
                unsafe_allow_html=True)
    currentSmoker = 1 if st.selectbox("Fumeur", ["Non","Oui"]) == "Oui" else 0
    cigsPerDay    = st.slider("Cigarettes / jour", 0, 60, 10) if currentSmoker else 0
    if not currentSmoker: st.caption("0 cigarette/jour")
    st.markdown("---")

    st.markdown('<div class="section-title">🏥 Antécédents</div>',
                unsafe_allow_html=True)
    BPMeds          = int(st.checkbox("Médicaments anti-hypertenseurs"))
    prevalentStroke = int(st.checkbox("Antécédent d'AVC"))
    prevalentHyp    = int(st.checkbox("Hypertension préexistante"))
    diabetes        = int(st.checkbox("Diabète préexistant"))
    st.markdown("---")

    st.markdown('<div class="section-title">🩺 Mesures cliniques</div>',
                unsafe_allow_html=True)
    totChol   = st.slider("Cholestérol total (mg/dL)",   100, 400, 235)
    sysBP     = st.slider("Pression systolique (mmHg)",  80,  220, 130)
    diaBP     = st.slider("Pression diastolique (mmHg)", 50,  130, 82)
    BMI       = st.slider("IMC (kg/m²)",                 15.0, 55.0, 25.5, 0.1)
    heartRate = st.slider("Fréquence cardiaque (bpm)",   40,  150, 75)
    glucose   = st.slider("Glycémie (mg/dL)",            40,  400, 80)
    st.markdown("---")

    st.markdown('<div class="section-title">⚙️ Seuil</div>',
                unsafe_allow_html=True)
    seuil = st.slider("Seuil de classification",
                      min_value=0.10, max_value=0.70,
                      value=float(round(seuil_optimal, 2)),
                      step=0.01)
    if   seuil <= 0.25: st.warning("⚡ Très sensible")
    elif seuil <= 0.38: st.success(f"✅ Recommandé ({seuil_optimal:.2f})")
    elif seuil <= 0.50: st.info("⚖️ Équilibré")
    else:               st.error("🎯 Risque de manquer des malades")

    st.markdown("")
    predict_btn = st.button("🔍 Analyser le risque", use_container_width=True)


# ── RÉSULTATS ─────────────────────────────────────────────────
donnees = {
    'male': male, 'age': age, 'education': education,
    'currentSmoker': currentSmoker, 'cigsPerDay': cigsPerDay,
    'BPMeds': BPMeds, 'prevalentStroke': prevalentStroke,
    'prevalentHyp': prevalentHyp, 'diabetes': diabetes,
    'totChol': totChol, 'sysBP': sysBP, 'diaBP': diaBP,
    'BMI': BMI, 'heartRate': heartRate, 'glucose': glucose
}

if predict_btn:
    vecteur = pretraiter(donnees)
    res     = predire(model, vecteur, seuil)
    prob    = res["probabilite"]
    pred    = res["prediction"]

    # Carte résultat + Jauge
    col1, col2 = st.columns([1, 1.1])

    with col1:
        st.markdown("### Résultat")
        css   = "card-danger" if pred == 1 else "card-safe"
        icone = "🚨" if pred == 1 else "✅"
        st.markdown(f"""
        <div class="{css}">
            <div style="font-size:2.8em;margin-bottom:8px">{icone}</div>
            <div style="font-size:1.7em;font-weight:700;margin-bottom:6px">
                {res['label']}
            </div>
            <div style="font-size:1.05em;opacity:0.85">
                Probabilité : <strong>{prob:.1%}</strong>
            </div>
            <div style="font-size:0.85em;opacity:0.65;margin-top:6px">
                Seuil : {seuil:.0%} &nbsp;|&nbsp; Marge : {res['marge']:.1%}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        mc1, mc2 = st.columns(2)
        mc3, mc4 = st.columns(2)
        mc1.markdown(f'<div class="card-metric"><div style="color:#64748b;font-size:0.78em">Âge</div><div style="font-size:1.5em;font-weight:700">{age} ans</div></div>', unsafe_allow_html=True)
        mc2.markdown(f'<div class="card-metric"><div style="color:#64748b;font-size:0.78em">Glycémie</div><div style="font-size:1.5em;font-weight:700">{glucose} mg/dL</div></div>', unsafe_allow_html=True)
        mc3.markdown(f'<div class="card-metric"><div style="color:#64748b;font-size:0.78em">Systolique</div><div style="font-size:1.5em;font-weight:700">{sysBP} mmHg</div></div>', unsafe_allow_html=True)
        mc4.markdown(f'<div class="card-metric"><div style="color:#64748b;font-size:0.78em">IMC</div><div style="font-size:1.5em;font-weight:700">{BMI:.1f}</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### Jauge de risque")
        st.pyplot(tracer_jauge(prob, seuil), use_container_width=True)

    st.markdown("---")

    # Facteurs de risque
    col3, col4 = st.columns([1.1, 1])

    with col3:
        st.markdown("### 🔎 Facteurs de risque identifiés")
        facteurs = []
        if age >= 55:        facteurs.append(("🔴", f"Âge élevé ({age} ans)",               "Fort",   "#dc2626"))
        elif age >= 45:      facteurs.append(("🟠", f"Âge intermédiaire ({age} ans)",        "Modéré", "#f59e0b"))
        if sysBP >= 140:     facteurs.append(("🔴", f"Hypertension sévère ({sysBP} mmHg)",   "Fort",   "#dc2626"))
        elif sysBP >= 130:   facteurs.append(("🟠", f"Pression élevée ({sysBP} mmHg)",       "Modéré", "#f59e0b"))
        if glucose >= 126:   facteurs.append(("🔴", f"Hyperglycémie ({glucose} mg/dL)",      "Fort",   "#dc2626"))
        elif glucose >= 100: facteurs.append(("🟠", f"Prédiabète ({glucose} mg/dL)",         "Modéré", "#f59e0b"))
        if totChol >= 240:   facteurs.append(("🟠", f"Cholestérol élevé ({totChol} mg/dL)",  "Modéré", "#f59e0b"))
        if diabetes:         facteurs.append(("🔴", "Diabète préexistant",                    "Fort",   "#dc2626"))
        if prevalentHyp:     facteurs.append(("🟠", "Hypertension connue",                    "Modéré", "#f59e0b"))
        if cigsPerDay >= 20: facteurs.append(("🔴", f"Tabagisme important ({cigsPerDay}/j)",  "Fort",   "#dc2626"))
        elif cigsPerDay > 0: facteurs.append(("🟠", f"Tabagisme ({cigsPerDay}/j)",            "Modéré", "#f59e0b"))
        if BMI >= 30:        facteurs.append(("🟠", f"Obésité (IMC {BMI:.1f})",               "Modéré", "#f59e0b"))
        if male:             facteurs.append(("🟡", "Sexe masculin",                           "Faible", "#ca8a04"))

        if facteurs:
            for ic, desc, niv, c in facteurs:
                st.markdown(
                    f'{ic} **{desc}** '
                    f'<span style="background:{c}22;color:{c};padding:2px 8px;'
                    f'border-radius:10px;font-size:0.78em;font-weight:600">{niv}</span>',
                    unsafe_allow_html=True)
        else:
            st.success("✅ Aucun facteur de risque majeur identifié.")

    with col4:
        st.markdown("### 📊 Profil complet")
        st.dataframe(pd.DataFrame({
            "Variable": ["Sexe","Âge","Cholestérol","P. systolique","P. diastolique",
                         "IMC","Fréq. cardiaque","Glycémie","Cigarettes/j",
                         "Hypertension","Diabète"],
            "Valeur":   [sexe_lbl, f"{age} ans", f"{totChol} mg/dL",
                         f"{sysBP} mmHg", f"{diaBP} mmHg", f"{BMI:.1f} kg/m²",
                         f"{heartRate} bpm", f"{glucose} mg/dL", str(cigsPerDay),
                         "Oui" if prevalentHyp else "Non",
                         "Oui" if diabetes else "Non"]
        }), use_container_width=True, hide_index=True, height=380)

else:
    # État initial
    st.markdown("### 👈 Renseignez le profil du patient dans le panneau gauche")
    st.markdown("")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Fichiers chargés")
        st.markdown("""
        | Fichier | Contenu |
        |---|---|
        | `model_only.pkl` | Instance `LogisticRegression` |
        | `best_threshold.pkl` | Seuil optimal |
        """)
        st.info(f"🎯 Seuil chargé : **{seuil_optimal:.4f}**")

        st.markdown("#### ⚠️ Ce que tu dois gérer manuellement")
        st.markdown("""
        Puisque seul le modèle est chargé, le prétraitement
        est codé en dur dans `pretraiter()` :

        - **log1p** → glucose, cigsPerDay
        - **Winsorisation** → bornes fixes dans `WINSOR_BOUNDS`
        - **Standardisation** → `SCALER_MEAN` et `SCALER_SCALE`
          à mettre à jour avec les valeurs de `scaler.mean_`
          et `scaler.scale_` depuis `train_and_save.py`
        """)

    with col_b:
        st.markdown("#### Comparaison des 3 approches")
        st.markdown("""
        | Approche | Fichiers .pkl | Prétraitement |
        |---|---|---|
        | **Pipeline complet** | 1 seul fichier | Automatique |
        | **Sans pipeline** | 5 fichiers | Manuel dans app.py |
        | **Modèle seul** *(ici)* | 2 fichiers | Codé en dur dans app.py |
        """)
        st.warning("""
        **Risque principal** : si tu ré-entraînes le modèle
        avec des données différentes, les constantes
        `WINSOR_BOUNDS`, `SCALER_MEAN` et `SCALER_SCALE`
        doivent être mises à jour manuellement.
        Sinon les prédictions seront incorrectes sans erreur visible.
        """)
