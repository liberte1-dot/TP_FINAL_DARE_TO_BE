import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(
    page_title="Prédiction du risque de maladie coronarienne",
    page_icon=None,
    layout="wide"
)

# Titre et description
st.title("Prédiction du risque de maladie coronarienne")
st.markdown("""
Cette application utilise un modèle de régression logistique pour predire le risque de developper 
une maladie coronarienne dans les 10 prochaines annees, base sur l'etude Framingham Heart Study.
""")

# Chargement du modèle et du seuil
@st.cache_resource
def load_model():
    """Charge le modèle et le seuil optimal"""
    try:
        model = joblib.load('model_only.pkl')
        threshold = joblib.load('best_threshold.pkl')
        return model, threshold
    except FileNotFoundError:
        st.error("Fichiers du modele non trouves. Veuillez executer l'entrainement d'abord.")
        return None, None

# Initialisation du scaler
@st.cache_resource
def init_scaler():
    """Initialise le scaler avec les parametres de l'entrainement"""
    scaler = StandardScaler()
    return scaler

# Sidebar pour les entrees utilisateur
st.sidebar.header("Informations du patient")

def create_input_section():
    """Cree les sections de saisie des donnees patient"""
    
    with st.sidebar.expander("Informations demographiques", expanded=True):
        gender = st.selectbox(
            "Sexe",
            options=[("Femme", 0), ("Homme", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        age = st.number_input(
            "Age (ans)",
            min_value=32,
            max_value=70,
            value=50,
            step=1
        )
        
        education = st.selectbox(
            "Niveau d'education",
            options=[
                ("Niveau 1", 1),
                ("Niveau 2", 2),
                ("Niveau 3", 3),
                ("Niveau 4", 4)
            ],
            format_func=lambda x: x[0]
        )[1]

    with st.sidebar.expander("Habitudes de vie", expanded=True):
        cigs_per_day = st.number_input(
            "Cigarettes par jour",
            min_value=0,
            max_value=70,
            value=0,
            step=1
        )

    with st.sidebar.expander("Antecedents medicaux", expanded=True):
        bp_meds = st.selectbox(
            "Traitement pour l'hypertension",
            options=[("Non", 0), ("Oui", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        prevalent_stroke = st.selectbox(
            "Antecedent d'AVC",
            options=[("Non", 0), ("Oui", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        prevalent_hyp = st.selectbox(
            "Hypertension",
            options=[("Non", 0), ("Oui", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        diabetes = st.selectbox(
            "Diabete",
            options=[("Non", 0), ("Oui", 1)],
            format_func=lambda x: x[0]
        )[1]

    with st.sidebar.expander("Mesures cliniques", expanded=True):
        tot_chol = st.number_input(
            "Cholesterol total (mg/dL)",
            min_value=100,
            max_value=600,
            value=200,
            step=1
        )
        
        bmi = st.number_input(
            "IMC",
            min_value=15.0,
            max_value=50.0,
            value=25.0,
            step=0.1
        )
        
        heart_rate = st.number_input(
            "Frequence cardiaque (bpm)",
            min_value=40,
            max_value=150,
            value=75,
            step=1
        )
        
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=40,
            max_value=400,
            value=90,
            step=1
        )
        
        # Calcul de la pression arterielle moyenne
        sys_bp = st.number_input(
            "Pression systolique (mmHg)",
            min_value=80,
            max_value=250,
            value=120,
            step=1
        )
        
        dia_bp = st.number_input(
            "Pression diastolique (mmHg)",
            min_value=50,
            max_value=150,
            value=80,
            step=1
        )
        
        # Calcul de la pression arterielle moyenne
        map_value = (sys_bp + 2 * dia_bp) / 3

    return {
        'gender': gender,
        'age': age,
        'education': education,
        'cigsPerDay': cigs_per_day,
        'BPMeds': bp_meds,
        'prevalentStroke': prevalent_stroke,
        'prevalentHyp': prevalent_hyp,
        'diabetes': diabetes,
        'totChol': tot_chol,
        'BMI': bmi,
        'heartRate': heart_rate,
        'glucose': glucose,
        'pression_arterielle_moyenne': map_value
    }

# Recuperation des entrees
patient_data = create_input_section()

# Fonction de pretraitement des donnees
def preprocess_input(data):
    """Pre traite les donnees d'entree comme lors de l'entrainement"""
    
    # Creation du DataFrame
    df = pd.DataFrame([data])
    
    # Colonnes dans le bon ordre (correspondant a l'entrainement)
    feature_order = [
        'gender', 'age', 'education', 'cigsPerDay', 'BPMeds',
        'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
        'BMI', 'heartRate', 'glucose', 'pression_arterielle_moyenne'
    ]
    
    df = df[feature_order]
    
    return df

# Chargement du modele
model, threshold = load_model()

if model is not None:
    # Bouton de prediction
    if st.sidebar.button("Predire le risque", type="primary", use_container_width=True):
        
        # Pretraitement
        input_df = preprocess_input(patient_data)
        
        # Pour la demo, on utilise un scaler approximatif
        # En production, il faudrait sauvegarder le scaler aussi
        scaler = StandardScaler()
        # On fit sur des valeurs de reference (approximatives)
        reference_data = pd.DataFrame({
            'gender': [0, 1], 'age': [50, 50], 'education': [2, 2],
            'cigsPerDay': [0, 20], 'BPMeds': [0, 0], 'prevalentStroke': [0, 0],
            'prevalentHyp': [0, 1], 'diabetes': [0, 0], 'totChol': [200, 200],
            'BMI': [25, 25], 'heartRate': [75, 75], 'glucose': [90, 90],
            'pression_arterielle_moyenne': [93.33, 93.33]
        })
        scaler.fit(reference_data)
        
        # Normalisation
        input_scaled = scaler.transform(input_df)
        
        # Prediction
        probability = model.predict_proba(input_scaled)[0, 1]
        prediction = (probability >= threshold).astype(int)
        
        # Affichage des resultats
        st.markdown("---")
        st.header("Resultats de la prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Probabilite de risque",
                value=f"{probability:.1%}"
            )
        
        with col2:
            st.metric(
                label="Seuil optimal",
                value=f"{threshold:.2%}"
            )
        
        with col3:
            risk_level = "ELEVE" if prediction == 1 else "FAIBLE"
            st.metric(
                label="Niveau de risque",
                value=risk_level
            )
        
        # Visualisation
        st.markdown("### Visualisation du risque")
        
        # Jauge de risque
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risque (%)"},
            delta = {'reference': threshold * 100},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                'steps': [
                    {'range': [0, threshold * 100], 'color': "lightgreen"},
                    {'range': [threshold * 100, 100], 'color': "lightcoral"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("### Interpretation")
        
        if prediction == 1:
            st.error("""
            **Risque eleve de maladie coronarienne**
            
            Le patient presente un risque significatif de developper une maladie coronarienne 
            dans les 10 prochaines annees. Une consultation medicale est recommandee pour :
            - Evaluer les facteurs de risque modifiables
            - Envisager un traitement preventif si necessaire
            - Adopter des mesures de prevention cardiovasculaire
            """)
        else:
            st.success("""
            **Risque faible de maladie coronarienne**
            
            Le patient presente un risque faible de developper une maladie coronarienne 
            dans les 10 prochaines annees. Il est recommande de :
            - Maintenir un mode de vie sain
            - Effectuer un suivi medical regulier
            - Surveiller les facteurs de risque
            """)
        
        # Facteurs de risque
        st.markdown("### Facteurs de risque principaux")
        
        # Coefficients du modele
        coefficients = {
            'Age': 0.165,
            'Pression arterielle moyenne': 0.102,
            'Hypertension': 0.088,
            'Sexe masculin': 0.063,
            'Medicaments tension': 0.061,
            'Diabete': 0.054,
            'Tabac (cig/jour)': 0.043,
            'IMC': 0.040,
            'Education': -0.039,
            'Cholesterol total': 0.038,
            'AVC anterieur': 0.031,
            'Glucose': 0.015,
            'Frequence cardiaque': -0.003
        }
        
        # Creation d'un DataFrame pour l'affichage
        coef_df = pd.DataFrame([
            {"Facteur": k, "Importance": abs(v), "Direction": "+" if v > 0 else "-"}
            for k, v in coefficients.items()
        ]).sort_values("Importance", ascending=False)
        
        st.dataframe(
            coef_df,
            column_config={
                "Facteur": "Facteur de risque",
                "Importance": st.column_config.NumberColumn("Importance", format="%.3f"),
                "Direction": "Impact"
            },
            hide_index=True,
            use_container_width=True
        )

else:
    st.warning("Le modele n'a pas pu etre charge. Veuillez verifier que les fichiers 'model_only.pkl' et 'best_threshold.pkl' sont presents dans le repertoire.")

# Pied de page
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developpe pour la prediction du risque cardiovasculaire</p>
    <p>Base sur l'etude Framingham Heart Study</p>
</div>
""", unsafe_allow_html=True)

# Instructions pour executer l'application
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### Instructions
    1. Remplissez toutes les informations du patient
    2. Cliquez sur "Predire le risque"
    3. Interpretez les resultats
    """)