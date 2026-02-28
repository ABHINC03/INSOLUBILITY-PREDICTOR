import streamlit as st
import pandas as pd
import pickle
import joblib
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from cal import get_molecular_features
import py3Dmol
from stmol import showmol
import pubchempy as pcp
from streamlit_lottie import st_lottie

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="InSilico Solubility Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS ---
st.markdown("""
<style>
    /* Card Styling */
    .metric-card {
        background-color: #262730;
        border: 1px solid #444;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card h2 {
        margin: 0;
        font-size: 32px;
        font-weight: 700;
    }
    .metric-card h3 {
        margin: 10px 0;
        font-size: 24px;
        color: #E0E0E0;
    }
    .metric-card p {
        color: #A0A0A0;
        margin: 0;
    }
    
    /* Molecule Name Styling */
    .molecule-name {
        font-size: 48px;
        font-weight: 800;
        color: #00C9FF;
        text-align: center;
        margin-top: -20px;
        margin-bottom: 30px;
        text-transform: capitalize;
        text-shadow: 0px 0px 15px rgba(0, 201, 255, 0.6);
    }
    
    /* Custom Title Styling to match Animation */
    .custom-title {
        font-size: 50px;
        font-weight: bold;
        margin-top: 10px;
        background: -webkit-linear-gradient(left, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Container Borders */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 10px;
        background-color: #1E1E1E;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Load Resources ---
@st.cache_resource
def load_model_and_features():
    try:
        with open('solubility_model.pkl', 'rb') as file:
            model = pickle.load(file)
        features = joblib.load('solubilty_features.pkl')
        return model, features
    except Exception:
        return None, None

# Function to load Lottie animations
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

model, feature_order = load_model_and_features()

# --- 4. Lottie Animation URLs (UPDATED & TESTED) ---
# Main Logo: Bubbling Science Beaker
lottie_science_logo = load_lottieurl("https://lottie.host/5a80572d-3c22-4096-857c-65239a039744/lottie.json")

# Status Animations
 # Rock/Solid

# --- 5. Helper Functions ---
def make_3d_view(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        view = py3Dmol.view(width=500, height=400)
        view.addModel(Chem.MolToMolBlock(mol), 'mol')
        view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}})
        view.setBackgroundColor('#1E1E1E')
        view.zoomTo()
        return view
    return None

def get_molecule_name(smiles):
    try:
        compounds = pcp.get_compounds(smiles, namespace='smiles')
        if compounds:
            name = compounds[0].synonyms[0] if compounds[0].synonyms else compounds[0].iupac_name
            return name
    except:
        return None
    return "Custom Molecule"

# --- 6. Sidebar ---
with st.sidebar:
    # Small sidebar logo
    if lottie_science_logo:
        st_lottie(lottie_science_logo, height=100, key="sidebar_logo")
    
    st.title("Molecule Input")
    st.markdown("### Quick Fill")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Paracetamol"): st.session_state.smiles = "CC(=O)Nc1ccc(O)cc1"
    with col2:
        if st.button("Benzene"): st.session_state.smiles = "c1ccccc1"
            
    if "smiles" not in st.session_state: st.session_state.smiles = ""
    smiles_input = st.text_input("SMILES String", key="smiles_input", value=st.session_state.smiles)
    predict_btn = st.button("🚀 Predict Solubility", type="primary")

# --- 7. Main Dashboard ---

# *** UPDATED TITLE SECTION WITH ANIMATION ***
head_c1, head_c2 = st.columns([1, 6]) # 1 part logo, 6 parts text

with head_c1:
    if lottie_science_logo:
        st_lottie(lottie_science_logo, height=90, key="main_logo")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/123/123392.png", width=80)

with head_c2:
    st.markdown('<div class="custom-title">InSilico Solubility Predictor</div>', unsafe_allow_html=True)

st.divider()

if not model:
    st.error("❌ Critical Error: Model files missing.")
    st.stop()

if predict_btn and smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    
    if not mol:
        st.error(f"❌ Invalid SMILES: `{smiles_input}`")
    else:
        with st.spinner("Analyzing Chemical Properties..."):
            features_dict = get_molecular_features(smiles_input)
            mol_name = get_molecule_name(smiles_input)
            
        if features_dict:
            # Predict
            df_input = pd.DataFrame([features_dict])
            model_cols = [f for f in feature_order if f != 'Solubility']
            X_input = df_input[model_cols]
            prediction = model.predict(X_input)[0]
            
            # Classification & Animation Logic
            if prediction > -2.0:
                solubility_class = "Soluble"
                color = "#00FF00" # Green
                
            elif prediction > -4.0:
                solubility_class = "Partially Soluble"
                color = "#FFA500" # Orange
                
            else:
                solubility_class = "Insoluble"
                color = "#FF4B4B" # Red
                

            # --- Layout ---
            
            # 1. Title
            if mol_name:
                st.markdown(f'<div class="molecule-name">{mol_name}</div>', unsafe_allow_html=True)

            # 2. Visualizations
            vis_c1, vis_c2 = st.columns(2)
            with vis_c1:
                with st.container():
                    st.markdown("<h4 style='text-align: center; color: #ccc;'>2D Structure</h4>", unsafe_allow_html=True)
                    img = Draw.MolToImage(mol, size=(450, 400))
                    st.image(img, use_container_width=True)

            with vis_c2:
                with st.container():
                    st.markdown("<h4 style='text-align: center; color: #ccc;'>3D Interactive</h4>", unsafe_allow_html=True)
                    view = make_3d_view(smiles_input)
                    if view: showmol(view, height=400, width=500)

            st.divider()

            # 3. Animated Prediction Result
            st.subheader("Prediction Result")
            
            # Split into Animation (Left) and Text Data (Right) for a clean look
            res_c1, res_c2 = st.columns([1, 2])
            
            
                # Display Lottie Animation
                
            
            with res_c2:
                # Display Data Card
                st.markdown(f"""
                <div class="metric-card" style="border-left: 10px solid {color}; text-align: left; padding-left: 30px;">
                    <h2 style="color: {color} !important;">{solubility_class}</h2>
                    <h3>LogS: {prediction:.4f}</h3>
                    <p>Concentration (mol/L)</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"**Insight:** This molecule has a LogS of **{prediction:.2f}**, which falls into the **{solubility_class}** category.")

            # 4. Data Table
            st.divider()
            with st.expander("🔬FEATURES && METRICS"):
                st.dataframe(X_input.T.rename(columns={0: "Value"}).style.format("{:.4f}"), use_container_width=True)
else:
    st.info("👈 Enter a SMILES string in the sidebar to start.")