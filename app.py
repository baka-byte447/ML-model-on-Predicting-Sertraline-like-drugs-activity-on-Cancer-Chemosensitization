# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import AllChem # Use AllChem for Morgan Fingerprints

# # --- 1. SET PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
# # This line was moved from below to be the first st. command
# st.set_page_config(page_title="SERT Inhibitor Predictor")


# # --- 2. Load Model and Metadata ---
# @st.cache_resource
# def load_model():
#     # Load the new fingerprint-based model
#     with open('best_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model

# @st.cache_resource
# def load_metadata():
#     # Load the new metadata
#     with open('model_metadata.pkl', 'rb') as f:
#         metadata = pickle.load(f)
#     return metadata

# model = load_model()
# metadata = load_metadata()

# # Get fingerprint parameters from metadata
# FP_RADIUS = metadata['features']['radius']
# FP_SIZE = metadata['features']['size']

# # --- 3. Feature Calculation Function ---
# def calculate_morgan_fp(smiles, radius=FP_RADIUS, n_bits=FP_SIZE):
#     """
#     Calculates the required Morgan Fingerprint from a SMILES string.
#     Returns a numpy array or None if SMILES is invalid.
#     """
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return None # Handle invalid SMILES
        
#         fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
#         return np.array(fp)
    
#     except Exception as e:
#         # Print error to console, but let main app handle UI
#         print(f"Error calculating features: {e}") 
#         return None

# # --- 4. Streamlit User Interface ---
# # The st.set_page_config() line was removed from here
# st.title("🔬 SERT Inhibitor Activity Predictor (Fingerprint Model)")
# st.write(f"""
# Enter a SMILES string to predict if the compound is **Active** or **Inactive**.
# This model uses a {FP_SIZE}-bit Morgan Fingerprint (Radius={FP_RADIUS}).
# """)

# # Example SMILES
# st.code("C(c1ccc(C(F)(F)F)cc1)C[NH2+]C", "Fluoxetine (Prozac)")

# st.write("---")

# # Input form
# smiles_input = st.text_input("Enter SMILES String:", "CN(C)CC[C@H](c1ccccc1)c2ccc(F)cc2")

# if st.button("Predict Activity"):
#     if smiles_input:
#         # 1. Calculate features
#         features_fp = calculate_morgan_fp(smiles_input)
        
#         if features_fp is not None:
#             # 2. Reshape for model (no scaling needed)
#             features_array = features_fp.reshape(1, -1)
            
#             # 3. Make prediction
#             prediction = model.predict(features_array)[0]
#             probability = model.predict_proba(features_array)[0]
            
#             # 4. Display results
#             st.subheader("Prediction Result")
#             if prediction == 1:
#                 st.success(f"**Result: Active** (Likely an inhibitor)")
#                 st.write(f"**Confidence:** {probability[1]*100:.2f}%")
#             else:
#                 st.error(f"**Result: Inactive** (Unlikely an inhibitor)")
#                 st.write(f"**Confidence:** {probability[0]*100:.2f}%")
            
#         else:
#             st.error("Invalid SMILES string. Please check your input.")
#     else:
#         st.warning("Please enter a SMILES string.")






# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import time  # To add a small delay for the spinner

# # --- 1. SET PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
# st.set_page_config(
#     page_title="SERT Inhibitor Predictor",
#     page_icon="🔬",
#     layout="centered"
# )

# # --- 2. Load Model and Metadata ---
# @st.cache_resource
# def load_model():
#     with open('best_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model

# @st.cache_resource
# def load_metadata():
#     with open('model_metadata.pkl', 'rb') as f:
#         metadata = pickle.load(f)
#     return metadata

# try:
#     model = load_model()
#     metadata = load_metadata()
    
#     # Get fingerprint parameters from metadata
#     FP_RADIUS = metadata['features']['radius']
#     FP_SIZE = metadata['features']['size']
    
# except FileNotFoundError:
#     st.error("ERROR: Model files not found. Please ensure 'best_model.pkl' and 'model_metadata.pkl' are in the same directory.")
#     st.stop()

# # --- 3. Custom CSS for Attractive UI ---
# st.markdown("""
# <style>
#     /* Main app background */
#     [data-testid="stAppViewContainer"] {
#         background-color: #f0f2f6;
#     }

#     /* Sidebar style */
#     [data-testid="stSidebar"] {
#         background-color: #f8f9fa;
#         border-right: 1px solid #ddd;
#     }
    
#     /* Main content area */
#     [data-testid="stMain"] > div {
#         background-color: #ffffff;
#         padding: 2rem;
#         border-radius: 10px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#     }

#     /* Custom result card */
#     .result-card {
#         padding: 25px;
#         border-radius: 10px;
#         text-align: center;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.05);
#         border: 1px solid #eee;
#         margin-top: 20px;
#     }
#     .result-active {
#         border-left: 8px solid #28a745;
#         background-color: #f0fff4;
#     }
#     .result-inactive {
#         border-left: 8px solid #dc3545;
#         background-color: #fff0f0;
#     }
#     .result-text {
#         font-size: 2.5rem; /* Larger text */
#         font-weight: bold;
#         margin-bottom: 10px;
#     }
#     .active-text { color: #28a745; }
#     .inactive-text { color: #dc3545; }
#     .confidence-text {
#         font-size: 1.8rem;
#         font-weight: 600;
#     }
# </style>
# """, unsafe_allow_html=True)


# # --- 4. Sidebar Information ---
# st.sidebar.title("About This App")
# st.sidebar.info(
#     "This app predicts the biological activity of a chemical compound against the "
#     "Serotonin Transporter (SERT). It uses a Machine Learning model trained on "
#     "publicly available data from the ChEMBL database."
# )

# st.sidebar.title("Model Performance")
# try:
#     metrics = metadata['metrics']
#     st.sidebar.metric("Best Model", metadata['best_model_name'])
#     st.sidebar.metric("Test Accuracy", f"{metrics['Test Accuracy'] * 100:.2f} %")
#     st.sidebar.metric("Test F1-Score", f"{metrics['F1-Score']:.4f}")
#     st.sidebar.metric("Test ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
# except KeyError:
#     st.sidebar.error("Could not load all metrics from metadata.")

# st.sidebar.subheader("Feature Parameters")
# st.sidebar.write(f"**Type:** {metadata['features']['type']}")
# st.sidebar.write(f"**Radius:** {metadata['features']['radius']}")
# st.sidebar.write(f"**Size (bits):** {metadata['features']['size']}")


# # --- 5. Feature Calculation Function ---
# def calculate_morgan_fp(smiles, radius=FP_RADIUS, n_bits=FP_SIZE):
#     """
#     Calculates the required Morgan Fingerprint from a SMILES string.
#     Returns a numpy array or None if SMILES is invalid.
#     """
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return None
#         fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
#         return np.array(fp)
#     except Exception:
#         return None

# # --- 6. Main Page UI ---
# st.title("🔬 SERT Inhibitor Activity Predictor")
# st.write(f"""
# Enter a SMILES string to predict if the compound is **Active** or **Inactive**.
# """)

# # Input form
# with st.form(key='smiles_form'):
#     smiles_input = st.text_input(
#         "Enter SMILES String:", 
#         "CN(C)CC[C@H](c1ccccc1)c2ccc(F)cc2" # Example: Escitalopram
#     )
#     submit_button = st.form_submit_button(label='Predict Activity')

# with st.expander("Show example SMILES strings"):
#     st.code("C(c1ccc(C(F)(F)F)cc1)C[NH2+]C", "Fluoxetine (Prozac)")
#     st.code("CN(C)CCCC(c1ccccc1)c2ccc(C#N)cc2", "Citalopram (Celexa)")
#     st.code("Clc1ccc(C2c3ccccc3C[C@H]2N(C)C)c(Cl)c1", "Sertraline (Zoloft)")

# st.write("---")

# # --- 7. Prediction Logic ---
# if submit_button:
#     if smiles_input:
#         with st.spinner("Calculating features and running prediction..."):
#             time.sleep(0.5) # Small delay to make spinner visible
            
#             # 1. Calculate features
#             features_fp = calculate_morgan_fp(smiles_input)
            
#             if features_fp is not None:
#                 # 2. Reshape for model (no scaling needed)
#                 features_array = features_fp.reshape(1, -1)
                
#                 # 3. Make prediction
#                 prediction = model.predict(features_array)[0]
#                 probability = model.predict_proba(features_array)[0]
                
#                 # 4. Display attractive results
#                 st.subheader("Prediction Result")
                
#                 if prediction == 1:
#                     prob = probability[1]
#                     label = "Active"
#                     color_class = "active"
#                     icon = "✅"
#                 else:
#                     prob = probability[0]
#                     label = "Inactive"
#                     color_class = "inactive"
#                     icon = "❌"

#                 # Custom HTML card
#                 st.markdown(f"""
#                 <div class="result-card result-{color_class}">
#                     <p style="font-size: 1.2rem; color: #555; margin-bottom: 10px;">PREDICTION</p>
#                     <p class="result-text {color_class}-text">{icon} {label}</p>
#                     <hr style="border-top: 1px solid #eee; margin: 15px 0;">
#                     <p style="font-size: 1.2rem; color: #555; margin-bottom: 10px;">CONFIDENCE</p>
#                     <p class="confidence-text {color_class}-text">
#                         {prob*100:.2f}%
#                     </p>
#                 </div>
#                 """, unsafe_allow_html=True)

#                 st.write("") # Add some space
                
#                 # Show probability breakdown
#                 st.subheader("Probability Breakdown")
#                 prob_df = pd.DataFrame({
#                     'Class': ['Inactive (0)', 'Active (1)'],
#                     'Probability': probability
#                 })
#                 st.bar_chart(prob_df.set_index('Class'))
                
#             else:
#                 st.error("⚠️ Invalid SMILES string. Please check your input.")
#     else:
#         st.warning("Please enter a SMILES string to predict.")






import streamlit as st
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import time

# --- 1. SET PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="SERT Inhibitor Predictor",
    page_icon="🔬",  # Adds a nice icon to the browser tab
    layout="centered"  # Keeps the app clean and focused
)

# --- 2. Load Model and Metadata ---
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_metadata():
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return metadata

try:
    model = load_model()
    metadata = load_metadata()
    
    # Get fingerprint parameters from metadata
    FP_RADIUS = metadata['features']['radius']
    FP_SIZE = metadata['features']['size']
    
except FileNotFoundError:
    st.error("ERROR: Model files not found. Please ensure 'best_model.pkl' and 'model_metadata.pkl' are in the same directory.")
    st.stop()

# --- 3. Custom CSS for Attractive UI (Theme-Aware) ---
st.markdown("""
<style>
    /* --- Base Theme --- */
    /* Let Streamlit handle main app and sidebar background for theme compatibility */
    [data-testid="stAppViewContainer"] {
        /* No background override */
    }
    
    [data-testid="stMain"] > div {
        /* Keep padding/border/shadow, remove hardcoded background */
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* --- Result Card Base --- */
    .result-card {
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-top: 20px;
        border: 1px solid #eee; /* Default light mode border */
    }

    /* --- Light Mode Card Colors --- */
    .result-active {
        border-left: 8px solid #28a745;
        background-color: #f0fff4; /* Light green bg */
    }
    .result-inactive {
        border-left: 8px solid #dc3545;
        background-color: #fff0f0; /* Light red bg */
    }
    .active-text { color: #28a745; }
    .inactive-text { color: #dc3545; }
    .card-label {
        font-size: 1.2rem;
        color: #555; /* Dark gray for light mode */
        margin-bottom: 10px;
    }

    /* --- Dark Mode Card Colors (This is the fix) --- */
    [data-theme="dark"] .result-card {
        border: 1px solid #333; /* Dark mode border */
    }
    [data-theme="dark"] .result-active {
        background-color: #0e2114; /* Dark green bg */
    }
    [data-theme="dark"] .result-inactive {
        background-color: #2c0b0e; /* Dark red bg */
    }
    /* Text colors (.active-text / .inactive-text) remain the same, as they are bright */
    [data-theme="dark"] .card-label {
        color: #aaa; /* Light gray for dark mode */
    }

    /* --- Text Styles inside Card --- */
    .result-text {
        font-size: 2.5rem; 
        font-weight: bold;
        margin-bottom: 10px;
    }
    .confidence-text {
        font-size: 1.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- 4. Sidebar Information ---
st.sidebar.title("About This App")
st.sidebar.info(
    "This app predicts the biological activity of a chemical compound against the "
    "Serotonin Transporter (SERT). It uses a Machine Learning model trained on "
    "publicly available data from the ChEMBL database."
)

st.sidebar.title("Model Performance")
try:
    metrics = metadata['metrics']
    st.sidebar.metric("Best Model", metadata['best_model_name'])
    # Using columns for a cleaner layout
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Test Accuracy", f"{metrics['Test Accuracy'] * 100:.2f} %")
    col2.metric("Test F1-Score", f"{metrics['F1-Score']:.4f}")
    col1.metric("Test ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
    col2.metric("CV Accuracy", f"{metrics['CV Accuracy']:.4f}")
except KeyError:
    st.sidebar.error("Could not load all metrics from metadata.")
except TypeError:
    st.sidebar.error("Metadata format error. Could not load metrics.")

st.sidebar.subheader("Feature Parameters")
st.sidebar.write(f"**Type:** {metadata['features']['type']}")
st.sidebar.write(f"**Radius:** {metadata['features']['radius']}")
st.sidebar.write(f"**Size (bits):** {metadata['features']['size']}")


# --- 5. Feature Calculation Function ---
def calculate_morgan_fp(smiles, radius=FP_RADIUS, n_bits=FP_SIZE):
    """
    Calculates the required Morgan Fingerprint from a SMILES string.
    Returns a numpy array or None if SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None

# --- 6. Main Page UI ---
st.title("🔬 SERT Inhibitor Activity Predictor")
st.write(f"""
Enter a SMILES string to predict if the compound is **Active** or **Inactive** (Activity threshold < {metadata.get('threshold', 100)} nM).
""")

# Input form
with st.form(key='smiles_form'):
    smiles_input = st.text_input(
        "Enter SMILES String:", 
        "CN(C)CC[C@H](c1ccccc1)c2ccc(F)cc2" # Example: Escitalopram
    )
    submit_button = st.form_submit_button(label='🚀 Predict Activity')

# Expander for examples to keep the UI clean
with st.expander("Show example SMILES strings"):
    st.code("C(c1ccc(C(F)(F)F)cc1)C[NH2+]C", "Fluoxetine (Prozac)")
    st.code("CN(C)CCCC(c1ccccc1)c2ccc(C#N)cc2", "Citalopram (Celexa)")
    st.code("Clc1ccc(C2c3ccccc3C[C@H]2N(C)C)c(Cl)c1", "Sertraline (Zoloft)")

st.write("---")

# --- 7. Prediction Logic ---
if submit_button:
    if smiles_input:
        # Show a loading spinner while processing
        with st.spinner("Calculating features and running prediction..."):
            time.sleep(0.5) # Small delay to make spinner visible
            
            features_fp = calculate_morgan_fp(smiles_input)
            
            if features_fp is not None:
                features_array = features_fp.reshape(1, -1)
                
                prediction = model.predict(features_array)[0]
                probability = model.predict_proba(features_array)[0]
                
                if prediction == 1:
                    prob = probability[1]
                    label = "Active"
                    color_class = "active"
                    icon = "✅"
                else:
                    prob = probability[0]
                    label = "Inactive"
                    color_class = "inactive"
                    icon = "❌"

                # Display the custom HTML card
                st.subheader("Prediction Result")
                st.markdown(f"""
                <div class="result-card result-{color_class}">
                    <p class="card-label">PREDICTION</p>
                    <p class="result-text {color_class}-text">{icon} {label}</p>
                    <hr style="border-top: 1px solid #eee; margin: 15px 0;">
                    <p class="card-label">CONFIDENCE</p>
                    <p class="confidence-text {color_class}-text">
                        {prob*100:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.write("") # Add some space
                
                # Show probability breakdown
                st.subheader("Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Class': ['Inactive (0)', 'Active (1)'],
                    'Probability': probability
                })
                # Use st.bar_chart for theme-aware plotting
                st.bar_chart(prob_df.set_index('Class'))
                
            else:
                st.error("⚠️ Invalid SMILES string. Please check your input.")
    else:
        st.warning("Please enter a SMILES string to predict.")