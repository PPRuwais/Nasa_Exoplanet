import numpy as np
import pandas as pd
import streamlit as st
import joblib

# --- IMPORTANT: Load Model and Scaler ---
MODEL_PATH = "exoplanet_binary_classifier_rf.pkl"
SCALER_PATH = "feature_scaler_rf.pkl"

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

# --- CRITICAL: Features for the Application ---
MODEL_FEATURES = [
    'koi_period', 
    'koi_prad', 
    'koi_teq', 
    'koi_impact', 
    'koi_duration', 
    'koi_depth'
]

# Feature descriptions for better UI
FEATURE_DESCRIPTIONS = {
    'koi_period': 'Orbital Period (days)',
    'koi_prad': 'Planetary Radius (Earth radii)',
    'koi_teq': 'Equilibrium Temperature (K)',
    'koi_impact': 'Impact Parameter',
    'koi_duration': 'Transit Duration (hours)',
    'koi_depth': 'Transit Depth (ppm)'
}

def main():
    st.set_page_config(
        page_title="Exoplanet Classifier",
        page_icon="🪐",
        layout="centered"
    )
    
    st.title("🪐 Exoplanet Classifier")
    st.markdown("""
    This app uses a Random Forest model to classify Kepler Objects of Interest (KOIs) as:
    - **CONFIRMED EXOPLANET**
    - **FALSE POSITIVE**
    - **CANDIDATE** (Uncertain prediction)
    """)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if not model or not scaler:
        st.error("Model or Scaler could not be loaded. Please check the file paths.")
        return
    
    st.header("Input Parameters")
    st.markdown("Enter the following parameters to classify the exoplanet candidate:")
    
    # Create input fields for each feature
    input_features = {}
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        input_features['koi_period'] = st.number_input(
            FEATURE_DESCRIPTIONS['koi_period'], 
            min_value=0.0, 
            value=10.0, 
            step=0.1,
            help="The time it takes for the planet to complete one orbit around its star."
        )
        input_features['koi_prad'] = st.number_input(
            FEATURE_DESCRIPTIONS['koi_prad'], 
            min_value=0.0, 
            value=1.0, 
            step=0.1,
            help="The size of the planet relative to Earth."
        )
        input_features['koi_teq'] = st.number_input(
            FEATURE_DESCRIPTIONS['koi_teq'], 
            min_value=0.0, 
            value=300.0, 
            step=10.0,
            help="The temperature of the planet if it were in thermal equilibrium."
        )
    
    with col2:
        input_features['koi_impact'] = st.number_input(
            FEATURE_DESCRIPTIONS['koi_impact'], 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            help="The sky-projected distance between the center of the stellar disk and the center of the planet disk at conjunction."
        )
        input_features['koi_duration'] = st.number_input(
            FEATURE_DESCRIPTIONS['koi_duration'], 
            min_value=0.0, 
            value=5.0, 
            step=0.1,
            help="The duration of the observed transit."
        )
        input_features['koi_depth'] = st.number_input(
            FEATURE_DESCRIPTIONS['koi_depth'], 
            min_value=0.0, 
            value=100.0, 
            step=1.0,
            help="The fractional loss of light caused by the transit."
        )
    
    # Create a predict button
    if st.button("Classify", type="primary"):
        try:
            # Convert the dictionary to a Pandas DataFrame
            features_df = pd.DataFrame([input_features])
            
            # Scale the input data using the LOADED scaler
            scaled_features = scaler.transform(features_df)
            
            # Make a prediction on the SCALED data
            prediction = model.predict(scaled_features)
            prediction_proba = model.predict_proba(scaled_features)
            
            # Get the probability of the predicted class
            probability = np.max(prediction_proba) * 100
            
            # Use thresholds to determine final label
            if probability < 70:  # Low confidence
                output_label = 'CANDIDATE (Uncertain)'
                color = "orange"
            elif prediction[0] == 1:  # High confidence for confirmed
                output_label = 'CONFIRMED EXOPLANET'
                color = "green"
            else:  # High confidence for false positive
                output_label = 'FALSE POSITIVE'
                color = "red"
            
            # Display the results
            st.header("Classification Results")
            
            # Display the prediction with appropriate color
            st.markdown(f"<h2 style='color:{color}'>{output_label}</h2>", unsafe_allow_html=True)
            
            # Display confidence
            st.metric("Confidence", f"{probability:.2f}%")
            
            # Display detailed probabilities
            with st.expander("Detailed Probabilities"):
                st.write(f"False Positive: {prediction_proba[0][0]*100:.2f}%")
                st.write(f"Confirmed Exoplanet: {prediction_proba[0][1]*100:.2f}%")
            
            # Display debug information
            with st.expander("Debug Information"):
                st.write("Input Features:")
                st.json(input_features)
                st.write("Scaled Features:")
                st.write(scaled_features)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()