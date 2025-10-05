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

def classify_batch(df, model, scaler):
    """Process batch predictions"""
    try:
        # Ensure all required features are present
        missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
        if missing_features:
            return None, f"Missing required features: {', '.join(missing_features)}"
        
        # Select only the required features in correct order
        features_df = df[MODEL_FEATURES].copy()
        
        # Scale the features
        scaled_features = scaler.transform(features_df)
        
        # Make predictions
        predictions = model.predict(scaled_features)
        prediction_probas = model.predict_proba(scaled_features)
        
        # Get max probability for each prediction
        max_probas = np.max(prediction_probas, axis=1) * 100
        
        # Determine labels based on confidence
        labels = []
        for pred, prob in zip(predictions, max_probas):
            if prob < 70:
                labels.append('CANDIDATE (Uncertain)')
            elif pred == 1:
                labels.append('CONFIRMED EXOPLANET')
            else:
                labels.append('FALSE POSITIVE')
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['Prediction'] = labels
        result_df['Confidence'] = max_probas
        result_df['Prob_False_Positive'] = prediction_probas[:, 0] * 100
        result_df['Prob_Confirmed'] = prediction_probas[:, 1] * 100
        
        return result_df, None
        
    except Exception as e:
        return None, str(e)

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
    
    # Add tabs for single and batch processing
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Processing"])
    
    # --- TAB 1: Single Prediction (Original functionality) ---
    with tab1:
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
    
    # --- TAB 2: Batch Processing ---
    with tab2:
        st.header("Batch File Processing")
        st.markdown("""
        Upload a CSV file containing multiple exoplanet candidates for batch classification.
        
        **Required columns:**
        - koi_period
        - koi_prad
        - koi_teq
        - koi_impact
        - koi_duration
        - koi_depth
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Uploaded Data Preview")
                st.dataframe(df.head())
                
                st.write(f"Total rows: {len(df)}")
                
                # Process batch predictions
                if st.button("Process Batch", type="primary"):
                    with st.spinner("Processing predictions..."):
                        result_df, error = classify_batch(df, model, scaler)
                        
                        if error:
                            st.error(f"Error during batch processing: {error}")
                        else:
                            st.success(f"Successfully processed {len(result_df)} records!")
                            
                            # Display summary statistics
                            st.subheader("Classification Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            confirmed = len(result_df[result_df['Prediction'] == 'CONFIRMED EXOPLANET'])
                            false_pos = len(result_df[result_df['Prediction'] == 'FALSE POSITIVE'])
                            candidates = len(result_df[result_df['Prediction'] == 'CANDIDATE (Uncertain)'])
                            
                            col1.metric("Confirmed Exoplanets", confirmed)
                            col2.metric("False Positives", false_pos)
                            col3.metric("Candidates", candidates)
                            
                            # Display results
                            st.subheader("Results")
                            st.dataframe(result_df)
                            
                            # Download button for results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="exoplanet_predictions.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    main()
