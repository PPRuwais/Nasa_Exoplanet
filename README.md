Got it 👍 You want your README to look properly formatted in GitHub (with sections, code blocks, and highlights), not just pasted text. Right now, your draft mixes plain text with "Line Wrapping / Collapse Copy" which comes from a copy-paste, so we’ll clean it.

Here’s a fixed and polished version of your README.md (ready to paste into your repo):

🌌 NASA Kepler Exoplanet Classifier

A machine learning web application that classifies Kepler Objects of Interest (KOIs) as either confirmed exoplanets, false positives, or uncertain candidates.

🔗 Live Demo → Exoplanet Classifier App

🌟 Overview

This application uses a Random Forest model trained on data from NASA’s Kepler mission to classify celestial objects based on their observable characteristics.

The Kepler Space Telescope has discovered thousands of potential exoplanets, but distinguishing between true exoplanets and false positives (such as eclipsing binaries or stellar activity) remains a challenge.
This tool helps automate part of that classification process.

✨ Key Features

Interactive Web Interface: Built with Streamlit

Real-time Classification: Instant predictions with confidence scores

Feature Analysis: See how different parameters influence classification

Educational Tool: Learn about exoplanet detection

Robust ML Model: Random Forest Classifier

🚀 Quick Start
Live Demo

Try the app online:
👉 Streamlit App

Local Installation
# Clone the repository
git clone https://github.com/PPRuwais/Nasa_Exoplanet.git
cd Nasa_Exoplanet

# Install required packages
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
# Run the app with default values
streamlit run streamlit_app.py
nasa_exoplanet/
├── streamlit_app.py                 # Main Streamlit application
├── requirements.txt                 # Dependencies
├── exoplanet_binary_classifier_rf.pkl # Trained Random Forest model
├── feature_scaler_rf.pkl            # Feature scaler for preprocessing
└── README.md                        # Documentation
🔧 System Components
1. Streamlit Application (streamlit_app.py)

Input form for exoplanet parameters

Data preprocessing with scaler

Prediction using Random Forest

Results with confidence scores

2. Random Forest Model (exoplanet_binary_classifier_rf.pkl)

Algorithm: Random Forest Classifier

Training Data: NASA Kepler mission

Features: 6 exoplanet parameters

3. Feature Scaler (feature_scaler_rf.pkl)

StandardScaler for preprocessing

Ensures input distribution matches training data

📊 Data Flow
flowchart LR
A[User Input] --> B[Feature Validation]
B --> C[Data Scaling]
C --> D[Random Forest Model]
D --> E[Prediction + Confidence Score]
E --> F[Results Display]
🔬 Features Used

Orbital Period (koi_period) – time to complete orbit (days)

Planetary Radius (koi_prad) – size relative to Earth

Equilibrium Temperature (koi_teq) – thermal balance temperature (K)

Impact Parameter (koi_impact) – transit alignment (0–1)

Transit Duration (koi_duration) – observed transit length (hours)

Transit Depth (koi_depth) – dip in brightness (ppm)

🎯 Model Architecture

Algorithm: Random Forest

Trees: 100

Features: 6 parameters

Output: Confirmed | False Positive | Candidate

📈 Performance Metrics

Accuracy: High classification accuracy

AUC Score: ROC curve analysis

Precision / Recall: Balance false positives vs true positives

Confusion Matrix: Detailed breakdown

🔍 Usage Examples
Example 1: Known Exoplanet (Kepler-186f)
input_data = {
  'koi_period': 129.9,
  'koi_prad': 1.11,
  'koi_teq': 188,
  'koi_impact': 0.5,
  'koi_duration': 4.5,
  'koi_depth': 100
}
# Expected: CONFIRMED EXOPLANET (high confidence)
input_data = {
  'koi_period': 0.8,
  'koi_prad': 15.0,
  'koi_teq': 3500,
  'koi_impact': 0.95,
  'koi_duration': 2.0,
  'koi_depth': 60000
}
# Expected: FALSE POSITIVE (high confidence)
🤝 Contributing

Fork the project

Create a feature branch (git checkout -b feature/NewFeature)

Commit changes (git commit -m 'Add feature')

Push branch (git push origin feature/NewFeature)

Open a Pull Request

📚 References

NASA Exoplanet Archive

Kepler Mission

Transit Method

Random Forest Classifier

📄 License

This project is licensed under the MIT License.

📞 Contact

👤 Ruwais P P
🔗 GitHub Repo

Happy Exoplanet Hunting! 🪐✨
