# 🌌 Exoplanet Classifier — NASA Space Apps Challenge 2025

A machine learning system that classifies exoplanet candidates as confirmed, false positives, or uncertain to help prioritize follow-up observations.

🔗 **Live Demo:** [https://nasaexoplanet-v9i5osfgpf8cedmlaunhtv.streamlit.app/](https://nasaexoplanet-v9i5osfgpf8cedmlaunhtv.streamlit.app/)

**Team:** Ctrl+Space

---

## 🎯 NASA Space Apps Challenge Alignment

This project addresses the **challenge considerations** outlined in the NASA Space Apps requirements:

### ✅ **Aimed at Multiple Audiences**
> "Your project could be aimed at researchers wanting to classify new data or in the field who want to interact with exoplanet data and do not know where to start."

- **For Researchers:** Confidence scores help prioritize follow-up observations
- **For Field Workers:** Simple interface requires no ML expertise
- **For Novices:** Clear explanations, example values, and educational content
- **For Educators:** Interactive tool teaches exoplanet characteristics

### ✅ **Data Ingestion Capability**
> "Your interface could enable your tool to ingest new data and train the models as it does so."

The system includes:
- Complete retraining pipeline (`training ML code.py`)
- Modular architecture allowing feature updates
- StandardScaler for consistent preprocessing
- Ability to retrain on updated Kepler data

### ✅ **Model Statistics & Transparency**
> "Your interface could show statistics about the accuracy of the current model."

Our interface provides:
- **Precision, Recall, F1-Score:** Generated during training with classification reports
- **Confusion Matrix:** Visualizes true/false positives and negatives
- **ROC Curve & AUC Score:** Shows model discrimination ability
- **Feature Importance:** Reveals which parameters drive predictions
- **Confidence Scores:** Real-time probability breakdown for each prediction
- **Probability Distribution:** Shows how candidates cluster in confidence space

### ✅ **Hyperparameter Tuning**
> "Your model could allow hyperparameter tweaking from the interface."

Current implementation includes:
- Configurable Random Forest parameters in training code
- SMOTE for handling class imbalance
- Adjustable confidence thresholds (currently 70%)
- Extensible architecture for parameter experimentation

---

## 📊 Model Performance & Statistics

### Training Data
- **Total labeled objects:** 5,195 from Kepler mission
- **Confirmed exoplanets:** 2,241 
- **False positives:** 2,954
- **Train/Test split:** 80/20 with stratification
- **Test set size:** 1,517 objects

### Classification Report

|                | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| FALSE POSITIVE | 0.95      | 0.86   | 0.90     | 968     |
| CONFIRMED      | 0.79      | 0.92   | 0.85     | 549     |
| **Accuracy**   |           |        | **0.88** | 1517    |
| **Macro Avg**  | 0.87      | 0.89   | 0.88     | 1517    |
| **Weighted Avg**| 0.89     | 0.88   | 0.88     | 1517    |

### Model Metrics
- **Overall Accuracy:** 88.33%
- **ROC AUC Score:** 0.9591 (excellent discrimination ability)

### Feature Importance Analysis

Features ranked by contribution to predictions:

| Rank | Feature       | Importance | Interpretation                           |
|------|---------------|------------|------------------------------------------|
| 1    | koi_prad      | 0.2518     | Planetary radius is the strongest predictor |
| 2    | koi_period    | 0.2183     | Orbital period second most important     |
| 3    | koi_depth     | 0.1707     | Transit depth significantly contributes  |
| 4    | koi_teq       | 0.1413     | Temperature provides useful signal       |
| 5    | koi_impact    | 0.1280     | Impact parameter aids classification     |
| 6    | koi_duration  | 0.0899     | Duration has lowest but still useful contribution |

**Key Insight:** Planetary radius and orbital period together account for ~47% of the model's decision-making process, highlighting their critical role in distinguishing real exoplanets from false positives.

---

## 🧠 How It Works

### The Challenge: Ambiguous Data
The Kepler dataset has three labels:
- **CONFIRMED** — verified through follow-up observations
- **FALSE POSITIVE** — proven not to be planets  
- **CANDIDATE** — awaiting confirmation (inherently uncertain)

### Our Solution: Binary Classification + Confidence Thresholds
1. **Train on what we know:** Model learns patterns from CONFIRMED vs FALSE POSITIVE
2. **Apply to unknowns:** Classify new candidates based on learned patterns
3. **Acknowledge uncertainty:** When confidence < 70%, label as "CANDIDATE (Uncertain)"

This approach is scientifically sound because you cannot train a model to predict "we don't know" as a distinct class.

---

## 🔬 Technical Architecture

### Model: Random Forest Classifier
```python
Pipeline([
    ('smote', SMOTE(random_state=42)),  # Handle class imbalance
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ))
])
```

### Feature Scaling
```python
StandardScaler()  # Normalizes all features to same scale
```

### Confidence Thresholds
```python
if confidence >= 70%:
    return "CONFIRMED" or "FALSE POSITIVE"
else:
    return "CANDIDATE (Uncertain)"
```

---

## 🪐 Input Features

The model analyzes **six key transit parameters**:

| Feature          | Description                    | Scientific Importance                      |
| ---------------- | ------------------------------ | ------------------------------------------ |
| **koi_period**   | Orbital period (days)          | Distinguishes planets from stellar activity |
| **koi_prad**     | Planetary radius (Earth radii) | Separates planets from eclipsing binaries   |
| **koi_teq**      | Equilibrium temperature (K)    | Indicates distance from host star           |
| **koi_impact**   | Impact parameter (0–1)         | Grazing transits often indicate false positives |
| **koi_duration** | Transit duration (hours)       | Validates orbital geometry                  |
| **koi_depth**    | Transit depth (ppm)            | Measures planet-to-star size ratio          |

---

## 🚀 Quick Start

### Online (Easiest)
Just visit: [https://nasaexoplanet-v9i5osfgpf8cedmlaunhtv.streamlit.app/](https://nasaexoplanet-v9i5osfgpf8cedmlaunhtv.streamlit.app/)

### Local Installation
```bash
# Clone the repository
git clone https://github.com/PPRuwais/Nasa_Exoplanet.git
cd Nasa_Exoplanet

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### Try These Test Cases

**Confirmed Exoplanet (Hot Super-Earth) — 93% confidence:**
```
Period: 10 | Radius: 2.5 | Temperature: 900
Impact: 0.3 | Duration: 3.0 | Depth: 1500
```

**False Positive (Grazing Transit) — 92% confidence:**
```
Period: 365 | Radius: 1.0 | Temperature: 288
Impact: 0.2 | Duration: 6.5 | Depth: 84
```

**Uncertain Candidate — 54% confidence:**
```
Period: 1 | Radius: 1 | Temperature: 1
Impact: 1 | Duration: 1 | Depth: 1
```

---

## 📁 Project Structure

```
Nasa_Exoplanet/
├── streamlit_app.py                      # Main web application
├── app.py                                # Flask version (alternative)
├── requirements.txt                      # Python dependencies
├── exoplanet_binary_classifier_rf.pkl    # Trained Random Forest model
├── feature_scaler_rf.pkl                 # Feature preprocessing scaler
├── training ML code.py                   # Complete training pipeline
├── templates/                            # Flask HTML templates
│   ├── index.html
│   └── results.html
└── README.md                             # This file
```

---

## 🔄 Retraining the Model

Want to retrain with updated data or different features?

```bash
python "training ML code.py"
```

This script:
1. Loads Kepler CSV data
2. Selects the 6 key features
3. Trains Random Forest with SMOTE (handles class imbalance)
4. Generates evaluation plots:
   - **Confusion Matrix** (binary_confusion_matrix_6features.png)
   - **ROC Curve** (roc_curve_6features.png)
   - **Feature Importance** (feature_importance_6features.png)
   - **Classification Report** with Precision, Recall, F1-Score
5. Saves new model and scaler files

**Modify features by editing:**
```python
MODEL_FEATURES = [
    'koi_period', 'koi_prad', 'koi_teq', 
    'koi_impact', 'koi_duration', 'koi_depth'
]
```

---

## 🎓 Use Cases

**For Researchers:**
- Prioritize follow-up observations based on confidence scores
- Quickly screen large batches of KOIs
- Export probability distributions for analysis

**For Educators:**
- Demonstrate ML applications in astronomy
- Interactive tool for teaching exoplanet characteristics
- Real-world example of handling imbalanced datasets

**For Science Communicators:**
- Make cutting-edge astronomy accessible
- Visualize how AI assists scientific discovery
- Engage public with interactive predictions

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more features (stellar parameters, multiple transit observations)
- Implement ensemble methods (combine multiple models)
- Create data visualization dashboard
- Add real-time updates from NASA Exoplanet Archive API

1. Fork the project
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open pull request

---

## 📚 Data Sources & References

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission Overview](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [Kepler Objects of Interest (KOI) Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [SMOTE for Imbalanced Data](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

## 🏆 NASA Space Apps Challenge 2025

**Team:** Ctrl+Space  
**Challenge:** Exoplanet Detection & Classification  

### Impact & Innovation
- **88.33% accuracy** with 95.91% AUC score on 1,517 test objects
- **Handles 5,195 labeled Kepler objects** with full retraining capability
- **Three-tier confidence system** (Confirmed/False Positive/Uncertain) based on 70% threshold
- **Transparent AI:** Full probability breakdowns and feature importance analysis
- **Production-ready:** Deployed web interface with real-time predictions

### Addressing NASA Space Apps Requirements

**✓ Multi-audience design:** Researchers get confidence scores; novices get guided interface  
**✓ Data ingestion:** Complete retraining pipeline with modular feature selection  
**✓ Model statistics:** Precision, recall, F1-scores, confusion matrices, ROC curves  
**✓ Hyperparameter access:** Configurable Random Forest, SMOTE, threshold tuning  
**✓ Educational value:** Makes exoplanet science accessible to all skill levels

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 🙏 Acknowledgments

- NASA Kepler Mission Team for the data
- NASA Space Apps Challenge organizers
- scikit-learn and Streamlit communities

**Built for NASA Space Apps Challenge 2025 by Team Ctrl+Space**



