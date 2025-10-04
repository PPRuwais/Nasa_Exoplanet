# 🌌 NASA Kepler Exoplanet Classifier

A Streamlit web app for predicting the existence of exoplanets based on NASA's dataset using a **Random Forest Classifier**.

---

## 🌟 Overview
This application uses data from NASA's Kepler mission to classify Kepler Objects of Interest (KOIs) as **confirmed exoplanets**, **false positives**, or **uncertain candidates**.

This project was developed as part of a **NASA Space Apps Challenge** submission.

---

## 🚀 Live Demo
Access the live app here: [Exoplanet Classifier App](https://nasaexoplanet-v9i5osfgpf8cedmlaunhtv.streamlit.app/)

---

## 🔧 Local Installation

```bash
# Clone the repository
git clone https://github.com/PPRuwais/Nasa_Exoplanet.git
cd Nasa_Exoplanet

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```plaintext
nasa_exoplanet/
├── streamlit_app.py                     # Main Streamlit app
├── requirements.txt                     # Python dependencies
├── exoplanet_binary_classifier_rf.pkl   # Trained Random Forest model
├── feature_scaler_rf.pkl                # Scaler for input features
└── README.md                            # Project documentation
```

---

## 🧠 Model Info

- **Algorithm:** Random Forest Classifier
- **Training Data:** NASA Kepler mission dataset
- **Features Used:**
  - Orbital Period (koi_period)
  - Planetary Radius (koi_prad)
  - Equilibrium Temperature (koi_teq)
  - Impact Parameter (koi_impact)
  - Transit Duration (koi_duration)
  - Transit Depth (koi_depth)
- **Output:** Confirmed Exoplanet | False Positive | Candidate

### Optional: Retrain Model
The training script is available in `training/train_model.py`. Run it to generate the Random Forest model and feature scaler.

```bash
# Example to retrain the model
python training/train_model.py
```

*Note: Retraining is optional since pre-trained model files are included.*

---

## 🔍 Usage Examples

### Example 1: Known Exoplanet (Kepler-186f)
```python
input_data = {
  'koi_period': 129.9,
  'koi_prad': 1.11,
  'koi_teq': 188,
  'koi_impact': 0.5,
  'koi_duration': 4.5,
  'koi_depth': 100
}
# Expected Output: CONFIRMED EXOPLANET (high confidence)
```

### Example 2: False Positive
```python
input_data = {
  'koi_period': 0.8,
  'koi_prad': 15.0,
  'koi_teq': 3500,
  'koi_impact': 0.95,
  'koi_duration': 2.0,
  'koi_depth': 60000
}
# Expected Output: FALSE POSITIVE (high confidence)
```

---

## 🛠️ Troubleshooting

- **Model Loading Error:** Ensure `.pkl` files are in the correct directory.
- **Prediction Error:** Enter valid numerical values in all fields.
- **Module Not Found:** Run `pip install -r requirements.txt`.

---

## 🤝 Contributing
1. Fork the project
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## 📚 References
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [Transit Method](https://exoplanets.nasa.gov/alien-worlds/ways-to-find-a-planet/#transits)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 📞 Contact
👤 **Ruwais P P**  
🔗 [GitHub Repo](https://github.com/PPRuwais/Nasa_Exoplanet)  

Happy Exoplanet Hunting! 🪐✨



