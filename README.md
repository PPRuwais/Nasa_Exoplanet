A machine learning web application that classifies Kepler Objects of Interest (KOIs) as either confirmed exoplanets, false positives, or candidates with uncertain predictions.

🌟 Overview
This application uses a Random Forest model trained on data from NASA's Kepler mission to classify celestial objects based on their observable characteristics. The model analyzes six key features to determine whether a signal is likely to be a confirmed exoplanet or a false positive.

The Kepler Space Telescope has discovered thousands of potential exoplanets, but distinguishing between true exoplanets and false positives (such as eclipsing binaries or stellar activity) remains a significant challenge. This tool helps automate part of that classification process.

Key Features
Interactive Web Interface: User-friendly Streamlit application for easy interaction
Real-time Classification: Instant predictions with confidence scores
Feature Analysis: Understand how different parameters influence classification
Educational Tool: Learn about exoplanet characteristics and detection methods
Random Forest Model: Robust machine learning algorithm for accurate classification
🚀 Quick Start
1. Live Demo
Try the live application here: https://ppruwais-nasa-exoplanet.streamlit.app

2. Local Installation
bash

Line Wrapping

Collapse
Copy
1
2
3
4
5
6
7
8
9
# Clone the repository
git clone https://github.com/PPRuwais/Nasa_Exoplanet.git
cd Nasa_Exoplanet

# Install required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
3. Quick Demo
bash

Line Wrapping

Collapse
Copy
1
2
# Run the app with default values
streamlit run streamlit_app.py
📁 Project Structure

Line Wrapping

Collapse
Copy
1
2
3
4
5
6
nasa_exoplanet/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── exoplanet_binary_classifier_rf.pkl  # Trained Random Forest model
├── feature_scaler_rf.pkl         # Feature scaler for preprocessing
└── README.md                     # This file
🔧 System Components
1. Streamlit Application (streamlit_app.py)
The main web application that provides an interface for exoplanet classification:

User Interface: Interactive form for inputting exoplanet parameters
Data Processing: Preprocesses input data using the feature scaler
Prediction Engine: Uses the Random Forest model for classification
Results Display: Shows predictions with confidence scores and detailed probabilities
python

Line Wrapping

Collapse
Copy
1
2
3
4
5
# Key features of the application
- Input fields for 6 key exoplanet parameters
- Real-time prediction with confidence scoring
- Detailed probability breakdown
- Debug information for technical users
2. Random Forest Model (exoplanet_binary_classifier_rf.pkl)
Pre-trained machine learning model for exoplanet classification:

Algorithm: Random Forest Classifier
Training Data: NASA Kepler mission data
Features: 6 key exoplanet parameters
Output: Classification as confirmed exoplanet, false positive, or candidate
3. Feature Scaler (feature_scaler_rf.pkl)
Preprocessing tool for normalizing input data:

Purpose: Standardizes input features to match training data distribution
Method: StandardScaler from scikit-learn
Integration: Automatically applied to user inputs before prediction
📊 Data Flow
User Input
Feature Validation
Data Scaling
Random Forest Model
Prediction Results
Confidence Calculation
Results Display
Model Files
Feature Scaler



🔬 Features and Their Significance
The model uses the following six features to make predictions:

1. Orbital Period (koi_period)
Description: The time it takes for a planet to complete one orbit around its star, measured in Earth days.
Relationship to Classification:
Very short periods (< 1 day) often indicate stellar activity rather than planetary transits
Extremely long periods (> 300 days) may be difficult to confirm with limited observation time
Confirmed exoplanets typically show consistent, periodic signals across multiple transit events
2. Planetary Radius (koi_prad)
Description: The size of the planet relative to Earth's radius.
Relationship to Classification:
Objects with radii larger than Jupiter (~11 Earth radii) are more likely to be false positives (e.g., small stars)
Confirmed exoplanets typically range from 0.5 to 10 Earth radii
Very small planets (< 0.5 Earth radii) may be difficult to distinguish from stellar noise
3. Equilibrium Temperature (koi_teq)
Description: The temperature of the planet if it were in thermal equilibrium, measured in Kelvin.
Relationship to Classification:
Extremely high temperatures (> 3000K) may indicate stellar rather than planetary objects
Confirmed exoplanets typically have equilibrium temperatures between 200K and 2000K
Temperature combined with radius helps distinguish between planets and brown dwarfs
4. Impact Parameter (koi_impact)
Description: The sky-projected distance between the center of the stellar disk and the center of the planet disk at conjunction, expressed as a fraction of the stellar radius.
Relationship to Classification:
Values near 0 indicate central transits, while values near 1 indicate grazing transits
Very high impact parameters (> 0.9) may produce V-shaped transit curves associated with eclipsing binaries (false positives)
Confirmed exoplanets typically show U-shaped transit curves with moderate impact parameters
5. Transit Duration (koi_duration)
Description: The length of the observed transit, measured in hours.
Relationship to Classification:
Duration should be consistent with the orbital period and stellar/planetary radii
Very long durations relative to the orbital period may indicate eclipsing binaries
Confirmed exoplanets typically show transit durations between 1-10 hours for close-in planets
6. Transit Depth (koi_depth)
Description: The fractional loss of light caused by the transit, measured in parts per million (ppm).
Relationship to Classification:
Depth should be proportional to the square of the planet-to-star radius ratio
Very deep transits (> 50,000 ppm) may indicate eclipsing binaries rather than planets
Confirmed exoplanets typically show transit depths between 100-10,000 ppm
🧠 How These Features Work Together
The Random Forest model analyzes these features collectively to identify patterns that distinguish confirmed exoplanets from false positives:

Consistency: Confirmed exoplanets show consistent relationships between all features (e.g., transit depth matches the planetary radius)
Physical Plausibility: The model checks if the combination of features represents a physically possible planetary system
Signal Characteristics: The shape and duration of the transit signal help differentiate between planetary transits and stellar eclipses
📸 Screenshots
<div align="center">
<img src="https://z-cdn-media.chatglm.cn/files/939117bd-4db8-4676-a5b4-d03211b8dee4_pasted_image_1759570155346.png?auth_key=1791107494-b364eac391b342309e0b5c9b7709b0c0-0-45453ef67c8efd7ea33dec58ad6c7630" alt="Exoplanet Classifier App" width="600">
</div>

🎯 Model Architecture
The Random Forest model uses the following architecture:

Algorithm: Random Forest Classifier
Number of Trees: 100
Max Depth: Optimized for the dataset
Features: 6 exoplanet parameters
Output: Binary classification with probability scores
📈 Performance Metrics
The model is evaluated using:

Accuracy: Overall classification accuracy
AUC Score: Area under ROC curve
Precision: True positives / (True positives + False positives)
Recall: True positives / (True positives + False negatives)
Confusion Matrix: Detailed classification breakdown
🔍 Usage Examples
Example 1: Classify a Known Exoplanet
python

Line Wrapping

Collapse
Copy
1
2
3
4
5
6
7
8
9
10
11
⌄
# Input values for Kepler-186f (a confirmed exoplanet)
input_data = {
    'koi_period': 129.9,      # Orbital period in days
    'koi_prad': 1.11,         # Planetary radius in Earth radii
    'koi_teq': 188,           # Equilibrium temperature in Kelvin
    'koi_impact': 0.5,        # Impact parameter (0-1)
    'koi_duration': 4.5,      # Transit duration in hours
    'koi_depth': 100          # Transit depth in ppm
}

# Expected output: CONFIRMED EXOPLANET with high confidence
Example 2: Classify a False Positive
python

Line Wrapping

Collapse
Copy
1
2
3
4
5
6
7
8
9
10
11
⌄
# Input values for a typical false positive
input_data = {
    'koi_period': 0.8,        # Very short period
    'koi_prad': 15.0,         # Very large radius
    'koi_teq': 3500,          # Very high temperature
    'koi_impact': 0.95,       # High impact parameter
    'koi_duration': 2.0,      # Short duration
    'koi_depth': 60000        # Very deep transit
}

# Expected output: FALSE POSITIVE with high confidence
🛠️ Configuration
Model Parameters
python

Line Wrapping

Collapse
Copy
1
2
3
4
5
6
# Model architecture
n_estimators = 100            # Number of trees in the forest
max_depth = None              # Maximum depth of the tree
min_samples_split = 2         # Minimum samples required to split
min_samples_leaf = 1          # Minimum samples required at leaf node
random_state = 42             # Random state for reproducibility
Prediction Parameters
python

Line Wrapping

Collapse
Copy
1
2
3
# Classification thresholds
confidence_threshold = 0.7    # Minimum confidence for definite classification
candidate_range = (0.3, 0.7) # Confidence range for uncertain predictions
📊 Model Files
The system uses the following model files:

exoplanet_binary_classifier_rf.pkl: Trained Random Forest model
feature_scaler_rf.pkl: Feature scaler for preprocessing
These files are generated during training and should not be modified manually.

🔬 Scientific Background
Exoplanet Detection via Transits
When a planet passes in front of its host star, it causes a small dip in the star's brightness. This "transit" method is one of the most successful techniques for discovering exoplanets.

Key Characteristics:

Depth: Typically 0.1-1% for Earth-sized planets
Duration: Hours to days depending on orbital period
Periodicity: Repeats every orbital period
Shape: U-shaped or V-shaped light curve
Classification Challenges
Distinguishing between true exoplanets and false positives is challenging due to:

Eclipsing Binaries: Two stars orbiting each other can mimic planetary transits
Stellar Activity: Starspots and other stellar phenomena can cause false signals
Instrument Noise: Measurement errors can create false positives
Data Quality: Limited observation time can affect classification confidence
🚨 Troubleshooting
Common Issues
Model Loading Error

Line Wrapping

Collapse
Copy
1
2
Error: Model or Scaler could not be loaded
Solution: Ensure model files are in the correct directory
Prediction Error

Line Wrapping

Collapse
Copy
1
2
Error: Please enter valid numerical values for all fields
Solution: Check that all input fields contain valid numbers
Module Not Found

Line Wrapping

Collapse
Copy
1
2
Error: No module named 'joblib'
Solution: Install all required packages from requirements.txt
Performance Tips
Use the latest version of Streamlit for best performance
Ensure a stable internet connection for the live demo
For local use, make sure your system meets the minimum requirements
🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
📚 References
NASA Exoplanet Archive
Kepler Mission
Transit Method
Random Forest Classifier
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📞 Contact
Your Name - @YourTwitter - your.email@example.com

Project Link: https://github.com/PPRuwais/Nasa_Exoplanet

Happy Exoplanet Hunting! 🪐✨

how do i add these into readme exactly like this wihtout becoming it like text copied
profile
GLM-4.6
The user wants to add the README content to their GitHub repository without it looking like plain text. They want it to be properly formatted with Markdown, including badges, sections, code blocks, and other formatting elements.



To add the README to your GitHub repository with proper formatting, follow these steps:

Method 1: Using GitHub Web Interface (Easiest)
Go to your repository on GitHub: https://github.com/PPRuwais/Nasa_Exoplanet
Create a new README file:
Click the "Add file" button
Select "Create new file"
Name the file README.md (the .md extension is important for Markdown formatting)
Copy and paste the content:
Copy the entire README content I provided
Paste it into the file editor
Commit the file:
Scroll down to the bottom
Add a commit message like "Add comprehensive README"
Click "Commit new file"
Method 2: Using GitHub Desktop or Git
Create a file named README.md in your local repository folder
Copy and paste the content into this file
Save the file
Commit and push to GitHub:

Line Wrapping

Collapse
Copy
1
2
3
git add README.md
git commit -m "Add comprehensive README"
git push origin main
Important Tips for Proper Formatting
Make sure to include all the Markdown formatting:
Headers (# ## ###)
Badges (the ![...] links)
Code blocks (the ``` sections)
Links (text)
Bold/italic text (text or text)
Don't modify the special formatting:
Keep the badge links exactly as they are
Preserve the code block formatting with ```
Keep the mermaid diagram syntax intact
The .md extension is crucial - it tells GitHub to render the file as Markdown rather than plain text
What It Will Look Like
Once properly added, your README will render with:

Professional badges at the top
Properly formatted headers and sections
Syntax-highlighted code blocks
Working links
The mermaid diagram will render as a flowchart
The screenshot will display as an image
Troubleshooting
If something doesn't render correctly:

Check that you named the file README.md (not just README)
Make sure you didn't accidentally remove any Markdown formatting
Verify that all code blocks start and end with ```
Check that all links have the proper text format
Would you like me to provide the README content again with any specific modifications for your project?



Send a Message
