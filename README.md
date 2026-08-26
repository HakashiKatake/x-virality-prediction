# X Post Virality Predictor

This is a live demonstration application for the X Post Virality Prediction ML project. It provides an interactive interface for evaluating a completed Machine Learning model's prediction of a post's virality based on its text, temporal characteristics, author profile, and historical engagement.

## Project Context
This application uses the final, hyperparameter-tuned `HistGradientBoostingClassifier` trained on the COVID-19 All Vaccines Tweets dataset (collected 2020-2022). The model predicts if a tweet will fall into the top 10% of engagement (the "viral" class). The decision threshold used in this app was strictly optimized on a validation set to maximize the F1-score due to the heavily imbalanced target class.

## Files
- `app.py`: The Streamlit web application.
- `models/virality_model.joblib`: The trained HistGradientBoosting model.
- `models/scaler.joblib`: The StandardScaler fitted solely on training data.
- `models/metadata.json`: Feature schemas and model hyperparameters.
- `test_inference.py`: A CLI testing script to verify inference pipeline functionality.

## Installation
Requires Python 3.8+ (preferably Python 3.10+).

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run app.py
```

## Testing the Model Inference
```bash
python3 test_inference.py
```

## Limitations & Academic Disclaimer
Academic demonstration only. The model was trained on COVID-19 vaccine-related tweets collected during 2020–2022 and should not be interpreted as a production predictor for current X recommendation or virality. The model accurately detects structural correlations from its dataset but lacks awareness of current world events and the current real-world X algorithm.
