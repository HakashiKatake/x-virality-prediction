import json
import joblib
import pandas as pd
import numpy as np
import os
import sys

def main():
    if not os.path.exists('models/metadata.json'):
        print("Models not found.")
        sys.exit(1)
        
    print("Loading model artifacts...")
    model = joblib.load('models/virality_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
        
    features_list = metadata['features']
    prediction_threshold = metadata['prediction_threshold']
    
    # Create sample inputs
    raw_inputs = {
        'char_count': 120,
        'word_count': 22,
        'hashtag_count': 2,
        'mention_count': 1,
        'url_count': 1,
        'hour': 14,
        'day_of_week': 2,
        'month': 5,
        'year': 2021,
        'is_weekend': 0,
        
        # Author profile (raw)
        'followers': 25000,
        'following': 1200,
        'is_verified': 0,
        'account_age_days': 1800,
        
        # Author history (raw)
        'prev_tweet_count': 450,
        'prev_avg_engagement': 1250.0,
        'prev_viral_count': 52
    }
    
    # Feature Engineering (mirroring notebook exactly)
    raw_inputs['log_followers'] = np.log1p(raw_inputs['followers'])
    raw_inputs['log_following'] = np.log1p(raw_inputs['following'])
    raw_inputs['follower_following_ratio'] = raw_inputs['followers'] / (raw_inputs['following'] + 1)
    
    # Historical calculations
    raw_inputs['historical_viral_rate'] = raw_inputs['prev_viral_count'] / raw_inputs['prev_tweet_count'] if raw_inputs['prev_tweet_count'] > 0 else 0.0
    
    # Create dataframe
    df = pd.DataFrame([raw_inputs])
    
    # Ensure all required features are present and strictly ordered
    for f in features_list:
        if f not in df.columns:
            df[f] = 0.0 # Default fallback
            
    df = df[features_list]
    
    print("\nFinal Feature Vector:")
    print(df.iloc[0])
    
    X_scaled = scaler.transform(df)
    prob = model.predict_proba(X_scaled)[:, 1][0]
    
    print("\n--- INFERENCE RESULT ---")
    print(f"Model: {metadata['model']}")
    print(f"Prediction Threshold: {prediction_threshold:.4f}")
    print(f"Predicted Probability: {prob:.4f}")
    if prob >= prediction_threshold:
        print("Prediction: VIRAL")
    else:
        print("Prediction: NOT VIRAL")

if __name__ == "__main__":
    main()
