import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
from datetime import datetime

st.set_page_config(
    page_title="X Post Virality Predictor",
    page_icon="🚀",
    layout="wide"
)

# ----------------------------------------------------
# 1. LOAD ARTIFACTS
# ----------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists('models/metadata.json'):
        return None, None, None
    
    model = joblib.load('models/virality_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
        
    return model, scaler, metadata

model, scaler, metadata = load_artifacts()

# ----------------------------------------------------
# 2. HELPER FUNCTIONS
# ----------------------------------------------------
def extract_text_features(text):
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    hashtag_count = len(re.findall(r'#\w+', text))
    mention_count = len(re.findall(r'@\w+', text))
    url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    return char_count, word_count, hashtag_count, mention_count, url_count

def generate_features(text, post_date, post_time, followers, following, verified, 
                      account_age_days, prev_tweets, prev_avg_eng, prev_viral_count):
    
    char_count, word_count, hashtag_count, mention_count, url_count = extract_text_features(text)
    
    # Temporal
    hour = post_time.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_of_week = post_date.weekday()
    month = post_date.month
    year = post_date.year
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Author Profile Engineering
    log_followers = np.log1p(followers)
    log_following = np.log1p(following)
    follower_following_ratio = followers / (following + 1)
    
    # Historical Performance Engineering
    historical_viral_rate = prev_viral_count / prev_tweets if prev_tweets > 0 else 0.0
    
    # Assemble raw dict
    raw_dict = {
        'char_count': char_count,
        'word_count': word_count,
        'hashtag_count': hashtag_count,
        'mention_count': mention_count,
        'url_count': url_count,
        'hour': hour,
                'day_of_week': day_of_week,
        'month': month,
        'year': year,
        'is_weekend': is_weekend,
        'user_followers': followers, # Notebook uses user_followers
        'user_friends': following, # Notebook uses user_friends
        'is_verified': int(verified),
        'account_age_days': account_age_days,
        'log_followers': log_followers,
        'log_following': log_following,
        'follower_following_ratio': follower_following_ratio,
        'prev_tweet_count': prev_tweets,
        'prev_avg_engagement': prev_avg_eng,
        'prev_viral_count': prev_viral_count,
        'historical_viral_rate': historical_viral_rate
    }
    
    return raw_dict

# ----------------------------------------------------
# 3. UI LAYOUT
# ----------------------------------------------------
st.title("🚀 X Post Virality Predictor")
st.markdown("### *Predict whether a post is likely to exceed the project's historical virality threshold.*")
st.caption("Academic demonstration only. The model was trained on COVID-19 vaccine-related tweets collected during 2020–2022 and should not be interpreted as a production predictor for current X recommendation or virality.")

if model is None:
    st.error("Model artifacts not found! Please ensure `models/virality_model.joblib` and `models/metadata.json` exist.")
    st.stop()

# --- DEMO EXAMPLES ---
st.markdown("---")
col_d1, col_d2, col_d3 = st.columns(3)
if col_d1.button("Load Example 1: Small/New Account"):
    st.session_state.demo = 1
if col_d2.button("Load Example 2: Established Account"):
    st.session_state.demo = 2
if col_d3.button("Load Example 3: Large/Influential Account"):
    st.session_state.demo = 3

demo = st.session_state.get('demo', 0)

# Defaults based on demo mode
def_text = ""
def_followers = 0
def_following = 0
def_verified = False
def_age = 30
def_prev_tweets = 0
def_prev_avg_eng = 0.0
def_prev_viral = 0

if demo == 1:
    def_text = "Just joined X! Looking forward to learning and sharing. #firstpost"
    def_followers = 15
    def_following = 45
    def_verified = False
    def_age = 2
    def_prev_tweets = 0
    def_prev_avg_eng = 0.0
    def_prev_viral = 0
elif demo == 2:
    def_text = "I've been working on this data science project for 3 months, and I finally published the results today! Check it out here: https://example.com/project"
    def_followers = 2500
    def_following = 800
    def_verified = False
    def_age = 1500
    def_prev_tweets = 120
    def_prev_avg_eng = 45.5
    def_prev_viral = 2
elif demo == 3:
    def_text = "BREAKING: Huge updates coming to our platform tomorrow. We're redesigning the entire core experience. What feature are you hoping for most? 👇"
    def_followers = 450000
    def_following = 1500
    def_verified = True
    def_age = 3200
    def_prev_tweets = 4500
    def_prev_avg_eng = 2500.0
    def_prev_viral = 1850

with st.form("prediction_form"):
    # SECTION A
    st.header("A. Write Your Post")
    post_text = st.text_area("Post Text", value=def_text, height=100, 
                             placeholder="Just launched something we've been working on for months 🚀 What do you think?")
    
    char_c, word_c, hash_c, ment_c, url_c = extract_text_features(post_text)
    st.caption(f"**Post statistics:** Characters: {char_c} | Words: {word_c} | Hashtags: {hash_c} | Mentions: {ment_c} | URLs: {url_c}")
    
    col1, col2 = st.columns(2)
    with col1:
        # SECTION B
        st.header("B. Author Profile")
        followers = st.number_input("Followers", min_value=0, value=def_followers, step=10)
        following = st.number_input("Following", min_value=0, value=def_following, step=10)
        account_age = st.number_input("Account Age (days)", min_value=0, value=def_age, step=30)
        verified = st.checkbox("Verified Account", value=def_verified)
        
        # SECTION D
        st.header("D. Temporal Information")
        post_date = st.date_input("Posting Date", value=datetime.now().date())
        post_time = st.time_input("Posting Time", value=datetime.now().time())

    with col2:
        # SECTION C
        st.header("C. Author Historical Performance")
        st.info("These values represent information available from the author's previous posts *before* the new post.")
        prev_tweets = st.number_input("Previous Tweets Count", min_value=0, value=def_prev_tweets, step=5)
        prev_avg_eng = st.number_input("Average Previous Engagement", min_value=0.0, value=def_prev_avg_eng, step=1.0)
        prev_viral = st.number_input("Previous Viral Tweets", min_value=0, value=def_prev_viral, step=1)

    submit_button = st.form_submit_button(label="PREDICT VIRALITY", use_container_width=True)

# ----------------------------------------------------
# 4. PREDICTION LOGIC
# ----------------------------------------------------
if submit_button:
    # Validation
    if len(post_text.strip()) == 0:
        st.error("Please enter a post before predicting.")
    elif prev_viral > prev_tweets:
        st.error("Error: Previous viral tweets cannot exceed total previous tweets.")
    else:
        # Generate raw dict
        raw_dict = generate_features(
            post_text, post_date, post_time, followers, following, verified, 
            account_age, prev_tweets, prev_avg_eng, prev_viral
        )
        
        # Build dataframe using exact model features
        features_list = metadata['features']
        df = pd.DataFrame([raw_dict])
        
        # Add missing features with default 0 if any
        for f in features_list:
            if f not in df.columns:
                df[f] = 0.0
                
        # Reorder to match notebook strictly
        input_df = df[features_list]
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prob = model.predict_proba(input_scaled)[0, 1]
        threshold = metadata['prediction_threshold']
        
        is_viral = prob >= threshold
        
        st.markdown("---")
        st.markdown("## MODEL PREDICTION")
        
        res_col, score_col = st.columns([1, 2])
        
        with res_col:
            if is_viral:
                st.success("### 🔥 LIKELY VIRAL")
            else:
                st.error("### 📉 NOT VIRAL")
                
        with score_col:
            st.markdown(f"**Model Predicted Virality Score:** `{prob*100:.1f}%`")
            st.progress(float(prob))
            st.markdown(f"**Decision Threshold:** `{threshold*100:.1f}%`")
            
        st.markdown(f"**Model:** `{metadata['model']}`")
        if is_viral:
            st.markdown("Because the score is above the validation-optimized threshold, the model classifies this post as viral.")
        else:
            st.markdown("Because the score is below the validation-optimized threshold, the model classifies this post as not viral.")
            
        st.caption("The score represents the model's predicted probability for the viral class. It is not a guaranteed real-world probability.")
        
        # Output Summary
        with st.expander("Show Input Summary"):
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown("**POST**")
                st.write(f"- Characters: {raw_dict['char_count']}")
                st.write(f"- Words: {raw_dict['word_count']}")
                st.write(f"- Hashtags: {raw_dict['hashtag_count']}")
                st.write(f"- Mentions: {raw_dict['mention_count']}")
                st.write(f"- URLs: {raw_dict['url_count']}")
            with s2:
                st.markdown("**AUTHOR**")
                st.write(f"- Followers: {followers}")
                st.write(f"- Following: {following}")
                st.write(f"- Verified: {'Yes' if verified else 'No'}")
                st.write(f"- Account age: {account_age} days")
            with s3:
                st.markdown("**HISTORY**")
                st.write(f"- Previous tweets: {prev_tweets}")
                st.write(f"- Average engagement: {prev_avg_eng:.1f}")
                st.write(f"- Previous viral tweets: {prev_viral}")
                st.write(f"- Historical viral rate: {raw_dict['historical_viral_rate']*100:.1f}%")

with st.expander("How does this model work?"):
    st.markdown("""
    1. **Text parsing**: Post text and temporal metadata are converted into numerical features (counts, hour trigonometry).
    2. **Author inclusion**: Author profile information (followers, age) is incorporated.
    3. **Historical momentum**: Historical author behavior is incorporated using rigorous chronological aggregation (no future information leaking).
    4. **ML processing**: The trained Machine Learning model estimates the viral-class score.
    5. **Classification**: The score is compared against the validation-optimized threshold.
    
    *Note: In the original project analysis, the model's strongest signals were historical engagement and follower-related features. Structural text elements (hashtags, character counts) were shown to be extremely weak predictors independently.*
    """)
