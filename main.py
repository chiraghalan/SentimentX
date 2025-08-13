# streamlit_sentiment_app_with_confidence.py

import streamlit as st
import joblib
import re
from sklearn.linear_model import LogisticRegression

# ===== 1️⃣ Load pre-trained model and TF-IDF =====
voting_clf = joblib.load("voting_sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ===== 2️⃣ Preprocessing function =====
def preprocess_tweet(tweet):
    tweet = tweet.lower()  # Lowercase
    tweet = re.sub(r'http\S+|www\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#', '', tweet)  # Remove hashtags symbol
    tweet = re.sub(r'\s+', ' ', tweet).strip()  # Remove extra spaces
    return tweet

# ===== 3️⃣ Prediction function with confidence =====
def predict_sentiment(tweet):
    tweet_clean = preprocess_tweet(tweet)
    X = tfidf.transform([tweet_clean])

    # Hard voting prediction
    label = voting_clf.predict(X)[0]

    # Confidence from Logistic Regression
    # Assumes LR is the first estimator in the voting ensemble
    lr_model = voting_clf.estimators_[0]
    prob_positive = lr_model.predict_proba(X)[0][1]  # probability of positive
    confidence = prob_positive if label == 1 else 1 - prob_positive

    return "positive" if label == 1 else "negative", confidence

# ===== 4️⃣ Streamlit UI =====
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Tweet Sentiment Analyzer")
st.write("Analyze the sentiment of your tweets in real-time!")

# Sidebar info
st.sidebar.header("About")
st.sidebar.write(
    """
    This app predicts whether a tweet is **positive** or **negative**.
    It uses a trained ensemble model (Voting Classifier) on the Sentiment140 dataset.
    The confidence/probability is estimated from Logistic Regression.
    """
)

# Text input
user_input = st.text_area("Enter your tweet here:", "", height=150)

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a tweet to analyze.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        percent = confidence * 100

        if sentiment == "positive":
            st.success(f"✅ Predicted sentiment: **{sentiment.upper()}** ({percent:.2f}% confidence)")
            st.progress(int(percent))
        else:
            st.error(f"❌ Predicted sentiment: **{sentiment.upper()}** ({percent:.2f}% confidence)")
            st.progress(int(percent))

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using **Python & Streamlit**")
