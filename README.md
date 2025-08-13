😊 Tweet Sentiment Analyzer

Analyze the mood of tweets instantly!
An end-to-end sentiment analysis system that predicts whether a tweet is positive or negative using a trained ensemble machine learning model. Built with Python, scikit-learn, and Streamlit, this project includes preprocessing, model training, and a sleek, interactive web interface.

✨ Key Features
Smart Preprocessing: Cleans tweets by removing URLs, mentions, hashtags, and extra spaces.

Powerful Ensemble Modeling: Combines Logistic Regression and Linear SVM for better predictions.

Real-Time Sentiment: Get instant feedback on any tweet you enter.

Confidence Score: Shows how strongly the model believes in its prediction.

Interactive UI: Color-coded sentiment display and visual confidence bar.

Reproducible: Fully saved models and vectorizers for easy reuse.

🎯 Why This Project is Cool
End-to-End Pipeline: From raw tweets → preprocessing → model → live web app.

Data-Driven Insights: Evaluated on Accuracy, Precision, Recall, F1-score, and ROC-AUC.

Fast & Efficient: Optimized for speed without losing prediction quality.



🖼 Demo Preview



🚀 How to Use
Visit the Streamlit app .

Type your tweet in the text box.

Click Predict → see the sentiment along with a confidence indicator.

Get insights instantly with a clean and interactive interface.

📂 Project Structure
Streamlit App: User interface for real-time tweet prediction.

Saved Models: Pre-trained ensemble and TF-IDF vectorizer for immediate use.

Training Scripts: Optional scripts to retrain the model from the dataset.

Requirements: All necessary dependencies listed for reproducibility.

🔮 Future Improvements
Add neutral sentiment for more nuanced analysis.

Enable multi-language support for global tweets.

Add batch processing to analyze multiple tweets at once.

Deploy a fully public app on Streamlit Cloud.

💻 Technologies Used
Python 3.11 – Core programming language

Pandas & NumPy – Data handling

scikit-learn – Machine learning models and metrics

Streamlit – Web app interface

Joblib – Model serialization and loading
