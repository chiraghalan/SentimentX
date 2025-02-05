
# Twitter Sentiment Analysis

## 📌 Project Overview
This project performs sentiment analysis on tweets using machine learning models. The goal is to classify tweets into positive and negative sentiments based on their text. The workflow includes:
- **Data Preprocessing & Cleaning**
- **Exploratory Data Analysis (EDA) with Visualizations**
- **Feature Engineering (Text Vectorization)**
- **Machine Learning Model Training & Evaluation**
- **Comparison of Different Models**
- **Results & Performance Analysis**

The best-performing model, **Logistic Regression**, achieved **92% accuracy** on the test dataset, making it the most suitable choice for sentiment classification.

---
## 📂 Dataset Details
- **Source**: Kaggle (Mental Health Dataset)
- **File Used**: `mental_health.csv`
- **Number of Records**: Varies
- **Features**:
  - `text`: Tweet content expressing opinions or emotions.
  - `label`: Sentiment classification (positive/negative).

### **Preprocessing Steps**:
- Removed null values.
- Cleaned text data using `dataprep.clean.clean_text` (removed stopwords, punctuation, and special characters).
- Lowercased all words for uniformity.
- Applied tokenization and text normalization.

---
## 🛠️ Exploratory Data Analysis (EDA)
EDA helps in understanding the data distribution and patterns. We performed:
- **Label Distribution Analysis** (Bar Chart - Seaborn Visualization)
- **Word Frequency Analysis** (Most Common Words in Tweets)
- **Word Clouds for Positive & Negative Sentiments**
- **Text Length Distribution** (Histogram)
- **Correlation Heatmaps** (Feature Dependencies)

---
## 🚀 Machine Learning Models Used
We trained the following models and compared their performance:
1. **Logistic Regression** ✅ (Best Performance - **92% Accuracy**)
2. **Support Vector Machine (SVM)**
3. **Decision Tree Classifier**

### **Model Training Approach**:
- **Feature Engineering**: Text data was converted into numerical format using **CountVectorizer**.
- **Train-Test Split**: 80-20 ratio was used for training and testing.
- **Model Training**: Implemented a function `train_model()` to train models efficiently.
- **Hyperparameter Tuning**: Adjusted key parameters for improved accuracy.
- **Cross-Validation**: K-Fold cross-validation was used to prevent overfitting.

### **Performance Metrics Used**:
- **Accuracy**
- **F1 Score** (Harmonic mean of precision & recall)
- **R2 Score** (Measures goodness of fit)
- **Confusion Matrix Visualization** (For error analysis)

---
## 🎯 Results & Model Performance
| Model | Accuracy | F1 Score | R2 Score |
|--------|----------|----------|----------|
| **Logistic Regression** | **92%** | **High** | **High** |
| Support Vector Machine | 89% | Medium | Medium |
| Decision Tree Classifier | 85% | Low | Low |

The **Logistic Regression** model outperformed others, making it the best choice for Twitter sentiment analysis.

---
## 💡 Installation & Setup
### **1️⃣ Clone Repository**
```bash
git clone https://github.com/chiraghalan/Mental-health-sentiment-classifier
cd Mental-health-sentiment-classifier
```

### **2️⃣ Install Dependencies**
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Analysis**
```bash
python sentimentclassifier.py
```

---
## 📝 Usage Instructions
1. **Load the Dataset**
   - The script automatically loads `mental_health.csv`.
2. **Perform EDA**
   - Run the notebook to visualize data insights.
3. **Train Models**
   - Use `train_model()` to train ML models.
4. **Evaluate Performance**
   - Confusion matrix and accuracy scores are displayed.
5. **Predict Sentiments**
   - Provide new tweets to predict their sentiment.

---
## 🔥 Future Improvements
To further enhance this project, we plan to:
- **Integrate Live Twitter Scraping** (Using Tweepy API)
- **Use Deep Learning Models** (LSTMs, BERT for NLP tasks)
- **Deploy a Web App** (Flask/Streamlit for real-time predictions)
- **Improve Model Explainability** (SHAP, LIME for interpretability)



---
## 🤝 Contributing
Contributions are welcome! If you wish to improve the project, follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

---
## 📞 Contact
For queries, reach out to: **halanchirag@gmail.com**
