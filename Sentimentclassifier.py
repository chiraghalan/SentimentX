# ============================== #
# 📦 Library Imports
# ============================== #

# Standard
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
import gc
import pickle

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Text Preprocessing
from dataprep.clean import clean_text
from sklearn.feature_extraction.text import CountVectorizer

# Model Selection & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Deep Learning (optional)
import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback
import keras_tuner as kt


# ============================== #
# 📥 Load and Preprocess Dataset
# ============================== #

# Load CSV
df = pd.read_csv(r'E:\Papers\mentalhealth\mentalhealthdataset\mental_health.csv')

# Drop rows with nulls
df.dropna(inplace=True)

# Clean text column
df = clean_text(df, 'text')

# Preview dataset
print("📄 Dataset Preview:")
print(df.head())


# ============================== #
# 📊 Label Distribution
# ============================== #

plt.figure(figsize=(6, 6))
sns.countplot(x='label', data=df, palette='pastel')
plt.title("Label Distribution")
plt.tight_layout()
plt.show()


# ============================== #
# 🧠 Feature Engineering
# ============================== #

# Features and labels
sentences = df['text'].values
labels = df['label'].values

# Train-test split (80-20)
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.2, random_state=42, stratify=labels
)

# Vectorization
vectorizer = CountVectorizer(lowercase=True, min_df=0)
vectorizer.fit(sentences_train)
x_train = vectorizer.transform(sentences_train)
x_test = vectorizer.transform(sentences_test)


# ============================== #
# 🧪 Model Evaluation Framework
# ============================== #

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVC": SVC()
}

# Result table
results_df = pd.DataFrame(columns=["Model", "Accuracy", "F1 Score", "R² Score"])


# ============================== #
# 🔁 Train and Evaluate Models
# ============================== #

def evaluate_models(model_dict):
    global results_df
    for name, model in model_dict.items():
        print(f"\n🔧 Training: {name}")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')  # weighted for imbalance
        r2 = r2_score(y_test.astype(int), predictions.astype(int))

        results_df.loc[len(results_df)] = [name, acc, f1, r2]

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.title(f"Confusion Matrix: {name}")
        plt.show()

    return results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)


# Train models
final_results = evaluate_models(models)

# Display Results
print("\n📊 Final Model Comparison:")
print(final_results)


# ============================== #
# 💾 Optional: Save Best Model
# ============================== #

# Save best performing model (Logistic Regression)
best_model_name = final_results.iloc[0]['Model']
best_model = models[best_model_name]

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n✅ Saved best model: {best_model_name}")
