# standard libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.simplefilter('ignore')
import gc

# visualisation
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

# dataprep
from dataprep.clean import clean_text

# tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback
import keras_tuner as kt
import pickle

# load data
df = pd.read_csv('E:\Papers\mentalhealth\mentalhealthdataset\mental_health.csv')
# drop NULLs
df.dropna(inplace=True)
# view
df.head()

"""Cleaning Text"""

df = clean_text(df,'text')

"""Label Distribution"""

fig = plt.figure(figsize=(6, 6))
plt.title("label distribution")
sns.countplot(x=df['label'],palette='pastel')
fig.tight_layout()
plt.show()

"""Word to Vector
Count Vectorizer
"""

# vectorize
vectorizer = CountVectorizer(min_df=0, lowercase=True)
vectorizer.fit(df['text'])
# vectorizer.vocabulary_

# feature engineering
sentences = df['text'].values
y = df['label'].values

# train-test split [80-20]
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=42)

# vectorize (again!)
vectorizer.fit(sentences_train)
x_train = vectorizer.transform(sentences_train)
x_test  = vectorizer.transform(sentences_test)

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,f1_score,r2_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

model_list={"Decision Tree Classifier":DecisionTreeClassifier(),"Logistic Regression":LogisticRegression(),"SVC":SVC()}

"""Result DataFrame to store Scores, Accuracy, Error"""

result=pd.DataFrame(columns=['Name of Model','accuracy','f1_score',"r2score"])
result

"""Trainning Model on Training Dataset"""

def train_model(models):
    for model_name,model in models.items():
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
        result.loc[len(result.index)]=[model_name,accuracy_score(pred,y_test),
        f1_score(pred,y_test),
        r2_score(pred,y_test),]
        cm = confusion_matrix(y_test, pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.show()

train_model(model_list)

"""Result-->Logistic Regression with highest accuracy of 94%"""
result
