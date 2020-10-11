import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from sklearn.svm import SVC, LinearSVC


df = pd.read_csv('/content/drive/My Drive/wine_review/file1.csv')
x = df.iloc[:,2].values.astype('U') # Message column as input
y = df.iloc[:,-1].values.astype('U') # Label column as output
st.title("Wine Review Classifier")
st.subheader('Count Vectorizer')
st.write('This project is based on LinearSVC Classifier')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30, random_state = 0)
text_model = Pipeline([('bow',CountVectorizer(ngram_range = (1, 3))),('model',LinearSVC())])
text_model.fit(x_train,y_train)
message = st.text_area("Enter Text","Type Here ..")
op = text_model.predict([message])
if st.button("Predict"):
  st.title(op)
