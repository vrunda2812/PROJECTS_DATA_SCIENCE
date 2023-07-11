# -*- coding: utf-8 -*-
"""MODEL_DEPLOYMENT_CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C_gXAMKWeKv3hY2SvJ-u3MllM-FDx6mW
"""

# !pip install streamlit

import pickle
import numpy as np
import streamlit as st
from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import CountVectorizer

st.title('NLP DEPLOYMENT')

Email = st.text_input("Enter content")

import re #regular expression
import string
stemmer = PorterStemmer()
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

def clean_text(text):
    '''
    Make text lowercase, remove text in square brackets,
    remove punctuation and remove words containing numbers.
    '''
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = text.strip()
#     text = text.split()
#     text = [x for x in text if x not in set(stop)]  
#     text = [stemmer.stem(word) for word in text]
#     text = " ".join(text)
    return text

Email=clean_text(Email)

Email=[Email]

vectorizer=pickle.load(open("P242_CNN_CountV.sav","rb"))

from tensorflow.keras.models import load_model
model = load_model('P242_cnn_model.h5')

Email=vectorizer.transform(Email)

result=model.predict(Email)

import numpy as np
res=result[0][0]
result_type = np.where(res > 0.5, 1, 0)

content = ""
if result_type==0:
    content ="Abusive"
elif result_type==1:
    content ="Non-Abusive"
    
if st.button("Predict"):
    st.success(f" Email content  is: {content}")


