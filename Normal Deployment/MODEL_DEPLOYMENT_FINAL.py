#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install streamlit

import pickle
import numpy as np
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

st.title('NLP DEPLOYMENT')

to_predict_list=[]


# In[2]:


Email = st.text_input("Enter content")
Email = "i can't believe i've been reduced to pathetic begging and empty threats.  you're killing me."
to_predict_list.append(Email)


# In[3]:


import re #regular expression
import string

def clean_text(text):
    '''
    Make text lowercase, remove text in square brackets,
    remove punctuation and remove words containing numbers.
    '''
    text = text.lower()
    text = text.strip()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
#     text = re.sub('\[.*?\]', '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\w*\d\w*', '', text)
#     text = re.sub('\n*\r\n*', '', text)
#     text = re.sub('\t*', '', text)
#     text = re.sub("[0-9" "]+"," ",text)
#     text = re.sub('[‘’“”…]', '', text)
    
    return text

Email = clean_text(Email)
print(Email)


# In[4]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop = stopwords.words('english')
stop.append('in')
stop.append('you')


# In[5]:


stemmer = PorterStemmer()
def stemming(data):
    text = [word_tokenize(word) for word in data]
    text = [word for word in data if not word in stop]

    
    text = [stemmer.stem(word) for word in text]
    return text


# In[6]:


text = word_tokenize(Email)
text = [word for word in text if not word in stop]


# In[7]:


Email = stemming(Email)


# In[8]:


loaded_cv=pickle.load(open("P242_cv.sav","rb"))
loaded_model=pickle.load(open("P242.sav","rb"))


# In[9]:


Email=loaded_cv.transform(Email)


# In[10]:


result=loaded_model.predict(Email)
print(result[0])
content = ""
if result[0]==0:
    content ="Abusive"
elif result[0]==1:
    content ="Non-Abusive"


# In[11]:


if st.button("Predict"):
    st.success(f"These parameters belongs to {content}")


# In[ ]:





# In[ ]:




