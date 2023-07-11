#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models


# In[2]:


# opening and creating new .txt file
with open(
    "D:/DATA SCIENCE/EXCELR/PROJECT/PROJECT 2/emails.txt", 'r') as r, open(
        "D:/DATA SCIENCE/EXCELR/PROJECT/PROJECT 2/output.txt", 'w') as o:
      
    for line in r:
        #strip() function
        if line.strip():
            o.write(line)
  
f = open("D:/DATA SCIENCE/EXCELR/PROJECT/PROJECT 2/output.txt", "r")
print("New text file:\n")


# In[3]:


get_ipython().system('pip install imbalanced-learn')


# In[4]:


data = pd.read_csv("D:\DATA SCIENCE\EXCELR\PROJECT\PROJECT 2\output.txt")

data.to_csv('output.csv', index = None)
data


# In[5]:


data = data.drop(['Unnamed: 0', 'filename' ,'Message-ID'], axis = 1)
data


# In[6]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[7]:


data['Class'] = encoder.fit_transform(data['Class'])


# In[8]:


data


# In[9]:


data.duplicated().sum()


# In[10]:


data = data.drop_duplicates(keep = 'first')
data


# In[11]:


from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['content'], data['Class'].values, test_size=0.30)


# In[12]:


from collections import Counter

Counter(y_train)


# In[13]:


Counter({1: 31241, 0: 2412})


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)


# In[15]:


tf_wb= TfidfVectorizer()
# X_tf = tf_wb.fit_transform(data['content'])
# X_tf = X_tf.toarray()


# In[16]:


X_tf = tf_wb.fit_transform(data['content'])


# In[17]:


# X_tf = X_tf.toarray()


# In[18]:


X_train_tf = vectorizer.transform(X_train)


# In[19]:


# X_train_tf = X_train_tf.toarray()


# In[20]:


X_test_tf = vectorizer.transform(X_test)


# In[21]:


# X_test_tf = X_test_tf.toarray()


# In[22]:


ROS = RandomOverSampler(sampling_strategy=1)


# In[23]:


X_train_ros, y_train_ros = ROS.fit_resample(X_train_tf, y_train)


# In[24]:


Counter(y_train_ros)


# In[25]:


X_train_ros


# In[26]:


y_train_ros


# In[ ]:




