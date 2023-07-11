#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
# %matplotlib inline


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


import pandas
dataset = pd.read_csv("D:/DATA SCIENCE/EXCELR/PROJECT/PROJECT 2/output.txt")


# In[4]:


#convert the list into datafrme
from pandas import DataFrame
data = DataFrame (dataset,columns=['content','Class'])
data


# In[5]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[6]:


data['Class'] = encoder.fit_transform(data['Class'])
data


# In[7]:


data = data.drop_duplicates(keep = 'first')
data


# In[8]:


import re #regular expression
import string

def clean_text(text):
    '''
    Make text lowercase, remove text in square brackets,
    remove punctuation and remove words containing numbers.
    '''
    text = text.lower()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n*\r\n*', '', text)
    text = re.sub('\t*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    
    return text

clean = lambda x: clean_text(x)

data['content'] = data['content'].apply(clean)
data['content']


# In[9]:


import nltk

nltk.download('stopwords')

data = data.iloc[:, 0:2]
data

stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

data['content'] = data['content'].apply(lambda x: stemming(x))

data['content']

content = list(data['content'])
content


# In[10]:


cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(content).toarray()
y = data.iloc[:, -1].values

## Balancing the imbalanced data
from imblearn.over_sampling import SMOTE
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[11]:


from keras.models import Sequential
from keras import layers
import numpy
import pandas as pd
from keras.layers import Dense

input_dim = X_train.shape  # Number of features
print(input_dim)


# In[12]:


# create model
model = Sequential()
model.add(layers.Dense(1, input_dim= 3000, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['mean_absolute_percentage_error'])

model.summary


# In[13]:


# Fit the model
history = model.fit(X_train, y_train, validation_split=0.33, epochs=20, batch_size=10)


# In[14]:


model.evaluate(X_test, y_test)


# In[15]:


# evaluate the model
scores = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[16]:


y_predicted = model.predict(X_test)


# In[17]:


y_predicted = y_predicted.flatten()
y_predicted


# In[18]:


import numpy as np

y_predicted = np.where(y_predicted > 0.5, 1, 0)
y_predicted


# In[19]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_predicted)
cm 


# In[20]:


acc = accuracy_score(y_test, y_predicted)
acc


# In[21]:


from matplotlib import pyplot as plt
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[22]:


print(classification_report(y_test, y_predicted))


# In[26]:


import pickle
pickle.dump(model, open('P242_CNN.h5', 'wb'))
pickle.dump(model, open('P242CNN_cv.sav','wb'))


# In[ ]:




