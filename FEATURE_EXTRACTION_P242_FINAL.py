#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES
# 

# In[1]:


# !pip install spacy
# !pip install wordcloud


# In[2]:


# ##Execute below command through anaconda command prompt
# !python -m spacy download en


# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
# %matplotlib inline


# In[4]:


# removing the blank rows


# In[5]:


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


# In[6]:


f


# In[7]:


import pandas
dataset = pd.read_csv("D:/DATA SCIENCE/EXCELR/PROJECT/PROJECT 2/output.txt")


# In[8]:


dataset


# In[9]:


dataset.to_csv("D:/DATA SCIENCE/EXCELR/PROJECT/PROJECT 2/output.csv")


# In[10]:


data=pd.read_csv("D:/DATA SCIENCE/EXCELR/PROJECT/PROJECT 2/output.csv")


# In[11]:


data.head()


# In[12]:


data['Class'].value_counts()


# In[13]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[14]:


data['Class'] = encoder.fit_transform(data['Class'])
data


# In[15]:


data.duplicated().sum()


# In[16]:


data = data.drop_duplicates(keep = 'first')
data


# In[17]:


data['content'].astype('string')


# In[18]:


data['content']


# In[19]:


data1=data['content'].to_string()


# In[20]:


data1


# In[21]:


data1 = data1.replace("\\r", "")
data1 = data1.replace("\\n", "")
data1 = data1.replace("\n", "")
data1 = data1.replace("\'", "")
data1 = data1.replace("\\t", "")

data1


# In[22]:


no_punc_text = data1.translate(str.maketrans('', '', string.punctuation)) #with arguments (x, y, z) where 'x' and 'y'
no_punc_text


# In[23]:


no_punc_text=" ".join(no_punc_text.split())##removing extra spaces
no_punc_text


# In[24]:


no_punc_text = ''.join([i for i in no_punc_text if not i.isdigit()])##removing digits
no_punc_text


# In[25]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[26]:


from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])


# In[27]:


len(text_tokens)


# In[28]:


#Remove stopwords
from nltk.corpus import stopwords

my_stop_words = stopwords.words('english')
my_stop_words.append('the')
my_stop_words.append('Its')

no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens[0:40])


# In[29]:


#Normalize the data
lower_words = [x.lower() for x in no_stop_tokens]
print(lower_words[0:25])


# In[30]:


#Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:40])


# In[31]:


nlp = spacy.load('en_core_web_sm')


# In[32]:


# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(no_stop_tokens[0:100000]))
print(doc[0:40])


# In[33]:


lemmas = [token.lemma_ for token in doc]
print(lemmas[0:25])


# In[34]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas)


# In[35]:


pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=False).head(20)


# In[36]:


print(vectorizer.vocabulary_)


# In[37]:


print(vectorizer.get_feature_names_out()[50:100])
print(X.toarray()[50:100])


# In[38]:


print(X.toarray().shape)


# In[39]:


vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 100)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(data['content'])


# In[40]:


bow_matrix_ngram


# In[41]:


print(vectorizer_ngram_range.get_feature_names_out())
print(bow_matrix_ngram.toarray())


# In[42]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 10)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(data['content'])
print(vectorizer_n_gram_max_features.get_feature_names_out())
print(tf_idf_matrix_n_gram_max_features.toarray())


# In[43]:


# Import packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[44]:


# Generate wordcloud
stopwords = STOPWORDS

wordcloud = WordCloud(width = 3000, height = 2000, 
                      background_color='black', max_words=100,
                      colormap='Set2',stopwords=stopwords).generate(data1)
# Plot
plot_cloud(wordcloud)


# In[ ]:




