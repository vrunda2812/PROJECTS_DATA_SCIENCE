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
from collections import Counter
from textblob import TextBlob
get_ipython().run_line_magic('matplotlib', 'inline')

# from text.blob import TextBlob, Word, Blobber
# from text.classifiers import NaiveBayesClassifier
# from text.taggers import NLTKTagger
# !pip install sweetviz


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


data = pd.read_csv("D:\DATA SCIENCE\EXCELR\PROJECT\PROJECT 2\output.txt")

data.to_csv('output.csv', index = None)
data


# In[4]:


data = data.drop(['Unnamed: 0', 'filename' ,'Message-ID'], axis = 1)
data


# In[5]:


data['Class'].value_counts()


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


data.reset_index(drop=True, inplace=True)
data


# In[12]:


data['Class'].value_counts()


# In[13]:


plt.pie(data['Class'].value_counts(), labels = ['Non Abusive', 'Abusive'], autopct= "%0.2f")
plt.show()


# In[14]:


data = data.loc[data['Class'] == 0, 'content']
data


# In[15]:


# clean up the dataset
data = [x.strip() for x in data] # remove both the leading and the trailing characters
data = [x for x in data if x] # removes empty strings, because they are considered in Python as False
data[0:10]


# In[16]:


##Part Of Speech Tagging
#nlp = spacy.load('en')
nlp = spacy.load('en_core_web_sm')

one_block = data[2]
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter=True)


# In[17]:


one_block


# In[18]:


for token in doc_block[0:100]:
    print(token, token.pos_)


# In[19]:


nouns_verbs = [token.text for token in doc_block if token.pos_ in ('NOUN', 'VERB')]
print(nouns_verbs[5:100])


# In[20]:


#Counting tokens again
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(nouns_verbs)
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
wf_df = pd.DataFrame(words_freq)
wf_df.columns = ['word', 'count']

wf_df[0:10]


# In[21]:


#Sentiment analysis
afinn = pd.read_csv("D:/DATA SCIENCE/EXCELR/PROJECT/PROJECT 2/Afinn.csv", sep=',', encoding='latin-1')
afinn.shape


# In[22]:


# import nltk
# nltk.download('punkt')
from nltk import punkt


# In[23]:


from nltk import tokenize
sentences = tokenize.sent_tokenize(" ".join(data))
sentences[0:10]


# In[24]:


sent_df = pd.DataFrame(sentences, columns=['sentence'])
sent_df


# In[25]:


affinity_scores = afinn.set_index('word')['value'].to_dict()


# In[26]:


#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
nlp = spacy.load('en_core_web_sm')
# nlp.max_length = 1500000
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        print(sentence)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score


# In[27]:


sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)


# In[28]:


sent_df


# In[29]:


# Sentiment score of the whole review # Negative 
negative = sent_df[sent_df['sentiment_value']<0]
negative


# In[30]:


neutral = sent_df[sent_df['sentiment_value']==0]   ## NEUTRAL
neutral


# In[31]:


sent_df[sent_df['sentiment_value']<0]['sentence'].tolist()


# In[32]:


positive = sent_df[sent_df['sentiment_value']>0]   ## POSITIVE
positive


# In[33]:


import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


# In[34]:


def polarity(text):
    return TextBlob(text).sentiment.polarity


# In[35]:


sent_df['polarity'] = sent_df['sentence'].apply(polarity)
sent_df['polarity']


# In[36]:


def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"


# In[37]:


sent_df['sentiment'] = sent_df['polarity'].apply(sentiment)
sent_df['sentiment']


# In[38]:


import seaborn as sns
fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = sent_df)


# In[39]:


fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = sent_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')


# In[40]:


# neg_words = sent_df[sent_df.sentiment == 'Negative']
neg_words = sent_df[sent_df['sentiment_value']<0]   
neg_words = neg_words.sort_values(['polarity'], ascending= True)
neg_words
neg_words.reset_index(drop=True, inplace=True)


# In[41]:


text = ' '.join([word for word in neg_words['sentence']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=3000, height=1500).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative mails', fontsize=19)
plt.show()


# In[42]:


# pos_words = sent_df[sent_df.sentiment == 'Positive']
pos_words = sent_df[sent_df['sentiment_value']>0]   
pos_words = pos_words.sort_values(['polarity'], ascending= False)
pos_words


# In[43]:


text = ' '.join([word for word in pos_words['sentence']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive mails', fontsize=19)
plt.show()


# In[44]:


# neutral_words = sent_df[sent_df.sentiment == 'Neutral
neutral_words = sent_df[sent_df['sentiment_value']==0]   
neutral_words = neutral_words.sort_values(['polarity'], ascending= False)
neutral_words


# In[45]:


text = ' '.join([word for word in neutral_words['sentence']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral mails', fontsize=19)
plt.show()


# In[ ]:





# In[ ]:




