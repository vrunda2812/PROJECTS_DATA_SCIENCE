#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models
import seaborn as sns
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# !pip install nltk
# !pip install scikit-learn
# !pip install textblob
# !pip install textstat
# !pip install pyldavis
# !pip install gensim


# In[3]:


data = pd.read_csv("D:\DATA SCIENCE\EXCELR\PROJECT\PROJECT 2\emails.txt")

data.to_csv('emails.csv', index = None)


# In[4]:


data


# In[5]:


data = data.drop(['Unnamed: 0', 'filename' ,'Message-ID'], axis = 1)
data


# In[6]:


data['Class'].value_counts()


# In[7]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[8]:


data['Class'] = encoder.fit_transform(data['Class'])
data


# In[9]:


data.duplicated().sum()


# In[10]:


data = data.drop_duplicates(keep = 'first')
data


# In[11]:


data['Class'].value_counts()


# In[12]:


# Pie Chart based on Class
plt.pie(data['Class'].value_counts(), labels = ['Non Abusive', 'Abusive'], autopct= "%0.2f")
plt.show()


# In[13]:


data['content'].str.len()


# In[14]:


data['content'].str.len().hist()


# In[15]:


from nltk.corpus import stopwords
from collections import  Counter


# In[16]:


def plot_word_number_histogram(text):
    text.str.split().\
        map(lambda x: len(x)).\
        hist()


# In[17]:


plot_word_number_histogram(data['content'])


# In[18]:


def plot_word_length_histogram(text):
    text.str.split().\
        apply(lambda x : [len(i) for i in x]). \
        map(lambda x: np.mean(x)).\
        hist()


# In[19]:


plot_word_length_histogram(data['content'])


# In[20]:


def plot_top_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    from collections import defaultdict
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1
            
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:20] 
    x,y=zip(*top)
    plt.bar(x,y)


# In[21]:


plot_top_stopwords_barchart(data['content'])


# In[22]:


def plot_top_non_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:50]:
        if (word not in stop):
            x.append(word)
            y.append(count)
            
    sns.barplot(x=y,y=x)


# In[23]:


plot_top_non_stopwords_barchart(data['content'])


# In[24]:


from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer


# In[25]:


def plot_top_ngrams_barchart(text, n=2):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:20]

    top_n_bigrams=_get_top_ngram(text,n)[:20]
    x,y=map(list,zip(*top_n_bigrams))
    sns.barplot(x=y,y=x)


# In[26]:


plot_top_ngrams_barchart(data['content'],2)    #bigrams


# In[27]:


plot_top_ngrams_barchart(data['content'],n=3)    #trigrams 


# In[28]:


import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


# In[29]:


import nltk
nltk.download('omw-1.4')


# In[30]:


# !pip install pyLDAvis


# In[31]:


from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim
import nltk
import gensim
# !pip install numpy
# !pip install pyLDAvis


# In[32]:


def get_lda_objects(text):
    nltk.download('stopwords')    
    stop=set(stopwords.words('english'))

    
    def _preprocess_text(text):
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for data in text:
            words=[w for w in word_tokenize(data) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus
    
    corpus=_preprocess_text(text)
    
    dic=gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    
    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
    
    return lda_model, bow_corpus, dic

def plot_lda_vis(lda_model, bow_corpus, dic):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    return vis


# In[33]:


lda_model, bow_corpus, dic = get_lda_objects(data['content'])


# In[34]:


lda_model.show_topics()


# In[35]:


plot_lda_vis(lda_model, bow_corpus, dic)


# In[36]:


from wordcloud import WordCloud, STOPWORDS


# In[37]:


def plot_wordcloud(text):
    nltk.download('stopwords')
    stop=set(stopwords.words('english'))

    def _preprocess_text(text):
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for data in text:
            words=[w for w in word_tokenize(data) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus
    
    corpus=_preprocess_text(text)
    
    wordcloud = WordCloud(
        background_color='black',
        stopwords=set(STOPWORDS),
        max_words=150,
        max_font_size=30, 
        scale=3,
        random_state=1)
    
    wordcloud=wordcloud.generate(str(corpus))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
 
    plt.imshow(wordcloud)
    plt.show()


# In[38]:


plot_wordcloud(data['content'])


# In[39]:


# NEW FILE

## Data Cleaning
import re #regular expression
import string

def clean_text(text):
    '''
    Make text lowercase, remove text in square brackets,
    remove punctuation and remove words containing numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n*\r\n*', '', text)
    text = re.sub('\t*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    
    return text

clean = lambda x: clean_text(x)


# In[40]:


data['content'] = data.content.apply(clean)
data.content


# In[41]:


#Word frequency
freq = pd.Series(' '.join(data['content']).split()).value_counts()[:20] # for top 20
freq


# In[42]:


#removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[43]:


#word frequency after removal of stopwords
freq_Sw = pd.Series(' '.join(data['content']).split()).value_counts()[:20] # for top 20
freq_Sw


# In[44]:


# count vectoriser tells the frequency of a word.
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
vectorizer = CountVectorizer(min_df = 1, max_df = 5)
X = vectorizer.fit_transform(data["content"])
word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names_out(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
word_freq_df['frequency'] = word_freq_df['occurrences']/np.sum(word_freq_df['occurrences'])
#print(word_freq_df.sort('occurrences',ascending = False).head())


# In[45]:


word_freq_df.head(9)


# In[46]:


#TFIDF - Term frequency inverse Document Frequencyt
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000, max_df = 0.5, smooth_idf=True) #keep top 1000 words
doc_vec = vectorizer.fit_transform(data["content"])
names_features = vectorizer.get_feature_names_out()
dense = doc_vec.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns = names_features)


# In[47]:


#Bi-gram
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  #for tri-gram, put ngram_range=(3,3)
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[48]:


top2_words = get_top_n2_words(data["content"], n=200) #top 200
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
top2_df.head(50)


# In[49]:


#Bi-gram plot
import matplotlib.pyplot as plt
import seaborn as sns
top20_bigram = top2_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_bigram["Bi-gram"],y=top20_bigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_bigram["Bi-gram"])


# In[50]:


#Tri-gram
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[51]:


top3_words = get_top_n3_words(data["content"], n=200)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]


# In[52]:


top3_df


# In[53]:


#Tri-gram plot
import seaborn as sns
top20_trigram = top3_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_trigram["Tri-gram"],y=top20_trigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_trigram["Tri-gram"])


# In[54]:


string_Total = " ".join(data["content"])


# In[55]:


#wordcloud for entire corpus
from wordcloud import WordCloud
wordcloud_stw = WordCloud(
                background_color= 'black',
                width = 3000,
                height = 2000
                ).generate(string_Total)
plt.imshow(wordcloud_stw)


# In[56]:


# HEAT MAP 

