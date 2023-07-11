#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pip install autoviz
# !pip install xlrd
from autoviz.AutoViz_Class import AutoViz_Class


# In[4]:


# # EDA using Autoviz
autoviz = AutoViz_Class().AutoViz('D:\DATA SCIENCE\EXCELR\PROJECT\PROJECT 2\output.csv')
autoviz


# In[5]:


# !pip install dtale


# In[8]:


import dtale
import pandas as pd
dataset = pd.read_csv("D:/DATA SCIENCE/EXCELR/PROJECT/PROJECT 2/output.csv")
d = dtale.show(dataset)
d.open_browser()


# In[ ]:




