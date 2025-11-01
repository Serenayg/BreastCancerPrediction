#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd


# ##  Upload and Explore dataset

# In[6]:


# fetch dataset 
data = fetch_ucirepo(id=15)

X = data.data.features
y = data.data.targets


# In[7]:


df = pd.concat([X, y], axis=1)


# In[8]:


df.info()


# In[9]:


df.head(10)


# In[10]:


df['Class'] = df['Class'].map({2: 0, 4: 1}).astype(int)


# In[11]:


df.duplicated().sum() 


# In[12]:


df2 = df.drop_duplicates().copy()


# In[13]:


df2.isnull().sum()


# In[14]:


df3 = df2.dropna().copy()


# In[15]:


df3.describe().T


# In[16]:


df3['Class'].value_counts()


# In[17]:


df3.info()


# In[18]:


df3.shape





import os
print(os.getcwd())


# In[48]:


df3.to_csv("/Users/serenaygoler/Breast_Cancer_Data.csv", index=False)


# In[ ]:





# In[ ]:




