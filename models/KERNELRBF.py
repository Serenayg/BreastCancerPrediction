#!/usr/bin/env python
# coding: utf-8

# ## LOADING the DATA

# In[2]:


import pandas as pd
import numpy as np
from pathlib import Path


# In[3]:


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "Breast_Cancer_Data.csv"
df = pd.read_csv(DATA_FILE, na_values=["?"])
df.info()


# In[4]:


df.head()


# ## Preparing the Data¶

# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X = df.drop("Class", axis=1).astype("float64")
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# In[8]:


X_test.shape , y_test.shape


# ## Support Vector Machine (SVM)

# In[10]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[11]:


#Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### KERNEL RBF

# In[13]:


# TRAIN 
model = SVC(kernel="rbf")
model.fit(X_train, y_train)


# In[14]:


# EVALUATION
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train Accuracy:", round(model.score(X_train, y_train), 3))
print("Test Accuracy :", round(model.score(X_test, y_test), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))


# In[15]:


print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

