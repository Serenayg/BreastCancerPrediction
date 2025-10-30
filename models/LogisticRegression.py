#!/usr/bin/env python
# coding: utf-8

# ## LOADING the DATA

# In[24]:


import pandas as pd
import numpy as np
from pathlib import Path

# In[32]:




DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "Breast_Cancer_Data.csv"
df = pd.read_csv(DATA_FILE, na_values=["?"])
df.info()


# In[36]:


df.head()


# ## Preparing the Data

# In[4]:


from sklearn.model_selection import train_test_split


# In[39]:


X = df.drop("Class", axis=1).astype("float64")
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# In[41]:


X_test.shape , y_test.shape


# ## Logistic Regression 

# In[47]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[49]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[51]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[57]:


# EVALUATE
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

print("Train Accuracy:", round(model.score(X_train, y_train), 3))
print("Test Accuracy :", round(model.score(X_test, y_test), 3))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))


# In[59]:


print("\nClassification Report:\n", classification_report(y_test, y_pred_test))


# In[ ]:




