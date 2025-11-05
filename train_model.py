

#!/usr/bin/env python
# coding: utf-8

# ## LOADING the DATA

# In[2]:


import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# In[3]:


DATA_FILE = Path("data") / "Breast_Cancer_Data.csv"
df = pd.read_csv(DATA_FILE, na_values=["?"])
df.info()



# ## Preparing the Data

# In[6]:

FEATURES = [
    "Clump_thickness",
    "Uniformity_of_cell_size",
    "Uniformity_of_cell_shape",
    "Marginal_adhesion",
    "Single_epithelial_cell_size",
    "Bare_nuclei",
    "Bland_chromatin",
    "Normal_nucleoli",
    "Mitoses"
]



from sklearn.model_selection import train_test_split


# In[7]:

X = df[FEATURES].astype("float64")
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)


# In[8]:


X_test.shape , y_test.shape


# ## XGBoost

# In[10]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[11]:


# TRAIN 
model = XGBClassifier()
model.fit(X_train, y_train)


# In[12]:


# EVALUATION
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

print("Train Accuracy:", round(model.score(X_train, y_train), 3))
print("Test Accuracy :", round(model.score(X_test, y_test), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))


# In[13]:


print("\nClassification Report:\n", classification_report(y_test, y_pred_test))



# ---- Save model ----
Path("artifacts").mkdir(exist_ok=True)
joblib.dump({"model": model, "features": FEATURES}, "artifacts/model.pkl")
print("💾 Model saved to artifacts/model.pkl")
