# 💗🎗️ Breast Cancer Prediction using Machine Learning

## Overview  
This project aims to classify **breast tumors as malignant or benign** using various supervised machine learning algorithms.  
The main goals of this project are to:  
- Compare the performance of multiple classifiers on the same dataset.  
- Identify the most accurate and reliable model for prediction.  
- Prepare the best-performing model for **future deployment in a real-world diagnostic system** (machine learning deployment readiness).  

---

## 📊 Dataset
- **Source:** UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Number of samples:** 699  
- **Number of features:** 9 numerical attributes + target variable  
- **Target variable:** Class (2 = Benign, 4 = Malignant)  

---

## 🧹 Data Cleaning & Preprocessing
- Removed duplicates and handled missing values.  
- Recoded the target variable for binary classification:  
  - 0 → Benign  
  - 1 → Malignant  
- Split the data into training and testing sets using a 75:25 ratio.  
- Standardized feature values to ensure fair comparison across models.  

---

## 🤖 Machine Learning Models Applied
Each model was implemented in a separate `.py` file for modular organization:  

1. Logistic Regression  
2. K-Nearest Neighbors (k = 5)  
3. Linear SVM (kernel = linear)  
4. Kernel SVM (kernel = rbf)  
5. Naïve Bayes  
6. Decision Tree  
7. Random Forest (n_estimators = 10)  
8. XGBoost Classifier  

---

### 🏆 Model Performance Summary  

| Algorithm | Test Accuracy | Confusion Matrix (TP, FN, FP, TN) |
|------------|----------------|----------------------------------|
| Logistic Regression | **0.956** | (57, 1, 4, 51) |
| KNN (k = 5) | **0.947** | (57, 1, 5, 50) |
| Linear SVM (linear kernel) | **0.952** | (56, 2, 4, 51) |
| Kernel SVM (rbf kernel) | **0.956** | (56, 2, 3, 52) |
| Naïve Bayes | **0.956** | (55, 3, 2, 53) |
| Decision Tree | **0.920** | (55, 3, 6, 49) |
| Random Forest (n_estimators = 10) | **0.965** | (56, 2, 2, 53) |
| XGBoost | **0.965** | (55, 3, 1, 54) |

---

## 🏁 Conclusion  
Among all models, **Random Forest** and **XGBoost** achieved the highest test accuracy (**0.965**).  
These models show strong potential for deployment in diagnostic prediction systems, demonstrating both accuracy and stability.  ******
