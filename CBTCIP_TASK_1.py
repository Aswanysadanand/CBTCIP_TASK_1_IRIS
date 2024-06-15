#!/usr/bin/env python
# coding: utf-8

# CIPHERBYTE TECHNOLOGIES DATA SCIENCE INTERNSHIP

# MAY - JUNE 2024

# TASK 1

# IRIS FLOWER CLASSIFICATION USING MACHINE LEARNING MODEL

# #importing necessary libraries

# In[154]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# #loading iris dataset

# In[169]:


iris = load_iris()
X = iris.data
y = iris.target


# #loading KNN and logistic regression 

# In[181]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


# #splitting the dataset into X and y

# In[182]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# #standardise the features

# In[184]:


scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


# #training KNN Classifier

# In[173]:


knn = KNeighborsClassifier(n_neighbors=5)  # You can choose the number of neighbors
knn.fit(X_train, y_train)


# #training Logistic Regression

# In[188]:


logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)


# In[189]:


#making predictions


# In[190]:


y_pred_knn = knn.predict(X_test)
y_pred_logreg = logreg.predict(X_test)


# #evaluating the models

# In[192]:


print("KNN Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

print("\nLogistic Regression Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))


# #Calculating the Confusion Matrices

# In[193]:


cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_logreg = confusion_matrix(y_test, y_pred_logreg)


# #plotting the confusion matrices

# In[194]:


plt.figure(figsize=(14, 6))


# #Confusion Matrices for KNN

# In[197]:


plt.subplot(1, 2, 1)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# #Confusion Matrices for Logistic Regression

# In[198]:


plt.subplot(1, 2, 2)
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')



# In[ ]:





# In[ ]:





# In[ ]:




