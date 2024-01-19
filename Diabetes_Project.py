#!/usr/bin/env python
# coding: utf-8

# # CS 513 C - Knowledge Discovery and Data Mining Project

# **Problem Definition: Algorithm Performance Analysis for Diabetes Classification**

# **Objective:** The primary goal is to analyze and compare the performance of various machine learning algorithms in accurately classifying individuals into categories such as diabetic or non-diabetic. This involves understanding and quantifying how effectively each algorithm can handle the data provided, make predictions, and how their predictions align with actual clinical diagnoses.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:\\Users\Vanshika\Downloads\projects\diabetes_binary_health_indicators_BRFSS2015.csv")


# ## Step 1: Exploratory Data Analysis 
# Data Understanding and Quality Checks

# In[3]:


df.head() #Initial Inspection, essential to understand how the data is structured, including column names and row indices.


# In[4]:


df.shape #helps in understanding data dimensions, validation, quality, memory and performance consideration. 


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.sample(20)


# In[8]:


df.isnull().sum() #Checking for the duplicates in the features


# In[9]:


x = df.drop(['Diabetes_binary'], axis = 1)
y = df['Diabetes_binary']


# In[10]:


pip install scikit-learn --upgrade


# In[11]:


from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# # Naive Bayes

# In[12]:


from sklearn.naive_bayes import GaussianNB

# Create a Naive Bayes classifier
classifier = GaussianNB()

# Train the classifier on the training data
classifier.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(x_test)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Compute specificity
specificity = tn / (tn + fp)
print('Specificity:', specificity)


# Compute recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Compute sensitivity
sensitivity= tp / (tp + fn)
print('Sensitivity:', sensitivity)

# Compute F1-score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))

print(y_pred)
print(y_test)


# # K-Nearest Neighbour

# In[13]:


from sklearn.neighbors import KNeighborsClassifier


# In[14]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)


# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Knn Output")

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Compute specificity
specificity = tn / (tn + fp)
print('Specificity:', specificity)

# Compute recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Compute sensitivity
sensitivity= tp / (tp + fn)
print('Sensitivity:', sensitivity)

# Compute F1-score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))



# # CART
# 

# In[15]:


from sklearn.tree import DecisionTreeClassifier


CART = DecisionTreeClassifier()

CART.fit(x_train, y_train)
y_pred = CART.predict(x_test)
# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Compute specificity
specificity = tn / (tn + fp)
print('Specificity:', specificity)

# Compute recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Compute sensitivity
sensitivity= tp / (tp + fn)
print('Sensitivity:', sensitivity)

# Compute F1-score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))



# # Decision Tree

# In[16]:


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(max_depth = 6)
DecisionTree.fit(x_train,y_train)
y_pred = DecisionTree.predict(x_test)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(" Decision Tree Output")

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Compute specificity
specificity = tn / (tn + fp)
print('Specificity:', specificity)

# Compute recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Compute sensitivity
sensitivity= tp / (tp + fn)
print('Sensitivity:', sensitivity)

# Compute F1-score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))


# # Random Forest

# In[17]:


from sklearn.ensemble import RandomForestClassifier

Random_forest = RandomForestClassifier()
Random_forest.fit(x_train,y_train)
y_pred = Random_forest.predict(x_test)
#classification_report(y_test,y_pred)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Output Random Forest")

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Compute specificity
specificity = tn / (tn + fp)
print('Specificity:', specificity)


# Compute recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Compute sensitivity
sensitivity= tp / (tp + fn)
print('Sensitivity:', sensitivity)

# Compute F1-score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))



# # C_50

# In[18]:


import seaborn as sns
from sklearn.tree import plot_tree


model = DecisionTreeClassifier(criterion='entropy', max_depth=3,splitter='best',max_leaf_nodes=5)
model.fit(x_train,y_train)
target_pred = model.predict(x_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Compute specificity
specificity = tn / (tn + fp)
print('Specificity:', specificity)


# Compute recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Compute sensitivity
sensitivity= tp / (tp + fn)
print('Sensitivity:', sensitivity)

# Compute F1-score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)
print(f"\n Classification Report:")
print(classification_report(y_test,y_pred))

plt.figure(figsize=(50,30), dpi=250)
plot_tree(model,fontsize=20,filled=True,feature_names=x.columns);


# # Support Vector Machine

# In[19]:


from sklearn.svm import SVC


svm = SVC(gamma = 'auto')
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)


# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Support Vector Machines (SVM) Output")

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Compute specificity
specificity = tn / (tn + fp)
print('Specificity:', specificity)

# Compute recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Compute sensitivity
sensitivity= tp / (tp + fn)
print('Sensitivity:', sensitivity)

# Compute F1-score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))



# # ANN

# In[20]:


from sklearn.neural_network import MLPClassifier

ann = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter = 1000)
ann.fit(x_train,y_train.values.ravel())
y_pred = ann.predict(x_test)


# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(" Artifical Neural Network ANN Output")

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

# Compute specificity
specificity = tn / (tn + fp)
print('Specificity:', specificity)

# Compute recall
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Compute sensitivity
sensitivity= tp / (tp + fn)
print('Sensitivity:', sensitivity)

# Compute F1-score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))

