#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.preprocessing import normalize


# In[ ]:


url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
df = pd.read_csv(url)


# In[ ]:


df.Class.value_counts()


# In[ ]:


Amount = normalize([df['Amount']])[0]
df['Amount'] = Amount
df = df.iloc[:, 1:]
df.head()


# In[ ]:


x = df.drop('Class', axis=1)
y = df['Class']

sampler = RandomOverSampler(sampling_strategy=0.95)
x_resample, y_resample = sampler.fit_resample(x, y)

print(y_resample.value_counts())


# In[ ]:


resample = pd.concat([x_resample, y_resample], axis=1)
resample


# In[ ]:


n = int((1.96*1.96 * 0.5*0.5)/(0.05**2))
SimpleSampling = resample.sample(n=n, random_state=42)
SimpleSampling.shape



# In[2]:


X = SimpleSampling.drop('Class', axis=1)
y = SimpleSampling['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression()
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()

models = [rf_model, lr_model, nb_model, dt_model, knn_model]
model_names = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Decision Trees', 'KNN']

accuracies = []

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"{name} : {accuracy:.4f}")


# In[3]:


import random

SystematicSampling = resample.sample(frac=1, random_state=42).reset_index(drop=True)

sampling_interval = 2
SystematicSample = SystematicSampling.iloc[::sampling_interval]
SystematicSample.shape


# In[4]:


X = SystematicSample.drop('Class', axis=1)
y = SystematicSample['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression()
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()

models = [rf_model, lr_model, nb_model, dt_model, knn_model]
model_names = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Decision Trees', 'KNN']

accuracies = []

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"{name} : {accuracy:.4f}")


# In[5]:


from sklearn.cluster import KMeans

num_clusters = 10

kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)

clusters = kmeans.fit_predict(resample)
clusters = pd.Series(clusters)

selected_clusters = random.sample(range(num_clusters), 3)
ClusterSample = resample.loc[clusters.isin(selected_clusters)]
print(ClusterSample.shape)


# In[6]:


X = ClusterSample.drop('Class', axis=1)
y = ClusterSample['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression()
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()

models = [rf_model, lr_model, nb_model, dt_model, knn_model]
model_names = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Decision Trees', 'KNN']

accuracies = []

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"{name} : {accuracy:.4f}")


# In[7]:


n = int((1.96*1.96 * 0.5*0.5)/((0.05)**2))
StratifiedSampling = resample.groupby('Class')
StratifiedSample=StratifiedSampling.sample(frac= 0.45)
StratifiedSample.shape


# In[8]:


X = StratifiedSample.drop('Class', axis=1)
y = StratifiedSample['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
lr_model= LogisticRegression()
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()

models = [rf_model, lr_model, nb_model, dt_model, knn_model]
model_names = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Decision Trees', 'KNN']

accuracies = []

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"{name} : {accuracy:.4f}")


# In[9]:


n_bootstrap = 100
desired_sample_size = 400
BootstrapSamples = pd.DataFrame()
for _ in range(n_bootstrap):
    resampled_data = resample.sample(n=len(df), replace=True, random_state=42)
    BootstrapSamples = pd.concat([BootstrapSamples, resampled_data])
    if BootstrapSamples.shape[0] >= desired_sample_size:
        break
BootstrapSamples = BootstrapSamples.iloc[:desired_sample_size, :]
print("Final Shape of Bootstrap Samples DataFrame:", BootstrapSamples.shape)


# In[10]:


X = BootstrapSamples.drop('Class', axis=1)
y = BootstrapSamples['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
lr_model= LogisticRegression()
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()

models = [rf_model, lr_model, nb_model, dt_model, knn_model]
model_names = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Decision Trees', 'KNN']

accuracies = []

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"{name} : {accuracy:.4f}")


# In[ ]:





# In[ ]:




