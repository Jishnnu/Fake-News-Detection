#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("C:\Fake News Detection\Dataset\Dataset.csv")
data.head()


# In[2]:


x = np.array(data["title"])
y = np.array(data["label"])

cv = CountVectorizer()
x = cv.fit_transform(x)


# In[3]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
model = MultinomialNB()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))


# In[4]:


news = "You Can Smell Hillary’s Fear"
data = cv.transform([news]).toarray()
print(model.predict(data))


# In[5]:


news = "Cow dung can cure Corona Virus"
data = cv.transform([news]).toarray()
print(model.predict(data))

