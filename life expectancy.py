#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
plt.style.use ("dark_background")

import os


# In[2]:


pwd


# In[3]:


life = pd.read_csv('life.csv')


# In[4]:


life.head(3)


# In[5]:


life.info()


# In[6]:


total = life.isnull().sum().sort_values(ascending=False)
percent = (life.isnull().sum()/life.isnull().count()).sort_values(ascending=False)
missing_data =pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[7]:


life.isnull().sum()


# In[8]:


#calculate the percentage of null value
life.isnull().sum()/life.shape[0]*100


# In[9]:


#finding the duplicates
life.duplicated().sum()


# In[10]:


life.describe().T


# In[11]:


life.describe(include='object')


# In[12]:


#histogram to undertsand the dstribution
import warnings
warnings.filterwarnings('ignore')
for i in life.select_dtypes(include='number').columns:
    sns.histplot(data=life, x=i)
    plt.show()


# In[13]:


#boxplot to identify outlier
import warnings
warnings.filterwarnings('ignore')
for i in life.select_dtypes(include='number').columns:
    sns.boxplot(data=life, x=i)
    plt.show()


# In[14]:


life.select_dtypes(include='number').columns


# # correaltion with heatmap to interprete the relation and multicolliniarity

# In[15]:


s=life.select_dtypes(include='number').corr()


# In[16]:


sns.heatmap(s)


# In[17]:


plt.figure(figsize=(15,15))
sns.heatmap(s, annot=True)


# In[18]:


from sklearn.impute import KNNImputer
impute = KNNImputer()


# In[19]:


for i in life.select_dtypes(include='number').columns:
    life[i]=impute.fit_transform(life[[i]])


# In[20]:


life.isnull().sum()


# In[21]:


''' What is an example of an outlier?
Outlier in Statistics | Definition & Examples - Lesson ...
When a value is called an outlier it usually means that that value 
deviates from all other values in a data set. For example,
in a group of 5 students the test grades were 9, 8, 9, 7, and 2. The last value
seems to be an outlier because it falls below the main pattern of the other grades '''
life


# # outlier

# In[22]:


''' whiskers : the vertical lines extending to the most extreme, non-outlier data points.
q1 means 25, q3 means 75 '''

def wisker(col):
    q1, q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw


# In[23]:


wisker(life['GDP'])


# In[24]:


for i in ['GDP','Total expenditure',' thinness  1-19 years', ' thinness 5-9 years' ]:
    lw,uw=wisker(life[i])
    life[i]=np.where(life[i]<lw,lw,life[i])
    life[i]=np.where(life[i]>uw,uw,life[i])


# In[25]:


for i in ['GDP','Total expenditure',' thinness  1-19 years', ' thinness 5-9 years' ]:
    sns.boxplot(life[i])
    plt.show()


# In[26]:


import plotly.express as px


# In[27]:


fig = px.pie(life, names='Status')
fig


# In[28]:


life.columns


# In[29]:


import plotly.graph_objects as go


# In[30]:


go.Figure(
    data=[go.Histogram(x=life["Life expectancy "], xbins={"start": 36.0, "end": 90.0, "size": 1.0})],
    layout=go.Layout(title="Histogram of Life expectancy", yaxis={"title": "Count"}, bargap=0.05),
)


# In[31]:


go.Figure(
    data=[go.Histogram(x=life["Adult Mortality"], xbins={"start": 36.0, "end": 90.0, "size": 1.0})],
    layout=go.Layout(title="Histogram of Adult Mortality", yaxis={"title": "Count"}, bargap=0.05),
)


# In[32]:


life_corr = life.corr()

fig = px.imshow(life_corr,
                labels=dict(x="Features", y="Features"),
                x=life_corr.columns,
                y=life_corr.columns,
                color_continuous_scale="Blues",
                color_continuous_midpoint=0)

fig.update_layout(
    title="Correlation Heatmap",
    width=800,
    height=500,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed')

fig.show()


# In[33]:


life.columns


# # note for choosing feature, you already pick your x as your dataframe(data)

# In[34]:


#Choosing x and y values

#x is our features except diagnosis (classification columns) 
#y is diagnosis
x = life[['Schooling','Income composition of resources',' HIV/AIDS',' BMI ','Adult Mortality']]
y = life['Life expectancy ']


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[37]:


X_train


# In[38]:


X_test


# In[39]:


y_test


# In[40]:


y_train


# In[41]:


#scalling
from sklearn.preprocessing import LabelEncoder,StandardScaler


# In[42]:


sdc = StandardScaler()


# In[43]:


X_train = sdc.fit_transform(X_train)
X_test = sdc.transform(X_test)


# In[44]:


X_test


# In[45]:


X_train


# In[46]:


y_test


# In[47]:


from sklearn.linear_model import LinearRegression


# In[48]:


get_ipython().run_line_magic('pinfo', 'LinearRegression')


# In[49]:


model= LinearRegression(n_jobs=15)


# In[50]:


model.fit(X_train, y_train)


# In[51]:


model.score(X_test, y_test)


# In[52]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# In[53]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[54]:


regressor = SVR(kernel='rbf')


# In[55]:


get_ipython().run_line_magic('pinfo', 'regressor')


# In[56]:


regressor.fit(X_train, y_train)


# In[57]:


y_pred = regressor.predict(X_test)


# In[58]:


y_pred = regressor.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:




