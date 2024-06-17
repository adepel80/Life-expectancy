# Life-expectancy
Implementing life expectancy prediction using machine learning to estimate the average number of years a person is expected to live based on various socio-economic and health-related factors.

# LIBRARIES
```
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
```
# LOADING THE DATASET
```
life = pd.read_csv('life.csv')
```
# EXPLORATORY DATA ANALYSIS
## TABULATING THE NULL AND THE PERCENTAGE NULL VALUE 
```
total = life.isnull().sum().sort_values(ascending=False)
percent = (life.isnull().sum()/life.isnull().count()).sort_values(ascending=False)
missing_data =pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
```
# DUPLICATED VALUE
```
#finding the duplicates
life.duplicated().sum()
```
# DISTRIBUTION
```
#histogram to undertsand the dstribution
import warnings
warnings.filterwarnings('ignore')
for i in life.select_dtypes(include='number').columns:
    sns.histplot(data=life, x=i)
    plt.show()
```
# OUTLIER
```
#boxplot to identify outlier
import warnings
warnings.filterwarnings('ignore')
for i in life.select_dtypes(include='number').columns:
    sns.boxplot(data=life, x=i)
    plt.show()
```
## correlation with heatmap to interprete the relation and multicolliniarity
```
s=life.select_dtypes(include='number').corr()
```

## HEATMAP
```
sns.heatmap(s)
```
## SUBPLOT FOR HEATMAP
```
plt.figure(figsize=(15,15))
sns.heatmap(s, annot=True)
```
## IMPORTING KNN IMPUTERS FOR NULL VALUES (FILL ALL NULL VALUES)
```
from sklearn.impute import KNNImputer
impute = KNNImputer()
```
## FILL ALL NULL VALUES
```
for i in life.select_dtypes(include='number').columns:
    life[i]=impute.fit_transform(life[[i]])
```
## OUTLIER
```
''' whiskers : the vertical lines extending to the most extreme, non-outlier data points.
q1 means 25, q3 means 75 '''

def wisker(col):
    q1, q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw
```
## OUTLIER FOR GDP
```
wisker(life['GDP'])
```
## 
```
for i in ['GDP','Total expenditure',' thinness  1-19 years', ' thinness 5-9 years' ]:
    lw,uw=wisker(life[i])
    life[i]=np.where(life[i]<lw,lw,life[i])
    life[i]=np.where(life[i]>uw,uw,life[i])
for i in ['GDP','Total expenditure',' thinness  1-19 years', ' thinness 5-9 years' ]:
    sns.boxplot(life[i])
    plt.show()
```
