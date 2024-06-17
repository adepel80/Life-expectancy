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
```
for i in ['GDP','Total expenditure',' thinness  1-19 years', ' thinness 5-9 years' ]:
    lw,uw=wisker(life[i])
    life[i]=np.where(life[i]<lw,lw,life[i])
    life[i]=np.where(life[i]>uw,uw,life[i])
```
```for i in ['GDP','Total expenditure',' thinness  1-19 years', ' thinness 5-9 years' ]:
    sns.boxplot(life[i])
    plt.show()
```
## import plotly.express as px
```
fig = px.pie(life, names='Status')
fig
```
![LIFE PIE](https://github.com/adepel80/Life-expectancy/assets/123180341/84af6f23-fdae-4161-a01d-dd166c8e7a54)
## import plotly.graph_objects as go
```
go.Figure(
    data=[go.Histogram(x=life["Life expectancy "], xbins={"start": 36.0, "end": 90.0, "size": 1.0})],
    layout=go.Layout(title="Histogram of Life expectancy", yaxis={"title": "Count"}, bargap=0.05),
)
```
![LIFE HISTO](https://github.com/adepel80/Life-expectancy/assets/123180341/dca279b9-d5c1-4748-ac1c-6abe3ce444fe)

```
go.Figure(
    data=[go.Histogram(x=life["Adult Mortality"], xbins={"start": 36.0, "end": 90.0, "size": 1.0})],
    layout=go.Layout(title="Histogram of Adult Mortality", yaxis={"title": "Count"}, bargap=0.05),
)
```
![LIFE GRAHP](https://github.com/adepel80/Life-expectancy/assets/123180341/4cb22135-ab36-4429-aece-2c0d07fea64e)

## CORRELATION
```
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
```
![LIFE CORRE FEA](https://github.com/adepel80/Life-expectancy/assets/123180341/41b28d99-b361-48f7-b612-17d3848d117a)

## CHOOSING A FEATURE
```
#Choosing x and y values

#x is our features except diagnosis (classification columns) 
#y is diagnosis
x = life[['Schooling','Income composition of resources',' HIV/AIDS',' BMI ','Adult Mortality']]
y = life['Life expectancy ']

```
## TEST AND TRAIN
```
from sklearn.model_selection import train_test_split
```
```
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
```
X_train
X_test
y_test
y_train
```

## SCALING 
```
#scalling
from sklearn.preprocessing import LabelEncoder,StandardScaler
```
```
sdc = StandardScaler()
```
```
X_train = sdc.fit_transform(X_train)
X_test = sdc.transform(X_test)
```
```
X_test
X_train
y_test
```
## IMPORT LINEAR REGRESSION
```
from sklearn.linear_model import LinearRegression
```
### LINEAR REGRESSION
```
LinearRegression?
```
```
model= LinearRegression(n_jobs=15)
```
```
model.fit(X_train, y_train)
```
```
model.score(X_test, y_test)
```
![LIFE LINERA RESUKT](https://github.com/adepel80/Life-expectancy/assets/123180341/2a4fe5eb-b404-492c-b04a-53fc376fe49d)
# IMPORTING THE SVM
```
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
```
### STANDARD SCALER
```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```
regressor = SVR(kernel='rbf')
```
```
regressor?
```
```
regressor.fit(X_train, y_train)
```
