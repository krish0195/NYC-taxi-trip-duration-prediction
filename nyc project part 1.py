#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                         
sns.set(color_codes = True)                   
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression


# In[29]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# In[30]:


df = pd.read_csv(r"E:\data science\analytics vidya\Applied ML\Assignment submission\NYC taxi trip duration prediction\nyc_taxi_trip_duration.csv")


# In[31]:


df.head(5)


# In[32]:


df.info()


# In[33]:


print(df.duplicated().sum())
print(df.isnull().sum())


# In[34]:


#outliner check

numerical_cols=df.select_dtypes(include=['int64',"float"]).columns
print(numerical_cols)

for i in df[numerical_cols]:
    q1,q3=df[i].quantile([0.25,0.75])
    iqr=q3-q1
    upper=q3+1.5*iqr
    lower=q1-1.5*iqr
    df[i]=np.where(df[i]>upper , upper , df[i])
    df[i]=np.where(df[i]<lower ,lower , df[i])


# In[35]:


df.head(5)


# In[36]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr())


# DATA PRE-PROCESSING-Feature Enginerring:

# In[37]:


# Data Formatting
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)


# In[38]:


# Lets move on to Pickup and Drop off columns
# Converting to datetime
df["pickup_datetime"]=pd.to_datetime(df['pickup_datetime'])
df["dropoff_datetime"]=pd.to_datetime(df['dropoff_datetime'])

df['pickup_dayname']=df['pickup_datetime'].dt.day_name()
df['pickup_day_no']=df['pickup_datetime'].dt.weekday
df['pickup_hour']=df['pickup_datetime'].dt.hour
df['pickup_month']=df['pickup_datetime'].dt.month
df['pickup_day'] = df['pickup_datetime'].dt.day

df['dropoff_dayname']=df['dropoff_datetime'].dt.day_name()
df['dropoff_day_no']=df['dropoff_datetime'].dt.weekday
df['dropoff_hour']=df['dropoff_datetime'].dt.hour
df['dropoff_month']=df['dropoff_datetime'].dt.month
df['dropoff_day'] = df['dropoff_datetime'].dt.day


# In[39]:


# featire creation: distance column

from geopy.distance import great_circle
def cal_distance(pickup_lat,pickup_long,dropoff_lat,dropoff_long):
    start_coordinates=(pickup_lat,pickup_long)
    stop_coordinates=(dropoff_lat,dropoff_long)
    return great_circle(start_coordinates,stop_coordinates).km


df["distance"] = df.apply(lambda x: cal_distance(x["pickup_latitude"],x["pickup_longitude"],x["dropoff_latitude"],x["dropoff_longitude"] ), axis=1)


# In[40]:


# featire creation: speed column

#Calculate Speed in km/h for further insights
df['speed'] = (df.distance/(df.trip_duration/3600))


# In[44]:


def time_day(x):
      if x in range(6,12):
        return "Morning"
      elif x in range(12,16):
        return "Afternoon"
      elif x in range(16,22):
        return "Evening"
      else:
        return "Late night"

df["pickup_timeofday"]=df["pickup_hour"].apply(time_day)

df["dropoff_timeofday"]=df["dropoff_hour"].apply(time_day)


# In[45]:


#seeing the categorical varaibles:
categorical_cols=df.select_dtypes(include=["object"]).columns
categorical_cols


# In[46]:


for i in df.columns:
    if df[i].dtype==np.dtype('O'):
        print(i,df[i].nunique())


# In[47]:


list(zip(range(0,len(df.columns)),df.columns,df.dtypes))


# In[48]:


#Dummify all the categorical features like "store_and_fwd_flag, vendor_id, month, weekday_num, pickup_hour, passenger_count" except the label i.e. "trip_duration"
dummy = pd.get_dummies(df.vendor_id, prefix='vendor_id')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
df = pd.concat([df,dummy], axis = 1)


# In[49]:


#seeing the Numerical varaibles:
Numerical=df.select_dtypes(include=["int","float"]).columns
Numerical


# In[50]:


#Dropping unwated columns:
df.drop(["id","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","pickup_datetime","dropoff_datetime","pickup_dayname","dropoff_dayname","pickup_timeofday","dropoff_timeofday","vendor_id"],axis=1,inplace =True)


# In[51]:


df.head()


# In[52]:


#Standard scaling is anyways a necessity when it comes to linear models and we have done that here after doing log transformation on all balance features

num_cols = ['passenger_count', 'store_and_fwd_flag', 'trip_duration',
       'pickup_day_no', 'pickup_hour', 'pickup_month', 'pickup_day',
       'dropoff_day_no', 'dropoff_hour', 'dropoff_month', 'dropoff_day',
       'distance', 'speed', 'vendor_id_2.0']
for i in num_cols:
    df[i] = np.log(df[i] + 17000)

from sklearn.preprocessing import StandardScaler
    
std = StandardScaler()
scaled = std.fit_transform(df[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)


# In[53]:


scaled


# In[54]:


# train test split:
x=df.drop(["trip_duration"],axis=1)
y=df["trip_duration"]
#Train Test Split to create a validation set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=11)


# ALGORITHM DEVELOPMENT

# Linear Relationship: The relationship between the independent and dependent variables is linear.
# Multivariate Normality: The variables (features) are normally distributed. If not, a non-linear transformation (e.g., log-transformation) may be needed to fix the issue.
# No or Little Multicollinearity: The independent variables (features) are not highly correlated with each other.
# No Auto-Corrlation: Residuals are independent of one another (i.e., outcome is independent of a previous outcome).
# Homoscedasticity: Residuals are equal across the regression line.

# In[55]:


#Linear Regression Model:
regression_model = LinearRegression()
regression_model.fit(xtrain,ytrain)
y_pred_lreg = regression_model.predict(xtest)


# In[56]:


from sklearn import metrics
print('\nLinear Regression Performance Metrics')
print('R^2=',metrics.explained_variance_score(ytest,y_pred_lreg))
print('MAE:',metrics.mean_absolute_error(ytest,y_pred_lreg))
print('MSE:',metrics.mean_squared_error(ytest,y_pred_lreg))
print('RMSE:',np.sqrt(metrics.mean_squared_error(ytest,y_pred_lreg)))


# In[57]:


from sklearn.model_selection import GridSearchCV
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(regression_model,parameters, cv=5)
grid.fit(xtrain, ytrain)

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
print("\n The best score across ALL searched params:\n",grid.best_score_)
print("\n The best parameters across ALL searched params:\n",grid.best_params_)


# In[ ]:





# In[58]:


#Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
# Instantiate model object
dtree = DecisionTreeRegressor()
# Fit to training data
dtree.fit(xtrain,ytrain)

# Predict
y_pred_dtree = dtree.predict(xtest)

# Score It
from sklearn import metrics
print('\nDecision Tree Regression Performance Metrics')
print('R^2=',metrics.explained_variance_score(ytest,y_pred_dtree))
print('MAE:',metrics.mean_absolute_error(ytest,y_pred_dtree))
print('MSE:',metrics.mean_squared_error(ytest,y_pred_dtree))
print('RMSE:',np.sqrt(metrics.mean_squared_error(ytest,y_pred_dtree)))


# In[ ]:





# In[61]:


#Random Forest 
from sklearn.ensemble import RandomForestRegressor 
# Instantiate model object
rforest = RandomForestRegressor(n_estimators = 20, n_jobs = -1)
# Fit to training data
rforest = rforest.fit(xtrain,ytrain)
print(rforest)

# Predict
y_pred_rforest = rforest.predict(xtest)

# Score It
from sklearn import metrics
print('\nRandom Forest Regression Performance Metrics')
print('R^2 =',metrics.explained_variance_score(ytest,y_pred_rforest))
print('MAE',metrics.mean_absolute_error(ytest, y_pred_rforest))
print('MSE',metrics.mean_squared_error(ytest, y_pred_rforest))
print('RMSE',np.sqrt(metrics.mean_squared_error(ytest, y_pred_rforest)))


# In[60]:


#we decided to use the Random Forest algorithm as Final model with least rmse error and high r2 variance:

