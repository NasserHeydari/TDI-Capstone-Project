
# coding: utf-8

# In[14]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn import base
from sklearn.neighbors import KNeighborsRegressor


# # Read and Pre-process Data

# In[15]:


train = pd.read_csv('train.csv')
wf1 = pd.read_csv('windforecasts_wf1.csv')
wf2 = pd.read_csv('windforecasts_wf2.csv')
wf3 = pd.read_csv('windforecasts_wf3.csv')
wf4 = pd.read_csv('windforecasts_wf4.csv')
wf5 = pd.read_csv('windforecasts_wf5.csv')
wf6 = pd.read_csv('windforecasts_wf6.csv')
wf7 = pd.read_csv('windforecasts_wf7.csv')


# In[17]:


wf1.head()


# ## Covert date to datetime format

# In[18]:


train['date'] = pd.to_datetime(train['date'], format='%Y%m%d%H', errors='ignore')


# In[19]:


wf1['date'] = pd.to_datetime(wf1['date'], format='%Y%m%d%H', errors='ignore') + pd.to_timedelta(wf1['hors'], unit = 'h')
wf2['date'] = pd.to_datetime(wf2['date'], format='%Y%m%d%H', errors='ignore') + pd.to_timedelta(wf2['hors'], unit = 'h')
wf3['date'] = pd.to_datetime(wf3['date'], format='%Y%m%d%H', errors='ignore') + pd.to_timedelta(wf3['hors'], unit = 'h')
wf4['date'] = pd.to_datetime(wf4['date'], format='%Y%m%d%H', errors='ignore') + pd.to_timedelta(wf4['hors'], unit = 'h')
wf5['date'] = pd.to_datetime(wf5['date'], format='%Y%m%d%H', errors='ignore') + pd.to_timedelta(wf5['hors'], unit = 'h')
wf6['date'] = pd.to_datetime(wf6['date'], format='%Y%m%d%H', errors='ignore') + pd.to_timedelta(wf6['hors'], unit = 'h')
wf7['date'] = pd.to_datetime(wf7['date'], format='%Y%m%d%H', errors='ignore') + pd.to_timedelta(wf7['hors'], unit = 'h')


# In[21]:


wf1.head()


# ## Read until 2011/1/1 where full data are available 

# In[30]:


train = train[0:13176]
wf1 = wf1[0:52559]
wf2 = wf2[0:52559]
wf3 = wf3[0:52559]
wf4 = wf4[0:52559]
wf5 = wf5[0:52559]
wf6 = wf6[0:52559]
wf7 = wf7[0:52559]


# ## Since there are several values for a specific date and time, we decided to groupby the values 

# In[31]:


wf1 = wf1.groupby(['date']).mean()
wf2 = wf2.groupby(['date']).mean()
wf3 = wf3.groupby(['date']).mean()
wf4 = wf4.groupby(['date']).mean()
wf5 = wf5.groupby(['date']).mean()
wf6 = wf6.groupby(['date']).mean()
wf7 = wf7.groupby(['date']).mean()


# In[32]:


wf1.head()


# ## In the next few cells we are trying to organize training datasets for each wind farm, some steps might seem too long!

# In[33]:


train1 = pd.DataFrame(columns=['date','WF','Energy'])
train2 = pd.DataFrame(columns=['date','WF','Energy'])
train3 = pd.DataFrame(columns=['date','WF','Energy'])
train4 = pd.DataFrame(columns=['date','WF','Energy'])
train5 = pd.DataFrame(columns=['date','WF','Energy'])
train6 = pd.DataFrame(columns=['date','WF','Energy'])
train7 = pd.DataFrame(columns=['date','WF','Energy'])

train1['date']=train['date']
train1['Energy']=train['wp1']
train1['WF']=1

train2['date']=train['date']
train2['Energy']=train['wp2']
train2['WF']=2

train3['date']=train['date']
train3['Energy']=train['wp3']
train3['WF']=3

train4['date']=train['date']
train4['Energy']=train['wp4']
train4['WF']=4

train5['date']=train['date']
train5['Energy']=train['wp5']
train5['WF']=5

train6['date']=train['date']
train6['Energy']=train['wp6']
train6['WF']=6

train7['date']=train['date']
train7['Energy']=train['wp7']
train7['WF']=7


# In[34]:


s1 = wf1.join(train1.set_index('date'), on='date')
s2 = wf2.join(train2.set_index('date'), on='date')
s3 = wf3.join(train3.set_index('date'), on='date')
s4 = wf4.join(train4.set_index('date'), on='date')
s5 = wf5.join(train5.set_index('date'), on='date')
s6 = wf6.join(train6.set_index('date'), on='date')
s7 = wf7.join(train7.set_index('date'), on='date')


# In[35]:


s1 = s1[[ 'Energy','WF',  'u', 'v','ws','wd']].dropna()
s2 = s2[[ 'Energy','WF',  'u', 'v','ws','wd']].dropna()
s3 = s3[[ 'Energy','WF',  'u', 'v','ws','wd']].dropna()
s4 = s4[[ 'Energy','WF',  'u', 'v','ws','wd']].dropna()
s5 = s5[[ 'Energy','WF',  'u', 'v','ws','wd']].dropna()
s6 = s6[[ 'Energy','WF',  'u', 'v','ws','wd']].dropna()
s7 = s7[[ 'Energy','WF',  'u', 'v','ws','wd']].dropna()


# ### We normalized data to make sure all values fall between 0 and 1

# In[36]:


s1.ws = s1.ws/s1.ws.max()
s1.wd = s1.wd/s1.wd.max()
s2.ws = s2.ws/s2.ws.max()
s2.wd = s2.wd/s2.wd.max()
s3.ws = s3.ws/s3.ws.max()
s3.wd = s3.wd/s3.wd.max()
s4.ws = s4.ws/s4.ws.max()
s4.wd = s4.wd/s4.wd.max()
s5.ws = s5.ws/s5.ws.max()
s5.wd = s5.wd/s5.wd.max()
s6.ws = s6.ws/s6.ws.max()
s6.wd = s6.wd/s6.wd.max()
s7.ws = s7.ws/s7.ws.max()
s7.wd = s7.wd/s7.wd.max()


# ## We identified that the mean energy production in the last four hours can significantly improve the performance of the model we will show later

# In[37]:


s1['mean_energy_last_4_hrs'] = (s1.Energy.rolling(5).sum() - s1.Energy)/4
s2['mean_energy_last_4_hrs'] = (s2.Energy.rolling(5).sum() - s2.Energy)/4
s3['mean_energy_last_4_hrs'] = (s3.Energy.rolling(5).sum() - s3.Energy)/4
s4['mean_energy_last_4_hrs'] = (s4.Energy.rolling(5).sum() - s4.Energy)/4
s5['mean_energy_last_4_hrs'] = (s5.Energy.rolling(5).sum() - s5.Energy)/4
s6['mean_energy_last_4_hrs'] = (s6.Energy.rolling(5).sum() - s6.Energy)/4
s7['mean_energy_last_4_hrs'] = (s7.Energy.rolling(5).sum() - s7.Energy)/4


# # Here we concat the training data (tables) created for each wind farm in the previous steps 

# In[169]:


frames = [s1.dropna(), s2.dropna(), s3.dropna(), s4.dropna(), s5.dropna(), s6.dropna(), s7.dropna()]
train_new = pd.concat(frames,sort=False)


# # The below figure shows the variation of hourly wind energy production 

# In[170]:


train_new.groupby(train_new.index.date).mean().Energy.plot()


#  # Create training and test dataset
#  ## we keep ~75% of data for model training and ~25% of data for testing the model

# In[205]:


CUT_YEAR = '2010-8-15'

train_set = train_new[train_new.index < CUT_YEAR]
test_set = train_new[train_new.index >= CUT_YEAR]


# In[206]:


train_set.shape[0] / train_new.shape[0] , test_set.shape[0] / train_new.shape[0]


# In[207]:


train_set.head(3)


# # Let's do modeling

# ## Since we want to predict hourly wind energy production, thus it seems reasonable to have hour and month as two independent variables

# In[208]:


df_train = train_set
df_test = test_set


# In[209]:


df_train.head()


# In[210]:


# new_index = [i for i in range(0, df.shape[0])]
# df.index = new_index 


# In[212]:


df_train['Julian'] = df_train.index.to_julian_date()
df_train['sin(day)'] = np.sin(df_train.index.hour / 24.0 * 2* np.pi)
df_train['cos(day)'] = np.cos(df_train.index.hour / 24.0 * 2* np.pi)
df_train['sin(3mo)'] = np.sin(df_train['Julian'] / (365.25 / 4) * 2 * np.pi)
df_train['cos(3mo)'] = np.cos(df_train['Julian'] / (365.25 / 4) * 2 * np.pi)
df_train['hour'] = df_train.index.hour
df_train['month'] = df_train.index.month
df_train['date'] = df_train.index
new_index = [i for i in range(0, df_train.shape[0])]
df_train.index = new_index 


df_test['Julian'] = df_test.index.to_julian_date()
df_test['sin(day)'] = np.sin(df_test.index.hour / 24.0 * 2* np.pi)
df_test['cos(day)'] = np.cos(df_test.index.hour / 24.0 * 2* np.pi)
df_test['sin(3mo)'] = np.sin(df_test['Julian'] / (365.25 / 4) * 2 * np.pi)
df_test['cos(3mo)'] = np.cos(df_test['Julian'] / (365.25 / 4) * 2 * np.pi)
df_test['hour'] = df_test.index.hour
df_test['month'] = df_test.index.month
df_test['date'] = df_test.index
new_index = [i for i in range(0, df_test.shape[0])]
df_test.index = new_index 


# In[213]:


df_train.head()


# ## We decided to include mean energy production for each hour for each wind farm and as independent variable

# In[214]:


group_by_farm_hour_train = df_train.groupby(['hour','WF']).mean()['Energy']
group_by_farm_hour_test = df_test.groupby(['hour','WF']).mean()['Energy']


# In[215]:


mean_by_hour_train = pd.DataFrame(group_by_farm_hour_train).rename(columns = {'Energy':'mean_Energy_hour'})
mean_by_hour_test = pd.DataFrame(group_by_farm_hour_test).rename(columns = {'Energy':'mean_Energy_hour'})


# In[216]:


df_train = df_train.merge(mean_by_hour_train, left_on=['hour','WF'], right_index=True)
df_test = df_test.merge(mean_by_hour_test, left_on=['hour','WF'], right_index=True)


# In[217]:


df_train = df_train.sort_values(['WF', 'date'], ascending=[True, True])
df_test = df_test.sort_values(['WF', 'date'], ascending=[True, True])


# In[218]:


df_train.head()


# In[219]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor

class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col_names):
        self.col_names = col_names  
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.col_names]


# In[220]:


class GroupbyEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, column):
        self.column = column                      
        self.WF_model = dict()                                          
 
    def fit(self, X, y=None):
        for WF, group in X.groupby(self.column):
            group = group.sample(frac=1).reset_index(drop=True)
            MY_FU = []
            MY_FU = FeatureUnion([("cst", ColumnSelectTransformer(['ws','wd','mean_Energy_hour','mean_energy_last_4_hrs','sin(day)','cos(day)','sin(3mo)','cos(3mo)'])),
                                  ("month", Pipeline([("cst", ColumnSelectTransformer(['month'])),("onehot", OneHotEncoder())])),
                                  ("hour", Pipeline([("cst", ColumnSelectTransformer(['hour'])),("onehot", OneHotEncoder())]))])    
            
            pipe = Pipeline([("Union", MY_FU ),
                             ("rf", RandomForestRegressor())])
            
            param_grid = {'rf__min_samples_leaf': range(1, 20, 5)}

            model = GridSearchCV(pipe, param_grid, iid=False, cv= 4)
            
            self.WF_model[WF] = model.fit(group,group.Energy)
            
        return self                                 
  
    def predict(self, X):          
        predictions = {}
        final = []
        for WF, group in X.groupby(self.column):    
            pred = self.WF_model[WF].predict(group)
            for i in range(0,len(group.index)):
                predictions[group.index[i]] = pred[i] 
        for key in sorted(predictions.keys()):
            final.append(predictions[key])
        
        return final


# In[221]:


from sklearn.pipeline import Pipeline 

pipe = Pipeline([("Estimator", GroupbyEstimator('WF'))])
                  


# # check the performance of the model

# In[222]:


from sklearn.metrics import r2_score

pipe.fit(df_train)

r2_score(pipe.predict(df_train),df_train.Energy)


# In[241]:


fig, ax = plt.subplots()
ax.plot(df_train.Energy[0:4000])
ax.plot(pd.DataFrame(pipe.predict(df_train))[0:4000])


# In[223]:


r2_score(pipe.predict(df_test),df_test.Energy)


# In[240]:


fig, ax = plt.subplots()
ax.plot(df_test.Energy[0:4000])
ax.plot(pd.DataFrame(pipe.predict(df_test))[0:4000])

