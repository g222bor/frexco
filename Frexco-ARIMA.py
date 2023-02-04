#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('fxc-f.csv')


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.columns=["Date","Sales"]


# In[9]:


df.head()


# In[10]:


df.tail()


# In[12]:


df['Date']=pd.to_datetime(df['Date'])


# In[13]:


df.head()


# In[14]:


df.info()


# In[15]:


df.set_index('Date',inplace=True)


# In[16]:


df.head()


# In[17]:


df.describe()


# In[18]:


df.plot()


# In[21]:


df['Sales dif 1'] = df['Sales'] - df['Sales'].shift(1)


# In[22]:


df['Saz dif 1'] = df['Sales'] - df['Sales'].shift(12)


# In[23]:


df.head(14)


# In[24]:


df['Saz dif 1'].plot()


# In[28]:


from pandas.plotting import autocorrelation_plot


# In[29]:


autocorrelation_plot(df['Sales'])


# In[30]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[35]:


import statsmodels as sm


# In[37]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[49]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsaplots.plot_acf(df['Saz dif 1'].iloc[13:],lags=15,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsaplots.plot_pacf(df['Saz dif 1'].iloc[13:],lags=15,ax=ax2)


# In[53]:


from statsmodels.tsa.arima.model import ARIMA


# In[80]:


model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()


# In[81]:


model_fit.summary()


# In[88]:


df['forecast']=model_fit.predict(start=90,end=100,dynamic=False)
df[['Sales','forecast']].plot(figsize=(12,8))


# In[89]:


import statsmodels.api as sm


# In[90]:


model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1,1,0),seasonal_order=(1,1,1,30))
results=model.fit()


# In[91]:


df['forecast']=results.predict(start=30,end=100,dynamic=False)
df[['Sales','forecast']].plot(figsize=(12,8))


# In[105]:


from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(days=x) for x in range(0,24)]


# In[106]:


future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)


# In[107]:


future_datest_df.tail()


# In[108]:


future_df=pd.concat([df,future_datest_df])


# In[120]:


future_df['forecast'] = results.predict(start = 20, end = 120, dynamic = False)
future_df[['Sales', 'forecast']].plot(figsize=(12,6))


# In[152]:


future_df['forecast'] = results.predict(start = 0, end = 55, dynamic = False)
future_df[['Sales', 'forecast']].plot(figsize=(9,6))


# In[ ]:




