#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install fbprophet')


# In[4]:


get_ipython().system('pip install pystan')


# In[5]:


get_ipython().system('pip install fbprophet')


# In[7]:


get_ipython().system('pip install prophet')


# In[8]:


from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()


# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[10]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[11]:


file = 'fxc-f.csv'

df = pd.read_csv(file)


# In[12]:


df.head()


# In[14]:


df['Data'] = pd.DatetimeIndex(df['Data'])
df.dtypes


# In[15]:


df = df.rename(columns={'Data': 'ds',
                        'Vendas': 'y'})

df.head()


# In[18]:


ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Vendas por data')
ax.set_xlabel('Data')

plt.show()


# In[19]:


my_model = Prophet(interval_width=0.95)


# In[20]:


my_model.fit(df)


# In[46]:


future_dates = my_model.make_future_dataframe(periods=33, freq='D', include_history = True)
future_dates.head()


# In[47]:


forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[48]:


my_model.plot(forecast, uncertainty=True)


# In[57]:


from prophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# In[58]:


m = Prophet(changepoint_prior_scale=0.5)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)


# In[59]:


m = Prophet(changepoint_prior_scale=0.001)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)


# In[63]:


m = Prophet(changepoints=['2023-01-10'])
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)


# In[69]:


df.set_index('ds').plot();


# In[70]:


m = Prophet()
m = m.fit(df)
future = m.make_future_dataframe(periods=33)
forecast = m.predict(future)


# In[71]:


m.plot(forecast)
plt.axhline(y=0, color='red')
plt.title('Default Prophet');


# In[72]:


m.plot_components(forecast);


# In[ ]:




