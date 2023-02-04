#!/usr/bin/env python
# coding: utf-8

# In[1]:




if "google.colab" in str(get_ipython()):
    get_ipython().system('pip install git+https://github.com/ourownstory/neural_prophet.git # may take a while')
    #!pip install neuralprophet # much faster, but may not have the latest upgrades/bugfixes

import pandas as pd
from neuralprophet import NeuralProphet, set_log_level

set_log_level("ERROR")


# In[2]:


data_location = "fxc-f.csv"
# df = pd.read_csv(data_location + "air_passengers.csv")


# In[5]:


m = NeuralProphet(seasonality_mode="multiplicative", learning_rate=0.1)

df = pd.read_csv("fxc-dsy.csv")
df_train, df_test = m.split_df(df=df, freq="MS", valid_p=0.2)

metrics_train = m.fit(df=df_train, freq="MS")
metrics_test = m.test(df=df_test)

metrics_test


# In[25]:


m = NeuralProphet(seasonality_mode="additive", learning_rate=0.1)
metrics_train2 = m.fit(df=df, freq="D")
future = m.make_future_dataframe(df, periods=10, n_historic_predictions=40)
forecast = m.predict(future)
fig = m.plot(forecast)


# In[15]:


import matplotlib.ticker as ticker


# In[22]:




m = NeuralProphet(seasonality_mode="multiplicative", learning_rate=0.1)

df = pd.read_csv('fxc-dsy.csv')
df_train, df_test = m.split_df(df=df, freq="D", valid_p=0.2)

metrics = m.fit(df=df_train, freq="D", validation_df=df_test, progress="plot")


# In[23]:




metrics.tail(1)


# In[ ]:




