#!/usr/bin/env python
# coding: utf-8

# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

# Criando o ambiente do gráfico 
sns.set_style("white")
plt.figure(figsize=(10, 10))


# In[5]:


df = pd.read_csv('fxc-f.csv')


# In[6]:


import pandas as pd


# In[7]:


df = pd.read_csv('fxc-f.csv')


# In[11]:


# Criando o ambiente do gráfico 
sns.set_style("white")
plt.figure(figsize=(10, 10))

# Gráfico de Dispersão
g = sns.scatterplot(x="Data", y="Vendas", data=df)
plt.show()


# In[13]:


cmap = sns.cubehelix_palette(rot=-.4, as_cmap=True)
g = sns.scatterplot(x="Data", y="Vendas", hue="Vendas", size="Vendas",palette=cmap, data=df) #df98 calculado anteriormente
g.set_title("Número de vendas ao longo do período")
g.set_xlabel("Data")
g.yaxis.set_major_locator(ticker.MultipleLocator(1))

for ind, label in enumerate(g.get_xticklabels()):
    if ind % 3 == 0:  # Mantém apenas os rótulos múltiplos de 4 no eixo x
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.show()


# In[17]:


import matplotlib.ticker as ticker


# In[44]:


cmap = sns.cubehelix_palette(rot=-.4, as_cmap=True)
g = sns.scatterplot(x="Data", y="Vendas", hue="Vendas", size="Vendas",palette=cmap, data=df) #df98 calculado anteriormente
g.set_title("Número de vendas ao longo do período")
g.set_xlabel("Data")
g.xaxis.set_major_locator(ticker.MultipleLocator(2))

for ind, label in enumerate(g.get_xticklabels()):
    if ind % 2 == 0:  # Mantém apenas os rótulos múltiplos de 3 no eixo x
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.xticks(rotation=30)
plt.legend(loc='upper left')
plt.show()


# In[ ]:




