#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install mlxtend')


# In[2]:


from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules
import pandas as pd


# In[3]:


df1 = pd.read_csv('Pakistan Largest Ecommerce Dataset.csv', encoding="ISO-8859-1") 
df1.head()


# In[4]:


df1 = df1[df1.status == 'complete'] 
df1


# In[5]:


df1['category_name_1'] = df1['category_name_1'].str.strip()
df1


# In[6]:


df1 = df1[df1.qty_ordered >0]
df1


# In[7]:


basket = pd.pivot_table(data=df1,index='item_id',columns='category_name_1',values='qty_ordered', 
                        aggfunc='sum',fill_value=0)
basket.head()


# In[8]:


def convert_into_binary(x): 
    if x > 0:
        return 1
    else:
        return 0


# In[9]:


basket_sets = basket.applymap(convert_into_binary)
basket_sets.head(20)


# In[10]:


frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
frequent_itemsets


# In[12]:


rules_mlxtend = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules_mlxtend.head() 


# In[13]:


rules_mlxtend[ (rules_mlxtend['lift'] >= 4) & (rules_mlxtend['confidence'] >= 0.8) ]

