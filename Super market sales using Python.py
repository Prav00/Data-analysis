#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Uplod the CSV file Using Pandas 


# In[2]:


df = pd.read_csv('supermarket_sales - Sheet1.csv')


# In[ ]:


# See the Data using pandas operation


# In[3]:


df


# In[4]:


df.info


# In[5]:


df.describe


# In[7]:


df.index
print('\n')


# In[8]:


df.columns


# In[9]:


df.info()


# In[ ]:


# Finding the duplicates or Null value 


# In[10]:


df.drop_duplicates(inplace = True)


# In[11]:


uni_num_rows =df.shape[0]


# In[12]:


uni_num_rows


# In[14]:


miss_values = df.isnull().sum().sort_values(ascending = False)
miss_values


# In[15]:


null_value =df.isnull()==True
df.fillna(np.nan,inplace = True)


# In[16]:


df


# In[17]:


df.describe()


# In[18]:


print(df .apply(lambda col:col.unique()))


# In[20]:


miss_values[:4]


# In[21]:





# In[22]:


df.info()


# In[23]:


df.head()


# In[ ]:


# Finding the data types and Correct it 


# In[24]:


df.dtypes


# In[25]:


from pandas import to_datetime


# In[27]:


df['Date']=to_datetime(df['Date'])


# In[28]:


type(df['Date'][0])


# In[29]:


df['Date'].dtype


# In[33]:


df['Time']=to_datetime(df['Time'])


# In[34]:


df['Time'].dtype


# In[36]:


df['Date']


# In[37]:


df['Time']


# In[42]:


def fetch_att(x):
    day=x.day
    month=x.month
    year=x.year
    return pd.Series([day,month,year])
    


# In[44]:


df[['day','month','year']]=df['Date'].apply(fetch_att)


# In[45]:


df.head()


# In[46]:


df['Date'].apply(lambda x:x.day)


# In[48]:


df['Date'].dt.year


# In[49]:


df.columns


# In[54]:


df['hour']=df['Time'].apply(lambda x:x.hour)


# In[55]:


df.head()


# In[57]:


df['Time'][0].split(' ')


# In[58]:


type('2023-02-19 13:08:00')


# In[61]:


'2023-02-19 13:08:00'.split(' ')[1].split(':')[0]


# In[63]:


df.describe().T


# In[ ]:


#Finding the relationship between the data


# In[72]:


plt.figure(figsize=(14,8))
sns.heatmap(np.round(df.corr(),2),annot=True)


# In[ ]:


#Finding the mean rating and visualize it 


# In[74]:


sns.scatterplot(x='Tax 5%',y ='gross income',data =df)


# In[75]:


sns.scatterplot(x='Quantity',y ='cogs',data =df,color ='green')


# In[76]:


sns.regplot(x='Quantity',y ='cogs',data =df,color ='green')


# In[77]:


sns.scatterplot(x='Tax 5%',y ='Total',data =df,color ='r')


# In[ ]:


#defining the own custom function


# In[78]:


df['City'].unique()


# In[79]:


df.groupby(['City'])['gross income'].median()


# In[81]:


sns.distplot(df['Rating'],kde =False)


# In[82]:


df['Rating'].mean()


# In[83]:


np.mean(df['Rating'])


# In[84]:


sns.distplot(df['Rating'],kde =False)

plt.axvline(x=np.mean(df['Rating']),c='r',label='Avg rating')


# In[ ]:


#analysis distribution of various branches


# In[168]:


def return_countplot(column):
    
    return sns.countplot(x = column,data=df)
  


# In[95]:


def return_boxplot(x_column,y_column):
    return sns.boxplot(x =x_column,y=y_column,data=df)
 
    


# In[96]:


def return_lineplot(x_column,y_column):
     return sns.lineplot(x =x_column,y=y_column,data=df)


# In[115]:


def return_relplot(x_col,y_col,col_name=None,row_name=None,rel_type=None,hue_name=None,style_name=None,name=None):
    return sns.relplot(x=x_col,y=y_col,col=col_name,row=row_name,kind=rel_type,hue=hue_name,style=style_name,data=df)


# In[ ]:


# How much sales occurs in each every branch with respect to each and eb=very month 


# In[113]:


return_boxplot('Branch','Rating')


# In[106]:


return_lineplot('hour','Quantity')


# In[102]:


df.dtypes


# In[107]:


df.columns


# In[116]:


return_relplot(x_col ='hour',y_col='Quantity',col_name='month',row_name='Branch',rel_type='line',hue_name='Gender',style_name='Gender')


# In[118]:


return_relplot(x_col='hour',y_col='Total',col_name='month',row_name='Branch',rel_type='line')


# In[ ]:


#Analisis the treand of sales


# In[121]:


return_relplot(x_col='hour',y_col='Total',col_name='Product line',row_name='Branch',rel_type='line')


# In[120]:


df.columns


# In[124]:


df['Product line'].unique()


# In[125]:


return_boxplot('Quantity','Product line')


# In[149]:


plt.figure(figsize=(14,10))
return_countplot('Product line')


# In[ ]:


# Relationship beteween Gross income and product line


# In[151]:


return_relplot('gross income','Product line',rel_type='scatter')


# In[ ]:


# Customer make payment in this business 


# In[169]:


return_countplot('Payment')#,hue_name='Branch'


# In[172]:


df.groupby('Customer type')['Total'].sum()


# In[173]:


df.groupby('Customer type').agg({'Total':'sum'})


# In[ ]:


# Do the customer type influence customer rating


# In[175]:


return_boxplot('Rating','Customer type')


# In[ ]:


# 


# In[209]:



sns.swarmplot(x='Customer type',y ='Rating',data=df,hue='City')


# In[ ]:


# Most fevorite product of the user.......


# In[201]:


get_ipython().system('pip install WordCloud')


# In[203]:


from wordcloud import WordCloud


# In[185]:


df['Product line']


# In[186]:


' '.join(df['Product line'])


# In[205]:


plt.figure(figsize=(14,10))
wordcloud=WordCloud(width=1920,height=1000).generate(' '.join(df['Product line']))

plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:




