#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing pandas library
import pandas as pd


# In[2]:


# importing plotly library
import plotly.express as px


# In[3]:


# Reading Data_set
ChessData = pd.read_csv("games.csv")


# In[4]:


# To know the fields of data
ChessData.head(1)


# In[5]:


# data inforamtion
ChessData.info()


# In[6]:


# checking Null values
ChessData.isnull().sum()


# # Data cleaning

# In[7]:


# dropping Unwanted columns
ChessData = ChessData.drop(['id'], axis=1)


# In[8]:


ChessData.head()


# In[9]:


#checking the columns
ChessData.columns


# In[10]:


ChessData['winner'].unique()


# In[11]:


ChessData.shape


# In[12]:


ChessData['increment_code'].unique()


# In[13]:


ChessData = ChessData.drop(['white_id','black_id'], axis=1)


# In[14]:


ChessData.head()


# In[15]:


ChessData.columns


# In[16]:


ChessData = ChessData.drop(['increment_code','moves','opening_eco','opening_name'], axis=1)


# In[17]:


ChessData.columns


# In[18]:


ChessData.head()


# In[19]:


ChessData.tail()


# In[20]:


# Bool to int
ChessData['rated']=ChessData['rated'].astype(int)


# In[21]:


ChessData.head()


# In[22]:


ChessData['victory_status'].unique()


# In[23]:


ChessData['winner'].unique()


# In[24]:


portsvictory_status=pd.get_dummies(ChessData.victory_status,prefix='victory_status')


# In[25]:


ChessData=ChessData.join(portsvictory_status)


# In[26]:


ChessData=ChessData.drop(['victory_status'],axis=1)


# In[27]:


ChessData.head()


# In[28]:


portswinner=pd.get_dummies(ChessData.winner,prefix='winner')


# In[29]:


ChessData=ChessData.join(portswinner)


# In[30]:


ChessData=ChessData.drop(['winner'],axis=1)


# In[31]:


ChessData.head()


# In[32]:


px.scatter(ChessData.opening_ply,ChessData.rated)


# In[33]:


ChessData.columns


# # Predictions

# In[34]:


#calling traintestsplit
from sklearn.model_selection import train_test_split


# In[35]:


X=ChessData[['created_at', 'last_move_at', 'turns', 'white_rating',
       'black_rating', 'opening_ply', 'victory_status_draw',
       'victory_status_mate', 'victory_status_outoftime',
       'victory_status_resign', 'winner_black', 'winner_draw', 'winner_white']]


# In[36]:


y=ChessData.rated


# In[37]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.80)


# In[38]:


X_train.shape


# In[39]:


y_train.shape


# In[40]:


#calling LogisticRegression
from sklearn.linear_model import LogisticRegression


# In[41]:


model=LogisticRegression()


# In[42]:


model.fit(X_train,y_train)


# In[43]:


y_pred=model.predict(X_test)


# In[44]:


y_pred


# In[45]:


y_test


# In[46]:


#calling confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix


# In[47]:


confusion_matrix(y_test,y_pred)


# In[48]:


print(classification_report(y_test,y_pred))


# # Accuracy_Score

# In[49]:


from sklearn.metrics import accuracy_score


# In[50]:


regression_model_sklearn_accuracy=model.score(X_test,y_test)


# In[51]:


regression_model_sklearn_accuracy


# In[ ]:





# In[ ]:




