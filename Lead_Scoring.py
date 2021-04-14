#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve
from sklearn import tree


# In[2]:


x = pd.read_csv(r'C:/ML/Lead/Lead_Scoring.csv')


# In[3]:


x.head()


# In[4]:


x.info()


# In[5]:


x.describe()


# In[6]:


x.isnull().sum()


# In[7]:


x.shape


# In[8]:


sum(x.duplicated(subset = 'Prospect ID')) == 0


# In[9]:


sum(x.duplicated(subset = 'Lead Number')) ==0


# In[10]:


x=x.drop(columns=['Prospect ID','Lead Number'])


# In[11]:


x.isnull().sum()


# In[12]:


x['Country'].value_counts(dropna=False)


# In[13]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x.Country, hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[14]:


x['Country'] = x['Country'].replace(np.nan,'India')


# In[15]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x.Country, hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[16]:


x.isnull().sum()


# In[17]:


cdrop=['Country']


# In[18]:


x.isnull().sum()


# In[19]:


x['City'].value_counts(dropna=False)


# In[20]:


x = x.replace('Select', np.nan)


# In[21]:


x['City'].value_counts(dropna=False)


# In[22]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x.City, hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[23]:


x['City'] = x['City'].replace(np.nan,'Mumbai')


# In[24]:


x.isnull().sum()


# In[25]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x.City, hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[26]:


x['Tags'].value_counts(dropna=False)


# In[27]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x.Tags, hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[28]:


x['Tags'] = x['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized'], 'Other_Tags')

x['Tags'] = x['Tags'].replace(['switched off','Already a student','Not doing further education','invalid number','wrong number given','Interested  in full time MBA'] , 'Other_Tags')


# In[29]:


x['Tags'].value_counts(dropna=False)


# In[30]:


x['Tags'] = x['Tags'].replace(np.nan,'Not Specified')


# In[31]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x.Tags, hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[32]:


x.isnull().sum()


# In[33]:


x['Specialization'].value_counts(dropna=False)


# In[34]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x.Specialization, hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[35]:


x['Specialization'] = x['Specialization'].replace(['Finance Management','Human Resource Management','Marketing Management','Operations Management','IT Projects Management','Supply Chain Management','Healthcare Management','Hospitality Management','Retail Management'] ,'Management_Specializations')  


# In[36]:


x['Specialization'].value_counts(dropna=False)


# In[37]:


x['Specialization'] = x['Specialization'].replace(np.nan,'Not Specified')


# In[38]:


x['Specialization'].value_counts(dropna=False)


# In[39]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x.Specialization, hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[40]:


x['Lead Source'].value_counts(dropna=False)


# In[41]:


x['Lead Source'] = x['Lead Source'].replace(np.nan,'Others')


# In[42]:


x['Lead Source'] = x['Lead Source'].replace('google','Google')
x['Lead Source'] = x['Lead Source'].replace('Facebook','Social Media')
x['Lead Source'] = x['Lead Source'].replace(['bing','Click2call','Press_Release','youtubechannel','welearnblog_Home','WeLearn','blog','Pay per Click Ads','testone','NC_EDM'] ,'Others') 


# In[43]:


x['Lead Source'].value_counts(dropna=False)


# In[44]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x['Lead Source'], hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[45]:


x.isnull().sum()


# In[46]:


x['What is your current occupation'].value_counts(dropna=False)


# In[47]:


x['What is your current occupation'] = x['What is your current occupation'].replace(np.nan,'Unemployed')


# In[48]:


x['What is your current occupation'].value_counts(dropna=False)


# In[49]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x['What is your current occupation'], hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[50]:


x.isnull().sum()


# In[51]:


x['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[52]:


x['What matters most to you in choosing a course'] = x['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[53]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x['What matters most to you in choosing a course'], hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[54]:


x.isnull().sum()


# In[55]:


cdrop.append('What matters most to you in choosing a course')
cdrop


# In[56]:


x['Last Activity'].value_counts(dropna=False)


# In[57]:


x['Last Activity'] = x['Last Activity'].replace(np.nan,'Others')
x['Last Activity'] = x['Last Activity'].replace(['Unreachable','Unsubscribed','Had a Phone Conversation','Approached upfront','View in browser link Clicked','Email Marked Spam','Email Received','Resubscribed to emails','Visited Booth in Tradeshow'],'Others')


# In[58]:


x['Last Activity'].value_counts(dropna=False)


# In[59]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x['Last Activity'], hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[60]:


cols=x.columns
for i in cols:
    if((100*(x[i].isnull().sum()/len(x.index))) >= 45):
        x.drop(i, 1, inplace = True)


# In[61]:


x.isnull().sum()


# In[62]:


x['Lead Origin'].value_counts(dropna=False)


# In[63]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x['Lead Origin'], hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[64]:


x['Do Not Email'].value_counts(dropna=False)


# In[65]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x['Do Not Email'], hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[66]:


x['Do Not Call'].value_counts(dropna=False)


# In[67]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x['Do Not Call'], hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[68]:


cdrop.append('Do Not Call')
cdrop


# In[69]:


x['Search'].value_counts(dropna=False)


# In[70]:


x['Magazine'].value_counts(dropna=False)


# In[71]:


x['Newspaper Article'].value_counts(dropna=False)


# In[72]:


x['X Education Forums'].value_counts(dropna=False)


# In[73]:


x['Newspaper'].value_counts(dropna=False)


# In[74]:


x['Digital Advertisement'].value_counts(dropna=False)


# In[75]:


x['Through Recommendations'].value_counts(dropna=False)


# In[76]:


x['Receive More Updates About Our Courses'].value_counts(dropna=False)


# In[77]:


x['Update me on Supply Chain Content'].value_counts(dropna=False)


# In[78]:


x['Get updates on DM Content'].value_counts(dropna=False)


# In[79]:


x['A free copy of Mastering The Interview'].value_counts(dropna=False)


# In[80]:


x['Last Notable Activity'].value_counts(dropna=False)


# In[81]:


x['Last Notable Activity'] = x['Last Notable Activity'].replace(['Had a Phone Conversation','Email Marked Spam','Unreachable','Unsubscribed','Email Bounced','Resubscribed to emails','View in browser link Clicked','Approached upfront','Form Submitted on Website','Email Received'],'Other_Notable_activity')


# In[82]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x['Last Notable Activity'], hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[83]:


x['Last Notable Activity'].value_counts(dropna=False)


# In[84]:


cdrop.extend(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque'])


# In[85]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x['Last Notable Activity'], hue=x.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[86]:


cdrop


# In[87]:


x = x.drop(cdrop,1)


# In[88]:


x.isnull().sum()


# In[89]:


corr_matrix=x.corr().round(1)
plt.figure(figsize=(12,10))
sns.heatmap(data=corr_matrix, annot=True, linewidths=0.1, square=True)


# In[90]:


x.info()


# In[91]:


x['TotalVisits'].describe()


# In[92]:


plt.figure(figsize=(6,4))
sns.boxplot(y=x['TotalVisits'])
plt.show()


# In[93]:


Q3 = x.TotalVisits.quantile(0.99)
x = x[(x.TotalVisits <= Q3)]
Q1 = x.TotalVisits.quantile(0.01)
x = x[(x.TotalVisits >= Q1)]
sns.boxplot(y=x['TotalVisits'])
plt.show()


# In[94]:


x['Total Time Spent on Website'].describe()


# In[95]:


plt.figure(figsize=(6,4))
sns.boxplot(y=x['Total Time Spent on Website'])
plt.show()


# In[96]:


x['Page Views Per Visit'].describe()


# In[97]:


plt.figure(figsize=(6,4))
sns.boxplot(y=x['Page Views Per Visit'])
plt.show()


# In[98]:


Q3 = x['Page Views Per Visit'].quantile(0.99)
x = x[x['Page Views Per Visit'] <= Q3]
Q1 = x['Page Views Per Visit'].quantile(0.01)
x = x[x['Page Views Per Visit'] >= Q1]
sns.boxplot(y=x['Page Views Per Visit'])
plt.show()


# In[99]:


Converted = (sum(x['Converted'])/len(x['Converted'].index))*100
Converted


# In[100]:


sns.boxplot(y = 'TotalVisits', x = 'Converted', data = x)
plt.show()


# In[101]:


sns.boxplot(x=x.Converted, y=x['Total Time Spent on Website'])
plt.show()


# In[102]:


sns.boxplot(x=x.Converted,y=x['Page Views Per Visit'])
plt.show()


# In[103]:


x.isnull().sum()


# In[104]:


cat_cols= x.select_dtypes(include=['object']).columns
cat_cols


# In[105]:


dummy = pd.get_dummies(x[['Lead Origin','What is your current occupation','City']], drop_first=True)
x = pd.concat([x,dummy],1)


# In[106]:


dummy = pd.get_dummies(x['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
x = pd.concat([x, dummy], axis = 1)


# In[107]:


dummy = pd.get_dummies(x['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
x = pd.concat([x, dummy], axis = 1)


# In[108]:


dummy = pd.get_dummies(x['Last Activity'], prefix  = 'Last Activity')
dummy = dummy.drop(['Last Activity_Others'], 1)
x = pd.concat([x, dummy], axis = 1)


# In[109]:


dummy = pd.get_dummies(x['Last Notable Activity'], prefix  = 'Last Notable Activity')
dummy = dummy.drop(['Last Notable Activity_Other_Notable_activity'], 1)
x = pd.concat([x, dummy], axis = 1)


# In[110]:


dummy = pd.get_dummies(x['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
x = pd.concat([x, dummy], axis = 1)


# In[111]:


x.drop(cat_cols,1,inplace = True)


# In[112]:


y = x['Converted']
y.head()
X=x.drop('Converted', axis=1)


# In[113]:


X.head()


# In[114]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[115]:


from sklearn.linear_model import LogisticRegression


# In[116]:


logmodel = LogisticRegression()


# In[117]:


logmodel.fit(X_train,y_train)


# In[118]:


y_pred= logmodel.predict(X_test)


# In[119]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(logmodel,X,y)
print(scores)


# In[120]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[121]:


from sklearn.metrics import classification_report


# In[122]:


print(classification_report(y_test,y_pred))


# In[123]:


cm = confusion_matrix(y_test,y_pred)


# In[124]:


print(cm)


# In[125]:


DT = tree.DecisionTreeClassifier(random_state = 0,class_weight="balanced",
    min_weight_fraction_leaf=0.01)
DT = DT.fit(X_train,y_train)
y_pred = DT.predict(X_test)


# In[126]:


print ("Decision Tree Accuracy is %2.2f" % accuracy_score(y_test, y_pred))


# In[127]:


score_DT = cross_val_score(DT, X, y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_DT)


# In[128]:


print(classification_report(y_test, y_pred))


# In[129]:


cm = confusion_matrix(y_test,y_pred)


# In[130]:


print(cm)


# In[131]:


X_test.head()


# In[132]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(logmodel,X,y)
print(scores)


# In[133]:


accuracy_score(y_test, logmodel.predict(X_test))


# In[134]:


print(classification_report(y_test, logmodel.predict(X_test)))


# In[135]:


#Difference between K-Means and Knn
#        K-means                                                        Knn
#1.Unsupervised Learning                                     1.Supervised Learning.

#2.Used for Clustering problems.                              2.Used for Classification  
#                                                               and Regression problems.

#3.It makes division of objects                              3.It makes predictions by 
#  into clusters.                                              learning the past data.


# In[136]:


# Working of K-Means:
# Step 1: Select number of K to decide number of Clusters.
# Step 2: Select random K points or centroids.
# Step 3: Assign each data point to their closest Centroid which will form the predefined K Clusters.
# Step 4: Calculate the Variance and place a new Centroid of each Cluster.
# Step 5: Goto Step 3(Reassign each datapoint to the new closest Centroid of each Cluster).
# Step 6: If any value is Reassigned then Goto Step 4 or else Goto Step 7.
# Step 7: Model is Ready.


# In[137]:


# Working of KNN:
# Step 1: Select number of K of the Neighbours.
# Step 2: Calculate Euclidean Distance of K Neighbours.
# Step 3: Take K nearest neighbours as per Euclidean distance.
# Step 4: Count number of data in data points in each Category.
# Step 5: Assign the new data points to that Category for which the number of the Neighbours is Maximum.
# Step 6: Model is Ready.

