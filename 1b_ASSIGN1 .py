#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.cm as cm
import pickle
import itertools
import datetime as dt
import warnings
import random
import gc
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
import os
import sys
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.datasets import load_files
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
from sklearn import cluster
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


### DATA CLEANING


# In[5]:


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)


# In[6]:


# specify encoding to deal with different formats
df = pd.read_csv('transaction_data.csv', encoding = 'ISO-8859-1')


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df_minus1 = df.loc[df ['UserId'] ==-1]


# In[10]:


df_minus1


# In[11]:


df_ok = df.loc[df ['UserId'] !=-1]


# In[12]:


df_ok


# In[13]:


df_ok.sort_values(by=['UserId'], inplace=True)


# In[14]:


df_ok


# In[15]:



dfTranId = pd.DataFrame(df_minus1["TransactionId"].unique().tolist()) 


# In[16]:


len(df_minus1["TransactionId"].unique().tolist())


# In[17]:


dfTranId 


# In[18]:


dfUseridVTranid=pd.read_csv("UseridVTranid.csv")


# In[19]:


dfUseridVTranid


# In[20]:


for i, j in zip(dfUseridVTranid["Tran_ID"], dfUseridVTranid["User_ID"]):
    df_minus1.loc[df_minus1.TransactionId==i,"UserId"]=j


# In[21]:


df_minus1


# In[22]:


df_updated_UserId=pd.concat([df_minus1, df_ok])


# In[23]:


df_updated_UserId


# In[24]:


df_updated_UserId.info()


# In[25]:


df_updated_UserId.sort_values(by=['UserId'], inplace=True)


# In[26]:


df_updated_UserId


# In[27]:


df_updated_UserId.isnull().sum(axis = 0)


# In[28]:


df_updated_UserId=df_updated_UserId.dropna()


# In[29]:


df_updated_UserId.isnull().sum(axis = 0)


# In[30]:


df_updated_UserId.drop(df_updated_UserId[df_updated_UserId['ItemCode']==-1].index, inplace = True) 


# In[31]:


df_updated_UserId


# In[32]:


len(df_updated_UserId[df_updated_UserId["ItemCode"]==-1])


# In[33]:


df_updated_UserId=df_updated_UserId.drop_duplicates()


# In[34]:


df_updated_UserId.info()


# In[35]:


df_updated_UserId.rename(index=str, columns={'TransactionId':'InvoiceNo',
                              'ItemCode':'StockCode',
                              'ItemDescription':'Description',
                              'NumberOfItemsPurchased':'Quantity',
                              'TransactionTime':'InvoiceDate',
                              'CostPerItem':'UnitPrice',
                              'UserId':'CustomerID'
                              }, inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


### BASIC EXPLORATORY DATA ANALYSIS


# In[ ]:


#1.Country


# In[37]:


data=df_updated_UserId.copy()


# In[38]:


data.Country.nunique()


# In[39]:


customer_country=data[['Country','CustomerID']].drop_duplicates()
customer_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)


# In[40]:


# We note that more than 90% of the data is coming from UK !!!


# In[ ]:





# In[41]:


#2.Quantity


# In[42]:


data.describe()


# In[43]:


data[(data['Quantity']<0)].head(5)


# In[44]:


# negative Quantity perhaps refer to cancelled transaction


# In[ ]:





# In[45]:


#3.InvoiceNo - Cancelation Code


# In[46]:


# Constucting a basket for later use
temp = data.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
nb_products_per_basket = temp.rename(columns = {'InvoiceDate':'Number of products'})


# In[47]:


data[(data['Quantity']<0)].count()


# In[48]:


# We see that many rows with negative quantity has an equivalent counterpart with positive quantity implying that 
#it was a misentry, ant the remaining ones with negative quantity implies cancelled transaction
# So we can safely remove them without much effect on our results


# In[49]:


data.drop(data[data['Quantity']<0].index, inplace = True) 


# In[50]:


data[(data['Quantity']<0)].count()


# In[ ]:





# In[51]:


#4.UnitPrice


# In[52]:


data.describe()


# In[53]:


data[(data['UnitPrice'] == 0)].head(5)


# In[54]:


#I am tempted to replace the null values by the most common one but it might be a special discount or 
#something else so I'll leave it like that. But we have already removed the transactions which were cancelled


# In[ ]:





# In[ ]:





# In[ ]:





# In[55]:


### FEATURE ENGINEERING


# In[56]:


def unique_counts(data):
   for i in data.columns:
       count = data[i].nunique()
       print(i, ": ", count)
unique_counts(data)


# In[57]:


#1 Total Price


# In[58]:



data['TotalPrice'] = data['UnitPrice'] * data['Quantity']


# In[59]:


revenue_per_countries = data.groupby(["Country"])["TotalPrice"].sum().sort_values()
revenue_per_countries.plot(kind='barh', figsize=(15,12))
plt.title("Revenue per Country")


# In[60]:


No_invoice_per_country = data.groupby(["Country"])["InvoiceNo"].count().sort_values()
No_invoice_per_country.plot(kind='barh', figsize=(15,12))
plt.title("Number of Invoices per Country")


# In[61]:


#This is very interesting since we can see that Netherlands is the 2nd country in value even though it has 
#less invoices than countries like Germany or France for example and 10 times less customers. 


# In[62]:


le = preprocessing.LabelEncoder()
le.fit(data['Country'])


# In[63]:


l = [i for i in range(37)]
dict(zip(list(le.classes_), l))


# In[64]:


data['Country'] = le.transform(data['Country'])


# In[65]:


with open('labelencoder.pickle', 'wb') as g:
    pickle.dump(le, g)


# In[66]:


data.head(5)


# In[ ]:





# In[67]:


#2 RFM Analysis


# In[68]:


#I'll implement here the RFM principle to classify the customers in this database. 
#RFM stands for Recency, Frequency and Monetary. It is a customer segmentation technique that uses
#past purchase behavior to divide customers into groups.


# In[69]:


data['InvoiceDate'].min()


# In[70]:


data['InvoiceDate'].max()


# In[71]:


max(data['InvoiceDate'])


# In[72]:


# I'll just fix the date to be the beginning of next month after the last month entry in the dataframe
#data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate,format='%m/%d/%Y %H:%M' )




data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])


# In[73]:


# I'll just fix the date to be the beginning of next month after the last month entry in the dataframe
import datetime
import pytz
NOW=pd.Timestamp('2019-03-01 00:00', tz=pytz.FixedOffset(330))


# In[74]:


custom_aggregation = {}
custom_aggregation["InvoiceDate"] = lambda x:x.iloc[0]
custom_aggregation["CustomerID"] = lambda x:x.iloc[0]
custom_aggregation["TotalPrice"] = "sum"


rfmTable = data.groupby("InvoiceNo").agg(custom_aggregation)


# In[75]:


rfmTable["Recency"] = NOW - rfmTable["InvoiceDate"]
rfmTable["Recency"] = pd.to_timedelta(rfmTable["Recency"]).astype("timedelta64[D]")


# In[76]:


custom_aggregation = {}

custom_aggregation["Recency"] = ["min", "max"]
custom_aggregation["InvoiceDate"] = lambda x: len(x)
custom_aggregation["TotalPrice"] = "sum"

rfmTable_final = rfmTable.groupby("CustomerID").agg(custom_aggregation)


# In[77]:


rfmTable.head(5)


# In[78]:


custom_aggregation = {}

custom_aggregation["Recency"] = ["min", "max"]
custom_aggregation["InvoiceDate"] = lambda x: len(x)
custom_aggregation["TotalPrice"] = "sum"

rfmTable_final = rfmTable.groupby("CustomerID").agg(custom_aggregation)


# In[79]:


rfmTable_final.columns = ["min_recency", "max_recency", "frequency", "monetary_value"]


# In[80]:


rfmTable_final.head(5)


# In[81]:


first_customer = data[data['CustomerID']==325731]
first_customer.head(5)


# In[82]:


quantiles = rfmTable_final.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[83]:


segmented_rfm = rfmTable_final


# In[84]:


def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[85]:


#Here we'll apply a score on each feature of RFM


# In[86]:


segmented_rfm['r_quartile'] = segmented_rfm['min_recency'].apply(RScore, args=('min_recency',quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))
segmented_rfm.head()


# In[87]:


#Finally we'll set a score for each customer in the dataframe!!


# In[88]:


segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)
segmented_rfm.head()


# In[89]:


segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False)


# In[90]:


#Here we have an example of customers with a score of 111 which means that they are classified as our best customers.


# In[91]:


segmented_rfm.head(5)


# In[92]:


segmented_rfm = segmented_rfm.reset_index()


# In[93]:


segmented_rfm.head(5)


# In[94]:


data = pd.merge(data,segmented_rfm, on='CustomerID')


# In[95]:


data.columns


# In[96]:


# We don't need the quartiles anymore, let's drop them.


# In[97]:


data = data.drop(columns=['r_quartile', 'f_quartile', 'm_quartile'])


# In[ ]:





# In[98]:


# 3 Time Features


# In[99]:


# I'll now create some time features, although I might not use them. It could be interesting to see if 
# there are any paterns due to seasonality.


# In[100]:


data['Month'] = data["InvoiceDate"].map(lambda x: x.month)


# In[101]:


data['Month'].value_counts()


# In[102]:


data['Weekday'] = data["InvoiceDate"].map(lambda x: x.weekday())
data['Day'] = data["InvoiceDate"].map(lambda x: x.day)
data['Hour'] = data["InvoiceDate"].map(lambda x: x.hour)


# In[103]:


data.head(5)


# In[ ]:





# In[104]:


# 4 Product categories


# In[105]:


X = data["Description"].unique()

stemmer = nltk.stem.porter.PorterStemmer()
stopword = nltk.corpus.stopwords.words('english')

def stem_and_filter(doc):
    tokens = [stemmer.stem(w) for w in analyzer(doc)]
    return [token for token in tokens if token.isalpha()]

analyzer = TfidfVectorizer().build_analyzer()
CV = TfidfVectorizer(lowercase=True, stop_words="english", analyzer=stem_and_filter, min_df=0.00, max_df=0.3)  # we remove words if it appears in more than 30 % of the corpus (not found stopwords like Box, Christmas and so on)
TF_IDF_matrix = CV.fit_transform(X)
print("TF_IDF_matrix :", TF_IDF_matrix.shape, "of", TF_IDF_matrix.dtype)


# In[106]:


svd = TruncatedSVD(n_components = 100)
normalizer = Normalizer(copy=False)

TF_IDF_embedded = svd.fit_transform(TF_IDF_matrix)
TF_IDF_embedded = normalizer.fit_transform(TF_IDF_embedded)
print("TF_IDF_embedded :", TF_IDF_embedded.shape, "of", TF_IDF_embedded.dtype)


# In[107]:


score_tfidf = []

x = list(range(5, 155, 10))

for n_clusters in x:
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10)
    kmeans.fit(TF_IDF_embedded)
    clusters = kmeans.predict(TF_IDF_embedded)
    silhouette_avg = silhouette_score(TF_IDF_embedded, clusters)

    rep = np.histogram(clusters, bins = n_clusters-1)[0]
    score_tfidf.append(silhouette_avg)


# In[108]:


plt.figure(figsize=(20,16))

plt.subplot(2, 1, 1)
plt.plot(x, score_tfidf, label="TF-IDF matrix")
plt.title("Evolution of the Silhouette Score")
plt.legend()


# In[109]:


# The highest value for the silhouette score is when there are 135 clusters. So we'll chose this value.


# In[110]:


n_clusters = 135

kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30, random_state=0)
proj = kmeans.fit_transform(TF_IDF_embedded)
clusters = kmeans.predict(TF_IDF_embedded)
plt.figure(figsize=(10,10))
plt.scatter(proj[:,0], proj[:,1], c=clusters)
plt.title("ACP with 135 clusters", fontsize="20")


# In[111]:


tsne = TSNE(n_components=2)
proj = tsne.fit_transform(TF_IDF_embedded)

plt.figure(figsize=(10,10))
plt.scatter(proj[:,0], proj[:,1], c=clusters)
plt.title("Visualization of the clustering with TSNE", fontsize="20")


# In[112]:


plt.figure(figsize=(20,8))
wc = WordCloud()

for num, cluster in enumerate(random.sample(range(100), 12)) :
    plt.subplot(3, 4, num+1)
    wc.generate(" ".join(X[np.where(clusters==cluster)]))
    plt.imshow(wc, interpolation='bilinear')
    plt.title("Cluster {}".format(cluster))
    plt.axis("off")
plt.figure()


# In[113]:


pd.Series(clusters).hist(bins=100)


# In[114]:


dict_article_to_cluster = {article : cluster for article, cluster in zip(X, clusters)}


# In[115]:


with open('product_clusters.pickle', 'wb') as h:
    pickle.dump(dict_article_to_cluster, h)


# In[ ]:





# In[ ]:





# In[ ]:





# In[116]:


### CREATING CUSTOMER SEGMENTS


# In[117]:


# 1 Intermediate dataset grouped by invoices


# In[118]:


cluster = data['Description'].apply(lambda x : dict_article_to_cluster[x])
df2 = pd.get_dummies(cluster, prefix="Cluster").mul(data["TotalPrice"], 0)
df2 = pd.concat([data['InvoiceNo'], df2], axis=1)
df2_grouped = df2.groupby('InvoiceNo').sum()


# In[119]:


custom_aggregation = {}
custom_aggregation["TotalPrice"] = lambda x:x.iloc[0]
custom_aggregation["min_recency"] = lambda x:x.iloc[0]
custom_aggregation["max_recency"] = lambda x:x.iloc[0]
custom_aggregation["frequency"] = lambda x:x.iloc[0]
custom_aggregation["monetary_value"] = lambda x:x.iloc[0]
custom_aggregation["CustomerID"] = lambda x:x.iloc[0]
custom_aggregation["Quantity"] = "sum"
custom_aggregation["Country"] = lambda x:x.iloc[0]


df_grouped = data.groupby("InvoiceNo").agg(custom_aggregation)


# In[ ]:





# In[120]:


# 2 Final dataset grouped by customers


# In[121]:


df2_grouped_final = pd.concat([df_grouped['CustomerID'], df2_grouped], axis=1).set_index("CustomerID").groupby("CustomerID").sum()
df2_grouped_final = df2_grouped_final.div(df2_grouped_final.sum(axis=1), axis=0)
df2_grouped_final = df2_grouped_final.fillna(0)


# In[122]:


custom_aggregation = {}
custom_aggregation["TotalPrice"] = ['min','max','mean']
custom_aggregation["min_recency"] = lambda x:x.iloc[0]
custom_aggregation["max_recency"] = lambda x:x.iloc[0]
custom_aggregation["frequency"] = lambda x:x.iloc[0]
custom_aggregation["monetary_value"] = lambda x:x.iloc[0]
custom_aggregation["Quantity"] = "sum"
custom_aggregation["Country"] = lambda x:x.iloc[0]

df_grouped_final = df_grouped.groupby("CustomerID").agg(custom_aggregation)


# In[123]:


df_grouped_final.head(5)


# In[124]:


df_grouped_final.columns = ["min", "max", "mean", "min_recency", "max_recency", "frequency", "monetary_value", "quantity", "country"]


# In[125]:


df_grouped_final.head(5)


# In[126]:


df2_grouped_final.head(5)


# In[ ]:





# In[127]:


# 3 Clustering customers


# In[128]:


X1 = df_grouped_final.to_numpy()
X2 = df2_grouped_final.to_numpy()

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
X_final_std_scale = np.concatenate((X1, X2), axis=1)


# In[129]:


x = list(range(2, 11))
y_std = []
for n_clusters in x:
    print("n_clusters =", n_clusters)
    
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10)
    kmeans.fit(X_final_std_scale)
    clusters = kmeans.predict(X_final_std_scale)
    silhouette_avg = silhouette_score(X_final_std_scale, clusters)
    y_std.append(silhouette_avg)
    print("The average silhouette_score is :", silhouette_avg, "with Std Scaling")


# In[130]:


#We want to have at least 5, 6 clusters so we won't take 2 or 3 clusters even though they have the 
#highest silhouette scores, 6 clusters would fit the best here.


# In[131]:


kmeans = KMeans(init='k-means++', n_clusters = 6, n_init=50, random_state=0)  # random state just to be able to provide cluster number durint analysis
kmeans.fit(X_final_std_scale)
clusters = kmeans.predict(X_final_std_scale)


# In[132]:


plt.figure(figsize = (20,8))
n, bins, patches = plt.hist(clusters, bins=6)
plt.xlabel("Cluster")
plt.title("Number of customers per cluster")
plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Cluster {}".format(x) for x in range(6)])

for rect in patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    space = 5
    va = 'bottom'
    label = str(int(y_value))
    
    plt.annotate(
        label,                      
        (x_value, y_value),         
        xytext=(0, space),          
        textcoords="offset points", 
        ha='center',                
        va=va)


# In[133]:


df_grouped_final["cluster"] = clusters


# In[134]:


final_dataset = pd.concat([df_grouped_final, df2_grouped_final], axis = 1)
final_dataset.head()


# In[135]:


final_dataset_V2 = final_dataset.reset_index()


# In[136]:


final_dataset_V2.to_csv("final_solution_dataset.csv",index=False)


# In[137]:


with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)


# In[ ]:





# In[138]:


# 4 Interpreting the clusters


# In[139]:


tsne = TSNE(n_components=2)
proj = tsne.fit_transform(X_final_std_scale)

plt.figure(figsize=(10,10))
plt.scatter(proj[:,0], proj[:,1], c=clusters)
plt.title("Visualization of the clustering with TSNE", fontsize="25")


# In[140]:


#Graphically the clusters are distinctive enough. Let's take a closer look at the clusters 
#that contain few customers.


# In[ ]:





# In[ ]:





# In[ ]:





# In[141]:


### ANALYSING THE MAJOR CLUSTER(Cluster 0) IN DETAIL


# In[142]:


final_dataset[final_dataset['cluster']==0]


# In[143]:


final_dataset[final_dataset['cluster']==0].mean()


# In[144]:


temp_final_df = final_dataset.reset_index()


# In[145]:


cust0 = list(temp_final_df[temp_final_df['cluster']==0]['CustomerID'])


# In[146]:


cluster0 = data[data['CustomerID'].isin(cust0)]
cluster0[['Quantity', 'UnitPrice', 'TotalPrice', 'frequency', 'min_recency'
         , 'monetary_value']].mean()


# In[147]:


cluster0['Description'].value_counts()[:10]


# In[148]:


custom_aggregation = {}
custom_aggregation["Country"] = lambda x:x.iloc[0]
custom_aggregation["RFMScore"] = lambda x:x.iloc[0]

cluster0_grouped = cluster0.groupby("CustomerID").agg(custom_aggregation)


# In[149]:


cluster0_grouped['RFMScore'].value_counts()


# In[150]:


cluster0_grouped['Country'].value_counts()


# In[151]:


cluster0['Month'].value_counts()


# In[152]:


plt.figure(figsize = (20,8))
n, bins, patches = plt.hist(cluster0['Month'], bins=12)
plt.xlabel("Cluster")
plt.title("Number of invoices per month")
plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Month {}".format(x) for x in range(1, 13)])

for rect in patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    space = 5
    va = 'bottom'
    label = str(int(y_value))
    
    plt.annotate(
        label,                      
        (x_value, y_value),         
        xytext=(0, space),          
        textcoords="offset points", 
        ha='center',                
        va=va)


# In[153]:


temp['Year'] = cluster0[cluster0['Month']==12]['InvoiceDate'].map(lambda x: x.year)
temp['Year'].value_counts()


# In[154]:


plt.figure(figsize = (20,8))
n, bins, patches = plt.hist(cluster0['Weekday'], bins=7)
plt.xlabel("Cluster")
plt.title("Number of invoices per day of the week")
plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Day {}".format(x) for x in range(0, 7)])

for rect in patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    space = 5
    va = 'bottom'
    label = str(int(y_value))
    
    plt.annotate(
        label,                      
        (x_value, y_value),         
        xytext=(0, space),          
        textcoords="offset points", 
        ha='center',                
        va=va)


# In[155]:


cluster0['Day'].nunique()


# In[156]:


plt.figure(figsize = (20,8))
n, bins, patches = plt.hist(cluster0['Day'], bins=31)
plt.xlabel("Cluster")
plt.title("Number of invoices per day of the month")
plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Day {}".format(x) for x in range(1,32)])

for rect in patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    space = 5
    va = 'bottom'
    label = str(int(y_value))
    
    plt.annotate(
        label,                      
        (x_value, y_value),         
        xytext=(0, space),          
        textcoords="offset points", 
        ha='center',                
        va=va)


# In[157]:


cluster0['Hour'].nunique()


# In[158]:



plt.figure(figsize = (20,8))
n, bins, patches = plt.hist(cluster0['Hour'], bins=14)
plt.xlabel("Cluster")
plt.title("Number of invoices per hour of the day")
plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Hour {}".format(x) for x in (sorted(cluster0['Hour'].unique()))])

for rect in patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    space = 5
    va = 'bottom'
    label = str(int(y_value))
    
    plt.annotate(
        label,                      
        (x_value, y_value),         
        xytext=(0, space),          
        textcoords="offset points", 
        ha='center',                
        va=va)


# In[ ]:





# In[ ]:




