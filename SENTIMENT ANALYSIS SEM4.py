#!/usr/bin/env python
# coding: utf-8

# In[6]:


from textblob import TextBlob
import tweepy
import sys
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt



# In[7]:


#config = pd.read_csv("./config.csv")
twitterApiKey = 'W64e4SP0Q2uIYrLMCaJ7qwGko'#config ['twitterApiKey'][0]
twitterApiSecret = 'bNSrPgrho2GaQZ35kZEQoUIm9yG6FIiWjZv0lTLAfefJnsUKc1'#config ['twitterApiSecret'][0]
twitterApiAccessToken = '1462723329714884615-T7owFOkzNzXHaGIZRBWXJ1UQQpdIPA'#config ['twitterApiAccessToken'][0]
twitterApiAccessTokenSecret = 'IywjRHc1OUaYXFFb9XbzfQglpVdhX6tmXmIYScQ2ORoPP'#config ['twitterApiAccessTokenSecret'][0]

auth = tweepy.OAuthHandler(twitterApiKey,twitterApiSecret)
auth.set_access_token(twitterApiAccessToken,twitterApiAccessTokenSecret)
twitterApi=tweepy.API(auth,wait_on_rate_limit=True)


# In[8]:



search_term = 'ps5 experience'
tweet_amount = 50
date_since = '2022-4-12'
tweets= tweepy.Cursor(twitterApi.search_tweets,q=search_term,lang='en',until=date_since).items(tweet_amount)


# In[9]:


df =pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['Tweet'])


# In[10]:


df.head(50)


# In[11]:


def cleanUpTweet (txt):
    txt= re.sub(r'@[A-Za-z0-9_]+','',txt)
    txt= re.sub(r'@','',txt)
    txt= re.sub(r'RT: ','',txt)
    txt= re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',txt)
    return txt


# In[12]:


df['Tweet']= df['Tweet'].apply(cleanUpTweet)


# In[13]:


def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity


# In[14]:


def getTextPolarity(txt) :
    return TextBlob(txt).sentiment.polarity


# In[15]:


df['Subjectivity']= df['Tweet'].apply(getTextSubjectivity)
df['Polarity']= df['Tweet'].apply(getTextPolarity)

df.head(50)


# In[16]:


def getTextAnalysis(a):
    if a<0:
        return "negative"
    elif a==0:
        return "neutral"
    else :
        return "positive"


# In[17]:


df["Score"]=df['Polarity'].apply(getTextAnalysis)


# In[18]:


df.head(50)


# In[19]:


positive=df[df['Score']=="positive"]
print(str(positive.shape[0]/(df.shape[0])*100 )+"% of Positive tweets")
pos=positive.shape[0]/df.shape[0]*100


# In[20]:


negative=df[df['Score']=="negative"]
print(str(negative.shape[0]/(df.shape[0])*100 )+"% of negative tweets")
neg=negative.shape[0]/df.shape[0]*100


# In[21]:


neutral=df[df['Score']=="neutral"]
print(str(neutral.shape[0]/(df.shape[0])*100 )+"% of neutral tweets")
neutral1=neutral.shape[0]/df.shape[0]*100


# In[22]:


explode=(0,0.1,0)
labels='positive','negative','neutral'
sizes=[pos,neg,neutral1]
colors=['yellowgreen','lightcoral','gold']


# In[23]:


plt.pie(sizes,explode=explode,colors=colors,autopct='%1.1f%%',startangle=120)
plt.legend(labels,loc=(-0.5,0.5),shadow=True)
plt.axis('equal')
plt.savefig("Sentiment_Analysis.png")


# In[30]:


labels = df.groupby('Score').count().index.values
values = df.groupby('Score').size().values
plt.bar(labels,values)


# In[33]:


for index, row in df.iterrows():
    if row['Score']=='positive':
        plt.scatter (row['Polarity'], row['Subjectivity'],color='green') 
    elif row['Score']=='negative':
        plt.scatter (row['Polarity'], row['Subjectivity'],color='red')
    elif row['Score']=='neutral':
        plt.scatter (row['Polarity'], row[ 'Subjectivity' ],color='blue')
plt.title('Twitter Sentiment Analysis')
plt.xlabel('Polarity') 
plt.ylabel('Subjectivity') 
plt.show()


# In[ ]:





# In[ ]:




