#!/usr/bin/env python
# coding: utf-8

# 
# Shark Tank India is an Indian Hindi-language business reality television series that airs on Sony Entertainment Television. The show is the Indian franchise of the American show Shark Tank. It shows entrepreneurs making business presentations to a panel of investors or sharks, who decide whether to invest in their company.

# In[ ]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# In[ ]:


df=pd.read_csv("D:\python_datascience\data sets\Shark Tank India.csv")


# In[ ]:


df.shape


# ## Observations
# 
# * There are total **64 Attributes/columns** available in the dataset.
# * There are total **320 Records/Rows** available in the dataset.

# In[ ]:


df.sample(2)  ## fetching 5 sample records/rows


# ## Shark Tank India Data set Description
# Shark Tank India - Season 1 and Season 2 information, with 64 fields/columns and 320+ records.
# 
# One season of 🦈 SHARKTANK INDIA 🇮🇳 was broadcasted on SonyLiv OTT/Sony TV.
# 
# Here is the data dictionary for Shark Tank (India) season's dataset.
# 
# - Season Number - Season number
# - Season Start - Season first aired date
# - Season End - Season last aired date
# - Episode Number - Episode number within the season
# - Episode Title - Episode title in SonyLiv
# - Pitch Number - Overall pitch number
# - Startup Name - Startup company name
# - Industry - Industry name or type
# - Business Description - Business Description
# - Company Website - Company Website URL
# - Number of Presenters - Number of presenters
# - Male Presenters - Number of male presenters
# - Female Presenters - Number of female presenters
# - Transgender Presenters - Number of transgender/LGBTQ presenters
# - Couple Presenters - Are presenters wife/husband ? 1-yes, 0-no
# - Pitchers Average Age - Pitchers average age, <30 young, 30-50 middle, >50 old
# - Started in - Year in which startup was started/incorporated
# - Pitchers City - Presenter's town/city
# - Pitchers State - Indian state pitcher hails from
# - Yearly Revenue - Yearly revenue, in lakhs INR, -1 means negative revenue, 0 means pre-revenue
# - Monthly Sales - Total monthly sales, in lakhs
# - Gross Margin - Gross margin/profit of company, in percentages
# - Net Margin - Net margin/profit of company, in percentages
# - Original Ask Amount - Original Ask Amount, in lakhs INR
# - Original Offered Equity - Original Offered Equity, in percentages
# - Valuation Requested - Valuation Requested, in lakhs INR
# - Received Offer - Received offer or not, 1-received, 0-not received
# - Accepted Offer - Accepted offer or not, 1-accepted, 0-rejected
# - Total Deal Amount - Total Deal Amount, in lakhs INR
# - Total Deal Equity - Total Deal Equity, in percentages
# - Total Deal Debt - Total Deal Debt, in lakhs INR
# - Debt Interest - Debt interest rate, in percentages
# - Deal Valuation - Deal Valuation, in lakhs INR
# - Number of sharks in deal - Number of sharks involved in deal
# - Deal has conditions - Deal has conditions or not?
# - Has Patents - Pitcher has Patents? 1-yes, 0-no
# - Ashneer Investment Amount - Ashneer Investment Amount, in lakhs INR
# - Ashneer Investment Equity - Ashneer Investment Equity, in percentages
# - Ashneer Debt Amount - Ashneer Debt Amount, in lakhs INR
# - Namita Investment Amount - Namita Investment Amount, in lakhs INR
# - Namita Investment Equity - Namita Investment Equity, in percentages
# - Namita Debt Amount - Namita Debt Amount, in lakhs INR
# - Anupam Investment Amount - Anupam Investment Amount, in lakhs INR
# - Anupam Investment Equity - Anupam Investment Equity, in percentages
# - Anupam Debt Amount - Anupam Debt Amount, in lakhs INR
# - Vineeta Investment Amount - Vineeta Investment Amount, in lakhs INR
# - Vineeta Investment Equity - Vineeta Investment Equity, in percentages
# - Vineeta Debt Amount - Vineeta Debt Amount, in lakhs INR
# - Aman Investment Amount - Aman Investment Amount, in lakhs INR
# - Aman Investment Equity - Aman Investment Equity, in percentages
# - Aman Debt Amount - Aman Debt Amount, in lakhs INR
# - Peyush Investment Amount - Peyush Investment Amount, in lakhs INR
# - Peyush Investment Equity - Peyush Investment Equity, in percentages
# - Peyush Debt Amount - Peyush Debt Amount, in lakhs INR
# - Ghazal Investment Amount - Ghazal Investment Amount, in lakhs INR
# - Ghazal Investment Equity - Ghazal Investment Equity, in percentages
# - Ghazal Debt Amount - Ghazal Debt Amount, in lakhs INR
# - Amit Investment Amount - Amit Investment Amount, in lakhs INR
# - Amit Investment Equity - Amit Investment Equity, in percentages
# - Amit Debt Amount - Amit Debt Amount, in lakhs INR
# - Guest Investment Amount - Guest Investment Amount, in lakhs INR
# - Guest Investment Equity - Guest Investment Equity, in percentages
# - Guest Debt Amount - Guest Debt Amount, in lakhs INR
# - Guest Name - Name of Guest

# In[ ]:


df1=df.dtypes.to_frame()


# In[ ]:


df1.rename(columns={0:"datatypes"},inplace=True)
df1.T


# ## Observation
# 
# *  The colunms dtypes are of mix types , we have float , object, int .
# *  Column name **started in**  indicates startup year that should not be in float . so we have to convert it into int
#    Columns name **Has patents**,**Male Presenters**,**Female Presenters**,**Transgender Presenters**,**Couple Presenters**   should not be in float . so we have to convert them into int
# 
# 

# In[ ]:


df.info()


# ## Observation
# 
# *  Some columns have missing values
# *  columns of float types: 47
# *  columns of int types : 5
# *  columns of object types : 12    
# *  memory occupied by this data set : 160.1 KB

# ## Checking missing values of each columns

# In[ ]:


df.isnull().sum().to_frame().rename(columns={0:"missing values count"}).T


# ## Checking missing values % of each column

# In[ ]:


for i in df.columns:
    if df[i].isnull().sum()>0:
        print(i,"---------------",df[i].isnull().sum()*100/df.shape[0],"%")


# ### Checking the missing values of those columns which are of object type

# In[ ]:


for i in df.columns:
    if df[i].isnull().sum()>0 and df[i].dtype=="object":
        print(i,"---------------",df[i].isnull().sum()*100/df.shape[0],"%")


# ## since company website is not relevant with respect to analysis so we drop it

# In[ ]:


df.drop(columns=["Company Website"],inplace=True)   ## feature engineering


# ## Imputing the missing values in Deal has conditions columns

# In[ ]:


df["Deal has conditions"].unique()


# here nan indicates no conditions so fill nan in Deal has conditions

# In[ ]:


df["Deal has conditions"]=df["Deal has conditions"].fillna("no")


# In[ ]:


## verifying the result
df["Deal has conditions"].unique()


# ## Imputing the missing values in Guest Name  column

# In[ ]:


df["Guest Name"]=df["Guest Name"].fillna("not present")


# In[ ]:


df["Guest Name"].isnull().sum()


# ## missing values % of columns which are of numeric type

# In[ ]:


for i in df.columns:
    if df[i].isnull().sum()>0 and (df[i].dtype=="int32" or df[i].dtype=="float64"):
        print(i,"---------------",df[i].isnull().sum()*100/df.shape[0],"%")


# ## though some columns has missing values % greater than 70% but we wont drop them as they are important columns so fill the misiing values of numeric columns as per domain knowledge

# Taking  **male presenters,Female Presenters,Transgender Presenters, Couple Presenters** columns

# In[ ]:


df[["Number of Presenters","Male Presenters","Female Presenters","Transgender Presenters","Couple Presenters"]]


# ## from this above output we can see that NaN indicates 0 so we fill NaN with 0 in male presenters ,female  presenters,couple presenters , transgenders

# In[ ]:


presenters=["Male Presenters","Female Presenters","Couple Presenters","Transgender Presenters"]
for i in presenters:
    df[i].fillna(0,inplace=True)


# ## verifying the results whether the missing values in above mentioned columns has been filled  or not

# In[ ]:


df[presenters].isnull().sum().sum()


# In[ ]:


df[presenters].dtypes


# ## since the datatypes of above is float they should be in int as presenters count would be discrete so convert them in int

# In[ ]:


df[presenters]=df[presenters].astype(int)   ## do necessary column conversion


# In[ ]:


df[presenters].dtypes


# Taking **Started in** column

# In[ ]:


df["Started in"].unique()


# In[ ]:


df["Started in"].isnull().sum()


# fill nan in started in columns with 0 as nan indicates brands did nt mention their start up year  or some pitchers were about to start

# In[ ]:


df["Started in"]=df["Started in"].fillna(0)
df["Started in"].unique()


# In[ ]:


df["Started in"]=df["Started in"].astype(int)   ## convert started in column in int as years should be discrete type so change it  in int


# In[ ]:


df["Started in"].dtypes


# Taking **Yearly Revenue ,Monthly Sales , Gross Margin ,Net Margin** columns

# ## checking the correlation to see relation between them

# In[ ]:


df[["Yearly Revenue","Monthly Sales","Gross Margin","Net Margin"]].corr()


# from this we can see there a moderate positive correlation between monthly sales and yearly  sales and with rest columns theere is no relation

# In[ ]:


df[["Yearly Revenue","Monthly Sales","Gross Margin","Net Margin"]]


# from this above output nan indicates not available info but these columns are in float so we fill them with 0

# In[ ]:


df[["Yearly Revenue","Monthly Sales","Gross Margin","Net Margin"]]=df[["Yearly Revenue","Monthly Sales","Gross Margin","Net Margin"]].fillna(0)


# Taking **Accepted offer**  column

# In[ ]:


df["Accepted Offer"].unique()


# In[ ]:


df[["Received Offer","Accepted Offer","Total Deal Amount"]]


# nan in accepted offers indicates deals were not finalized so we fill nan with 0

# In[ ]:


df["Accepted Offer"]=df["Accepted Offer"].fillna(0)
df["Accepted Offer"].isnull().sum()


# In[ ]:


df["Accepted Offer"].dtypes


# #### the dtypes of Accepted Offer column to int as it should be in discrete

# In[ ]:


df["Accepted Offer"]=df["Accepted Offer"].astype(int)


# In[ ]:


df["Accepted Offer"].dtype   ## verifying the result


# ## taking Has patent column

# In[ ]:


df["Has Patents"].isnull().sum()


# #### In Patent Has column  Nan indicates that brand did not have any patents so we fill nan with 0

# In[ ]:


df["Has Patents"]=df["Has Patents"].fillna(0)


# In[ ]:


df["Has Patents"].dtypes


# In[ ]:


df["Has Patents"]=df["Has Patents"].astype(int)  ## convert it into int


# #### Taking Total Deal Amount","Original Offered Equity","Total Deal Equity","Total Deal Debt","Debt Interest","Deal Valuation column

# In[ ]:


df[["Original Ask Amount","Total Deal Amount","Original Offered Equity","Total Deal Equity","Total Deal Debt","Debt Interest","Deal Valuation","Accepted Offer"]]


# here nan indicates deals were not finalized no we fill with 0

# In[ ]:


df[["Total Deal Amount","Original Offered Equity","Total Deal Equity","Total Deal Debt","Debt Interest","Deal Valuation"]]=df[["Total Deal Amount","Original Offered Equity","Total Deal Equity","Total Deal Debt","Debt Interest","Deal Valuation"]].fillna(0)


# In[ ]:


## verifying the results


# In[ ]:


df[["Total Deal Amount","Original Offered Equity","Total Deal Equity","Total Deal Debt","Debt Interest","Deal Valuation"]].isnull().sum()


# ## Taking sharks

# In[ ]:


df[["Number of sharks in deal","Ashneer Investment Amount","Namita Investment Amount","Anupam Investment Amount","Vineeta Investment Amount","Aman Investment Amount","Peyush Investment Amount","Ghazal Investment Amount","Amit Investment Amount"]]


# nan in Number of sharks in deal columns indicates deals were not finalized  so we fill with 0

# In[ ]:


df["Number of sharks in deal"]=df["Number of sharks in deal"].fillna(0)


# In[ ]:


df.columns


# ### Taking Ashneer Investment Amount,Namita Investment Amount,Anupam Investment Amount,Vineeta Investment Amount,Aman Investment Amount,Peyush Investment Amount,Ghazal Investment Amount,Amit Investment Amount columns,Guest Investment Amount', 'Guest Investment Equity','Guest Debt Amount

# In[ ]:


df.dtypes


# In[ ]:


df.columns[35:-1]


# nan in above columns indicates some deals werenot accpeted by corresponding sharks

# In[ ]:


df[df.columns[35:-1]]=df[df.columns[35:-1]].fillna(0)


# In[ ]:


df[df.columns[35:-1]].isnull().sum().sum()  ## verifying the result


# ### verifying whether the  missing values have been filled or not

# In[ ]:


df.isnull().sum().sum()


# ## the data is now free from missing values

# ## checking duplicacy in the data

# In[ ]:


df.duplicated().sum()


# <div style="border-radius:10px; border:#CD5C5C solid; padding: 15px; background-color: #FFFAF1; font-size:100%; text-align:left">
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:pink;
#            font-size:200%;
#            font-family:Serif;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#           color:black;
#           font-size:120%;
#           text-align:center;">
# Exploratory Data Analysis
# </div>

# ## Descriptive stats

# In[ ]:


df.describe().T


# In[ ]:


df.head(2)


# ## How many  seasons of Shark Tank India

# In[ ]:


pd.DataFrame(df["Season Number"].unique(),columns=["seasons"],index=["season1","season2"])


# #### Insight : 2 seasons

# ## when both the seasons of shark  Tank India aired

# In[ ]:


season1=df["Season Start"].unique()
season1


# In[ ]:


season2=df["Season End"].unique()
season2


# In[ ]:


x=pd.DataFrame([[season1[0],season2[0]],[season1[1],season2[1]]],index=["season1","season2"],columns=["start_aired_date","end_aired_date"])
x


# ### Insight:
# - season 1: 20 -Dec-2021   to 04-feb-2022
# - season 2 : 2-jan-2023  to 10-march-2023

# ## Total number of episodes telecasted till now

# In[ ]:


df[["Season Number","Episode Number","Pitch Number"]]


# In[ ]:


df["Episode Number"].unique()


# there is episode number is 0 means unseen pitch so we exclude unseen pitches in order to find out the total number of episodes

# In[ ]:


df[df["Episode Number"]==0].head()


# In[ ]:


episodes=df[df["Episode Number"]!=0]  ## fetching telecasted episodes
episodes


# In[ ]:


#episodes["Episode Number"].unique()


# ## Total number of episodes in each season

# In[ ]:


df[["Season Number","Episode Number","Pitch Number"]]


# there is episode number is 0 means unseen pitch so we exclude unseen pitches in order to find out the total number of episodes of each season :

# In[ ]:


episodes.groupby(["Season Number"])["Episode Number"].nunique()


# In[ ]:


import numpy as np
np.sum(episodes.groupby(["Season Number"])["Episode Number"].nunique().values)


# #### Insights:
# ## Total number of episodes :87

# In[ ]:


x["no_of_episodes"]=episodes.groupby(["Season Number"])["Episode Number"].nunique().values


# In[ ]:


x


# ### Insight:
# - season 1 : no episodes : 36
# - season 2 : no episodes : 51

# ## find the total number of enterprenuers who presented their ideas in Shark tank India

# In[ ]:


df.iloc[:,0:8]


# In[ ]:


df["Pitch Number"].nunique()


# ####  Insight:
# - 320 entrepreneurs participated in the shark Tank

# ## Out of 320, How many enterprenuers participated in each seasons?

# In[ ]:


df.groupby(["Season Number"])["Pitch Number"].count()


# In[ ]:


plt.pie(df.groupby(["Season Number"])["Pitch Number"].count().values,labels=df.groupby(["Season Number"])["Pitch Number"].count().index,autopct="%.2f%%",colors=sns.color_palette("muted"));


# ####  Insight:
# - season 1: 52.5% entrepreneurs partcipated while in season 2 while in season 1 47% of entrepreneurs paricipated
# 
# 
# -- no of  presenters  in season 2 were higher than season 1

# ## Total pitches in each season excluding unseen
# or
# ## How many's brands participated in each season excluding non telecasted pitch

# In[ ]:


episodes.groupby(["Season Number"])["Pitch Number"].count()


# In[ ]:


plt.pie(episodes.groupby(["Season Number"])["Pitch Number"].count().values,labels=episodes.groupby(["Season Number"])["Pitch Number"].count().index,autopct="%.2f%%",colors=sns.color_palette("muted"))
circle = plt.Circle( (0,0), 0.4, color='white')
p=plt.gcf()
p.gca().add_artist(circle)


# In[ ]:


df.head(2)


# In[ ]:


df[["Season Number","Episode Number","Pitch Number"]]


# ## How many brands participated in each epsiode of each season

# In[ ]:


df.groupby(["Season Number","Episode Number"])["Pitch Number"].agg(["count"]).sort_values(by="count",ascending=False)


# ## Insight:
# - max no of brands participated per episode: 4 except unseen pitches
# - Min. number of brands participated per episode : 2

# ## Industry

# In[ ]:


df.head(2)


# In[ ]:


df["Industry"].unique()


# In[ ]:


df[["Season Number","Industry"]]


# ## of which sector/Industry brands participated

# In[ ]:


sns.countplot(x="Industry",data=df,palette="muted")
plt.xticks(rotation=90);


# #### Insight:
# Entrpreneurs from food industry particpated most while from hardware and entertainment sectors were less

# ## Of which sector brands participated with respect to each season

# In[ ]:


plt.figure(figsize=(10,9))
sns.countplot(x="Season Number",data=df,hue="Industry",palette="muted")
plt.xticks(rotation=90);


# ## Insight:
# Entrpreneurs from food sector in both the season participated most

# In[ ]:


df.head(2)


# ## Which  team  size particapted most
# 

# In[ ]:


sns.countplot(df["Number of Presenters"],palette="muted");


# #### Insight:
# Team size of 2 participated most

# In[ ]:


df[((df["Number of Presenters"]==6) | (df["Number of Presenters"]==5))]


# ## Team participation size in each seasons

# In[ ]:


plt.figure(figsize=(10,4))

sns.countplot(x="Season Number",palette="pastel",hue="Number of Presenters",data=df)


# Team of 2 participated most in both season and only one team with 5 or 6 members  participated in season 2 and season 1 respectively.

# In[ ]:


df.head(2)


# ## How many % were couples ?

# In[ ]:


df["Couple Presenters"].unique()


# In[ ]:


df["Couple Presenters"].value_counts()


# In[ ]:


plt.pie(df["Couple Presenters"].value_counts().values,labels=df["Couple Presenters"].unique(),autopct="%.2f%%",colors=sns.color_palette("muted"));


# In[ ]:


df.groupby(["Season Number"])["Couple Presenters"].value_counts()


# ## who participated most  male,female,transgender

# In[ ]:


df.head(2)


# In[ ]:


l=["Male Presenters","Female Presenters","Transgender Presenters"]
l1=[]
for i in l:
    l1.append(df[i].sum())
l1


# In[ ]:


plt.pie(l1,labels=l,autopct="%.2f%%",colors=sns.color_palette("muted"),explode=[0,0,2]);


# #### Insight:
# male entrepreneurs participated most

# In[ ]:


df["Transgender Presenters"].unique()


# In[ ]:


df[df["Transgender Presenters"]==1]


# ## Of which age entreprenurs participated  most and least

# In[ ]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(x="Pitchers Average Age",data=df,palette="muted")
plt.title("pitcher's age count")
plt.subplot(1,2,2)
plt.pie(df["Pitchers Average Age"].value_counts().values,labels=df["Pitchers Average Age"].unique(),autopct="%.2f%%",explode=[0,0,2],colors=sns.color_palette("muted"))
plt.title("pitcher's age distribution");


# #### Insight:
# - 72% of pitchers were of  middle age participated
# - mostly middle age participated most

# In[ ]:


df.head(2)


# In[ ]:


df["Pitchers City"].nunique()


# ## of which states  Pitchers participated most and least

# In[ ]:


df["Pitchers State"].unique()


# In[ ]:


df["Pitchers State"].nunique()


# In[ ]:


plt.figure(figsize=(16,4))
sns.countplot(x="Pitchers State",data=df)
plt.xticks(rotation=90);


# In[ ]:


df["Pitchers State"]


# In[ ]:


states=df["Pitchers State"].values
state = ' '.join(states)
state


# In[ ]:


#pip install wordcloud


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


wordcloud = WordCloud(width=800, height=800, background_color='black').generate(state)
plt.figure(figsize=(14,8), facecolor=None)
plt.imshow(wordcloud)
#plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


# ## or

# In[ ]:


plt.figure(figsize=(7,5))
sns.countplot(x="Pitchers State",data=df)
plt.xticks(rotation=90);


# #### Insight:
# from Mahrashtra entrepreneurs partcipated most

# In[ ]:


df["Pitchers State"].value_counts()


# ## to see the distribution of each in %  use pie chart

# In[ ]:


plt.figure(figsize=(15,15))
plt.pie(x=df["Pitchers State"].value_counts().values,labels=df["Pitchers State"].value_counts().index,autopct="%.2f%%");


# ## because of too many categories as states so pie chart becomes messy so we do the feature engineering in such way that number of categories is reduced

# In[ ]:


df["Pitchers State"].unique()


# from above output we can see in some team each entreprenurs came from different state like in Karnataka,West Bengal so wo are making these states as hybrid state as follows

# In[ ]:


df1=df.copy()
df1


# In[ ]:


index=0
for i in df1["Pitchers State"]:
    if "," in i:
        df1.loc[index,"Pitchers State"]="hybrid state"
    else:
        df1.loc[index,"Pitchers State"]=i
    index=index+1


# In[ ]:


df1["Pitchers State"].unique()


# In[ ]:


plt.figure(figsize=(15,15))
plt.pie(x=df1["Pitchers State"].value_counts().values,labels=df1["Pitchers State"].value_counts().index,autopct="%.2f%%");


# ## still messyness is there in above output we further reduce the categories as below

# ### make  new  others categories/state  for  those state whose corresponding  count less than 5

# In[ ]:


df1["Pitchers State"].value_counts()


# In[ ]:


l=["Madhya Pradesh","Bihar","Jammu & Kashmir","Goa","Kerala","Uttarakhand","Himachal Pradesh","Jharkhand","Chhattisgarh","Arunachal Pradesh"]
index=0
for i in df1["Pitchers State"]:
    if i in l:
        df1.loc[index,"Pitchers State"]="others"
    index=index+1


# In[ ]:


df1["Pitchers State"].nunique()


# In[ ]:


df1["Pitchers State"].value_counts()


# In[ ]:


plt.figure(figsize=(10,8))
plt.pie(x=df1["Pitchers State"].value_counts().values,labels=df1["Pitchers State"].value_counts().index,autopct="%.2f%%",colors=sns.color_palette("muted"));


# ## Top 5 participating states

# In[ ]:


df1["Pitchers State"].value_counts().head(5)


# In[ ]:


plt.bar(df1["Pitchers State"].value_counts().head(5).index,df1["Pitchers State"].value_counts().head(5).values,color="yellow",edgecolor="red")
plt.tight_layout()


# #### Insight:
# - Top 5 participating states:
#    -  Mahrashtra
#    -  Delhi
#    -  Karnatka
#    -  Gujarat
#    -  others is treated as miscellaneous

# ## most and least active participating city with respect to above states

# In[ ]:


x=df1[((df1["Pitchers State"]=="Maharashtra")|(df["Pitchers State"]=="Delhi")|(df["Pitchers State"]=="Karnataka")|(df["Pitchers State"]=="Gujarat")|(df["Pitchers State"]=="others"))]


# In[ ]:


x


# In[ ]:


plt.figure(figsize=(14,9))
sns.countplot(x="Pitchers State",data=x,hue="Pitchers City",palette="muted");
plt.xticks(rotation=90);


# ##  Feature engineering:
# - adding  zones  column by doing feature extraction from pitcher state

# In[ ]:


df["Pitchers State"].unique()


# In[ ]:


north=["Delhi","Punjab","Delhi,Punjab","Haryana","Jammu & Kashmir","Uttar Pradesh","Uttarakhand","Himachal Pradesh","Uttarakhand,Uttar Pradesh"]
central=["Madhya Pradesh","Chhattisgarh"]
south=["Karnataka","Telangana",'Kerala',"Tamil Nadu","Goa","Karnataka,Telangana","Karnataka,Andhra Pradesh"]
west=["Gujarat","Maharashtra","Rajasthan"]
east=["Bihar","West Bengal","Jharkhand","Arunachal Pradesh"]
hybrid=["Karnataka,West Bengal","Delhi,Maharashtra","Haryana,Madhya Pradesh","Telangana,Maharashtra","Kerala,Maharashtra","Haryana,West Bengal","Haryana,Maharashtra","Jharkhand,Chhattisgarh","Gujarat,Uttar Pradesh"]



# In[ ]:


index=0
df["zones"]=pd.Series()

for i in df["Pitchers State"]:
    if i in north:
        df.loc[index,"zones"]="north"
    elif i in south:
        df.loc[index,"zones"]="south"
    elif i in central:
        df.loc[index,"zones"]="central"
    elif i in west:
        df.loc[index,"zones"]="west"
    elif i in east:
        df.loc[index,"zones"]="east"

    else:
        df.loc[index,"zones"]="hybrid zone"
    index=index+1


# In[ ]:


df.head(2)


# In[ ]:


df[["Pitchers State","zones"]]


# ## Most and Least Actively participating zone

# In[ ]:


df["zones"].value_counts()


# In[ ]:


plt.figure(figsize=(8,5))
plt.pie(df["zones"].value_counts().values,labels=df["zones"].value_counts().index,autopct="%.2f%%",colors=sns.color_palette("muted"),explode=[0,0,0,0,2,2])
circle = plt.Circle( (0,0), 0.4, color='white')
p=plt.gcf()
p.gca().add_artist(circle);


# #### Insight:
# - actively participating zone is west
# - least participating is central zone

# In[ ]:


df.head(2)


#  ## How many enterprenurs patent their ideas

# In[ ]:


df["Has Patents"].value_counts()


# ## only 7 entreprenurs has patents . Who were they ?

# In[ ]:


df[df["Has Patents"]==1].iloc[:,[6,7,8,16,17]]


# ## Which was the oldest and latest startup participated in shark tank

# In[ ]:


df["Started in"].min()


# In[ ]:


year=df[df["Started in"]!=0]  ## 0 means no information or about to start
year


# In[ ]:


year[year["Started in"]==year["Started in"].min()]


# startname : Agritourism
# startup year: 2005

# In[ ]:


year[year["Started in"]==year["Started in"].max()]


# ## latest year is 2022

# 

# ## How many deals were finalized

# In[ ]:


df["Accepted Offer"].value_counts()


# - 176 deals were accepted
# - 144 deals were rejected
# 
# mostly deals were finalized

# In[ ]:


sns.countplot(x="Accepted Offer",data=df,palette="muted")


# ###  The ideas and details of brands whose deals were accpted.

# In[ ]:


df_ac=df[df["Accepted Offer"]==1]
df_ac[["Startup Name","Industry","Business Description"]]


# ## How many deals were finalized in both season

# In[ ]:


df.groupby(["Season Number"])["Accepted Offer"].value_counts()


# In[ ]:


sns.countplot(x="Season Number",data=df,palette="muted",hue="Accepted Offer");


# #### Insight:
# - in  season 1 - more deals were rejected than accepted
# -  in season 2 - more deals were accepted than rejected

# ## How many deals were finalized in both season in seen piches

# In[ ]:


#episodes_ac=episodes[episodes["Accepted Offer"]==1]


# In[ ]:


sns.countplot(x="Season Number",data=episodes,palette="muted",hue="Accepted Offer");


# #### Insight:
# - in  season 1 - more deals were accepted than rejected
# -  in season 2 - more deals were accepted than rejected

# ### How many offers were rejected

# In[ ]:


len(df[df["Received Offer"]!=df["Accepted Offer"]])


# ## Details of thoses whose offers have been rejected

# In[ ]:


df[df["Received Offer"]!=df["Accepted Offer"]]


# In[ ]:


df_ac  ## data where their offers were   accepted


# ## Find the highest deal amount that were accepted

# In[ ]:


#df[["Original Ask Amount","Accepted Offer","Total Deal Amount"]]


# In[ ]:


df_ac["Total Deal Amount"].max()


# ## Details of highest deal amount

# In[ ]:


df_ac[df_ac["Total Deal Amount"]==df["Total Deal Amount"].max()]


# ### highest deal amount accepted for startup name and their corresponding ideas  
# 
# "UnStop"-Connecting talent colleges recruiters
# "MeduLance" - One-stop solution for all healthcare needs
# "Pharmallama" - Simplified pharmacy
# 
# 
# these deals were from season season 2 with  was accpted by 4 sharks :
# 
# 
# - Namita
# - Anupam ,
# - Aman
# - Amit  
# 
# who gave them an investment of Rs 200 lakhs  for 4 per cent equity, by far the highest deal on Shark Tank India.
# 

# ###  Find the lowest deal amount that were accepted

# In[ ]:


df_ac["Total Deal Amount"].min()


# In[ ]:


df_ac[df_ac["Total Deal Amount"]==df_ac["Total Deal Amount"].min()]


# since 0 rs in deal amont shows not an invalid entry this was case for the pitchers who demanded 100 hours from sharks for 0.5% equity.

# In[ ]:


x=df_ac[df_ac["Total Deal Amount"]!=0]
x


# In[ ]:


x[x["Total Deal Amount"]==x["Total Deal Amount"].min()]


# ### now find the min deal amount that had been accepted

# In[ ]:


x["Total Deal Amount"].min()


# "Cocofit"-Coconut based beverage franchise
# 
# 
# 
# this deal was from  season 1 which was accpted by 3 sharks :
#     
# 
# - Namita
# - Anupam
# - Aman
# 
# who gave them an investment of Rs 5   for 5 per cent equity, by far the highest deal on Shark Tank India.
# 

# ## min and max deal amount accpted by sharks of both season

# In[ ]:


x.groupby(["Season Number"])["Total Deal Amount"].agg(["max","min"])


# In[ ]:


sns.heatmap(x.groupby(["Season Number"])["Total Deal Amount"].agg(["max","min"]),annot=True)


# ## Find most dealing episodes of both the season

# In[ ]:


best_episodes=df_ac.groupby(["Season Number",'Episode Number'])['Accepted Offer'].sum()
best_episodes.to_frame().sort_values(by=["Season Number","Accepted Offer"],ascending=False)


# ### Find the most liking episodes

# In[ ]:


df_ac.groupby(["Season Number","Episode Number"])["Number of sharks in deal"].max().to_frame().sort_values(by=["Season Number","Number of sharks in deal"],ascending=False)


# ## Find Most Expensive dealing Episodes

# In[ ]:


df_ac[["Season Number",'Episode Number',"Total Deal Amount"]]


# In[ ]:


expensive_episodes=df_ac.groupby(["Season Number",'Episode Number'])["Total Deal Amount"].sum().reset_index().sort_values(by=["Season Number","Total Deal Amount"],ascending=False)
expensive_episodes


# ### most expensive episode :
# 
# Episode 20  of season 2
# 
# Episode 17  of season 1
# 
# ### least expensive episode :
# 
# Episode 21  of season 2
# 
# Episode 18  of season 1
# 
# 

# In[ ]:


df.head(2)


# In[ ]:


df.columns


# ### How many sharks participated in this show and What were their names ?¶
# 

# In[ ]:


sharks_names=[]

for i in df.columns[35:-5:3]:
    sharks=i.split(maxsplit=1)
    #print(sharks)
    #print(sharks[0])
    sharks_names.append(sharks[0])
print(len(sharks_names), " sharks participated \n")
print("following are the names \n\n",sharks_names)



# ## How many sharks participated in each season
# 
# 

# In[ ]:


df_1=df[df["Season Number"]==1]  ## df_1 for season 1
for i in df.columns[35:-5:3]:
    if df_1[i].sum()==0:
        print(i.split(maxsplit=1)[0],"did not participate in season 1   \n")
    else:
        print(i.split(maxsplit=1)[0],"participated in season 1\n")



# In[ ]:


df_2=df[df["Season Number"]==2]  ## df_2 for season 2
for i in df.columns[35:-4:3]:
    if df_2[i].sum()==0:
        print(i.split(maxsplit=1)[0],"did not participate in season 2   \n")
    else:
        print(i.split(maxsplit=1)[0],"participated in season 2\n")


# #### insight
# 
# - shark Amit did not participate in season 1  
# - Sharks Ashneer and Ghazal did not participate in season 2

# ## How much total amount each investers/sharks invested in the the deals?

# In[ ]:


#df.columns[35:-4:3]


# In[ ]:


#df[df.columns[35:-4:3]]


# In[ ]:


l=[]
for i in df.columns[35:-5:3]:
    s=df[i].sum()

    l.append(s)
l


# In[ ]:


plt.bar(sharks_names,l,color="yellow",edgecolor="red")
plt.tight_layout()


# - aman invested the highest amount in the show
# - ghazal invested the least amount in the show

# ## How much total amount each investers/sharks invested in the the deals in each season?

# In[ ]:


l_1=[]
for i in df_1.columns[35:-5:3]:  ## df_1 for season 1
    s=df_1[i].sum()

    l_1.append(s)
l_1


# In[ ]:


l_2=[]
for i in df_2.columns[35:-5:3]:
    s=df_2[i].sum()

    l_2.append(s)
l_2


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.bar(sharks_names,l_1,color="pink",edgecolor="red")
plt.title("Total amount invested by sharks in season 1 ",fontweight="black",pad=5,size=15,color="Red")

plt.subplot(1,2,2)
plt.bar(sharks_names,l_2,color="cyan",edgecolor="red")
plt.title("Total amount invested by sharks in season 2 ",fontweight="black",pad=5,size=15,color="Red")
plt.show()


# ####  In season 1 Aman invested the highest amount and ghazal invested least , no amount is showing correspond to Amit since Amit did not participate the season 1
# #### In season 2 Aman invested the highest amount and Vineeta,Amit invested least , no amount is showing correspond to Ashneer and Ghazal since they did not participate the season 2
# 
# ### Aman found to be highest invester

# ## Find the equity percent that each sharks gets

# ## How much total equity each investers/sharks gets in the show?

# In[ ]:


df.columns[36:-4:3]


# In[ ]:


l=[]
for i in df.columns[36:-4:3]:
    s=df[i].sum()

    l.append(s)
l


# In[ ]:


plt.bar(sharks_names,l,color="yellow",edgecolor="red")
plt.tight_layout()


# ## How much total equity each shark gets in each season

# In[ ]:


l_1=[]
for i in df_1.columns[36:-4:3]:
    s=df_1[i].sum()

    l_1.append(s)
l_1


# In[ ]:


l_2=[]
for i in df_2.columns[36:-4:3]:
    s=df_2[i].sum()

    l_2.append(s)
l_2


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.bar(sharks_names,l_1,color="grey",edgecolor="red")
plt.title("Total Equity each sharks gets in season 1 ",fontweight="black",pad=5,size=15,color="Red")

plt.subplot(1,2,2)
plt.bar(sharks_names,l_2,color="pink",edgecolor="red")
plt.title("Total Equity each sharks gets in season 2 ",fontweight="black",pad=5,size=15,color="Red")
plt.show()


# ### though Aamn invested the highest amount in both season but peyush and Namita gets highest equity as compare to Aman

# ## Find the most attracted ideas accepted by sharks

# In[ ]:


df["Number of sharks in deal"].unique()


# In[ ]:


df[df["Number of sharks in deal"]==5][["Startup Name","Business Description"]]


# ### Find how many % entrepreneurs got more,less and same amount in successful deals

# In[ ]:


print(len(df_ac[df_ac["Total Deal Amount"]>df_ac["Original Ask Amount"]])/len(df_ac)*100," %entrepreneurs got more amount than they asked\n")
print(len(df_ac[df_ac["Total Deal Amount"]<df_ac["Original Ask Amount"]])/len(df_ac)*100,"% entrepreneurs got less amount than they asked\n")
print(len(df_ac[df_ac["Total Deal Amount"]==df_ac["Original Ask Amount"]])/len(df_ac)*100," %entrepreneurs got same amount as  they asked\n")


# #### Insight:
# - less bar gaining deals
# - mostly entreprenurs got same amount as they asked.

# ## Find how many % entrepreneurs got more,less and same amount in successful deals in season 1

# In[ ]:


df_ac_1=df_1[df_1["Accepted Offer"]==1]  ## for season1


# In[ ]:


print(len(df_ac_1[df_ac_1["Total Deal Amount"]>df_ac_1["Original Ask Amount"]])/len(df_ac_1)*100," %entrepreneurs got more amount than they asked\n")
print(len(df_ac_1[df_ac_1["Total Deal Amount"]<df_ac_1["Original Ask Amount"]])/len(df_ac_1)*100,"% entrepreneurs got less amount than they asked\n")
print(len(df_ac_1[df_ac_1["Total Deal Amount"]==df_ac_1["Original Ask Amount"]])/len(df_ac_1)*100," %entrepreneurs got same amount as  they asked\n")


# ## Find how many % entrepreneurs got more,less and same amount in successful deals in season 2

# In[ ]:


df_ac_2=df_2[df_2["Accepted Offer"]==1]


# In[ ]:


print(len(df_ac_2[df_ac_2["Total Deal Amount"]>df_ac_2["Original Ask Amount"]])/len(df_ac_2)*100," %entrepreneurs got more amount than they asked\n")
print(len(df_ac_2[df_ac_2["Total Deal Amount"]<df_ac_2["Original Ask Amount"]])/len(df_ac_2)*100,"% entrepreneurs got less amount than they asked\n")
print(len(df_ac_2[df_ac_2["Total Deal Amount"]==df_ac_2["Original Ask Amount"]])/len(df_ac_2)*100," %entrepreneurs got same amount as  they asked\n")

