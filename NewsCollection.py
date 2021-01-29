# -*- coding: utf-8 -*-
'''
Author: Wei
Date: 11/2020
'''
from bs4 import BeautifulSoup
import requests
from pymongo import MongoClient
import random 
import time
import pytz
from datetime import datetime
import csv

with open('ticker_name.csv',newline='') as f:
    reader=csv.reader(f)
    tickers=[row for row in reader]
tickers=dict(tickers)
for t in tickers:
    tickers[t]=tickers[t].split()
    
mdict=dict(zip(['January','February','March','April','May','June','July','August','September','Octorber','November','December'],
                  range(1,13)))
est = pytz.timezone('US/Eastern')
utc = pytz.utc
fmt = '%Y-%m-%d %H:%M:%S'

ii=0

lurl="https://www.reuters.com/news/archive/businessnews?view=page&page=%d&pageSize=10"
urls=[]
for i in range(0,500):
    response=requests.get(lurl%i)
    contentTmp=response.content.decode('utf-8')
    soup=BeautifulSoup(contentTmp,'html.parser')
    links=soup.find_all('div',class_='story-content')
    urls=urls+["https://www.reuters.com"+lk.a.get('href') for lk in links]
    if i%10==0:
        print(i)
        
newslist=[]
for u in urls:
    if ii%100==0:
        time.sleep(random.randint(0,20)*0.1)
        print(ii)
    response=requests.get(u)
    contentTmp = response.content.decode('utf-8')
    soup=BeautifulSoup(contentTmp,'html.parser')
    if soup==None:
        continue
    
    date=soup.find('head')
    if date==None:
        continue
    date=date.find('meta',{'property':"og:article:published_time"})
    if date==None:
        continue
    date=date.attrs['content'].replace('Z','').split('T')
    year,month,day=[int(x) for x in date[0].split('-')]
    h,m,s=[int(x) for x in date[1].split(':')]
    zdate=datetime(year,month,day,h,m,s, tzinfo=utc)
    edate=zdate.astimezone(est).strftime(fmt)
   
    div=soup.find('div',class_="ArticleBodyWrapper")
    texts=div.find_all('p')
    links=div.find_all('a')
    news=''
    firms=[]
    for l in links:
        if "." in l.text:
            name=l.text.split(".")[0]
            if name!='' and name in tickers.keys():
                firms.append(name)
                
    titleh=soup.find('h1')
    title=titleh.text
    if len(firms)>1:
        tfirms=[]
        for f in firms:
            fullname=tickers[f]
            for fw in fullname:
                if fw in title:
                    tfirms.append(f)
                    break
        firms=tfirms
    if len(firms)==1:
        for tt in texts[2:-2]:
            news=news+tt.text
        news=news.replace('\xa0','')
        newsdict={'no':ii,'tickers':firms[0],'date':edate,'article':news}
        newslist.append(newsdict)
        ii=ii+1
        

client=MongoClient("mongodb+srv://Wei:Mangodb01@cluster0.lhndh.mongodb.net/Reuters")
db=client.Reuters
db.rnews.insert_many(newslist)
print(db.rnews.count_documents({}))
