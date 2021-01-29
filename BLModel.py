# -*- coding: utf-8 -*-
"""
Author: Wei
Date: 11/2020
"""


from pymongo import MongoClient
from yahoofinancials import YahooFinancials
import re
from nltk import FreqDist
from nltk.corpus import stopwords
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import itertools
import pandas as pd
import json
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer("english")


#convert news to bag of words and screen non-neutral words
def getbow(art):
    art=np.array(re.sub(r'[^a-zA-Z]',' ',art.lower()).split())
    artindex=list(map(lambda w : w not in stopwords.words("english"),art))
    art=list(map(lambda w:stemmer.stem(w),art[artindex]))
    bow=FreqDist(art)
    return bow

def getSenbow(art,Swds):
    art=np.array(re.sub(r'[^a-zA-Z]',' ',art.lower()).split())
    art=np.array(list(map(lambda w :stemmer.stem(w),art)))
    artindex=list(map(lambda w : w in Swds,art))
    art=art[artindex]
    bow=FreqDist(art)
    return bow

def getM(articles):
    #generate bag of words
    mlist=list(zip(*(map(lambda a:(getbow(a),list(getbow(a).keys())),articles))))
    freqs,D=list(mlist[0]),list(itertools.chain(*mlist[1]))
    D=list(set(D))
    
    #generate matrix
    n=len(articles)
    m=len(D)
    Matrix=np.zeros((n,m))
    for i in range(n):
        iwds=list(freqs[i].keys())
        for w in iwds:
            j=D.index(w)
            Matrix[i,j]=freqs[i][w]
    return D,Matrix

def screen(D,Matrix,ret,alphap,alpham,k):
    dmatrix=getdM(Matrix)
    dens=dmatrix.sum(0)
    
    rflag=ret>0
    nums=np.dot(rflag,dmatrix)
    
    f=nums/dens
    kindex=np.where(dens>k)[0]
    findex=np.where((f>alphap) | (f<alpham))[0]
    finalindex=np.intersect1d(kindex,findex)
    
    S=Matrix[:,finalindex]
    Swds=np.array(D)[finalindex]
    
    return S,Swds

#Topic Modeling for "Positive Topic" and "Negative Topic"
def SentiDist(ret,S):
    index=np.where(S.sum(1)!=0)
    S,ret=S[index],ret[index]
    S=S/S.sum(1).reshape(-1,1)
    p=(np.array(list(map(sorted(ret).index,ret)))+0.99999)/len(ret)
    W=np.vstack([p,1-p])
    O=np.dot(S.T,np.dot(W.T,np.matrix(np.dot(W,W.T)).I))
    O=np.clip(O,0,None)
    O=O/O.sum(0)
    return O

    
###Hyperparameter Combinations
#lambda=1,5,10
#k=5%,10%,15% of testing article counts
#alphap,alpham: 25,50,100

#Generate hyperparameter combinations
def getdM(Matrix):
    dmatrix=Matrix
    dmatrix[Matrix.nonzero()]=1
    return dmatrix

def hyperparam(dmatrix,ret):
    alphams,alphaps=[],[]
    count=dmatrix.shape[0]
    ks=np.array([0.02,0.05,0.1])*count
    dens=dmatrix.sum(0)

    rflag=(ret>0)
    nums=np.dot(rflag,dmatrix)

    for k in ks:
        f=(nums/dens)[dens>k]
        alqs=np.array([25,50,100])/len(f)
        alphams.append(np.quantile(f,alqs,interpolation='nearest'))
        alphaps.append(np.quantile(f,1-alqs,interpolation='nearest'))

    ks=np.tile(ks,(3,1)).T.flatten()
    alphams=np.array(alphams).flatten()
    alphaps=np.array(alphaps).flatten()
    hyperparams=np.tile(np.vstack([ks,alphaps,alphams]),3)
    hyperparams=np.vstack([hyperparams,np.tile([1,5,10],(9,1)).T.flatten()])
    return hyperparams
            
#Sentiment engine training
def train(article,ret,hyperparams,fD,fMatrix):
    kf3 = KFold(n_splits=3, shuffle=False)
    loss=[]
    ks1,alphaps1,alphams1,lams1=tuple(hyperparams)
    for x in range(len(ks1)):
        xloss=[]
        alphap,alpham,k,lam=alphaps1[x],alphams1[x],ks1[x],lams1[x]
        for tindex,vindex in kf3.split(ret):
            vart=article[vindex]
            tret,vret=ret[tindex],ret[vindex]

            Matrix=fMatrix[tindex]
            Shat,Swds=screen(fD,Matrix,tret,alphap,alpham,k)
            O=SentiDist(tret,Shat)

            phats=np.array([predict(lam,O,a,Swds) for a in vart])
            ps=(np.array(list(map(sorted(vret).index,vret)))+1)/len(vret)
            ls=np.array(list(map(abs,phats-ps))).sum()
            xloss.append(ls)
        loss.append(np.array(xloss).mean())
    bestindex=loss.index(min(loss))
    balphap,balpham,bk,blam=alphaps1[bestindex],alphams1[bestindex],ks1[bestindex],lams1[bestindex]
    
    
    fShat,fSwds=screen(fD,fMatrix,ret,balphap,balpham,bk)
    fO=SentiDist(ret,fShat)
    return blam,fO,fSwds

#Use trained sentiment engine to predict sentiment score for out-of-sample news articles
def predict(lam,O,art,Swds):
    O=np.array(O)
    ds=np.zeros(len(Swds))
    bow=getSenbow(art,Swds)
    if len(bow.keys())>0:
        for k in bow.keys():
            kindex=np.where(Swds==k)[0][0]
            ds[kindex]=bow[k]
        objfunc=lambda p:-(np.dot(ds,np.log(p*O[:,0]+(1-p)*O[:,1]))/sum(bow.values())+lam*np.log(p*(1-p)))
        #res=minimize(objfunc,0.5,method='nelder-mead')
        res=minimize(objfunc,0.5,bounds=[(0.00001,0.99999)],method='TNC')
        if res.success==True:
            phat=res.x[0]
        else:
            phat=0.5
        return phat
    else:
        return 0.5


#generate a view for Black-Litterman Model based on sentiment scores
def BLview(lam,arts,O,Swds,firms,stkname):
    numstk=len(stkname)
    phats=np.array([predict(lam,O,a,Swds) for a in arts])
    pdict={}
    for i in range(len(firms)):
        if not firms[i] in stkname:
            continue
        if firms[i] not in pdict.keys():
            pdict[firms[i]]=[phats[i]]
        else:
            pdict[firms[i]]=pdict[firms[i]]+phats[i]
    pdict=dict(zip(pdict.keys(),list(map(lambda k:np.mean(pdict[k]),pdict.keys()))))
    sortedfirms=np.array(sorted(pdict,key=pdict.get,reverse=True))
    sortedps=np.array(sorted(pdict.values()))
    
    long=sortedfirms[sortedps>0.5]
    short=sortedfirms[sortedps<0.5]
    P=pd.DataFrame(np.zeros(numstk).reshape(1,-1),columns=stkname)
    
    if len(long)>0 and len(short)>0:
        longp,shortp=sortedps[sortedps>0.5],sortedps[sortedps<0.5]
        
    elif len(long)>0:
        short=list(set(list(stkname))-set(list(long)))
        longp,shortp=sortedps[sortedps>0.5],np.ones(len(short))*0.5
    elif len(short)>0:
        long=list(set(list(stkname))-set(list(short)))
        longp,shortp=np.ones(len(long))*0.5,sortedps[sortedps<0.5]
    else:
        return np.matrix(np.zeros(numstk)),'NoView'
        
    lw,sw=longp/np.sum(longp),-shortp/np.sum(shortp)
    P[long],P[short]=lw,sw
    P=np.matrix(P.values.flatten())
    if (np.mean(longp)-np.mean(shortp))>0.5: 
        flag=2
    elif (np.mean(longp)-np.mean(shortp))>0.2:
        flag=0.2
    elif (np.mean(longp)-np.mean(shortp))>0.1:
        flag=0.1
    else:
        flag=0.05
    return P,flag


#prepare training set and testing set
def getRetOpen(prices,firmset):
    pricedf=pd.DataFrame(columns=pd.MultiIndex.from_product([['blank'],['open','ret']]))
    pricedf.index.name="formatted_date"
    for firm in firmset:
        if 'prices' in prices[firm].keys():
            firmdf=pd.DataFrame(prices[firm]['prices'])[['formatted_date','open','adjclose']]
            firmdf=firmdf.set_index('formatted_date')
            p2close=firmdf['adjclose'].shift(2)
            n1close=firmdf['adjclose'].shift(-1)
            firmdf['ret']=np.log(n1close/p2close)
            firmdf=firmdf.drop('adjclose',axis=1)
            firmdf.columns=pd.MultiIndex.from_product([[firm],['open','ret']])
            pricedf=pricedf.merge(firmdf,how='outer',left_index=True,right_index=True)
    pricedf=pricedf.drop([('blank','open'),('blank','ret')],axis=1)
    idx=pd.IndexSlice
    retdf=pricedf.loc[:,idx[:,'ret']]
    retdf.columns=retdf.columns.droplevel(1)

    opendf=pricedf.loc[:,idx[:,'open']]
    opendf.columns=opendf.columns.droplevel(1)
    return retdf,opendf    

def getTRset(clct,sdate,retdf):
    dataset=list(db.rnews.find({'date':{"$gt":sdate}})) #use sdate to define
    artset=np.array(list(map(lambda x:x["article"],dataset)))
    firmset=np.array(list(map(lambda x:x["tickers"],dataset)))
    datetime=pd.to_datetime(list(map(lambda x:x["date"],dataset)))
    udates=sorted(list(map(lambda x:x+" 16:00:00",list(set(datetime.strftime("%Y-%m-%d"))))))
    
    TRset={}
    for i in range(1,len(udates)):
        isdate=udates[i-1]
        iedate=udates[i]
        
        dindex=(datetime>isdate)*(datetime<=iedate)
        
        if dindex.any():
            TRset[udates[i][:10]]={}
            TRset[udates[i][:10]]['art']=artset[dindex]
            datesset=list(datetime.strftime("%Y-%m-%d")[dindex])

            fnames=firmset[dindex]
            retdates=retdf.index
            TRset[udates[i][:10]]['ret']=np.array([retdf.loc[retdates[np.searchsorted(retdates,d)],fn] for d,fn in zip(datesset,fnames)])
        
    return TRset
    
def getTTset(clct,sdate):
    dataset=list(db.rnews.find({'date':{"$gt":sdate}})) #use sdate to define
    artset=np.array(list(map(lambda x:x["article"],dataset)))
    firmset=np.array(list(map(lambda x:x["tickers"],dataset)))
    datetime=pd.to_datetime(list(map(lambda x:x["date"],dataset)))
    udates=sorted(list(set(datetime.strftime("%Y-%m-%d"))))
    
    TTset={}
    for i in range(1,len(udates)):
        isdate=udates[i-1]+" 09:30:00"
        iedate=udates[i]+" 09:00:00"
        
        dindex=(datetime>isdate)*(datetime<=iedate)
        if dindex.any():
            TTset[udates[i][:10]]={}
            TTset[udates[i][:10]]['art']=artset[dindex]
            TTset[udates[i][:10]]['firms']=firmset[dindex]
        
    return TTset

#BL model parameters calculation
def getvar(hisprice):
    hisret=(np.log(hisprice)).diff().dropna().values
    var=np.cov(hisret.T)
    return var

def EQret(mktcap,var,lam):
    w=mktcap/np.sum(mktcap)
    PI=lam*np.dot(var,w)
    return PI

def getOmega(tau,P,var):
    Omega=np.diag(np.diag(tau*np.dot(P,np.dot(var,P.T))))
    return Omega

def getQ(P,PI,Omega,flag):
    return np.dot(P,PI)+flag*np.sqrt(Omega)

def BLret(P,Q,tau,Omega,PI,var):
    beta=np.dot(((np.matrix((tau*var)).I+np.dot(P.T,np.dot(np.matrix(Omega).I,P))).I),(np.dot(np.matrix(tau*var).I,PI).flatten()+np.dot(P.T,np.dot(np.matrix(Omega).I,Q)).flatten()).T)
    return beta

def BLvar(var,tau,P,Omega):
    M=(np.matrix(tau*var).I+np.dot(P.T,np.dot(np.matrix(Omega).I,P))).I
    varBL=var+M
    return varBL

def getw(ret,var,lam):
    w=1/lam*np.dot(np.matrix(var).I,ret)
    w=np.array(w).flatten()
    w=w/np.sum(w)
    return w

#Connect to news database
client=MongoClient("mongodb+srv://Wei:Mangodb01@cluster0.lhndh.mongodb.net/Reuters")
db=client.Reuters

##get and store stock price data
dataset=list(db.rnews.find({'date':{"$gt":"2017-12-31"}}).sort('date'))
firmset=list(set(list(map(lambda x:x["tickers"],dataset))))
yahoo_financials = YahooFinancials(firmset)
prices=yahoo_financials.get_historical_price_data('2018-01-01','2020-11-22','daily')
with open('prices112820.json','w') as fj:
    json.dump(prices,fj)
    
##Start Training
retdf,openprice=getRetOpen(prices,firmset)
dates=openprice.index
TRdataset=getTRset(db.rnews,"2017-12-31",retdf)
TTdataset=getTTset(db.rnews,"2017-12-31")
portret=[]
longret=[]
shortret=[]
traindates=np.array(list(TRdataset.keys()))
testdates=np.array(list(TTdataset.keys()))

trainx=list(db.rnews.find({'date':{"$lt":"2019-12-31","$gt":"2017-12-31"}},{'tickers':1,"_id":0}))
trainx=[xx['tickers'] for xx in trainx]
unique, counts = np.unique(trainx, return_counts=True)
nunique,ncounts=[],[]
for i,j in zip(unique,counts):
    if j>=10:
        nunique.append(i)
        ncounts.append(j)
stkname=nunique

traint=np.searchsorted(traindates,'2019-12-31','right')-1
trainkeys=traindates[:traint]
trainart=np.array(list(itertools.chain(*list(map(lambda x:x['art'],list(map(TRdataset.get,trainkeys)))))))
trainret=np.array(list(itertools.chain(*list(map(lambda x:x['ret'],list(map(TRdataset.get,trainkeys)))))))
trainret=np.nan_to_num(trainret)

D,Matrix=getM(trainart)
dmatrix=getdM(Matrix)
hyperparams=hyperparam(dmatrix,trainret)
blam,fO,fSwds=train(trainart,trainret,hyperparams,D,Matrix)

##Backtesting
mktcaps=pd.read_excel('MktCap.xlsx')
mktcaps=mktcaps.set_index('Date')
mktcaps=mktcaps.ffill()
mktcaps=mktcaps.loc[:,stkname]

RAlam=3 #risk-averse lambda
var=getvar(openprice.loc['2018-01-02':'2019-12-31',stkname])
tau=1/len(openprice.loc['2018-01-02':'2019-12-31'])
K=[10000]
Km=[10000]
BLws,Mws=[],[]

for t in range(502,len(dates)-1):
        
    testartdate=testdates[np.searchsorted(testdates,dates[t],side='right')-1]
    testart=TTdataset[testartdate]['art']
    testfirms=TTdataset[testartdate]['firms']
    
    if t>502:
        Kt=np.sum(sharet*openprice.loc[dates[t+1],stkname])
        MKt=np.sum(Msharet*openprice.loc[dates[t+1],stkname])
        K.append(Kt)
        Km.append(MKt)
    
    P,flag=BLview(blam,testart,fO,fSwds,testfirms,stkname)
    
    mktcap=mktcaps.loc[dates[t]]
    PI=EQret(mktcap,var,RAlam)
    if not flag=='NoView':
        Omega=getOmega(tau,P,var)
        Q=getQ(P,PI,Omega,flag)
        blret=BLret(P,Q,tau,Omega,PI,var)
        blvar=BLvar(var,tau,P,Omega)
        BLw=getw(blret,blvar,RAlam)
    else:
        BLw=mktcap/np.sum(mktcap)
    Mw=mktcap/np.sum(mktcap)
    
    BLws.append(BLw)
    Mws.append(Mw)
    
    opent=openprice.loc[dates[t+1],stkname]
    sharet=K[-1]*BLw/opent
    Msharet=Km[-1]*Mw/opent
    
##Visualize cumulative return and compare to value-weighted benchmark portfolio
figure,(ax1,ax2)=plt.subplots(2,1,figsize=(6,8))
ax1.plot(np.array([K,Km]).T)
ax1.legend(['Black-Litterman','Market Equilibrium'])
ax1.set_title('Portfolio Value')

ax2.plot(np.cumsum(np.diff(np.log([K,Km]).T,axis=0),axis=0))
ax2.legend(['Black-Litterman','Market Equilibrium'])
ax2.set_title('Cumulative Return')
plt.tight_layout();



