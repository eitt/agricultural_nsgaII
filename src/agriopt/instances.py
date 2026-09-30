from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from .model import ProblemData
from .parameters import PRODUCT_SETS,MATURITY_WEEKS,BOTANICAL,SALES_GROUP_NAME,SALES_GROUP_ID
from .productivity import estimate_yield,weekly_sow_calendar,weekly_yield_tensor

CATALOG=pd.DataFrame([
 ['I01_micro',1,2,0.6,5,52,'low',101],
 ['I02_small',1,4,1.8,8,52,'medium',102],
 ['I03_diverse',1,8,4.5,12,104,'high',103],
 ['I04_two_farms',2,8,6.0,8,52,'medium',104],
 ['I05_coop_small',5,20,15.0,12,104,'medium',105],
 ['I06_coop_medium',10,40,30.0,12,104,'high',106],
 ['I07_network',20,80,60.0,21,104,'high',107],
 ['I08_network_large',30,150,100.0,21,104,'high',108],
],columns=['instance','farms','lots','total_area_ha','products','horizon_weeks','heterogeneity','seed'])

def _lot_allocation(farms,lots,total_area_ha,heterogeneity,rng):
    # Allocate total area across farms then lots. Every individual farm remains <=5 ha.
    avg=total_area_ha/farms
    sigma={'low':0.15,'medium':0.4,'high':0.75}[heterogeneity]
    raw=rng.lognormal(np.log(max(avg,0.2)),sigma,size=farms)
    raw=np.minimum(raw,5.0)
    farm_area=raw/raw.sum()*total_area_ha
    # if rescaling exceeds 5, iteratively cap and redistribute
    for _ in range(10):
        over=farm_area>5
        if not over.any():break
        excess=(farm_area[over]-5).sum();farm_area[over]=5
        under=~over
        if under.any():farm_area[under]+=excess*farm_area[under]/farm_area[under].sum()
    counts=np.ones(farms,dtype=int)
    for _ in range(lots-farms): counts[np.argmin(counts/farm_area)]+=1
    areas=[];ids=[]
    for f,(a,n) in enumerate(zip(farm_area,counts)):
        alpha=np.full(n,4.0 if heterogeneity=='low' else (2.0 if heterogeneity=='medium' else 0.8))
        shares=rng.dirichlet(alpha)
        areas.extend((a*10000*shares).tolist());ids.extend([f]*n)
    return np.asarray(areas),np.asarray(ids)

def make_instance(row, productivity, price_forecast:pd.DataFrame, price_history:pd.DataFrame, demand:pd.DataFrame|None=None):
    if not isinstance(row,pd.Series): row=pd.Series(row)
    rng=np.random.default_rng(int(row.seed))
    products=PRODUCT_SETS[int(row.products)]
    T=int(row.horizon_weeks)
    areas,farm_ids=_lot_allocation(int(row.farms),int(row.lots),float(row.total_area_ha),str(row.heterogeneity),rng)
    mun_counts=productivity['Municipio'].value_counts()
    muns=mun_counts.index.to_numpy(); probs=(mun_counts/mun_counts.sum()).to_numpy()
    farm_mun=rng.choice(muns,size=int(row.farms),replace=True,p=probs)
    municipalities=[str(farm_mun[f]) for f in farm_ids]
    Y=np.zeros((len(areas),len(products)))
    support=np.zeros_like(Y,dtype=int)
    for l,m in enumerate(municipalities):
        for k,p in enumerate(products):
            Y[l,k],support[l,k],_=estimate_yield(productivity,m,p)
    sow=weekly_sow_calendar(productivity,products,T,expand_sparse=True)
    Yweek=weekly_yield_tensor(productivity,municipalities,products,T)
    P=price_forecast[products].iloc[:T].to_numpy(float).T
    hist=price_history[products].to_numpy(float)
    ret=np.diff(np.log(hist),axis=0)
    cov=LedoitWolf().fit(ret).covariance_
    maturity=np.array([MATURITY_WEEKS[p] for p in products],dtype=int)
    botan=np.array([BOTANICAL[p] for p in products],dtype=object)
    groups=np.array([SALES_GROUP_ID[SALES_GROUP_NAME[p]] for p in products],dtype=int)
    eligible=support>=3
    # If a lot would have no eligible crop because the updated census support is sparse,
    # keep the highest-support crop eligible rather than making the lot structurally unusable.
    for l in range(len(areas)):
        if not eligible[l].any(): eligible[l,int(np.argmax(support[l]))]=True
    demand_arr=None
    if demand is not None:
        # Demand/capacity columns follow the model group labels VH, F, TRP, OG.
        labels=['VH','F','TRP','OG']
        d=demand.copy()
        missing=[g for g in labels if g not in d.columns]
        if missing:
            raise ValueError(f'demand is missing required group columns: {missing}')
        if len(d)<T:
            raise ValueError(f'demand has {len(d)} periods but instance {row.instance} requires {T}')
        demand_arr=d[labels].iloc[:T].to_numpy(float).T
    data=ProblemData(products,areas,farm_ids,municipalities,Y,P,cov,maturity,botan,groups,sow,eligible=eligible,yield_by_sow_week=Yweek,demand=demand_arr)
    metadata={'instance':row.instance,'yield_support_min':int(support.min()),'yield_support_median':float(np.median(support)),'yield_support_zero_share':float((support==0).mean()),'eligible_share':float(eligible.mean())}
    return data,metadata
