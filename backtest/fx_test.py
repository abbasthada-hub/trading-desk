"""Forex and metals swing-rule tests. Usage: python -m backtest.fx_test backtest/data"""
import sys, numpy as np, pandas as pd

from desk.indicators import sma, rsi_wilder
from backtest.backtest import load_daily, perf
D=sys.argv[1]
pairs=["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD"]
C=pd.DataFrame({p:load_daily(f"{D}/{p}_daily.csv")["close"] for p in pairs})
C=C[C.index.dayofweek<5].dropna(how="all").ffill()
R=C.pct_change()
print("fx days",len(C),C.index[0].date(),C.index[-1].date(),"max abs daily ret per pair",R.abs().max().round(3).to_dict())
R=R.clip(-0.08,0.08)   # guard against bad ticks
COST=0.0002
def port(pos,lag=1,cost=COST):
    p=pos.shift(lag).fillna(0)
    r=(p*R).mean(axis=1)-(p.diff().abs().fillna(0)*cost).mean(axis=1)
    return r["2008":]
def donch(c,n_in,n_out):
    hi=c.rolling(n_in).max().shift(1); lo=c.rolling(n_in).min().shift(1)
    xhi=c.rolling(n_out).max().shift(1); xlo=c.rolling(n_out).min().shift(1)
    pos=np.zeros(len(c)); cur=0
    for i in range(len(c)):
        x=c.iloc[i]
        if cur==0:
            if x>hi.iloc[i]: cur=1
            elif x<lo.iloc[i]: cur=-1
        elif cur==1 and x<xlo.iloc[i]: cur=0
        elif cur==-1 and x>xhi.iloc[i]: cur=0
        pos[i]=cur
    return pd.Series(pos,index=c.index)
def rsi2(c):
    r=rsi_wilder(c,2); m=sma(c,5); pos=np.zeros(len(c)); cur=0
    for i in range(len(c)):
        if cur==0:
            if r.iloc[i]<10: cur=1
            elif r.iloc[i]>90: cur=-1
        elif cur==1 and c.iloc[i]>m.iloc[i]: cur=0
        elif cur==-1 and c.iloc[i]<m.iloc[i]: cur=0
        pos[i]=cur
    return pd.Series(pos,index=c.index)
S={}
for n in (50,100,200): S[f"trend SMA{n} long/short"]=np.sign(C-C.rolling(n).mean())
for n in (21,63,252): S[f"momentum {n}d sign"]=np.sign(C/C.shift(n)-1)
S["breakout 20/10"]=C.apply(lambda c:donch(c,20,10)); S["breakout 55/20"]=C.apply(lambda c:donch(c,55,20))
S["RSI2 reversal"]=C.apply(rsi2)
def line(name,r):
    a=perf(r,252); h1=perf(r[:"2016"],252); h2=perf(r["2017":],252)
    yr=r.groupby(r.index.year).apply(lambda x:(1+x).prod()-1)
    print(f"{name:26s} cagr={a['cagr']*100:+5.1f}% vol={a['vol']*100:4.1f}% sharpe={a['sharpe']:+.2f} maxdd={a['max_dd']*100:5.1f}% | 08-16 {h1['cagr']*100:+5.1f}% sh {h1['sharpe']:+.2f} | 17-26 {h2['cagr']*100:+5.1f}% sh {h2['sharpe']:+.2f} | pos yrs {(yr>0).sum()}/{len(yr)}")
print("--- FX, 7 majors, equal weight, 1x notional, 2bp/side, signal traded at same close")
for k,v in S.items(): line(k,port(v))
print("--- same, traded one day late")
for k in ("trend SMA100 long/short","breakout 20/10","breakout 55/20","momentum 63d sign","RSI2 reversal"): line(k,port(S[k],lag=2))
print("--- zero cost"); 
for k in ("breakout 20/10","RSI2 reversal","trend SMA100 long/short"): line(k,port(S[k],cost=0))
# metals
for m in ("GCUSD","SIUSD"):
    c=load_daily(f"{D}/{m}_daily.csv")["close"]; c=c[c.index.dayofweek<5]; r=c.pct_change().clip(-0.15,0.15)
    for nm,pos in (("long if >SMA200",(c>c.rolling(200).mean()).astype(float)),("long/short SMA100",np.sign(c-c.rolling(100).mean())),("breakout 55/20 L/S",donch(c,55,20)),("buy&hold",pd.Series(1.0,index=c.index))):
        p=pos.shift(1).fillna(0); rr=(p*r-p.diff().abs().fillna(0)*0.0005)["2008":]; line(f"{m} {nm}",rr)
