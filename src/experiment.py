"""Independent-episode experiments for continuous-speed conformal braking."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, json, time, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel, delayed

VMAX=3.0
SCENARIOS=('gaussian','heteroscedastic','heavy_tail','mixture')
ROOT=Path(__file__).resolve().parents[1]

def population(n, rng, scenario):
    x=rng.uniform(size=(n,5))
    mean=np.column_stack((.08+.20*x[:,0]+.10*x[:,1]**2,
                          .18+.20*x[:,2]+.12*x[:,3]**2+.05*np.sin(np.pi*x[:,4])))
    z=rng.normal(size=(n,2))
    rho=.5 if scenario=='gaussian' else (-.65 if scenario=='heteroscedastic' else .2)
    z[:,1]=rho*z[:,0]+np.sqrt(1-rho*rho)*z[:,1]
    scale=np.column_stack((.025+.08*x[:,0]**2,.03+.12*x[:,2]**2))
    if scenario=='gaussian': scale[:]=[.04,.06]
    if scenario=='heavy_tail': z/=np.sqrt(rng.chisquare(5,size=(n,1))/3.)
    y=mean+scale*z
    if scenario=='mixture':
        mode=rng.choice(3,size=n,p=[.76,.12,.12])
        y[:,0]+=(mode==1)*(.12+.16*x[:,0])
        y[:,1]+=(mode==2)*(.15+.18*x[:,2])
    return x,np.maximum(y,[.015,.04])

def fit_predict(x,y,xs,ys,xc,xt,seed,constant=False):
    t0=time.perf_counter(); pc=[];pt=[];sc=[];st=[]
    for j in range(2):
        model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,
            min_samples_leaf=35,learning_rate=.07,l2_regularization=1.,early_stopping=False,random_state=seed+j)
        model.fit(x,y[:,j])
        floor=[.01,.015][j]
        scale_target=np.log(np.maximum(np.abs(ys[:,j]-model.predict(xs)),floor))
        sm=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=9,
            min_samples_leaf=45,learning_rate=.06,l2_regularization=1.,early_stopping=False,random_state=seed+10+j)
        sm.fit(xs,scale_target)
        pc.append(np.maximum(model.predict(xc),floor)); pt.append(np.maximum(model.predict(xt),floor))
        sc.append(np.maximum(np.exp(sm.predict(xc)),floor)); st.append(np.maximum(np.exp(sm.predict(xt)),floor))
    return tuple(np.column_stack(z) for z in (pc,pt,sc,st))+(time.perf_counter()-t0,)

def quantile(scores,alpha,pac=False,delta=.05):
    n=len(scores)
    k=int(binom.ppf(1-delta,n,1-alpha))+1 if pac else int(np.ceil((n+1)*(1-alpha)-1e-12))
    return np.inf if k>n else float(max(0.,np.partition(scores,k-1)[k-1]))

def endpoint_score(y,p,s):
    e=y-p
    return np.maximum.reduce((np.zeros(len(y)),e[:,0]/s[:,0],(e[:,0]+VMAX*e[:,1])/(s[:,0]+VMAX*s[:,1])))

def joint_grid_score(y,p,s,points=201):
    """One jointly calibrated supremum, including the zero-speed limit."""
    v=np.linspace(0,VMAX,points)
    e=y-p
    return np.maximum(0,np.max((e[:,0,None]+v*e[:,1,None])/(s[:,0,None]+v*s[:,1,None]),axis=1))

def score_at(y,p,s,v):
    e=y-p
    return (e[:,0]+v*e[:,1])/(s[:,0]+v*s[:,1])

def stop_distance(y,v,cubic=0.):
    return y[:,0]*v+y[:,1]*v*v+cubic*v**3

def safe_speed(coef,d):
    a,b=coef[:,0],coef[:,1]
    # Rationalized quadratic root avoids cancellation.
    with np.errstate(invalid='ignore',over='ignore',divide='ignore'):
        v=2*d/(a+np.sqrt(a*a+4*b*d))
    return np.clip(np.nan_to_num(v,nan=0,posinf=VMAX),0,VMAX)

def oracle_speed(y,d,cubic=0.):
    if cubic==0: return safe_speed(y,d)
    lo=np.zeros(len(y));hi=np.full(len(y),VMAX)
    for _ in range(45):
        mid=(lo+hi)/2;ok=stop_distance(y,mid,cubic)<=d
        lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
    return lo

def make_bounds(y,p,s,alpha,vlog,dcal):
    e=endpoint_score(y,p,s)
    vstar=safe_speed(y,dcal)
    policy=np.where(stop_distance(y,np.full(len(y),VMAX))<=dcal,0.,score_at(y,p,s,vstar))
    out={'nominal':('coef',np.array([0.,0.])),
         'pooled':('coef',np.repeat(quantile(score_at(y,p,s,vlog),alpha),2)),
         'endpoint':('coef',np.repeat(quantile(e,alpha),2)),
         'joint_grid':('coef',np.repeat(quantile(joint_grid_score(y,p,s),alpha),2)),
         'policy':('coef',np.repeat(quantile(policy,alpha),2)),
         'coefficient':('coef',np.array([quantile((y[:,j]-p[:,j])/s[:,j],alpha/2) for j in range(2)])),
         'tolerance':('coef',np.repeat(quantile(e,alpha,True),2))}
    grid=np.linspace(0,VMAX,21)
    q=np.array([0.]+[quantile(score_at(y,p,s,v),alpha/(len(grid)-1)) for v in grid[1:]])
    out['grid_bonferroni']=('grid',q)
    # Constant scales use training-independent constants with correct units.
    out['constant_scale']=('constant',np.repeat(quantile(endpoint_score(y,p,np.full_like(s,.05)),alpha),2))
    return out

def evaluate(y,p,s,d,bound,cubic=0.,correction=0.):
    typ,q=bound
    if typ in ('coef','constant'):
        ss=s if typ=='coef' else np.full_like(s,.05)
        coef=p+ss*q
        if correction:
            v=oracle_speed(coef,d,correction)
        else: v=safe_speed(coef,d)
        # Exact curve comparison for quadratic law. With a cubic truth residual/v
        # is convex, so its maximum over [0,V] is also at an endpoint.
        dif=y-coef
        full=((dif[:,0]>1e-12)|(dif[:,0]+VMAX*dif[:,1]+(cubic-correction)*VMAX**2>1e-12)).mean()
        margin=np.mean((coef[:,0]-p[:,0])*1.5+(coef[:,1]-p[:,1])*1.5**2+correction*1.5**3)
    else:
        grid=np.linspace(0,VMAX,21)
        with np.errstate(invalid='ignore'):
            b=p[:,0,None]*grid+p[:,1,None]*grid**2+(s[:,0,None]*grid+s[:,1,None]*grid**2)*q
        b[:,0]=0.
        # A cumulative maximum retains all valid grid bounds and monotonicity.
        b=np.maximum.accumulate(b,axis=1)
        idx=np.sum(b<=d[:,None],axis=1)-1
        idx=np.clip(idx,0,len(grid)-2);rows=np.arange(len(y))
        width=b[rows,idx+1]-b[rows,idx]
        with np.errstate(invalid='ignore',divide='ignore'):
            frac=np.clip((d-b[rows,idx])/width,0,1)
        frac=np.nan_to_num(frac,nan=0)
        v=grid[idx]+(grid[idx+1]-grid[idx])*frac
        v=np.where(b[:,-1]<=d,VMAX,v)
        # Convex true quadratic/cubic lies below each secant if knots covered.
        truth=y[:,0,None]*grid+y[:,1,None]*grid**2+cubic*grid**3
        full=np.any(truth>b+1e-12,axis=1).mean()
        margin=np.mean(b[:,10]-(p[:,0]*1.5+p[:,1]*1.5**2))
    oracle=oracle_speed(y,d,cubic)
    risk=stop_distance(y,v,cubic)>d+1e-10
    return {'curve_failure':float(full),'stop_failure':float(risk.mean()),'mean_speed':float(v.mean()),
            'speed_ratio':float(np.mean(v/oracle)),'conservative_loss':float(np.mean(np.maximum(oracle-v,0)/oracle)),
            'mean_margin':float(margin),'abstain':float(np.mean(v<1e-10))},v,risk

def one_run(scenario,seed,ntest=10000,full=True):
    start=time.perf_counter()
    streams=[np.random.default_rng(np.random.SeedSequence([seed,SCENARIOS.index(scenario),j])) for j in range(6)]
    x,y=population(4000,streams[0],scenario);xs,ys=population(2000,streams[1],scenario)
    xc,yc=population(2047,streams[2],scenario);xt,yt=population(ntest,streams[3],scenario)
    pc,pt,sc,st,fit_s=fit_predict(x,y,xs,ys,xc,xt,seed)
    vlog=streams[4].uniform(0,VMAX,len(yc))
    dcal=np.exp(streams[4].uniform(np.log(.08),np.log(4.5),len(yc)))
    clearances={name:np.exp(streams[5].uniform(np.log(lo),np.log(hi),ntest)) for name,lo,hi in [('standard',.08,4.5),('low',.03,.4),('high',.8,4.5)]}
    rows=[]; traces={}
    configs=[(511,.05)]
    if full: configs+= [(n,.05) for n in (63,127,255,1023,2047)]+[(511,a) for a in (.01,.02,.1)]
    for n,alpha in configs:
        bounds=make_bounds(yc[:n],pc[:n],sc[:n],alpha,vlog[:n],dcal[:n])
        settings=list(clearances.items()) if (n,alpha)==(511,.05) else [('standard',clearances['standard'])]
        for condition,d in settings:
            for method,b in bounds.items():
                metrics,v,r=evaluate(yt,pt,st,d,b)
                rows.append(dict(scenario=scenario,seed=seed,n=n,alpha=alpha,condition=condition,method=method,**metrics))
                if seed==1000 and n==511 and alpha==.05 and condition=='standard':
                    traces[method+'_speed']=v;traces[method+'_failure']=r
        if (n,alpha)==(511,.05):
            shifted=yt.copy();shifted[:,1]*=1.25
            for method,b in bounds.items():
                for condition,yy,cubic,correction in [('brake_shift',shifted,0,0),('cubic',yt,.04,0),('cubic_corrected',yt,.04,.04)]:
                    if condition=='cubic_corrected' and method!='endpoint':continue
                    metrics,_,_=evaluate(yy,pt,st,clearances['standard'],b,cubic,correction)
                    rows.append(dict(scenario=scenario,seed=seed,n=n,alpha=alpha,condition=condition,method=method,**metrics))
            if seed==1000:
                traces.update(x=xt,parameters=yt,prediction=pt,scale=st,clearance=clearances['standard'],
                              cal_y=yc,cal_prediction=pc,cal_scale=sc,vlog=vlog,dcal=dcal)
    if seed==1000:
        np.savez_compressed(ROOT/'results'/f'example_{scenario}.npz',**traces)
    meta={'scenario':scenario,'seed':seed,'fit_seconds':fit_s,'total_seconds':time.perf_counter()-start}
    return rows,meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seeds',type=int,default=40)
    ap.add_argument('--pilot',action='store_true');ap.add_argument('--test-size',type=int,default=10000)
    args=ap.parse_args();(ROOT/'results').mkdir(exist_ok=True)
    t0=time.perf_counter();seeds=[7] if args.pilot else range(1000,1000+args.seeds)
    result=Parallel(n_jobs=args.workers,verbose=10)(delayed(one_run)(s,k,args.test_size,not args.pilot) for s in SCENARIOS for k in seeds)
    rows=[r for run,_ in result for r in run];meta=[m for _,m in result]
    name='pilot' if args.pilot else 'metrics'
    pd.DataFrame(rows).to_csv(ROOT/'results'/f'{name}.csv',index=False)
    env={'python':sys.version,'platform':platform.platform(),'processors_requested':args.workers,'wall_seconds':time.perf_counter()-t0,'settings':vars(args),'runs':meta}
    (ROOT/'results'/f'{name}_runtime.json').write_text(json.dumps(env,indent=2))
    print(pd.DataFrame(rows).query("n==511 and alpha==.05 and condition=='standard'").groupby('method')[['curve_failure','stop_failure','mean_speed']].mean().to_string())
if __name__=='__main__':main()
