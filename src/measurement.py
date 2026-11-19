"""Calibration from paired stopping distances with bounded observation error."""
from experiment import (np,pd,ROOT,SCENARIOS,VMAX,population,fit_predict,
                        endpoint_score,joint_grid_score,quantile,evaluate)
from analyze import summarize
from joblib import Parallel,delayed
from scipy.stats import t
import argparse,json,time,platform,sys

def reconstruct(distances,u1=.75,u2=2.5):
    z=distances/np.array([u1,u2])
    beta=(z[:,1]-z[:,0])/(u2-u1)
    return np.column_stack((z[:,0]-u1*beta,beta))

def measured_labels(y,uniforms,eta,profile,u1=.75,u2=2.5):
    u=np.array([u1,u2])
    error=eta*((2*uniforms-1) if profile=='symmetric' else (uniforms-1))
    distances=y[:,0,None]*u+y[:,1,None]*u*u+error
    return reconstruct(distances,u1,u2),distances

def error_allowance(v,eta,u1=.75,u2=2.5):
    """Support function of the two-distance rectangular error set, after /v."""
    return abs((u2-v)/(u2-u1))*eta/u1+abs((v-u1)/(u2-u1))*eta/u2

def robust_score(y_measured,p,s,eta,u1=.75,u2=2.5):
    e=y_measured-p
    return np.maximum.reduce((np.zeros(len(e)),
        (e[:,0]+error_allowance(0,eta,u1,u2))/s[:,0],
        (e[:,0]+VMAX*e[:,1]+error_allowance(VMAX,eta,u1,u2))/(s[:,0]+VMAX*s[:,1])))

def one_measurement_run(scenario,seed,ntest=10000):
    tic=time.perf_counter()
    rng=[np.random.default_rng(np.random.SeedSequence([seed,SCENARIOS.index(scenario),j])) for j in range(9)]
    x,y=population(4000,rng[0],scenario);xs,ys=population(2000,rng[1],scenario)
    xc,yc=population(511,rng[2],scenario);xt,yt=population(ntest,rng[3],scenario)
    un=[rng[4+j].uniform(size=(len(yy),2)) for j,yy in enumerate((y,ys,yc))]
    d=np.exp(rng[7].uniform(np.log(.08),np.log(4.5),ntest))
    settings=[(0.,'symmetric',.75)]+[(eta,profile,.75) for eta in (.01,.025,.05) for profile in ('symmetric','under_read')]
    settings +=[(.025,'under_read',u1) for u1 in (.3,1.5)]
    rows=[];fits=[]
    for eta,profile,u1 in settings:
        rec=[measured_labels(yy,uu,eta,profile,u1) for yy,uu in zip((y,ys,yc),un)]
        # Only training labels are floored. Calibration keeps the exact affine inversion.
        pc,pt,sc,st,fit_s=fit_predict(x,np.maximum(rec[0][0],[.015,.04]),xs,
                                     np.maximum(rec[1][0],[.015,.04]),xc,xt,seed)
        fits.append(fit_s)
        scores={'oracle_label':endpoint_score(yc,pc,sc),
                'naive_label':endpoint_score(rec[2][0],pc,sc),
                'robust_label':robust_score(rec[2][0],pc,sc,eta,u1)}
        if not np.all(scores['robust_label']>=scores['oracle_label']-2e-12):
            raise AssertionError('Bounded-noise score failed to dominate latent score')
        trace={}
        for method,ss in scores.items():
            q=quantile(ss,.05)
            metrics,v,r=evaluate(yt,pt,st,d,('coef',np.repeat(q,2)))
            rows.append(dict(scenario=scenario,seed=seed,n=511,alpha=.05,eta=eta,
                noise=profile,u1=u1,u2=2.5,method=method,q=q,**metrics))
            trace[method+'_speed']=v;trace[method+'_failure']=r
        if seed==2000 and scenario=='heteroscedastic' and eta==.025 and profile=='under_read' and u1==.75:
            np.savez_compressed(ROOT/'results/measurement_example.npz',cal_y=yc,cal_measured=rec[2][1],
                cal_reconstructed=rec[2][0],cal_prediction=pc,cal_scale=sc,test_y=yt,
                test_prediction=pt,test_scale=st,clearance=d,**trace)
    return rows,dict(scenario=scenario,seed=seed,fit_seconds=fits,total_seconds=time.perf_counter()-tic)

def analyze_measurements():
    d=pd.read_csv(ROOT/'results/measurement_metrics.csv')
    groups=['eta','noise','u1','u2','method']
    summarize(d,groups).to_csv(ROOT/'results/measurement_summary.csv',index=False)
    summarize(d,['scenario']+groups).to_csv(ROOT/'results/measurement_scenarios.csv',index=False)
    blocks=d.groupby(['eta','noise','u1','u2','seed','method'])[['mean_speed','curve_failure','stop_failure']].mean().reset_index()
    rows=[]
    for key,z in blocks.groupby(['eta','noise','u1','u2']):
        speed=z.pivot(index='seed',columns='method',values='mean_speed')
        curve=z.pivot(index='seed',columns='method',values='curve_failure')
        a=speed.robust_label.to_numpy();b=speed.oracle_label.to_numpy();ratio=a.mean()/b.mean()
        infl=(a-ratio*b)/b.mean();h=t.ppf(.975,len(a)-1)*infl.std(ddof=1)/np.sqrt(len(a))
        diff=curve.naive_label-curve.robust_label;dh=t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff))
        rows.append(dict(zip(['eta','noise','u1','u2'],key),speed_loss_percent=100*(1-ratio),
            speed_loss_lo=100*(1-ratio-h),speed_loss_hi=100*(1-ratio+h),
            curve_risk_reduction=diff.mean(),curve_reduction_lo=diff.mean()-dh,curve_reduction_hi=diff.mean()+dh))
    pd.DataFrame(rows).to_csv(ROOT/'results/measurement_contrasts.csv',index=False)

def benchmark(repeats=1000):
    a=np.load(ROOT/'results/example_heteroscedastic.npz');n=511
    y,p,s=[a[key][:n] for key in ('cal_y','cal_prediction','cal_scale')]
    funcs={'endpoint':lambda:endpoint_score(y,p,s),'joint_grid_201':lambda:joint_grid_score(y,p,s),
           'robust_endpoint':lambda:robust_score(y,p,s,.025)}
    records=[]
    for name,fun in funcs.items():
        for _ in range(20):fun()
        timings=[]
        for _ in range(repeats):
            start=time.perf_counter_ns();fun();timings.append((time.perf_counter_ns()-start)/1000)
        records.append(dict(method=name,n=n,repeats=repeats,median_microseconds=float(np.median(timings)),
                            p05_microseconds=float(np.quantile(timings,.05)),p95_microseconds=float(np.quantile(timings,.95))))
    # Score and threshold equivalence verified on every stored original split through metrics.
    d=pd.read_csv(ROOT/'results/metrics.csv');keys=['scenario','seed','n','alpha','condition']
    e=d[d.method=='endpoint'].set_index(keys);g=d[d.method=='joint_grid'].set_index(keys)
    common=e.index.intersection(g.index);cols=['curve_failure','stop_failure','mean_speed']
    err=float(np.max(np.abs(e.loc[common,cols].to_numpy()-g.loc[common,cols].to_numpy())))
    assert err<1e-12
    out=dict(timings=records,joint_endpoint_max_metric_difference=err,matched_settings=len(common),
             platform=platform.platform(),python=sys.version,scope='score computation only; fitting, sorting and control integration excluded')
    (ROOT/'results/score_benchmark.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--seeds',type=int,default=40);ap.add_argument('--test-size',type=int,default=10000)
    ap.add_argument('--analyze-only',action='store_true');ap.add_argument('--benchmark-only',action='store_true');args=ap.parse_args()
    if args.benchmark_only:benchmark();return
    if not args.analyze_only:
        t0=time.perf_counter()
        runs=Parallel(n_jobs=args.workers,verbose=10)(delayed(one_measurement_run)(s,k,args.test_size)
            for s in SCENARIOS for k in range(2000,2000+args.seeds))
        pd.DataFrame([r for rows,_ in runs for r in rows]).to_csv(ROOT/'results/measurement_metrics.csv',index=False)
        (ROOT/'results/measurement_runtime.json').write_text(json.dumps(dict(settings=vars(args),
            wall_seconds=time.perf_counter()-t0,platform=platform.platform(),python=sys.version,
            runs=[m for _,m in runs]),indent=2))
    analyze_measurements()
    print(pd.read_csv(ROOT/'results/measurement_summary.csv').query('u1==.75')[['eta','noise','method','curve_failure','stop_failure','mean_speed']].to_string(index=False))
if __name__=='__main__':main()
