"""Summarize independent-split results; intervals use 40 seed blocks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]

def summarize(frame,groups):
    metrics=['curve_failure','stop_failure','mean_speed','speed_ratio','conservative_loss','abstain']
    blocks=frame.groupby(groups+['seed'])[metrics].mean().reset_index()
    rows=[]
    for key,a in blocks.groupby(groups):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(groups,key));row['replications']=len(a)
        for metric in metrics:
            z=a[metric].to_numpy();mu=float(z.mean());h=float(t.ppf(.975,len(z)-1)*z.std(ddof=1)/np.sqrt(len(z)))
            row[metric]=mu;row[metric+'_lo']=mu-h;row[metric+'_hi']=mu+h
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    d=pd.read_csv(ROOT/'results/metrics.csv')
    summarize(d,['n','alpha','condition','method']).to_csv(ROOT/'results/summary_macro.csv',index=False)
    summarize(d,['scenario','n','alpha','condition','method']).to_csv(ROOT/'results/summary_scenarios.csv',index=False)
    primary=d.query("n==511 and alpha==.05 and condition=='standard'")
    table=primary.pivot_table(index=['seed'],columns='method',values='mean_speed')
    contrasts=[]
    for method in ['coefficient','grid_bonferroni','constant_scale','policy','tolerance']:
        z=table.endpoint-table[method];h=t.ppf(.975,len(z)-1)*z.std(ddof=1)/np.sqrt(len(z))
        # Paired delta-method interval for the ratio of mean speeds.
        a=table.endpoint.to_numpy();b=table[method].to_numpy();r=a.mean()/b.mean()
        inf=(a-r*b)/b.mean();rh=t.ppf(.975,len(z)-1)*inf.std(ddof=1)/np.sqrt(len(z))
        contrasts.append(dict(baseline=method,speed_difference=z.mean(),difference_lo=z.mean()-h,difference_hi=z.mean()+h,
            relative_percent=100*(r-1),relative_lo=100*(r-1-rh),relative_hi=100*(r-1+rh)))
    pd.DataFrame(contrasts).to_csv(ROOT/'results/paired_contrasts.csv',index=False)
    details={'rows':len(d),'independent_splits':int(primary[['scenario','seed']].drop_duplicates().shape[0]),
             'unique_test_episodes':int(primary[['scenario','seed']].drop_duplicates().shape[0]*json.loads((ROOT/'results/metrics_runtime.json').read_text())['settings']['test_size']),
             'risk_above_target_split_fraction':primary.groupby('method')['curve_failure'].apply(lambda x:float((x>.05).mean())).to_dict()}
    (ROOT/'results/analysis_details.json').write_text(json.dumps(details,indent=2))
    print(pd.DataFrame(contrasts).round(5).to_string(index=False))
if __name__=='__main__':main()
