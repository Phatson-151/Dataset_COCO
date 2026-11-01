"""Execute staged, validation-only selection; subprocess logs preserve failures."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/workspace/.cache/matplotlib'
os.environ['MPLBACKEND']='Agg'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import concurrent.futures
import json
import subprocess
import sys
import time
import traceback
from paper_scsc.experiments import OUT,config,tag,directory
from native_final import jwrite

STATUS=OUT/'experiment_configs/experiment_status.json'
registry={}


def run_one(cfg):
    key=tag(cfg['K'],cfg['gamma'],cfg['N'])
    args=['--K',str(cfg['K']),'--gamma',str(cfg['gamma']),'--N',str(cfg['N'])]
    start=time.time()
    with (directory('logs')/f'{key}.log').open('a') as log:
        for stage in ['train','eval']:
            result=subprocess.run([sys.executable,'-m','paper_scsc.experiments',stage,*args],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                return key,dict(status='failed',stage=stage,returncode=result.returncode,seconds=time.time()-start,config=cfg)
    return key,dict(status='completed',seconds=time.time()-start,config=cfg)


def batch(configs):
    todo=[c for c in configs if tag(c['K'],c['gamma'],c['N']) not in registry]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_one,c) for c in todo]
        for future in concurrent.futures.as_completed(futures):
            key,result=future.result();registry[key]=result
            jwrite(STATUS,registry)
            print(key,result['status'],f'{result["seconds"]:.1f}s',flush=True)


def select(configs):
    scores=[]
    for c in configs:
        key=tag(c['K'],c['gamma'],c['N'])
        path=OUT/'metrics'/f'{key}_val.json'
        if path.exists():
            m=json.loads(path.read_text())
            scores.append((m['aggregate']['macro_AP_foreground'],c))
    if not scores:raise RuntimeError('No successful validation candidates')
    scores.sort(key=lambda r:(-r[0],r[1]['K'],r[1]['gamma']))
    return scores[0][1],scores


def main():
    if STATUS.exists():registry.update(json.loads(STATUS.read_text()))
    gamma_configs=[config(4,g,10) for g in [0,.05,.1,.15,.2,.5,1,2,5,10,100]]
    batch(gamma_configs)
    # SCSC gamma is selected amongst supervised candidates; CSC is reported separately.
    best_g,rank_g=select([c for c in gamma_configs if c['gamma']>0])
    jwrite(OUT/'experiment_configs/gamma_selection.json',dict(selected=best_g,ranking=rank_g))
    ks=[4,8,12,16,32,49,100]
    k_configs=[config(k,g,10) for k in ks for g in [0,best_g['gamma']]]
    batch(k_configs)
    best,rank_k=select([c for c in k_configs if c['gamma']>0])
    # A controlled N learning curve need not repeat the most expensive K.
    # Fix K=4 (the gamma-sweep reference) for all N, with the VAL-selected gamma.
    n_configs=[config(4,g,n) for n in [1,5,10] for g in [0,best_g['gamma']]]
    batch(n_configs)
    # N is a learning curve only: final model always uses all 10 TRAIN images.
    pair=[config(best['K'],g,10) for g in [0,best_g['gamma']]]
    jwrite(OUT/'experiment_configs/selection.json',dict(selected=best,paired_models=pair,
        gamma_ranking=rank_g,K_ranking=rank_k,rule='Maximum pooled VAL foreground AP; no TEST selection',
        gamma_configs=gamma_configs,K_configs=k_configs,N_configs=n_configs,N_reference_K=4))
    for c in pair:
        key=tag(c['K'],c['gamma'],c['N'])
        with (directory('logs')/f'{key}_final.log').open('a') as log:
            for action,extra in [('eval',['--split','test_sealed']),('inpaint',[])]:
                result=subprocess.run([sys.executable,'-m','paper_scsc.experiments',action,
                    '--K',str(c['K']),'--gamma',str(c['gamma']),'--N',str(c['N']),*extra],stdout=log,stderr=subprocess.STDOUT)
                registry[f'{key}_{action}_final']=dict(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode)
                jwrite(STATUS,registry)
                print(key,action,result.returncode,flush=True)


if __name__=='__main__':main()
