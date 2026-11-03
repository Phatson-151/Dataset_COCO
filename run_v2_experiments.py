"""Run every v2 experiment; resume only matching completed configurations.
Selection uses VAL all-class mIoU with explicit Background constraints. TEST is
opened only after the full recipe, gamma and K are fixed. K=100 is excluded.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,sys,subprocess,time,concurrent.futures
from pathlib import Path
from paper_scsc import v2 as ex
from native_final import jwrite


def job(cfg):
    key=ex.tag(cfg);log=ex.directory('logs')/f'{key}.log'
    with log.open('a') as f:
        r=subprocess.run([sys.executable,__file__,'--job',json.dumps(cfg)],stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'{key} failed; see {log}')
    return key


def batch(configs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for key in pool.map(job,configs):print('COMPLETE',key,flush=True)


def score(cfg):
    a=json.loads((ex.OUT/'metrics'/f'{ex.tag(cfg)}_val.json').read_text())['aggregate']
    feasible=a['per_class']['Background']['recall']>=.1 and a['per_class']['Background']['iou']>=.1
    return (feasible,a['all_class_mIoU'],a['macro_AP_foreground'])


def main():
    import platform
    ex.directory('experiment_configs')
    prereg=dict(protocol='V2_TASK_DRIVEN_LABEL_FREE_FISTA_SOFTMAX',K=[4,8,12,16,32,49],gamma=[0,.05,.1,.15,.2,.5,1,2,5,10,100],N=[1,5,10],seeds=[42,43,44],
        joint_refinement='Top two K from initial VAL sweep, gamma in {initial_gamma,.2,1}; choose VAL gamma, then complete all K at that fixed gamma',
        rule='VAL all-class mIoU; prefer Background IoU and recall >= .1, then foreground AP tiebreak; no TEST selection',
        adaptations=['differentiable warmstarted 24-step label-free FISTA task encoder','signed/absolute/pooled sparse features plus native RGB context for both arms','fresh classifier for both arms','shared VAL CE temperature/bias calibration','12 native tiles per image: 50% ROI, 50% uniformly sampled','30 outer iterations, reconstruction warmup and supervision ramp'],
        known_unavailable=['N>10 unique TRAIN images','new independent holdout','other-dataset inpainting'],
        source_base='80a44dd6001e542b1637e511d70fe23a7cbdaca1',python=platform.python_version())
    jwrite(ex.OUT/'experiment_configs/protocol.json',prereg)
    # Small preregistered staged search, never a full Cartesian product.
    recipes=[dict(variant="recipe_default"),dict(beta=.005,variant='beta005'),dict(beta=.03,variant='beta03'),dict(alpha=.1,variant='alpha01'),dict(alpha=10.,variant='alpha10'),dict(filter_size=21,stride=32,variant='filter21')]
    trials=[ex.config(4,g,10,**r) for r in recipes for g in [0,.2]];batch(trials)
    best_recipe=max([c for c in trials if c['gamma']>0],key=score)
    common={k:best_recipe[k] for k in ['beta','alpha','filter_size','stride']}
    # Keep recipe names only on tuning trials; primary tags remain paper-compatible.
    gs=[ex.config(4,g,10,**common) for g in prereg['gamma']];batch(gs)
    bestg=max([c for c in gs if c['gamma']>0],key=score)['gamma']
    ks=[ex.config(k,g,10,**common) for k in prereg['K'] for g in [0,bestg]];batch(ks)
    initial_ks=ks
    top=sorted([c for c in ks if c['gamma']>0],key=score,reverse=True)[:2]
    joint=[ex.config(c['K'],g,10,**common) for c in top for g in sorted({bestg,.2,1.})]
    batch(joint)
    joint_best=max(joint,key=score);bestg=joint_best['gamma']
    ks=[ex.config(k,g,10,**common) for k in prereg['K'] for g in [0,bestg]];batch(ks)
    best=max([c for c in ks if c['gamma']>0],key=score);pair=[ex.config(best['K'],g,10,**common) for g in [0,bestg]]
    ns=[ex.config(4,g,n,**common) for n in prereg['N'] for g in [0,bestg]];batch(ns)
    selection=dict(selected=best,paired_models=pair,gamma_configs=gs,K_configs=ks,N_configs=ns,N_reference_K=4,
        recipe_search=trials,selected_recipe=best_recipe,joint_refinement=joint,initial_K_configs=initial_ks,rule=prereg['rule'],K_excluded=[100],all_recipe_settings=common)
    jwrite(ex.OUT/'experiment_configs/selection.json',selection)
    jwrite(ex.OUT/'experiment_configs/gamma_selection.json',dict(selected=ex.config(4,bestg,10,**common),ranking=[dict(id=ex.tag(c),score=score(c)) for c in gs]))
    # Fully crossed solver/loss ablation at K=4 with both arms; task gradients
    # always use the declared differentiable encoder, solver varies exact coding/D.
    ablations=[ex.config(4,g,10,**common,solver=solver,loss=loss,**(dict(d_maxiter=1000,variant='reference_budget') if solver=='admm' else {})) for solver in ['fista','admm'] for loss in ['softmax','ova'] for g in [0,bestg]]
    batch(ablations)
    seeds=[ex.config(best['K'],g,10,**common,seed=seed) for seed in prereg['seeds'] for g in [0,bestg]];batch(seeds)
    # Feature and sampling controls isolate the added context and larger bank.
    controls=[ex.config(4,g,10,**common,feature='signed',variant='signed_only') for g in [0,bestg]]
    controls += [ex.config(4,g,10,**common,tiles_per_image=4,variant='four_tiles') for g in [0,bestg]]
    batch(controls)
    selection.update(ablations=ablations,seed_configs=seeds,controls=controls)
    jwrite(ex.OUT/'experiment_configs/selection.json',selection)
    # Nothing below this line influences selection.
    for cfg in seeds:ex.evaluate(cfg,'test_sealed')
    for cfg in pair:ex.inpaint(cfg)
    jwrite(ex.OUT/'experiment_configs/experiment_status.json',{p.parent.name:json.loads(p.read_text())['status'] for p in (ex.OUT/'checkpoints').glob('*/completed.json')})
    print('ALL V2 EXPERIMENTS COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job');args=p.parse_args()
    if args.job:
        import fcntl
        c=json.loads(args.job)
        with (ex.directory('checkpoints')/(ex.tag(c)+'.lock')).open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ex.train(c);ex.evaluate(c)
    else:main()
