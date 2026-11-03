"""Matched-objective solver benchmark; no iteration-count speed claims."""
import time,json
import numpy as np
from native_final import jwrite
from paper_scsc.solver import init_dict,solve_z,conv,kernel_fft,make_mask
from paper_scsc.v2_solver import code
from paper_scsc.benchmark_admm import solve_z_reference
from paper_scsc.v2 import OUT,directory

def main():
 rows=[]
 for K in [4,16,49]:
  x=np.random.default_rng(42).normal(size=(2,32,32)).astype('float32')*.05;d=init_dict(x,K,7);m=make_mask(x.shape[-2:],7)
  objective=lambda z:float(.5*np.sum((m*(conv(kernel_fft(d,x.shape[-2:]),z)-x))**2)+.015*abs(z).sum())
  for repeat in range(3):
   for solver in ['fista','admm']:
    start=time.perf_counter()
    if solver=='fista':z,diag=code(d,x,.015,2000,1e-5)
    else:z,diag=solve_z_reference(d,x,np.zeros((4,K),'float32'),np.zeros(4,'float32'),None,0,.015,maxiter=5000,tol=1e-6,adaptive=True,absolute_tolerance=1e-7)
    rows.append(dict(K=K,repeat=repeat,solver=solver,seconds=time.perf_counter()-start,objective=objective(z),converged=diag['converged'],iterations=diag['iterations'],relative_tolerance=(1e-5 if solver=='fista' else 1e-6),absolute_tolerance=(0 if solver=='fista' else 1e-7)))
 for row in rows:
  best=min(r['objective'] for r in rows if r['K']==row['K']);row['relative_objective_gap']=(row['objective']-best)/max(abs(best),1e-12)
  row['matched_accuracy']=row['relative_objective_gap']<=1e-3
 jwrite(directory('metrics')/'solver_benchmark.json',dict(protocol='same fixed dictionary, signal, beta and mask; strict stopping; 3 repeats, shared host load; speed interpretation requires matched_accuracy',rows=rows))
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
