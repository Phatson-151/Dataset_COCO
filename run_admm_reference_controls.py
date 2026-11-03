"""Repeat ADMM ablations with an adequate D-step ceiling; preserve budget pilots."""
import json
from run_v2_experiments import batch
from paper_scsc import v2 as ex
from native_final import jwrite

def main():
 p=ex.OUT/'experiment_configs/selection.json';s=json.loads(p.read_text());common=s['all_recipe_settings'];g=s['selected']['gamma']
 refs=[ex.config(4,gamma,10,**common,solver='admm',loss=loss,d_maxiter=1000,variant='reference_budget') for loss in ['softmax','ova'] for gamma in [0,g]]
 batch(refs)
 s['budgeted_ADMM_pilots']=[c for c in s['ablations'] if c['solver']=='admm']
 s['ablations']=[c for c in s['ablations'] if c['solver']=='fista']+refs
 s['ADMM_budget_note']='Production default D ceiling160 pilots did not converge. Reference ablation repeats with ceiling1000, same tolerances; failures still retained.'
 jwrite(p,s)
 print('REFERENCE ADMM ABLATIONS COMPLETE',flush=True)
if __name__=='__main__':main()
