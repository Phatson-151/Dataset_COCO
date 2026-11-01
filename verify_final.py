"""Independently re-read all selected artifacts and fail on inconsistencies."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.special import expit
from sklearn.metrics import average_precision_score
from native_final import native_image,jwrite,NAMES
from paper_scsc.data import ROOT,manifest
from paper_scsc.experiments import OUT,tag


def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)


def main():
    selection=read(OUT/'experiment_configs/selection.json');nonconverged=[];model_count=0
    for p in sorted((OUT/'checkpoints').glob('*/completed.json')):
        r=read(p);h=read(p.parent/'history.json');cfg=read(p.parent/'config.json')
        assert len(h)==cfg['outer']==10
        assert len(list(p.parent.glob('outer_*.npz')))==10
        model_count+=1
        for row in h:
            for step in ['z_admm','d_admm']:
                d=row[step];last=d['residuals'][-1]
                assert all(np.isfinite(last[f]) for f in ['primal_residual','dual_residual','primal_tolerance','dual_tolerance'])
                if d['converged']:
                    assert last['primal_residual']<=last['primal_tolerance']
                    assert last['dual_residual']<=last['dual_tolerance']
                else:nonconverged.append(dict(id=p.parent.name,outer=row['outer'],step=step,
                    primal_ratio=last['primal_relative'],dual_ratio=last['dual_relative_proxy']))
        a=np.load(p.parent/'model.npz')
        assert a['D'].shape==(cfg['K'],cfg['filter_size'],cfg['filter_size'])
        assert np.linalg.norm(a['D'],axis=(1,2)).max()<=1.00001
        assert all(np.isfinite(a[k]).all() for k in a.files)
        assert r['z_converged']==sum(row['z_admm']['converged'] for row in h)
        assert r['d_converged']==sum(row['d_admm']['converged'] for row in h)
    pair=selection['paired_models'];ids=[tag(c['K'],c['gamma'],c['N']) for c in pair]
    banks=[read(OUT/'checkpoints'/k/'training_bank.json') for k in ids]
    assert banks[0]['tile_signals_sha256']==banks[1]['tile_signals_sha256']
    assert banks[0]['tile_labels_sha256']==banks[1]['tile_labels_sha256']
    native_checked=0
    for split in ['train','val','test_sealed']:
        for e in manifest()[split]:native_image(e);native_checked+=1
    predictions=0;inpainted=0
    for k in ids:
        for split in ['val','test_sealed']:
            metric=read(OUT/'metrics'/f'{k}_{split}.json')
            total=np.zeros((4,4),dtype='int64');all_y=[];all_s=[];psnrs=[]
            for e in manifest()[split]:
                im=native_image(e);a=np.load(OUT/'score_maps'/f'{k}_{split}_{im["id"]}.npz')
                logits=a['logits'];assert logits.shape==(4,*im['shape'])
                assert all(np.isfinite(a[f]).all() for f in ['logits','probabilities','reconstruction'])
                close(a['probabilities'],expit(logits))
                pred=logits.argmax(0);np.testing.assert_array_equal(pred,a['prediction_internal'])
                np.testing.assert_array_equal(a['prediction'],np.array([1,2,3,0])[pred])
                valid=~im['ignore'];y=im['gt'].argmax(0)[valid];p=pred[valid]
                cm=np.bincount(y*4+p,minlength=16).reshape(4,4);total+=cm
                row=next(r for r in metric['per_image'] if r['image_id']==im['id'])
                np.testing.assert_array_equal(cm,row['confusion_gt_rows_pred_cols'])
                psnr=float(-10*np.log10(max(float(np.mean((im['x']+im['low']-a['reconstruction'])**2)),1e-12)))
                close(psnr,row['PSNR_db']);psnrs.append(psnr)
                all_y.append(y);all_s.append(logits[:,valid]);predictions+=1
            m=metric['aggregate'];np.testing.assert_array_equal(total,m['confusion_gt_rows_pred_cols'])
            close(np.mean(psnrs),m['PSNR_db_mean_image'])
            y=np.concatenate(all_y);s=np.concatenate(all_s,axis=1)
            for c,name in enumerate(NAMES):
                close(average_precision_score(y==c,s[c]),m['pooled_pixel_AP'][name])
                iou=total[c,c]/(total[c].sum()+total[:,c].sum()-total[c,c])
                close(iou,m['per_class'][name]['iou'])
        inp=read(OUT/'metrics'/f'{k}_inpainting.json')
        for e in manifest()['test_sealed']:
            im=native_image(e);original=im['x']+im['low']
            a=np.load(OUT/'inpainting'/f'{k}_{im["id"]}.npz')
            b=np.load(OUT/'inpainting'/f'{ids[1] if k==ids[0] else ids[0]}_{im["id"]}.npz')
            observed=a['observed'];assert observed.shape==original.shape
            np.testing.assert_array_equal(observed,b['observed'])
            assert np.count_nonzero(~observed)==observed.size//2
            np.testing.assert_array_equal(a['masked'],np.where(observed,original,0))
            row=next(r for r in inp['per_image'] if r['image_id']==im['id'])
            assert hashlib.sha256(observed.tobytes()).hexdigest()==row['observed_mask_sha256']
            for f,mask in [('PSNR_all_db',np.ones_like(observed)),('PSNR_missing_db',~observed)]:
                value=-10*np.log10(max(float(np.mean((original[mask]-a['reconstruction'][mask])**2)),1e-12))
                close(value,row[f])
            inpainted+=1
    figures=list((OUT/'figures').rglob('*.png'))
    assert figures
    for p in figures:
        with Image.open(p) as image:
            image.verify()
    required=['figures/figure3_dictionaries','figures/figure4_multiclass','figures/figure5_ap_psnr',
        'figures/figure6_class_psnr','tables/table1_gamma_ap_psnr','tables/table2_inpainting']
    for name in required:assert (OUT/name).is_dir() and any((OUT/name).iterdir())
    for c in selection['gamma_configs']+selection['K_configs']+selection['N_configs']:
        k=tag(c['K'],c['gamma'],c['N']);assert (OUT/'metrics'/f'{k}_val.json').exists()
    status=read(OUT/'experiment_configs/experiment_status.json')
    for p in (OUT/'checkpoints').glob('*/completed.json'):
        c=read(p)
        if p.parent.name in status:
            ratio=(c['z_converged']+c['d_converged'])/c['total_subproblems']
            status[p.parent.name]['ADMM_convergence_ratio']=ratio
            status[p.parent.name]['optimization_status']='converged_subproblems' if ratio==1 else 'inconclusive_inner_convergence'
    jwrite(OUT/'experiment_configs/experiment_status.json',status)
    failed={k:v for k,v in status.items() if v['status']!='completed'}
    report=dict(artifact_consistency_passed=True,models_checked=model_count,
        original_images_verified=native_checked,selected_native_predictions_recomputed=predictions,
        paired_inpaintings_verified=inpainted,PNG_files_verified=len(figures),
        nonconverged_primary_subproblems=nonconverged,failed_or_unrun_registered_experiments=failed,
        all_primary_subproblems_converged=not nonconverged,
        independent_holdout_available=False,full_frame_training=False,training_seed_count=1,
        fully_verified_final=False,
        limitations=['TEST previously seen; no new independent holdout','N>=20 unavailable; 18 total images',
            'sampled grayscale native tiles, not full-frame RGB dictionary learning',
            'no other-dataset inpainting','single training seed'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    jwrite(OUT/'verification.json',report)
    inventory={str(p.relative_to(ROOT)):digest(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name!='SHA256SUMS.json' and p.suffix!='.lock' and 'pilot' not in p.parts}
    jwrite(OUT/'SHA256SUMS.json',inventory)
    print(json.dumps({k:v for k,v in report.items() if k!='nonconverged_primary_subproblems'},indent=2))


if __name__=='__main__':main()
