"""Render ONLY new stored predictions/metrics; no training or result fabrication."""
import os
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from paper_scsc.experiments import OUT,tag,NAMES,config
from paper_scsc.data import ROOT,manifest
from native_final import native_image,jwrite

COLORS=['#ef5450','#428bf2','#ffb734','#30343b']
CMAP=ListedColormap(COLORS)


def load(path):return json.loads(Path(path).read_text())
def outdir(name):
    p=OUT/name;p.mkdir(parents=True,exist_ok=True);return p
def save(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)
def csvwrite(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def key(cfg):return tag(cfg['K'],cfg['gamma'],cfg['N'])
def result(cfg,split='val'):return load(OUT/'metrics'/f'{key(cfg)}_{split}.json')
def ap(cfg,name=None):
    m=result(cfg)['aggregate'];return m['macro_AP_foreground'] if name is None else m['pooled_pixel_AP'][name]
def psnr(cfg,name=None):
    m=result(cfg)['aggregate'];return m['PSNR_db_mean_image'] if name is None else m['class_PSNR_db_mean_image'][name]


def dictionaries(selection):
    folder=outdir('figures/figure3_dictionaries')
    configs=selection['K_configs'];pairs={}
    for cfg in configs:pairs.setdefault(cfg['K'],[]).append(cfg)
    for K,pair in pairs.items():
        if len(pair)!=2:continue
        arrays=[np.load(OUT/'checkpoints'/key(c)/'model.npz')['D'] for c in pair]
        vmax=max(np.max(np.abs(a)) for a in arrays);cols=int(np.ceil(np.sqrt(K)));rows=int(np.ceil(K/cols))
        fig,axs=plt.subplots(1,2,figsize=(12,6))
        for ax,D,cfg in zip(axs,arrays,pair):
            f=D.shape[-1];mosaic=np.full((rows*(f+1),cols*(f+1)),np.nan)
            for i,d in enumerate(D):
                y,x=divmod(i,cols);mosaic[y*(f+1):y*(f+1)+f,x*(f+1):x*(f+1)+f]=d
            ax.imshow(mosaic,cmap='gray',vmin=-vmax,vmax=vmax);ax.axis('off')
            ax.set_title(f'{"CSC" if cfg["gamma"]==0 else "SCSC"} | K={K} | gamma={cfg["gamma"]:g}')
        fig.suptitle('Figure 3: all learned native-pixel grayscale filters')
        save(fig,folder/f'figure3_K{K}.png')
    pair=selection['paired_models'];fig,axs=plt.subplots(2,3,figsize=(12,7))
    vmax=max(abs(np.load(OUT/'checkpoints'/key(c)/f'outer_{o:02}.npz')['D']).max() for c in pair for o in [1,5,10])
    for row,c in enumerate(pair):
        for col,o in enumerate([1,5,10]):
            D=np.load(OUT/'checkpoints'/key(c)/f'outer_{o:02}.npz')['D'];K,f,_=D.shape
            cols=int(np.ceil(np.sqrt(K)));rows=int(np.ceil(K/cols));a=np.full((rows*(f+1),cols*(f+1)),np.nan)
            for i,d in enumerate(D):
                y,x=divmod(i,cols);a[y*(f+1):y*(f+1)+f,x*(f+1):x*(f+1)+f]=d
            axs[row,col].imshow(a,cmap='gray',vmin=-vmax,vmax=vmax);axs[row,col].axis('off')
            axs[row,col].set_title(f'{"CSC" if row==0 else "SCSC"}, outer {o}')
    save(fig,folder/'dictionary_evolution.png')


def maps(selection):
    folder=outdir('figures/figure4_multiclass');pair=selection['paired_models']
    for entry in manifest()['test_sealed']:
        im=native_image(entry);id=im['id'];gt=im['gt'].argmax(0)
        arrays=[np.load(OUT/'score_maps'/f'{key(c)}_test_sealed_{id}.npz') for c in pair]
        # All images are rendered, not chosen by advantage for either arm.
        fig,axs=plt.subplots(4,4,figsize=(16,13))
        for c,name in enumerate(NAMES):
            axs[c,0].imshow(im['rgb'])
            for arm in [0,1]:axs[c,arm+1].imshow(arrays[arm]['probabilities'][c],cmap='magma',vmin=0,vmax=1)
            ground=np.ma.masked_where(im['ignore'],im['gt'][c].astype('float32'))
            axs[c,3].imshow(ground,cmap='gray',vmin=0,vmax=1)
            for j,title in enumerate(['Original RGB',f'CSC {name} OVA probability',f'SCSC {name} OVA probability',f'{name} GT']):
                axs[c,j].set_title(title);axs[c,j].axis('off')
        fig.suptitle(f'Figure 4 — {id} | previously-seen TEST | fixed probability scale 0–1')
        save(fig,folder/f'figure4_all_classes_{id}.png')
        for c,name in enumerate(NAMES):
            fig,axs=plt.subplots(1,4,figsize=(16,4))
            axs[0].imshow(im['rgb']);axs[1].imshow(arrays[0]['probabilities'][c],cmap='magma',vmin=0,vmax=1)
            axs[2].imshow(arrays[1]['probabilities'][c],cmap='magma',vmin=0,vmax=1)
            axs[3].imshow(np.ma.masked_where(im['ignore'],im['gt'][c].astype(float)),cmap='gray',vmin=0,vmax=1)
            for a,t in zip(axs,['(a) Input',f'(b) CSC {name}',f'(c) SCSC {name}','(d) GT']):a.set_title(t);a.axis('off')
            fig.suptitle(f'{name} raw OVA probability, ID {id}; shared 0–1 scale')
            save(fig,folder/f'figure4_{name}_{id}.png')
        fig,axs=plt.subplots(4,4,figsize=(16,13))
        for c,name in enumerate(NAMES):
            axs[c,0].imshow(im['rgb'])
            for arm in [0,1]:axs[c,arm+1].imshow(arrays[arm]['prediction_internal']==c,cmap='gray',vmin=0,vmax=1)
            axs[c,3].imshow(np.ma.masked_where(im['ignore'],im['gt'][c].astype(float)),cmap='gray',vmin=0,vmax=1)
            for a,t in zip(axs[c],['Input',f'CSC {name} argmax',f'SCSC {name} argmax','GT']):a.set_title(t);a.axis('off')
        save(fig,folder/f'binary_predictions_{id}.png')
        fig,axs=plt.subplots(1,4,figsize=(16,4))
        axs[0].imshow(im['rgb'])
        for a in [0,1]:axs[a+1].imshow(arrays[a]['prediction_internal'],cmap=CMAP,vmin=0,vmax=3,interpolation='nearest')
        axs[3].imshow(np.ma.masked_where(im['ignore'],gt),cmap=CMAP,vmin=0,vmax=3,interpolation='nearest')
        for a,t in zip(axs,['Original','CSC argmax','SCSC argmax','GT (ignore masked)']):a.set_title(t);a.axis('off')
        fig.suptitle('Person=red; Car=blue; Motorcycle=yellow; Background=dark grey')
        save(fig,folder/f'multiclass_{id}.png')
        fig,axs=plt.subplots(1,3,figsize=(12,4));original=im['x']+im['low']
        for ax,image,title in zip(axs,[original,*[a['reconstruction'] for a in arrays]],['Original grayscale','CSC reconstruction','SCSC reconstruction']):
            ax.imshow(image,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
        save(fig,outdir('figures/reconstruction')/f'reconstruction_{id}.png')


def curves(selection):
    gammas=selection['gamma_configs'];ks=selection['K_configs'];ns=selection['N_configs']
    folder=outdir('figures/figure5_ap_psnr')
    for name in [None,*NAMES]:
        fig,axs=plt.subplots(2,3,figsize=(15,8))
        for col,(axis,configs) in enumerate([('gamma',gammas),('K',ks),('N',ns)]):
            for row,fun in enumerate([ap,psnr]):
                ax=axs[row,col]
                if axis=='gamma':
                    baseline=[c for c in configs if c['gamma']==0][0]
                    supervised=[c for c in configs if c['gamma']>0]
                    ax.axhline(fun(baseline,name),label='CSC gamma=0',color='tab:blue')
                    ax.plot([c['gamma'] for c in supervised],[fun(c,name) for c in supervised],'o-',label='SCSC',color='tab:orange')
                    ax.set_xscale('symlog',linthresh=.05)
                else:
                    for arm in [0,1]:
                        selected=sorted([c for c in configs if (c['gamma']>0)==bool(arm)],key=lambda c:c[axis])
                        ax.plot([c[axis] for c in selected],[fun(c,name) for c in selected],'o-',label='SCSC' if arm else 'CSC')
                ax.set_xlabel(dict(gamma='gamma (classification weight)',K='Number of filters K',N='Number of TRAIN images N')[axis])
                ax.set_ylabel('Pixel AP (0–1)' if row==0 else 'PSNR (dB)');ax.grid(alpha=.25);ax.legend()
                ax.set_title(f'({"abcdef"[row*3+col]}) VAL {"foreground macro" if name is None else name}')
                one,oa=plt.subplots(figsize=(6,4))
                for line in ax.get_lines():
                    if axis=='gamma' and line.get_label()=='CSC gamma=0':
                        oa.axhline(line.get_ydata()[0],label=line.get_label(),color=line.get_color())
                    else:
                        oa.plot(line.get_xdata(),line.get_ydata(),marker=line.get_marker(),color=line.get_color(),label=line.get_label())
                if axis=='gamma':oa.set_xscale('symlog',linthresh=.05)
                oa.set_xlabel(ax.get_xlabel());oa.set_ylabel(ax.get_ylabel());oa.set_title(ax.get_title())
                oa.legend();oa.grid(alpha=.25)
                save(one,folder/f'figure5_{name or "macro"}_{"abcdef"[row*3+col]}.png')
        fig.suptitle('Figure 5 — native-pixel tile adaptation; VAL is used for selection')
        save(fig,folder/f'figure5_{name or "macro"}.png')
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    baseline=gammas[0];sup=[c for c in gammas if c['gamma']>0]
    for ax,name in zip(axs,NAMES[:3]):
        ax.axhline(psnr(baseline,name),label='CSC',color='tab:blue')
        ax.plot([c['gamma'] for c in sup],[psnr(c,name) for c in sup],'o-',label='SCSC',color='tab:orange')
        ax.set_xscale('symlog',linthresh=.05);ax.set_xlabel('gamma');ax.set_ylabel('GT-region PSNR (dB)')
        ax.set_title(name);ax.legend();ax.grid(alpha=.25)
    fig.suptitle('Figure 6 — VAL native grayscale reconstruction inside each class mask')
    save(fig,outdir('figures/figure6_class_psnr')/'figure6_class_PSNR.png')


def tables(selection):
    folder=outdir('tables/table1_gamma_ap_psnr');rows=[]
    for cfg in selection['gamma_configs']:
        m=result(cfg)['aggregate'];row=dict(id=key(cfg),gamma=cfg['gamma'],split='VAL')
        row.update({f'{n}_AP':m['pooled_pixel_AP'][n] for n in NAMES})
        row['foreground_macro_AP']=m['macro_AP_foreground'];row['global_PSNR_db']=m['PSNR_db_mean_image']
        row.update({f'{n}_PSNR_db':m['class_PSNR_db_mean_image'][n] for n in NAMES})
        rows.append(row)
    csvwrite(folder/'table1_VAL.csv',rows)
    # Paper-oriented layout: metrics in rows and gamma columns.
    fields=[f for f in rows[0] if f not in ['id','gamma','split']]
    csvwrite(folder/'table1_VAL_transposed.csv',[dict(metric=f,**{f'gamma_{r["gamma"]:g}':r[f] for r in rows}) for f in fields])
    allrows=[];curve=[];convrows=[];diversity=[]
    for path in sorted((OUT/'metrics').glob('*_val.json')):
        r=load(path);c=r['config'];m=r['aggregate'];t=load(OUT/'checkpoints'/r['id']/'completed.json')
        h=load(OUT/'checkpoints'/r['id']/'history.json')
        curve.append(dict(id=r['id'],K=c['K'],gamma=c['gamma'],N=c['N'],macro_AP=m['macro_AP_foreground'],PSNR_db=m['PSNR_db_mean_image'],**{f'{n}_AP':m['pooled_pixel_AP'][n] for n in NAMES},**{f'{n}_PSNR_db':m['class_PSNR_db_mean_image'][n] for n in NAMES}))
        convrows.append(dict(id=r['id'],seconds=t['seconds'],peak_rss_MiB=t['peak_rss_MiB'],Z_converged=t['z_converged'],D_converged=t['d_converged'],subproblems=t['total_subproblems'],ADMM_convergence_ratio=(t['z_converged']+t['d_converged'])/t['total_subproblems'],fresh_encoding_converged=t['fresh_encoding']['converged'],fresh_classifier_success=t['fresh_classifier']['success']))
        D=np.load(OUT/'checkpoints'/r['id']/'model.npz')['D'].reshape(c['K'],-1)
        dn=D/np.maximum(np.linalg.norm(D,axis=1,keepdims=True),1e-12);corr=np.abs(dn@dn.T);off=corr[~np.eye(c['K'],dtype=bool)]
        diversity.append(dict(id=r['id'],mean_absolute_filter_correlation=float(off.mean()),max_absolute_filter_correlation=float(off.max()),max_filter_norm=float(np.linalg.norm(D,axis=1).max())))
    csvwrite(outdir('tables')/'figure5_6_numeric_source.csv',curve)
    csvwrite(outdir('tables')/'convergence_runtime.csv',convrows)
    csvwrite(outdir('tables')/'dictionary_diversity.csv',diversity)
    for split in ['val','test_sealed']:
        selected=[]
        for cfg in selection['paired_models']:
            r=result(cfg,split);m=r['aggregate'];arm='CSC' if cfg['gamma']==0 else 'SCSC'
            for name in NAMES:
                selected.append(dict(arm=arm,split=split,class_name=name,**m['per_class'][name],AP=m['pooled_pixel_AP'][name],class_PSNR_db=m['class_PSNR_db_mean_image'][name]))
            cm=np.array(m['confusion_gt_rows_pred_cols']);fig,axs=plt.subplots(1,2,figsize=(10,4))
            order=[3,0,1,2]
            for ax,a,title in zip(axs,[cm,np.array(m['normalized_confusion'])],['Pixel counts','GT-row normalized']):
                a=a[np.ix_(order,order)];ax.imshow(a,cmap='Blues');ax.set_xticks(range(4),[NAMES[i] for i in order],rotation=30);ax.set_yticks(range(4),[NAMES[i] for i in order]);ax.set_xlabel('Prediction');ax.set_ylabel('Ground truth');ax.set_title(title)
                for y in range(4):
                    for x in range(4):ax.text(x,y,f'{a[y,x]:.2f}' if title!='Pixel counts' else str(int(a[y,x])),ha='center',va='center',fontsize=8)
            fig.suptitle(f'{arm} | {split} | all class confusions retained')
            save(fig,outdir('figures/confusion_matrices')/f'{arm}_{split}.png')
            csvwrite(outdir('tables/multiclass_metrics')/f'confusion_{arm}_{split}.csv',[
                dict(GT=NAMES[y],**{NAMES[x]:int(cm[y,x]) for x in order}) for y in order])
            norm=np.array(m['normalized_confusion'])
            csvwrite(outdir('tables/multiclass_metrics')/f'normalized_confusion_{arm}_{split}.csv',[
                dict(GT=NAMES[y],**{NAMES[x]:float(norm[y,x]) for x in order}) for y in order])
        csvwrite(outdir('tables/multiclass_metrics')/f'metrics_{split}.csv',selected)
    rows=[];pair=selection['paired_models']
    for cfg in pair:
        r=load(OUT/'metrics'/f'{key(cfg)}_inpainting.json')
        for row in r['per_image']:rows.append(dict(arm='CSC' if cfg['gamma']==0 else 'SCSC',**{k:v for k,v in row.items() if k!='encode'}))
    csvwrite(outdir('tables/table2_inpainting')/'table2_per_image.csv',rows)
    csvwrite(outdir('tables/table2_inpainting')/'table2_mean.csv',[
        dict(arm=arm,**{f:float(np.mean([r[f] for r in rows if r['arm']==arm])) for f in ['PSNR_all_db','PSNR_missing_db','SSIM','relative_error']}) for arm in ['CSC','SCSC']])
    for e in manifest()['test_sealed']:
        im=native_image(e);arrays=[np.load(OUT/'inpainting'/f'{key(c)}_{im["id"]}.npz') for c in pair]
        assert np.array_equal(arrays[0]['observed'],arrays[1]['observed'])
        fig,axs=plt.subplots(1,4,figsize=(16,4))
        images=[im['x']+im['low'],arrays[0]['masked'],arrays[0]['reconstruction'],arrays[1]['reconstruction']]
        for ax,image,title in zip(axs,images,['Original grayscale','50% missing (zeroed)','CSC inpainting','SCSC inpainting']):ax.imshow(image,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
        save(fig,outdir('figures/inpainting')/f'inpainting_{im["id"]}.png')
    fig,ax=plt.subplots(figsize=(8,4))
    for arm in ['CSC','SCSC']:
        r=[r for r in rows if r['arm']==arm];ax.plot([str(v['image_id']) for v in r],[v['PSNR_missing_db'] for v in r],'o-',label=arm)
    ax.legend();ax.set_ylabel('Missing-pixel PSNR (dB)');ax.set_xlabel('Previously-seen TEST image ID');ax.grid(alpha=.25)
    save(fig,outdir('figures/inpainting')/'inpainting_PSNR_comparison.png')


def convergence():
    folder=outdir('figures/convergence')
    for p in sorted((OUT/'checkpoints').glob('*/history.json')):
        h=load(p);fig,axs=plt.subplots(1,3,figsize=(15,4));x=[r['outer'] for r in h]
        for field in ['reconstruction','sparsity','classification','total']:
            axs[0].plot(x,[r['after_classifier'][field] for r in h],label=field)
        axs[0].set_yscale('symlog',linthresh=1);axs[0].legend(fontsize=8);axs[0].set_title('Objective components (classification unweighted)')
        for ax,step in zip(axs[1:],['z_admm','d_admm']):
            for field,label in [('primal_relative','primal / tolerance'),('dual_relative_proxy','dual / tolerance')]:
                ax.plot(x,[r[step]['residuals'][-1][field] for r in h],'o-',label=label)
            ax.axhline(1,c='black',ls='--');ax.set_yscale('log');ax.legend(fontsize=8)
            ax.set_title(f'{step}: {sum(r[step]["converged"] for r in h)}/{len(h)} converged')
        for ax in axs:ax.set_xlabel('Outer iteration');ax.grid(alpha=.2)
        fig.suptitle(p.parent.name);save(fig,folder/f'{p.parent.name}.png')


def bootstrap(selection):
    a,b=[result(c,'test_sealed')['per_image'] for c in selection['paired_models']]
    assert [r['image_id'] for r in a]==[r['image_id'] for r in b]
    rng=np.random.default_rng(42);draws=[]
    from paper_scsc.experiments import from_cm
    for _ in range(2000):
        ix=rng.integers(0,len(a),len(a))
        cm=[np.sum([np.array(r[j]['confusion_gt_rows_pred_cols']) for j in ix],axis=0) for r in [a,b]]
        draws.append([from_cm(cm[1])['foreground_mIoU']-from_cm(cm[0])['foreground_mIoU'],
            np.mean([b[j]['macro_ap_foreground']-a[j]['macro_ap_foreground'] for j in ix]),
            np.mean([b[j]['PSNR_db']-a[j]['PSNR_db'] for j in ix])])
    jwrite(OUT/'metrics/paired_bootstrap.json',dict(seed=42,replicates=2000,num_images=len(a),
        caveat='Previously-seen TEST; image variability only, one training seed; no population significance claim',
        SCSC_minus_CSC_95_percentile_CI={n:np.percentile(np.array(draws)[:,i],[2.5,97.5]) for i,n in enumerate(['pooled_foreground_mIoU','mean_image_foreground_AP','mean_image_PSNR_db'])}))


def additional_metrics(selection):
    from sklearn.metrics import precision_recall_curve
    folder=outdir('figures/multiclass_metrics');fig,axs=plt.subplots(2,2,figsize=(10,8))
    pair=selection['paired_models'];table=[]
    fig_iou,ax_iou=plt.subplots(figsize=(9,5))
    for arm,cfg in zip(['CSC','SCSC'],pair):
        ys=[];ss=[]
        for e in manifest()['test_sealed']:
            im=native_image(e);valid=~im['ignore'];ys.append(im['gt'].argmax(0)[valid])
            a=np.load(OUT/'score_maps'/f'{key(cfg)}_test_sealed_{im["id"]}.npz')
            ss.append(a['logits'][:,valid])
        y=np.concatenate(ys);s=np.concatenate(ss,axis=1)
        for c,name in enumerate(NAMES):
            p,r,t=precision_recall_curve(y==c,s[c],drop_intermediate=True)
            ix=np.unique(np.linspace(0,len(p)-1,min(1001,len(p))).astype(int))
            csvwrite(outdir('tables/multiclass_metrics')/f'PR_{arm}_{name}.csv',[
                dict(recall=float(r[i]),precision=float(p[i]),logit_threshold=float(t[i]) if i<len(t) else None) for i in ix])
            axs.flat[c].plot(r[ix],p[ix],label=arm);axs.flat[c].set_title(name)
        m=result(cfg,'test_sealed')['aggregate'];offset=-.18 if arm=='CSC' else .18
        labels=[*NAMES,'Foreground mIoU','All-class mIoU']
        values=[m['per_class'][n]['iou'] for n in NAMES]+[m['foreground_mIoU'],m['all_class_mIoU']]
        ax_iou.bar(np.arange(6)+offset,values,width=.36,label=arm)
        row=dict(arm=arm,gamma=cfg['gamma'],K=cfg['K'],split='TEST_PREVIOUSLY_SEEN',
            **{f'{n}_AP':m['pooled_pixel_AP'][n] for n in NAMES},
            foreground_macro_AP=m['macro_AP_foreground'],global_PSNR_db=m['PSNR_db_mean_image'],
            **{f'{n}_PSNR_db':m['class_PSNR_db_mean_image'][n] for n in NAMES})
        table.append(row)
    for ax in axs.flat:ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.legend();ax.grid(alpha=.25)
    fig.suptitle('Pixel precision–recall on previously-seen TEST; no threshold selection')
    save(fig,folder/'precision_recall_curves.png')
    ax_iou.set_xticks(np.arange(6),labels,rotation=20);ax_iou.set_ylabel('IoU (0–1)');ax_iou.legend();ax_iou.grid(axis='y',alpha=.25)
    save(fig_iou,folder/'IoU_comparison.png')
    csvwrite(outdir('tables/table1_gamma_ap_psnr')/'table1_TEST_selected_only.csv',table)


def main():
    selection=load(OUT/'experiment_configs/selection.json')
    dictionaries(selection);maps(selection);curves(selection);tables(selection);convergence();bootstrap(selection);additional_metrics(selection)
    jwrite(OUT/'REPORT_METRICS.json',dict(selection=selection,
        VAL={('CSC' if c['gamma']==0 else 'SCSC'):result(c)['aggregate'] for c in selection['paired_models']},
        TEST_PREVIOUSLY_SEEN={('CSC' if c['gamma']==0 else 'SCSC'):result(c,'test_sealed')['aggregate'] for c in selection['paired_models']}))
    print('Figures 3–6, Tables 1–2, multiclass metrics and bootstrap generated from new results',flush=True)


if __name__=='__main__':main()
