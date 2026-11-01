"""Verify, visualize and package the local-only contextual experiment."""
import os
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
import json,hashlib,zipfile,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from rapid_context import OUT,ARMS,read,metrics,sha,samples
from native_final import native_image,NAMES,jwrite
from paper_scsc.data import ROOT,manifest

COLORS=np.array([[239,84,80],[66,139,242],[255,183,52],[32,32,32]],dtype='uint8')

def main():
    result=read(OUT/'results.json');family=result['selected_family']
    protocol=read(OUT/'protocol.json')
    for arm,k in ARMS.items():assert sha(ROOT/'final output/checkpoints'/k/'model.npz')==protocol['dictionary_sha256'][arm]
    sample_record=read(OUT/'training_samples.json')
    for e,row in zip(manifest()['train'],sample_record):
        _,idx,y=samples(e)
        assert hashlib.sha256(idx.tobytes()).hexdigest()==row['indices_sha256']
        np.testing.assert_array_equal(np.bincount(y,minlength=4),row['counts'])
    fresh_encoding=[]
    for path in (OUT/'codes').glob('*.json'):
        diag=read(path)
        if diag['source'].startswith('fresh'):
            fresh_encoding.append(dict(name=path.stem,converged_batches=diag['converged_batches'],batch_count=diag['batch_count']))
    records=[r for r in result['records'] if 'TEST_PREVIOUSLY_SEEN' in r]
    verified=0
    for r in result['records']:
        for split,label in [('val','VAL'),('test_sealed','TEST_PREVIOUSLY_SEEN')]:
            if label not in r:continue
            short='test' if split=='test_sealed' else split;rows=[]
            for e in manifest()[split]:
                p=np.load(OUT/f'{r["name"]}_{short}_{e["image_id"]}.npz')['probabilities']
                assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                np.testing.assert_allclose(p.sum(0),1,atol=1e-5);rows.append((e,p));verified+=1
            got=metrics(rows,r['VAL']['background_logit_bias']);expected=r[label]
            np.testing.assert_array_equal(got['confusion_gt_rows_pred_cols'],expected['confusion_gt_rows_pred_cols'])
            for f in ['all_class_mIoU','foreground_mIoU','foreground_AP','pixel_accuracy','background_FPR']:
                np.testing.assert_allclose(got[f],expected[f],atol=1e-8)
            for c in NAMES:np.testing.assert_allclose(got['AP'][c],expected['AP'][c],atol=1e-8)
    pair={r['arm']:r for r in records if r['family']==family}
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for e in manifest()['test_sealed']:
        im=native_image(e);H,W=im['shape'];probs=[];preds=[]
        for arm in ARMS:
            r=pair[arm];p=np.load(OUT/f'{r["name"]}_test_{im["id"]}.npz')['probabilities']
            d=np.log(np.maximum(p,1e-12));d[3]+=r['VAL']['background_logit_bias']
            pred=d.argmax(0);probs.append(p);preds.append(pred)
            np.savez_compressed(OUT/f'{arm}_final_multiclass_{im["id"]}.npz',probabilities=p,prediction_internal=pred.astype('uint8'),prediction_public=np.array([1,2,3,0],dtype='uint8')[pred],class_order=np.array(NAMES))
        fig,axs=plt.subplots(4,4,figsize=(14,13))
        for c,name in enumerate(NAMES):
            axs[c,0].imshow(im['rgb']);axs[c,1].imshow(probs[0][c],vmin=0,vmax=1,cmap='magma')
            axs[c,2].imshow(probs[1][c],vmin=0,vmax=1,cmap='magma');axs[c,3].imshow(im['gt'][c],vmin=0,vmax=1,cmap='gray')
            for ax,title in zip(axs[c],[f'{name}: input','CSC + contextual head','SCSC + contextual head','GT (display only)']):ax.set_title(title,fontsize=9);ax.axis('off')
        fig.suptitle(f'ID {im["id"]} | {family} | probabilities 0–1 | previously-seen TEST')
        fig.tight_layout();fig.savefig(folder/f'activation_all_classes_{im["id"]}.png',dpi=120);plt.close(fig)
        fig,axs=plt.subplots(1,4,figsize=(16,4))
        images=[im['rgb'],COLORS[preds[0]],COLORS[preds[1]],COLORS[im['gt'].argmax(0)]]
        for ax,img,title in zip(axs,images,['Input','CSC + contextual head','SCSC + contextual head','GT']):
            ax.imshow(img);ax.set_title(title);ax.axis('off')
        fig.suptitle(f'ID {im["id"]} | Person red, Car blue, Motorcycle yellow, Background dark | ignored GT excluded from metrics')
        fig.tight_layout();fig.savefig(folder/f'multiclass_{im["id"]}.png',dpi=120);plt.close(fig)
    image_rows=[]
    for e in manifest()['test_sealed']:
        row=dict(image_id=e['image_id'])
        for arm,r in pair.items():
            p=np.load(OUT/f'{r["name"]}_test_{e["image_id"]}.npz')['probabilities']
            m=metrics([(e,p)],r['VAL']['background_logit_bias'])
            row.update({arm+'_'+k:m[k] for k in ['foreground_AP','foreground_mIoU','all_class_mIoU']})
        image_rows.append(row)
    bootstrap={};rng=np.random.default_rng(20261101)
    for field in ['foreground_AP','foreground_mIoU','all_class_mIoU']:
        d=np.array([r['SCSC_'+field]-r['CSC_'+field] for r in image_rows]);draws=rng.choice(d,size=(10000,len(d))).mean(1)
        bootstrap[field]=dict(mean_per_image_delta=float(d.mean()),SCSC_wins=int((d>0).sum()),images=len(d),CI95_mean_per_image_delta=np.quantile(draws,[.025,.975]).tolist(),note='Paired image bootstrap, mean per-image metric; distinct from pooled metric; only 5 previously-seen images')
    jwrite(OUT/'per_image_and_bootstrap.json',dict(images=image_rows,bootstrap=bootstrap))
    rows=[]
    for r in records:
        for split in ['VAL','TEST_PREVIOUSLY_SEEN']:
            m=r[split];rows.append(dict(name=r['name'],split=split,foreground_AP=m['foreground_AP'],foreground_mIoU=m['foreground_mIoU'],all_class_mIoU=m['all_class_mIoU'],pixel_accuracy=m['pixel_accuracy'],background_FPR=m['background_FPR'],**{f'AP_{c}':m['AP'][c] for c in NAMES}))
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    a=pair['CSC']['TEST_PREVIOUSLY_SEEN'];b=pair['SCSC']['TEST_PREVIOUSLY_SEEN']
    original={arm:read(ROOT/'final output/metrics'/f'{k}_test_sealed.json')['aggregate'] for arm,k in ARMS.items()}
    lines=['# ผลทดลองเร่งด่วน: contextual multiclass readout','',
      'งานนี้เป็นการฝึกตัวจำแนกใหม่บน dictionary เดิมที่ถูกตรึงไว้ ไม่ใช่การฝึก dictionary ใหม่ และไม่ใช่การทำซ้ำ paper แบบตรงตัว ไม่ได้เผยแพร่หรือ push การแก้ไขใด ๆ','',
      '## วิธีที่เปลี่ยน','',
      '- เพิ่ม TRAIN sampling สูงสุด 1,200 จุดต่อคลาสต่อภาพ ใช้จุดเดียวกันทั้ง CSC/SCSC รวม 10 ภาพ',
      '- ใช้ signed/absolute sparse codes และค่าเฉลี่ยขนาด 5/17/49/97 พิกเซล เพิ่มบริบทโดยไม่ resize ภาพ',
      '- ทดสอบเพิ่ม RGB/Lab, สีเฉลี่ยหลายขนาด และ local texture variance; มี RGB-only control',
      '- ใช้ HistGradientBoosting multiclass classifier แบบเดียวกัน: 120 iterations, 15 leaves, seed เดียวกัน',
      '- เลือก decision bias ของ Background บน VAL แยกต่อโมเดลด้วย grid เดียวกัน ไม่ใช้ TEST labels เลือกค่า',
      f'- เลือก feature family **{family}** จากค่าเฉลี่ย VAL all-class mIoU ของทั้งสอง arm ไม่เลือกจากขนาดช่องว่าง SCSC−CSC','',
      '## ผล previously-seen TEST 5 ภาพ','',
      '| Metric | CSC เดิม | SCSC เดิม | CSC + contextual | SCSC + contextual |',
      '|---|---:|---:|---:|---:|']
    for title,old,new in [('Foreground AP','macro_AP_foreground','foreground_AP'),('Foreground mIoU','foreground_mIoU','foreground_mIoU'),('All-class mIoU','all_class_mIoU','all_class_mIoU'),('Pixel accuracy','pixel_accuracy','pixel_accuracy'),('Background FPR (lower better)','background_FPR','background_FPR')]:
        lines.append(f'| {title} | {original["CSC"][old]:.6f} | {original["SCSC"][old]:.6f} | {a[new]:.6f} | {b[new]:.6f} |')
    lines+=['','| Class | CSC contextual AP | SCSC contextual AP | CSC contextual IoU | SCSC contextual IoU |','|---|---:|---:|---:|']
    for c in NAMES:lines.append(f'| {c} | {a["AP"][c]:.6f} | {b["AP"][c]:.6f} | {a["per_class"][c]["iou"]:.6f} | {b["per_class"][c]["iou"]:.6f} |')
    control=next(r for r in records if r['family']=='rgb_context_only')['TEST_PREVIOUSLY_SEEN']
    lines+=['','## การแปลผล','',
      f'SCSC−CSC contextual: foreground AP={b["foreground_AP"]-a["foreground_AP"]:+.6f}, foreground mIoU={b["foreground_mIoU"]-a["foreground_mIoU"]:+.6f}, all-class mIoU={b["all_class_mIoU"]-a["all_class_mIoU"]:+.6f}. ต้องแยกการดีขึ้นจากการเปลี่ยน classifier ออกจากการดีขึ้นจาก supervised dictionary ห้ามเทียบ SCSC รุ่นเพิ่มขั้นตอนกับ CSC รุ่นเดิมเพียงอย่างเดียวแล้วสรุปว่า supervision ชนะ',
      f'RGB-only control: foreground AP={control["foreground_AP"]:.6f}, all-class mIoU={control["all_class_mIoU"]:.6f}. ถ้าใกล้เคียงหรือดีกว่า ไม่ควรอ้างว่า sparse coding เป็นเหตุหลักของผลดีขึ้น','',
      '## ข้อจำกัด','',
      'ข้อมูลทั้งหมด 18 ภาพ แบ่ง TRAIN/VAL/TEST=10/3/5 ตามเดิม TEST เคยถูกดูมาก่อน จึงไม่ใช่ independent unseen evaluation; seed เดียวและยังไม่มีการยืนยันนัยสำคัญทางสถิติ ภาพยังเป็น native resolution แต่บริบทเฉลี่ยทำให้ขอบเบลอได้ การปรับ bias มีผลต่อ prediction แต่ไม่เปลี่ยน probability maps ที่แสดงหรือ AP ซึ่งคำนวณจาก raw classifier probabilities',
      'ไม่ได้ฝึก dictionary เพิ่ม ไม่ได้ทำ reconstruction/inpainting ใหม่ เพราะการแก้รอบนี้เปลี่ยนตัวจำแนกเท่านั้น ผลการทดลองที่ไม่ชนะยังเก็บครบใน validation.json/results.json','',
      '## ตรวจสอบและรันซ้ำ','',
      f'คำนวณ confusion/AP/IoU ซ้ำจากผลดิบ {verified} ภาพแล้วผ่าน ตรวจ probabilities อยู่ใน [0,1] และรวมเป็น 1 ต่อพิกเซล; ตรวจการกู้ sparse codes จาก saved logits เทียบ fresh encoding บน VAL หนึ่งภาพใน code_recovery_check.json',
      'ใช้ /workspace/.venvs/dataset-coco/bin/python rapid_context.py แล้ว report_rapid_context.py ใน repository เดิม Dependencies ใช้ requirements-experiment-lock.txt ผลอยู่ rapid_context_output/ ทั้งหมด ไม่มีการเขียนทับ final output/','',
      f'เวลาของ main experiment: {result["elapsed_s"]:.1f} วินาที (ไม่รวมการเขียนโค้ด/verification/figures). Protocol และ source/dictionary hashes อยู่ protocol.json']
    gamma_path=OUT/'gamma_reselection/results.json'
    if gamma_path.exists():
        gr=read(gamma_path);selected=gr['selected'];gamma=selected['gamma']
        for r in gr['all_VAL']:
            if r['gamma']==2:continue
            gamma_rows=[(e,np.load(OUT/f'gamma_reselection/g{r["gamma"]:g}_val_{e["image_id"]}.npz')['probabilities']) for e in manifest()['val']]
            got=metrics(gamma_rows,r['VAL']['background_logit_bias'])
            for field in ['foreground_AP','all_class_mIoU','foreground_mIoU']:np.testing.assert_allclose(got[field],r['VAL'][field],atol=1e-8)
            verified+=len(gamma_rows)
        gamma_rows=[(e,np.load(OUT/f'gamma_reselection/selected_test_{e["image_id"]}.npz')['probabilities']) for e in manifest()['test_sealed']]
        gm=metrics(gamma_rows,selected['VAL']['background_logit_bias']);verified+=len(gamma_rows)
        for field in ['foreground_AP','all_class_mIoU','foreground_mIoU']:np.testing.assert_allclose(gm[field],selected['TEST_PREVIOUSLY_SEEN'][field],atol=1e-8)
        baseline=next(r for r in result['records'] if r['name']=='rgb_sparse_context_CSC')['TEST_PREVIOUSLY_SEEN']
        lines+=['','## เพิ่มเติม: เลือก supervision ใหม่สำหรับ contextual readout','',
          'กำหนด candidates γ=0.2/2/10/100 และเลือกจาก SCSC VAL all-class mIoU เท่านั้น ไม่เลือกจากผล TEST หรือ SCSC−CSC เปรียบเทียบกับ CSC ที่ใช้ RGB+context และ classifier configuration เดียวกัน','',
          '| Gamma | VAL foreground AP | VAL all-class mIoU |','|---:|---:|---:|']
        for r in gr['all_VAL']:lines.append(f'| {r["gamma"]} | {r["VAL"]["foreground_AP"]:.6f} | {r["VAL"]["all_class_mIoU"]:.6f} |')
        lines+=['',f'VAL เลือก **γ={gamma}**. เป็นการเลือกใน workflow ใหม่ ไม่ได้เลือกจาก TEST ที่ดูดีกว่า','',
          '| Metric, previously-seen TEST | CSC + RGB/context | SCSC selected γ + RGB/context |','|---|---:|---:|']
        for field in ['foreground_AP','foreground_mIoU','all_class_mIoU','pixel_accuracy','background_FPR']:lines.append(f'| {field} | {baseline[field]:.6f} | {gm[field]:.6f} |')
        lines+=['','| Class | CSC AP | Selected SCSC AP | CSC IoU | Selected SCSC IoU |','|---|---:|---:|---:|---:|']
        for c in NAMES:lines.append(f'| {c} | {baseline["AP"][c]:.6f} | {gm["AP"][c]:.6f} | {baseline["per_class"][c]["iou"]:.6f} | {gm["per_class"][c]["iou"]:.6f} |')
        gamma_image_rows=[]
        for e,p in gamma_rows:
            im=native_image(e);basep=np.load(OUT/f'rgb_sparse_context_CSC_test_{im["id"]}.npz')['probabilities']
            dec=np.log(np.maximum(p,1e-12));dec[3]+=selected['VAL']['background_logit_bias'];pred=dec.argmax(0)
            bd=np.log(np.maximum(basep,1e-12));bd[3]+=next(r for r in result['records'] if r['name']=='rgb_sparse_context_CSC')['VAL']['background_logit_bias']
            pm=metrics([(e,p)],selected['VAL']['background_logit_bias']);bm=metrics([(e,basep)],0)
            gamma_image_rows.append(dict(image_id=im['id'],**{field:pm[field]-bm[field] for field in ['foreground_AP','foreground_mIoU','all_class_mIoU']}))
            np.savez_compressed(OUT/f'gamma_reselection/SCSC_final_multiclass_{im["id"]}.npz',probabilities=p,prediction_internal=pred.astype('uint8'),prediction_public=np.array([1,2,3,0],dtype='uint8')[pred],class_order=np.array(NAMES))
            fig,axs=plt.subplots(3,4,figsize=(14,10))
            for ax,img,title in zip(axs[0],[im['rgb'],COLORS[im['gt'].argmax(0)],COLORS[bd.argmax(0)],COLORS[pred]],['Input','GT','CSC contextual',f'SCSC contextual gamma={gamma}']):ax.imshow(img);ax.set_title(title);ax.axis('off')
            for c,name in enumerate(NAMES):
                for row,proba,arm in [(1,basep,'CSC'),(2,p,'SCSC')]:
                    axs[row,c].imshow(proba[c],vmin=0,vmax=1,cmap='magma');axs[row,c].set_title(arm+' '+name);axs[row,c].axis('off')
            fig.suptitle(f'VAL-selected gamma={gamma} | ID {im["id"]} | native resolution | probabilities 0–1 | previously-seen TEST')
            fig.tight_layout();fig.savefig(folder/f'gamma_selected_activation_{im["id"]}.png',dpi=120);plt.close(fig)
        gamma_bootstrap={};rng=np.random.default_rng(20261101)
        for field in ['foreground_AP','foreground_mIoU','all_class_mIoU']:
            d=np.array([r[field] for r in gamma_image_rows]);draws=rng.choice(d,size=(10000,len(d))).mean(1)
            gamma_bootstrap[field]=dict(mean_per_image_delta=float(d.mean()),SCSC_wins=int((d>0).sum()),images=len(d),CI95_mean_per_image_delta=np.quantile(draws,[.025,.975]).tolist(),note='Mean per-image metric, distinct from pooled metric; only five previously-seen images')
        jwrite(OUT/'gamma_reselection/per_image_and_bootstrap.json',dict(images=gamma_image_rows,bootstrap=gamma_bootstrap))
        lines+=['','Paired image bootstrap (10,000 resamples) ของค่าเฉลี่ย metric ต่อภาพ อยู่ใน gamma_reselection/per_image_and_bootstrap.json เป็นคนละนิยามกับ pooled AP/IoU ในตารางและใช้ภาพเพียง 5 ภาพ จึงไม่อ้างว่ายืนยัน generalization ได้']
    spatial_path=OUT/'spatial_refinement/results.json'
    if spatial_path.exists():
        sr=read(spatial_path);sigma=sr['sigma']
        for arm,record in sr['arms'].items():
            for split,label in [('val','VAL'),('test_sealed','TEST_PREVIOUSLY_SEEN')]:
                short='test' if split=='test_sealed' else split
                rows=[]
                from rapid_spatial import load_p
                for e in manifest()[split]:
                    saved=np.load(OUT/f'spatial_refinement/{arm}_{short}_{e["image_id"]}.npz');p=saved['probabilities']
                    original=load_p(arm,e,short,sr['gamma'])
                    expected=gaussian_filter(original,sigma=(0,sigma,sigma),mode='reflect') if sigma else original.copy()
                    expected/=expected.sum(0,keepdims=True)
                    np.testing.assert_allclose(p,expected,atol=1e-7)
                    if short=='test':
                        d=np.log(np.maximum(p,1e-12));d[3]+=record['bias']
                        np.testing.assert_array_equal(d.argmax(0),saved['prediction_internal'])
                    rows.append((e,p))
                got=metrics(rows,record['bias']);verified+=len(rows)
                for field in ['foreground_AP','foreground_mIoU','all_class_mIoU','pixel_accuracy','background_FPR']:np.testing.assert_allclose(got[field],record[label][field],atol=1e-8)
        lines+=['','## เพิ่มเติม: Spatial consistency refinement','',
          f'ใช้ Gaussian smoothing ของ probability maps โดย VAL เลือก sigma={sigma} พิกเซลจาก [0,3,7,15,31] โดยเลือกค่าเฉลี่ย all-class mIoU ของ CSC/SCSC ไม่เลือกช่องว่างระหว่างโมเดล ใช้ sigma เดียวกันทั้งสอง arm และปรับ Background decision bias ด้วย VAL grid เดียวกัน การปรับนี้เป็น postprocessing ที่ระบุชัด ไม่ใช่ raw activation และไม่ได้ใช้ GT ระหว่าง inference','',
          '| Metric, previously-seen TEST | CSC contextual + refinement | SCSC contextual + refinement | RGB-only + refinement |','|---|---:|---:|---:|']
        for field in ['foreground_AP','foreground_mIoU','all_class_mIoU','pixel_accuracy','background_FPR']:
            vals=[sr['arms'][arm]['TEST_PREVIOUSLY_SEEN'][field] for arm in ['CSC','SCSC','RGB_only']]
            lines.append('| '+field+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
        lines+=['','| Class | Refined CSC AP | Refined SCSC AP | Refined CSC IoU | Refined SCSC IoU |','|---|---:|---:|---:|---:|']
        a=sr['arms']['CSC']['TEST_PREVIOUSLY_SEEN'];b=sr['arms']['SCSC']['TEST_PREVIOUSLY_SEEN']
        for c in NAMES:lines.append(f'| {c} | {a["AP"][c]:.6f} | {b["AP"][c]:.6f} | {a["per_class"][c]["iou"]:.6f} | {b["per_class"][c]["iou"]:.6f} |')
        spatial_deltas=[]
        for e in manifest()['test_sealed']:
            im=native_image(e);arrays={arm:np.load(OUT/f'spatial_refinement/{arm}_test_{im["id"]}.npz') for arm in ['CSC','SCSC']}
            delta=dict(image_id=im['id']);per={arm:metrics([(e,arr['probabilities'])],sr['arms'][arm]['bias']) for arm,arr in arrays.items()}
            delta.update({field:per['SCSC'][field]-per['CSC'][field] for field in ['foreground_AP','foreground_mIoU','all_class_mIoU']});spatial_deltas.append(delta)
            fig,axs=plt.subplots(3,4,figsize=(14,10))
            for ax,img,title in zip(axs[0],[im['rgb'],COLORS[im['gt'].argmax(0)],COLORS[arrays['CSC']['prediction_internal']],COLORS[arrays['SCSC']['prediction_internal']]],['Input','GT','Refined CSC','Refined SCSC']):ax.imshow(img);ax.set_title(title);ax.axis('off')
            for row,arm in [(1,'CSC'),(2,'SCSC')]:
                for c,name in enumerate(NAMES):axs[row,c].imshow(arrays[arm]['probabilities'][c],vmin=0,vmax=1,cmap='magma');axs[row,c].set_title(arm+' '+name);axs[row,c].axis('off')
            fig.suptitle(f'Spatially refined probabilities (not raw), sigma={sigma}, gamma={sr["gamma"]} | ID {im["id"]} | previously-seen TEST')
            fig.tight_layout();fig.savefig(folder/f'spatially_refined_activation_{im["id"]}.png',dpi=120);plt.close(fig)
        spatial_bootstrap={};rng=np.random.default_rng(20261101)
        for field in ['foreground_AP','foreground_mIoU','all_class_mIoU']:
            d=np.array([r[field] for r in spatial_deltas]);draws=rng.choice(d,size=(10000,len(d))).mean(1)
            spatial_bootstrap[field]=dict(mean_per_image_delta=float(d.mean()),SCSC_wins=int((d>0).sum()),images=len(d),CI95_mean_per_image_delta=np.quantile(draws,[.025,.975]).tolist())
        jwrite(OUT/'spatial_refinement/per_image_and_bootstrap.json',dict(images=spatial_deltas,bootstrap=spatial_bootstrap,note='Mean per-image metric, not pooled AP/IoU; five previously-seen images only'))
        old_csc=read(ROOT/'final output/metrics/K4_g0_N10_test_sealed.json')['aggregate']
        old_scsc=read(ROOT/'final output/metrics/K4_g2_N10_test_sealed.json')['aggregate']
        summary=['## ผลสรุปรอบปรับปรุง','',
          f'Workflow ที่ปรับแล้วใช้ frozen SCSC dictionary γ={sr["gamma"]} + contextual multiclass classifier ที่ฝึกใหม่ + spatial refinement σ={sigma} ซึ่งเลือกด้วย VAL ทั้งหมด ไม่ได้ฝึก dictionary ใหม่ ผลต่อไปนี้เทียบกับ CSC ที่ได้ขั้นตอนใหม่เหมือนกัน','',
          '| Workflow | TEST foreground AP | TEST all-class mIoU |','|---|---:|---:|',
          f'| CSC เดิม | {old_csc["macro_AP_foreground"]:.6f} | {old_csc["all_class_mIoU"]:.6f} |',
          f'| SCSC เดิม | {old_scsc["macro_AP_foreground"]:.6f} | {old_scsc["all_class_mIoU"]:.6f} |',
          f'| CSC + contextual classifier + refinement | {a["foreground_AP"]:.6f} | {a["all_class_mIoU"]:.6f} |',
          f'| SCSC + contextual classifier + refinement | {b["foreground_AP"]:.6f} | {b["all_class_mIoU"]:.6f} |','',
          'ใน matched comparison นี้ SCSC มี pooled AP และ IoU สูงกว่า CSC ทั้งสี่คลาส แต่ยังมีข้อจำกัดสำคัญ: ภาพทำนายยังหยาบและไม่ตรงรูปร่างวัตถุดีพอ โดยเฉพาะ Person มี AP ต่ำทั้งคู่ และ Person AP ของ workflow ใหม่น้อยกว่าตัวจำแนกเดิม จึงห้ามสรุปว่าทุกด้านดีขึ้นจากรุ่นเดิม','',
          f'ชนะด้าน per-image all-class mIoU {spatial_bootstrap["all_class_mIoU"]["SCSC_wins"]}/5 ภาพ; paired bootstrap 95% CI ของ mean per-image mIoU delta = {spatial_bootstrap["all_class_mIoU"]["CI95_mean_per_image_delta"]}. ช่วงคร่อมศูนย์ จึงยังไม่ยืนยันความเหนือกว่าเชิงสถิติ TEST ทั้งห้าภาพเคยถูกดูมาก่อน ไม่ใช่ independent unseen test','',
          'ไม่มีการ commit/push/publish งานรอบนี้ ผลเดิมไม่ถูกแก้ไข รายงานและ ZIP เป็นงานในฉบับร่างเท่านั้น','']
        lines=lines[:2]+summary+lines[2:]
    (OUT/'REPORT_TH.md').write_text('\n'.join(lines)+'\n')
    verification=dict(passed=True,raw_predictions_recomputed=verified,selected_family=family,source_sha256=sha(ROOT/'rapid_context.py'),report_source_sha256=sha(__file__),published=False,
        TRAIN_sampling_recomputed=True,frozen_dictionary_hashes_matched=True,fresh_encoding=fresh_encoding,
        nonconverged_fresh_encodings=[r for r in fresh_encoding if r['converged_batches']!=r['batch_count']])
    jwrite(OUT/'verification.json',verification)
    entries=[p for p in OUT.rglob('*') if p.is_file() and 'codes' not in p.relative_to(OUT).parts and p.suffix!='.log' and p.name!='SHA256SUMS.json']
    entries += [ROOT/name for name in ['rapid_context.py','rapid_gamma.py','rapid_spatial.py','report_rapid_context.py','requirements-experiment-lock.txt','native_final.py','coco_scsc_manifest.json']]
    entries.extend((ROOT/'paper_scsc').glob('*.py'))
    for folder in ['data','references']:entries.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for k in [*ARMS.values(),'K4_g0.2_N10','K4_g10_N10','K4_g100_N10']:entries.extend([ROOT/'final output/checkpoints'/k/'model.npz',ROOT/'final output/checkpoints'/k/'config.json'])
    for k in ARMS.values():entries.append(ROOT/f'final output/metrics/{k}_test_sealed.json')
    inventory={str(p.relative_to(ROOT)):sha(p) for p in entries};jwrite(OUT/'SHA256SUMS.json',inventory);entries.append(OUT/'SHA256SUMS.json')
    archive=ROOT.parent/'SCSC_COCO_CONTEXTUAL_30MIN_RESULTS.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in entries:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(verification=verification,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),indent=2))

if __name__=='__main__':main()
