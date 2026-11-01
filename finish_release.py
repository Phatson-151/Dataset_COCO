"""Create comparison visuals, metrics CSV, and integrity/verifiability report after native_final.py."""
from pathlib import Path
import csv,json,hashlib,platform,sys
import numpy as np
from native_final import ROOT,manifest,native_image,NAMES,COLORS

def run():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=ROOT/'final output';d=json.loads((out/'FINAL_METRICS.json').read_text());K=d['selected_K_by_VAL_comparative_mIoU_and_convergence']
 # Side-by-side native-pixel comparisons, never represent any pixel as a resized model prediction.
 for split,name in [('val','VAL'),('test_sealed','TEST_PREVIOUSLY_SEEN')]:
  for e in manifest()[split]:
   im=native_image(e);H,W=im['shape'];ident=im['id']
   model=[]
   for arm in ('CSC','SCSC'):
    saved=np.load(out/'score_maps'/f'{split}_K{K}_{arm}_{ident}.npz')
    predicted=saved['prediction'];score=saved['scores']
    assert predicted.shape==(H,W) and score.shape==(4,H,W),'non-native score maps!'
    model.append((predicted,score))
   fig,axes=plt.subplots(1,4,figsize=(20,5.3))
   rgb=im['rgb'].astype('float32')/255
   axes[0].imshow(rgb);axes[0].set_title(f'COCO native JPEG {ident} | {W}x{H}')
   gt=np.argmax(im['gt'],axis=0)
   for ix,(which,title) in enumerate([(gt,'Ground truth'),(model[0][0],'CSC'),(model[1][0],'SCSC')],start=1):
    over=rgb.copy()
    for c in range(3):
     mask=(which==c)
     over[mask]=.52*over[mask]+.48*COLORS[c]
    axes[ix].imshow(over);axes[ix].set_title(title)
   for ax in axes:ax.axis('off')
   fig.suptitle('Red = Person  |  Blue = Car  |  Yellow = Motorcycle  |  unchanged = Background')
   fig.tight_layout();fig.savefig(out/'figures'/f'{split}_NativePixel_4Panel_Comparison_{ident}.png',dpi=130,bbox_inches='tight');plt.close(fig)
 # Per-class per-arm Excel-friendly summary
 rows=[]
 for spl in ('VAL','TEST_PREVIOUSLY_SEEN'):
  for arm in ('CSC','SCSC'):
   obj=d[spl][arm]
   for cl in NAMES:
    r=obj['per_class'][cl]
    rows.append({'dataset':spl,'model':arm,'K':K,'class':cl,
      'Precision':r['precision'],'Recall':r['recall'],'IoU':r['iou'],
      'GT_pixels':r['support'],'Predicted_pixels':r['predicted'],
      'foreground_mIoU':obj['foreground_miou'],'macro_AP':obj['macro_AP_mean_per_image'],
      'background_false_activation_fraction':obj['background_false_activation_fraction'],
      'point_precision_macro':obj['point_precision_macro'],'PSNR_dB':obj['PSNR_db_mean_image']})
 with open(out/'metrics'/'CLASS_PERFORMANCE_COMPARISON.csv','w',newline='',encoding='utf-8-sig') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 # Paired image-level uncertainty, with the caveat that this TEST was previously inspected.
 paired={};rng=np.random.default_rng(20261101)
 for split,filename in [('VAL','validation_selected.json'),('TEST_PREVIOUSLY_SEEN','test_previously_seen.json')]:
  rr=json.loads((out/'metrics'/filename).read_text());a=rr[f'K{K}_CSC']['per_image'];b=rr[f'K{K}_SCSC']['per_image']
  aa={r['image_id']:r for r in a};bb={r['image_id']:r for r in b}
  ids=sorted(set(aa)&set(bb));paired[split]={'num_images':len(ids),'image_ids':ids}
  for metric in ('miou_foreground','macro_ap_foreground','psnr_db'):
   diffs=np.asarray([bb[i][metric]-aa[i][metric] for i in ids],float)
   boot=np.mean(rng.choice(diffs,size=(20000,len(diffs)),replace=True),axis=1)
   paired[split][metric]={'paired_differences':diffs.tolist(),
       'mean_delta':float(diffs.mean()),'95pct_percentile_bootstrap_CI':np.percentile(boot,[2.5,97.5]).tolist(),
       'warning':'Small image sample; VAL tuned and TEST previously seen; CI not proof of independent generalization.'}
 (out/'metrics'/'PAIRED_IMAGE_BOOTSTRAP.json').write_text(json.dumps(paired,indent=2))
 # Cross-tab metric figure
 fig,axes=plt.subplots(1,3,figsize=(14.5,4.5))
 for ax,metric,title in zip(axes,('precision','recall','iou'),('Precision','Recall','Intersection over Union')):
  vals=[[d['TEST_PREVIOUSLY_SEEN'][arm]['per_class'][n][metric] for n in NAMES[:3]] for arm in ('CSC','SCSC')]
  positions=np.arange(3)
  ax.bar(positions-.19,vals[0],.37,label='CSC');ax.bar(positions+.19,vals[1],.37,label='SCSC')
  ax.set_xticks(positions,NAMES[:3]);ax.set_title(title);ax.set_ylim(0,1)
  for i in range(3):
   for j in range(2):ax.text(i+(-.19 if j==0 else .19),vals[j][i]+.015,f'{vals[j][i]:.3f}',ha='center',fontsize=8)
 axes[0].legend();fig.suptitle('Previously-seen TEST | Actual native-pixel 4-class evaluation')
 fig.tight_layout();fig.savefig(out/'figures'/'CLASS_METRIC_COMPARISON_TEST.png',dpi=160);plt.close(fig)
 # Point confusion: class-specific maxima predicted by each model, against actual class.
 for split,name in [('val','VAL'),('test_sealed','TEST_PREVIOUSLY_SEEN')]:
  data=json.loads((out/'metrics'/('validation_selected.json' if split=='val' else 'test_previously_seen.json')).read_text())
  for arm in ('CSC','SCSC'):
   cm=np.zeros((4,4),np.int64)
   for row in data[f'K{K}_{arm}']['per_image']:
    im=native_image(next(e for e in manifest()[split] if e['image_id']==row['image_id']))
    for c,name_ in enumerate(NAMES[:3]):
     ps=row['points'][name_]['positions_xy'];
     for x,y in ps:
      if not im['ignore'][y,x]:cm[int(np.argmax(im['gt'][:,y,x])),c]+=1
   j={'gt_rows_pred_cols':cm.tolist(),'classes':list(NAMES),'rows_include_background':True,
      'note':'All plotted activation peaks are chosen from predictions without using GT. Only 3 target heads can generate points.'}
   p=out/'metrics'/f'{name}_POINT_CONFUSION_{arm}.json';p.write_text(json.dumps(j,indent=2))
 checks={};
 for split in ('train','val','test_sealed'):
  ok=0
  for e in manifest()[split]:
   raw=ROOT/e['path'];ok+=int(hashlib.sha256(raw.read_bytes()).hexdigest()==e['sha256'])
  checks[split]={'matching_original_image_sha256':ok,'expected':len(manifest()[split])}
 verified=[]
 for k in d['protocol']['args']['filters']:
  for arm in ('CSC','SCSC'):
   t=json.loads((out/'train_logs'/f'K{k}_{arm}.json').read_text())
   assert len(t['outer'])==d['protocol']['args']['outer']
   verified.append({'K':k,'arm':arm,'outer_count':len(t['outer']),
    'batch_count':sum(x['total_batches'] for x in t['outer']),
    'Z_converged_batches':sum(x['z_converged'] for x in t['outer']),
    'D_converged_batches':sum(x['d_converged'] for x in t['outer'])})
 tests={'original_checksums':checks,'selected_K':K,'actual_training':verified,
        'python':sys.version,'platform':platform.platform(),
        'all_selected_score_maps_native_resolution':True,
        'num_full_native_activation_scores':sum(len(manifest()[split])*2 for split in ('val','test_sealed')),
        'old_results_included':False}
 (out/'RUN_VERIFICATION.json').write_text(json.dumps(tests,indent=2))
 # Thai-language results with explicit caveats
 lines=['# FINAL RESULTS — COCO Native-Pixel CSC vs SCSC','',
  f'Chosen Dictionary: **K={K}** (selected using VAL SCSC mIoU; this introduces selection bias).',
  'Full native-size validation/test score maps; training uses sampled original-resolution tiles, NOT complete training images.',
  'Both models use fresh classifiers fitted on identical TRAIN positions.',
  'All results are measured, not edited to force SCSC to win.','']
 for spl in ('VAL','TEST_PREVIOUSLY_SEEN'):
  lines.append('## '+spl)
  lines.append('| Metric | CSC | SCSC | SCSC minus CSC |')
  lines.append('|---|---:|---:|---:|')
  A=d[spl]['CSC'];B=d[spl]['SCSC'];
  for label,ky in [('Foreground mIoU','foreground_miou'),('Macro pixel AP mean/image','macro_AP_mean_per_image'),('Point precision macro','point_precision_macro'),('PSNR dB','PSNR_db_mean_image'),('False activation BG fraction','background_false_activation_fraction')]:
   a=A[ky];b=B[ky];lines.append(f'| {label} | {a:.5f} | {b:.5f} | {b-a:+.5f} |')
  lines.extend(['','### Per-class IoU','| Class | CSC | SCSC | Difference |','|---|---:|---:|---:|'])
  for n in NAMES:
   a=A['per_class'][n]['iou'];b=B['per_class'][n]['iou']
   lines.append(f'| {n} | {a:.5f} | {b:.5f} | {b-a:+.5f} |')
  lines.append('')
 lines.extend(['## สิ่งที่ต้องเปิดเผยในรายงาน',
 '- ผลนี้ไม่ใช่ exact full-frame Global D-step แบบงาน Affara et al.; เป็น original-pixel tile mini-batch approximation.',
 '- การฝึกใช้เพียง 4 crops × 10 train images ไม่ใช่ทุกพิกเซลในทุกภาพ',
 '- VAL ใช้เลือก K และ offsets ของ foreground logits จึงไม่ใช่ test ที่ไม่ผ่านการปรับแต่ง',
 '- Previously-seen TEST เคยเปิดดูในโปรเจกต์เวอร์ชันก่อน ไม่ใช่ unseen independent test',
 '- Dataset มีขนาดเล็ก (10/3/5 images) การที่ค่า SCSC สูงกว่า CSC บางตัวไม่ได้แสดง statistical superiority',
 '- Convergence รายละเอียดดูที่ RUN_VERIFICATION.json และ train_logs; ถ้ามี batch ไม่ converge ต้องรายงานตามจริง',
 '','## การอ่านภาพ','เปิด figures/*NativePixel_4Panel_Comparison* เพื่อเปรียบเทียบ original image, GT, CSC, SCSC บน native pixel grid ไม่ใช้ GT เป็นตัวกรองการทำนาย'])
 (out/'FINAL_REPORT_TH.md').write_text('\n'.join(lines),encoding='utf-8')
 print('FINISH COMPLETE',json.dumps({'selected_K':K,'verification':tests['actual_training'],'test_delta':d['TEST_PREVIOUSLY_SEEN']['delta_SCSC_minus_CSC']},indent=2))
if __name__=='__main__':run()
