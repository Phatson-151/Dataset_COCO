"""Render sparse multiclass activation points from actual model predictions.
No ground truth is used to select, suppress, or change the predicted points.
"""
import json
import numpy as np
from native_final import ROOT,manifest,native_image,NAMES,COLORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=ROOT/'final output';report=json.loads((p/'FINAL_METRICS.json').read_text())
K=report['selected_K_by_VAL_comparative_mIoU_and_convergence']
for split in ('val','test_sealed'):
 f='validation_selected.json' if split=='val' else 'test_previously_seen.json'
 metrics=json.loads((p/'metrics'/f).read_text())
 for e in manifest()[split]:
  im=native_image(e);H,W=im['shape']
  fig,axs=plt.subplots(1,4,figsize=(20,5.5))
  axs[0].imshow(im['rgb']);axs[0].set_title(f'Original COCO {im["id"]}: {W} x {H} native pixels')
  rgb=im['rgb'].astype('float32')/255;show=rgb.copy()
  truth=np.argmax(im['gt'],axis=0)
  for c in range(3):show[truth==c]=.6*show[truth==c]+.4*COLORS[c]
  axs[1].imshow(show);axs[1].set_title('GT for visual checking (not used in prediction)')
  for col,arm in [(2,'CSC'),(3,'SCSC')]:
   axs[col].imshow(im['rgb']);axs[col].set_title(f'{arm}: top native-grid activation peaks')
   entry=next(x for x in metrics[f'K{K}_{arm}']['per_image'] if x['image_id']==im['id'])
   sc=np.load(p/'score_maps'/f'{split}_K{K}_{arm}_{im["id"]}.npz')['scores']
   for c,name in enumerate(NAMES[:3]):
    pos=entry['points'][name]['positions_xy']
    if not len(pos):continue
    xy=np.asarray(pos)
    strengths=sc[c,xy[:,1],xy[:,0]]
    sel=np.argsort(strengths)[-35:]
    chosen=xy[sel]
    axs[col].scatter(chosen[:,0],chosen[:,1],s=22,facecolors='none',edgecolors=COLORS[c],linewidth=1.2,label=name)
   axs[col].legend(loc='lower left',fontsize=8,facecolor='white')
  for ax in axs:ax.axis('off')
  fig.suptitle('Red Person / Blue Car / Yellow Motorcycle | Predicted peaks selected WITHOUT GT; false activations not removed')
  fig.tight_layout()
  fig.savefig(p/'figures'/f'{split}_Activation_Points_4Panel_{im["id"]}.png',dpi=135,bbox_inches='tight')
  plt.close(fig)
print('Created original-size point-only comparison figures',flush=True)
