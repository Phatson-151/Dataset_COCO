"""Plot actual matched-workflow metrics; native maps are separate artifacts."""
import os,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/scsc-context-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
r=json.loads((root/'spatial_refinement/results.json').read_text())
names=['Person','Car','Motorcycle','Background']
fig,axs=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(4)
for ax,metric in zip(axs,['AP','IoU']):
    for offset,arm,color in [(-.18,'CSC','#577590'),(.18,'SCSC','#f8961e')]:
        m=r['arms'][arm]['TEST_PREVIOUSLY_SEEN']
        values=[m['AP'][n] if metric=='AP' else m['per_class'][n]['iou'] for n in names]
        bars=ax.bar(x+offset,values,.36,label=arm+' + context/refinement',color=color)
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.set_xticks(x,names);ax.set_ylim(0,1);ax.set_ylabel(metric);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle(f'Matched modified workflows | gamma={r["gamma"]}, sigma={r["sigma"]} | five previously-seen TEST images\nVAL-selected settings; not independent generalization evidence',fontsize=10)
fig.tight_layout();fig.savefig(root/'figures/final_metric_comparison.png',dpi=150)
