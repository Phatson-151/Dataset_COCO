"""Read-only COCO manifest, native instance masks, and split verification.
No resizing/downsampling is implemented in this module.
"""
"""Read and audit the original 18-image COCO subset; no split leakage."""
import json,hashlib
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from scipy.fft import rfft2,irfft2
ROOT=Path(__file__).resolve().parents[1]
IDS=(1,3,4)
NAMES=('Person','Car','Motorcycle')

def _sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

@lru_cache(None)
def manifest():
    j=json.loads((ROOT/'coco_scsc_manifest.json').read_text())
    ids=[{int(t['image_id']) for t in j[s]} for s in ('train','val','test_sealed')]
    assert len(ids[0])==10 and len(ids[1])==3 and len(ids[2])==5
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    return j

@lru_cache(None)
def annotations():
    j=json.loads((ROOT/'data/coco/annotations_18img_subset.json').read_text())
    anns={}
    for a in j['annotations']: anns.setdefault(int(a['image_id']),[]).append(a)
    return anns

def compressed_rle_counts(s):
    # Reference COCO maskApi.c compressed run-count coding, pure Python fallback.
    p=0;m=0;counts=[]
    while p<len(s):
        x=0;k=0;more=1
        while more:
            c=ord(s[p])-48;p+=1
            x|=(c&0x1f)<<(5*k)
            more=c&0x20
            if not more and (c&0x10): x|=(-1)<<(5*(k+1))
            k+=1
        if m>2: x+=counts[m-2]
        counts.append(x);m+=1
    return counts

def mask_of(a, h, w):
    # COCO's own rasterizer is required: rounding polygon vertices and fillPoly
    # changes boundary pixels and consequently all pixel-level metrics.
    from pycocotools import mask as coco_mask
    seg=a.get('segmentation', [])
    if isinstance(seg,list):
        if not seg:return np.zeros((h,w),np.uint8)
        rle=coco_mask.merge(coco_mask.frPyObjects(seg,h,w))
    elif isinstance(seg,dict):
        rle=coco_mask.frPyObjects(seg,h,w) if isinstance(seg['counts'],list) else dict(seg)
        if isinstance(rle['counts'],str):rle['counts']=rle['counts'].encode('ascii')
    else:raise TypeError('Unsupported COCO segmentation')
    mask=coco_mask.decode(rle)
    if mask.shape!=(h,w):raise ValueError('COCO mask shape mismatch; resizing prohibited')
    return mask

def legacy_mask_of(a, h, w):
    """Retained for audit only; experiments use the official COCO rasterizer."""
    seg=a.get('segmentation',[])
    if isinstance(seg,list):
        mask=np.zeros((h,w),np.uint8)
        for poly in seg:
            pts=np.asarray(poly,np.float64).reshape(-1,2)
            if len(pts)>=3:
                pts=np.rint(pts).astype(np.int32)
                cv2.fillPoly(mask,[pts],1)
        return mask
    if isinstance(seg,dict):
        cnts=seg['counts']
        if isinstance(cnts,str): cnts=compressed_rle_counts(cnts)
        sh,sw=seg['size']
        flat=np.zeros(sh*sw,np.uint8);pos=0;val=0
        for n in cnts:
            if n<0 or pos+n>len(flat): raise ValueError('Invalid RLE')
            if val:flat[pos:pos+n]=1
            pos+=n;val^=1
        if pos!=len(flat):raise ValueError('Incomplete RLE')
        mask=flat.reshape((sh,sw),order='F')
        if (sh,sw)!=(h,w):raise ValueError('COCO RLE shape mismatch; resizing prohibited')
        return mask
    raise TypeError('Unsupported COCO segmentation')

