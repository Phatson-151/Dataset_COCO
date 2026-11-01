"""Native inference, official annotations and stored-release consistency."""
import json
import unittest
from pathlib import Path
import numpy as np
from pycocotools import mask as coco_mask
from paper_scsc.data import ROOT,mask_of
from paper_scsc.solver import init_dict,solve_z,solve_d,conv,kernel_fft,make_mask
from paper_scsc.experiments import infer,BASE,tag,from_cm


class ExperimentProtocolTests(unittest.TestCase):
    def test_official_polygon_and_rle_decoding_agree(self):
        poly=[[1.2,1.4,8.7,1.6,8.1,7.9,2.1,8.4]]
        rle=coco_mask.merge(coco_mask.frPyObjects(poly,12,13))
        ref=coco_mask.decode(rle)
        np.testing.assert_array_equal(mask_of({'segmentation':poly},12,13),ref)
        np.testing.assert_array_equal(mask_of({'segmentation':rle},12,13),ref)

    def test_inference_covers_non_multiple_native_shape(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(67,79)).astype('float32')*.02
        D=init_dict(x[None],2,5)
        cfg=dict(BASE,tile=32,stride=20,eval_admm=200)
        scores,rec,diag=infer(x,D,np.zeros((4,2),'float32'),np.arange(4,dtype='float32'),cfg)
        self.assertEqual(scores.shape,(4,67,79));self.assertEqual(rec.shape,x.shape)
        self.assertGreater(diag['min_weight'],0)
        self.assertTrue(np.isfinite(rec).all())
        np.testing.assert_allclose(scores,np.broadcast_to(np.arange(4)[:,None,None],scores.shape),atol=2e-6)

    def test_multiclass_metrics_against_hand_computed_confusion(self):
        cm=np.array([[3,1,0,0],[0,2,1,1],[0,0,2,0],[1,1,0,4]])
        m=from_cm(cm)
        self.assertAlmostEqual(m['per_class']['Person']['iou'],3/5)
        self.assertAlmostEqual(m['per_class']['Car']['f1'],4/8)
        self.assertAlmostEqual(m['background_FPR'],2/6)
        self.assertAlmostEqual(m['pixel_accuracy'],11/16)

    def test_low_rank_high_K_dictionary_update_never_returns_worse_point(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(1,16,16)).astype('float32')*.1
        d=init_dict(x,12,3);w=rng.normal(size=(4,12)).astype('float32')*.1
        z,_=solve_z(d,x,w,np.zeros(4,'float32'),np.ones((1,4,16,16),'int8'),.1,.01,maxiter=1000,adaptive=True)
        new,diag=solve_d(x,z,d,maxiter=1000,adaptive=True)
        m=make_mask((16,16),3)
        cost=lambda D:np.sum((m*(conv(kernel_fft(D,(16,16)),z)-x))**2)
        self.assertLessEqual(cost(new),cost(d)+1e-6)
        self.assertLessEqual(np.linalg.norm(new,axis=(1,2)).max(),1.00001)
        if diag['descent_guard_rejected']:self.assertFalse(diag['converged'])


class NewReleaseArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=ROOT/'final output'
        p=cls.out/'experiment_configs/selection.json'
        if not p.exists():raise unittest.SkipTest('New experiment sweep not complete')
        cls.selection=json.loads(p.read_text())

    def test_all_completed_candidates_have_ten_checkpoints(self):
        for p in (self.out/'checkpoints').glob('*/completed.json'):
            self.assertEqual(len(list(p.parent.glob('outer_*.npz'))),10)
            h=json.loads((p.parent/'history.json').read_text())
            self.assertEqual(len(h),10)
            for row in h:
                for name in ['z_admm','d_admm']:
                    diag=row[name];last=diag['residuals'][-1]
                    if diag['converged']:
                        self.assertLessEqual(last['primal_residual'],last['primal_tolerance'])
                        self.assertLessEqual(last['dual_residual'],last['dual_tolerance'])

    def test_inpainting_masks_are_paired(self):
        from paper_scsc.data import manifest
        a,b=self.selection['paired_models']
        for e in manifest()['test_sealed']:
            masks=[]
            for cfg in [a,b]:
                k=tag(cfg['K'],cfg['gamma'],cfg['N'])
                p=self.out/'inpainting'/f'{k}_{e["image_id"]}.npz'
                self.assertTrue(p.exists())
                masks.append(np.load(p)['observed'])
            np.testing.assert_array_equal(*masks)
            self.assertEqual(np.count_nonzero(~masks[0]),masks[0].size//2)

    def test_native_shapes_class_ids_and_argmax_match(self):
        from paper_scsc.data import manifest
        for cfg in self.selection['paired_models']:
            k=tag(cfg['K'],cfg['gamma'],cfg['N'])
            for split in ['val','test_sealed']:
                for e in manifest()[split]:
                    a=np.load(self.out/'score_maps'/f'{k}_{split}_{e["image_id"]}.npz')
                    self.assertEqual(a['logits'].shape,(4,e['height'],e['width']))
                    p=a['logits'].argmax(0)
                    np.testing.assert_array_equal(a['prediction_internal'],p)
                    np.testing.assert_array_equal(a['prediction'],np.array([1,2,3,0])[p])
