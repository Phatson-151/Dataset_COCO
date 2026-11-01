import unittest
import numpy as np
from native_final import native_image,choose_tiles,tile_signal,sample_targets,coverage_positions,postprocess,evaluate_one,point_metrics
from paper_scsc.data import manifest
from paper_scsc.solver import init_dict,solve_z,solve_d,make_mask

class NativePipelineTests(unittest.TestCase):
 def test_original_dimensions(self):
  e=manifest()['train'][0];m=native_image(e)
  self.assertEqual(m['rgb'].shape[:2],(e['height'],e['width']))
  self.assertEqual(m['x'].shape,(e['height'],e['width']))
 def test_no_split_leakage(self):
  ms=manifest();splits=[set(x['image_id'] for x in ms[n]) for n in ('train','val','test_sealed')]
  self.assertFalse(splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2])
 def test_four_exclusive_classes(self):
  m=native_image(manifest()['train'][1]);self.assertTrue(np.all(m['gt'].sum(axis=0)[~m['ignore']]==1))
 def test_stratified_training_tiles(self):
  m=native_image(manifest()['train'][0]);coords=choose_tiles(m,64,4,42)
  for y,x in coords:
   a,g,ign,h,w=tile_signal(m,y,x,64,True)
   labels=sample_targets(g,ign,11,np.random.default_rng(42),24)
   self.assertGreater(np.count_nonzero(labels),0)
   self.assertEqual(a.shape,(64,64))
 def test_all_native_pixels_covered(self):
  for L in (360,427,640):
   covers=np.zeros(L,bool)
   for p in coverage_positions(L,64,48):covers[p:p+64]=True
   self.assertTrue(covers.all())
 def test_background_is_unobserved_in_inference(self):
  scores=np.zeros((4,20,30),'float32');scores[3]=1
  pred,_=postprocess(scores,0,[0,0,0]);self.assertTrue(np.all(pred==3))
 def test_original_gamma_zero_same_solution(self):
  x=np.random.default_rng(15).normal(size=(1,32,32)).astype('float32')*.02
  D=init_dict(x,4,7);W=np.zeros((4,4),'float32');b=np.zeros(4,'float32')
  y=np.zeros((1,4,32,32),np.int8);y[:,0,10,10]=1
  za,_=solve_z(D,x,W,b,y,0,.02,maxiter=40)
  zb,_=solve_z(D,x,W,b,None,0,.02,maxiter=40)
  self.assertTrue(np.allclose(za,zb))
 def test_admm_convergence_is_reported_not_assumed(self):
  x=np.random.default_rng(15).normal(size=(1,32,32)).astype('float32')*.03
  D=init_dict(x,4,7);W=np.zeros((4,4),'float32');b=np.zeros(4,'float32')
  z,diag=solve_z(D,x,W,b,None,0,.02,maxiter=2)
  self.assertEqual(diag['iterations'],2);self.assertFalse(diag['converged'])
 def test_dictionary_ball(self):
  D=init_dict(np.zeros((1,32,32),'float32'),4,7)
  x=np.random.default_rng(15).normal(size=(1,32,32)).astype('float32')*.02
  z,_=solve_z(D,x,np.zeros((4,4),'float32'),np.zeros(4,'float32'),None,0,.015,maxiter=30)
  d,_=solve_d(x,z,D,maxiter=30)
  self.assertLessEqual(float(np.linalg.norm(d,axis=(1,2)).max()),1.00001)

if __name__=='__main__':unittest.main()
