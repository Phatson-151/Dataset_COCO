import unittest
import numpy as np
from paper_scsc import v2_solver as vs
from paper_scsc.solver import init_dict,solve_z,kernel_fft,conv,make_mask

class FistaTests(unittest.TestCase):
 def test_fista_matches_admm_objective(self):
  rng=np.random.default_rng(3);x=rng.normal(size=(2,20,23)).astype('float32')*.1;d=init_dict(x,3,5)
  z,a=vs.code(d,x,.015,2000,1e-5)
  zz,b=solve_z(d,x,np.zeros((4,3),'float32'),np.zeros(4,'float32'),None,0,.015,maxiter=2000,tol=1e-5,adaptive=True)
  m=make_mask(x.shape[-2:],5)
  obj=lambda v:float(.5*np.sum((m*(conv(kernel_fft(d,x.shape[-2:]),v)-x))**2)+.015*abs(v).sum())
  self.assertLess(abs(obj(z)-obj(zz))/max(obj(zz),1),.003)
  self.assertTrue(a['converged']);self.assertTrue(b['converged'])
 def test_projected_dictionary_descent_and_norm(self):
  x=np.random.default_rng(4).normal(size=(2,20,23)).astype('float32')*.1;d=init_dict(x,3,5)
  z,_=vs.code(d,x,maxiter=200);dn,diag=vs.dictionary(d,x,z,100)
  self.assertLessEqual(diag['candidate_feasible_objective'],diag['initial_feasible_objective']+1e-5)
  self.assertLessEqual(np.linalg.norm(dn,axis=(1,2)).max(),1.00001)
 def test_masked_code_cannot_see_hidden_pixels(self):
  x=np.random.default_rng(5).normal(size=(1,20,23)).astype('float32')*.1;d=init_dict(x,2,5);m=make_mask(x.shape[-2:],5)[None];m[:,:,::2]=0
  alt=x.copy();alt[m==0]=999
  a,_=vs.code(d,x,maxiter=80,mask=m);b,_=vs.code(d,alt,maxiter=80,mask=m)
  np.testing.assert_array_equal(a,b)
 def test_torch_label_free_code_matches_fixed_fista(self):
  import torch
  x=np.random.default_rng(6).normal(size=(1,20,23)).astype('float32')*.1;d=init_dict(x,2,5)
  # At sufficient accuracy acceleration restart and fixed unrolling agree in objective.
  z=vs.torch_code(torch.tensor(d),torch.tensor(x),.015,steps=1500).numpy();zz,_=vs.code(d,x,maxiter=1500,tol=1e-6)
  np.testing.assert_allclose(z,zz,atol=.003)
 def test_task_gradient_reaches_dictionary_and_probabilities_normalize(self):
  import torch
  from paper_scsc.v2_solver import torch_code
  x=torch.randn(1,20,23)*.1;d=torch.tensor(init_dict(x.numpy(),2,5),requires_grad=True)
  z=torch_code(d,x,.015,12);w=torch.randn(4,2);scores=torch.einsum('ck,nkhw->nchw',w,z)
  loss=torch.nn.functional.cross_entropy(scores,torch.zeros((1,20,23),dtype=torch.long));loss.backward()
  self.assertTrue(torch.isfinite(d.grad).all());self.assertGreater(float(d.grad.norm()),0)
  np.testing.assert_allclose(torch.softmax(scores,1).detach().numpy().sum(1),1,atol=1e-6)

class V2ProtocolTests(unittest.TestCase):
 def test_softmax_fresh_head_learns_four_exclusive_classes(self):
  from paper_scsc.v2 import fit_head,config,feature_array
  from scipy.special import softmax
  y=np.tile(np.arange(4)[None,None,:],(2,24,6)).astype(np.int64)
  colors=np.array([[1,0,0],[0,0,1],[1,1,0],[0,0,0]],np.float32)
  rgb=colors[y].transpose(0,3,1,2);z=np.zeros((2,2,24,24),np.float32);cfg=config(2,.1,1)
  w,b,mean,scale,diag=fit_head(z,rgb,y,cfg)
  f=feature_array(z,rgb,cfg);scores=np.einsum('cf,nfhw->nchw',w,(f-mean[None,:,None,None])/scale[None,:,None,None])+b[None,:,None,None]
  p=softmax(scores,axis=1)
  self.assertTrue(diag['success']);self.assertGreater(np.mean(p.argmax(1)==y),.99)
  np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
 def test_native_inference_has_no_ground_truth_dependency(self):
  from paper_scsc.v2 import infer,config
  rng=np.random.default_rng(8);shape=(35,43);d=init_dict(np.zeros((1,*shape),'float32'),2,5)
  cfg=config(2,.1,1,tile=32,filter_size=5,stride=20,eval_admm=20)
  model=dict(D=d,W=np.zeros((4,14),np.float32),b=np.arange(4,dtype='float32'),mean=np.zeros(14,np.float32),scale=np.ones(14,np.float32))
  im=dict(x=rng.normal(size=shape).astype('float32')*.05,rgb=rng.integers(0,256,(*shape,3),dtype='uint8'),shape=shape)
  s,r,diag=infer(im,model,cfg);other=dict(im,gt=np.ones((4,*shape),bool),ignore=np.ones(shape,bool))
  ss,rr,_=infer(other,model,cfg);np.testing.assert_array_equal(s,ss);np.testing.assert_array_equal(r,rr)
  self.assertEqual(s.shape,(4,*shape));self.assertGreater(diag['min_weight'],0)
 def test_crop_bank_is_deterministic_and_native(self):
  from paper_scsc.v2 import bank_for,config
  a,r,y,record=bank_for(config(4,.1,1));b,rr,yy,other=bank_for(config(4,.1,1))
  np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(r,rr);np.testing.assert_array_equal(y,yy)
  self.assertEqual(len(a),12);self.assertEqual(record,other);self.assertEqual(a.shape[1:],(64,64))

class TaskGradientOracleTests(unittest.TestCase):
 def test_unrolled_dictionary_gradient_matches_finite_difference_at_fixed_step(self):
  import torch
  from unittest.mock import patch
  rng=np.random.default_rng(17);x=torch.tensor(rng.normal(size=(1,12,13))*.3,dtype=torch.float64)
  d=torch.tensor(init_dict(x.numpy(),2,3).astype('float64'),requires_grad=True)
  # Line-search/spectral step is deliberately detached in production. Freeze it
  # in the oracle as well so this checks the declared partial derivative.
  with patch('torch.max',return_value=torch.tensor(20.,dtype=torch.float64)):
   loss=(vs.torch_code(d,x,.001,8)**2).sum();loss.backward();analytic=d.grad.detach().numpy()
   for index in [(0,0,0),(0,1,2),(1,2,1)]:
    plus=d.detach().clone();minus=d.detach().clone();plus[index]+=1e-6;minus[index]-=1e-6
    a=float((vs.torch_code(plus,x,.001,8)**2).sum());b=float((vs.torch_code(minus,x,.001,8)**2).sum())
    self.assertAlmostEqual(float(analytic[index]),(a-b)/2e-6,places=5)
