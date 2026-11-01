"""Run after experiment completion; checks deliverable rather than future model quality."""
import unittest,json
import numpy as np
from pathlib import Path
from paper_scsc.data import ROOT,manifest

class ReleaseArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.out=ROOT/'final output'
  if not (cls.out/'FINAL_METRICS.json').exists():raise unittest.SkipTest('Legacy artifact schema absent; new release checks are in test_new_experiment_protocol.NewReleaseArtifactTests')
  cls.report=json.loads((cls.out/'FINAL_METRICS.json').read_text())
 def test_ten_outer_each_candidate(self):
  for K in self.report['protocol']['args']['filters']:
   for arm in ('CSC','SCSC'):
    d=json.loads((self.out/'train_logs'/f'K{K}_{arm}.json').read_text())
    self.assertEqual(len(d['outer']),self.report['protocol']['args']['outer'])
 def test_native_score_shapes(self):
  k=self.report['selected_K_by_VAL_comparative_mIoU_and_convergence']
  for s in ('val','test_sealed'):
   for e in manifest()[s]:
    for arm in ('CSC','SCSC'):
     a=np.load(self.out/'score_maps'/f'{s}_K{k}_{arm}_{e["image_id"]}.npz')
     self.assertEqual(a['scores'].shape,(4,e['height'],e['width']))
     self.assertEqual(a['prediction'].shape,(e['height'],e['width']))
 def test_four_class_confusion(self):
  for s in ('VAL','TEST_PREVIOUSLY_SEEN'):
   for arm in ('CSC','SCSC'):
    m=self.report[s][arm]
    self.assertEqual(np.asarray(m['confusion_gt_rows_pred_cols']).shape,(4,4))
 def test_fair_training_source(self):
  p=self.report['protocol']
  self.assertTrue(p['native_resolution_no_resize'])
  self.assertTrue(p['background_explicit_classifier'])
  self.assertEqual(len(p['splits']['train']),10)
  self.assertEqual(len(p['splits']['val']),3)
  self.assertEqual(len(p['splits']['test_sealed']),5)
