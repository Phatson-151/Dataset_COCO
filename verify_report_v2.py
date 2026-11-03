"""Cross-check generated report/goal statements against checked numerical JSON."""
import json
from paper_scsc.v2 import OUT

def main():
 r=json.loads((OUT/'REPORT_METRICS.json').read_text());g=json.loads((OUT/'GOAL_STATUS.json').read_text());text=(OUT/'FINAL_REPORT_TH.md').read_text()
 a=r['TEST_PREVIOUSLY_SEEN']['CSC'];b=r['TEST_PREVIOUSLY_SEEN']['SCSC']
 for field,delta in g['seed42_deltas'].items():assert abs(delta-(b[field]-a[field]))<1e-10
 for field in ['macro_AP_foreground','macro_AP_all_classes','foreground_mIoU','all_class_mIoU','pixel_accuracy','background_FPR','PSNR_db_mean_image','SSIM_mean_image','relative_reconstruction_error_mean_image']:
  expected=f'| {field} | {a[field]:.6f} | {b[field]:.6f} | {b[field]-a[field]:+.6f} |';assert expected in text,field
 assert g['independent_unseen_superiority'] is False
 assert g['K100_excluded'] is True
 assert 'ยังไม่พิสูจน์ว่า SCSC ชนะ CSC อย่างสม่ำเสมอ' in text
 print('Report tables, deltas and limitations match numerical results')
if __name__=='__main__':main()
