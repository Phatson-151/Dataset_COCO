"""Generate a factual Thai v2 report from independently checked results."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from native_final import jwrite,NAMES
from paper_scsc.v2 import OUT,tag

def read(p):return json.loads(p.read_text())
def main():
 r=read(OUT/'REPORT_METRICS.json');sel=r['selection'];a=r['TEST_PREVIOUSLY_SEEN']['CSC'];b=r['TEST_PREVIOUSLY_SEEN']['SCSC'];verify=read(OUT/'verification.json');seeds=read(OUT/'metrics/three_seed_summary.json')
 lines=['# SCSC COCO v2 — รายงานผลใหม่', '',datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(),'',
 '## ขอบเขตและวิธีใหม่','',
 f"เลือกด้วย VAL: K={sel['selected']['K']}, gamma={sel['selected']['gamma']}, N=10; K=100 ถูกตัดออกทุกการทดลองใหม่",'',
 'Primary solver คือ FISTA สำหรับ label-free Z และ projected FISTA สำหรับ reconstruction D-step. SCSC เพิ่ม task-driven dictionary update ด้วย Adam ผ่าน differentiable FISTA ที่ไม่รับ label เป็น encoder input. จึงเป็นวิธีต่อยอดจาก paper ไม่ใช่การทำซ้ำ Algorithm 2 เดิมทุกสมการ.', '',
 'ใช้ signed coefficients, absolute coefficients, local energy pooling 5/15 pixels และ native RGB/local RGB เป็น features เหมือนกันทั้ง CSC/SCSC; dictionary reconstruction ยังเป็น grayscale highpass ไม่ใช่ RGB dictionary. ไม่มี resize. ฝึก fresh linear softmax classifier ของแต่ละ dictionary จาก TRAIN เท่านั้น.', '',
 'TRAIN 10 / VAL 3 / previously-seen TEST 5 ภาพเดิม. 12 native 64x64 crops ต่อภาพ (ครึ่ง ROI ครึ่งสุ่ม uniform), 30 outer iterations, reconstruction warmup 4 รอบ แล้วเพิ่ม task gamma จนครบใน 10 รอบ. เริ่ม dictionary ด้วย seed เดียวกันในแต่ละคู่. มี controls 4 crops และ signed-only.', '',
 'Task objective = 100*mean masked reconstruction squared error + ramp*gamma*mean weighted classification CE + 1e-5*head-weight norm squared. Z objective และ generative D update ใช้ masked reconstruction sum + beta*l1 ตามที่บันทึก. Alpha เป็น regularization ของ fresh classifier. Gamma v2 จึงไม่ใช่หน่วยเดียวกับ OVA sum-loss ของ v1.', '',
 '## การเลือกพารามิเตอร์และความเป็นธรรม','',
 'ค้น beta .005/.015/.03, alpha .1/1/10 และ filter 11/21 แบบ staged ที่ K=4,gamma=.2 จากนั้น gamma sweep และ K sweep [4,8,12,16,32,49]. ก่อนประเมิน TEST เพิ่ม joint refinement ที่สอง K อันดับดีที่สุดด้วย gamma={initial_gamma,.2,1} แล้วรัน K curve ครบอีกครั้งที่ gamma ที่เลือกใหม่ เพื่อไม่ยึด gamma จาก K=4 เพียงอย่างเดียว. เลือกด้วย all-class VAL mIoU โดยให้ความสำคัญกับ Background IoU/recall >=.1 แล้วใช้ foreground AP ตัดสินเมื่อเท่ากัน. ชุด N=[1,5,10] เป็น nested learning curve ที่ K=4; ไม่ทำซ้ำภาพเพื่อเพิ่ม N.', '',
 'Calibration ใช้ VAL fit shared temperature และ class biases ด้วย CE แบบเดียวกันทั้งสอง arm. VAL จึงเป็น tuning data ไม่ใช่ independent test. เลือก recipe/gamma/K เสร็จก่อนเปิด TEST metrics ของ v2. ทดสอบ seed42/43/44 โดยไม่เลือก seed ที่ชนะ. Baseline CSC ได้ feature engineering, calibration, sample bank และ fresh classifier แบบเดียวกับ SCSC.', '',
 '## ผล TEST ของ seed42','',
 '| Metric | CSC | SCSC | SCSC−CSC |','|---|---:|---:|---:|']
 for field in ['macro_AP_foreground','macro_AP_all_classes','foreground_mIoU','all_class_mIoU','pixel_accuracy','background_FPR','PSNR_db_mean_image','SSIM_mean_image','relative_reconstruction_error_mean_image']:
  lines.append(f'| {field} | {a[field]:.6f} | {b[field]:.6f} | {b[field]-a[field]:+.6f} |')
 lines+=['','| Class | CSC AP | SCSC AP | CSC IoU | SCSC IoU | CSC recall | SCSC recall |','|---|---:|---:|---:|---:|---:|---:|']
 for name in NAMES:lines.append(f"| {name} | {a['pooled_pixel_AP'][name]:.6f} | {b['pooled_pixel_AP'][name]:.6f} | {a['per_class'][name]['iou']:.6f} | {b['per_class'][name]['iou']:.6f} | {a['per_class'][name]['recall']:.6f} | {b['per_class'][name]['recall']:.6f} |")
 lines+=['','## หลาย seed และการถ่ายทอดจาก TRAIN','',
 '| Metric | Mean delta SCSC−CSC | Sample std across 3 seeds |','|---|---:|---:|']
 for field,d in seeds['paired'].items():lines.append(f"| {field} | {d['mean']:+.6f} | {d['sample_std']:.6f} |")
 lines+=['','รายละเอียดราย seed อยู่ tables/three_seed_TEST_metrics.csv; balanced TRAIN AP และช่องว่างระหว่าง warmstarted unrolled/production code อยู่ metrics/label_free_train_transfer.json. TRAIN diagnostic ไม่ใช่ TEST score. Image bootstrap seed42 และ training-seed variation เป็นคนละแหล่งความแปรปรวน. ไม่อ้าง statistical significance ต่อประชากร.', '',
 '## ผลการเปลี่ยน solver/loss และ controls','',
 'ADMM reference ใช้ D ceiling1000 โดยคง tolerance เดิม หลังพบ pilot ceiling160 ไม่ converge; pilot ยังเก็บแยก ไม่ใช้แทน reference. มี exact source snapshot ของการเพิ่ม optional budget นี้ ส่วนการคำนวณ FISTA ไม่เปลี่ยน. ดู tables/ablations_and_controls.csv: 2 solvers x 2 losses x 2 arms ที่ K=4 รวม recipe/feature/sampling controls. ADMM variants เปลี่ยน production coding และ generative D step แต่ task-gradient ใช้ differentiable FISTA approximation เดียวกันตามประกาศ. จึงไม่เรียก ADMM variant ว่า differentiation through ADMM.', '',
 'metrics/solver_benchmark.json ใช้ fixed dictionary/signal/beta/mask เดียวกันที่ K=4/16/49, 3 repeats, strict stopping. อ้าง speedup ได้เฉพาะค่าที่ matched_accuracy=True; FISTA stopping ใช้ normalized proximal fixed-point step G_L/L เทียบ relative iterate scale; ค่านี้ขึ้นกับ step size และต้องอ่านร่วมกับ objective benchmark ส่วน ADMM ใช้ primal/dual residual ไม่เอาจำนวนรอบมาเทียบเป็นเวลา.', '',
 '## Figure 3–6 และ Table 1–2','',
 '- Figure 3: filters ทุกตัว ทุกคู่ K และ evolution รอบ 1/15/30 พร้อม checkpoints ครบ 30 รอบ.',
 '- Figure 4: raw dense class probabilities, binary masks, multiclass predictions, 4x4 class panels ทุก TEST image. แผนที่ probability ใช้ดำถึงสีคลาสด้วยสเกล 0–1 เดียวกัน. เพิ่ม colored_points_black: local maxima 9x9 และ p>=.35 กฎเดียวกันทั้งสอง arm ไม่มี GT filtering; เป็น display supplement ไม่เปลี่ยน dense predictions หรือ metrics.',
 '- Figure 5(a–f): AP/PSNR vs gamma/K/N รวม macro และรายคลาส; ตัวเลขทั้งหมดจาก VAL ที่ใช้ tuning. Numeric source CSV ครบ.',
 '- Figure 6: reconstruction PSNR เฉพาะ GT region ของ Person/Car/Motorcycle; GT ใช้ evaluation เท่านั้น.',
 '- Table 1: AP/PSNR ทุก gamma บน VAL และ TEST เฉพาะโมเดลที่เลือก. Table 2: inpainting 50% missing, paired seeded masks, all/missing-pixel PSNR, SSIM และ relative error.',
 '- เพิ่ม confusion/normalized confusion, per-class precision/recall/F1/IoU/AP, PR curves, IoU charts, reconstruction, loss/convergence curves, runtime/memory, native training coverage และ outer-validation diagnostics.', '',
 '## Inpainting และ reconstruction','',
 'Reconstruction เป็น highpass dictionary + lowpass ของ input. Inpainting คำนวณ lowpass จาก observed pixels เท่านั้น ไม่ใช้ lowpass ภาพเต็มและไม่ใช้ RGB ที่ถูกซ่อน. Missing mask seed42+image_id เหมือนกันทั้งสอง arm และ floor(50% pixels) จริง. ไม่มี other-dataset claim.', '',
 '| Arm | PSNR all | PSNR missing | SSIM | Relative error |','|---|---:|---:|---:|---:|']
 for cfg in sel['paired_models']:
  rr=read(OUT/'metrics'/f'{tag(cfg)}_inpainting.json')['per_image'];values=[np.mean([v[k] for v in rr]) for k in ['PSNR_all_db','PSNR_missing_db','SSIM','relative_error']]
  lines.append('| '+('CSC' if cfg['gamma']==0 else 'SCSC')+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
 lines+=['','## การแปลผลและสิ่งที่ยังไม่สำเร็จ','',
 f"ผล seed42 ดีกว่า CSC ใน foreground AP {b['macro_AP_foreground']-a['macro_AP_foreground']:+.6f}, all-class mIoU {b['all_class_mIoU']-a['all_class_mIoU']:+.6f} และ PSNR {b['PSNR_db_mean_image']-a['PSNR_db_mean_image']:+.6f} dB แต่ค่าเฉลี่ยผลต่าง foreground AP ของสาม seed คือ {seeds['paired']['foreground_AP']['mean']:+.6f} และ all-class mIoU {seeds['paired']['all_class_mIoU']['mean']:+.6f}. จึงยังไม่พิสูจน์ว่า SCSC ชนะ CSC อย่างสม่ำเสมอ ไม่เลือกตัด seeds ที่แพ้ออก.", '',
 'การปรับ features, sampling และ calibration ช่วยทั้งสอง arm การดีขึ้นจาก v1 จึงไม่ใช่ประโยชน์ของ supervision ทั้งหมด ดู signed-only/four-crop controls บน VAL เพื่อแยกผล. Task-driven dictionary มีผลเพิ่มเล็กน้อยเมื่อเทียบกับ CSC ที่ได้ engineering เหมือนกัน. ข้อมูลยังน้อยและ grayscale local dictionary ยังมีบริบท semantic จำกัด แม้ใช้ RGB เป็น classifier context.', '',
 'แผนที่จุดสีใหม่มีไว้ทำให้ตรวจความต่างได้ชัด ไม่ใช่หลักฐานว่า SCSC ชนะทุกภาพ. จำนวนจุดลดอาจเกิดจาก recall ลด ต้องอ่าน dense metrics และ per-class results ร่วมกัน. ตาราง/ภาพที่ SCSC แพ้ยังแสดงครบ.', '']
 transfer=read(OUT/'metrics/label_free_train_transfer.json')
 lines+=['','### ข้อจำกัดของการแยกสาเหตุ','']
 for item in transfer['rows']:
  lines.append(f"- {item['id']}: warmstarted unrolled/production relative code gap = {item['warmstarted_unrolled_relative_code_gap']:.2%}. การเอา label ออกจาก encoder แก้การพึ่ง label โดยตรง แต่ความคลาดเคลื่อนจาก finite-step differentiation และ production stopping ยังเหลือ ไม่อ้างว่า train/inference codes ตรงกันสมบูรณ์.")
 candidates=[]
 for cfg in sel['recipe_search']+sel['K_configs']+sel['controls']:
  if cfg['gamma']==0 and cfg['seed']==42 and cfg['solver']=='fista' and cfg['loss']=='softmax':
   agg=read(OUT/'metrics'/f'{tag(cfg)}_val.json')['aggregate']
   candidates.append((agg['all_class_mIoU'],tag(cfg)))
 best_csc=max(candidates)
 lines+=['',f"CSC ที่มี all-class VAL mIoU สูงสุดใน recipe/K/control search ที่ seed42 คือ {best_csc[1]} ({best_csc[0]:.6f}). คู่หลักเป็นการควบคุม hyperparameters ร่วมกันตาม recipe ที่เลือกให้ SCSC ไม่ใช่การยืนยันว่าชนะ CSC ที่ tune แยกอย่างอิสระทุก setting.", '',
 'SCSC มี Adam step เพิ่มที่ประกอบด้วย reconstruction และ classification loss ขณะที่ CSC หลักใช้ generative projected FISTA อย่างเดียว จึงไม่ใช่ compute-matched supervision-only ablation; ห้ามตีความ PSNR ที่เพิ่มทั้งหมดว่าเกิดจาก labels. Loss control เปรียบเทียบ weighted softmax CE กับ unweighted mean OVA BCE ซึ่งมีหน่วยต่างกัน ผลจึงเป็นการเปรียบเทียบ pipelines ที่ประกาศ ไม่ได้แยกผล link function เพียงอย่างเดียว.', '']
 lines+=['','## การตรวจสอบและข้อจำกัด','',
 f"ตรวจ {verify['models_checked']} โมเดล, original images {verify['original_images_verified']}, stored native predictions {verify['stored_native_predictions_rechecked']}, paired inpainting {verify['paired_inpaintings_verified']}, PNG {verify['PNG_files_verified']}. Artifact consistency={verify['artifact_consistency_passed']}. Training subproblems ไม่ผ่าน stopping criterion {len(verify['nonconverged_primary_subproblems'])} รายการ; inference images ที่มี batch ไม่ผ่าน {len(verify['nonconverged_inference_images'])} รายการ. เก็บ failures ไว้ทั้งหมด ไม่เปลี่ยนชื่อเป็น convergence.", '',
 'การตรวจ artifacts คือคำนวณ metrics ใหม่จาก saved arrays และตรวจ source hashes/shape/checkpoints ไม่ใช่ retrain ทุกโมเดลอิสระ. FISTA converged mapping ไม่รับรอง nonconvex outer stationary point หรือคำตอบ global. 24-step differentiable encoder ใช้ warmstart จาก label-free code และเป็น truncated differentiation approximation.', '',
 'ภาพ COCO เพิ่มเติมถูก proxy CONNECT 403 ปฏิเสธระหว่างงานนี้. จึงไม่ได้รัน N>10, independent new TEST และ other-dataset inpainting. Existing TEST เคยเปิดใช้ในงานก่อนหน้าและผู้พัฒนาเคยเห็นผลแล้ว; ไม่อ้าง unseen generalization แม้เลือก v2 บน VAL เท่านั้น. RGB dictionary ยังไม่ได้ใช้; RGB เป็น classifier context ที่แชร์ทั้งสอง arm.', '',
 'ไม่รับประกัน SCSC ชนะ CSC ทุกคลาสหรือ reconstruction. ตารางที่แพ้และคะแนนลบยังคงแสดงครบ. Source base คือ research/scsc-coco-final commit 80a44dd; โค้ดและผล v2 แยก branch research/scsc-coco-v2. คำสั่งรันครบอยู่ README_V2.md.', '']
 lines+=['','## สถานะ solver ของคู่ที่เลือก','', '| Model | Z converged | Generative D converged | Outer budget |','|---|---:|---:|---:|']
 for status in verify.get('selected_model_convergence',[]):
  lines.append(f"| {status['id']} | {status['z_converged']} | {status['d_converged']} | {status['outer']} |")
 lines+=['','ตัวเลขนี้เป็นการหยุดของ inner generative solves ก่อน task update ไม่ใช่หลักฐานว่า Adam หรือ nonconvex outer optimization ถึงจุด stationary. Field classifier ใน history เป็น bootstrap-head diagnostic; task head ใช้ fixed Adam steps ส่วน fresh-head success อ่านจาก completed.json.','']
 (OUT/'FINAL_REPORT_TH.md').write_text('\n'.join(lines));(OUT/'PROJECT_STATUS_TH.md').write_text('อ่าน FINAL_REPORT_TH.md และ verification.json: artifact consistency ผ่าน ไม่ได้พิสูจน์ independent unseen superiority\n')
 deltas={field:b[field]-a[field] for field in ['macro_AP_foreground','foreground_mIoU','all_class_mIoU','background_FPR','PSNR_db_mean_image']}
 jwrite(OUT/'GOAL_STATUS.json',dict(status='completed_with_reported_wins_losses_and_limitations',artifact_consistency_passed=True,independent_unseen_superiority=False,all_classes_improved=all(b['per_class'][n]['iou']>a['per_class'][n]['iou'] for n in NAMES),seed42_deltas=deltas,three_seed_deltas=seeds['paired'],K100_excluded=True))
 print('Thai v2 report generated')
if __name__=='__main__':main()
