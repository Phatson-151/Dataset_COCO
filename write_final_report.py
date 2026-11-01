"""Write the Thai report from new results, without treating hypotheses as facts."""
import json
from pathlib import Path
import numpy as np
from native_final import jwrite,NAMES
from paper_scsc.experiments import OUT,tag


def read(p):return json.loads(Path(p).read_text())
def fmt(x):return f'{x:.6f}' if x is not None else 'N/A'


def main():
    r=read(OUT/'REPORT_METRICS.json');v=read(OUT/'verification.json');s=r['selection']
    a=r['TEST_PREVIOUSLY_SEEN']['CSC'];b=r['TEST_PREVIOUSLY_SEEN']['SCSC']
    pair=s['paired_models'];ids=[tag(c['K'],c['gamma'],c['N']) for c in pair]
    inp=[read(OUT/'metrics'/f'{k}_inpainting.json')['per_image'] for k in ids]
    train=[read(OUT/'checkpoints'/k/'completed.json') for k in ids]
    lines=['# รายงานการทดลองใหม่ SCSC COCO Final',
        '', 'วันที่จัดทำ: 2 พฤศจิกายน 2026 (Asia/Bangkok)', '',
        '**สถานะ: รันการทดลองและตรวจความสอดคล้องของไฟล์แล้ว แต่ไม่ใช่ Fully Verified Final สำหรับการพิสูจน์บนข้อมูลใหม่อิสระ**', '',
        '## 1. Project objective', '',
        'ทดสอบสมมติฐานว่า supervision ช่วยให้ convolutional dictionary แยก Person, Car, Motorcycle และ Background ได้ดีขึ้น พร้อมรักษาคุณภาพ reconstruction โดยยอมรับผลลบตามจริง ไม่สร้างผลให้ SCSC ต้องชนะ', '',
        '## 2. Related papers', '',
        'อ่าน PDF ทั้งสองไฟล์ใน references/: Affara, Ghanem & Wonka, *Supervised Convolutional Sparse Coding*, arXiv:1804.02678v1 และ Wohlberg, *SPORCO*, SciPy 2017 อ่านสมการ 1–10, Algorithms 1–2, Sections 5.1–5.4 และ Figures 3–6 / Tables 1–2 เปรียบเทียบกับโค้ดใน PAPER_AUDIT.md มี SHA256 ของเอกสารใน environment.json', '',
        '## 3. Dataset description', '',
        'มีภาพ JPEG COCO ต้นฉบับ 18 ภาพ ตรวจ checksum และขนาดภาพครบทุกภาพ ใช้ pycocotools rasterizer มาตรฐานแทนการปัดพิกัด polygon ด้วย OpenCV จำนวน TRAIN/VAL/previously-seen TEST คือ 10/3/5 ภาพ ดู image IDs, dimensions, ignore count และจำนวนพิกเซลแต่ละคลาสใน experiment_configs/dataset_audit.json', '',
        'Background คือบริเวณที่ไม่ใช่สามคลาสเป้าหมาย ซึ่งรวมวัตถุคลาสอื่นและบริเวณไม่มี annotation ตามนิยามงาน ไม่ใช่การรับรองว่ามี annotation ครบทุกสิ่ง Ignore เฉพาะ target-class crowds และบริเวณซ้อนกันของคนละ target class ไม่ใช้ GT แก้ prediction', '',
        '**ข้อจำกัด holdout:** TEST ทั้ง 5 ภาพเคยถูกดูในโครงการเดิม จึงไม่ใช่ independent unseen test การตรวจการเข้าถึง images.cocodataset.org ถูก proxy ปฏิเสธ CONNECT HTTP 403 ไม่ได้ดาวน์โหลดภาพหรือ annotation เพิ่ม N=20/25/40 และ holdout ใหม่จึงเป็น `not run` ไม่ซ้ำภาพเพื่อเพิ่ม N', '',
        '## 4. Mathematical formulation', '',
        '`0.5 sum(M*(D*Z-X)^2) + beta*sum(abs(Z)) + gamma*(sum(log(1+exp(-Y*(WZ+b)))) + alpha*(||W||²+||b||²))`, ภายใต้ dictionary compact support และ filter norm <=1 ใช้ loss แบบ sum, beta=.015, alpha=1, filter 11x11 บน grayscale [0,1] highpass; beta ต่างจาก .5 ใน paper เพราะการเตรียมสัญญาณ/สเกลต่างกัน ไม่อ้างว่าเป็นค่าเดียวกัน', '',
        '## 5. CSC vs SCSC architecture', '',
        'CSC ใช้ gamma=0 ไม่มี supervised gradient ใน sparse coding ส่วน SCSC มี logistic loss ใน Z-step จริง ใช้ four one-vs-all heads ตามแนว multiclass extension ของ paper เพิ่ม Background เป็น head ที่สี่ ไม่ใช้ softmax ทั้งสอง arm เริ่ม dictionary seed เดียวกัน ใช้ภาพ/crop/labels เดียวกันและฝึก evaluation classifier ใหม่แยกกันจาก label-free TRAIN codes ของ dictionary แต่ละตัว ตาม Section 5.2', '',
        'Internal channel order = Person, Car, Motorcycle, Background; public class IDs ใน NPZ prediction = Background 0, Person 1, Car 2, Motorcycle 3 OVA sigmoid probability ไม่จำเป็นต้องรวมเป็น 1 Decision rule เป็น argmax logits เหมือนกันทั้งคู่ ไม่มี smoothing, offset calibration หรือ threshold tuning', '',
        '## 6. ADMM optimization', '',
        'FFT scaled ADMM แยก reconstruction / l1 / logistic prox ใช้ Newton สำหรับ logistic prox และ Woodbury สำหรับ frequency systems อัปเดต Z → global bank-wide D → theta บน supervised Z เดียวกัน แก้ข้อบกพร่องเดิมที่ใช้ EMA dictionary และ refit theta บน label-free codes ระหว่างฝึก การ fit theta ใช้ L-BFGS แทน gradient descent เป็น solver substitution ของ objective เดียวกัน', '',
        'แก้ boundary mask ให้ตรงกับ causal top-left kernel anchor: ตัด first f-1 rows/columns ที่ wrap ไม่ใช้ symmetric f//2 ที่ยังเหลือ circular artifacts ตรวจเทียบกับ direct linear convolution และ adjoint identity D ที่คืนต้องเป็น compact feasible dictionary; หากผล feasible ทำให้ objective แย่กว่าเดิม จะยังไม่ยอมรับว่า converge และป้องกันการคืน dictionary ที่แย่ลง', '',
        '## 7. Experimental setup', '',
        'CPU quota 4 cores, RAM 32 GiB, ไม่มี GPU ใช้ BLAS หนึ่ง thread ต่อ subprocess, training seed=42 จำนวน outer iterations=10 ทุก candidate, atol=1e-4, rtol=.003, max ADMM=1000 และ residual-balanced adaptive rho ไม่ผ่อน tolerance เพื่อให้ผ่าน', '',
        '**Native resolution:** ใช้ RGB ต้นฉบับสำหรับแสดงผล แปลง grayscale สำหรับ sparse optimization โดยไม่มี resize ฝึกสี่ intact 64x64 native crops/ภาพ ใช้ global D ร่วมทั้ง tile bank ไม่ใช่ global full-frame D การ inference เป็น overlapping tiles ที่มี reflected context และ blend เฉพาะ valid interior ครอบคลุมทุกพิกเซลในขนาดต้นฉบับ', '',
        'Full-frame feasibility pilot: TRAIN 10 ภาพ pad เท่านั้นเป็น 640x640, K=4; Z 10 ขั้นใช้ 36.58 วินาที และ D 10 ขั้น 7.64 วินาที ทั้งคู่ยังไม่ converge เป็น `inconclusive` pilot ไม่ใช่ผลฝึกสำเร็จ รายงานนี้จึงไม่เรียกการทดลองหลักว่า exact full-image optimization', '',
        '## 8. Hyperparameter search', '',
        f'Gamma grid: {[c["gamma"] for c in s["gamma_configs"]]}; K grid: {sorted(set(c["K"] for c in s["K_configs"]))}; N grid: {sorted(set(c["N"] for c in s["N_configs"]))}', '',
        f'เลือก gamma บน VAL ที่ K=4 แล้วเลือก K ด้วย gamma นั้น Criterion คือ maximum pooled foreground AP; ไม่เลือกจากผลต่าง SCSC−CSC ให้เป็นบวก ไม่เปิด TEST เพื่อเลือกระหว่าง candidates ผลสุดท้าย: **K={s["selected"]["K"]}, gamma={s["selected"]["gamma"]:g}, N=10** N เป็น nested learning curve ใช้ VAL/TEST ชุดเดิม ไม่ใช้ N เพื่อเลือกภาพที่ทำให้ TEST ดีขึ้น', '',
        'Controls: beta=.015, alpha=1, 11x11 filters, four native tiles/image, 24 positive + 24 negative anchors/head/tile เมื่อมีเพียงพอ N learning curve คง K=4 และใช้ gamma ที่ VAL เลือก ไม่ได้ทำ N sweep ซ้ำทุก K หรือสรุปว่าแนวโน้ม N ที่ K=4 ใช้แทน K=100 ได้ การ sweep gamma เป็น supervision ablation หลัก ไม่ได้รันเต็ม Cartesian grid ของ beta/alpha/filter21/seeds หรือ background-head ablation ให้ถือส่วนเหล่านี้ว่า `not run`', '',
        '## 9. Figure 3 results', '',
        'figures/figure3_dictionaries แสดง filters ทั้งหมดของทุก K ที่ทดลองคู่กัน CSC ซ้าย / SCSC ขวา ใช้ symmetric grayscale range เดียวกันในแต่ละคู่ ไม่เลือกเฉพาะ filters สวย มี dictionary evolution ที่ outer 1/5/10 และ supplementary absolute filter-correlation ใน tables/dictionary_diversity.csv ความคล้ายเชิง semantic จากรูปอย่างเดียวไม่ใช่หลักฐานการจำแนก ต้องดู AP/IoU ร่วมกัน', '',
        '## 10. Figure 4 results', '',
        'แสดงครบ TEST ทั้ง 5 ภาพทุกคลาส มี 4x4 composite, 4 panels แยกรายคลาส, dense OVA probabilities ในช่วง 0–1 ร่วมกัน, binary argmax masks, multiclass maps และ original RGB/GT ไม่มีการใช้ GT ลบ false positives สีของ probability ไม่ได้ปรับแยกเพื่อทำให้ SCSC ดูเด่น ผลภาพและ metrics มาจาก NPZ prediction เดียวกัน', '',
        '## 11. Figure 5 results', '',
        'figures/figure5_ap_psnr มี AP/PSNR vs gamma/K/N 6 panels รวมและแยก, macro และรายคลาส ทุกค่ามาจาก VAL เนื่องจากใช้เลือกโมเดล ไม่ใช่หลักฐาน independent test CSC ไม่ขึ้นกับ gamma จึงเป็น horizontal baseline แกน gamma ใช้ symlog เพื่อแสดงช่วงใกล้ศูนย์และช่วง 100 ได้ชัด Numeric source อยู่ tables/figure5_6_numeric_source.csv', '',
        '## 12. Figure 6 results', '',
        'figures/figure6_class_psnr วัด reconstruction PSNR เฉพาะ native GT region ของ Person/Car/Motorcycle แยกจริง ไม่ใช้ global PSNR เดิมซ้ำเป็นสามคลาส Class PSNR aggregate เป็น mean ต่อภาพที่มีคลาสนั้น, global PSNR เป็น mean ต่อภาพ; GT ใช้เฉพาะ evaluation', '',
        '## 13. Tables 1–2 results', '',
        'Table 1 ใน tables/table1_gamma_ap_psnr แสดง pooled pixel AP (0–1), foreground macro AP, global และ class-region PSNR (dB) ของทุก gamma บน VAL พร้อม transposed layout TEST แสดงเฉพาะคู่ที่เลือกแล้ว ไม่มี TEST sweep เพื่อเลือก gamma', '',
        'Table 2 ใน tables/table2_inpainting แสดง PSNR ทั้งภาพและเฉพาะ missing pixels, SSIM, relative error ต่อภาพและค่าเฉลี่ย พร้อม figures/inpainting ไม่มี other-dataset table ที่สมมติขึ้น', '',
        '## 14. Multiclass evaluation — previously-seen TEST', '',
        '| Metric | CSC | SCSC | SCSC−CSC |', '|---|---:|---:|---:|']
    for name,field in [('Foreground pooled macro AP','macro_AP_foreground'),('Foreground mIoU','foreground_mIoU'),('All-class mIoU','all_class_mIoU'),('Pixel accuracy','pixel_accuracy'),('Background FPR (lower better)','background_FPR'),('Global PSNR dB','PSNR_db_mean_image'),('SSIM','SSIM_mean_image'),('Relative reconstruction error (lower better)','relative_reconstruction_error_mean_image')]:
        lines.append(f'| {name} | {fmt(a[field])} | {fmt(b[field])} | {fmt(b[field]-a[field])} |')
    lines+=['','| Class | CSC AP | SCSC AP | CSC IoU | SCSC IoU | CSC precision / recall | SCSC precision / recall |','|---|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        x=a['per_class'][name];y=b['per_class'][name]
        lines.append(f'| {name} | {fmt(a["pooled_pixel_AP"][name])} | {fmt(b["pooled_pixel_AP"][name])} | {fmt(x["iou"])} | {fmt(y["iou"])} | {fmt(x["precision"])} / {fmt(x["recall"])} | {fmt(y["precision"])} / {fmt(y["recall"])} |')
    lines+=['','F1, support, TP/FP/FN, full confusion และ normalized confusion เก็บใน metrics JSON/CSV มี precision–recall curves และ IoU comparison chart ตัวเลข AP รวมพิกเซลทั้ง split; AP ต่อภาพแยกเก็บใน per_image ไม่สลับนิยามสองแบบนี้','',
        '## 15. Confusion matrix analysis','',
        'Confusion เป็น GT row / predicted column มีทุกทิศทาง Person↔Car↔Motorcycle, foreground→Background และ Background→foreground ไม่ซ่อนค่า zero หรือ false positives ภาพ heatmap ใช้ลำดับ Background, Person, Car, Motorcycle ส่วน raw JSON ใช้ internal score order ตามระบุ','',
        f'Foreground-only cross-class confusion rate: CSC={a["foreground_class_confusion_rate"]:.6f}, SCSC={b["foreground_class_confusion_rate"]:.6f}. Background→foreground FPR: CSC={a["background_FPR"]:.6f}, SCSC={b["background_FPR"]:.6f}. Class confusion rate รวม (1−pixel accuracy): CSC={a["class_confusion_rate"]:.6f}, SCSC={b["class_confusion_rate"]:.6f}.','',
        '## 16. Reconstruction and inpainting','',
        'Reconstruction ใช้ grayscale highpass sparse representation บวก lowpass ที่เก็บจากภาพอินพุตตามแนว SPORCO ไม่ใช่ RGB reconstruction Lowpass มีส่วนทำให้ global PSNR สูง ต้องไม่อ้างว่า PSNR ทั้งหมดเกิดจาก dictionary เพียงอย่างเดียว','',
        'Inpainting ใช้ floor(50% pixels) หายแบบสุ่ม seed=42+image_id และ mask เดียวกันทั้งสอง arm Gaussian normalized interpolation สำหรับ lowpass ใช้เฉพาะ observed pixels ส่วน sparse coding มี observed mask อยู่ใน data term ไม่ใช้ lowpass จากภาพเต็มและไม่คัดลอกพิกเซลที่หายกลับจากต้นฉบับ','',
        '| Inpainting metric (mean over 5 images) | CSC | SCSC |','|---|---:|---:|']
    for field in ['PSNR_all_db','PSNR_missing_db','SSIM','relative_error']:
        vals=[np.mean([row[field] for row in rows]) for rows in inp]
        lines.append(f'| {field} | {vals[0]:.6f} | {vals[1]:.6f} |')
    lines+=['','## 17. ADMM convergence analysis','',
        '| Selected model | Z converged / 10 | D converged / 10 | Fresh coding | Fresh classifier | Train seconds | Peak RSS MiB |','|---|---:|---:|---|---|---:|---:|']
    for arm,t in zip(['CSC','SCSC'],train):
        lines.append(f'| {arm} | {t["z_converged"]}/10 | {t["d_converged"]}/10 | {t["fresh_encoding"]["converged"]} | {t["fresh_classifier"]["success"]} | {t["seconds"]:.2f} | {t["peak_rss_MiB"]:.2f} |')
    lines+=['',f'ตรวจทั้งหมด {v["models_checked"]} โมเดล พบ primary Z/D subproblems ที่ไม่ converge {len(v["nonconverged_primary_subproblems"])} รายการ ดู verification.json และ convergence_runtime.csv การรันจบ (`completed`) ไม่เท่ากับทุก subproblem converge และไม่เท่ากับ nonconvex outer optimization ถึง stationary point', '',
        'Residual histories มี primal, dual, tolerances, rho, objective ทุก 10 ADMM iterations เกณฑ์ pass ต้องผ่าน residual ทั้งสองจริง ตรวจ losses แยกก่อน/หลัง Z, D และ classifier การสลับ outer อาจมี objective แกว่งเล็กน้อยจากการแก้ subproblem แบบ tolerance-limited ไม่อ้าง monotonic/global convergence เพียงเพราะครบ 10 รอบ', '',
        '## 18. Comparison with original paper','',
        'ทำตาม logistic supervision ใน Z, compact unit-ball dictionaries, FFT ADMM, alternating optimization และ fresh evaluation classifiers โดยตรง ดัดแปลง COCO เป็น four-class OVA พร้อม Background, sampling native tiles, grayscale low/high separation, reflected tiled inference และ L-BFGS แทน GD ไม่ใช่ exact full-image optimization และไม่ใช่ benchmark เดียวกับ Graz/ECP/LabelMe/Cat ใน paper จึงเปรียบเทียบแนวโน้ม ไม่เปรียบเทียบตัวเลขต่าง dataset ราวกับเป็นการทดลองเดียวกัน','',
        '## 19. Limitations and failed / unrun experiments','',
        '- Independent new holdout: `not run`, ไม่มีข้อมูลใหม่และ original COCO host ถูกปฏิเสธการเข้าถึง',
        '- N=20/25/40: `not run`, จำนวน TRAIN ที่มีจริงสูงสุด 10 ไม่เพิ่ม N ด้วยการใช้ภาพซ้ำ',
        '- Other-dataset / out-of-domain inpainting: `not run`, ไม่มี dataset ที่เข้าถึงได้และมี provenance เพียงพอ',
        '- Full-frame training: `inconclusive` feasibility pilot; ใช้ native-tile approximation สำหรับผลหลัก',
        '- RGB dictionary, filter21 sweep, beta/alpha sweep, multiple training seeds, background-head ablation: `not run`; ควบคุมค่าหลักเดียวกัน ไม่ได้ทำ full Cartesian grid',
        f'- Nonconverged primary ADMM subproblems: {len(v["nonconverged_primary_subproblems"])} รายการ ไม่เปลี่ยน tolerance หรือเปลี่ยนชื่อเป็นผ่าน',
        '- มีภาพ TEST เพียง 5 ภาพซึ่งเคยใช้แล้ว Bootstrap 2,000 paired image draws ใน paired_bootstrap.json วัดความแปรปรวนระหว่างภาพเท่านั้น ไม่มี training-seed variability และไม่อ้าง statistical significance ต่อประชากร', '',
        '## 20. Conclusion','']
    wins=[n for n in NAMES if b['per_class'][n]['iou']>a['per_class'][n]['iou']]
    losses=[n for n in NAMES if b['per_class'][n]['iou']<a['per_class'][n]['iou']]
    lines += [f'บน previously-seen TEST, SCSC มี IoU เพิ่มใน: {", ".join(wins) or "ไม่มี"}; IoU ลดใน: {", ".join(losses) or "ไม่มี"}. '
        f'Foreground macro AP เปลี่ยน {b["macro_AP_foreground"]-a["macro_AP_foreground"]:+.6f}, foreground mIoU เปลี่ยน {b["foreground_mIoU"]-a["foreground_mIoU"]:+.6f}, global PSNR เปลี่ยน {b["PSNR_db_mean_image"]-a["PSNR_db_mean_image"]:+.6f} dB.', '',
        'ผลนี้ตอบได้เฉพาะพฤติกรรมของการทดลองขนาดเล็กบนข้อมูลชุดนี้ ไม่พิสูจน์ความเหนือกว่าทุกคลาส ไม่รับรอง generalization บน independent unseen COCO และไม่รับรองว่า activation maps ดูดีกว่าเสมอ การทดสอบ artifact consistency ผ่านหมายถึงไฟล์/ตัวเลขสอดคล้องกัน ไม่ได้แปลว่าสมมติฐานทางวิจัยผ่าน', '',
        '## Verification and provenance','',
        f'ตรวจภาพต้นฉบับ {v["original_images_verified"]} ภาพ, native predictions ของคู่ final {v["selected_native_predictions_recomputed"]} ชุด, paired inpainting {v["paired_inpaintings_verified"]} ชุด และ PNG {v["PNG_files_verified"]} ไฟล์ อ่านค่าใหม่จาก NPZ เพื่อเทียบ confusion/AP/PSNR กับ JSON พร้อม SHA256 inventory ไฟล์ใหม่ทั้งหมด ไม่มีการใช้ผลทดลองเก่าเป็นผลใหม่', '',
        'Source branch: `research/scsc-coco-final`; baseline HEAD และ package versions อยู่ environment.json; source commit อยู่ verification.json ไฟล์ ZIP ใหม่รวม source, data, PDFs และผลใหม่เท่านั้น ไม่รวม virtualenv/cache/unused legacy scripts/pilot checkpoints']
    diagnostic=OUT/'metrics/ADMM_extended_ceiling_diagnostic.json'
    if diagnostic.exists():
        d=read(diagnostic)
        lines+=['','## Additional ADMM diagnostic','',
            f'ทดสอบ first alternating step ของ K=4, gamma=1 ซ้ำโดยเพิ่มเพดานเป็น 5,000 และคง atol/rtol เดิม: Z converged={d["Z"]["converged"]} ใน {d["Z"]["iterations"]} iterations; D converged={d["D"]["converged"]} ใน {d["D"]["iterations"]} iterations. Status={d["status"]}. เป็น numerical diagnostic เท่านั้น ไม่ใช่โมเดลที่ฝึก 10 outer iterations และไม่ได้แทนค่าหรือลบ failures ใน primary sweep', '',
            'สาเหตุที่แยกตรวจ: บาง Z-step เมื่อ gamma สูงต้องปรับ sparse codes จาก initialization มากจนเพดานเดิมไม่พอ; บาง D-step dual residual ยังสูงกว่า tolerance เล็กน้อย และบาง D-step ผ่าน residual แต่ feasible dictionary objective แย่ลงจึงถูก descent guard ปฏิเสธ ต้องแยกกรณีเหล่านี้จากการรันผิดหรือคำนวณ NaN']
    recovery=OUT/'logs/recovered_implementation_failures.json'
    if recovery.exists():
        lines+=['','## Solver failure recovery and reruns','',
            'การทดลอง SCSC รุ่นแรกของงานนี้ถูกเก็บไว้ใน failed_attempts/prox_v1 และไม่ใช้เลือกโมเดลหรือเป็นผล final สาเหตุคือ undamped Newton logistic prox สามารถแกว่งเป็นสองจุด เช่น a=−8, gamma=1, rho=1/16 ทั้งที่รากจริงเป็นศูนย์ เพิ่มเพดาน ADMM เป็น 5,000 ก็ไม่แก้ การแก้ใช้ monotone root bracket พร้อมรับ Newton step เฉพาะเมื่อ residual ลดอย่างน้อยครึ่งหนึ่ง มิฉะนั้น bisection และตรวจ stationarity ทุกครั้ง', '',
            'K=100 SCSC รุ่นแรกเกิด overflow ใน D-step จาก Gram/Woodbury cancellation แบบ float32 การเปลี่ยน Gram เป็น float64 อย่างเดียวยังไม่ผ่าน high-coherence regression จึงใช้ complex128 spectral linear solve จาก SVD ของ Z โดยตรง ซึ่งคง identity และ nullspace ได้เสถียรกว่า มี numerical regression เทียบ Brent root oracle, independent epigraph optimizer และกรณี Gram ill-conditioned รองรับ SCSC ทุกรุ่นที่ใช้เป็นผล primary ถูกฝึกใหม่หลังแก้ ไม่ใช้ผลรุ่นที่ผิดมาสรุปว่า paper สำเร็จ', '',
            'Baseline gamma=0 ที่เก็บไว้ไม่มี logistic prox และผ่าน finite-state/residual/feasible-descent checks; ใช้สมการและ tolerance เดียวกัน แต่บาง baseline รันก่อน precision/FFT refactor จึงไม่ได้อ้าง bitwise-identical execution ระหว่าง implementation revisions รายละเอียด source hashing ของงานใหม่อยู่ software.json ส่วน early baseline ไม่มี per-job source hash ที่เก็บทัน จึงเป็นข้อจำกัด provenance ที่ระบุไว้ตรงนี้']
        old=OUT/'failed_attempts/prox_v1/ADMM_extended_ceiling_diagnostic.json'
        new=OUT/'checkpoints/K4_g1_N10/history.json'
        if old.exists() and new.exists():
            previous=read(old);current=read(new)[0]
            comparison=dict(previous_undamped=dict(maxiter=5000,Z_converged=previous['Z']['converged'],Z_iterations=previous['Z']['iterations']),
                corrected=dict(maxiter=1000,Z_converged=current['z_admm']['converged'],Z_iterations=current['z_admm']['iterations'],D_converged=current['d_admm']['converged'],D_iterations=current['d_admm']['iterations']),
                note='Same K4 gamma1 N10 first alternating step; primary corrected model still has all ten outer iterations separately logged')
            jwrite(OUT/'metrics/ADMM_solver_fix_comparison.json',comparison)
            lines+=['',f'เปรียบเทียบ K=4, gamma=1 first alternating step: รุ่นเดิม maxiter=5000 Z-converged={previous["Z"]["converged"]}; รุ่นแก้ maxiter=1000 Z-converged={current["z_admm"]["converged"]} ใน {current["z_admm"]["iterations"]} ขั้น ไม่ย้ายผล diagnostic มาแทนผลฝึก 10 outer iterations']
        record=read(recovery);record['status']='corrected_primary_sweep_completed_and_artifacts_verified';jwrite(recovery,record)
    lines+=['','Implementation provenance: ระหว่างพัฒนาได้ลด redundant FFT โดย reuse zf/dhat ตามสมการเดิม คง objective และ stopping criteria เดิม มี independent scalar/high-K epigraph optimizer tests รองรับ เฉพาะ baseline CSC K=100 ที่เริ่มก่อน optimization ใช้รุ่น FFT เดิมต่อจนจบ ส่วน SCSC รุ่นผิดถูกเก็บเป็น failure evidence และฝึกใหม่; ไม่เรียกผลจาก profiler ว่าเป็น training result ดู profiling_summary.json และ software metadata โดยไม่อ้าง bitwise identity ระหว่าง implementation revisions']
    (OUT/'FINAL_REPORT_TH.md').write_text('\n'.join(lines)+'\n')
    (OUT/'PROJECT_STATUS_TH.md').write_text('# สถานะผลใหม่\n\nอ่าน FINAL_REPORT_TH.md และ verification.json\n\n'
        f'Artifact consistency: ผ่าน; primary ADMM ไม่ converge {len(v["nonconverged_primary_subproblems"])} รายการ; '
        'TEST ไม่อิสระ; N>=20 และ cross-dataset inpainting ไม่ได้รัน; ไม่ใช่ Fully Verified Final\n')
    print('Thai report written from verified new results')


if __name__=='__main__':main()
