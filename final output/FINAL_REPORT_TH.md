# SCSC COCO v2 — รายงานผลใหม่

2026-11-03T22:16:15.685408+07:00

## ขอบเขตและวิธีใหม่

เลือกด้วย VAL: K=4, gamma=0.05, N=10; K=100 ถูกตัดออกทุกการทดลองใหม่

Primary solver คือ FISTA สำหรับ label-free Z และ projected FISTA สำหรับ reconstruction D-step. SCSC เพิ่ม task-driven dictionary update ด้วย Adam ผ่าน differentiable FISTA ที่ไม่รับ label เป็น encoder input. จึงเป็นวิธีต่อยอดจาก paper ไม่ใช่การทำซ้ำ Algorithm 2 เดิมทุกสมการ.

ใช้ signed coefficients, absolute coefficients, local energy pooling 5/15 pixels และ native RGB/local RGB เป็น features เหมือนกันทั้ง CSC/SCSC; dictionary reconstruction ยังเป็น grayscale highpass ไม่ใช่ RGB dictionary. ไม่มี resize. ฝึก fresh linear softmax classifier ของแต่ละ dictionary จาก TRAIN เท่านั้น.

TRAIN 10 / VAL 3 / previously-seen TEST 5 ภาพเดิม. 12 native 64x64 crops ต่อภาพ (ครึ่ง ROI ครึ่งสุ่ม uniform), 30 outer iterations, reconstruction warmup 4 รอบ แล้วเพิ่ม task gamma จนครบใน 10 รอบ. เริ่ม dictionary ด้วย seed เดียวกันในแต่ละคู่. มี controls 4 crops และ signed-only.

Task objective = 100*mean masked reconstruction squared error + ramp*gamma*mean weighted classification CE + 1e-5*head-weight norm squared. Z objective และ generative D update ใช้ masked reconstruction sum + beta*l1 ตามที่บันทึก. Alpha เป็น regularization ของ fresh classifier. Gamma v2 จึงไม่ใช่หน่วยเดียวกับ OVA sum-loss ของ v1.

## การเลือกพารามิเตอร์และความเป็นธรรม

ค้น beta .005/.015/.03, alpha .1/1/10 และ filter 11/21 แบบ staged ที่ K=4,gamma=.2 จากนั้น gamma sweep และ K sweep [4,8,12,16,32,49]. ก่อนประเมิน TEST เพิ่ม joint refinement ที่สอง K อันดับดีที่สุดด้วย gamma={initial_gamma,.2,1} แล้วรัน K curve ครบอีกครั้งที่ gamma ที่เลือกใหม่ เพื่อไม่ยึด gamma จาก K=4 เพียงอย่างเดียว. เลือกด้วย all-class VAL mIoU โดยให้ความสำคัญกับ Background IoU/recall >=.1 แล้วใช้ foreground AP ตัดสินเมื่อเท่ากัน. ชุด N=[1,5,10] เป็น nested learning curve ที่ K=4; ไม่ทำซ้ำภาพเพื่อเพิ่ม N.

Calibration ใช้ VAL fit shared temperature และ class biases ด้วย CE แบบเดียวกันทั้งสอง arm. VAL จึงเป็น tuning data ไม่ใช่ independent test. เลือก recipe/gamma/K เสร็จก่อนเปิด TEST metrics ของ v2. ทดสอบ seed42/43/44 โดยไม่เลือก seed ที่ชนะ. Baseline CSC ได้ feature engineering, calibration, sample bank และ fresh classifier แบบเดียวกับ SCSC.

## ผล TEST ของ seed42

| Metric | CSC | SCSC | SCSC−CSC |
|---|---:|---:|---:|
| macro_AP_foreground | 0.210225 | 0.211749 | +0.001524 |
| macro_AP_all_classes | 0.335789 | 0.336034 | +0.000245 |
| foreground_mIoU | 0.097479 | 0.098533 | +0.001053 |
| all_class_mIoU | 0.209102 | 0.210588 | +0.001486 |
| pixel_accuracy | 0.506046 | 0.509566 | +0.003520 |
| background_FPR | 0.239552 | 0.228060 | -0.011492 |
| PSNR_db_mean_image | 40.449266 | 40.976237 | +0.526971 |
| SSIM_mean_image | 0.982158 | 0.983554 | +0.001396 |
| relative_reconstruction_error_mean_image | 0.021465 | 0.020222 | -0.001242 |

| Class | CSC AP | SCSC AP | CSC IoU | SCSC IoU | CSC recall | SCSC recall |
|---|---:|---:|---:|---:|---:|---:|
| Person | 0.100727 | 0.101759 | 0.056427 | 0.056445 | 0.139495 | 0.137452 |
| Car | 0.173506 | 0.177699 | 0.044378 | 0.053153 | 0.048275 | 0.058696 |
| Motorcycle | 0.356443 | 0.355789 | 0.191634 | 0.186001 | 0.279049 | 0.263981 |
| Background | 0.712480 | 0.708888 | 0.543970 | 0.546755 | 0.760448 | 0.771940 |

## หลาย seed และการถ่ายทอดจาก TRAIN

| Metric | Mean delta SCSC−CSC | Sample std across 3 seeds |
|---|---:|---:|
| foreground_AP | -0.000339 | 0.001638 |
| foreground_mIoU | -0.000391 | 0.001577 |
| all_class_mIoU | +0.000095 | 0.001219 |
| Background_FPR | -0.005735 | 0.006642 |
| PSNR | +0.517541 | 0.160707 |

รายละเอียดราย seed อยู่ tables/three_seed_TEST_metrics.csv; balanced TRAIN AP และช่องว่างระหว่าง warmstarted unrolled/production code อยู่ metrics/label_free_train_transfer.json. TRAIN diagnostic ไม่ใช่ TEST score. Image bootstrap seed42 และ training-seed variation เป็นคนละแหล่งความแปรปรวน. ไม่อ้าง statistical significance ต่อประชากร.

## ผลการเปลี่ยน solver/loss และ controls

ADMM reference ใช้ D ceiling1000 โดยคง tolerance เดิม หลังพบ pilot ceiling160 ไม่ converge; pilot ยังเก็บแยก ไม่ใช้แทน reference. มี exact source snapshot ของการเพิ่ม optional budget นี้ ส่วนการคำนวณ FISTA ไม่เปลี่ยน. ดู tables/ablations_and_controls.csv: 2 solvers x 2 losses x 2 arms ที่ K=4 รวม recipe/feature/sampling controls. ADMM variants เปลี่ยน production coding และ generative D step แต่ task-gradient ใช้ differentiable FISTA approximation เดียวกันตามประกาศ. จึงไม่เรียก ADMM variant ว่า differentiation through ADMM.

metrics/solver_benchmark.json ใช้ fixed dictionary/signal/beta/mask เดียวกันที่ K=4/16/49, 3 repeats, strict stopping. อ้าง speedup ได้เฉพาะค่าที่ matched_accuracy=True; FISTA stopping ใช้ normalized proximal fixed-point step G_L/L เทียบ relative iterate scale; ค่านี้ขึ้นกับ step size และต้องอ่านร่วมกับ objective benchmark ส่วน ADMM ใช้ primal/dual residual ไม่เอาจำนวนรอบมาเทียบเป็นเวลา.

## Figure 3–6 และ Table 1–2

- Figure 3: filters ทุกตัว ทุกคู่ K และ evolution รอบ 1/15/30 พร้อม checkpoints ครบ 30 รอบ.
- Figure 4: raw dense class probabilities, binary masks, multiclass predictions, 4x4 class panels ทุก TEST image. แผนที่ probability ใช้ดำถึงสีคลาสด้วยสเกล 0–1 เดียวกัน. เพิ่ม colored_points_black: local maxima 9x9 และ p>=.35 กฎเดียวกันทั้งสอง arm ไม่มี GT filtering; เป็น display supplement ไม่เปลี่ยน dense predictions หรือ metrics.
- Figure 5(a–f): AP/PSNR vs gamma/K/N รวม macro และรายคลาส; ตัวเลขทั้งหมดจาก VAL ที่ใช้ tuning. Numeric source CSV ครบ.
- Figure 6: reconstruction PSNR เฉพาะ GT region ของ Person/Car/Motorcycle; GT ใช้ evaluation เท่านั้น.
- Table 1: AP/PSNR ทุก gamma บน VAL และ TEST เฉพาะโมเดลที่เลือก. Table 2: inpainting 50% missing, paired seeded masks, all/missing-pixel PSNR, SSIM และ relative error.
- เพิ่ม confusion/normalized confusion, per-class precision/recall/F1/IoU/AP, PR curves, IoU charts, reconstruction, loss/convergence curves, runtime/memory, native training coverage และ outer-validation diagnostics.

## Inpainting และ reconstruction

Reconstruction เป็น highpass dictionary + lowpass ของ input. Inpainting คำนวณ lowpass จาก observed pixels เท่านั้น ไม่ใช้ lowpass ภาพเต็มและไม่ใช้ RGB ที่ถูกซ่อน. Missing mask seed42+image_id เหมือนกันทั้งสอง arm และ floor(50% pixels) จริง. ไม่มี other-dataset claim.

| Arm | PSNR all | PSNR missing | SSIM | Relative error |
|---|---:|---:|---:|---:|
| CSC | 24.898718 | 21.974835 | 0.826290 | 0.130277 |
| SCSC | 25.160606 | 22.238586 | 0.834772 | 0.126455 |

## การแปลผลและสิ่งที่ยังไม่สำเร็จ

ผล seed42 ดีกว่า CSC ใน foreground AP +0.001524, all-class mIoU +0.001486 และ PSNR +0.526971 dB แต่ค่าเฉลี่ยผลต่าง foreground AP ของสาม seed คือ -0.000339 และ all-class mIoU +0.000095. จึงยังไม่พิสูจน์ว่า SCSC ชนะ CSC อย่างสม่ำเสมอ ไม่เลือกตัด seeds ที่แพ้ออก.

การปรับ features, sampling และ calibration ช่วยทั้งสอง arm การดีขึ้นจาก v1 จึงไม่ใช่ประโยชน์ของ supervision ทั้งหมด ดู signed-only/four-crop controls บน VAL เพื่อแยกผล. Task-driven dictionary มีผลเพิ่มเล็กน้อยเมื่อเทียบกับ CSC ที่ได้ engineering เหมือนกัน. ข้อมูลยังน้อยและ grayscale local dictionary ยังมีบริบท semantic จำกัด แม้ใช้ RGB เป็น classifier context.

แผนที่จุดสีใหม่มีไว้ทำให้ตรวจความต่างได้ชัด ไม่ใช่หลักฐานว่า SCSC ชนะทุกภาพ. จำนวนจุดลดอาจเกิดจาก recall ลด ต้องอ่าน dense metrics และ per-class results ร่วมกัน. ตาราง/ภาพที่ SCSC แพ้ยังแสดงครบ.


### ข้อจำกัดของการแยกสาเหตุ

- K4_g0_N10: warmstarted unrolled/production relative code gap = 15.47%. การเอา label ออกจาก encoder แก้การพึ่ง label โดยตรง แต่ความคลาดเคลื่อนจาก finite-step differentiation และ production stopping ยังเหลือ ไม่อ้างว่า train/inference codes ตรงกันสมบูรณ์.
- K4_g0.05_N10: warmstarted unrolled/production relative code gap = 15.65%. การเอา label ออกจาก encoder แก้การพึ่ง label โดยตรง แต่ความคลาดเคลื่อนจาก finite-step differentiation และ production stopping ยังเหลือ ไม่อ้างว่า train/inference codes ตรงกันสมบูรณ์.

CSC ที่มี all-class VAL mIoU สูงสุดใน recipe/K/control search ที่ seed42 คือ K4_g0_N10_beta005 (0.242769). คู่หลักเป็นการควบคุม hyperparameters ร่วมกันตาม recipe ที่เลือกให้ SCSC ไม่ใช่การยืนยันว่าชนะ CSC ที่ tune แยกอย่างอิสระทุก setting.

SCSC มี Adam step เพิ่มที่ประกอบด้วย reconstruction และ classification loss ขณะที่ CSC หลักใช้ generative projected FISTA อย่างเดียว จึงไม่ใช่ compute-matched supervision-only ablation; ห้ามตีความ PSNR ที่เพิ่มทั้งหมดว่าเกิดจาก labels. Loss control เปรียบเทียบ weighted softmax CE กับ unweighted mean OVA BCE ซึ่งมีหน่วยต่างกัน ผลจึงเป็นการเปรียบเทียบ pipelines ที่ประกาศ ไม่ได้แยกผล link function เพียงอย่างเดียว.


## การตรวจสอบและข้อจำกัด

ตรวจ 57 โมเดล, original images 18, stored native predictions 201, paired inpainting 10, PNG 168. Artifact consistency=True. Training subproblems ไม่ผ่าน stopping criterion 132 รายการ; inference images ที่มี batch ไม่ผ่าน 0 รายการ. เก็บ failures ไว้ทั้งหมด ไม่เปลี่ยนชื่อเป็น convergence.

การตรวจ artifacts คือคำนวณ metrics ใหม่จาก saved arrays และตรวจ source hashes/shape/checkpoints ไม่ใช่ retrain ทุกโมเดลอิสระ. FISTA converged mapping ไม่รับรอง nonconvex outer stationary point หรือคำตอบ global. 24-step differentiable encoder ใช้ warmstart จาก label-free code และเป็น truncated differentiation approximation.

ภาพ COCO เพิ่มเติมถูก proxy CONNECT 403 ปฏิเสธระหว่างงานนี้. จึงไม่ได้รัน N>10, independent new TEST และ other-dataset inpainting. Existing TEST เคยเปิดใช้ในงานก่อนหน้าและผู้พัฒนาเคยเห็นผลแล้ว; ไม่อ้าง unseen generalization แม้เลือก v2 บน VAL เท่านั้น. RGB dictionary ยังไม่ได้ใช้; RGB เป็น classifier context ที่แชร์ทั้งสอง arm.

ไม่รับประกัน SCSC ชนะ CSC ทุกคลาสหรือ reconstruction. ตารางที่แพ้และคะแนนลบยังคงแสดงครบ. Source base คือ research/scsc-coco-final commit 80a44dd; โค้ดและผล v2 แยก branch research/scsc-coco-v2. คำสั่งรันครบอยู่ README_V2.md.


## สถานะ solver ของคู่ที่เลือก

| Model | Z converged | Generative D converged | Outer budget |
|---|---:|---:|---:|
| K4_g0_N10 | 30 | 30 | 30 |
| K4_g0.05_N10 | 30 | 30 | 30 |

ตัวเลขนี้เป็นการหยุดของ inner generative solves ก่อน task update ไม่ใช่หลักฐานว่า Adam หรือ nonconvex outer optimization ถึงจุด stationary. Field classifier ใน history เป็น bootstrap-head diagnostic; task head ใช้ fixed Adam steps ส่วน fresh-head success อ่านจาก completed.json.
