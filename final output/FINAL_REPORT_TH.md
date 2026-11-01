# รายงานการทดลองใหม่ SCSC COCO Final

วันที่จัดทำ: 2 พฤศจิกายน 2026 (Asia/Bangkok)

**สถานะ: รันการทดลองและตรวจความสอดคล้องของไฟล์แล้ว แต่ไม่ใช่ Fully Verified Final สำหรับการพิสูจน์บนข้อมูลใหม่อิสระ**

## 1. Project objective

ทดสอบสมมติฐานว่า supervision ช่วยให้ convolutional dictionary แยก Person, Car, Motorcycle และ Background ได้ดีขึ้น พร้อมรักษาคุณภาพ reconstruction โดยยอมรับผลลบตามจริง ไม่สร้างผลให้ SCSC ต้องชนะ

## 2. Related papers

อ่าน PDF ทั้งสองไฟล์ใน references/: Affara, Ghanem & Wonka, *Supervised Convolutional Sparse Coding*, arXiv:1804.02678v1 และ Wohlberg, *SPORCO*, SciPy 2017 อ่านสมการ 1–10, Algorithms 1–2, Sections 5.1–5.4 และ Figures 3–6 / Tables 1–2 เปรียบเทียบกับโค้ดใน PAPER_AUDIT.md มี SHA256 ของเอกสารใน environment.json

## 3. Dataset description

มีภาพ JPEG COCO ต้นฉบับ 18 ภาพ ตรวจ checksum และขนาดภาพครบทุกภาพ ใช้ pycocotools rasterizer มาตรฐานแทนการปัดพิกัด polygon ด้วย OpenCV จำนวน TRAIN/VAL/previously-seen TEST คือ 10/3/5 ภาพ ดู image IDs, dimensions, ignore count และจำนวนพิกเซลแต่ละคลาสใน experiment_configs/dataset_audit.json

Background คือบริเวณที่ไม่ใช่สามคลาสเป้าหมาย ซึ่งรวมวัตถุคลาสอื่นและบริเวณไม่มี annotation ตามนิยามงาน ไม่ใช่การรับรองว่ามี annotation ครบทุกสิ่ง Ignore เฉพาะ target-class crowds และบริเวณซ้อนกันของคนละ target class ไม่ใช้ GT แก้ prediction

**ข้อจำกัด holdout:** TEST ทั้ง 5 ภาพเคยถูกดูในโครงการเดิม จึงไม่ใช่ independent unseen test การตรวจการเข้าถึง images.cocodataset.org ถูก proxy ปฏิเสธ CONNECT HTTP 403 ไม่ได้ดาวน์โหลดภาพหรือ annotation เพิ่ม N=20/25/40 และ holdout ใหม่จึงเป็น `not run` ไม่ซ้ำภาพเพื่อเพิ่ม N

## 4. Mathematical formulation

`0.5 sum(M*(D*Z-X)^2) + beta*sum(abs(Z)) + gamma*(sum(log(1+exp(-Y*(WZ+b)))) + alpha*(||W||²+||b||²))`, ภายใต้ dictionary compact support และ filter norm <=1 ใช้ loss แบบ sum, beta=.015, alpha=1, filter 11x11 บน grayscale [0,1] highpass; beta ต่างจาก .5 ใน paper เพราะการเตรียมสัญญาณ/สเกลต่างกัน ไม่อ้างว่าเป็นค่าเดียวกัน

## 5. CSC vs SCSC architecture

CSC ใช้ gamma=0 ไม่มี supervised gradient ใน sparse coding ส่วน SCSC มี logistic loss ใน Z-step จริง ใช้ four one-vs-all heads ตามแนว multiclass extension ของ paper เพิ่ม Background เป็น head ที่สี่ ไม่ใช้ softmax ทั้งสอง arm เริ่ม dictionary seed เดียวกัน ใช้ภาพ/crop/labels เดียวกันและฝึก evaluation classifier ใหม่แยกกันจาก label-free TRAIN codes ของ dictionary แต่ละตัว ตาม Section 5.2

Internal channel order = Person, Car, Motorcycle, Background; public class IDs ใน NPZ prediction = Background 0, Person 1, Car 2, Motorcycle 3 OVA sigmoid probability ไม่จำเป็นต้องรวมเป็น 1 Decision rule เป็น argmax logits เหมือนกันทั้งคู่ ไม่มี smoothing, offset calibration หรือ threshold tuning

## 6. ADMM optimization

FFT scaled ADMM แยก reconstruction / l1 / logistic prox ใช้ Newton สำหรับ logistic prox และ Woodbury สำหรับ frequency systems อัปเดต Z → global bank-wide D → theta บน supervised Z เดียวกัน แก้ข้อบกพร่องเดิมที่ใช้ EMA dictionary และ refit theta บน label-free codes ระหว่างฝึก การ fit theta ใช้ L-BFGS แทน gradient descent เป็น solver substitution ของ objective เดียวกัน

แก้ boundary mask ให้ตรงกับ causal top-left kernel anchor: ตัด first f-1 rows/columns ที่ wrap ไม่ใช้ symmetric f//2 ที่ยังเหลือ circular artifacts ตรวจเทียบกับ direct linear convolution และ adjoint identity D ที่คืนต้องเป็น compact feasible dictionary; หากผล feasible ทำให้ objective แย่กว่าเดิม จะยังไม่ยอมรับว่า converge และป้องกันการคืน dictionary ที่แย่ลง

## 7. Experimental setup

CPU quota 4 cores, RAM 32 GiB, ไม่มี GPU ใช้ BLAS หนึ่ง thread ต่อ subprocess, training seed=42 จำนวน outer iterations=10 ทุก candidate, atol=1e-4, rtol=.003, max ADMM=1000 และ residual-balanced adaptive rho ไม่ผ่อน tolerance เพื่อให้ผ่าน

Runtime เป็น process wall time ใน Cloud ที่มีงานขนาน ไม่ใช่ controlled method-speed benchmark กรณี resume รวม process segments และค่า restart/งานที่ยกเลิก/การรอ lock ตาม resume_event.json จึงไม่ใช่เวลาคำนวณสุทธิหรือ end-to-end latency ที่ใช้เปรียบเทียบความเร็ว CSC/SCSC โดยตรง เก็บ D/theta checkpoint เดิมและ reinitialize Z ก่อนแก้ convex Z-step เดิม source/retained-prefix hashes ทำให้ตรวจได้ว่ารอบก่อนหน้าไม่ได้ถูกเขียนทับ

**Native resolution:** ใช้ RGB ต้นฉบับสำหรับแสดงผล แปลง grayscale สำหรับ sparse optimization โดยไม่มี resize ฝึกสี่ intact 64x64 native crops/ภาพ ใช้ global D ร่วมทั้ง tile bank ไม่ใช่ global full-frame D การ inference เป็น overlapping tiles ที่มี reflected context และ blend เฉพาะ valid interior ครอบคลุมทุกพิกเซลในขนาดต้นฉบับ

Full-frame feasibility pilot: TRAIN 10 ภาพ pad เท่านั้นเป็น 640x640, K=4; Z 10 ขั้นใช้ 36.58 วินาที และ D 10 ขั้น 7.64 วินาที ทั้งคู่ยังไม่ converge เป็น `inconclusive` pilot ไม่ใช่ผลฝึกสำเร็จ รายงานนี้จึงไม่เรียกการทดลองหลักว่า exact full-image optimization

## 8. Hyperparameter search

Gamma grid: [0, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 5, 10, 100]; K grid: [4, 8, 12, 16, 32, 49, 100]; N grid: [1, 5, 10]

เลือก gamma บน VAL ที่ K=4 แล้วเลือก K ด้วย gamma นั้น Criterion คือ maximum pooled foreground AP; ไม่เลือกจากผลต่าง SCSC−CSC ให้เป็นบวก ไม่เปิด TEST เพื่อเลือกระหว่าง candidates ผลสุดท้าย: **K=4, gamma=2, N=10** N เป็น nested learning curve ใช้ VAL/TEST ชุดเดิม ไม่ใช้ N เพื่อเลือกภาพที่ทำให้ TEST ดีขึ้น

Controls: beta=.015, alpha=1, 11x11 filters, four native tiles/image, 24 positive + 24 negative anchors/head/tile เมื่อมีเพียงพอ N learning curve คง K=4 และใช้ gamma ที่ VAL เลือก ไม่ได้ทำ N sweep ซ้ำทุก K หรือสรุปว่าแนวโน้ม N ที่ K=4 ใช้แทน K=100 ได้ การ sweep gamma เป็น supervision ablation หลัก ไม่ได้รันเต็ม Cartesian grid ของ beta/alpha/filter21/seeds หรือ background-head ablation ให้ถือส่วนเหล่านี้ว่า `not run`

| N (K=4 fixed, VAL) | CSC foreground AP | SCSC foreground AP | CSC PSNR dB | SCSC PSNR dB |
|---|---:|---:|---:|---:|
| 1 | 0.163005 | 0.165858 | 39.607793 | 38.344712 |
| 5 | 0.165092 | 0.163080 | 40.580565 | 38.601013 |
| 10 | 0.163634 | 0.165282 | 40.584077 | 38.114173 |

จำนวน labeled anchor entries (positive+negative) ตามลำดับ Person/Car/Motorcycle/Background: N=1: [144, 48, 192, 144]; N=5: [528, 580, 768, 912]; N=10: [1080, 1152, 1536, 1872]. Positive/negative สมดุลภายในแต่ละ head เมื่อมีข้อมูล แต่ยอดรวมระหว่าง heads ต่างกันตาม coverage ไม่ตีความ N curve ว่าแยกผลจำนวนภาพจากความหลากหลายของภาพ/สัดส่วนคลาสได้สมบูรณ์

ขอบเขตพิกเซล TRAIN ที่สุ่มจริงและ unique reconstruction-support pixels หลัง boundary mask อยู่ tables/native_training_coverage.csv แยก raw crop area กับ data-term area; lowpass preprocessing อ่านภาพต้นฉบับเต็ม แต่ dictionary optimization ใช้เฉพาะ tile bank

## 9. Figure 3 results

figures/figure3_dictionaries แสดง filters ทั้งหมดของทุก K ที่ทดลองคู่กัน CSC ซ้าย / SCSC ขวา ใช้ symmetric grayscale range เดียวกันในแต่ละคู่ ไม่เลือกเฉพาะ filters สวย มี dictionary evolution ที่ outer 1/5/10 และ supplementary absolute filter-correlation ใน tables/dictionary_diversity.csv ความคล้ายเชิง semantic จากรูปอย่างเดียวไม่ใช่หลักฐานการจำแนก ต้องดู AP/IoU ร่วมกัน

## 10. Figure 4 results

แสดงครบ TEST ทั้ง 5 ภาพทุกคลาส มี 4x4 composite, 4 panels แยกรายคลาส, dense OVA probabilities ในช่วง 0–1 ร่วมกัน, binary argmax masks, multiclass maps และ original RGB/GT ไม่มีการใช้ GT ลบ false positives สีของ probability ไม่ได้ปรับแยกเพื่อทำให้ SCSC ดูเด่น ผลภาพและ metrics มาจาก NPZ prediction เดียวกัน

## 11. Figure 5 results

figures/figure5_ap_psnr มี AP/PSNR vs gamma/K/N 6 panels รวมและแยก, macro และรายคลาส ทุกค่ามาจาก VAL เนื่องจากใช้เลือกโมเดล ไม่ใช่หลักฐาน independent test CSC ไม่ขึ้นกับ gamma จึงเป็น horizontal baseline แกน gamma ใช้ symlog เพื่อแสดงช่วงใกล้ศูนย์และช่วง 100 ได้ชัด Numeric source อยู่ tables/figure5_6_numeric_source.csv

## 12. Figure 6 results

figures/figure6_class_psnr วัด reconstruction PSNR เฉพาะ native GT region ของ Person/Car/Motorcycle แยกจริง ไม่ใช้ global PSNR เดิมซ้ำเป็นสามคลาส Class PSNR aggregate เป็น mean ต่อภาพที่มีคลาสนั้น, global PSNR เป็น mean ต่อภาพ; GT ใช้เฉพาะ evaluation

## 13. Tables 1–2 results

Table 1 ใน tables/table1_gamma_ap_psnr แสดง pooled pixel AP (0–1), foreground macro AP, global และ class-region PSNR (dB) ของทุก gamma บน VAL พร้อม transposed layout TEST แสดงเฉพาะคู่ที่เลือกแล้ว ไม่มี TEST sweep เพื่อเลือก gamma

Table 2 ใน tables/table2_inpainting แสดง PSNR ทั้งภาพและเฉพาะ missing pixels, SSIM, relative error ต่อภาพและค่าเฉลี่ย พร้อม figures/inpainting ไม่มี other-dataset table ที่สมมติขึ้น

## 14. Multiclass evaluation — previously-seen TEST

| Metric | CSC | SCSC | SCSC−CSC |
|---|---:|---:|---:|
| Foreground pooled macro AP | 0.155174 | 0.157368 | 0.002194 |
| All-class pooled macro AP | 0.251167 | 0.252381 | 0.001214 |
| Foreground mIoU | 0.070237 | 0.104124 | 0.033887 |
| All-class mIoU | 0.095848 | 0.098699 | 0.002850 |
| Pixel accuracy | 0.204291 | 0.224117 | 0.019826 |
| Background FPR (lower better) | 0.795725 | 0.910607 | 0.114881 |
| Global PSNR dB | 40.821456 | 38.566210 | -2.255246 |
| SSIM | 0.983250 | 0.975794 | -0.007456 |
| Relative reconstruction error (lower better) | 0.020561 | 0.026693 | 0.006132 |

| Class | CSC AP | SCSC AP | CSC IoU | SCSC IoU | CSC precision / recall | SCSC precision / recall |
|---|---:|---:|---:|---:|---:|---:|
| Person | 0.106370 | 0.106173 | 0.096786 | 0.053147 | 0.103377 / 0.602852 | 0.105062 / 0.097112 |
| Car | 0.101216 | 0.101439 | 0.052194 | 0.059265 | 0.096688 / 0.101867 | 0.095630 / 0.134836 |
| Motorcycle | 0.257938 | 0.264493 | 0.061732 | 0.199960 | 0.276652 / 0.073614 | 0.227476 / 0.623075 |
| Background | 0.539144 | 0.537417 | 0.172681 | 0.082423 | 0.527527 / 0.204275 | 0.513868 / 0.089393 |

F1, support, TP/FP/FN, full confusion และ normalized confusion เก็บใน metrics JSON/CSV มี precision–recall curves และ IoU comparison chart ตัวเลข AP รวมพิกเซลทั้ง split; AP ต่อภาพแยกเก็บใน per_image ไม่สลับนิยามสองแบบนี้

## 15. Confusion matrix analysis

Confusion เป็น GT row / predicted column มีทุกทิศทาง Person↔Car↔Motorcycle, foreground→Background และ Background→foreground ไม่ซ่อนค่า zero หรือ false positives ภาพ heatmap ใช้ลำดับ Background, Person, Car, Motorcycle ส่วน raw JSON ใช้ internal score order ตามระบุ

Foreground-only cross-class confusion rate: CSC=0.571393, SCSC=0.507039. Background→foreground FPR: CSC=0.795725, SCSC=0.910607. Class confusion rate รวม (1−pixel accuracy): CSC=0.795709, SCSC=0.775883.

## 16. Reconstruction and inpainting

Reconstruction ใช้ grayscale highpass sparse representation บวก lowpass ที่เก็บจากภาพอินพุตตามแนว SPORCO ไม่ใช่ RGB reconstruction Lowpass มีส่วนทำให้ global PSNR สูง ต้องไม่อ้างว่า PSNR ทั้งหมดเกิดจาก dictionary เพียงอย่างเดียว

Inpainting ใช้ floor(50% pixels) หายแบบสุ่ม seed=42+image_id และ mask เดียวกันทั้งสอง arm Gaussian normalized interpolation สำหรับ lowpass ใช้เฉพาะ observed pixels ส่วน sparse coding มี observed mask อยู่ใน data term ไม่ใช้ lowpass จากภาพเต็มและไม่คัดลอกพิกเซลที่หายกลับจากต้นฉบับ

| Inpainting metric (mean over 5 images) | CSC | SCSC |
|---|---:|---:|
| PSNR_all_db | 24.782673 | 24.420964 |
| PSNR_missing_db | 21.851995 | 21.513569 |
| SSIM | 0.824952 | 0.809768 |
| relative_error | 0.132063 | 0.137705 |

## 17. ADMM convergence analysis

| Selected model | Z converged / 10 | D converged / 10 | Fresh coding | Fresh classifier | Train seconds | Peak RSS MiB |
|---|---:|---:|---|---|---:|---:|
| CSC | 10/10 | 10/10 | True | True | 43.50 | 202.42 |
| SCSC | 10/10 | 10/10 | True | True | 103.06 | 221.24 |

ตรวจทั้งหมด 27 โมเดล พบ primary Z/D subproblems ที่ไม่ converge 19 รายการ, auxiliary coding ไม่ converge 0 รายการ, classifier success=False 0 รายการ และ inference/inpainting batches ไม่ converge 0 รายการ ดู verification.json และ convergence_runtime.csv การรันจบ (`completed`) ไม่เท่ากับทุก subproblem converge และไม่เท่ากับ nonconvex outer optimization ถึง stationary point

Residual histories มี primal, dual, tolerances, rho, objective ทุก 10 ADMM iterations เกณฑ์ pass ต้องผ่าน residual ทั้งสองจริง ตรวจ losses แยกก่อน/หลัง Z, D และ classifier การสลับ outer อาจมี objective แกว่งเล็กน้อยจากการแก้ subproblem แบบ tolerance-limited ไม่อ้าง monotonic/global convergence เพียงเพราะครบ 10 รอบ

มี outer validation diagnostic ครบ 10 checkpoints ของทั้งสอง arm โดยใช้ native centre crop 64x64 คงที่หนึ่ง crop ต่อ VAL image รวม 3 crops และใช้ training heads ของ checkpoint นั้น ไม่ใช่ fresh evaluation heads กราฟนี้มีไว้ตรวจแนวโน้มระหว่างฝึกเท่านั้น AP ข้ามคลาสที่ไม่มี positive pixels ใน subset ตามที่ระบุชื่อ field; class support/coordinates/encoding convergence เก็บไว้ใน outer_validation_training_head_sampled.json ไม่ใช้กราฟ sampled นี้แทน full-VAL final metrics หรือเลือกโมเดล

## 18. Comparison with original paper

ทำตาม logistic supervision ใน Z, compact unit-ball dictionaries, FFT ADMM, alternating optimization และ fresh evaluation classifiers โดยตรง ดัดแปลง COCO เป็น four-class OVA พร้อม Background, sampling native tiles, grayscale low/high separation, reflected tiled inference และ L-BFGS แทน GD ไม่ใช่ exact full-image optimization และไม่ใช่ benchmark เดียวกับ Graz/ECP/LabelMe/Cat ใน paper จึงเปรียบเทียบแนวโน้ม ไม่เปรียบเทียบตัวเลขต่าง dataset ราวกับเป็นการทดลองเดียวกัน

## 19. Limitations and failed / unrun experiments

- Independent new holdout: `not run`, ไม่มีข้อมูลใหม่และ original COCO host ถูกปฏิเสธการเข้าถึง
- N=20/25/40: `not run`, จำนวน TRAIN ที่มีจริงสูงสุด 10 ไม่เพิ่ม N ด้วยการใช้ภาพซ้ำ
- Other-dataset / out-of-domain inpainting: `not run`, ไม่มี dataset ที่เข้าถึงได้และมี provenance เพียงพอ
- Full-frame training: `inconclusive` feasibility pilot; ใช้ native-tile approximation สำหรับผลหลัก
- RGB dictionary, filter21 sweep, beta/alpha sweep, multiple training seeds, background-head ablation: `not run`; ควบคุมค่าหลักเดียวกัน ไม่ได้ทำ full Cartesian grid
- Nonconverged primary ADMM subproblems: 19 รายการ ไม่เปลี่ยน tolerance หรือเปลี่ยนชื่อเป็นผ่าน
- มีภาพ TEST เพียง 5 ภาพซึ่งเคยใช้แล้ว Bootstrap 2,000 paired image draws ใน paired_bootstrap.json วัดความแปรปรวนระหว่างภาพเท่านั้น ไม่มี training-seed variability และไม่อ้าง statistical significance ต่อประชากร

## 20. Conclusion

บน previously-seen TEST, SCSC มี IoU เพิ่มใน: Car, Motorcycle; IoU ลดใน: Person, Background. Foreground macro AP เปลี่ยน +0.002194, foreground mIoU เปลี่ยน +0.033887, global PSNR เปลี่ยน -2.255246 dB.

ผลนี้ตอบได้เฉพาะพฤติกรรมของการทดลองขนาดเล็กบนข้อมูลชุดนี้ ไม่พิสูจน์ความเหนือกว่าทุกคลาส ไม่รับรอง generalization บน independent unseen COCO และไม่รับรองว่า activation maps ดูดีกว่าเสมอ การทดสอบ artifact consistency ผ่านหมายถึงไฟล์/ตัวเลขสอดคล้องกัน ไม่ได้แปลว่าสมมติฐานทางวิจัยผ่าน

## Verification and provenance

ตรวจภาพต้นฉบับ 18 ภาพ, native predictions ทั้งหมด 91 ชุด (รวมคู่ final 16 ชุด), paired inpainting 10 ชุด และ PNG 124 ไฟล์ อ่านค่าใหม่จาก NPZ เพื่อเทียบ confusion/AP/PSNR กับ JSON พร้อม SHA256 inventory ไฟล์ใหม่ทั้งหมด ไม่มีการใช้ผลทดลองเก่าเป็นผลใหม่

Source branch: `research/scsc-coco-final`; baseline HEAD และ package versions อยู่ environment.json; source commit อยู่ verification.json ไฟล์ ZIP ใหม่รวม source, data, PDFs และผลใหม่เท่านั้น ไม่รวม virtualenv/cache/unused legacy scripts/pilot checkpoints

## Solver failure recovery and reruns

การทดลอง SCSC รุ่นแรกของงานนี้ถูกเก็บไว้ใน failed_attempts/prox_v1 และไม่ใช้เลือกโมเดลหรือเป็นผล final สาเหตุคือ undamped Newton logistic prox สามารถแกว่งเป็นสองจุด เช่น a=−8, gamma=1, rho=1/16 ทั้งที่รากจริงเป็นศูนย์ เพิ่มเพดาน ADMM เป็น 5,000 ก็ไม่แก้ การแก้ใช้ monotone root bracket พร้อมรับ Newton step เฉพาะเมื่อ residual ลดอย่างน้อยครึ่งหนึ่ง มิฉะนั้น bisection และตรวจ stationarity ทุกครั้ง

K=100 SCSC รุ่นแรกเกิด overflow ใน D-step จาก Gram/Woodbury cancellation แบบ float32 การเปลี่ยน Gram เป็น float64 อย่างเดียวยังไม่ผ่าน high-coherence regression จึงใช้ complex128 spectral linear solve จาก SVD ของ Z โดยตรง ซึ่งคง identity และ nullspace ได้เสถียรกว่า มี numerical regression เทียบ Brent root oracle, independent epigraph optimizer และกรณี Gram ill-conditioned รองรับ SCSC ทุกรุ่นที่ใช้เป็นผล primary ถูกฝึกใหม่หลังแก้ ไม่ใช้ผลรุ่นที่ผิดมาสรุปว่า paper สำเร็จ

Baseline gamma=0 ที่เก็บไว้ไม่มี logistic prox และผ่าน finite-state/residual/feasible-descent checks; ใช้สมการและ tolerance เดียวกัน แต่บาง baseline รันก่อน precision/FFT refactor จึงไม่ได้อ้าง bitwise-identical execution ระหว่าง implementation revisions รายละเอียด source hashing ของงานใหม่อยู่ software.json ส่วน early baseline ไม่มี per-job source hash ที่เก็บทัน จึงเป็นข้อจำกัด provenance ที่ระบุไว้ตรงนี้

เปรียบเทียบ K=4, gamma=1 first alternating step: รุ่นเดิม maxiter=5000 Z-converged=False; รุ่นแก้ maxiter=1000 Z-converged=True ใน 210 ขั้น ไม่ย้ายผล diagnostic มาแทนผลฝึก 10 outer iterations

## Additional penalty diagnostic (not adopted)

ทดสอบ K=32, gamma=2 first D-step ด้วย native TRAIN bank เดิม ลด adaptive rho floor จาก 1/64 เป็น 1e-6 โดยคง maxiter=1000 และ atol/rtol เดิม ไม่ใช้ TEST ในการทดลองนี้

| Rho floor | D converged | Final primal / tolerance | Final dual / tolerance | Candidate feasible objective |
|---|---|---:|---:|---:|
| 1/64 | False | 0.241667 | 10.887269 | 21.992065 |
| 1e-6 | False | 1.802281 | 7.217176 | 149.441025 |

ทั้งสองแบบยังไม่ผ่านและ candidate ที่ทำให้ objective แย่ลงถูกปฏิเสธ จึงไม่ใช้ setting ใหม่ใน primary sweep ผลนี้แสดงว่าเพิ่มช่วง rho อย่างเดียวไม่แก้ conditioning/finite-budget convergence ทุกกรณี ไม่ผ่อน tolerance เพื่อเปลี่ยน failure ให้เป็น pass

Implementation provenance: ระหว่างพัฒนาได้ลด redundant FFT โดย reuse zf/dhat ตามสมการเดิม คง objective และ stopping criteria เดิม มี independent scalar/high-K epigraph optimizer tests รองรับ เฉพาะ baseline CSC K=100 ที่เริ่มก่อน optimization ใช้รุ่น FFT เดิมต่อจนจบ ส่วน SCSC รุ่นผิดถูกเก็บเป็น failure evidence และฝึกใหม่; ไม่เรียกผลจาก profiler ว่าเป็น training result ดู profiling_summary.json และ software metadata โดยไม่อ้าง bitwise identity ระหว่าง implementation revisions
