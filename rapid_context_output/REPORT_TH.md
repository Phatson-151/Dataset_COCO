# ผลทดลองเร่งด่วน: contextual multiclass readout

## ผลสรุปรอบปรับปรุง

Workflow ที่ปรับแล้วใช้ frozen SCSC dictionary γ=0.2 + contextual multiclass classifier ที่ฝึกใหม่ + spatial refinement σ=31 ซึ่งเลือกด้วย VAL ทั้งหมด ไม่ได้ฝึก dictionary ใหม่ ผลต่อไปนี้เทียบกับ CSC ที่ได้ขั้นตอนใหม่เหมือนกัน

| Workflow | TEST foreground AP | TEST all-class mIoU |
|---|---:|---:|
| CSC เดิม | 0.155174 | 0.095848 |
| SCSC เดิม | 0.157368 | 0.098699 |
| CSC + contextual classifier + refinement | 0.248708 | 0.234874 |
| SCSC + contextual classifier + refinement | 0.265934 | 0.275133 |

ใน matched comparison นี้ SCSC มี pooled AP และ IoU สูงกว่า CSC ทั้งสี่คลาส แต่ยังมีข้อจำกัดสำคัญ: ภาพทำนายยังหยาบและไม่ตรงรูปร่างวัตถุดีพอ โดยเฉพาะ Person มี AP ต่ำทั้งคู่ และ Person AP ของ workflow ใหม่น้อยกว่าตัวจำแนกเดิม จึงห้ามสรุปว่าทุกด้านดีขึ้นจากรุ่นเดิม

ชนะด้าน per-image all-class mIoU 3/5 ภาพ; paired bootstrap 95% CI ของ mean per-image mIoU delta = [-0.013603093160647528, 0.10255945231327064]. ช่วงคร่อมศูนย์ จึงยังไม่ยืนยันความเหนือกว่าเชิงสถิติ TEST ทั้งห้าภาพเคยถูกดูมาก่อน ไม่ใช่ independent unseen test

ไม่มีการ commit/push/publish งานรอบนี้ ผลเดิมไม่ถูกแก้ไข รายงานและ ZIP เป็นงานในฉบับร่างเท่านั้น

งานนี้เป็นการฝึกตัวจำแนกใหม่บน dictionary เดิมที่ถูกตรึงไว้ ไม่ใช่การฝึก dictionary ใหม่ และไม่ใช่การทำซ้ำ paper แบบตรงตัว ไม่ได้เผยแพร่หรือ push การแก้ไขใด ๆ

## วิธีที่เปลี่ยน

- เพิ่ม TRAIN sampling สูงสุด 1,200 จุดต่อคลาสต่อภาพ ใช้จุดเดียวกันทั้ง CSC/SCSC รวม 10 ภาพ
- ใช้ signed/absolute sparse codes และค่าเฉลี่ยขนาด 5/17/49/97 พิกเซล เพิ่มบริบทโดยไม่ resize ภาพ
- ทดสอบเพิ่ม RGB/Lab, สีเฉลี่ยหลายขนาด และ local texture variance; มี RGB-only control
- ใช้ HistGradientBoosting multiclass classifier แบบเดียวกัน: 120 iterations, 15 leaves, seed เดียวกัน
- เลือก decision bias ของ Background บน VAL แยกต่อโมเดลด้วย grid เดียวกัน ไม่ใช้ TEST labels เลือกค่า
- เลือก feature family **rgb_sparse_context** จากค่าเฉลี่ย VAL all-class mIoU ของทั้งสอง arm ไม่เลือกจากขนาดช่องว่าง SCSC−CSC

## ผล previously-seen TEST 5 ภาพ

| Metric | CSC เดิม | SCSC เดิม | CSC + contextual | SCSC + contextual |
|---|---:|---:|---:|---:|
| Foreground AP | 0.155174 | 0.157368 | 0.239515 | 0.249676 |
| Foreground mIoU | 0.070237 | 0.104124 | 0.172138 | 0.180032 |
| All-class mIoU | 0.095848 | 0.098699 | 0.219807 | 0.228003 |
| Pixel accuracy | 0.204291 | 0.224117 | 0.402840 | 0.418208 |
| Background FPR (lower better) | 0.795725 | 0.910607 | 0.593198 | 0.574424 |

| Class | CSC contextual AP | SCSC contextual AP | CSC contextual IoU | SCSC contextual IoU |
|---|---:|---:|---:|
| Person | 0.095967 | 0.094541 | 0.065215 | 0.063703 |
| Car | 0.226361 | 0.239040 | 0.172608 | 0.174521 |
| Motorcycle | 0.396219 | 0.415446 | 0.278591 | 0.301873 |
| Background | 0.739751 | 0.736281 | 0.362813 | 0.371916 |

## การแปลผล

SCSC−CSC contextual: foreground AP=+0.010160, foreground mIoU=+0.007894, all-class mIoU=+0.008196. ต้องแยกการดีขึ้นจากการเปลี่ยน classifier ออกจากการดีขึ้นจาก supervised dictionary ห้ามเทียบ SCSC รุ่นเพิ่มขั้นตอนกับ CSC รุ่นเดิมเพียงอย่างเดียวแล้วสรุปว่า supervision ชนะ
RGB-only control: foreground AP=0.231204, all-class mIoU=0.234230. ถ้าใกล้เคียงหรือดีกว่า ไม่ควรอ้างว่า sparse coding เป็นเหตุหลักของผลดีขึ้น

## ข้อจำกัด

ข้อมูลทั้งหมด 18 ภาพ แบ่ง TRAIN/VAL/TEST=10/3/5 ตามเดิม TEST เคยถูกดูมาก่อน จึงไม่ใช่ independent unseen evaluation; seed เดียวและยังไม่มีการยืนยันนัยสำคัญทางสถิติ ภาพยังเป็น native resolution แต่บริบทเฉลี่ยทำให้ขอบเบลอได้ การปรับ bias มีผลต่อ prediction แต่ไม่เปลี่ยน probability maps ที่แสดงหรือ AP ซึ่งคำนวณจาก raw classifier probabilities
ไม่ได้ฝึก dictionary เพิ่ม ไม่ได้ทำ reconstruction/inpainting ใหม่ เพราะการแก้รอบนี้เปลี่ยนตัวจำแนกเท่านั้น ผลการทดลองที่ไม่ชนะยังเก็บครบใน validation.json/results.json

## ตรวจสอบและรันซ้ำ

คำนวณ confusion/AP/IoU ซ้ำจากผลดิบ 30 ภาพแล้วผ่าน ตรวจ probabilities อยู่ใน [0,1] และรวมเป็น 1 ต่อพิกเซล; ตรวจการกู้ sparse codes จาก saved logits เทียบ fresh encoding บน VAL หนึ่งภาพใน code_recovery_check.json
ใช้ /workspace/.venvs/dataset-coco/bin/python rapid_context.py แล้ว report_rapid_context.py ใน repository เดิม Dependencies ใช้ requirements-experiment-lock.txt ผลอยู่ rapid_context_output/ ทั้งหมด ไม่มีการเขียนทับ final output/

เวลาของ main experiment: 302.4 วินาที (ไม่รวมการเขียนโค้ด/verification/figures). Protocol และ source/dictionary hashes อยู่ protocol.json

## เพิ่มเติม: เลือก supervision ใหม่สำหรับ contextual readout

กำหนด candidates γ=0.2/2/10/100 และเลือกจาก SCSC VAL all-class mIoU เท่านั้น ไม่เลือกจากผล TEST หรือ SCSC−CSC เปรียบเทียบกับ CSC ที่ใช้ RGB+context และ classifier configuration เดียวกัน

| Gamma | VAL foreground AP | VAL all-class mIoU |
|---:|---:|---:|
| 0.2 | 0.381976 | 0.323079 |
| 10 | 0.354615 | 0.297788 |
| 100 | 0.359499 | 0.312140 |
| 2 | 0.363806 | 0.303981 |

VAL เลือก **γ=0.2**. เป็นการเลือกใน workflow ใหม่ ไม่ได้เลือกจาก TEST ที่ดูดีกว่า

| Metric, previously-seen TEST | CSC + RGB/context | SCSC selected γ + RGB/context |
|---|---:|---:|
| foreground_AP | 0.239515 | 0.259358 |
| foreground_mIoU | 0.172138 | 0.184912 |
| all_class_mIoU | 0.219807 | 0.243202 |
| pixel_accuracy | 0.402840 | 0.453894 |
| background_FPR | 0.593198 | 0.510467 |

| Class | CSC AP | Selected SCSC AP | CSC IoU | Selected SCSC IoU |
|---|---:|---:|---:|---:|
| Person | 0.095967 | 0.104082 | 0.065215 | 0.067901 |
| Car | 0.226361 | 0.256078 | 0.172608 | 0.181761 |
| Motorcycle | 0.396219 | 0.417914 | 0.278591 | 0.305075 |
| Background | 0.739751 | 0.747509 | 0.362813 | 0.418072 |

Paired image bootstrap (10,000 resamples) ของค่าเฉลี่ย metric ต่อภาพ อยู่ใน gamma_reselection/per_image_and_bootstrap.json เป็นคนละนิยามกับ pooled AP/IoU ในตารางและใช้ภาพเพียง 5 ภาพ จึงไม่อ้างว่ายืนยัน generalization ได้

## เพิ่มเติม: Spatial consistency refinement

ใช้ Gaussian smoothing ของ probability maps โดย VAL เลือก sigma=31 พิกเซลจาก [0,3,7,15,31] โดยเลือกค่าเฉลี่ย all-class mIoU ของ CSC/SCSC ไม่เลือกช่องว่างระหว่างโมเดล ใช้ sigma เดียวกันทั้งสอง arm และปรับ Background decision bias ด้วย VAL grid เดียวกัน การปรับนี้เป็น postprocessing ที่ระบุชัด ไม่ใช่ raw activation และไม่ได้ใช้ GT ระหว่าง inference

| Metric, previously-seen TEST | CSC contextual + refinement | SCSC contextual + refinement | RGB-only + refinement |
|---|---:|---:|---:|
| foreground_AP | 0.248708 | 0.265934 | 0.241963 |
| foreground_mIoU | 0.174636 | 0.209890 | 0.204767 |
| all_class_mIoU | 0.234874 | 0.275133 | 0.237705 |
| pixel_accuracy | 0.431412 | 0.510663 | 0.420722 |
| background_FPR | 0.547815 | 0.454619 | 0.652175 |

| Class | Refined CSC AP | Refined SCSC AP | Refined CSC IoU | Refined SCSC IoU |
|---|---:|---:|---:|---:|
| Person | 0.082939 | 0.088974 | 0.039585 | 0.051561 |
| Car | 0.249316 | 0.276714 | 0.197978 | 0.221031 |
| Motorcycle | 0.413868 | 0.432113 | 0.286346 | 0.357078 |
| Background | 0.779529 | 0.784147 | 0.415587 | 0.470862 |
