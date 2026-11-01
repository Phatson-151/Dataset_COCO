# คู่มืออ่านผล / สถานะงานวิจัย

## จุดสำคัญที่สุด

โครงการนี้ไม่ย่อภาพ COCO: ภาพต้นฉบับถูกตรวจ checksum และใช้งานบนพิกัดเดิมจริงทั้งหมด แต่การฝึกแบบใช้ทรัพยากรจำกัด **สุ่มเลือกภาพตัดย่อยจากภาพต้นฉบับ** (4 crops ต่อ 1 ภาพ TRAIN รวม 40 crops) ไม่ใช่การ optimize ทั้ง 640×427 pixels พร้อมกันหรือใช้ทุก crop ของทุกภาพในการฝึก

- **TRAIN:** ภาพ COCO 10 ภาพ; ตัด patch 64×64 พิกเซลจริง, ใช้ mini-batch online D-step
- **VAL:** ภาพ 3 ภาพ ขนาดจริง เพื่อเลือก K และปรับ offsets ของ logits แต่ละโมเดล
- **TEST:** ภาพ 5 ภาพขนาดจริง แต่ชุด TEST เคยถูกเปิดดูในเวอร์ชันก่อน จึงเรียก `previously_seen_TEST` ไม่ใช่ pristine held-out test
- **CSC:** sparse codes ที่ใช้ฝึก Dictionary ไม่รับ logistic loss (gamma=0)
- **SCSC:** logistic supervised coding มี 4 OVA heads คือ Person, Car, Motorcycle, Background
- **Fresh classifier:** ทั้งคู่เรียนใหม่บน TRAIN ตาม sparse codes ของ Dictionary ของตัวเอง เพื่อทดสอบคุณภาพของ Dictionary ไม่ได้ใช้ weights ระหว่าง training มาอ้างเปรียบเทียบโดยตรง
- **Metrics:** foreground mIoU, class IoU/precision/recall/AP, confusion 4×4, activation peak precision, PSNR, ADMM convergence

## ข้อจำกัดที่ต้องกล่าวในรายงาน

1. Global D-step ของต้นฉบับถูกแทนด้วย online tiled minibatch D-step จึงเป็น **approximation**
2. มีเพียง 10/3/5 ภาพและ VAL ใช้ปรับ model/calibration — ไม่เหมาะที่จะสรุปผลทางสถิติทั่วไป
3. TEST เคยถูกใช้ในงานเวอร์ชันก่อนแล้ว จึงจำกัดความเป็นอิสระ
4. Convergence ต้องอ่านตามจำนวน batch ที่ converge จริงจากไฟล์ train_logs ไม่ใช่ดูแค่จำนวน outer ที่รันจบ
5. คะแนน foreground purity ที่นับผิดคลาสเป็น foreground ได้ ไม่ถูกใช้แทน class-specific IoU
6. ผล SCSC ไม่ได้รับการคัดหรือตัดทิ้งตามความสวยงามของภาพ และไม่มีการใช้ GT มาช่วยกรองภาพที่ทำนาย

โปรดอ่านผลใหม่จาก `final output/FINAL_METRICS.json` และ `final output/figures/` เท่านั้น ไม่ใช่ผลใน ZIP เก่า
