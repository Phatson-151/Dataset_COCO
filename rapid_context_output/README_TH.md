# ผลการปรับปรุง SCSC ภายในกรอบเวลา 30 นาที

เริ่มอ่าน REPORT_TH.md และ verification.json

เป็น workflow ใหม่: dictionary เดิมที่ตรึงไว้ + contextual multiclass classifier ที่ฝึกใหม่ + การเลือก supervision จาก VAL + spatial probability refinement ที่เลือกจาก VAL ทั้ง CSC/SCSC ได้ขั้นตอนและงบ classifier แบบเดียวกัน

- ผลหลักและ comparison ที่ปรับเหมือนกัน: spatial_refinement/results.json
- ผลก่อน spatial refinement: results.json และ gamma_reselection/results.json
- ภาพทุกคลาส: figures/spatially_refined_activation_*.png
- ภาพ raw probabilities ของ classifier ใหม่: figures/activation_all_classes_*.png และ gamma_selected_activation_*.png
- Raw/refined probabilities และ predictions: ไฟล์ NPZ ใน folder ของแต่ละวิธี
- TRAIN sampling, protocol, hashes และผล bootstrap ต่อภาพ: JSON ที่แนบ
- Logs การฝึกตัวจำแนกและการประเมิน: logs/*.txt

คำว่า raw ในชุดนี้หมายถึง output ของ contextual classifier ก่อน spatial smoothing ไม่ใช่ linear classifier ใน paper

สำหรับรันซ้ำใน repository/ZIP ที่มี data และ frozen dictionaries:
1. ติดตั้ง requirements-experiment-lock.txt
2. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python rapid_context.py
3. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python rapid_gamma.py
4. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python rapid_spatial.py
5. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python report_rapid_context.py

ไม่มีการฝึก dictionary ใหม่หรือเปลี่ยนผลใน final output/ ไม่มีการ commit/push/publish งานรอบนี้ ผล TEST มาจากภาพเดิม 5 ภาพที่เคยใช้แล้ว จึงยังไม่ใช่หลักฐาน generalization บนข้อมูลใหม่อิสระ
