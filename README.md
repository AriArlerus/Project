# Ultrasonic Sensor Calibration with Metaheuristic Algorithms

โปรเจคนี้เป็นงานปรับเทียบ (Calibration) ค่าระยะทางของเซนเซอร์อัลตราโซนิก `HC-SR04` เพื่อลดความคลาดเคลื่อนในการวัด โดยใช้อัลกอริทึมเมตาฮิวริสติก ได้แก่

- `PSO` (Particle Swarm Optimization)
- `POA` (Pelican Optimization Algorithm)
- `Hybrid POA-PSO`

---

## Objectives

- ลด Error ของค่าที่วัดได้จากเซนเซอร์เมื่อเทียบกับค่าระยะจริง
- เปรียบเทียบประสิทธิภาพของ `PSO`, `POA`, และ `Hybrid POA-PSO`
- ประเมินผลด้วย `MAE` และ `RMSE`

---

## Project Structure

- `Summary.py`  
  สคริปต์หลักสำหรับรันอัลกอริทึมทั้งหมด คำนวณผลเปรียบเทียบ และพล็อตกราฟสรุปผล
- `error_trend.py`  
  สคริปต์วิเคราะห์แนวโน้ม Error ก่อน Optimization (Quadratic trend + Huber trend)
- `Distance(CM) - SensorData.csv`  
  ชุดข้อมูลการทดลอง (Measured vs Desired)

---

## Dataset Columns

ข้อมูลที่ใช้ต้องมีคอลัมน์:

- `Measured (cm)` : ระยะที่เซนเซอร์วัดได้
- `Desired (cm)` : ระยะจริงอ้างอิง
- `Index` : ลำดับข้อมูล

---

## Installation

แนะนำ Python 3.10+ และติดตั้งไลบรารีดังนี้:

```bash
pip install numpy pandas matplotlib requests scikit-learn