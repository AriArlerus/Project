import numpy as np          # นำเข้า NumPy สำหรับการคำนวณเชิงตัวเลข เช่น array และคณิตศาสตร์
import matplotlib.pyplot as plt  # นำเข้า Matplotlib สำหรับวาดกราฟ
import pandas as pd         # นำเข้า Pandas สำหรับจัดการข้อมูลในรูป DataFrame
import io                   # นำเข้า io สำหรับแปลง bytes เป็น stream ที่ pandas อ่านได้
import requests             # นำเข้า requests สำหรับส่ง HTTP request ดึงข้อมูลจากเว็บ
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
# ============================================================
# ส่วนที่ 1: Load Dataset (ดึงจาก Google Sheets)
# ============================================================

SHEET_NAME = "SensorData"   # ชื่อชีตที่ใช้อ้างอิงในข้อความ log (ไม่ได้ใช้ใน URL โดยตรง)
SHEET_ID = "169b1I4Gos8UhkzDkxH6uX9ty3yaQ_8kCqRGjqnpb0dU"  # ID ของ Google Spreadsheet
GID = "1511238558"          # GID ระบุหมายเลขชีตย่อยภายใน Spreadsheet
sheet_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
# สร้าง URL สำหรับ export ชีตเป็นไฟล์ CSV โดยฝัง SHEET_ID และ GID ลงใน URL

try:
    print(f"[INFO] กำลังดึงข้อมูลจากชีต: {SHEET_NAME}...")  # แสดงข้อความว่ากำลังดึงข้อมูล
    response = requests.get(sheet_url, timeout=20)            # ส่ง GET request ไปยัง URL พร้อม timeout 20 วินาที
    response.raise_for_status()                               # โยน exception ทันทีถ้า HTTP status code เป็น error (4xx, 5xx)

    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    # แปลง bytes ที่ได้จาก response เป็น string UTF-8 แล้วอ่านเป็น DataFrame
    df.columns = df.columns.str.strip()
    # ลบช่องว่างหน้า-หลังชื่อคอลัมน์ทุกคอลัมน์ เพื่อป้องกัน KeyError จากช่องว่างแฝง

    required_cols = ["Measured (cm)", "Desired (cm)", "Index"]  # รายชื่อคอลัมน์ที่จำเป็นต้องมี
    for col in required_cols:               # วนตรวจสอบทุกคอลัมน์ที่กำหนด
        if col not in df.columns:           # ถ้าคอลัมน์นั้นไม่อยู่ใน DataFrame
            raise KeyError(f"ไม่พบคอลัมน์: {col}")  # โยน KeyError พร้อมบอกชื่อคอลัมน์ที่หาย

    df = df.sort_values(by="Index").reset_index(drop=True)
    # เรียงลำดับแถวตามคอลัมน์ Index และ reset index ให้เริ่มที่ 0 ใหม่
    print(f"[SUCCESS] โหลดข้อมูลสำเร็จทั้งหมด {len(df)} แถว")  # แสดงจำนวนแถวที่โหลดได้
    print(df.head())  # แสดง 5 แถวแรกของ DataFrame เพื่อตรวจสอบ

except Exception as e:                          # รับ exception ทุกประเภทที่เกิดขึ้นในบล็อก try
    print(f"[ERROR] ไม่สามารถโหลดข้อมูลได้: {e}")  # แสดงข้อความ error พร้อมสาเหตุ
    df = pd.DataFrame()                         # สร้าง DataFrame ว่างเพื่อให้โปรแกรมทำงานต่อได้โดยไม่ crash


# ============================================================
# ส่วนที่ 2: Pelican Optimization Algorithm (POA)
# ------------------------------------------------------------
# ค้นหาสัมประสิทธิ์สมการชดเชย (calibration equation)
#     Desired_pred = a * Measured^2 + b * Measured + c
# ใช้ Huber loss เพื่อให้ทนต่อ outlier
# ตัวแปรค้นหา m = 3 :  [a, b, c]
# ============================================================

class POA_Calibration:      # ประกาศ class สำหรับอัลกอริทึม POA เพื่อหาสมการ calibration
    def __init__(self, measured, desired,
                 n=50, m=3, T=200, R=0.2, huber_delta=6.0):
        # constructor รับพารามิเตอร์: ข้อมูล measured/desired และค่าควบคุมอัลกอริทึม
        self.measured = measured        # เก็บ array ค่าที่วัดได้จากเซนเซอร์
        self.desired  = desired         # เก็บ array ค่าที่ต้องการ (ค่าอ้างอิง)
        self.n = n                      # จำนวนนก (ประชากร) ในอัลกอริทึม
        self.m = m                      # จำนวนมิติของสารละลาย = 3 ([a, b, c])
        self.T = T                      # จำนวนรอบการวนซ้ำ (iterations)
        self.R = R                      # รัศมี exploration เริ่มต้นสำหรับ Phase 2
        self.huber_delta = huber_delta  # ค่า delta ของ Huber loss ที่แบ่ง quadratic/linear
        self.fitness_history = []       # list สำหรับเก็บค่า fitness ที่ดีที่สุดของแต่ละ iteration

        # ขอบเขตของ [a, b, c]
        # a ใกล้ 0 (เทอม non-linear เล็กน้อย), b ใกล้ 1, c คือ offset
        self.lb = np.array([-0.0003,  0.90, -5.0])  # lower bound ของแต่ละตัวแปร [a, b, c]
        self.ub = np.array([ 0.0003,  1.20, 12.0])  # upper bound ของแต่ละตัวแปร [a, b, c]

    def _huber(self, r):
        """Huber loss: quadratic เมื่อ |r|<=delta, linear เมื่อเกินกว่านั้น
        ทำให้ outlier (จุดที่ residual ใหญ่มาก) ไม่ครอบงำการ fit"""
        d = self.huber_delta            # นำค่า delta มาใช้งานภายในเมธอด
        absr = np.abs(r)                # คำนวณค่าสัมบูรณ์ของ residual ทุกจุด
        quad = 0.5 * r ** 2             # สูตร Huber แบบ quadratic: ½r² (ใช้เมื่อ |r| ≤ delta)
        lin  = d * (absr - 0.5 * d)     # สูตร Huber แบบ linear: δ(|r| - ½δ) (ใช้เมื่อ |r| > delta)
        return np.where(absr <= d, quad, lin).mean()
        # เลือกใช้ quad หรือ lin ตามเงื่อนไข แล้วคืนค่าเฉลี่ยของทุกจุดเป็น scalar

    def fitness_function(self, w):      # ฟังก์ชันวัดคุณภาพของสารละลาย w = [a, b, c]
        a, b, c = w                     # แตก array w ออกเป็นสัมประสิทธิ์แต่ละตัว
        pred = a * self.measured ** 2 + b * self.measured + c
        # คำนวณค่า predicted จากสมการ quadratic กับทุกจุดใน measured
        residual = pred - self.desired  # หา residual = ผลต่างระหว่าง predicted กับค่าจริง
        return self._huber(residual)    # คืน Huber loss ของ residual เป็นค่า fitness

    def run(self):                      # เมธอดหลักสำหรับรันอัลกอริทึม POA
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)
        # สร้างประชากรเริ่มต้น n ตัว แต่ละตัวมี m มิติ กระจายสุ่มภายใน [lb, ub]
        F = np.array([self.fitness_function(s) for s in X])
        # คำนวณค่า fitness ของทุกสมาชิกในประชากร เก็บเป็น array F

        self.best_fitness  = float(np.min(F))       # เก็บค่า fitness ต่ำสุดในประชากรเริ่มต้น
        self.best_solution = X[np.argmin(F)].copy() # เก็บสารละลายที่ให้ fitness ต่ำสุด
        self.fitness_history = [self.best_fitness]  # เริ่มต้น history ด้วยค่า fitness รอบแรก

        for t in range(1, self.T + 1):  # วนซ้ำ T รอบ (iteration ที่ 1 ถึง T)
            prey   = self.lb + np.random.rand(self.m) * (self.ub - self.lb)
            # สุ่มตำแหน่ง "เหยื่อ" (prey) ภายใน [lb, ub] เพื่อใช้เป็นเป้าหมาย exploration
            f_prey = self.fitness_function(prey)    # คำนวณ fitness ของตำแหน่งเหยื่อ

            for i in range(self.n):     # วนอัพเดตสมาชิกแต่ละตัวในประชากร
                # Phase 1: Exploration
                I = np.random.choice([1, 2])        # สุ่ม intensity factor I เป็น 1 หรือ 2
                if f_prey < F[i]:                   # ถ้าเหยื่อดีกว่าสมาชิกปัจจุบัน
                    X_p1 = X[i] + np.random.rand(self.m) * (prey - I * X[i])
                    # เคลื่อนที่เข้าหาเหยื่อตามสูตร Phase 1 (เหยื่อดีกว่า)
                else:                               # ถ้าสมาชิกปัจจุบันดีกว่าเหยื่อ
                    X_p1 = X[i] + np.random.rand(self.m) * (X[i] - prey)
                    # เคลื่อนที่ออกจากเหยื่อตามสูตร Phase 1 (เหยื่อแย่กว่า)
                X_p1 = np.clip(X_p1, self.lb, self.ub)  # จำกัดตำแหน่งใหม่ให้อยู่ภายใน [lb, ub]
                f_p1 = self.fitness_function(X_p1)       # คำนวณ fitness ของตำแหน่งใหม่
                if f_p1 < F[i]:                          # ถ้าตำแหน่งใหม่ดีกว่าเดิม
                    X[i], F[i] = X_p1, f_p1             # อัพเดตสมาชิก i ด้วยตำแหน่งและ fitness ใหม่

                # Phase 2: Exploitation
                # หมายเหตุ: สูตรต้นฉบับใช้ * X[i] ซึ่งเมื่อ X[i] ใกล้ 0 (เช่นค่า a)
                # จะทำให้ขยับไม่ได้เลย จึงเปลี่ยนเป็นคูณช่วง search (ub - lb)
                # เพื่อให้ทุก dimension มีโอกาสปรับเท่ากัน
                radius = self.R * (1 - t / self.T)
                # คำนวณรัศมีการค้นหาที่ค่อยๆ ลดลงตามจำนวน iteration ที่ผ่านไป
                X_p2 = X[i] + radius * (2 * np.random.rand(self.m) - 1) * (self.ub - self.lb)
                # สุ่มตำแหน่งใหม่รอบๆ X[i] ในรัศมีที่กำหนด (exploitation แบบ local search)
                X_p2 = np.clip(X_p2, self.lb, self.ub)  # จำกัดตำแหน่งใหม่ให้อยู่ภายใน [lb, ub]
                f_p2 = self.fitness_function(X_p2)       # คำนวณ fitness ของตำแหน่งใหม่
                if f_p2 < F[i]:                          # ถ้าตำแหน่งใหม่ดีกว่าเดิม
                    X[i], F[i] = X_p2, f_p2             # อัพเดตสมาชิก i ด้วยตำแหน่งและ fitness ใหม่

            cur = float(np.min(F))          # หาค่า fitness ต่ำสุดของประชากรในรอบนี้
            if cur < self.best_fitness:     # ถ้าดีกว่าค่าที่เคยบันทึกไว้
                self.best_fitness  = cur                      # อัพเดต best fitness
                self.best_solution = X[np.argmin(F)].copy()  # อัพเดต best solution

            self.fitness_history.append(self.best_fitness)   # บันทึก best fitness ของรอบนี้ลง history

        return self.best_solution, self.best_fitness  # คืนสารละลายที่ดีที่สุดและค่า fitness ของมัน


# ============================================================
# ส่วนที่ 3: เมนโปรแกรม
# ============================================================

if __name__ == "__main__":              # รันเฉพาะเมื่อเรียกไฟล์นี้โดยตรง (ไม่ใช่ import)
    if len(df) == 0:                    # ตรวจสอบว่า DataFrame ว่างเปล่าหรือไม่
        print("[ERROR] ไม่มีข้อมูลสำหรับรันอัลกอริทึม")  # แจ้งเตือนถ้าไม่มีข้อมูล
        raise SystemExit(1)             # ออกจากโปรแกรมด้วย exit code 1 (แสดงว่า error)

    np.random.seed(42)                  # กำหนด seed ให้ผลการสุ่มซ้ำได้ทุกครั้งที่รัน

    measured_all = df["Measured (cm)"].astype(float).values  # ดึงคอลัมน์ค่าวัดเป็น float array
    desired_all  = df["Desired (cm)"].astype(float).values   # ดึงคอลัมน์ค่าอ้างอิงเป็น float array
    raw_error = measured_all - desired_all
    n_samples    = len(measured_all)    # นับจำนวนจุดข้อมูลทั้งหมด

    # ----- รัน POA หาสมการชดเชย -----
    print("\n--- เริ่มหาสมการชดเชยด้วย POA (Quadratic + Huber loss) ---")
    # แสดงข้อความแจ้งว่ากำลังเริ่ม POA
    poa = POA_Calibration(measured_all, desired_all,
                          n=50, m=3, T=200, R=0.2, huber_delta=6.0)
    # สร้าง object POA_Calibration พร้อมข้อมูลและพารามิเตอร์ที่กำหนด
    best_w, best_loss = poa.run()       # รันอัลกอริทึม รับสารละลายที่ดีที่สุดและค่า loss กลับมา
    a, b, c = best_w                    # แตกสัมประสิทธิ์ที่ดีที่สุด [a, b, c] ออกเป็นตัวแปร
    print(f"\n[RESULT] สมการชดเชย:")   # พิมพ์หัวข้อผลลัพธ์
    print(f"   Desired_pred = ({a:.6e}) * M^2 + ({b:.6f}) * M + ({c:.6f})")
    # แสดงสมการ calibration พร้อมสัมประสิทธิ์ที่ค้นหาได้

    # ----- ใช้สมการกับ measurement -----
    corrected = a * measured_all ** 2 + b * measured_all + c
    # คำนวณค่า corrected ของทุกจุดโดยใส่สัมประสิทธิ์ที่ได้ลงสมการ quadratic
    residual  = corrected - desired_all     # หา residual = ค่า corrected ลบค่าอ้างอิง
    abs_err   = np.abs(residual)            # คำนวณ absolute error ทุกจุด

    # ----- แยก inlier / outlier ด้วย MAD -----
    med = np.median(abs_err)                # คำนวณ median ของ absolute error
    mad = np.median(np.abs(abs_err - med))  # คำนวณ Median Absolute Deviation (MAD)
    threshold = med + 5.0 * 1.4826 * mad
    # กำหนด threshold โดยใช้ 1.4826*MAD เป็น robust std และบวก 5 เท่าเพื่อจับ outlier
    inliers = abs_err <= threshold          # boolean mask: True = inlier, False = outlier
    n_out   = int((~inliers).sum())         # นับจำนวน outlier (ค่าที่เกิน threshold)

    # ----- Metrics -----
    classical_mae = np.mean(np.abs(measured_all - desired_all))  # MAE ก่อน calibration
    poa_mae_all   = np.mean(abs_err)            # MAE หลัง calibration รวมทุกจุด
    poa_mae_in    = np.mean(abs_err[inliers])   # MAE หลัง calibration เฉพาะ inlier
    rmse_all      = np.sqrt(np.mean(residual ** 2))          # RMSE รวมทุกจุด
    rmse_in       = np.sqrt(np.mean(residual[inliers] ** 2)) # RMSE เฉพาะ inlier

    print("\n=== Summary ===")           # พิมพ์หัวข้อสรุปผล
    print(f"จำนวนข้อมูล                 : {n_samples} จุด")        # แสดงจำนวนจุดข้อมูลทั้งหมด
    print(f"Outlier ที่ตรวจพบ (>5*MAD)  : {n_out} จุด")             # แสดงจำนวน outlier
    print("-" * 55)                      # พิมพ์เส้นคั่น 55 ขีด
    print(f"MAE  ก่อนคาลิเบรท (classical) : {classical_mae:7.4f} cm")  # แสดง MAE ก่อน calibrate
    print(f"MAE  หลังคาลิเบรท (ทุกจุด)    : {poa_mae_all:7.4f} cm  "
          f"(ลด {(1 - poa_mae_all / classical_mae) * 100:.2f}%)")
    # แสดง MAE หลัง calibrate ทุกจุด พร้อมเปอร์เซ็นต์ที่ลดลง
    print(f"MAE  หลังคาลิเบรท (inlier)    : {poa_mae_in:7.4f} cm  "
          f"(ลด {(1 - poa_mae_in / classical_mae) * 100:.2f}%)")
    # แสดง MAE หลัง calibrate เฉพาะ inlier พร้อมเปอร์เซ็นต์ที่ลดลง
    print(f"RMSE หลังคาลิเบรท (ทุกจุด)    : {rmse_all:7.4f} cm")  # แสดง RMSE ทุกจุด
    print(f"RMSE หลังคาลิเบรท (inlier)    : {rmse_in:7.4f} cm")   # แสดง RMSE เฉพาะ inlier

    # --------------------------------------------------------
    # Figure 8: Convergence
    # --------------------------------------------------------
    history = np.array(poa.fitness_history)     # แปลง fitness_history เป็น NumPy array
    plt.figure(figsize=(7, 5))                  # สร้าง figure ขนาด 7x5 นิ้ว
    it_x = np.arange(len(history))             # สร้าง array [0, 1, 2, ...] สำหรับแกน x (iteration)
    plt.plot(it_x, history, linewidth=2,
             label="POA fitness (Huber loss)", color="tab:blue")
    # วาดเส้น convergence curve สีน้ำเงิน ความกว้าง 2
    best_idx = int(np.argmin(history))          # หา index ของ iteration ที่ fitness ต่ำสุด
    plt.plot(it_x[best_idx], history[best_idx], "ro",
             label="Best fitness")
    # วาดจุดสีแดงที่ตำแหน่ง best fitness
    plt.xlabel("Iterations")                    # ตั้งชื่อแกน x
    plt.ylabel("POA fitness function")          # ตั้งชื่อแกน y
    plt.xlim(0, len(history) - 1)              # กำหนดช่วงแกน x ตั้งแต่ 0 ถึง iteration สุดท้าย
    plt.grid(True, alpha=0.5)                   # เปิด grid โปร่งใส 50%
    plt.legend()                                # แสดง legend
    plt.tight_layout()                          # จัดพื้นที่กราฟให้พอดีอัตโนมัติ
    plt.show()                                  # แสดง Figure 8

    # --------------------------------------------------------
    # Figure 9: Desired vs Optimized
    # --------------------------------------------------------
    pop_idx = np.arange(1, n_samples + 1)       # สร้าง array [1, 2, ..., n_samples] สำหรับแกน x
    plt.figure(figsize=(8, 6))                  # สร้าง figure ขนาด 8x6 นิ้ว
    plt.plot(pop_idx, desired_all, "ro", markersize=5,
             markerfacecolor="none", label="Desired Distances")
    # วาดจุดกลวงสีแดงแทนค่าอ้างอิง (desired)
    plt.plot(pop_idx, corrected, "b*", markersize=5,
             label="Optimized Measured Distances")
    # วาดดาวสีน้ำเงินแทนค่า corrected หลัง calibration
    plt.xlabel("number of population")          # ตั้งชื่อแกน x
    plt.ylabel("Distances (cm)")                # ตั้งชื่อแกน y
    plt.xlim(0, max(450, n_samples + 20))       # กำหนดช่วงแกน x ให้กว้างอย่างน้อย 450
    plt.ylim(0, max(desired_all) + 20)          # กำหนดช่วงแกน y ตั้งแต่ 0 ถึงค่าสูงสุด+20
    plt.legend(loc="upper left")                # แสดง legend มุมบนซ้าย
    plt.grid(True, alpha=0.4)                   # เปิด grid โปร่งใส 40%
    plt.tight_layout()                          # จัดพื้นที่กราฟให้พอดีอัตโนมัติ
    plt.show()                                  # แสดง Figure 9

    # --------------------------------------------------------
    # Figure 10: Error ก่อน vs หลังคาลิเบรท
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))                  # สร้าง figure ขนาด 8x5 นิ้ว
    plt.plot(pop_idx, measured_all - desired_all, "r.",
             markersize=4, alpha=0.6, label="Error before Optimization")
    # วาดจุดสีแดงแทน error ก่อน calibration (measured - desired)
    plt.plot(pop_idx, residual, "b.",
             markersize=4, alpha=0.6, label="Error After Optimization")
    # วาดจุดสีน้ำเงินแทน residual หลัง calibration
    plt.axhline(0, color="k", linewidth=0.8)    # วาดเส้นแนวนอนที่ y=0 (เส้นอ้างอิง error=0)
    plt.xlabel("number of population")          # ตั้งชื่อแกน x
    plt.ylabel("Error (cm)")                    # ตั้งชื่อแกน y
    plt.grid(True, alpha=0.4)                   # เปิด grid โปร่งใส 40%
    plt.legend()                                # แสดง legend
    plt.tight_layout()                          # จัดพื้นที่กราฟให้พอดีอัตโนมัติ
    plt.show()                                  # แสดง Figure 10
