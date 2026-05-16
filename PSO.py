import numpy as np          # นำเข้า NumPy สำหรับการคำนวณเชิงตัวเลขและเมทริกซ์
import matplotlib.pyplot as plt  # นำเข้า Matplotlib สำหรับวาดกราฟ
import pandas as pd         # นำเข้า Pandas สำหรับจัดการข้อมูลตาราง
import io                   # นำเข้า io สำหรับแปลง bytes เป็น stream ที่ pandas อ่านได้
import requests             # นำเข้า requests สำหรับดึงข้อมูลผ่าน HTTP

# ============================================================
# ส่วนที่ 1: Load Dataset (ดึงจาก Google Sheets)
# ============================================================

SHEET_NAME = "SensorData"   # ชื่อชีตที่ต้องการดึงข้อมูล (ใช้แสดงใน log เท่านั้น)
SHEET_ID = "169b1I4Gos8UhkzDkxH6uX9ty3yaQ_8kCqRGjqnpb0dU"  # ID ของ Google Spreadsheet
GID = "1511238558"          # GID ของชีตเฉพาะภายใน Spreadsheet
sheet_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"  # สร้าง URL สำหรับ export ชีตเป็น CSV

try:                         # เริ่มบล็อก try เพื่อจัดการข้อผิดพลาดขณะโหลดข้อมูล
    print(f"[INFO] กำลังดึงข้อมูลจากชีต: {SHEET_NAME}...")  # แสดงข้อความแจ้งว่ากำลังดึงข้อมูล
    response = requests.get(sheet_url, timeout=20)  # ส่ง HTTP GET ไปยัง URL โดยกำหนด timeout 20 วินาที
    response.raise_for_status()  # โยน exception ถ้า HTTP status code เป็น 4xx หรือ 5xx

    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))  # แปลง response bytes เป็น CSV แล้วโหลดเป็น DataFrame
    df.columns = df.columns.str.strip()  # ตัดช่องว่างหัว-ท้ายออกจากชื่อคอลัมน์ทุกคอลัมน์

    required_cols = ["Measured (cm)", "Desired (cm)", "Index"]  # กำหนดรายชื่อคอลัมน์ที่จำเป็นต้องมี
    for col in required_cols:           # วนลูปตรวจสอบทีละคอลัมน์
        if col not in df.columns:       # ถ้าคอลัมน์นั้นไม่มีใน DataFrame
            raise KeyError(f"ไม่พบคอลัมน์: {col}")  # โยน KeyError พร้อมชื่อคอลัมน์ที่หายไป

    df = df.sort_values(by="Index").reset_index(drop=True)  # เรียงข้อมูลตาม Index แล้วรีเซ็ต index ของ DataFrame
    print(f"[SUCCESS] โหลดข้อมูลสำเร็จทั้งหมด {len(df)} แถว")  # แสดงจำนวนแถวที่โหลดสำเร็จ
    print(df.head())            # แสดง 5 แถวแรกของข้อมูลเพื่อตรวจสอบ

except Exception as e:          # รับ exception ทุกประเภทที่เกิดในบล็อก try
    print(f"[ERROR] ไม่สามารถโหลดข้อมูลได้: {e}")  # แสดงข้อความแจ้งข้อผิดพลาด
    df = pd.DataFrame()         # กำหนด df เป็น DataFrame ว่างเพื่อให้โปรแกรมทำงานต่อได้


# ============================================================
# ส่วนที่ 2: Particle Swarm Optimization (PSO)
# อ้างอิง: https://en.wikipedia.org/wiki/Particle_swarm_optimization
# ------------------------------------------------------------
# ค้นหาสัมประสิทธิ์สมการชดเชย (calibration equation)
#     Desired_pred = a * Measured^2 + b * Measured + c
# ใช้ Huber loss เพื่อให้ทนต่อ outlier
# ตัวแปรค้นหา m = 3 :  [a, b, c]
#
# Velocity update (Kennedy & Eberhart, 1995 / Clerc & Kennedy, 2002):
#     v_{i}(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
#     x_{i}(t+1) = x_i(t) + v_{i}(t+1)
# ค่ามาตรฐาน Clerc constriction:  w=0.7298, c1=c2=1.49618
# ============================================================

class PSO_Calibration:          # ประกาศคลาสสำหรับรัน PSO เพื่อหาสมการคาลิเบรท
    def __init__(self, measured, desired,   # constructor รับ array ข้อมูลวัดและค่าจริง
                 n=50, m=3, T=200,          # n=จำนวน particle, m=มิติ (a,b,c), T=จำนวนรอบ
                 w=0.7298, c1=1.49618, c2=1.49618,  # ค่า inertia weight และ acceleration coefficient
                 huber_delta=6.0):          # delta สำหรับ Huber loss
        self.measured = measured    # เก็บ array ค่า Measured ที่ได้จากเซนเซอร์
        self.desired  = desired     # เก็บ array ค่า Desired ที่เป็นค่าจริง
        self.n = n                  # จำนวน particle ใน swarm
        self.m = m                  # จำนวนมิติของ search space (3 ตัวแปร: a, b, c)
        self.T = T                  # จำนวน iteration สูงสุดที่จะรัน
        self.w  = w                 # inertia weight ควบคุมความเร็วเดิมที่นำมาใช้
        self.c1 = c1                # cognitive coefficient ดึง particle ไปหา personal best
        self.c2 = c2                # social coefficient ดึง particle ไปหา global best
        self.huber_delta = huber_delta  # threshold สำหรับแยก quadratic/linear ใน Huber loss
        self.fitness_history = []   # list เก็บประวัติค่า fitness ที่ดีที่สุดในแต่ละรอบ

        # ขอบเขตของ [a, b, c]  (เหมือน POA เพื่อเทียบกันได้ตรง ๆ)
        self.lb = np.array([-0.0003,  0.90, -5.0])  # lower bound ของแต่ละตัวแปร [a, b, c]
        self.ub = np.array([ 0.0003,  1.20, 12.0])  # upper bound ของแต่ละตัวแปร [a, b, c]

        # จำกัดความเร็วสูงสุดที่ 20% ของช่วง search
        self.v_max = 0.2 * (self.ub - self.lb)  # คำนวณความเร็วสูงสุดเป็น 20% ของขนาด search space

    def _huber(self, r):
        """Huber loss: quadratic เมื่อ |r|<=delta, linear เมื่อเกินกว่านั้น"""
        d = self.huber_delta        # นำค่า delta มาใช้งานในตัวแปรท้องถิ่น
        absr = np.abs(r)            # คำนวณค่าสัมบูรณ์ของ residual ทุกจุด
        quad = 0.5 * r ** 2         # คำนวณ quadratic loss สำหรับจุดที่ |r| <= delta
        lin  = d * (absr - 0.5 * d) # คำนวณ linear loss สำหรับจุดที่ |r| > delta
        return np.where(absr <= d, quad, lin).mean()  # เลือก loss ที่เหมาะสมแต่ละจุด แล้วหาค่าเฉลี่ย

    def fitness_function(self, w):  # ฟังก์ชันคำนวณค่า fitness ของ particle หนึ่งตัว
        a, b, c = w                 # แตก array [a, b, c] ออกเป็น 3 ตัวแปร
        pred = a * self.measured ** 2 + b * self.measured + c  # คำนวณค่าพยากรณ์จากสมการ quadratic
        residual = pred - self.desired  # คำนวณ residual (ส่วนต่างระหว่างพยากรณ์กับค่าจริง)
        return self._huber(residual)    # ส่งคืนค่า Huber loss เป็นค่า fitness

    def run(self):                  # เมธอดหลักสำหรับรันกระบวนการ PSO
        # Initialization: ตำแหน่งและความเร็วสุ่ม
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)  # สุ่มตำแหน่งเริ่มต้นของ particle ทุกตัวในขอบเขต
        V = np.random.uniform(-self.v_max, self.v_max, (self.n, self.m))    # สุ่มความเร็วเริ่มต้นในช่วง [-v_max, v_max]

        # Personal best (pbest)
        pbest     = X.copy()        # คัดลอกตำแหน่งเริ่มต้นเป็น personal best ของแต่ละ particle
        pbest_fit = np.array([self.fitness_function(p) for p in X])  # คำนวณ fitness ของ personal best ทุกตัว

        # Global best (gbest)
        g_idx     = int(np.argmin(pbest_fit))  # หา index ของ particle ที่มี fitness ดีที่สุด
        gbest     = pbest[g_idx].copy()         # คัดลอกตำแหน่ง particle ที่ดีที่สุดเป็น global best
        gbest_fit = float(pbest_fit[g_idx])     # เก็บค่า fitness ของ global best

        self.fitness_history = [gbest_fit]      # เริ่มบันทึกประวัติ fitness ด้วยค่าแรก

        for t in range(1, self.T + 1):          # วนลูปตาม iteration ที่กำหนด (1 ถึง T)
            for i in range(self.n):             # วนลูปอัปเดตทุก particle ใน swarm
                r1 = np.random.rand(self.m)     # สุ่มเลข [0,1) สำหรับ cognitive component แต่ละมิติ
                r2 = np.random.rand(self.m)     # สุ่มเลข [0,1) สำหรับ social component แต่ละมิติ

                # Velocity update
                V[i] = (self.w  * V[i]                      # ส่วน inertia: ความเร็วเดิมคูณน้ำหนัก
                        + self.c1 * r1 * (pbest[i] - X[i])  # ส่วน cognitive: ดึงไปหา personal best
                        + self.c2 * r2 * (gbest    - X[i])) # ส่วน social: ดึงไปหา global best
                # clip ความเร็ว ป้องกันการระเบิด (velocity clamping)
                V[i] = np.clip(V[i], -self.v_max, self.v_max)  # จำกัดความเร็วไม่ให้เกิน v_max

                # Position update
                X[i] = X[i] + V[i]             # อัปเดตตำแหน่ง particle ด้วยความเร็วใหม่
                # clip ตำแหน่งให้อยู่ในขอบเขต search
                X[i] = np.clip(X[i], self.lb, self.ub)  # จำกัดตำแหน่งไม่ให้ออกนอก search space

                # Evaluate
                fit_curr = self.fitness_function(X[i])  # คำนวณ fitness ของตำแหน่งใหม่

                # Update pbest
                if fit_curr < pbest_fit[i]:     # ถ้า fitness ใหม่ดีกว่า personal best เดิม
                    pbest[i]     = X[i].copy()  # อัปเดต personal best เป็นตำแหน่งปัจจุบัน
                    pbest_fit[i] = fit_curr      # อัปเดตค่า fitness ของ personal best

                    # Update gbest
                    if fit_curr < gbest_fit:    # ถ้า fitness ใหม่ดีกว่า global best เดิม
                        gbest     = X[i].copy() # อัปเดต global best เป็นตำแหน่งปัจจุบัน
                        gbest_fit = float(fit_curr)  # อัปเดตค่า fitness ของ global best

            self.fitness_history.append(gbest_fit)  # บันทึกค่า fitness ดีที่สุดของรอบนี้

        self.best_solution = gbest      # เก็บตำแหน่ง (a, b, c) ที่ดีที่สุดไว้ใน attribute
        self.best_fitness  = gbest_fit  # เก็บค่า fitness ที่ดีที่สุดไว้ใน attribute
        return gbest, gbest_fit         # ส่งคืนตำแหน่งและค่า fitness ที่ดีที่สุด


# ============================================================
# ส่วนที่ 3: เมนโปรแกรม
# ============================================================

if __name__ == "__main__":              # รันเฉพาะเมื่อรันไฟล์นี้โดยตรง ไม่ใช่ import
    if len(df) == 0:                    # ตรวจสอบว่า DataFrame มีข้อมูลหรือไม่
        print("[ERROR] ไม่มีข้อมูลสำหรับรันอัลกอริทึม")  # แจ้งข้อผิดพลาดถ้าไม่มีข้อมูล
        raise SystemExit(1)             # หยุดโปรแกรมด้วย exit code 1

    np.random.seed(42)                  # กำหนด random seed เพื่อให้ผลลัพธ์ reproducible

    measured_all = df["Measured (cm)"].astype(float).values  # ดึงคอลัมน์ค่าวัดเป็น float array
    desired_all  = df["Desired (cm)"].astype(float).values   # ดึงคอลัมน์ค่าจริงเป็น float array
    n_samples    = len(measured_all)    # นับจำนวนข้อมูลทั้งหมด

    # ----- รัน PSO หาสมการชดเชย -----
    print("\n--- เริ่มหาสมการชดเชยด้วย PSO (Quadratic + Huber loss) ---")  # แสดงหัวข้อเริ่มต้น
    pso = PSO_Calibration(measured_all, desired_all,  # สร้าง object PSO พร้อมข้อมูล
                          n=50, m=3, T=200,           # 50 particle, 3 ตัวแปร, 200 iteration
                          w=0.7298, c1=1.49618, c2=1.49618,  # ค่า Clerc constriction มาตรฐาน
                          huber_delta=6.0)             # delta สำหรับ Huber loss
    best_w, best_loss = pso.run()       # รัน PSO และรับค่า (a, b, c) ที่ดีที่สุดกับค่า loss
    a, b, c = best_w                    # แตกค่า (a, b, c) จาก array ผลลัพธ์
    print(f"\n[RESULT] สมการชดเชย:")    # แสดงหัวข้อผลลัพธ์
    print(f"   Desired_pred = ({a:.6e}) * M^2 + ({b:.6f}) * M + ({c:.6f})")  # แสดงสมการ quadratic ที่ได้

    # ----- ใช้สมการกับ measurement -----
    corrected = a * measured_all ** 2 + b * measured_all + c  # คำนวณค่าที่ผ่านการคาลิเบรทแล้วทุกจุด
    residual  = corrected - desired_all  # คำนวณ residual ระหว่างค่าคาลิเบรทกับค่าจริง
    abs_err   = np.abs(residual)         # คำนวณ absolute error ทุกจุด

    # ----- แยก inlier / outlier ด้วย MAD -----
    med = np.median(abs_err)             # คำนวณ median ของ absolute error
    mad = np.median(np.abs(abs_err - med))  # คำนวณ Median Absolute Deviation (MAD)
    threshold = med + 5.0 * 1.4826 * mad   # คำนวณ threshold = median + 5*sigma โดยใช้ scaling factor MAD
    inliers = abs_err <= threshold       # สร้าง boolean mask สำหรับจุดที่เป็น inlier
    n_out   = int((~inliers).sum())      # นับจำนวน outlier (จุดที่ไม่ใช่ inlier)

    # ----- Metrics -----
    classical_mae = np.mean(np.abs(measured_all - desired_all))  # คำนวณ MAE ก่อนคาลิเบรท (raw sensor)
    pso_mae_all   = np.mean(abs_err)         # คำนวณ MAE หลังคาลิเบรทของข้อมูลทุกจุด
    pso_mae_in    = np.mean(abs_err[inliers])  # คำนวณ MAE หลังคาลิเบรทเฉพาะ inlier
    rmse_all      = np.sqrt(np.mean(residual ** 2))          # คำนวณ RMSE ของข้อมูลทุกจุด
    rmse_in       = np.sqrt(np.mean(residual[inliers] ** 2)) # คำนวณ RMSE เฉพาะ inlier

    print("\n=== Summary ===")           # แสดงหัวข้อสรุปผล
    print(f"จำนวนข้อมูล                 : {n_samples} จุด")  # แสดงจำนวนข้อมูลทั้งหมด
    print(f"Outlier ที่ตรวจพบ (>5*MAD)  : {n_out} จุด")     # แสดงจำนวน outlier ที่ตรวจพบ
    print("-" * 55)                      # แสดงเส้นคั่น
    print(f"MAE  ก่อนคาลิเบรท (classical) : {classical_mae:7.4f} cm")  # แสดง MAE ก่อนคาลิเบรท
    print(f"MAE  หลังคาลิเบรท (ทุกจุด)    : {pso_mae_all:7.4f} cm  "
          f"(ลด {(1 - pso_mae_all / classical_mae) * 100:.2f}%)")  # แสดง MAE หลังคาลิเบรทพร้อม % ที่ลดลง
    print(f"MAE  หลังคาลิเบรท (inlier)    : {pso_mae_in:7.4f} cm  "
          f"(ลด {(1 - pso_mae_in / classical_mae) * 100:.2f}%)")   # แสดง MAE เฉพาะ inlier พร้อม % ที่ลดลง
    print(f"RMSE หลังคาลิเบรท (ทุกจุด)    : {rmse_all:7.4f} cm")  # แสดง RMSE ของทุกจุด
    print(f"RMSE หลังคาลิเบรท (inlier)    : {rmse_in:7.4f} cm")   # แสดง RMSE เฉพาะ inlier

    # --------------------------------------------------------
    # Figure 8: Convergence
    # --------------------------------------------------------
    history = np.array(pso.fitness_history)  # แปลง list ประวัติ fitness เป็น numpy array
    plt.figure(figsize=(7, 5))          # สร้าง figure ขนาด 7x5 นิ้ว
    it_x = np.arange(len(history))     # สร้าง array ลำดับ iteration สำหรับแกน x
    plt.plot(it_x, history, linewidth=2,
             label="PSO fitness (Huber loss)", color="tab:green")  # วาดเส้น convergence curve สีเขียว
    best_idx = int(np.argmin(history))  # หา index ของ iteration ที่ fitness ดีที่สุด
    plt.plot(it_x[best_idx], history[best_idx], "ro",
             label="Best fitness")      # วาดจุดแดงที่ตำแหน่ง fitness ดีที่สุด
    plt.xlabel("Iterations")            # ตั้งชื่อแกน x เป็น Iterations
    plt.ylabel("PSO fitness function")  # ตั้งชื่อแกน y เป็นชื่อ fitness function
    plt.xlim(0, len(history) - 1)      # กำหนดช่วงแกน x ตั้งแต่ 0 ถึง iteration สุดท้าย
    plt.grid(True, alpha=0.5)           # แสดง grid โปร่งแสง 50%
    plt.legend()                        # แสดง legend ของกราฟ
    plt.tight_layout()                  # ปรับ layout ให้พอดีกรอบ
    plt.show()                          # แสดงกราฟ

    # --------------------------------------------------------
    # Figure 9: Desired vs Calibrated
    # --------------------------------------------------------
    pop_idx = np.arange(1, n_samples + 1)  # สร้าง array ลำดับตัวอย่าง 1 ถึง n_samples
    plt.figure(figsize=(8, 6))          # สร้าง figure ขนาด 8x6 นิ้ว
    plt.plot(pop_idx, desired_all, "ro", markersize=5,
             markerfacecolor="none", label="desired distances")  # วาดจุดวงกลมแดงสำหรับค่าจริง
    plt.plot(pop_idx, corrected, "g*", markersize=5,
             label="calibrated measured distances")  # วาดจุดดาวเขียวสำหรับค่าหลังคาลิเบรท
    plt.xlabel("number of population")  # ตั้งชื่อแกน x
    plt.ylabel("distances (cm)")        # ตั้งชื่อแกน y
    plt.xlim(0, max(450, n_samples + 20))  # กำหนดช่วงแกน x อย่างน้อย 450 หรือมากกว่าจำนวนข้อมูล
    plt.ylim(0, max(desired_all) + 20)  # กำหนดช่วงแกน y ตั้งแต่ 0 ถึงค่าสูงสุด+20
    plt.legend(loc="upper left")        # แสดง legend มุมบนซ้าย
    plt.grid(True, alpha=0.4)           # แสดง grid โปร่งแสง 40%
    plt.tight_layout()                  # ปรับ layout ให้พอดีกรอบ
    plt.show()                          # แสดงกราฟ

    # --------------------------------------------------------
    # Figure 10: Error ก่อน vs หลังคาลิเบรท
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))          # สร้าง figure ขนาด 8x5 นิ้ว
    plt.plot(pop_idx, measured_all - desired_all, "r.",
             markersize=4, alpha=0.6, label="error before calibration")  # วาดจุดแดงแสดง error ก่อนคาลิเบรท
    plt.plot(pop_idx, residual, "g.",
             markersize=4, alpha=0.6, label="error after calibration")   # วาดจุดเขียวแสดง error หลังคาลิเบรท
    plt.axhline(0, color="k", linewidth=0.8)  # วาดเส้นแนวนอนที่ y=0 เป็นเส้นอ้างอิง
    plt.xlabel("number of population")  # ตั้งชื่อแกน x
    plt.ylabel("error (cm)")            # ตั้งชื่อแกน y
    plt.grid(True, alpha=0.4)           # แสดง grid โปร่งแสง 40%
    plt.legend()                        # แสดง legend ของกราฟ
    plt.tight_layout()                  # ปรับ layout ให้พอดีกรอบ
    plt.show()                          # แสดงกราฟ