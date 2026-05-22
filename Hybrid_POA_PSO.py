import numpy as np          # นำเข้า NumPy สำหรับคำนวณเมทริกซ์และตัวเลข
import matplotlib.pyplot as plt  # นำเข้า Matplotlib สำหรับวาดกราฟ
import pandas as pd         # นำเข้า Pandas สำหรับจัดการข้อมูลตาราง
import io                   # นำเข้า io สำหรับแปลง bytes เป็น stream ที่ pandas อ่านได้
import requests             # นำเข้า requests สำหรับดึงข้อมูลจาก URL

# ============================================================
# ส่วนที่ 1: Load Dataset (ดึงจาก Google Sheets)
# ============================================================

SHEET_NAME = "SensorData"  # ชื่อชีตที่ต้องการดึงข้อมูล (ใช้แสดงใน log เท่านั้น)
SHEET_ID = "169b1I4Gos8UhkzDkxH6uX9ty3yaQ_8kCqRGjqnpb0dU"  # ID ของ Google Sheets
GID = "1511238558"          # GID ของชีตย่อยที่ต้องการ (tab ภายใน Spreadsheet)
sheet_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
# สร้าง URL สำหรับ export ชีตเป็นไฟล์ CSV โดยฝัง SHEET_ID และ GID ลงไป

try:
    print(f"[INFO] กำลังดึงข้อมูลจากชีต: {SHEET_NAME}...")  # แจ้งสถานะเริ่มดึงข้อมูล
    response = requests.get(sheet_url, timeout=20)  # ส่ง HTTP GET ไปยัง URL พร้อมกำหนด timeout 20 วินาที
    response.raise_for_status()  # ถ้า HTTP status code เป็น error (4xx/5xx) จะ raise exception ทันที

    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    # แปลง bytes ที่ได้จาก response เป็น string (UTF-8) แล้วห่อด้วย StringIO
    # จากนั้น pandas อ่านเป็น DataFrame
    df.columns = df.columns.str.strip()  # ตัด whitespace ซ้าย-ขวาออกจากชื่อคอลัมน์ทุกคอลัมน์

    required_cols = ["Measured (cm)", "Desired (cm)", "Index"]  # กำหนดชื่อคอลัมน์ที่จำเป็นต้องมี
    for col in required_cols:          # วนตรวจสอบทีละคอลัมน์
        if col not in df.columns:      # ถ้าคอลัมน์นั้นไม่อยู่ใน DataFrame
            raise KeyError(f"ไม่พบคอลัมน์: {col}")  # โยน KeyError เพื่อหยุดการทำงาน

    df = df.sort_values(by="Index").reset_index(drop=True)
    # เรียงแถวตามคอลัมน์ Index (น้อยไปมาก) แล้ว reset index ให้เริ่มจาก 0 ใหม่
    print(f"[SUCCESS] โหลดข้อมูลสำเร็จทั้งหมด {len(df)} แถว")  # แสดงจำนวนแถวที่โหลดได้
    print(df.head())  # แสดง 5 แถวแรกของ DataFrame เพื่อตรวจสอบ

except Exception as e:  # ดักจับ exception ทุกประเภทที่เกิดใน try block
    print(f"[ERROR] ไม่สามารถโหลดข้อมูลได้: {e}")  # แสดงข้อความ error
    df = pd.DataFrame()  # สร้าง DataFrame เปล่าเพื่อให้โปรแกรมไม่พัง (จะตรวจสอบอีกทีตอนรัน)


# ============================================================
# ส่วนที่ 2: Hybrid POA -> PSO
# ------------------------------------------------------------
# แนวคิด:
#   Phase A (POA):  ใช้ POA ทำ exploration กว้าง ๆ ก่อน
#                   (POA มี random prey + radius ที่ค่อย ๆ หด ช่วย
#                    หลีกเลี่ยง local optima ในช่วงต้น)
#   Phase B (PSO):  ใช้ PSO ลู่เข้าจุดที่ดีที่สุดอย่างรวดเร็ว
#                   โดยรับ population สุดท้ายของ POA เป็นจุดเริ่ม
#                   (PSO เก่ง local refinement เพราะมี pbest/gbest guide)
#
#   ค้นหาสัมประสิทธิ์สมการชดเชย:
#       Desired_pred = a * Measured^2 + b * Measured + c
#   ใช้ Huber loss (robust ต่อ outlier),  m = 3 :  [a, b, c]
# ============================================================

class Hybrid_POA_PSO:  # ประกาศ class ที่รวม POA และ PSO ทำงานต่อกัน
    def __init__(self, measured, desired,
                 n=50, m=3,
                 T_poa=150, T_pso=50,
                 R=0.2,
                 w=0.7298, c1=1.49618, c2=1.49618,
                 huber_delta=6.0):
        # constructor รับพารามิเตอร์ทั้งหมดและเก็บเป็น attribute
        self.measured = measured  # array ค่าที่เซ็นเซอร์วัดได้ (cm)
        self.desired  = desired   # array ค่าที่ต้องการจริง (cm)
        self.n  = n               # จำนวน particle/agent ในประชากร
        self.m  = m               # จำนวนมิติของปัญหา (3 ตัวแปร: a, b, c)
        self.T_poa = T_poa        # จำนวน iteration ของ Phase A (POA)
        self.T_pso = T_pso        # จำนวน iteration ของ Phase B (PSO)
        # POA params
        self.R = R                # รัศมีเริ่มต้นสำหรับ exploitation ของ POA
        # PSO params
        self.w  = w               # inertia weight ของ PSO (ควบคุมความเฉื่อยของ velocity)
        self.c1 = c1              # cognitive coefficient (แรงดึงดูดของ particle ต่อ pbest ตัวเอง)
        self.c2 = c2              # social coefficient (แรงดึงดูดของ particle ต่อ gbest)
        self.huber_delta = huber_delta  # threshold δ ของ Huber loss
        self.fitness_history = []  # list เก็บ best fitness ของแต่ละ iteration

        self.lb = np.array([-0.0003,  0.90, -5.0])   # lower bound ของแต่ละมิติ [a, b, c]
        self.ub = np.array([ 0.0003,  1.20, 12.0])   # upper bound ของแต่ละมิติ [a, b, c]
        self.v_max = 0.2 * (self.ub - self.lb)  # velocity สูงสุดที่อนุญาต = 20% ของช่วงค้นหา

    def _huber(self, r):
        # คำนวณ Huber loss เฉลี่ยของ residual vector r
        d = self.huber_delta        # ดึงค่า δ มาใช้งาน
        absr = np.abs(r)            # คำนวณค่าสัมบูรณ์ของ residual ทุกตัว
        quad = 0.5 * r ** 2         # ส่วน quadratic: ใช้เมื่อ |r| <= δ  →  0.5 * r²
        lin  = d * (absr - 0.5 * d) # ส่วน linear: ใช้เมื่อ |r| > δ  →  δ*(|r| - 0.5*δ)
        return np.where(absr <= d, quad, lin).mean()
        # เลือก quad หรือ lin ตามเงื่อนไข แล้วคืนค่าเฉลี่ยทั้งหมด

    def fitness_function(self, w):
        # คำนวณ fitness (ยิ่งน้อยยิ่งดี) ของ solution หนึ่งตัว
        a, b, c = w  # แตกค่า [a, b, c] ออกจาก array
        pred = a * self.measured ** 2 + b * self.measured + c
        # คำนวณค่าทำนายด้วยสมการ quadratic สำหรับทุก sample
        return self._huber(pred - self.desired)
        # คืนค่า Huber loss ระหว่างค่าทำนายกับค่าจริง

    # ---------- Phase A: POA ----------
    def _run_poa(self, X):
        # รัน Pelican Optimization Algorithm (POA) บน population X
        F = np.array([self.fitness_function(s) for s in X])
        # คำนวณ fitness ของทุก agent ใน population เก็บเป็น array F
        best_fitness  = float(np.min(F))       # หา fitness ดีที่สุด (ต่ำสุด) ใน generation แรก
        best_solution = X[np.argmin(F)].copy() # เก็บ solution ที่ให้ fitness ดีที่สุด
        self.fitness_history.append(best_fitness)  # บันทึก best fitness ของ generation แรก

        for t in range(1, self.T_poa + 1):  # วนลูปตามจำนวน iteration ที่กำหนด
            prey   = self.lb + np.random.rand(self.m) * (self.ub - self.lb)
            # สุ่มตำแหน่ง "เหยื่อ" แบบ random ภายใน search space
            f_prey = self.fitness_function(prey)  # คำนวณ fitness ของเหยื่อ

            for i in range(self.n):  # วนผ่านทุก agent
                # Phase 1: Exploration
                I = np.random.choice([1, 2])  # สุ่มค่า I เป็น 1 หรือ 2 (ปัจจัยขยาย)
                if f_prey < F[i]:  # ถ้าเหยื่ออยู่ในตำแหน่งที่ดีกว่า agent ปัจจุบัน
                    X_p1 = X[i] + np.random.rand(self.m) * (prey - I * X[i])
                    # เคลื่อน agent เข้าหาเหยื่อ (exploration แบบ aggressive)
                else:  # ถ้า agent ดีกว่าเหยื่อ
                    X_p1 = X[i] + np.random.rand(self.m) * (X[i] - prey)
                    # เคลื่อน agent หนีออกจากเหยื่อ (exploration แบบ diversification)
                X_p1 = np.clip(X_p1, self.lb, self.ub)  # จำกัดค่าให้อยู่ใน boundary
                f_p1 = self.fitness_function(X_p1)       # คำนวณ fitness ของตำแหน่งใหม่
                if f_p1 < F[i]:                          # ถ้าตำแหน่งใหม่ดีกว่าเดิม
                    X[i], F[i] = X_p1, f_p1             # อัปเดตตำแหน่งและ fitness

                # Phase 2: Exploitation (ใช้ (ub-lb) แทน X[i] เพื่อกัน a ขยับไม่ได้)
                radius = self.R * (1 - t / self.T_poa)
                # คำนวณรัศมีการค้นหา: เริ่มต้นใหญ่แล้วค่อยๆ หดลงตามเวลา (linear decay)
                X_p2 = X[i] + radius * (2 * np.random.rand(self.m) - 1) * (self.ub - self.lb)
                # สุ่มตำแหน่งใหม่รอบๆ agent ปัจจุบันภายในรัศมีที่กำหนด
                X_p2 = np.clip(X_p2, self.lb, self.ub)  # จำกัดค่าให้อยู่ใน boundary
                f_p2 = self.fitness_function(X_p2)       # คำนวณ fitness ของตำแหน่งใหม่
                if f_p2 < F[i]:                          # ถ้าตำแหน่งใหม่ดีกว่าเดิม
                    X[i], F[i] = X_p2, f_p2             # อัปเดตตำแหน่งและ fitness

            cur = float(np.min(F))          # หา best fitness ของ iteration นี้
            if cur < best_fitness:          # ถ้าดีกว่า global best ที่เคยเจอ
                best_fitness  = cur                   # อัปเดต global best fitness
                best_solution = X[np.argmin(F)].copy()  # อัปเดต global best solution
            self.fitness_history.append(best_fitness)  # บันทึก best fitness ของ iteration นี้

        return X, F, best_solution, best_fitness
        # คืน population สุดท้าย, fitness array, best solution, และ best fitness

    # ---------- Phase B: PSO ----------
    def _run_pso(self, X, F, gbest, gbest_fit):
        # รัน Particle Swarm Optimization (PSO) โดยรับ population จาก POA เป็น input
        # เริ่ม V ใกล้ 0 (ไม่ random เต็ม v_max) เพื่อไม่ให้ particle
        # ที่ POA หาตำแหน่งดีไว้แล้ว "หลุด" ออกจากบริเวณ optimum
        # ค่อย ๆ เร่งตาม pbest/gbest guide เอง
        V = np.random.uniform(-0.1 * self.v_max, 0.1 * self.v_max,
                              (self.n, self.m))
        # สุ่ม velocity เริ่มต้นให้เล็กมาก (±10% ของ v_max) เพื่อรักษาตำแหน่งดีจาก POA
        # ใช้ตำแหน่ง/fitness จาก POA เป็น pbest เริ่มต้น
        pbest     = X.copy()   # personal best position ของแต่ละ particle เริ่มจากตำแหน่ง POA สุดท้าย
        pbest_fit = F.copy()   # personal best fitness เริ่มจาก fitness POA สุดท้าย

        for t in range(1, self.T_pso + 1):  # วนลูปตามจำนวน iteration PSO
            for i in range(self.n):          # วนผ่านทุก particle
                r1 = np.random.rand(self.m)  # สุ่มเวกเตอร์ random สำหรับ cognitive component
                r2 = np.random.rand(self.m)  # สุ่มเวกเตอร์ random สำหรับ social component
                V[i] = (self.w  * V[i]                      # inertia: รักษาทิศทางเดิม
                        + self.c1 * r1 * (pbest[i] - X[i])  # cognitive: ดึงเข้า pbest ตัวเอง
                        + self.c2 * r2 * (gbest    - X[i]))  # social: ดึงเข้า gbest ทั้งฝูง
                V[i] = np.clip(V[i], -self.v_max, self.v_max)  # จำกัด velocity ไม่ให้เกิน v_max
                X[i] = np.clip(X[i] + V[i], self.lb, self.ub) # อัปเดตตำแหน่ง แล้วจำกัดใน boundary
                fit_curr = self.fitness_function(X[i])  # คำนวณ fitness ของตำแหน่งใหม่
                if fit_curr < pbest_fit[i]:              # ถ้าดีกว่า personal best เดิม
                    pbest[i]     = X[i].copy()           # อัปเดต personal best position
                    pbest_fit[i] = fit_curr              # อัปเดต personal best fitness
                    if fit_curr < gbest_fit:             # ถ้าดีกว่า global best ด้วย
                        gbest     = X[i].copy()          # อัปเดต global best position
                        gbest_fit = float(fit_curr)      # อัปเดต global best fitness
            self.fitness_history.append(gbest_fit)  # บันทึก best fitness ของ iteration นี้

        return gbest, gbest_fit  # คืน global best solution และ fitness หลัง PSO เสร็จ

    # ---------- run ----------
    def run(self):
        # เมธอดหลักสำหรับรัน Hybrid POA -> PSO ทั้งหมด
        # reset history เผื่อมีการเรียก run() ซ้ำบน object เดิม
        self.fitness_history = []  # ล้างประวัติ fitness ก่อนเริ่มรันใหม่
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)
        # สุ่ม population เริ่มต้น n ตัว ในช่วง [lb, ub] ทุกมิติ

        # Phase A: POA
        print("[Hybrid] >> Phase A: POA ...")  # แจ้งสถานะเริ่ม Phase A
        X_after_poa, F_after_poa, gbest, gbest_fit = self._run_poa(X)
        # รัน POA แล้วรับ population สุดท้าย, fitness, best solution, best fitness
        print(f"           POA best fitness = {gbest_fit:.6f}")  # แสดงผล best fitness หลัง POA

        # Phase B: PSO refine
        print("[Hybrid] >> Phase B: PSO refine ...")  # แจ้งสถานะเริ่ม Phase B
        best_sol, best_fit = self._run_pso(
            X_after_poa.copy(), F_after_poa.copy(), gbest.copy(), gbest_fit)
        # รัน PSO โดยส่ง population/fitness จาก POA และ gbest เป็นจุดเริ่มต้น
        print(f"           PSO refined fitness = {best_fit:.6f}")  # แสดง fitness หลัง PSO

        self.best_solution = best_sol   # เก็บ best solution สุดท้ายเป็น attribute
        self.best_fitness  = best_fit   # เก็บ best fitness สุดท้ายเป็น attribute
        self.split_iter    = self.T_poa + 1  # บันทึก iteration ที่เปลี่ยนจาก POA เป็น PSO
        return best_sol, best_fit  # คืนค่า best solution และ fitness


# ============================================================
# ส่วนที่ 3: เมนโปรแกรม
# ============================================================

if __name__ == "__main__":  # รันเฉพาะเมื่อเรียกไฟล์นี้โดยตรง (ไม่รันถ้า import)
    if len(df) == 0:  # ตรวจสอบว่าโหลดข้อมูลได้หรือไม่
        print("[ERROR] ไม่มีข้อมูลสำหรับรันอัลกอริทึม")  # แจ้ง error
        raise SystemExit(1)  # หยุดโปรแกรมด้วย exit code 1

    np.random.seed(42)  # กำหนด random seed เพื่อให้ผลลัพธ์ reproducible ทุกครั้ง

    measured_all = df["Measured (cm)"].astype(float).values  # ดึงคอลัมน์ค่าวัดเป็น numpy array
    desired_all  = df["Desired (cm)"].astype(float).values   # ดึงคอลัมน์ค่าจริงเป็น numpy array
    n_samples    = len(measured_all)  # นับจำนวน sample ทั้งหมด

    print("\n--- Hybrid POA -> PSO (Quadratic + Huber loss) ---")  # พิมพ์หัวข้อ
    hyb = Hybrid_POA_PSO(measured_all, desired_all,
                         n=50, m=3,
                         T_poa=150, T_pso=50,
                         R=0.2,
                         w=0.7298, c1=1.49618, c2=1.49618,
                         huber_delta=6.0)
    # สร้าง object Hybrid_POA_PSO พร้อมกำหนดพารามิเตอร์ทั้งหมด
    best_w, best_loss = hyb.run()  # รัน algorithm และรับ best solution กับ loss
    a, b, c = best_w  # แตก best solution เป็นสัมประสิทธิ์ a, b, c
    print(f"\n[RESULT] สมการชดเชย:")  # แสดงหัวข้อผลลัพธ์
    print(f"   Desired_pred = ({a:.6e}) * M^2 + ({b:.6f}) * M + ({c:.6f})")
    # แสดงสมการ quadratic ที่ได้จาก optimization

    corrected = a * measured_all ** 2 + b * measured_all + c
    # คำนวณค่าชดเชยสำหรับทุก sample ด้วยสมการที่หาได้
    residual  = corrected - desired_all  # คำนวณ residual (ค่าชดเชย - ค่าจริง) ทุก sample
    abs_err   = np.abs(residual)         # คำนวณค่า absolute error ทุก sample

    med = np.median(abs_err)                      # หาค่ามัธยฐานของ absolute error
    mad = np.median(np.abs(abs_err - med))        # หาค่า MAD (Median Absolute Deviation)
    threshold = med + 5.0 * 1.4826 * mad          # กำหนด threshold สำหรับตรวจ outlier (5*σ_robust)
    inliers = abs_err <= threshold                 # boolean mask: True = inlier, False = outlier
    n_out   = int((~inliers).sum())                # นับจำนวน outlier

    classical_mae = np.mean(np.abs(measured_all - desired_all))  # MAE ก่อน calibrate (raw error)
    mae_all = np.mean(abs_err)             # MAE หลัง calibrate (ทุก sample รวม outlier)
    mae_in  = np.mean(abs_err[inliers])   # MAE หลัง calibrate เฉพาะ inlier
    rmse_all = np.sqrt(np.mean(residual ** 2))          # RMSE หลัง calibrate (ทุก sample)
    rmse_in  = np.sqrt(np.mean(residual[inliers] ** 2)) # RMSE หลัง calibrate เฉพาะ inlier

    print("\n=== Summary ===")  # พิมพ์หัวข้อ summary
    print(f"จำนวนข้อมูล                 : {n_samples} จุด")   # แสดงจำนวน sample
    print(f"Outlier ที่ตรวจพบ (>5*MAD)  : {n_out} จุด")       # แสดงจำนวน outlier
    print("-" * 55)  # เส้นคั่น
    print(f"MAE  ก่อนคาลิเบรท (classical) : {classical_mae:7.4f} cm")  # MAE ก่อน calibrate
    print(f"MAE  หลังคาลิเบรท (ทุกจุด)    : {mae_all:7.4f} cm  "
          f"(ลด {(1 - mae_all / classical_mae) * 100:.2f}%)")  # MAE หลัง calibrate + % ที่ลดได้
    print(f"MAE  หลังคาลิเบรท (inlier)    : {mae_in:7.4f} cm  "
          f"(ลด {(1 - mae_in / classical_mae) * 100:.2f}%)")   # MAE inlier + % ที่ลดได้
    print(f"RMSE หลังคาลิเบรท (ทุกจุด)    : {rmse_all:7.4f} cm")  # RMSE ทุก sample
    print(f"RMSE หลังคาลิเบรท (inlier)    : {rmse_in:7.4f} cm")    # RMSE เฉพาะ inlier

    # --------------------------------------------------------
    # Figure: Convergence
    # --------------------------------------------------------
    history = np.array(hyb.fitness_history)  # แปลง list fitness history เป็น numpy array
    plt.figure(figsize=(8, 5))               # สร้าง figure ขนาด 8x5 นิ้ว
    it_x = np.arange(len(history))           # สร้าง array index สำหรับแกน x
    plt.plot(it_x, history, linewidth=2,
             label="Hybrid POA->PSO (Huber loss)", color="tab:orange")
    # วาดเส้น convergence curve ของ best fitness ตลอดการรัน
    plt.axvline(hyb.split_iter, color="k", linestyle="--", linewidth=1,
                label=f"switch POA -> PSO (iter {hyb.split_iter})")
    # วาดเส้นแนวตั้งแสดงจุดที่เปลี่ยนจาก POA เป็น PSO
    best_idx = int(np.argmin(history))       # หา index ของ iteration ที่ได้ fitness ดีที่สุด
    plt.plot(it_x[best_idx], history[best_idx], "ro", label="Best fitness")
    # วาดจุดแดงที่ตำแหน่ง best fitness
    plt.xlabel("Iterations")                 # กำหนด label แกน x
    plt.ylabel("Hybrid fitness function")    # กำหนด label แกน y
    plt.xlim(0, len(history) - 1)            # กำหนดช่วงแกน x ตามจำนวน iteration
    plt.grid(True, alpha=0.5)                # แสดง grid แบบ semi-transparent
    plt.legend()                             # แสดง legend
    plt.tight_layout()                       # จัดวาง layout อัตโนมัติไม่ให้ขอบตัด
    plt.show()                               # แสดงกราฟ

    # --------------------------------------------------------
    # Figure: Desired vs Optimized
    # --------------------------------------------------------
    pop_idx = np.arange(1, n_samples + 1)   # สร้าง array index 1 ถึง n_samples สำหรับแกน x
    plt.figure(figsize=(8, 6))              # สร้าง figure ขนาด 8x6 นิ้ว
    plt.plot(pop_idx, desired_all, "ro", markersize=5,
             markerfacecolor="none", label="Desired distances")
    # วาดจุดสีแดง (วงกลมกลวง) แสดงค่า desired ทุก sample
    plt.plot(pop_idx, corrected, color="darkorange", marker="*",
             linestyle="None", markersize=5,
             label="Optimized Measured Distances")
    # วาดจุดรูปดาวสีส้มแสดงค่าหลัง calibrate ทุก sample
    plt.xlabel("number of population")     # กำหนด label แกน x
    plt.ylabel("Distances (cm)")           # กำหนด label แกน y
    plt.xlim(0, max(450, n_samples + 20))  # กำหนดช่วงแกน x (อย่างน้อย 450 หรือ n+20)
    plt.ylim(0, max(desired_all) + 20)     # กำหนดช่วงแกน y เหลือ margin 20 cm ด้านบน
    plt.legend(loc="upper left")           # แสดง legend มุมบนซ้าย
    plt.grid(True, alpha=0.4)              # แสดง grid แบบ semi-transparent
    plt.tight_layout()                     # จัดวาง layout อัตโนมัติ
    plt.show()                             # แสดงกราฟ

    # --------------------------------------------------------
    # Figure: Error ก่อน vs หลังคาลิเบรท
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))             # สร้าง figure ขนาด 8x5 นิ้ว
    plt.plot(pop_idx, measured_all - desired_all, "r.",
             markersize=4, alpha=0.6, label="Error Before Optimization")
    # วาดจุดแดงแสดง error ก่อน calibrate (measured - desired) ทุก sample
    plt.plot(pop_idx, residual, color="darkorange", marker=".",
             linestyle="None", markersize=4, alpha=0.6,
             label="Error After Optimization")
    # วาดจุดสีส้มแสดง residual หลัง calibrate ทุก sample
    plt.axhline(0, color="k", linewidth=0.8)  # วาดเส้นแนวนอนที่ error = 0 เป็นเส้นอ้างอิง
    plt.xlabel("number of population")        # กำหนด label แกน x
    plt.ylabel("Error (cm)")                  # กำหนด label แกน y
    plt.grid(True, alpha=0.4)                 # แสดง grid แบบ semi-transparent
    plt.legend()                              # แสดง legend
    plt.tight_layout()                        # จัดวาง layout อัตโนมัติ
    plt.show()                                # แสดงกราฟ
