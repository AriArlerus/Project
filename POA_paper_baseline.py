"""
POA_paper_baseline.py
============================================================
Pelican Optimization Algorithm (POA) — เวอร์ชันทำตาม paper เป๊ะ
อ้างอิง: Khaleel et al., Indonesian Journal of Science & Technology
         9(1) (2024) 145-162.

หมายเหตุสำคัญ (สำหรับการเปรียบเทียบ):
  - paper ตั้ง m = 2  คือ [measured, desired]  ไม่ใช่สัมประสิทธิ์สมการ
  - fitness ตาม Eq.(9):  f = |x[0] - x[1]|     (error = measured - desired)
  - lb/ub ตั้งล้อมรอบ (measured, desired) ของแต่ละแถว
  - ผลที่ได้:  fitness ลู่เข้า ~0  เพราะ optimum (x[0]=x[1]=desired) อยู่ใน
              ขอบเขตการค้นหา → improvement ~100%
  - ข้อจำกัด: ไม่ได้สร้าง "สมการชดเชย" ที่ใช้กับ measurement ใหม่ในอนาคต
              (จึงเอาไว้เปรียบเทียบกับเวอร์ชัน POA_300cm.py ที่ปรับปรุงแล้ว)

ใช้คู่กับ POA_300cm.py (Modified POA: Quadratic + Huber loss)
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import requests

# ============================================================
# ส่วนที่ 1: Load Dataset (ดึงจาก Google Sheets)
# ============================================================

SHEET_NAME = "SensorData"
SHEET_ID = "169b1I4Gos8UhkzDkxH6uX9ty3yaQ_8kCqRGjqnpb0dU"
GID = "1511238558"
sheet_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

try:
    print(f"[INFO] กำลังดึงข้อมูลจากชีต: {SHEET_NAME}...")
    response = requests.get(sheet_url, timeout=20)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    df.columns = df.columns.str.strip()

    required_cols = ["Measured (cm)", "Desired (cm)", "Index"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"ไม่พบคอลัมน์: {col}")

    df = df.sort_values(by="Index").reset_index(drop=True)
    print(f"[SUCCESS] โหลดข้อมูลสำเร็จทั้งหมด {len(df)} แถว")
    print(df.head())

except Exception as e:
    print(f"[ERROR] ไม่สามารถโหลดข้อมูลได้: {e}")
    df = pd.DataFrame()


# ============================================================
# ส่วนที่ 2: POA — ทำตาม Khaleel et al. (2024) เป๊ะ
# ============================================================

class POA_Paper:
    """
    Parameters ตาม Table 2 ของ paper:
        n = 431  (จำนวนประชากรนกกระทุง)
        m = 2    (ตัวแปร 2 ตัว = [measured, desired])
        T = 10   (จำนวนรอบสูงสุด)
        R = 0.2
    """

    def __init__(self, n=431, m=2, T=10, R=0.2):
        self.n = n
        self.m = m
        self.T = T
        self.R = R
        self.fitness_history = []

    # Eq.(9): error = measured - desired
    def fitness_function(self, x):
        return abs(x[0] - x[1])

    def run(self, measured_val, desired_val):
        # ตั้ง lb/ub ให้ครอบ gap ระหว่าง measured กับ desired
        # (ตาม paper: "After setting the lower bound and upper bound of two
        #  pelicans' inputs (measured and desired) distances...")
        margin = max(0.5, 0.01 * abs(desired_val))
        lo = min(measured_val, desired_val) - margin
        hi = max(measured_val, desired_val) + margin
        self.lb = np.array([lo, desired_val - margin])
        self.ub = np.array([hi, desired_val + margin])

        # Eq.(1): สุ่มประชากรเริ่มต้น
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)
        F = np.array([self.fitness_function(s) for s in X])

        self.best_fitness = float(np.min(F))
        self.best_solution = X[np.argmin(F)].copy()
        self.fitness_history = [self.best_fitness]

        for t in range(1, self.T + 1):
            # ตำแหน่ง "เหยื่อ" สุ่มในพื้นที่ค้นหา
            prey = self.lb + np.random.rand(self.m) * (self.ub - self.lb)
            f_prey = self.fitness_function(prey)

            for i in range(self.n):
                # ----- Phase 1: Exploration (Eq.4) -----
                I = np.random.choice([1, 2])
                if f_prey < F[i]:
                    X_p1 = X[i] + np.random.rand(self.m) * (prey - I * X[i])
                else:
                    X_p1 = X[i] + np.random.rand(self.m) * (X[i] - prey)
                X_p1 = np.clip(X_p1, self.lb, self.ub)
                f_p1 = self.fitness_function(X_p1)
                # Eq.(5): effective updating
                if f_p1 < F[i]:
                    X[i], F[i] = X_p1, f_p1

                # ----- Phase 2: Exploitation (Eq.6) -----
                radius = self.R * (1 - t / self.T)
                X_p2 = X[i] + radius * (2 * np.random.rand(self.m) - 1) * X[i]
                X_p2 = np.clip(X_p2, self.lb, self.ub)
                f_p2 = self.fitness_function(X_p2)
                # Eq.(7)
                if f_p2 < F[i]:
                    X[i], F[i] = X_p2, f_p2

            cur = float(np.min(F))
            if cur < self.best_fitness:
                self.best_fitness = cur
                self.best_solution = X[np.argmin(F)].copy()
            self.fitness_history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# ============================================================
# ส่วนที่ 3: เมนโปรแกรม
# ============================================================

if __name__ == "__main__":
    if len(df) == 0:
        print("[ERROR] ไม่มีข้อมูลสำหรับรันอัลกอริทึม")
        raise SystemExit(1)

    np.random.seed(42)

    measured_all = df["Measured (cm)"].astype(float).values
    desired_all  = df["Desired (cm)"].astype(float).values
    n_samples    = len(measured_all)

    print("\n--- POA Baseline (ตาม Khaleel et al. 2024 เป๊ะ) ---")
    print(f"    Parameters: n=431, m=2, T=10, R=0.2  |  fitness = |x[0]-x[1]|")

    corrected_all = np.zeros(n_samples)
    best_errors   = np.zeros(n_samples)
    histories     = []

    for i in range(n_samples):
        poa = POA_Paper(n=431, m=2, T=10, R=0.2)
        best_sol, best_err = poa.run(measured_all[i], desired_all[i])

        # paper เอา x[0] (measured ใหม่) มาเป็นค่าคาลิเบรท
        corrected_all[i] = best_sol[0]
        best_errors[i]   = best_err
        histories.append(poa.fitness_history)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:3d}/{n_samples}] m={measured_all[i]:7.3f}  "
                  f"d={desired_all[i]:7.3f}  best_err={best_err:.4e}")

    # convergence เฉลี่ยทุกแถว
    fitness_history = np.mean(np.array(histories), axis=0)

    classical_error = np.mean(np.abs(measured_all - desired_all))
    poa_error       = np.mean(best_errors)
    improvement     = (classical_error - poa_error) / classical_error * 100.0

    print("\n=== Summary (POA ตาม paper เป๊ะ) ===")
    print(f"Classical error  : {classical_error:.6f} cm")
    print(f"POA best error   : {poa_error:.6e} cm")
    print(f"Improvement      : {improvement:.4f} %")
    print()
    print("[หมายเหตุ] Improvement สูง (~100%) เป็น artifact จาก search space")
    print("          ที่ครอบจุด optimum (x[0]=x[1]=desired) อยู่แล้ว")
    print("          ไม่ได้สร้างสมการชดเชยที่ใช้งานจริงได้")
    print("          → ดูเวอร์ชัน POA_300cm.py สำหรับการคาลิเบรทจริง")

    # --------------------------------------------------------
    # Figure 8: Convergence
    # --------------------------------------------------------
    plt.figure(figsize=(7, 5))
    it_x = np.arange(len(fitness_history))
    plt.plot(it_x, fitness_history, linewidth=2,
             label="POA fitness function", color="tab:blue")
    best_idx = int(np.argmin(fitness_history))
    plt.plot(it_x[best_idx], fitness_history[best_idx], "ro",
             label="Best fitness function")
    plt.xlabel("Iterations")
    plt.ylabel("POA fitness function")
    plt.xlim(0, len(fitness_history) - 1)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.title("Figure 8 — POA Baseline (paper)")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Figure 9: Desired vs Calibrated
    # --------------------------------------------------------
    pop_idx = np.arange(1, n_samples + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(pop_idx, desired_all, "ro", markersize=5,
             markerfacecolor="none", label="desired distances")
    plt.plot(pop_idx, corrected_all, "b*", markersize=5,
             label="measured distances (POA baseline)")
    plt.xlabel("number of population")
    plt.ylabel("distances (cm)")
    plt.xlim(0, max(450, n_samples + 20))
    plt.ylim(0, max(desired_all) + 20)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.4)
    plt.title("Figure 9 — POA Baseline (paper)")
    plt.tight_layout()
    plt.show()
