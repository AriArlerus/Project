"""
POA_paper_real.py
============================================================
Pelican Optimization Algorithm (POA) — เวอร์ชันที่ให้ค่า "จริง"
อ้างอิงโครงสร้าง POA: Khaleel et al., Indonesian Journal of
                       Science & Technology 9(1) (2024) 145-162.

ความแตกต่างจาก POA_paper_baseline.py (ของเดิม):
  - ของเดิม: optimize [measured, desired] ต่อแถว → fitness ลู่ 0
    เพราะ optimum (x[0]=x[1]=desired) อยู่ในกล่องค้นหา
    ⇒ improvement ~100% เป็น artifact ใช้อ้างอิงไม่ได้
  - ไฟล์นี้: ใช้ POA ตัวเดียวกัน (Eq.1, 4–7) แต่เปลี่ยน
    "ตัวแปรที่หา" เป็น "สัมประสิทธิ์สมการชดเชยเชิงเส้น" [a, b]
    ที่ทำให้
        corrected = a * measured + b ≈ desired
    ฟิตบน train set แล้ววัด error บน test set ที่ไม่เคยเห็น
  - improvement ที่ได้จึงเทียบ apples-to-apples:
        before:  mean |measured       - desired|   (test)
        after :  mean |a*measured+b   - desired|   (test)
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

SHEET_NAME = "SensorDataTest"
SHEET_ID = "169b1I4Gos8UhkzDkxH6uX9ty3yaQ_8kCqRGjqnpb0dU"
GID = "1445438450"
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
# ส่วนที่ 2: POA — โครงสร้างตาม Khaleel et al. (Eq.1, 4–7)
#            แต่ optimize สัมประสิทธิ์สมการชดเชย [a, b]
# ============================================================

class POA_Calibration:
    """
    Parameters อิงตาม Table 2 ของ paper:
        n = 431  (ประชากรนกกระทุง)
        T = 10   (จำนวนรอบ)
        R = 0.2

    ตัวแปรที่หา (m=2):  x = [a, b]
    Fitness:  mean(|a*measured + b - desired|)   บน train set
    """

    def __init__(self, n=431, T=10, R=0.2,
                 lb=(-2.0, -50.0), ub=(2.0, 50.0)):
        self.n = n
        self.m = 2
        self.T = T
        self.R = R
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.fitness_history = []

    def fitness_function(self, x, measured, desired):
        a, b = x[0], x[1]
        return float(np.mean(np.abs(a * measured + b - desired)))

    def run(self, measured, desired):
        # Eq.(1): สุ่มประชากรเริ่มต้น
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)
        F = np.array([self.fitness_function(s, measured, desired) for s in X])

        self.best_fitness = float(np.min(F))
        self.best_solution = X[np.argmin(F)].copy()
        self.fitness_history = [self.best_fitness]

        for t in range(1, self.T + 1):
            # ตำแหน่ง "เหยื่อ" สุ่มในพื้นที่ค้นหา
            prey = self.lb + np.random.rand(self.m) * (self.ub - self.lb)
            f_prey = self.fitness_function(prey, measured, desired)

            for i in range(self.n):
                # ----- Phase 1: Exploration (Eq.4) -----
                I = np.random.choice([1, 2])
                if f_prey < F[i]:
                    X_p1 = X[i] + np.random.rand(self.m) * (prey - I * X[i])
                else:
                    X_p1 = X[i] + np.random.rand(self.m) * (X[i] - prey)
                X_p1 = np.clip(X_p1, self.lb, self.ub)
                f_p1 = self.fitness_function(X_p1, measured, desired)
                # Eq.(5): effective updating
                if f_p1 < F[i]:
                    X[i], F[i] = X_p1, f_p1

                # ----- Phase 2: Exploitation (Eq.6) -----
                radius = self.R * (1 - t / self.T)
                X_p2 = X[i] + radius * (2 * np.random.rand(self.m) - 1) * X[i]
                X_p2 = np.clip(X_p2, self.lb, self.ub)
                f_p2 = self.fitness_function(X_p2, measured, desired)
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
# ส่วนที่ 3: เมนโปรแกรม (train/test split + รายงาน improvement จริง)
# ============================================================

if __name__ == "__main__":
    if len(df) == 0:
        print("[ERROR] ไม่มีข้อมูลสำหรับรันอัลกอริทึม")
        raise SystemExit(1)

    np.random.seed(42)

    measured_all = df["Measured (cm)"].astype(float).values
    desired_all  = df["Desired (cm)"].astype(float).values
    n_samples    = len(measured_all)

    # ---- Train/Test split (80/20) ----
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    n_train = int(0.8 * n_samples)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    m_train, d_train = measured_all[train_idx], desired_all[train_idx]
    m_test,  d_test  = measured_all[test_idx],  desired_all[test_idx]

    print("\n--- POA Real (โครงสร้างตาม paper, optimize [a, b]) ---")
    print(f"    n_train={len(train_idx)}  n_test={len(test_idx)}")
    print(f"    Parameters: n=431, T=10, R=0.2  |  fitness = mean|a*m+b-d|")

    poa = POA_Calibration(n=431, T=10, R=0.2,
                          lb=(-2.0, -50.0), ub=(2.0, 50.0))
    best_sol, best_train_err = poa.run(m_train, d_train)
    a, b = best_sol
    fitness_history = np.array(poa.fitness_history)

    # ---- ประเมิน ----
    corrected_train = a * m_train + b
    corrected_test  = a * m_test  + b
    corrected_all   = a * measured_all + b

    classical_train_err = float(np.mean(np.abs(m_train - d_train)))
    classical_test_err  = float(np.mean(np.abs(m_test  - d_test)))
    poa_test_err        = float(np.mean(np.abs(corrected_test - d_test)))

    improvement_train = (classical_train_err - best_train_err) / classical_train_err * 100.0
    improvement_test  = (classical_test_err  - poa_test_err)   / classical_test_err  * 100.0

    print("\n=== Summary (POA — ค่าจริง, ไม่ใช่ artifact) ===")
    print(f"Calibration model :  corrected = {a:.6f} * measured + {b:.6f}")
    print()
    print(f"[TRAIN]  classical err = {classical_train_err:.4f} cm")
    print(f"         POA      err = {best_train_err:.4f} cm")
    print(f"         improvement  = {improvement_train:.2f} %")
    print()
    print(f"[TEST ]  classical err = {classical_test_err:.4f} cm")
    print(f"         POA      err = {poa_test_err:.4f} cm")
    print(f"         improvement  = {improvement_test:.2f} %   <-- ค่าที่อ้างอิงได้")

    # --------------------------------------------------------
    # Figure 8: Convergence (fitness บน train set)
    # --------------------------------------------------------
    plt.figure(figsize=(7, 5))
    it_x = np.arange(len(fitness_history))
    plt.plot(it_x, fitness_history, linewidth=2,
             label="POA fitness (train MAE)", color="tab:blue")
    best_idx = int(np.argmin(fitness_history))
    plt.plot(it_x[best_idx], fitness_history[best_idx], "ro",
             label="Best fitness")
    plt.xlabel("Iterations")
    plt.ylabel("Mean |a*m + b - d|  (cm)")
    plt.xlim(0, len(fitness_history) - 1)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.title("Figure 8 — POA Real Convergence")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Figure 9: Desired vs Calibrated vs Raw (ทั้ง dataset)
    # --------------------------------------------------------
    pop_idx = np.arange(1, n_samples + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(pop_idx, desired_all, "ro", markersize=5,
             markerfacecolor="none", label="desired distances")
    plt.plot(pop_idx, corrected_all, "b*", markersize=5,
             label=f"calibrated = {a:.3f}*m + {b:.3f}")
    plt.plot(pop_idx, measured_all, "g.", markersize=3, alpha=0.5,
             label="raw measured")
    plt.xlabel("sample index")
    plt.ylabel("distances (cm)")
    plt.xlim(0, max(450, n_samples + 20))
    plt.ylim(0, max(desired_all) + 20)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.4)
    plt.title("Figure 9 — POA Real (linear calibration)")
    plt.tight_layout()
    plt.show()
