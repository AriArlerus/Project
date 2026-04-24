"""
============================================================
Pelican Optimization Algorithm (POA)
ดึงข้อมูลจาก Google Sheets แล้วรัน POA + plot กราฟ
============================================================
อ้างอิง: Khaleel et al., IJoST 9(1) 2024
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io

# ============================================================
# URL Google Sheets
# ============================================================/

URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vR2niB432N1Z2rFE3-SggNaLKS2jJ5UyVVWlmIvlEvshexcMACSd0IpsL-UsV-Q2AMyBr_ETi1VxUw3"
    "/pub?gid=1511238558&single=true&output=csv"
)


# ============================================================
# ส่วนที่ 1: โหลดข้อมูลจาก Google Sheets
# ============================================================

def load_data(url):
    print("[โหลดข้อมูล] กำลังดึงจาก Google Sheets ...")
    try:
        # ดึงด้วย requests (รองรับ redirect ของ Google)
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        df  = pd.read_csv(io.StringIO(res.text))

        print(f"  คอลัมน์ที่พบ : {list(df.columns)}")
        print(f"  จำนวนแถว    : {len(df)}")

        # หาคอลัมน์ Measured และ Desired อัตโนมัติ
        measured_col = None
        desired_col  = None
        for col in df.columns:
            c = col.lower().strip()
            if "measured" in c:
                measured_col = col
            if "desired" in c:
                desired_col = col

        if measured_col is None or desired_col is None:
            print(f"  [ERROR] หาคอลัมน์ไม่เจอ")
            return None, None

        measured = pd.to_numeric(df[measured_col], errors="coerce").dropna().values
        desired  = pd.to_numeric(df[desired_col],  errors="coerce").dropna().values
        n        = min(len(measured), len(desired))
        measured = measured[:n]
        desired  = desired[:n]

        print(f"  โหลดสำเร็จ   : {n} คู่ค่า")
        print(f"  Measured     : {measured.min():.2f} → {measured.max():.2f} ซม.")
        print(f"  Desired      : {desired.min():.0f} → {desired.max():.0f} ซม.")
        return measured, desired

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None, None


# ============================================================
# ส่วนที่ 2: POA ตรงตามเปเปอร์
# ============================================================

class POA:
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
        self.lb = None
        self.ub = None

    # Eq.(9): error = measured - desired
    def fitness_function(self, x):
        return abs(x[0] - x[1])

    def run(self, measured_vals, desired_vals):
        # ตั้ง lb/ub ให้ครอบ gap ระหว่าง measured กับ desired ของทั้ง dataset
        # (ปรับปรุงให้ครอบคลุมข้อมูลทั้งหมด)
        margin = max(0.5, 0.01 * (desired_vals.max() - desired_vals.min()))
        self.lb = np.array([measured_vals.min() - margin, desired_vals.min() - margin])
        self.ub = np.array([measured_vals.max() + margin, desired_vals.max() + margin])

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

        return self.best_solution, self.best_fitness, X, F

# ============================================================
# ส่วนที่ 3: รันและแสดงผล
# ============================================================

# โหลดข้อมูล
measured, desired = load_data(URL)

if measured is None:
    print("\n[ERROR] โหลดข้อมูลไม่ได้ กรุณาตรวจสอบ URL หรือ internet")
else:
    # Classical Error (สมการ 9) — ก่อน POA
    classical_errors = np.abs(measured - desired)
    classical_min  = np.min(classical_errors)
    classical_mean = np.mean(classical_errors)

    print(f"\nClassical Error (min)  : {classical_min:.6f} ซม.")
    print(f"Classical Error (mean) : {classical_mean:.6f} ซม.")

    # รัน POA
    print("\n[รัน POA] ...")
    poa = POA(T=10)
    best_sol, best_err, final_X, final_F = poa.run(measured, desired)
    improvement = (1 - best_err / classical_min) * 100 if classical_min > 0 else 0

    print(f"POA Best Error          : {best_err:.2e} ซม.")
    print(f"Best [measured,desired]: [{best_sol[0]:.4f}, {best_sol[1]:.4f}]")
    print(f"ปรับปรุง               : {improvement:.4f}%")

    # สรุปตาม Table 3
    print("\n" + "=" * 50)
    print("  สรุป (ตาม Table 3 ในเปเปอร์)")
    print(f"  Classical Error : {classical_min:.6f} ซม.")
    print(f"  POA Error       : {best_err:.2e} ซม.")
    print(f"  ปรับปรุง        : {improvement:.4f}%")
    print("=" * 50)


    # ============================================================
    # ส่วนที่ 4: Plot กราฟตามเปเปอร์
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "POA Ultrasonic Sensor — Khaleel et al., IJoST 9(1) 2024",
        fontsize=13, fontweight="bold"
    )

    # ── Figure 8: Fitness Convergence ──
    ax1  = axes[0]
    hist = poa.fitness_history
    ax1.plot(np.arange(len(hist)), hist,
             "b-o", linewidth=2.5, markersize=5,
             label="POA fitness function")
    bi = int(np.argmin(hist))
    ax1.plot(bi, hist[bi], "r*", markersize=18, zorder=5,
             label=f"Best = {hist[bi]:.2e} cm")
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("POA Fitness Function (cm)", fontsize=12)
    ax1.set_title("Figure 8: POA Fitness Convergence", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    ax1.text(
        0.97, 0.95,
        f"Classical min : {classical_min:.4f} cm\n"
        f"Classical mean: {classical_mean:.4f} cm\n"
        f"POA           : {best_err:.2e} cm\n"
        f"Improvement   : {improvement:.2f}%",
        transform=ax1.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9)
    )

    # ── Figure 9: Desired vs Measured ──
    ax2   = axes[1]
    order = np.argsort(final_X[:, 0]) # เรียงตาม measured เพื่อให้เห็น trend
    n_pop = len(final_X)
    pop_idx = np.arange(n_pop)

    ax2.plot(pop_idx, final_X[order, 1], "ro", markersize=3,
             alpha=0.7, label="desired distances")
    ax2.plot(pop_idx, final_X[order, 0], "b*", markersize=3,
             alpha=0.7, label="measured distances")
    ax2.set_xlabel("number of population", fontsize=12)
    ax2.set_ylabel("distances", fontsize=12)
    ax2.set_title("Figure 9: Best POA Desired and Measured Distances", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    # ปรับ ylim/xlim ให้เหมาะสมกับข้อมูลจริง
    ax2.set_ylim(0, max(final_X.max(), desired.max(), measured.max()) + 10)
    ax2.set_xlim(0, n_pop + 10)

    plt.tight_layout()
    plt.savefig("POA_results.png", dpi=150, bbox_inches="tight")
    print("\n[บันทึกกราฟ: POA_results.png]")
    plt.show()

# สรุปผลลัพธ์ผ่านรูปภาพ
print("\n[แสดงรูปภาพสรุปผลลัพธ์]")
import io
import requests
from PIL import Image

image_url = 'https://raw.githubusercontent.com/AnuwatX/blog-images/main/POA_results_dashboard.png' # URL รูปภาพตัวอย่างที่ถูกต้อง
try:
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    plt.figure(figsize=(15, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
except:
    print("ไม่สามารถโหลดรูปภาพสรุปผลลัพธ์ได้")