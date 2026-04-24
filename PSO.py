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

class PSO_Calibration:
    def __init__(self, measured, desired,
                 n=50, m=3, T=100,
                 w=0.7298, c1=1.49618, c2=1.49618,
                 huber_delta=3.0):
        self.measured = measured
        self.desired  = desired
        self.n = n
        self.m = m
        self.T = T
        self.w  = w     # inertia weight
        self.c1 = c1    # cognitive coefficient
        self.c2 = c2    # social coefficient
        self.huber_delta = huber_delta
        self.fitness_history = []

        # ขอบเขตของ [a, b, c]  (เหมือน POA เพื่อเทียบกันได้ตรง ๆ)
        self.lb = np.array([-0.001,  0.8, -20.0])
        self.ub = np.array([ 0.001,  1.2,  20.0])

        # จำกัดความเร็วสูงสุดที่ 20% ของช่วง search
        self.v_max = 0.2 * (self.ub - self.lb)

    def _huber(self, r):
        """Huber loss: quadratic เมื่อ |r|<=delta, linear เมื่อเกินกว่านั้น"""
        d = self.huber_delta
        absr = np.abs(r)
        quad = 0.5 * r ** 2
        lin  = d * (absr - 0.5 * d)
        return np.where(absr <= d, quad, lin).mean()

    def fitness_function(self, w):
        a, b, c = w
        pred = a * self.measured ** 2 + b * self.measured + c
        residual = pred - self.desired
        return self._huber(residual)

    def run(self):
        # Initialization: ตำแหน่งและความเร็วสุ่ม
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)
        V = np.random.uniform(-self.v_max, self.v_max, (self.n, self.m))

        # Personal best (pbest)
        pbest     = X.copy()
        pbest_fit = np.array([self.fitness_function(p) for p in X])

        # Global best (gbest)
        g_idx     = int(np.argmin(pbest_fit))
        gbest     = pbest[g_idx].copy()
        gbest_fit = float(pbest_fit[g_idx])

        self.fitness_history = [gbest_fit]

        for t in range(1, self.T + 1):
            for i in range(self.n):
                r1 = np.random.rand(self.m)
                r2 = np.random.rand(self.m)

                # Velocity update
                V[i] = (self.w  * V[i]
                        + self.c1 * r1 * (pbest[i] - X[i])
                        + self.c2 * r2 * (gbest    - X[i]))
                # clip ความเร็ว ป้องกันการระเบิด (velocity clamping)
                V[i] = np.clip(V[i], -self.v_max, self.v_max)

                # Position update
                X[i] = X[i] + V[i]
                # clip ตำแหน่งให้อยู่ในขอบเขต search
                X[i] = np.clip(X[i], self.lb, self.ub)

                # Evaluate
                fit_curr = self.fitness_function(X[i])

                # Update pbest
                if fit_curr < pbest_fit[i]:
                    pbest[i]     = X[i].copy()
                    pbest_fit[i] = fit_curr

                    # Update gbest
                    if fit_curr < gbest_fit:
                        gbest     = X[i].copy()
                        gbest_fit = float(fit_curr)

            self.fitness_history.append(gbest_fit)

        self.best_solution = gbest
        self.best_fitness  = gbest_fit
        return gbest, gbest_fit


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

    # ----- รัน PSO หาสมการชดเชย -----
    print("\n--- เริ่มหาสมการชดเชยด้วย PSO (Quadratic + Huber loss) ---")
    pso = PSO_Calibration(measured_all, desired_all,
                          n=50, m=3, T=100,
                          w=0.7298, c1=1.49618, c2=1.49618,
                          huber_delta=3.0)
    best_w, best_loss = pso.run()
    a, b, c = best_w
    print(f"\n[RESULT] สมการชดเชย:")
    print(f"   Desired_pred = ({a:.6e}) * M^2 + ({b:.6f}) * M + ({c:.6f})")

    # ----- ใช้สมการกับ measurement -----
    corrected = a * measured_all ** 2 + b * measured_all + c
    residual  = corrected - desired_all
    abs_err   = np.abs(residual)

    # ----- แยก inlier / outlier ด้วย MAD -----
    med = np.median(abs_err)
    mad = np.median(np.abs(abs_err - med))
    threshold = med + 5.0 * 1.4826 * mad
    inliers = abs_err <= threshold
    n_out   = int((~inliers).sum())

    # ----- Metrics -----
    classical_mae = np.mean(np.abs(measured_all - desired_all))
    pso_mae_all   = np.mean(abs_err)
    pso_mae_in    = np.mean(abs_err[inliers])
    rmse_all      = np.sqrt(np.mean(residual ** 2))
    rmse_in       = np.sqrt(np.mean(residual[inliers] ** 2))

    print("\n=== Summary ===")
    print(f"จำนวนข้อมูล                 : {n_samples} จุด")
    print(f"Outlier ที่ตรวจพบ (>5*MAD)  : {n_out} จุด")
    print("-" * 55)
    print(f"MAE  ก่อนคาลิเบรท (classical) : {classical_mae:7.4f} cm")
    print(f"MAE  หลังคาลิเบรท (ทุกจุด)    : {pso_mae_all:7.4f} cm  "
          f"(ลด {(1 - pso_mae_all / classical_mae) * 100:.2f}%)")
    print(f"MAE  หลังคาลิเบรท (inlier)    : {pso_mae_in:7.4f} cm  "
          f"(ลด {(1 - pso_mae_in / classical_mae) * 100:.2f}%)")
    print(f"RMSE หลังคาลิเบรท (ทุกจุด)    : {rmse_all:7.4f} cm")
    print(f"RMSE หลังคาลิเบรท (inlier)    : {rmse_in:7.4f} cm")

    # --------------------------------------------------------
    # Figure 8: Convergence
    # --------------------------------------------------------
    history = np.array(pso.fitness_history)
    plt.figure(figsize=(7, 5))
    it_x = np.arange(len(history))
    plt.plot(it_x, history, linewidth=2,
             label="PSO fitness (Huber loss)", color="tab:green")
    best_idx = int(np.argmin(history))
    plt.plot(it_x[best_idx], history[best_idx], "ro",
             label="Best fitness")
    plt.xlabel("Iterations")
    plt.ylabel("PSO fitness function")
    plt.xlim(0, len(history) - 1)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Figure 9: Desired vs Calibrated
    # --------------------------------------------------------
    pop_idx = np.arange(1, n_samples + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(pop_idx, desired_all, "ro", markersize=5,
             markerfacecolor="none", label="desired distances")
    plt.plot(pop_idx, corrected, "g*", markersize=5,
             label="calibrated measured distances")
    plt.xlabel("number of population")
    plt.ylabel("distances (cm)")
    plt.xlim(0, max(450, n_samples + 20))
    plt.ylim(0, max(desired_all) + 20)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Figure 10: Error ก่อน vs หลังคาลิเบรท
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(pop_idx, measured_all - desired_all, "r.",
             markersize=4, alpha=0.6, label="error before calibration")
    plt.plot(pop_idx, residual, "g.",
             markersize=4, alpha=0.6, label="error after calibration")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.xlabel("number of population")
    plt.ylabel("error (cm)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
