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
# ส่วนที่ 2: Hybrid PSO -> POA
# ------------------------------------------------------------
# แนวคิด:
#   Phase A (PSO):  ใช้ PSO ลู่เข้าเร็ว ๆ เพื่อหาบริเวณคำตอบที่ดี
#                   (PSO เก่งเรื่อง global convergence ใน landscape เรียบ)
#   Phase B (POA):  ส่ง population สุดท้ายของ PSO ให้ POA ใช้กลไก
#                   exploration + exploitation รอบ ๆ คำตอบ
#                   (POA ช่วย refine และหลุดจาก local optima)
#
#   ค้นหาสัมประสิทธิ์สมการชดเชย:
#       Desired_pred = a * Measured^2 + b * Measured + c
#   ใช้ Huber loss (robust ต่อ outlier),  m = 3 :  [a, b, c]
# ============================================================

class Hybrid_PSO_POA:
    def __init__(self, measured, desired,
                 n=50, m=3,
                 T_pso=50, T_poa=50,
                 w=0.7298, c1=1.49618, c2=1.49618,
                 R=0.2, huber_delta=3.0):
        self.measured = measured
        self.desired  = desired
        self.n  = n
        self.m  = m
        self.T_pso = T_pso
        self.T_poa = T_poa
        # PSO params
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        # POA params
        self.R = R
        self.huber_delta = huber_delta
        self.fitness_history = []

        self.lb = np.array([-0.001,  0.8, -20.0])
        self.ub = np.array([ 0.001,  1.2,  20.0])
        self.v_max = 0.2 * (self.ub - self.lb)

    def _huber(self, r):
        d = self.huber_delta
        absr = np.abs(r)
        quad = 0.5 * r ** 2
        lin  = d * (absr - 0.5 * d)
        return np.where(absr <= d, quad, lin).mean()

    def fitness_function(self, w):
        a, b, c = w
        pred = a * self.measured ** 2 + b * self.measured + c
        return self._huber(pred - self.desired)

    # ---------- Phase A: PSO ----------
    def _run_pso(self, X):
        V = np.random.uniform(-self.v_max, self.v_max, (self.n, self.m))
        pbest     = X.copy()
        pbest_fit = np.array([self.fitness_function(p) for p in X])
        g_idx     = int(np.argmin(pbest_fit))
        gbest     = pbest[g_idx].copy()
        gbest_fit = float(pbest_fit[g_idx])

        self.fitness_history.append(gbest_fit)

        for t in range(1, self.T_pso + 1):
            for i in range(self.n):
                r1 = np.random.rand(self.m)
                r2 = np.random.rand(self.m)
                V[i] = (self.w  * V[i]
                        + self.c1 * r1 * (pbest[i] - X[i])
                        + self.c2 * r2 * (gbest    - X[i]))
                V[i] = np.clip(V[i], -self.v_max, self.v_max)
                X[i] = np.clip(X[i] + V[i], self.lb, self.ub)
                fit_curr = self.fitness_function(X[i])
                if fit_curr < pbest_fit[i]:
                    pbest[i]     = X[i].copy()
                    pbest_fit[i] = fit_curr
                    if fit_curr < gbest_fit:
                        gbest     = X[i].copy()
                        gbest_fit = float(fit_curr)
            self.fitness_history.append(gbest_fit)

        # คืน population สุดท้าย (= pbest = ตำแหน่งที่ดีที่สุดของแต่ละตัว)
        return pbest, pbest_fit, gbest, gbest_fit

    # ---------- Phase B: POA ----------
    def _run_poa(self, X, F, best_solution, best_fitness):
        for t in range(1, self.T_poa + 1):
            prey   = self.lb + np.random.rand(self.m) * (self.ub - self.lb)
            f_prey = self.fitness_function(prey)

            for i in range(self.n):
                # Phase 1: Exploration
                I = np.random.choice([1, 2])
                if f_prey < F[i]:
                    X_p1 = X[i] + np.random.rand(self.m) * (prey - I * X[i])
                else:
                    X_p1 = X[i] + np.random.rand(self.m) * (X[i] - prey)
                X_p1 = np.clip(X_p1, self.lb, self.ub)
                f_p1 = self.fitness_function(X_p1)
                if f_p1 < F[i]:
                    X[i], F[i] = X_p1, f_p1

                # Phase 2: Exploitation
                radius = self.R * (1 - t / self.T_poa)
                X_p2 = X[i] + radius * (2 * np.random.rand(self.m) - 1) * X[i]
                X_p2 = np.clip(X_p2, self.lb, self.ub)
                f_p2 = self.fitness_function(X_p2)
                if f_p2 < F[i]:
                    X[i], F[i] = X_p2, f_p2

            cur = float(np.min(F))
            if cur < best_fitness:
                best_fitness  = cur
                best_solution = X[np.argmin(F)].copy()

            self.fitness_history.append(best_fitness)

        return best_solution, best_fitness

    # ---------- run ----------
    def run(self):
        # Init population
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)

        # Phase A: PSO
        print("[Hybrid] >> Phase A: PSO ...")
        X_after_pso, F_after_pso, gbest, gbest_fit = self._run_pso(X)
        print(f"           PSO best fitness = {gbest_fit:.6f}")

        # Phase B: POA (รับ population ที่ PSO หาไว้ดีแล้ว)
        print("[Hybrid] >> Phase B: POA refine ...")
        best_sol, best_fit = self._run_poa(
            X_after_pso.copy(), F_after_pso.copy(), gbest.copy(), gbest_fit)
        print(f"           POA refined fitness = {best_fit:.6f}")

        self.best_solution = best_sol
        self.best_fitness  = best_fit
        self.split_iter    = self.T_pso + 1   # index เส้นแบ่งบนกราฟ
        return best_sol, best_fit


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

    print("\n--- Hybrid PSO -> POA (Quadratic + Huber loss) ---")
    hyb = Hybrid_PSO_POA(measured_all, desired_all,
                         n=50, m=3,
                         T_pso=50, T_poa=50,
                         w=0.7298, c1=1.49618, c2=1.49618,
                         R=0.2, huber_delta=3.0)
    best_w, best_loss = hyb.run()
    a, b, c = best_w
    print(f"\n[RESULT] สมการชดเชย:")
    print(f"   Desired_pred = ({a:.6e}) * M^2 + ({b:.6f}) * M + ({c:.6f})")

    corrected = a * measured_all ** 2 + b * measured_all + c
    residual  = corrected - desired_all
    abs_err   = np.abs(residual)

    med = np.median(abs_err)
    mad = np.median(np.abs(abs_err - med))
    threshold = med + 5.0 * 1.4826 * mad
    inliers = abs_err <= threshold
    n_out   = int((~inliers).sum())

    classical_mae = np.mean(np.abs(measured_all - desired_all))
    mae_all = np.mean(abs_err)
    mae_in  = np.mean(abs_err[inliers])
    rmse_all = np.sqrt(np.mean(residual ** 2))
    rmse_in  = np.sqrt(np.mean(residual[inliers] ** 2))

    print("\n=== Summary ===")
    print(f"จำนวนข้อมูล                 : {n_samples} จุด")
    print(f"Outlier ที่ตรวจพบ (>5*MAD)  : {n_out} จุด")
    print("-" * 55)
    print(f"MAE  ก่อนคาลิเบรท (classical) : {classical_mae:7.4f} cm")
    print(f"MAE  หลังคาลิเบรท (ทุกจุด)    : {mae_all:7.4f} cm  "
          f"(ลด {(1 - mae_all / classical_mae) * 100:.2f}%)")
    print(f"MAE  หลังคาลิเบรท (inlier)    : {mae_in:7.4f} cm  "
          f"(ลด {(1 - mae_in / classical_mae) * 100:.2f}%)")
    print(f"RMSE หลังคาลิเบรท (ทุกจุด)    : {rmse_all:7.4f} cm")
    print(f"RMSE หลังคาลิเบรท (inlier)    : {rmse_in:7.4f} cm")

    # --------------------------------------------------------
    # Figure 8: Convergence (มีเส้นแบ่ง phase)
    # --------------------------------------------------------
    history = np.array(hyb.fitness_history)
    plt.figure(figsize=(8, 5))
    it_x = np.arange(len(history))
    plt.plot(it_x, history, linewidth=2,
             label="Hybrid PSO->POA (Huber loss)", color="tab:purple")
    plt.axvline(hyb.split_iter, color="k", linestyle="--", linewidth=1,
                label=f"switch PSO -> POA (iter {hyb.split_iter})")
    best_idx = int(np.argmin(history))
    plt.plot(it_x[best_idx], history[best_idx], "ro", label="Best fitness")
    plt.xlabel("Iterations")
    plt.ylabel("Hybrid fitness function")
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
    plt.plot(pop_idx, corrected, "m*", markersize=5,
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
    plt.plot(pop_idx, residual, "m.",
             markersize=4, alpha=0.6, label="error after calibration")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.xlabel("number of population")
    plt.ylabel("error (cm)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
