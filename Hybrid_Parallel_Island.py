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
# ส่วนที่ 2: Hybrid Parallel Island (POA || PSO)
# ------------------------------------------------------------
# แนวคิด Island Model:
#   - แบ่ง population ออกเป็น 2 sub-swarm:
#       Island A: POA   (n_poa ตัว)
#       Island B: PSO   (n_pso ตัว)
#   - ทุก iteration ทั้ง 2 ฝ่ายทำงาน "ขนานกัน" บนปัญหาเดียวกัน
#   - ทุก K iterations ทำ migration:
#       ส่ง best ของแต่ละ island ไปแทนที่ worst ของอีก island
#       (สลับข้อมูลกันคล้าย "เกาะมีเรือพายระหว่างกัน")
#   - Global best = min(island A best, island B best)
#
# ค้นหาสัมประสิทธิ์สมการชดเชย:
#     Desired_pred = a * Measured^2 + b * Measured + c
# Huber loss, m = 3 :  [a, b, c]
# ============================================================

class Hybrid_Parallel_Island:
    def __init__(self, measured, desired,
                 n_poa=25, n_pso=25, m=3, T=100,
                 migration_interval=5, n_migrants=2,
                 R=0.2,
                 w=0.7298, c1=1.49618, c2=1.49618,
                 huber_delta=3.0):
        self.measured = measured
        self.desired  = desired
        self.n_poa = n_poa
        self.n_pso = n_pso
        self.m = m
        self.T = T
        self.migration_interval = migration_interval
        self.n_migrants = n_migrants
        # POA params
        self.R = R
        # PSO params
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        self.huber_delta = huber_delta

        # history (เก็บแยก 2 island + global)
        self.hist_poa    = []
        self.hist_pso    = []
        self.hist_global = []

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

    # ---------- POA: หนึ่ง iteration ----------
    def _step_poa(self, X, F, t):
        prey   = self.lb + np.random.rand(self.m) * (self.ub - self.lb)
        f_prey = self.fitness_function(prey)
        for i in range(self.n_poa):
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

            # Phase 2: Exploitation (ใช้ (ub-lb) แทน X[i])
            radius = self.R * (1 - t / self.T)
            X_p2 = X[i] + radius * (2 * np.random.rand(self.m) - 1) * (self.ub - self.lb)
            X_p2 = np.clip(X_p2, self.lb, self.ub)
            f_p2 = self.fitness_function(X_p2)
            if f_p2 < F[i]:
                X[i], F[i] = X_p2, f_p2
        return X, F

    # ---------- PSO: หนึ่ง iteration ----------
    def _step_pso(self, X, V, pbest, pbest_fit, gbest, gbest_fit):
        for i in range(self.n_pso):
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
        return X, V, pbest, pbest_fit, gbest, gbest_fit

    # ---------- Migration ----------
    def _migrate(self, X_poa, F_poa, X_pso, F_pso,
                 pbest, pbest_fit):
        """ส่ง best k ตัวจากแต่ละ island ไปแทน worst k ของอีก island"""
        k = self.n_migrants

        # เลือก best จาก POA (จะส่งไป PSO)
        idx_poa_best = np.argsort(F_poa)[:k]
        # เลือก worst ใน PSO (จะถูกแทน)
        idx_pso_worst = np.argsort(F_pso)[-k:]
        # เลือก best จาก PSO (จะส่งไป POA)
        idx_pso_best = np.argsort(F_pso)[:k]
        # เลือก worst ใน POA (จะถูกแทน)
        idx_poa_worst = np.argsort(F_poa)[-k:]

        # snapshot ก่อนแก้ (กัน aliasing เมื่อ swap)
        migrants_poa_to_pso = X_poa[idx_poa_best].copy()
        fits_poa_to_pso     = F_poa[idx_poa_best].copy()
        migrants_pso_to_poa = X_pso[idx_pso_best].copy()
        fits_pso_to_poa     = F_pso[idx_pso_best].copy()

        # POA -> PSO  (อัปเดต X, pbest, pbest_fit ด้วย)
        for j, dst in enumerate(idx_pso_worst):
            X_pso[dst]     = migrants_poa_to_pso[j]
            F_pso[dst]     = fits_poa_to_pso[j]
            pbest[dst]     = migrants_poa_to_pso[j].copy()
            pbest_fit[dst] = fits_poa_to_pso[j]

        # PSO -> POA
        for j, dst in enumerate(idx_poa_worst):
            X_poa[dst] = migrants_pso_to_poa[j]
            F_poa[dst] = fits_pso_to_poa[j]

        return X_poa, F_poa, X_pso, F_pso, pbest, pbest_fit

    # ---------- run ----------
    def run(self):
        # reset history
        self.hist_poa    = []
        self.hist_pso    = []
        self.hist_global = []

        # ---- Init Island A (POA) ----
        X_poa = self.lb + np.random.rand(self.n_poa, self.m) * (self.ub - self.lb)
        F_poa = np.array([self.fitness_function(s) for s in X_poa])

        # ---- Init Island B (PSO) ----
        X_pso = self.lb + np.random.rand(self.n_pso, self.m) * (self.ub - self.lb)
        V_pso = np.random.uniform(-self.v_max, self.v_max, (self.n_pso, self.m))
        pbest     = X_pso.copy()
        pbest_fit = np.array([self.fitness_function(p) for p in X_pso])
        g_idx     = int(np.argmin(pbest_fit))
        gbest     = pbest[g_idx].copy()
        gbest_fit = float(pbest_fit[g_idx])

        # global best (จาก 2 island รวมกัน)
        best_poa_fit = float(np.min(F_poa))
        best_poa_sol = X_poa[np.argmin(F_poa)].copy()
        if best_poa_fit < gbest_fit:
            global_best_fit = best_poa_fit
            global_best_sol = best_poa_sol.copy()
        else:
            global_best_fit = gbest_fit
            global_best_sol = gbest.copy()

        self.hist_poa.append(best_poa_fit)
        self.hist_pso.append(gbest_fit)
        self.hist_global.append(global_best_fit)

        n_migrations = 0
        for t in range(1, self.T + 1):
            # ---- ทั้ง 2 island ทำงาน "ขนานกัน" ----
            X_poa, F_poa = self._step_poa(X_poa, F_poa, t)
            (X_pso, V_pso, pbest, pbest_fit,
             gbest, gbest_fit) = self._step_pso(
                X_pso, V_pso, pbest, pbest_fit, gbest, gbest_fit)

            # ---- Migration ทุก K iter ----
            if t % self.migration_interval == 0 and t < self.T:
                (X_poa, F_poa, X_pso, F_pso,
                 pbest, pbest_fit) = self._migrate(
                    X_poa, F_poa, X_pso, np.copy(pbest_fit),
                    pbest, pbest_fit)
                n_migrations += 1

            # ---- update bests ----
            cur_poa = float(np.min(F_poa))
            if cur_poa < best_poa_fit:
                best_poa_fit = cur_poa
                best_poa_sol = X_poa[np.argmin(F_poa)].copy()

            # global
            if best_poa_fit < global_best_fit:
                global_best_fit = best_poa_fit
                global_best_sol = best_poa_sol.copy()
            if gbest_fit < global_best_fit:
                global_best_fit = gbest_fit
                global_best_sol = gbest.copy()

            self.hist_poa.append(best_poa_fit)
            self.hist_pso.append(gbest_fit)
            self.hist_global.append(global_best_fit)

        print(f"[Island] migrations performed: {n_migrations} times "
              f"(every {self.migration_interval} iter)")

        self.best_solution = global_best_sol
        self.best_fitness  = global_best_fit
        return global_best_sol, global_best_fit


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

    print("\n--- Hybrid Parallel Island (POA || PSO) ---")
    hyb = Hybrid_Parallel_Island(measured_all, desired_all,
                                 n_poa=25, n_pso=25, m=3, T=100,
                                 migration_interval=5, n_migrants=2,
                                 R=0.2,
                                 w=0.7298, c1=1.49618, c2=1.49618,
                                 huber_delta=3.0)
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
    # Figure 8: Convergence (3 เส้น: POA-island, PSO-island, global)
    # --------------------------------------------------------
    h_poa    = np.array(hyb.hist_poa)
    h_pso    = np.array(hyb.hist_pso)
    h_global = np.array(hyb.hist_global)
    it_x     = np.arange(len(h_global))

    plt.figure(figsize=(8, 5))
    plt.plot(it_x, h_poa,    linewidth=1.6, alpha=0.9,
             label="Island A (POA) best",  color="tab:blue")
    plt.plot(it_x, h_pso,    linewidth=1.6, alpha=0.9,
             label="Island B (PSO) best",  color="tab:green")
    plt.plot(it_x, h_global, linewidth=2.2, alpha=1.0,
             label="Global best",          color="tab:purple")

    # เส้น migration
    for t in range(hyb.migration_interval, hyb.T,
                   hyb.migration_interval):
        plt.axvline(t, color="gray", linestyle=":",
                    linewidth=0.6, alpha=0.5)

    best_idx = int(np.argmin(h_global))
    plt.plot(it_x[best_idx], h_global[best_idx], "ko",
             markersize=6, label="Best fitness")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness function")
    plt.xlim(0, len(h_global) - 1)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.title(f"Parallel Island: migrate every {hyb.migration_interval} iter "
              f"(dotted gray)")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Figure 9: Desired vs Calibrated
    # --------------------------------------------------------
    pop_idx = np.arange(1, n_samples + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(pop_idx, desired_all, "ro", markersize=5,
             markerfacecolor="none", label="desired distances")
    plt.plot(pop_idx, corrected, color="tab:purple", marker="*",
             linestyle="None", markersize=5,
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
    plt.plot(pop_idx, residual, color="tab:purple", marker=".",
             linestyle="None", markersize=4, alpha=0.6,
             label="error after calibration")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.xlabel("number of population")
    plt.ylabel("error (cm)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
