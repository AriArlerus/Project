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

except Exception as e:
    print(f"[ERROR] ไม่สามารถโหลดข้อมูลได้: {e}")
    df = pd.DataFrame()

if len(df) == 0:
    print("[ERROR] ไม่มีข้อมูลสำหรับรันอัลกอริทึม")
    raise SystemExit(1)

np.random.seed(42)
measured_all = df["Measured (cm)"].astype(float).values
desired_all  = df["Desired (cm)"].astype(float).values
raw_error = measured_all - desired_all
n_samples    = len(measured_all)

# ============================================================
# ส่วนที่ 2: คลาสของอัลกอริทึมทั้ง 3 ตัว
# ============================================================

# 1. PSO Class
class PSO_Calibration:
    def __init__(self, measured, desired, n=50, m=3, T=200, w=0.7298, c1=1.49618, c2=1.49618, huber_delta=6.0):
        self.measured, self.desired = measured, desired
        self.n, self.m, self.T = n, m, T
        self.w, self.c1, self.c2 = w, c1, c2
        self.huber_delta = huber_delta
        self.fitness_history = []
        self.lb = np.array([-0.0003,  0.90, -5.0])
        self.ub = np.array([ 0.0003,  1.20, 12.0])
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

    def run(self):
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)
        V = np.random.uniform(-self.v_max, self.v_max, (self.n, self.m))
        pbest = X.copy()
        pbest_fit = np.array([self.fitness_function(p) for p in X])
        g_idx = int(np.argmin(pbest_fit))
        gbest = pbest[g_idx].copy()
        gbest_fit = float(pbest_fit[g_idx])
        self.fitness_history = [gbest_fit]

        for t in range(1, self.T + 1):
            for i in range(self.n):
                r1, r2 = np.random.rand(self.m), np.random.rand(self.m)
                V[i] = self.w * V[i] + self.c1 * r1 * (pbest[i] - X[i]) + self.c2 * r2 * (gbest - X[i])
                V[i] = np.clip(V[i], -self.v_max, self.v_max)
                X[i] = np.clip(X[i] + V[i], self.lb, self.ub)
                fit_curr = self.fitness_function(X[i])
                if fit_curr < pbest_fit[i]:
                    pbest[i], pbest_fit[i] = X[i].copy(), fit_curr
                    if fit_curr < gbest_fit:
                        gbest, gbest_fit = X[i].copy(), float(fit_curr)
            self.fitness_history.append(gbest_fit)
        return gbest
# 2. POA Class
class POA_Calibration:
    def __init__(self, measured, desired, n=50, m=3, T=200, R=0.2, huber_delta=6.0):
        self.measured, self.desired = measured, desired
        self.n, self.m, self.T, self.R = n, m, T, R
        self.huber_delta = huber_delta
        self.fitness_history = []
        self.lb = np.array([-0.0003,  0.90, -5.0])
        self.ub = np.array([ 0.0003,  1.20, 12.0])

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

    def run(self):
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)
        F = np.array([self.fitness_function(s) for s in X])
        self.best_fitness = float(np.min(F))
        self.best_solution = X[np.argmin(F)].copy()
        self.fitness_history = [self.best_fitness]

        for t in range(1, self.T + 1):
            prey = self.lb + np.random.rand(self.m) * (self.ub - self.lb)
            f_prey = self.fitness_function(prey)
            for i in range(self.n):
                I = np.random.choice([1, 2])
                if f_prey < F[i]: X_p1 = X[i] + np.random.rand(self.m) * (prey - I * X[i])
                else:             X_p1 = X[i] + np.random.rand(self.m) * (X[i] - prey)
                X_p1 = np.clip(X_p1, self.lb, self.ub)
                f_p1 = self.fitness_function(X_p1)
                if f_p1 < F[i]: X[i], F[i] = X_p1, f_p1

                radius = self.R * (1 - t / self.T)
                X_p2 = X[i] + radius * (2 * np.random.rand(self.m) - 1) * (self.ub - self.lb)
                X_p2 = np.clip(X_p2, self.lb, self.ub)
                f_p2 = self.fitness_function(X_p2)
                if f_p2 < F[i]: X[i], F[i] = X_p2, f_p2

            cur = float(np.min(F))
            if cur < self.best_fitness:
                self.best_fitness = cur
                self.best_solution = X[np.argmin(F)].copy()
            self.fitness_history.append(self.best_fitness)
        return self.best_solution
    
# 3. Hybrid Class
class Hybrid_POA_PSO:
    def __init__(self, measured, desired, n=50, m=3, T_poa=150, T_pso=50, R=0.2, w=0.7298, c1=1.49618, c2=1.49618, huber_delta=6.0):
        self.measured, self.desired = measured, desired
        self.n, self.m = n, m
        self.T_poa, self.T_pso = T_poa, T_pso
        self.R, self.w, self.c1, self.c2 = R, w, c1, c2
        self.huber_delta = huber_delta
        self.fitness_history = []
        self.lb = np.array([-0.0003,  0.90, -5.0])
        self.ub = np.array([ 0.0003,  1.20, 12.0])
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

    def _run_poa(self, X):
        F = np.array([self.fitness_function(s) for s in X])
        best_fitness, best_solution = float(np.min(F)), X[np.argmin(F)].copy()
        self.fitness_history.append(best_fitness)
        for t in range(1, self.T_poa + 1):
            prey = self.lb + np.random.rand(self.m) * (self.ub - self.lb)
            f_prey = self.fitness_function(prey)
            for i in range(self.n):
                I = np.random.choice([1, 2])
                if f_prey < F[i]: X_p1 = X[i] + np.random.rand(self.m) * (prey - I * X[i])
                else:             X_p1 = X[i] + np.random.rand(self.m) * (X[i] - prey)
                X_p1 = np.clip(X_p1, self.lb, self.ub)
                f_p1 = self.fitness_function(X_p1)
                if f_p1 < F[i]: X[i], F[i] = X_p1, f_p1

                radius = self.R * (1 - t / self.T_poa)
                X_p2 = X[i] + radius * (2 * np.random.rand(self.m) - 1) * (self.ub - self.lb)
                X_p2 = np.clip(X_p2, self.lb, self.ub)
                f_p2 = self.fitness_function(X_p2)
                if f_p2 < F[i]: X[i], F[i] = X_p2, f_p2

            cur = float(np.min(F))
            if cur < best_fitness: best_fitness, best_solution = cur, X[np.argmin(F)].copy()
            self.fitness_history.append(best_fitness)
        return X, F, best_solution, best_fitness

    def _run_pso(self, X, F, gbest, gbest_fit):
        V = np.random.uniform(-0.1 * self.v_max, 0.1 * self.v_max, (self.n, self.m))
        pbest, pbest_fit = X.copy(), F.copy()
        for t in range(1, self.T_pso + 1):
            for i in range(self.n):
                r1, r2 = np.random.rand(self.m), np.random.rand(self.m)
                V[i] = self.w * V[i] + self.c1 * r1 * (pbest[i] - X[i]) + self.c2 * r2 * (gbest - X[i])
                V[i] = np.clip(V[i], -self.v_max, self.v_max)
                X[i] = np.clip(X[i] + V[i], self.lb, self.ub)
                fit_curr = self.fitness_function(X[i])
                if fit_curr < pbest_fit[i]:
                    pbest[i], pbest_fit[i] = X[i].copy(), fit_curr
                    if fit_curr < gbest_fit: gbest, gbest_fit = X[i].copy(), float(fit_curr)
            self.fitness_history.append(gbest_fit)
        return gbest

    def run(self):
        self.fitness_history = []
        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)
        X_after_poa, F_after_poa, gbest, gbest_fit = self._run_poa(X)
        best_sol = self._run_pso(X_after_poa.copy(), F_after_poa.copy(), gbest.copy(), gbest_fit)
        self.split_iter = self.T_poa + 1
        return best_sol

# ============================================================
# ส่วนที่ 3: รันโปรแกรมและรวบรวมค่า
# ============================================================
print("[INFO] รัน PSO...")
np.random.seed(42)
pso = PSO_Calibration(measured_all, desired_all)
best_pso = pso.run()
corrected_pso = best_pso[0] * measured_all**2 + best_pso[1] * measured_all + best_pso[2]
residual_pso = corrected_pso - desired_all

print("\n[INFO] รัน POA...")
np.random.seed(42)
poa = POA_Calibration(measured_all, desired_all)
best_poa = poa.run()
corrected_poa = best_poa[0] * measured_all**2 + best_poa[1] * measured_all + best_poa[2]
residual_poa = corrected_poa - desired_all

print("[INFO] รัน Hybrid POA-PSO...")
np.random.seed(42)
hyb = Hybrid_POA_PSO(measured_all, desired_all)
best_hyb = hyb.run()
corrected_hyb = best_hyb[0] * measured_all**2 + best_hyb[1] * measured_all + best_hyb[2]
residual_hyb = corrected_hyb - desired_all

def mae(x):
    return np.mean(np.abs(x))

def rmse(x):
    return np.sqrt(np.mean(x ** 2))

raw_mae = mae(raw_error)
raw_rmse = rmse(raw_error)

results = {
    "Before Optimization": (raw_mae, raw_rmse),
    "PSO": (mae(residual_pso), rmse(residual_pso)),
    "POA": (mae(residual_poa), rmse(residual_poa)),
    "Hybrid POA-PSO": (mae(residual_hyb), rmse(residual_hyb)),
}

print("\n=== Performance Comparison ===")
print(f"{'Method':<25} {'MAE (cm)':>12} {'RMSE (cm)':>12} {'MAE Reduction (%)':>18}")

for name, (m, r) in results.items():
    if name == "Before Optimization":
        reduction = 0
    else:
        reduction = (1 - m / raw_mae) * 100

    print(f"{name:<25} {m:12.4f} {r:12.4f} {reduction:18.2f}")
# ============================================================
# ส่วนที่ 4: พล็อตกราฟเปรียบเทียบแบบ Subplots (แยก 3 กราฟย่อยใน 1 รูป)
# ============================================================
pop_idx = np.arange(1, n_samples + 1)

# --------------------------------------------------------
# Figure 8: Convergence (Subplots: POA, PSO, Hybrid)
# --------------------------------------------------------
fig8, axes8 = plt.subplots(1, 3, figsize=(15, 5))
fig8.suptitle("Convergence Performance", fontsize=16, fontweight='bold')

all_fitness = np.concatenate([pso.fitness_history, poa.fitness_history, hyb.fitness_history])
y_min, y_max = np.min(all_fitness), np.max(all_fitness)
y_margin = (y_max - y_min) * 0.05

# [0] PSO
axes8[0].plot(pso.fitness_history, linewidth=2, color="tab:green")
axes8[0].plot(np.argmin(pso.fitness_history), np.min(pso.fitness_history), "go")
axes8[0].set_title("PSO Convergence")
axes8[0].set_xlabel("Iterations")
axes8[0].set_ylabel("Fitness (Huber loss)")
axes8[0].grid(True, alpha=0.5)

# [1] POA
axes8[1].plot(poa.fitness_history, linewidth=2, color="tab:blue")
axes8[1].plot(np.argmin(poa.fitness_history), np.min(poa.fitness_history), "bo")
axes8[1].set_title("POA Convergence")
axes8[1].set_xlabel("Iterations")
axes8[1].grid(True, alpha=0.5)

# [2] Hybrid
axes8[2].plot(hyb.fitness_history, linewidth=2, color="tab:orange")
axes8[2].axvline(hyb.split_iter, color="k", linestyle="--", linewidth=1, label=f"Switch at iter {hyb.split_iter}")
axes8[2].plot(np.argmin(hyb.fitness_history), np.min(hyb.fitness_history), "ro")
axes8[2].set_title("Hybrid POA-PSO Convergence")
axes8[2].set_xlabel("Iterations")
axes8[2].legend()
axes8[2].grid(True, alpha=0.5)

for ax in axes8:
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Figure 9: Desired vs Optimized (Subplots: POA, PSO, Hybrid)
# --------------------------------------------------------
# ใช้ sharey=True เพื่อให้แกน Y ของทั้ง 3 กราฟมีสเกลเท่ากันเป๊ะ
fig9, axes9 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig9.suptitle("Desired vs Optimized Distances", fontsize=16, fontweight='bold')

algorithms_9 = [
    ("PSO", corrected_pso, "g*"), 
    ("POA", corrected_poa, "b*"),
    ("Hybrid POA-PSO", corrected_hyb, "darkorange")
]

for i, (name, corrected_vals, color) in enumerate(algorithms_9):
    axes9[i].plot(pop_idx, desired_all, "ro", markersize=5, markerfacecolor="none", label="Desired Distances")
    
    if name == "Hybrid POA-PSO":
        axes9[i].plot(pop_idx, corrected_vals, color=color, marker="*", linestyle="None", markersize=5, label="Optimized")
    else:
        axes9[i].plot(pop_idx, corrected_vals, color, markersize=5, label="Optimized")
        
    axes9[i].set_title(name)
    axes9[i].set_xlabel("Measured (cm)")
    if i == 0: axes9[i].set_ylabel("Distances (cm)")
    axes9[i].set_xlim(0, max(450, n_samples + 20))
    axes9[i].set_ylim(0, max(desired_all) + 20)
    axes9[i].legend(loc="upper left")
    axes9[i].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Figure 10: Error Before vs After (Subplots: POA, PSO, Hybrid)
# --------------------------------------------------------
# ใช้ sharey=True เพื่อให้เปรียบเทียบช่วง Error ได้ชัดเจน
fig10, axes10 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig10.suptitle("Error Before vs After Optimization", fontsize=16, fontweight='bold')

algorithms_10 = [
    ("PSO", residual_pso, "g."),
    ("POA", residual_poa, "b."),
    ("Hybrid POA-PSO", residual_hyb, "darkorange")
]

for i, (name, resid_vals, color) in enumerate(algorithms_10):
    axes10[i].plot(pop_idx, raw_error, "r.", markersize=4, alpha=0.6, label="Error Before")
    
    if name == "Hybrid POA-PSO":
        axes10[i].plot(pop_idx, resid_vals, color=color, marker=".", linestyle="None", markersize=4, alpha=0.9, label="Error After")
    else:
        axes10[i].plot(pop_idx, resid_vals, color, markersize=4, alpha=0.8, label="Error After")
        
    axes10[i].axhline(0, color="k", linewidth=0.8)
    axes10[i].set_title(name)
    axes10[i].set_xlabel("Measured (cm)")
    if i == 0: axes10[i].set_ylabel("Error (cm)")
    axes10[i].legend()
    axes10[i].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()