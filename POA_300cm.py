import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import requests

# ============================================================
# ส่วนที่ 1: Load Dataset (เลือกชีตตามระยะ)
# ============================================================

# ===== Load Dataset (หลายระยะพร้อมกัน) =====


# 🔥 mapping ระยะ → gid
GID_MAP = {
    20: "232904181",
    30: "874540621",
    40: "370951230",
    50: "481672648",
    60: "1763098656",
    70: "1902096278",
    80: "1460590862",
    90: "1662531076",
    100: "11469892",
    110: "455809277",
    120: "742893182",
    130: "322684667",
    140: "240594711",
    150: "1267531949",
    160: "1942282072",
    #300: "96634151",
}

# 🔥 เลือกระยะที่ต้องการโหลด
DISTANCE_LIST = list(GID_MAP.keys())

# base URL
BASE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2niB432N1Z2rFE3-SggNaLKS2jJ5UyVVWlmIvlEvshexcMACSd0IpsL-UsV-Q2AMyBr_ETi1VxUw3/pub?output=csv&single=true&gid="

all_dfs = []

for dist in DISTANCE_LIST:
    if dist not in GID_MAP:
        print(f"[WARN] Distance {dist} cm not found in GID_MAP, skip")
        continue

    sheet_url = BASE_URL + GID_MAP[dist]

    try:
        response = requests.get(sheet_url, timeout=20)
        response.raise_for_status()

        df_temp = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        df_temp.columns = df_temp.columns.str.strip()

        # ตรวจ column สำคัญ
        required_cols = ["Measured (cm)", "Desired (cm)", "Index"]
        for col in required_cols:
            if col not in df_temp.columns:
                raise KeyError(f"Missing column: {col} in {dist} cm")

        # เรียง index
        df_temp = df_temp.sort_values(by="Index").reset_index(drop=True)

        # เพิ่มคอลัมน์บอกว่ามาจากระยะไหน
        df_temp["Source Distance (cm)"] = dist

        all_dfs.append(df_temp)
        print(f"[INFO] Loaded {len(df_temp)} rows from {dist} cm")

    except Exception as e:
        print(f"[ERROR] Failed to load {dist} cm: {e}")

# รวมทุกชีต
if not all_dfs:
    raise ValueError("No data loaded from any sheet")

df = pd.concat(all_dfs, ignore_index=True)

# extract numpy
measured_distances = df["Measured (cm)"].astype(float).values
desired_distances = df["Desired (cm)"].astype(float).values
n_samples = len(df)

print(f"\n[INFO] Total loaded rows = {n_samples}")
print(df.head())

# ============================================================
# ส่วนที่ 2: Pelican Optimization Algorithm (POA)
# ============================================================


class POA:
    def __init__(self, n=431, m=2, T=10):
        self.n = n
        self.m = m
        self.T = T
        self.R = 0.2
        self.fitness_history = []

    def fitness_function(self, x):
        # x[0] = Measured, x[1] = Desired
        return abs(x[0] - x[1])

    def run(self, measured_val, desired_val):
        margin = max(1.0, 0.1 * desired_val)
        self.lb = np.array([measured_val - margin, desired_val - margin])
        self.ub = np.array([measured_val + margin, desired_val + margin])

        X = self.lb + np.random.rand(self.n, self.m) * (self.ub - self.lb)
        F = np.array([self.fitness_function(sol) for sol in X])

        self.best_fitness = np.min(F)
        self.best_solution = X[np.argmin(F)].copy()
        self.fitness_history.append(self.best_fitness)

        for t in range(1, self.T + 1):
            prey = self.lb + np.random.rand(self.m) * (self.ub - self.lb)
            # Phase 1: Exploration
            for i in range(self.n):
                I = np.random.choice([1, 2])
                if self.fitness_function(prey) < F[i]:
                    X_p1 = X[i] + np.random.rand(self.m) * (prey - I * X[i])
                else:
                    X_p1 = X[i] + np.random.rand(self.m) * (X[i] - prey)

                X_p1 = np.clip(X_p1, self.lb, self.ub)
                f_p1 = self.fitness_function(X_p1)
                if f_p1 < F[i]:
                    X[i], F[i] = X_p1, f_p1

            # Phase 2: Exploitation
            radius = self.R * (1 - t / self.T)
            for i in range(self.n):
                X_p2 = X[i] + radius * (2 * np.random.rand(self.m) - 1) * X[i]
                X_p2 = np.clip(X_p2, self.lb, self.ub)
                f_p2 = self.fitness_function(X_p2)
                if f_p2 < F[i]:
                    X[i], F[i] = X_p2, f_p2

            if np.min(F) < self.best_fitness:
                self.best_fitness = np.min(F)
                self.best_solution = X[np.argmin(F)].copy()

            self.fitness_history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


# ============================================================
# ส่วนที่ 3: เมนโปรแกรม
# ============================================================

if __name__ == "__main__":
    print("\n--- เริ่มการปรับปรุงข้อมูลด้วย POA ---")
    results_list = []

    for index, row in df.iterrows():
        m_val = float(row["Measured (cm)"])
        d_val = float(row["Desired (cm)"])
        idx_val = row["Index"]
        src_dist = row["Source Distance (cm)"]

        poa_engine = POA(n=431, T=10)
        best_sol, best_err = poa_engine.run(m_val, d_val)

        print(
            f"Distance {src_dist} cm | Index {idx_val}: "
            f"Measured={m_val:.2f}, Desired={d_val:.2f} -> POA Error={best_err:.4e}"
        )

        results_list.append(
    {
        "Source Distance (cm)": src_dist,
        "Index": idx_val,
        "Measured": m_val,
        "Desired": d_val,
        "BestMeasured": best_sol[0],
        "BestDesired": best_sol[1],
        "BestError": best_err,
        "History": poa_engine.fitness_history,
    }
)


    if results_list:
        desired_plot = np.array([r["BestDesired"] for r in results_list], dtype=float)
        measured_plot = np.array([r["BestMeasured"] for r in results_list], dtype=float)

        # sort ให้เป็นเส้นตรงสวยแบบ paper
        sort_idx = np.argsort(desired_plot)
        desired_plot = desired_plot[sort_idx]
        measured_plot = measured_plot[sort_idx]

        pop_idx = np.arange(1, len(desired_plot) + 1)

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.set_facecolor("#f0f0f0")  # ให้เหมือนพื้นหลังใน paper

        plt.plot(
            pop_idx,
            desired_plot,
            "ro",
            markersize=5,
            markerfacecolor="none",
            label="desired distances",
        )
        plt.plot(
            pop_idx,
            measured_plot,
            "b*",
            markersize=5,
            label="measured distances",
        )

        plt.xlabel("number of population")
        plt.ylabel("distances")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()