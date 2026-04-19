import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import requests

# ============================================================
# ส่วนที่ 1: Load Dataset (เลือกชีตตามระยะ)
# ============================================================

# 🔥 เปลี่ยนระยะตรงนี้
DISTANCE_CM = 300   # เช่น 10, 20, 30, ..., 300

# 🔥 mapping ระยะ → gid (ใส่ของคุณให้ครบ)
GID_MAP = {
    20:  "232904181",
    300: "96634151",
}

# ตรวจว่ามีระยะนี้ไหม
if DISTANCE_CM not in GID_MAP:
    raise ValueError(f"Distance {DISTANCE_CM} cm not found in GID_MAP")

# base URL
BASE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2niB432N1Z2rFE3-SggNaLKS2jJ5UyVVWlmIvlEvshexcMACSd0IpsL-UsV-Q2AMyBr_ETi1VxUw3/pub?output=csv&single=true&gid="

# สร้าง URL
sheet_url = BASE_URL + GID_MAP[DISTANCE_CM]

# โหลดข้อมูล
response = requests.get(sheet_url, timeout=20)
response.raise_for_status()

df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))

# clean column
df.columns = df.columns.str.strip()

# ตรวจ column สำคัญ
required_cols = ["Measured (cm)", "Desired (cm)", "Index"]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")

# เรียง index
df = df.sort_values(by="Index").reset_index(drop=True)

# extract numpy
measured_distances = df["Measured (cm)"].astype(float).values
desired_distances  = df["Desired (cm)"].astype(float).values
n_samples = len(df)

print(f"[INFO] Loaded {n_samples} rows from {DISTANCE_CM} cm")

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
    data = load_data_from_gsheets(SHEET_CSV_URL)

    if data is not None:
        print("\n--- เริ่มการปรับปรุงข้อมูลด้วย POA ---")
        results_list = []

        for index, row in data.head(5).iterrows():
            try:
                m_val = float(row['Measured (cm)'])
                d_val = float(row['Desired (cm)'])
                idx_val = row['Index']

                poa_engine = POA(n=431, T=10)
                best_sol, best_err = poa_engine.run(m_val, d_val)

                print(f"Index {idx_val}: Measured={m_val:.2f}, Desired={d_val:.2f} -> POA Error={best_err:.4e}")

                results_list.append({
                    'Index':   idx_val,
                    'History': poa_engine.fitness_history,
                    'Measured': m_val,
                    'Desired':  d_val,
                })
            except KeyError as e:
                print(f"Error: ไม่พบคอลัมน์ชื่อ {e} ใน Google Sheet ของคุณ")
                break

        # ============================================================
        # ส่วนที่ 4: เตรียมข้อมูลตารางเปรียบเทียบ
        # ============================================================

        if results_list:
            compare_rows = []
            for index, row in data.head(5).iterrows():
                m_val = float(row['Measured (cm)'])
                d_val = float(row['Desired (cm)'])
                diff  = m_val - d_val
                compare_rows.append({
                    'Index':         row['Index'],
                    'Measured (cm)': round(m_val, 3),
                    'Desired (cm)':  round(d_val, 3),
                    'ผลต่าง (cm)':  round(diff, 3),
                    'สถานะ':        'ดี' if abs(diff) < 0.5 else 'ห่างมาก',
                })
            df_compare = pd.DataFrame(compare_rows)

            # ============================================================
            # ส่วนที่ 5: เตรียมข้อมูลกราฟ Population vs Error
            # ============================================================

            pop_sizes = [10, 20, 50, 100, 200, 300, 400, 500]
            first = compare_rows[0]
            errors_measured = []
            errors_desired  = []

            print("\n--- คำนวณ Population vs Error ---")
            for n in pop_sizes:
                poa_m = POA(n=n, T=10)
                _, err_m = poa_m.run(first['Measured (cm)'], first['Desired (cm)'])
                errors_measured.append(err_m)

                poa_d = POA(n=n, T=10)
                _, err_d = poa_d.run(first['Desired (cm)'], first['Desired (cm)'])
                errors_desired.append(err_d)
                print(f"  n={n:4d}: Error_Measured={err_m:.4e}, Error_Desired={err_d:.4e}")

            # ============================================================
            # ส่วนที่ 6: แสดงตารางและกราฟทั้งหมดพร้อมกันในหน้าต่างเดียว
            # ============================================================

            fig = plt.figure(figsize=(12, 12))
            fig.suptitle("POA Analysis Dashboard", fontsize=14, fontweight='bold', y=0.99)

            # --- แถวบน: ตารางเปรียบเทียบ ---
            ax_table = fig.add_subplot(3, 1, 1)
            ax_table.axis('off')

            col_labels = list(df_compare.columns)
            cell_data  = df_compare.values.tolist()

            tbl = ax_table.table(
                cellText=cell_data,
                colLabels=col_labels,
                cellLoc='center',
                loc='center',
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.6)

            # จัดสีหัวตาราง
            for j in range(len(col_labels)):
                tbl[0, j].set_facecolor('#2C6E9B')
                tbl[0, j].set_text_props(color='white', fontweight='bold')

            # จัดสีแถวข้อมูลตามสถานะ
            for i, row_data in enumerate(cell_data, start=1):
                status = row_data[-1]
                bg = '#E8F5E9' if status == 'ดี' else '#FFEBEE'
                for j in range(len(col_labels)):
                    tbl[i, j].set_facecolor(bg)

            ax_table.set_title("ตารางเปรียบเทียบ Measured vs Desired",
                               fontsize=11, pad=10, loc='left')

            # --- แถวกลาง: กราฟ POA Convergence ---
            ax_conv = fig.add_subplot(3, 1, 2)
            ax_conv.plot(results_list[-1]['History'], marker='o', color='#1D9E75', linewidth=2)
            ax_conv.set_title(f"POA Convergence — Index {results_list[-1]['Index']}",
                              fontsize=11, loc='left')
            ax_conv.set_xlabel("Iteration")
            ax_conv.set_ylabel("Error (Fitness)")
            ax_conv.set_yscale('log')
            ax_conv.grid(True, alpha=0.3)

            # --- แถวล่าง: กราฟ Population vs Error ---
            ax_pop = fig.add_subplot(3, 1, 3)
            ax_pop.plot(pop_sizes, errors_measured, marker='o', color='#1D9E75',
                        label=f"Measured (Index {first['Index']})", linewidth=2)
            ax_pop.plot(pop_sizes, errors_desired, marker='s', color='#378ADD',
                        linestyle='--', label=f"Desired (Index {first['Index']})", linewidth=2)
            ax_pop.set_title(f"POA Error vs จำนวนประชากร — Index {first['Index']}",
                             fontsize=11, loc='left')
            ax_pop.set_xlabel("จำนวนประชากร (n)")
            ax_pop.set_ylabel("ระยะทาง / Error (Fitness)")
            ax_pop.set_yscale('log')
            ax_pop.legend()
            ax_pop.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()