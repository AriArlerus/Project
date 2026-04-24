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
# ส่วนที่ 3: แสดงตารางเปรียบเทียบ
# ============================================================

def print_comparison_table(df_result):
    """แสดงตารางเปรียบเทียบค่าจริงกับค่าหลังผ่าน POA"""
    col_w = [8, 14, 14, 16, 12, 14]
    headers = ["Index", "Measured", "Desired", "POA Optimized", "Error", "Improve (%)"]
    
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    header_row = "|" + "|".join(f" {h:<{col_w[i]-1}}" for i, h in enumerate(headers)) + "|"
    
    print("\n" + "=" * 82)
    print("  ตารางเปรียบเทียบ: ค่าจริง vs ค่าหลังผ่าน POA")
    print("=" * 82)
    print(sep)
    print(header_row)
    print(sep.replace("-", "="))
    
    for _, row in df_result.iterrows():
        # คำนวณ % การปรับปรุง เทียบกับ error เดิม (Measured - Desired)
        original_err = abs(row['Measured (cm)'] - row['Desired (cm)'])
        improvement = ((original_err - row['POA Error']) / original_err * 100) if original_err > 0 else 0.0
        
        cells = [
            f" {str(row['Index']):<{col_w[0]-1}}",
            f" {row['Measured (cm)']:>{col_w[1]-2}.4f} ",
            f" {row['Desired (cm)']:>{col_w[2]-2}.4f} ",
            f" {row['POA Optimized']:>{col_w[3]-2}.4f} ",
            f" {row['POA Error']:>{col_w[4]-2}.4e} ",
            f" {improvement:>{col_w[5]-3}.2f} %  ",
        ]
        print("|" + "|".join(cells) + "|")
    
    print(sep)
    
    # สรุปค่าเฉลี่ย
    avg_err = df_result['POA Error'].mean()
    avg_orig = abs(df_result['Measured (cm)'] - df_result['Desired (cm)']).mean()
    avg_improve = ((avg_orig - avg_err) / avg_orig * 100) if avg_orig > 0 else 0.0
    print(f"  ค่าเฉลี่ย Error หลัง POA : {avg_err:.4e}   |   ปรับปรุงเฉลี่ย: {avg_improve:.2f}%")
    print("=" * 82 + "\n")

# ============================================================
# ส่วนที่ 4: เมนโปรแกรม
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
                
                poa_engine = POA(n=100, T=10)
                best_sol, best_err = poa_engine.run(m_val, d_val)
                
                # best_sol[0] คือค่า Measured ที่ถูก optimize แล้ว
                optimized_val = best_sol[0]
                
                print(f"Index {idx_val}: Measured={m_val:.4f}, Desired={d_val:.4f}, "
                      f"POA Optimized={optimized_val:.4f}, Error={best_err:.4e}")
                
                results_list.append({
                    'Index':          idx_val,
                    'Measured (cm)':  m_val,
                    'Desired (cm)':   d_val,
                    'POA Optimized':  optimized_val,   # ← ค่าหลังผ่านอัลกอริทึม
                    'POA Error':      best_err,
                    'History':        poa_engine.fitness_history
                })

            except KeyError as e:
                print(f"Error: ไม่พบคอลัมน์ชื่อ {e} ใน Google Sheet ของคุณ")
                break

        if results_list:
            # แปลงเป็น DataFrame แล้วแสดงตาราง
            df_results = pd.DataFrame(results_list)
            print_comparison_table(df_results)

            # ---- ส่วนแสดงผลด้วย pandas ก็ได้ (เพิ่มเติม) ----
            display_cols = ['Index', 'Measured (cm)', 'Desired (cm)', 'POA Optimized', 'POA Error']
            print("DataFrame สรุป:")
            print(df_results[display_cols].to_string(index=False, float_format="{:.4f}".format))

            # วาดกราฟ convergence แถวสุดท้าย
            plt.figure(figsize=(8, 5))
            plt.plot(results_list[-1]['History'], marker='o', color='blue')
            plt.title(f"POA Convergence - Index {results_list[-1]['Index']}")
            plt.xlabel("Iteration")
            plt.ylabel("Error (Fitness)")
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()