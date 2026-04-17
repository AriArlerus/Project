import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import requests

# ============================================================
# ส่วนที่ 1: ดึงข้อมูลจาก Google Sheets (Public)
# ============================================================

# ใส่ URL ที่ได้จาก Publish to Web ตรงนี้บรรทัดเดียวพอครับ
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR2niB432N1Z2rFE3-SggNaLKS2jJ5UyVVWlmIvlEvshexcMACSd0IpsL-UsV-Q2AMyBr_ETi1VxUw3/pub?gid=96634151&single=true&output=csv"

def load_data_from_gsheets(sheet_url):
    try:
        response = requests.get(sheet_url)
        response.raise_for_status()
        # อ่านข้อมูลและจัดการ Encoding เผื่อมีภาษาไทย
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        df.columns = df.columns.str.strip() # ตัดช่องว่างที่ชื่อคอลัมน์
        print(f"ดึงข้อมูลสำเร็จ! จำนวนทั้งหมด: {len(df)} แถว")
        return df
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
        return None

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
    # ดึงข้อมูลจาก URL ที่กำหนดไว้ด้านบน
    data = load_data_from_gsheets(SHEET_CSV_URL)
    
    if data is not None:
        print("\n--- เริ่มการปรับปรุงข้อมูลด้วย POA ---")
        results_list = []
        
        # วนลูปประมวลผล (ในที่นี้ทดสอบ 5 แถวแรก)
        for index, row in data.head(5).iterrows():
            try:
                # ดึงค่าจากคอลัมน์ (ตรวจสอบให้แน่ใจว่าใน Sheet ชื่อตรงเป๊ะ)
                m_val = float(row['Measured (cm)'])
                d_val = float(row['Desired (cm)'])
                idx_val = row['Index']
                
                poa_engine = POA(n=100, T=10)
                best_sol, best_err = poa_engine.run(m_val, d_val)
                
                print(f"Index {idx_val}: Measured={m_val:.2f}, Desired={d_val:.2f} -> POA Error={best_err:.4e}")
                
                results_list.append({
                    'Index': idx_val,
                    'History': poa_engine.fitness_history
                })
            except KeyError as e:
                print(f"Error: ไม่พบคอลัมน์ชื่อ {e} ใน Google Sheet ของคุณ")
                break

        # วาดกราฟของแถวสุดท้ายที่ประมวลผล
        if results_list:
            plt.figure(figsize=(8, 5))
            plt.plot(results_list[-1]['History'], marker='o', color='blue')
            plt.title(f"POA Convergence - Index {results_list[-1]['Index']}")
            plt.xlabel("Iteration")
            plt.ylabel("Error (Fitness)")
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.show()