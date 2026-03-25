import pandas as pd
import numpy as np
import os

# 1. Configuration ตามขอบเขตโครงงาน [cite: 52, 153, 155]
FILE_NAME = "Distance(CM) - HC1.csv"
D_ACTUAL = 79.5  # ระยะทางจริงที่ใช้ทดสอบ
N_POP = 50       # จำนวนประชากร (Pelicans/Particles)
MAX_ITER = 500    # จำนวนรอบต่อเฟส (รวมเป็น 100 รอบตามแผน)

def load_data(path):
    if not os.path.exists(path):
        print("Error: CSV file not found.")
        return None
    try:
        # อ่าน B2:B10001 ตามขอบเขตด้านข้อมูล [cite: 143, 145]
        df = pd.read_csv(path, usecols=[1], header=None, skiprows=1, nrows=10000)
        return df.iloc[:, 0].dropna().values.astype(float)
    except Exception as e:
        print("Error loading data:", e)
        return None

# 2. Objective Function: MAE ตามสมการในบทที่ 3.2.3.2 [cite: 154, 155]
def get_fitness(x):
    # Fitness = (1/N) * sum(|D_measured - D_actual|)
    return np.mean(np.abs(x - D_ACTUAL))

# 3. Hybrid POA-PSO Implementation [cite: 41, 156, 159]
def run_hybrid_calibration(measured_data):
    dim = len(measured_data)
    min_val, max_val = np.min(measured_data), np.max(measured_data)
    
    # --- STEP 1: POA Phase (Global Exploration) [cite: 132, 158] ---
    print("Phase 1: Running POA for Global Exploration...")
    X = np.random.uniform(min_val, max_val, (N_POP, dim))
    fit = np.array([get_fitness(p) for p in X])
    
    for t in range(MAX_ITER):
        best_idx = np.argmin(fit)
        X_prey = X[best_idx].copy()
        
        for i in range(N_POP):
            # Pelican movement logic
            I = np.random.randint(1, 3)
            X_new = X[i] + np.random.rand(dim) * (X_prey - I * X[i])
            
            f_new = get_fitness(X_new)
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new
    
    # --- STEP 2: PSO Phase (Local Exploitation) [cite: 41, 160, 162] ---
    print("Phase 2: Running PSO for Local Exploitation...")
    # นำคำตอบที่ดีที่สุดจาก POA มาเป็นจุดเริ่มต้นของ PSO [cite: 160]
    V = np.zeros((N_POP, dim))
    pbest = X.copy()
    pbest_fit = fit.copy()
    
    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    # PSO Parameters (Standard) [cite: 136]
    w, c1, c2 = 0.7, 1.5, 1.5
    
    for t in range(MAX_ITER):
        for i in range(N_POP):
            r1, r2 = np.random.rand(), np.random.rand()
            V[i] = w*V[i] + c1*r1*(pbest[i] - X[i]) + c2*r2*(gbest - X[i])
            X[i] = X[i] + V[i]
            
            f_curr = get_fitness(X[i])
            if f_curr < pbest_fit[i]:
                pbest[i], pbest_fit[i] = X[i].copy(), f_curr
                if f_curr < gbest_fit:
                    gbest, gbest_fit = X[i].copy(), f_curr
                    
        if (t + 1) % 10 == 0:
            print(f"Iteration {t+MAX_ITER} | Best MAE: {gbest_fit:.8f}")

    return gbest, gbest_fit

# 4. Main Execution ตามกระบวนการทำงาน (Workflows) [cite: 166, 175]
if __name__ == "__main__":
    # โหลดชุดข้อมูล (Data Preparation) [cite: 152]
    raw_data = load_data(FILE_NAME)
    
    if raw_data is not None:
        # เริ่มการประมวลผลและคาลิเบรท [cite: 187]
        best_calibrated, final_mae = run_hybrid_calibration(raw_data)
        
        print("\n" + "="*45)
        print("Hybrid POA-PSO Calibration Results")
        print(f"Target Distance: {D_ACTUAL} cm")
        print(f"Initial MAE: {np.mean(np.abs(raw_data - D_ACTUAL)):.6f}")
        print(f"Optimized MAE: {final_mae:.10f}")
        print(f"Improvement: {((np.mean(np.abs(raw_data - D_ACTUAL)) - final_mae) / np.mean(np.abs(raw_data - D_ACTUAL)) * 100):.2f}%")
        print("="*45)
        
        # บันทึกค่าพารามิเตอร์เพื่อนำไปใช้จริงในระบบ Prototype [cite: 48, 164]
        # pd.DataFrame(best_calibrated).to_csv("Calibrated_Data.csv", index=False)