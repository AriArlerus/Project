import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# 1. Configuration
FILE_NAME = "Distance(CM) - HC1.csv"
D_ACTUAL = 79.5
N_PELICANS = 100      # จำนวนประชากร 
ITERATIONS = 100000   # จำนวนรอบการทำงาน 

def load_data():
    if not os.path.exists(FILE_NAME):
        return None
    df = pd.read_csv(FILE_NAME, usecols=[1], header=None, skiprows=1, nrows=10000)
    return df.iloc[:, 0].dropna().values.astype(float)

# 2. Fitness Function (MAE)
def get_fitness(x):
    return np.mean(np.abs(x - D_ACTUAL))

def run_poa(data):
    dim = len(data)

    # กำหนดขอบเขตใหม่ให้ครอบคลุม 79.5
    lb = 70.0
    ub = 85.0

    # Initialization
    X = np.random.uniform(lb, ub, (N_PELICANS, dim))
    fit = np.array([get_fitness(p) for p in X])

    # Best solution เริ่มต้น
    best_idx = np.argmin(fit)
    X_prey = X[best_idx].copy()
    f_prey = fit[best_idx]

    convergence_history = []

    for t in range(ITERATIONS):
        for i in range(N_PELICANS):
            # Phase 1: Exploration
            I = np.random.randint(1, 3)

            if f_prey < fit[i]:
                X_new = X[i] + np.random.rand(dim) * (X_prey - I * X[i])
            else:
                X_new = X[i] + np.random.rand(dim) * (X[i] - X_prey)

            # บังคับให้อยู่ในขอบเขต
            X_new = np.clip(X_new, lb, ub)

            f_new = get_fitness(X_new)
            if f_new < fit[i]:
                X[i], fit[i] = X_new.copy(), f_new

            # Phase 2: Exploitation
            R = 0.2 * (1 - t / ITERATIONS)
            X_new = X[i] + R * (2 * np.random.rand(dim) - 1) * X[i]

            # บังคับให้อยู่ในขอบเขต
            X_new = np.clip(X_new, lb, ub)

            f_new = get_fitness(X_new)
            if f_new < fit[i]:
                X[i], fit[i] = X_new.copy(), f_new

        # อัปเดต best
        current_best_idx = np.argmin(fit)
        if fit[current_best_idx] < f_prey:
            f_prey = fit[current_best_idx]
            X_prey = X[current_best_idx].copy()

        convergence_history.append(f_prey)

        if (t + 1) % 10 == 0:
            print(f"POA Iteration {t+1} | Best MAE: {f_prey:.6f}")

    return X_prey, f_prey, convergence_history

if __name__ == "__main__":
    raw_data = load_data()
    if raw_data is not None:
        # เริ่มกระบวนการประมวลผลและคาลิเบรท 
        best_pos, best_mae, history = run_poa(raw_data)
        
        # 1. คำนวณตัวชี้วัดประสิทธิภาพ (Metrics) 
        target_final = np.full(len(best_pos), D_ACTUAL)
        final_rmse = np.sqrt(mean_squared_error(target_final, best_pos))
        
        # 2. แสดงผลลัพธ์การคาลิเบรทแบบละเอียด
        print("\n" + "="*50)
        print("สรุปผลการคาลิเบรทด้วย POA (Calibration Results)")
        print("="*50)
        print(f"ระยะทางจริง (Actual Distance): {D_ACTUAL} cm")
        print(f"ค่าเฉลี่ยก่อนคาลิเบรท (Raw Mean): {np.mean(raw_data):.4f} cm")
        print(f"ค่าเฉลี่ยหลังคาลิเบรท (Calibrated Mean): {np.mean(best_pos):.4f} cm")
        print("-" * 50)
        print(f"ค่า MAE (Mean Absolute Error): {best_mae:.10f}")
        print(f"ค่า RMSE (Root Mean Square Error): {final_rmse:.10f}")
        print("-" * 50)
        
        # 3. แสดงตัวอย่างข้อมูลที่คาลิเบรทแล้ว 10 แถวแรก 
        print("ตัวอย่างข้อมูลหลังการคาลิเบรท (10 แถวแรก):")
        df_result = pd.DataFrame({
            'Raw_Data (cm)': raw_data[:10],
            'Calibrated_Data (cm)': best_pos[:10]
        })
        print(df_result.to_string(index=False))
        print("="*50)

        # 4. แสดงกราฟการลู่เข้า (Convergence Curve) 
        plt.figure(figsize=(10, 5))
        plt.plot(history, label='Best Fitness (MAE)', color='orange')
        plt.title("POA Convergence Curve for HC-SR04 Calibration")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Value (MAE)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบชื่อไฟล์และที่อยู่ของไฟล์")