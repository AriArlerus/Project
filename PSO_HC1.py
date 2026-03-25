import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# 1. Configuration
FILE_NAME = "Distance(CM) - HC1.csv"
D_ACTUAL = 79.5
N_PARTICLES = 300
ITERATIONS = 200000

def load_data():
    if not os.path.exists(FILE_NAME):
        return None
    # Read Column B (index 1), Rows B2:B10001 
    df = pd.read_csv(FILE_NAME, usecols=[1], header=None, skiprows=1, nrows=10000)                                             
    return df.iloc[:, 0].dropna().values.astype(float)

# 2. Fitness Function (MAE) [cite: 155]
def get_fitness(x):
    # ใช้สูตร MAE ตามบทที่ 3.2.3.2 โดยใช้ NumPy (ประสิทธิภาพสูงกว่า)
    return np.mean(np.abs(x - D_ACTUAL))

def run_pso(data):
    dim = len(data)
    w, c1, c2 = 0.7298, 1.49618, 1.49618 # Standard coefficients [cite: 215]
    
    X = np.random.uniform(np.min(data), np.max(data), (N_PARTICLES, dim))
    V = np.zeros((N_PARTICLES, dim))
    
    pbest = X.copy()
    pbest_fit = np.array([get_fitness(p) for p in X])
    gbest = pbest[np.argmin(pbest_fit)].copy()
    gbest_fit = np.min(pbest_fit)
    
    convergence_history = []

    for t in range(ITERATIONS):
        for i in range(N_PARTICLES):
            r1, r2 = np.random.rand(), np.random.rand()
            # Velocity update [cite: 80, 216]
            V[i] = w*V[i] + c1*r1*(pbest[i] - X[i]) + c2*r2*(gbest - X[i])
            X[i] = X[i] + V[i]
            
            fit_curr = get_fitness(X[i])
            if fit_curr < pbest_fit[i]:
                pbest[i], pbest_fit[i] = X[i].copy(), fit_curr
                if fit_curr < gbest_fit:
                    gbest, gbest_fit = X[i].copy(), fit_curr
        
        convergence_history.append(gbest_fit)
        if (t + 1) % 10 == 0:
            print(f"PSO Iteration {t+1} | Best MAE: {gbest_fit:.6f}")
            
    return gbest, gbest_fit, convergence_history

if __name__ == "__main__":
    raw_data = load_data()
    if raw_data is not None:
        # เริ่มกระบวนการประมวลผลและคาลิเบรท [cite: 173, 187]
        best_pos, best_mae, history = run_pso(raw_data)
        
        # 1. คำนวณตัวชี้วัดประสิทธิภาพ (Metrics) [cite: 66, 207]
        target_final = np.full(len(best_pos), D_ACTUAL)
        final_rmse = np.sqrt(mean_squared_error(target_final, best_pos))
        
        # 2. แสดงผลลัพธ์การคาลิเบรทแบบละเอียด [cite: 194]
        print("\n" + "="*50)
        print("สรุปผลการคาลิเบรท (Calibration Results)")
        print("="*50)
        print(f"ระยะทางจริง (Actual Distance): {D_ACTUAL} cm")
        print(f"ค่าเฉลี่ยก่อนคาลิเบรท (Raw Mean): {np.mean(raw_data):.4f} cm")
        print(f"ค่าเฉลี่ยหลังคาลิเบรท (Calibrated Mean): {np.mean(best_pos):.4f} cm")
        print("-" * 50)
        print(f"ค่า MAE (Mean Absolute Error): {best_mae:.10f}")
        print(f"ค่า RMSE (Root Mean Square Error): {final_rmse:.10f}")
        print("-" * 50)
        
        # 3. แสดงตัวอย่างข้อมูลที่คาลิเบรทแล้ว 10 แถวแรก [cite: 194]
        print("ตัวอย่างข้อมูลหลังการคาลิเบรท (10 แถวแรก):")
        df_result = pd.DataFrame({
            'Raw_Data (cm)': raw_data[:10],
            'Calibrated_Data (cm)': best_pos[:10]
        })
        print(df_result.to_string(index=False))
        print("="*50)

        # 4. แสดงกราฟการลู่เข้า (Convergence Curve) [cite: 72, 208]
        plt.figure(figsize=(10, 5))
        plt.plot(history, label='Best Fitness (MAE)', color='green')
        plt.title("PSO Convergence Curve for HC-SR04 Calibration")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness Value (MAE)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบชื่อไฟล์และที่อยู่ของไฟล์")