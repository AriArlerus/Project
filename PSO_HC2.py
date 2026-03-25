import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# 1. Configuration
FILE_NAME = "Distance(CM) - HC2.csv"
D_ACTUAL = 79.5
N_PARTICLES = 100
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
        best_pos, best_mae, history = run_pso(raw_data)
        
        # Calculate RMSE using Scikit-learn 
        target_final = np.full(len(best_pos), D_ACTUAL)
        final_rmse = np.sqrt(mean_squared_error(target_final, best_pos))
        
        print(f"\nFinal PSO Metrics:")
        print(f"MAE: {best_mae:.10f}")
        print(f"RMSE: {final_rmse:.10f}")

        # Plot Convergence Curve [cite: 72, 208]
        plt.plot(history)
        plt.title("PSO Convergence Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness (MAE)")
        plt.grid(True)
        plt.show()