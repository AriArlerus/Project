import pandas as pd
import numpy as np
import os

# 1. Configuration
FILE_NAME = "Distance(CM) - HC1.csv"
D_ACTUAL = 79.5  # Actual target
N_PELICANS = 30  # Population size
MAX_ITERATIONS = 20000

def load_data(path):
    if not os.path.exists(path):
        print("Error: File not found.")
        return None
    try:
        # Read Column B (Index 1), B2:B10001
        df = pd.read_csv(path, usecols=[1], header=None, skiprows=1, nrows=10000)
        return df.iloc[:, 0].dropna().values.astype(float)
    except Exception as e:
        print("Error loading data:", e)
        return None

# 2. Fitness Function (MAE from your image)
def get_fitness(x):
    return np.mean(np.abs(x - D_ACTUAL))

# 3. POA Algorithm Implementation
def run_poa(measured_data):
    dim = len(measured_data)
    
    # Initialize Pelicans positions
    # X represents potential calibrated data sets
    X = np.random.uniform(np.min(measured_data), 
                          np.max(measured_data), 
                          (N_PELICANS, dim))
    
    # Evaluate initial fitness
    fit = np.array([get_fitness(p) for p in X])
    
    # Find the best pelican (Prey location)
    best_idx = np.argmin(fit)
    X_prey = X[best_idx].copy()
    f_prey = fit[best_idx]

    print(f"Algorithm: POA | Dimensions: {dim} | Target: {D_ACTUAL}")

    for t in range(MAX_ITERATIONS):
        for i in range(N_PELICANS):
            # --- Phase 1: Moving towards prey (Exploration) ---
            # Randomly choose a prey (could be the best or another pelican)
            k = np.random.randint(0, N_PELICANS)
            X_k = X[k].copy()
            
            I = np.random.randint(1, 3) # Parameter I (1 or 2)
            
            if fit[k] < fit[i]:
                # Move towards the prey
                X_new = X[i] + np.random.rand(dim) * (X_k - I * X[i])
            else:
                # Move away from the prey
                X_new = X[i] + np.random.rand(dim) * (X[i] - X_k)
            
            # Boundary check and Update if better (Update Phase 1)
            f_new = get_fitness(X_new)
            if f_new < fit[i]:
                X[i] = X_new.copy()
                fit[i] = f_new

            # --- Phase 2: Winging on the water surface (Exploitation) ---
            # Search locally around the current position
            R = 0.2 * (1 - t/MAX_ITERATIONS) # Radius decreases over time
            X_new = X[i] + R * (2 * np.random.rand(dim) - 1) * X[i]
            
            # Update if better (Update Phase 2)
            f_new = get_fitness(X_new)
            if f_new < fit[i]:
                X[i] = X_new.copy()
                fit[i] = f_new

        # Update global best (Prey)
        best_idx = np.argmin(fit)
        if fit[best_idx] < f_prey:
            f_prey = fit[best_idx]
            X_prey = X[best_idx].copy()

        if (t + 1) % 10 == 0:
            print(f"Iteration {t+1}/{MAX_ITERATIONS} | Best MAE: {f_prey:.8f}")

    return X_prey, f_prey

# 4. Main Program
if __name__ == "__main__":
    raw_measurements = load_data(FILE_NAME)
    
    if raw_measurements is not None:
        best_set, final_mae = run_poa(raw_measurements)
        
        print("-" * 45)
        print("POA Calibration Summary:")
        print(f"Target Value: {D_ACTUAL}")
        print(f"Final Best MAE: {final_mae:.10f}")
        print(f"Calibrated Mean: {np.mean(best_set):.4f}")
        print("-" * 45)