import numpy as np
import pandas as pd
import io
import requests

SHEET_ID = "169b1I4Gos8UhkzDkxH6uX9ty3yaQ_8kCqRGjqnpb0dU"
GID = "1511238558"
url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
df = pd.read_csv(io.StringIO(requests.get(url, timeout=20).content.decode("utf-8")))
df.columns = df.columns.str.strip()
df = df.sort_values("Index").reset_index(drop=True)
m = df["Measured (cm)"].astype(float).values
d = df["Desired (cm)"].astype(float).values

raw_err = m - d
print(f"BEFORE | MAE={np.mean(np.abs(raw_err)):.4f} | RMSE={np.sqrt(np.mean(raw_err**2)):.4f} | Max={np.max(np.abs(raw_err)):.4f} cm")

np.random.seed(42)
lb = np.array([-0.0003, 0.90, -5.0])
ub = np.array([ 0.0003, 1.20, 12.0])
vmax = 0.2 * (ub - lb)
n, T = 50, 200

def huber(r, delta=6.0):
    absr = np.abs(r)
    return np.where(absr <= delta, 0.5*r**2, delta*(absr - 0.5*delta)).mean()

def fitness(w):
    a, b, c = w
    pred = a*m**2 + b*m + c
    return huber(pred - d)

# ---- PSO ----
np.random.seed(42)
X = lb + np.random.rand(n, 3)*(ub - lb)
V = np.random.uniform(-vmax, vmax, (n, 3))
pbest = X.copy()
pbf = np.array([fitness(p) for p in X])
gi = int(np.argmin(pbf)); gb = pbest[gi].copy(); gf = float(pbf[gi])
for t in range(1, T+1):
    for i in range(n):
        r1, r2 = np.random.rand(3), np.random.rand(3)
        V[i] = 0.7298*V[i] + 1.49618*r1*(pbest[i]-X[i]) + 1.49618*r2*(gb-X[i])
        V[i] = np.clip(V[i], -vmax, vmax); X[i] = np.clip(X[i]+V[i], lb, ub)
        fc = fitness(X[i])
        if fc < pbf[i]:
            pbest[i], pbf[i] = X[i].copy(), fc
            if fc < gf: gb, gf = X[i].copy(), float(fc)
res_pso = gb[0]*m**2 + gb[1]*m + gb[2] - d
print(f"PSO    | MAE={np.mean(np.abs(res_pso)):.4f} | RMSE={np.sqrt(np.mean(res_pso**2)):.4f} | Max={np.max(np.abs(res_pso)):.4f} cm | Fitness={gf:.6f} | a={gb[0]:.7f} b={gb[1]:.6f} c={gb[2]:.4f}")

# ---- POA ----
np.random.seed(42)
X = lb + np.random.rand(n, 3)*(ub - lb)
F = np.array([fitness(s) for s in X])
bf = float(np.min(F)); bs = X[np.argmin(F)].copy()
for t in range(1, T+1):
    prey = lb + np.random.rand(3)*(ub - lb); fp = fitness(prey)
    for i in range(n):
        I = np.random.choice([1, 2])
        Xp1 = X[i] + np.random.rand(3)*(prey - I*X[i]) if fp < F[i] else X[i] + np.random.rand(3)*(X[i] - prey)
        Xp1 = np.clip(Xp1, lb, ub); fp1 = fitness(Xp1)
        if fp1 < F[i]: X[i], F[i] = Xp1, fp1
        r = 0.2*(1 - t/T); Xp2 = np.clip(X[i] + r*(2*np.random.rand(3)-1)*(ub-lb), lb, ub)
        fp2 = fitness(Xp2)
        if fp2 < F[i]: X[i], F[i] = Xp2, fp2
    cur = float(np.min(F))
    if cur < bf: bf, bs = cur, X[np.argmin(F)].copy()
res_poa = bs[0]*m**2 + bs[1]*m + bs[2] - d
print(f"POA    | MAE={np.mean(np.abs(res_poa)):.4f} | RMSE={np.sqrt(np.mean(res_poa**2)):.4f} | Max={np.max(np.abs(res_poa)):.4f} cm | Fitness={bf:.6f} | a={bs[0]:.7f} b={bs[1]:.6f} c={bs[2]:.4f}")

# ---- Hybrid POA-PSO ----
np.random.seed(42)
T_poa, T_pso = 150, 50
X = lb + np.random.rand(n, 3)*(ub - lb)
F = np.array([fitness(s) for s in X])
hbf = float(np.min(F)); hbs = X[np.argmin(F)].copy()
for t in range(1, T_poa+1):
    prey = lb + np.random.rand(3)*(ub - lb); fp = fitness(prey)
    for i in range(n):
        I = np.random.choice([1, 2])
        Xp1 = X[i] + np.random.rand(3)*(prey - I*X[i]) if fp < F[i] else X[i] + np.random.rand(3)*(X[i] - prey)
        Xp1 = np.clip(Xp1, lb, ub); fp1 = fitness(Xp1)
        if fp1 < F[i]: X[i], F[i] = Xp1, fp1
        r = 0.2*(1 - t/T_poa); Xp2 = np.clip(X[i] + r*(2*np.random.rand(3)-1)*(ub-lb), lb, ub)
        fp2 = fitness(Xp2)
        if fp2 < F[i]: X[i], F[i] = Xp2, fp2
    cur = float(np.min(F))
    if cur < hbf: hbf, hbs = cur, X[np.argmin(F)].copy()
# PSO phase
V = np.random.uniform(-0.1*vmax, 0.1*vmax, (n, 3))
pbest = X.copy(); pbf = F.copy(); gb = hbs.copy(); gf = hbf
for t in range(1, T_pso+1):
    for i in range(n):
        r1, r2 = np.random.rand(3), np.random.rand(3)
        V[i] = 0.7298*V[i] + 1.49618*r1*(pbest[i]-X[i]) + 1.49618*r2*(gb-X[i])
        V[i] = np.clip(V[i], -vmax, vmax); X[i] = np.clip(X[i]+V[i], lb, ub)
        fc = fitness(X[i])
        if fc < pbf[i]:
            pbest[i], pbf[i] = X[i].copy(), fc
            if fc < gf: gb, gf = X[i].copy(), float(fc)
res_hyb = gb[0]*m**2 + gb[1]*m + gb[2] - d
print(f"Hybrid | MAE={np.mean(np.abs(res_hyb)):.4f} | RMSE={np.sqrt(np.mean(res_hyb**2)):.4f} | Max={np.max(np.abs(res_hyb)):.4f} cm | Fitness={gf:.6f} | a={gb[0]:.7f} b={gb[1]:.6f} c={gb[2]:.4f}")
