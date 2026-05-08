import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# ============================================================
# Load Data
# ============================================================
df = pd.read_csv("D:\GitHub\Project\Distance(CM) - SensorData.csv")
df.columns = df.columns.str.strip()
df = df.sort_values("Index").reset_index(drop=True)

measured_all = df["Measured (cm)"].astype(float).values
desired_all  = df["Desired (cm)"].astype(float).values

# ============================================================
# Huber Loss + Fitness Function
# ============================================================
def huber(r, delta=6.0):
    absr = np.abs(r)
    return np.where(absr <= delta, 0.5 * r**2, delta * (absr - 0.5 * delta)).mean()

def fitness_fn(w, measured, desired):
    a, b, c = w
    pred = a * measured**2 + b * measured + c
    return huber(pred - desired)

lb = np.array([-0.0003,  0.90, -5.0])
ub = np.array([ 0.0003,  1.20, 12.0])

# ============================================================
# POA
# ============================================================
def run_poa(measured, desired, n=50, m=3, T=200, R=0.2, seed=42):
    np.random.seed(seed)
    X = lb + np.random.rand(n, m) * (ub - lb)
    F = np.array([fitness_fn(s, measured, desired) for s in X])
    bf = float(np.min(F)); bs = X[np.argmin(F)].copy()
    history = [bf]
    for t in range(1, T + 1):
        prey = lb + np.random.rand(m) * (ub - lb)
        fp   = fitness_fn(prey, measured, desired)
        for i in range(n):
            I  = np.random.choice([1, 2])
            X1 = X[i] + np.random.rand(m) * (prey - I * X[i]) if fp < F[i] \
                 else X[i] + np.random.rand(m) * (X[i] - prey)
            X1 = np.clip(X1, lb, ub)
            f1 = fitness_fn(X1, measured, desired)
            if f1 < F[i]: X[i], F[i] = X1, f1
            r  = R * (1 - t / T)
            X2 = np.clip(X[i] + r * (2 * np.random.rand(m) - 1) * (ub - lb), lb, ub)
            f2 = fitness_fn(X2, measured, desired)
            if f2 < F[i]: X[i], F[i] = X2, f2
        cur = float(np.min(F))
        if cur < bf: bf = cur; bs = X[np.argmin(F)].copy()
        history.append(bf)
    return bs, history

# ============================================================
# PSO
# ============================================================
def run_pso(measured, desired, n=50, m=3, T=200,
            w=0.7298, c1=1.49618, c2=1.49618, seed=42,
            X_init=None, F_init=None, v_scale=1.0):
    np.random.seed(seed)
    vmax = 0.2 * (ub - lb)
    X    = (lb + np.random.rand(n, m) * (ub - lb)) if X_init is None else X_init.copy()
    V    = np.random.uniform(-v_scale * vmax, v_scale * vmax, (n, m))
    pbest     = X.copy()
    pbest_fit = F_init.copy() if F_init is not None else \
                np.array([fitness_fn(p, measured, desired) for p in X])
    g_idx     = int(np.argmin(pbest_fit))
    gbest     = pbest[g_idx].copy(); gbest_fit = float(pbest_fit[g_idx])
    history   = [gbest_fit]
    for _ in range(1, T + 1):
        for i in range(n):
            r1, r2 = np.random.rand(m), np.random.rand(m)
            V[i]   = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            V[i]   = np.clip(V[i], -vmax, vmax)
            X[i]   = np.clip(X[i] + V[i], lb, ub)
            fc     = fitness_fn(X[i], measured, desired)
            if fc < pbest_fit[i]:
                pbest[i] = X[i].copy(); pbest_fit[i] = fc
                if fc < gbest_fit: gbest = X[i].copy(); gbest_fit = float(fc)
        history.append(gbest_fit)
    return gbest, history

# ============================================================
# Hybrid POA -> PSO
# ============================================================
def run_hybrid(measured, desired, n=50, m=3,
               T_poa=150, T_pso=50, R=0.2,
               w=0.7298, c1=1.49618, c2=1.49618, seed=42):
    np.random.seed(seed)
    vmax = 0.2 * (ub - lb)
    # Phase A: POA
    X = lb + np.random.rand(n, m) * (ub - lb)
    F = np.array([fitness_fn(s, measured, desired) for s in X])
    bf = float(np.min(F)); bs = X[np.argmin(F)].copy()
    history = [bf]
    for t in range(1, T_poa + 1):
        prey = lb + np.random.rand(m) * (ub - lb)
        fp   = fitness_fn(prey, measured, desired)
        for i in range(n):
            I  = np.random.choice([1, 2])
            X1 = X[i] + np.random.rand(m) * (prey - I * X[i]) if fp < F[i] \
                 else X[i] + np.random.rand(m) * (X[i] - prey)
            X1 = np.clip(X1, lb, ub)
            f1 = fitness_fn(X1, measured, desired)
            if f1 < F[i]: X[i], F[i] = X1, f1
            r  = R * (1 - t / T_poa)
            X2 = np.clip(X[i] + r * (2 * np.random.rand(m) - 1) * (ub - lb), lb, ub)
            f2 = fitness_fn(X2, measured, desired)
            if f2 < F[i]: X[i], F[i] = X2, f2
        cur = float(np.min(F))
        if cur < bf: bf = cur; bs = X[np.argmin(F)].copy()
        history.append(bf)
    # Phase B: PSO (low initial velocity)
    V = np.random.uniform(-0.1 * vmax, 0.1 * vmax, (n, m))
    pbest = X.copy(); pbest_fit = F.copy()
    gbest = bs.copy(); gbest_fit = bf
    for _ in range(1, T_pso + 1):
        for i in range(n):
            r1, r2 = np.random.rand(m), np.random.rand(m)
            V[i]   = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            V[i]   = np.clip(V[i], -vmax, vmax)
            X[i]   = np.clip(X[i] + V[i], lb, ub)
            fc     = fitness_fn(X[i], measured, desired)
            if fc < pbest_fit[i]:
                pbest[i] = X[i].copy(); pbest_fit[i] = fc
                if fc < gbest_fit: gbest = X[i].copy(); gbest_fit = float(fc)
        history.append(gbest_fit)
    return gbest, history

# ============================================================
# Run 3 Algorithms
# ============================================================
print("Running POA ...")
poa_w,  _ = run_poa(measured_all, desired_all, seed=42)

print("Running PSO ...")
pso_w,  _ = run_pso(measured_all, desired_all, seed=42)

print("Running Hybrid POA-PSO ...")
hyb_w,  _ = run_hybrid(measured_all, desired_all, seed=42)

def calibrate(w, m):
    a, b, c = w
    return a * m**2 + b * m + c

corr_poa = calibrate(poa_w, measured_all)
corr_pso = calibrate(pso_w, measured_all)
corr_hyb = calibrate(hyb_w, measured_all)

raw_err  = np.abs(measured_all - desired_all)
err_poa  = np.abs(corr_poa - desired_all)
err_pso  = np.abs(corr_pso - desired_all)
err_hyb  = np.abs(corr_hyb - desired_all)

# ============================================================
# Range-based Analysis
# ============================================================
ranges       = [(20, 100), (100, 200), (200, 300), (300, 400)]
range_labels = ["20–100 cm", "100–200 cm", "200–300 cm", "300–400 cm"]
colors       = {"Raw": "#d62728", "POA": "#1f77b4", "PSO": "#2ca02c", "Hybrid POA-PSO": "#ff7f0e"}

stats = []
for (lo, hi), label in zip(ranges, range_labels):
    mask = (desired_all >= lo) & (desired_all <= hi)
    stats.append({
        "label":    label,
        "n":        int(mask.sum()),
        "raw_mae":  float(np.mean(raw_err[mask])),
        "poa_mae":  float(np.mean(err_poa[mask])),
        "pso_mae":  float(np.mean(err_pso[mask])),
        "hyb_mae":  float(np.mean(err_hyb[mask])),
        "raw_rmse": float(np.sqrt(np.mean((measured_all[mask] - desired_all[mask])**2))),
        "poa_rmse": float(np.sqrt(np.mean((corr_poa[mask] - desired_all[mask])**2))),
        "pso_rmse": float(np.sqrt(np.mean((corr_pso[mask] - desired_all[mask])**2))),
        "hyb_rmse": float(np.sqrt(np.mean((corr_hyb[mask] - desired_all[mask])**2))),
        "mask":     mask,
    })

# ============================================================
# Figure 1: MAE & RMSE Bar Chart per Range
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Error Analysis by Distance Range\n"
             "HC-SR04 Ultrasonic Sensor — POA / PSO / Hybrid POA-PSO",
             fontsize=13, fontweight="bold", y=1.02)

x     = np.arange(len(ranges))
width = 0.18
algo_keys = [("Raw (Before Calibration)", "raw_mae", "#d62728"),
             ("POA",                      "poa_mae", "#1f77b4"),
             ("PSO",                      "pso_mae", "#2ca02c"),
             ("Hybrid POA-PSO",           "hyb_mae", "#ff7f0e")]

ax = axes[0]
for k, (label, key, color) in enumerate(algo_keys):
    vals = [s[key] for s in stats]
    bars = ax.bar(x + (k - 1.5) * width, vals, width, label=label,
                  color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=color)

ax.set_xlabel("Distance Range", fontsize=11)
ax.set_ylabel("MAE (cm)", fontsize=11)
ax.set_title("Mean Absolute Error (MAE) by Distance Range", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(range_labels, fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(s["raw_mae"] for s in stats) * 1.25)

algo_rmse = [("Raw (Before Calibration)", "raw_rmse", "#d62728"),
             ("POA",                      "poa_rmse", "#1f77b4"),
             ("PSO",                      "pso_rmse", "#2ca02c"),
             ("Hybrid POA-PSO",           "hyb_rmse", "#ff7f0e")]

ax2 = axes[1]
for k, (label, key, color) in enumerate(algo_rmse):
    vals = [s[key] for s in stats]
    bars = ax2.bar(x + (k - 1.5) * width, vals, width, label=label,
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=color)

ax2.set_xlabel("Distance Range", fontsize=11)
ax2.set_ylabel("RMSE (cm)", fontsize=11)
ax2.set_title("Root Mean Square Error (RMSE) by Distance Range", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(range_labels, fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, max(s["raw_rmse"] for s in stats) * 1.25)

plt.tight_layout()
plt.show()

# ============================================================
# Figure 2: % MAE Reduction Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

algo_red = [("POA",            "poa_mae", "#1f77b4"),
            ("PSO",            "pso_mae", "#2ca02c"),
            ("Hybrid POA-PSO", "hyb_mae", "#ff7f0e")]
width2 = 0.22

for k, (label, key, color) in enumerate(algo_red):
    reductions = [(1 - s[key] / s["raw_mae"]) * 100 for s in stats]
    bars = ax.bar(x + (k - 1) * width2, reductions, width2,
                  label=label, color=color, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
                color=color, fontweight="bold")

ax.axhline(0, color="k", linewidth=0.8)
ax.set_xlabel("Distance Range", fontsize=11)
ax.set_ylabel("MAE Reduction (%)", fontsize=11)
ax.set_title("MAE Reduction (%) by Distance Range\n"
             "Compared to Raw (Before Calibration)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(range_labels, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.show()

# ============================================================
# Figure 3: Scatter Error per Data Point (4 subplots by range)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Error Before and After Calibration by Distance Range\n"
             "(Each point = 1 measurement)",
             fontsize=13, fontweight="bold")

for idx, (s, ax) in enumerate(zip(stats, axes.flat)):
    mask        = s["mask"]
    desired_sub = desired_all[mask]

    ax.plot(desired_sub, measured_all[mask] - desired_all[mask],
            "r.", markersize=5, alpha=0.55, label="Raw (Before Calibration)")
    ax.plot(desired_sub, corr_poa[mask] - desired_all[mask],
            "b.", markersize=5, alpha=0.55, label="POA")
    ax.plot(desired_sub, corr_pso[mask] - desired_all[mask],
            "g.", markersize=5, alpha=0.55, label="PSO")
    ax.plot(desired_sub, corr_hyb[mask] - desired_all[mask],
            color="#ff7f0e", marker=".", linestyle="None",
            markersize=5, alpha=0.7, label="Hybrid POA-PSO")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    mae_txt = (f"MAE:  Raw={s['raw_mae']:.2f}  "
               f"POA={s['poa_mae']:.2f}  "
               f"PSO={s['pso_mae']:.2f}  "
               f"Hyb={s['hyb_mae']:.2f} cm")
    ax.set_title(f"{s['label']}  (n={s['n']})\n{mae_txt}", fontsize=10)
    ax.set_xlabel("True Distance (cm)", fontsize=10)
    ax.set_ylabel("Error = Measured − True (cm)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Figure 4: Box Plot of |Error| per Range
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 5.5), sharey=False)
fig.suptitle("Distribution of |Error| by Distance Range\n"
             "(Box plot — center line = median, box = Q1–Q3)",
             fontsize=13, fontweight="bold")

for idx, (s, ax) in enumerate(zip(stats, axes)):
    mask   = s["mask"]
    data   = [raw_err[mask], err_poa[mask], err_pso[mask], err_hyb[mask]]
    labels = ["Raw", "POA", "PSO", "Hybrid"]
    clrs   = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], clrs):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for whisker, color in zip(bp["whiskers"], [c for c in clrs for _ in range(2)]):
        whisker.set_color(color)
    for cap, color in zip(bp["caps"], [c for c in clrs for _ in range(2)]):
        cap.set_color(color)
    for flier, color in zip(bp["fliers"], clrs):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(s["label"], fontsize=11)
    ax.set_ylabel("|Error| (cm)" if idx == 0 else "", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Figure 5: Summary Table
# ============================================================
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis("off")

col_labels = ["Distance Range", "n",
              "Raw MAE", "POA MAE", "PSO MAE", "Hybrid MAE",
              "POA Red.", "PSO Red.", "Hybrid Red.",
              "Hybrid RMSE"]
rows = []
for s in stats:
    red_poa = (1 - s["poa_mae"] / s["raw_mae"]) * 100
    red_pso = (1 - s["pso_mae"] / s["raw_mae"]) * 100
    red_hyb = (1 - s["hyb_mae"] / s["raw_mae"]) * 100
    rows.append([
        s["label"], str(s["n"]),
        f"{s['raw_mae']:.2f}", f"{s['poa_mae']:.2f}",
        f"{s['pso_mae']:.2f}", f"{s['hyb_mae']:.2f}",
        f"{red_poa:.1f}%", f"{red_pso:.1f}%", f"{red_hyb:.1f}%",
        f"{s['hyb_rmse']:.2f}",
    ])

overall_red_poa = (1 - np.mean(err_poa) / np.mean(raw_err)) * 100
overall_red_pso = (1 - np.mean(err_pso) / np.mean(raw_err)) * 100
overall_red_hyb = (1 - np.mean(err_hyb) / np.mean(raw_err)) * 100
rows.append([
    "Overall", str(len(desired_all)),
    f"{np.mean(raw_err):.2f}", f"{np.mean(err_poa):.2f}",
    f"{np.mean(err_pso):.2f}", f"{np.mean(err_hyb):.2f}",
    f"{overall_red_poa:.1f}%", f"{overall_red_pso:.1f}%", f"{overall_red_hyb:.1f}%",
    f"{np.sqrt(np.mean((corr_hyb - desired_all)**2)):.2f}",
])

table = ax.table(cellText=rows, colLabels=col_labels,
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 1.8)

for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

for j in range(len(col_labels)):
    table[len(rows), j].set_facecolor("#ecf0f1")
    table[len(rows), j].set_text_props(fontweight="bold")

for i in range(1, len(rows)):
    table[i, 5].set_facecolor("#fff3cd")
    table[i, 8].set_facecolor("#d4edda")

ax.set_title("Summary Table: MAE and RMSE by Distance Range (unit: cm)",
             fontsize=12, fontweight="bold", pad=20)

plt.tight_layout()
plt.show()

print("\nDone!")
print("  1. MAE & RMSE bar chart by distance range")
print("  2. MAE reduction (%) bar chart by distance range")
print("  3. Scatter plot of error by distance range (4 subplots)")
print("  4. Box plot of |Error| by distance range")
print("  5. Summary table")