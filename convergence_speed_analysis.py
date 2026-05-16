"""
convergence_speed_analysis.py
=============================================================
วิเคราะห์ Convergence Speed และ Stability ของ POA / PSO / Hybrid POA-PSO
สำหรับการปรับเทียบเซนเซอร์ HC-SR04

วิธีใช้:
    python convergence_speed_analysis.py

ไฟล์ CSV ที่ต้องการ: Distance_CM__-_SensorData.csv (folder เดียวกัน)
หรือแก้ CSV_PATH ด้านล่างให้ตรงกับ path จริง
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import matplotlib
import matplotlib.pyplot as plt
import os

# ============================================================
# ตั้งค่าฟอนต์ (รองรับภาษาไทย)
# ============================================================
THAI_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/tlwg/Waree.ttf",
]
for fp in THAI_FONT_CANDIDATES:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break

# ============================================================
# โหลดข้อมูล
# ============================================================
CSV_PATH = "D:\GitHub\Project\Distance(CM) - SensorData.csv"
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df = df.sort_values("Index").reset_index(drop=True)

measured_all = df["Measured (cm)"].astype(float).values
desired_all  = df["Desired (cm)"].astype(float).values
print(f"[INFO] โหลดข้อมูล {len(df)} แถว | Desired: {desired_all.min():.0f}–{desired_all.max():.0f} cm")

# ============================================================
# Search bounds และ Huber Loss
# ============================================================
lb = np.array([-0.0003,  0.90, -5.0])
ub = np.array([ 0.0003,  1.20, 12.0])

def huber(r, delta=6.0):
    absr = np.abs(r)
    return np.where(absr <= delta, 0.5 * r**2, delta * (absr - 0.5 * delta)).mean()

def fitness_fn(w):
    a, b, c = w
    return huber(a * measured_all**2 + b * measured_all + c - desired_all)

# ============================================================
# อัลกอริทึม
# ============================================================
def run_poa(seed=42, T=200, R=0.2):
    np.random.seed(seed)
    n, m = 50, 3
    X = lb + np.random.rand(n, m) * (ub - lb)
    F = np.array([fitness_fn(s) for s in X])
    bf = float(np.min(F)); hist = [bf]
    for t in range(1, T + 1):
        prey = lb + np.random.rand(m) * (ub - lb)
        fp = fitness_fn(prey)
        for i in range(n):
            I = np.random.choice([1, 2])
            X1 = X[i] + np.random.rand(m) * (prey - I * X[i]) if fp < F[i] \
                 else X[i] + np.random.rand(m) * (X[i] - prey)
            X1 = np.clip(X1, lb, ub); f1 = fitness_fn(X1)
            if f1 < F[i]: X[i], F[i] = X1, f1
            r = R * (1 - t / T)
            X2 = np.clip(X[i] + r * (2 * np.random.rand(m) - 1) * (ub - lb), lb, ub)
            f2 = fitness_fn(X2)
            if f2 < F[i]: X[i], F[i] = X2, f2
        cur = float(np.min(F))
        if cur < bf: bf = cur
        hist.append(bf)
    return np.array(hist), X, F

def run_pso(seed=42, T=200, w=0.7298, c1=1.49618, c2=1.49618):
    np.random.seed(seed)
    vmax = 0.2 * (ub - lb); n, m = 50, 3
    X = lb + np.random.rand(n, m) * (ub - lb)
    V = np.random.uniform(-vmax, vmax, (n, m))
    pbest = X.copy()
    pf = np.array([fitness_fn(p) for p in X])
    gi = int(np.argmin(pf)); gbest = pbest[gi].copy(); gf = float(pf[gi])
    hist = [gf]
    for _ in range(T):
        for i in range(n):
            r1, r2 = np.random.rand(m), np.random.rand(m)
            V[i] = np.clip(w * V[i] + c1 * r1 * (pbest[i] - X[i])
                           + c2 * r2 * (gbest - X[i]), -vmax, vmax)
            X[i] = np.clip(X[i] + V[i], lb, ub)
            fc = fitness_fn(X[i])
            if fc < pf[i]:
                pbest[i] = X[i].copy(); pf[i] = fc
                if fc < gf: gbest = X[i].copy(); gf = float(fc)
        hist.append(gf)
    return np.array(hist)

def run_hybrid(seed=42, T_poa=150, T_pso=50, R=0.2, w=0.7298, c1=1.49618, c2=1.49618):
    np.random.seed(seed)
    vmax = 0.2 * (ub - lb); n, m = 50, 3
    X = lb + np.random.rand(n, m) * (ub - lb)
    F = np.array([fitness_fn(s) for s in X])
    bf = float(np.min(F)); bs = X[np.argmin(F)].copy(); hist = [bf]
    # Phase A: POA
    for t in range(1, T_poa + 1):
        prey = lb + np.random.rand(m) * (ub - lb); fp = fitness_fn(prey)
        for i in range(n):
            I = np.random.choice([1, 2])
            X1 = X[i] + np.random.rand(m) * (prey - I * X[i]) if fp < F[i] \
                 else X[i] + np.random.rand(m) * (X[i] - prey)
            X1 = np.clip(X1, lb, ub); f1 = fitness_fn(X1)
            if f1 < F[i]: X[i], F[i] = X1, f1
            r = R * (1 - t / T_poa)
            X2 = np.clip(X[i] + r * (2 * np.random.rand(m) - 1) * (ub - lb), lb, ub)
            f2 = fitness_fn(X2)
            if f2 < F[i]: X[i], F[i] = X2, f2
        cur = float(np.min(F))
        if cur < bf: bf = cur; bs = X[np.argmin(F)].copy()
        hist.append(bf)
    # Phase B: PSO (low initial velocity)
    V = np.random.uniform(-0.1 * vmax, 0.1 * vmax, (n, m))
    pbest = X.copy(); pf = F.copy(); gbest = bs.copy(); gf = bf
    for _ in range(T_pso):
        for i in range(n):
            r1, r2 = np.random.rand(m), np.random.rand(m)
            V[i] = np.clip(w * V[i] + c1 * r1 * (pbest[i] - X[i])
                           + c2 * r2 * (gbest - X[i]), -vmax, vmax)
            X[i] = np.clip(X[i] + V[i], lb, ub)
            fc = fitness_fn(X[i])
            if fc < pf[i]:
                pbest[i] = X[i].copy(); pf[i] = fc
                if fc < gf: gbest = X[i].copy(); gf = float(fc)
        hist.append(gf)
    return np.array(hist)

# ============================================================
# รัน 10 seeds
# ============================================================
SEEDS = [42, 7, 13, 21, 99, 123, 256, 512, 1024, 2024]
CONV_THRESHOLD = 10.15   # Huber Loss ที่ถือว่า converge เพียงพอ
T_TOTAL        = 200     # total iterations ของแต่ละอัลกอริทึม

print(f"\n[INFO] รัน {len(SEEDS)} seeds × 3 อัลกอริทึม ...")

all_hist   = {"POA": [], "PSO": [], "Hybrid": []}
final_fit  = {"POA": [], "PSO": [], "Hybrid": []}
conv_iters = {"POA": [], "PSO": [], "Hybrid": []}

for s in SEEDS:
    h_poa, _, _ = run_poa(s, T=T_TOTAL)
    h_pso       = run_pso(s, T=T_TOTAL)
    h_hyb       = run_hybrid(s, T_poa=150, T_pso=50)

    for name, h in [("POA", h_poa), ("PSO", h_pso), ("Hybrid", h_hyb)]:
        all_hist[name].append(h)
        final_fit[name].append(h[-1])
        # iteration แรกที่ถึง threshold (ถ้าไม่ถึงให้ใช้ T_TOTAL)
        idx = next((i for i, v in enumerate(h) if v <= CONV_THRESHOLD), T_TOTAL)
        conv_iters[name].append(idx)

# ============================================================
# พิมพ์ตารางสรุป
# ============================================================
print("\n" + "="*65)
print("STABILITY ANALYSIS  (final Huber Loss — 10 seeds)")
print("="*65)
print(f"{'Algorithm':12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-"*65)
for name in ["POA", "PSO", "Hybrid"]:
    v = final_fit[name]
    print(f"{name:12} {np.mean(v):10.6f} {np.std(v):10.6f} {np.min(v):10.6f} {np.max(v):10.6f}")

print("\n" + "="*65)
print(f"CONVERGENCE SPEED  (iter ถึง Huber <= {CONV_THRESHOLD})")
print("="*65)
print(f"{'Algorithm':12} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}  per seed")
print("-"*65)
for name in ["POA", "PSO", "Hybrid"]:
    v = conv_iters[name]
    print(f"{name:12} {np.mean(v):8.1f} {np.std(v):8.1f} {np.min(v):6} {np.max(v):6}  {v}")

# ============================================================
# Figure 1: Mean Convergence Curve + Std Band (3 subplots)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle("Convergence Performance — Mean ± Std (10 Seeds)",
             fontsize=13, fontweight="bold")

cfg = [
    ("POA",    "tab:blue",   axes[0]),
    ("PSO",    "tab:green",  axes[1]),
    ("Hybrid", "tab:orange", axes[2]),
]

for name, color, ax in cfg:
    hists   = np.array(all_hist[name])          # shape (10, T+1)
    mean_h  = hists.mean(axis=0)
    std_h   = hists.std(axis=0)
    iters   = np.arange(len(mean_h))

    ax.plot(iters, mean_h, color=color, linewidth=2, label="Mean fitness")
    ax.fill_between(iters, mean_h - std_h, mean_h + std_h,
                    color=color, alpha=0.2, label="±1 Std")
    ax.axhline(CONV_THRESHOLD, color="gray", linestyle="--",
               linewidth=1, label=f"Threshold {CONV_THRESHOLD}")

    # จุด mean convergence
    mean_conv = int(np.round(np.mean(conv_iters[name])))
    if mean_conv < len(mean_h):
        ax.axvline(mean_conv, color=color, linestyle=":", linewidth=1.5,
                   label=f"Mean conv. iter = {mean_conv}")

    if name == "Hybrid":
        ax.axvline(151, color="black", linestyle="--", linewidth=1,
                   label="Switch POA->PSO (iter 151)")

    ax.set_title(f"{name}  |  final mean = {np.mean(final_fit[name]):.6f}",
                 fontsize=11)
    ax.set_xlabel("Iterations", fontsize=10)
    ax.set_ylabel("Huber Loss (Fitness)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, T_TOTAL)

plt.tight_layout()
plt.savefig("convergence_mean_std.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: convergence_mean_std.png")

# ============================================================
# Figure 2: Convergence Speed Bar Chart + Stability Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Convergence Speed and Stability Comparison\n(10 Seeds)",
             fontsize=13, fontweight="bold")

names  = ["POA", "PSO", "Hybrid POA-PSO"]
keys   = ["POA", "PSO", "Hybrid"]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

# --- Bar: Mean convergence speed ---
ax = axes[0]
means_conv = [np.mean(conv_iters[k]) for k in keys]
stds_conv  = [np.std(conv_iters[k])  for k in keys]
bars = ax.bar(names, means_conv, color=colors, alpha=0.85,
              edgecolor="white", width=0.5, yerr=stds_conv,
              capsize=6, error_kw=dict(elinewidth=1.5, ecolor="black"))
for bar, v, s in zip(bars, means_conv, stds_conv):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.5,
            f"{v:.1f} iter", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel(f"Iterations to reach Huber <= {CONV_THRESHOLD}", fontsize=10)
ax.set_title("Convergence Speed\n(lower = faster)", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(means_conv) * 1.4)

# --- Bar: Stability (std of final fitness) ---
ax2 = axes[1]
stds_final = [np.std(final_fit[k]) for k in keys]
bars2 = ax2.bar(names, stds_final, color=colors, alpha=0.85,
                edgecolor="white", width=0.5)
for bar, v in zip(bars2, stds_final):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(stds_final) * 0.02,
             f"{v:.6f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_ylabel("Std of Final Huber Loss (10 seeds)", fontsize=10)
ax2.set_title("Stability\n(lower std = more stable)", fontsize=11)
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, max(stds_final) * 1.4)

plt.tight_layout()
plt.savefig("convergence_speed_stability_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: convergence_speed_stability_bar.png")

# ============================================================
# Figure 3: Box Plot ของ Convergence Iterations
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribution of Convergence Iterations and Final Fitness\n(10 Seeds)",
             fontsize=13, fontweight="bold")

# Box: convergence iterations
ax = axes[0]
data_conv = [conv_iters[k] for k in keys]
bp = ax.boxplot(data_conv, patch_artist=True, widths=0.5,
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=5, alpha=0.5))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
for w, color in zip(bp["whiskers"], [c for c in colors for _ in range(2)]):
    w.set_color(color)
for cap, color in zip(bp["caps"], [c for c in colors for _ in range(2)]):
    cap.set_color(color)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel(f"Iterations to Huber <= {CONV_THRESHOLD}", fontsize=10)
ax.set_title("Convergence Speed Distribution", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Box: final fitness
ax2 = axes[1]
data_fit = [final_fit[k] for k in keys]
bp2 = ax2.boxplot(data_fit, patch_artist=True, widths=0.5,
                  medianprops=dict(color="white", linewidth=2),
                  whiskerprops=dict(linewidth=1.2),
                  capprops=dict(linewidth=1.2),
                  flierprops=dict(marker="o", markersize=5, alpha=0.5))
for patch, color in zip(bp2["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
for w, color in zip(bp2["whiskers"], [c for c in colors for _ in range(2)]):
    w.set_color(color)
for cap, color in zip(bp2["caps"], [c for c in colors for _ in range(2)]):
    cap.set_color(color)
ax2.set_xticklabels(names, fontsize=10)
ax2.set_ylabel("Final Huber Loss", fontsize=10)
ax2.set_title("Final Fitness Stability", fontsize=11)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("convergence_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: convergence_boxplot.png")

# ============================================================
# Figure 4: Summary Table
# ============================================================
fig, ax = plt.subplots(figsize=(13, 3.5))
ax.axis("off")

col_labels = ["Algorithm", "Mean Conv. Iter",
              "Std Conv. Iter", "Min", "Max",
              "Mean Final Fitness", "Std Final Fitness"]
rows = []
for name, key in zip(names, keys):
    ci = conv_iters[key]; ff = final_fit[key]
    rows.append([
        name,
        f"{np.mean(ci):.1f}", f"{np.std(ci):.1f}",
        str(np.min(ci)), str(np.max(ci)),
        f"{np.mean(ff):.6f}", f"{np.std(ff):.6f}",
    ])

table = ax.table(cellText=rows, colLabels=col_labels,
                 loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.0)

for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# highlight Hybrid row
for j in range(len(col_labels)):
    table[3, j].set_facecolor("#fff3cd")

ax.set_title(
    f"Convergence Speed & Stability Summary  |  threshold = Huber <= {CONV_THRESHOLD}  |  10 seeds",
    fontsize=11, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("convergence_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: convergence_summary_table.png")

print("\nเสร็จสิ้น! ไฟล์ทั้งหมด:")
print("  1. convergence_mean_std.png         — Mean curve + std band")
print("  2. convergence_speed_stability_bar.png — Bar chart speed & stability")
print("  3. convergence_boxplot.png          — Box plot distribution")
print("  4. convergence_summary_table.png    — ตารางสรุป")
