import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import requests
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline

# ============================================================
# ส่วนที่ 1: Load Dataset จาก Google Sheets
# ============================================================

SHEET_NAME = "SensorData"
SHEET_ID = "169b1I4Gos8UhkzDkxH6uX9ty3yaQ_8kCqRGjqnpb0dU"
GID = "1511238558"

sheet_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"


def load_dataset():
    try:
        print(f"[INFO] กำลังดึงข้อมูลจากชีต: {SHEET_NAME}...")

        response = requests.get(sheet_url, timeout=20)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        df.columns = df.columns.str.strip()

        required_cols = ["Measured (cm)", "Desired (cm)", "Index"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"ไม่พบคอลัมน์: {col}")

        df = df.sort_values(by="Index").reset_index(drop=True)

        print(f"[SUCCESS] โหลดข้อมูลสำเร็จทั้งหมด {len(df)} แถว")
        print(df.head())

        return df

    except Exception as e:
        print(f"[ERROR] ไม่สามารถโหลดข้อมูลได้: {e}")
        return pd.DataFrame()

def plot_error_trend_before_optimization(measured_all, desired_all):
    # --------------------------------------------------------
    # Figure 7: Error vs Measured (Before Optimization)
    # Compare Ordinary Quadratic Trend vs Huber Quadratic Trend
    # --------------------------------------------------------

    # Error ก่อน Optimization
    raw_error = measured_all - desired_all

    # -----------------------------
    # 1) Ordinary quadratic trend
    #    ใช้ np.polyfit แบบเดิม
    # -----------------------------
    coeffs = np.polyfit(measured_all, raw_error, 2)
    x_fit = np.linspace(min(measured_all), max(measured_all), 200)
    y_fit_poly = np.polyval(coeffs, x_fit)

    # -----------------------------
    # 2) Huber quadratic trend
    #    Fit polynomial degree 2 ด้วย Huber Loss
    # -----------------------------
    X = measured_all.reshape(-1, 1)
    y = raw_error

    huber_model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000)
    )

    huber_model.fit(X, y)

    x_fit_reshape = x_fit.reshape(-1, 1)
    y_fit_huber = huber_model.predict(x_fit_reshape)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(7, 5))

    plt.scatter(
        measured_all,
        raw_error,
        s=20,
        alpha=0.5,
        label="Raw Error"
    )

    plt.plot(
        x_fit,
        y_fit_poly,
        linewidth=2,
        color="yellow",
        label="Quadratic Trend (Least Squares)"
    )

    plt.plot(
        x_fit,
        y_fit_huber,
        linewidth=2,
        linestyle="--",
        color="red",
        label="Quadratic Trend (Huber Loss)"
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel("Measured Distance (cm)")
    plt.ylabel("Error (cm)")
    plt.title("Error Pattern Before Optimization")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ============================================================
# ส่วนที่ 3: Main Program
# ============================================================

if __name__ == "__main__":
    df = load_dataset()

    if len(df) == 0:
        print("[ERROR] ไม่มีข้อมูลสำหรับสร้างกราฟ")
        raise SystemExit(1)

    measured_all = df["Measured (cm)"].astype(float).values
    desired_all = df["Desired (cm)"].astype(float).values

    plot_error_trend_before_optimization(measured_all, desired_all)