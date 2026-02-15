import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "logs/eval_by_level.csv"
OUT_DIR = "logs/plots"

# Choose smoothing style
SMOOTH_METHOD = "ema"   # "ema" or "rolling"
EMA_ALPHA = 0.05         # lower -> smoother (0.1 very smooth, 0.3 less smooth)
ROLLING_WINDOW = 5      # used only if SMOOTH_METHOD == "rolling"

RAW_ALPHA = 0.2         # transparency for raw line (TensorBoard-like)


def smooth_series(s: pd.Series) -> pd.Series:
    if SMOOTH_METHOD == "ema":
        return s.ewm(alpha=EMA_ALPHA, adjust=False).mean()
    if SMOOTH_METHOD == "rolling":
        return s.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    raise ValueError(f"Unknown SMOOTH_METHOD={SMOOTH_METHOD}")


def plot_metric_by_level(df: pd.DataFrame, metric: str, out_name: str):
    plt.figure()
    levels = sorted(df["eval_level"].unique())

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, lvl in enumerate(levels):
        sub = df[df["eval_level"] == lvl].sort_values("global_iter")
        x = sub["global_iter"].to_numpy()
        y_raw = sub[metric].to_numpy()
        y_smooth = smooth_series(sub[metric]).to_numpy()

        color = color_cycle[idx % len(color_cycle)]

        # Raw (transparent)
        plt.plot(
            x,
            y_raw,
            color=color,
            alpha=RAW_ALPHA,
            linewidth=1.0,
        )

        # Smoothed (solid)
        plt.plot(
            x,
            y_smooth,
            color=color,
            linewidth=2.0,
            label=f"L{lvl}",
        )

    plt.xlabel("Curriculum iteration")
    plt.ylabel(metric)

    title = f"{metric} (raw + {SMOOTH_METHOD})"
    if SMOOTH_METHOD == "ema":
        title += f" alpha={EMA_ALPHA}"
    else:
        title += f" window={ROLLING_WINDOW}"
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name))
    plt.close()

def plot_promotion_metric(df: pd.DataFrame, out_name: str):
    plt.figure()
    sub = df[df["eval_level"] == df["trained_max_difficulty"]].sort_values("global_iter")

    x = sub["global_iter"].to_numpy()
    y_raw = sub["success_rate"].to_numpy()
    y_smooth = smooth_series(sub["success_rate"]).to_numpy()

    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    plt.plot(x, y_raw, color=color, alpha=RAW_ALPHA, linewidth=1.0)
    plt.plot(x, y_smooth, color=color, linewidth=2.0)

    plt.xlabel("Curriculum iteration")
    plt.ylabel("success_rate (current level)")

    title = f"promotion metric (raw + {SMOOTH_METHOD})"
    if SMOOTH_METHOD == "ema":
        title += f" alpha={EMA_ALPHA}"
    else:
        title += f" window={ROLLING_WINDOW}"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name))
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Plot per-level curves
    plot_metric_by_level(df, "success_rate", f"success_rate_by_level_raw_plus_{SMOOTH_METHOD}.png")
    plot_metric_by_level(df, "death_rate", f"death_rate_by_level_raw_plus_{SMOOTH_METHOD}.png")
    plot_metric_by_level(df, "timeout_rate", f"timeout_rate_by_level_raw_plus_{SMOOTH_METHOD}.png")

    # Plot promotion metric (current trained difficulty only)
    plot_promotion_metric(df, f"promotion_success_rate_raw_plus_{SMOOTH_METHOD}.png")

    print(f"Saved TensorBoard-style plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
