import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _line_plot(df, x_col, y_col, group_col, title, xlabel, ylabel, out_path):
    plt.figure()
    for name, g in df.groupby(group_col):
        g = g.sort_values(x_col)
        plt.plot(g[x_col].values, g[y_col].values, marker="o", label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(sorted(df[x_col].unique()))
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _bar_plot(df, x_col, y_col, group_col, title, xlabel, ylabel, out_path):
    # grouped bar: x = levels, bars = agents
    levels = sorted(df[x_col].unique())
    agents = sorted(df[group_col].unique())

    width = 0.8 / max(1, len(agents))
    x = np.arange(len(levels))

    plt.figure()
    for i, agent in enumerate(agents):
        g = df[df[group_col] == agent].set_index(x_col).reindex(levels)
        plt.bar(x + i * width, g[y_col].values, width=width, label=agent)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x + width * (len(agents) - 1) / 2, levels)
    plt.grid(True, axis="y", linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _print_summary(df: pd.DataFrame):
    # Averages across levels (unweighted mean of per-level rates)
    cols = ["success_rate", "death_rate", "timeout_rate", "ep_len_mean", "ep_rew_mean"]
    if "mean_step_latency_ms" in df.columns:
        cols.append("mean_step_latency_ms")

    summary = df.groupby("agent")[cols].mean(numeric_only=True).sort_values("success_rate", ascending=False)
    print("\n=== Summary (mean over levels) ===")
    print(summary.round(4).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="logs/eval_all_agents.csv")
    p.add_argument("--out_dir", type=str, default="logs/plots_compare")
    p.add_argument("--bar", action="store_true", help="Use grouped bar plots (default: line plots).")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Missing file: {args.csv}. Run eval_all.py first.")

    _ensure_dir(args.out_dir)
    df = pd.read_csv(args.csv)

    # Basic validation
    needed = {"agent", "level", "success_rate", "death_rate", "timeout_rate", "ep_len_mean", "ep_rew_mean"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # Keep only rows with proper level
    df["level"] = df["level"].astype(int)

    # Plots to generate
    plots = [
        ("success_rate", "Success rate by level", "Level", "Success rate", "success_rate.png"),
        ("death_rate", "Death rate by level", "Level", "Death rate", "death_rate.png"),
        ("timeout_rate", "Timeout rate by level", "Level", "Timeout rate", "timeout_rate.png"),
        ("ep_len_mean", "Episode length mean by level", "Level", "Mean episode length", "ep_len_mean.png"),
        ("ep_rew_mean", "Env return mean by level", "Level", "Mean env return", "ep_rew_mean.png"),
    ]

    plot_fn = _bar_plot if args.bar else _line_plot

    for y_col, title, xlabel, ylabel, fname in plots:
        out_path = os.path.join(args.out_dir, fname)
        plot_fn(
            df=df,
            x_col="level",
            y_col=y_col,
            group_col="agent",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            out_path=out_path,
        )

    # Latency plot only if present (typically only llm_policy rows)
    if "mean_step_latency_ms" in df.columns:
        # Some agents may have NaN for latency; keep only non-NaN rows for plotting
        dlat = df.dropna(subset=["mean_step_latency_ms"])
        if not dlat.empty:
            out_path = os.path.join(args.out_dir, "mean_step_latency_ms.png")
            plot_fn(
                df=dlat,
                x_col="level",
                y_col="mean_step_latency_ms",
                group_col="agent",
                title="Mean step latency by level",
                xlabel="Level",
                ylabel="Mean step latency (ms)",
                out_path=out_path,
            )

    _print_summary(df)
    print(f"\nSaved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
