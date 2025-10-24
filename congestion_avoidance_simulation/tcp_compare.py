import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MSS_BYTES = 1460
DT = 0.1  # seconds
SIM_DURATION = 300.0  # seconds
STEPS = int(SIM_DURATION / DT)
RNG_SEED = 1337
DEFAULT_COLORS = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
]

ALGORITHM_COLOR_ORDER = [
    "TCP Tahoe",
    "TCP Reno",
    "TCP NewReno",
    "TCP Cubic",
    "TFRC",
]

ALGORITHM_COLORS: Dict[str, str] = {
    name: DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
    for idx, name in enumerate(ALGORITHM_COLOR_ORDER)
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 190,
            "savefig.dpi": 230,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


@dataclass(frozen=True)
class Scenario:
    name: str
    bandwidth_mbps: float
    rtt_ms: float
    loss_rate: float


class TCPFlow:
    def __init__(self, name: str, rtt_ms: float, mss_bytes: int = MSS_BYTES) -> None:
        self.name = name
        self.rtt_sec = rtt_ms / 1000.0
        self.mss_bytes = mss_bytes
        self.cwnd = 1.0
        self.ssthresh = 64.0

    # --- helper methods -------------------------------------------------
    def estimate_rate_bps(self) -> float:
        cwnd_packets = max(self.cwnd, 1.0)
        return cwnd_packets * self.mss_bytes * 8.0 / max(self.rtt_sec, 1e-6)

    def effective_cwnd(self) -> float:
        return max(self.cwnd, 1.0)

    # --- congestion control hooks --------------------------------------
    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float) -> None:
        raise NotImplementedError

    def on_loss(self, lost_packets: int, attempted_packets: int) -> None:
        raise NotImplementedError


class TCPTahoe(TCPFlow):
    def __init__(self, rtt_ms: float, mss_bytes: int = MSS_BYTES) -> None:
        super().__init__("TCP Tahoe", rtt_ms, mss_bytes)

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float) -> None:
        if acked_packets <= 0:
            return
        if self.cwnd < self.ssthresh:
            self.cwnd += acked_packets
        else:
            self.cwnd += acked_packets / max(self.cwnd, 1.0)

    def on_loss(self, lost_packets: int, attempted_packets: int) -> None:
        if lost_packets <= 0:
            return
        self.ssthresh = max(self.cwnd / 2.0, 1.0)
        self.cwnd = 1.0


class TCPReno(TCPFlow):
    def __init__(self, rtt_ms: float, mss_bytes: int = MSS_BYTES) -> None:
        super().__init__("TCP Reno", rtt_ms, mss_bytes)

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float) -> None:
        if acked_packets <= 0:
            return
        if self.cwnd < self.ssthresh:
            self.cwnd += acked_packets
        else:
            self.cwnd += acked_packets / max(self.cwnd, 1.0)

    def on_loss(self, lost_packets: int, attempted_packets: int) -> None:
        if lost_packets <= 0:
            return
        self.ssthresh = max(self.cwnd / 2.0, 1.0)
        self.cwnd = max(self.ssthresh, 1.0)


class TCPNewReno(TCPReno):
    def __init__(self, rtt_ms: float, mss_bytes: int = MSS_BYTES) -> None:
        super().__init__(rtt_ms, mss_bytes)
        self.name = "TCP NewReno"
        self._in_recovery = False

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float) -> None:
        if acked_packets <= 0:
            return
        if self.cwnd < self.ssthresh:
            self.cwnd += acked_packets
            return
        increase = acked_packets / max(self.cwnd, 1.0)
        if self._in_recovery:
            self.cwnd += 1.5 * increase
            if self.cwnd >= self.ssthresh:
                self._in_recovery = False
        else:
            self.cwnd += increase

    def on_loss(self, lost_packets: int, attempted_packets: int) -> None:
        if lost_packets <= 0:
            return
        self.ssthresh = max(self.cwnd / 2.0, 1.0)
        self.cwnd = max(self.ssthresh, 1.0)
        self._in_recovery = True


class TCPCubic(TCPFlow):
    def __init__(self, rtt_ms: float, mss_bytes: int = MSS_BYTES) -> None:
        super().__init__("TCP Cubic", rtt_ms, mss_bytes)
        self.C = 0.4
        self.beta = 0.3
        self.time_since_loss = 0.0
        self.w_max = self.cwnd

    def cubic_target(self) -> float:
        if self.w_max <= 0.0:
            return max(self.cwnd, 1.0)
        k = (self.w_max * (1.0 - self.beta) / self.C) ** (1.0 / 3.0)
        return self.C * (self.time_since_loss - k) ** 3 + self.w_max

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float) -> None:
        self.time_since_loss += dt
        if acked_packets <= 0:
            return
        additive = acked_packets / max(self.cwnd, 1.0)
        target = self.cubic_target()
        self.cwnd += max(additive, target - self.cwnd)

    def on_loss(self, lost_packets: int, attempted_packets: int) -> None:
        if lost_packets <= 0:
            return
        self.w_max = max(self.cwnd, 1.0)
        self.cwnd *= (1.0 - self.beta)
        self.ssthresh = max(self.cwnd, 1.0)
        self.time_since_loss = 0.0


class TCPFriendlyRateControl(TCPFlow):
    def __init__(self, rtt_ms: float, mss_bytes: int = MSS_BYTES) -> None:
        super().__init__("TFRC", rtt_ms, mss_bytes)
        self.send_rate_bps = self.estimate_rate_bps()

    def effective_cwnd(self) -> float:
        return max(self.send_rate_bps * self.rtt_sec / (self.mss_bytes * 8.0), 1.0)

    def _throughput_equation(self, loss_event_rate: float) -> float:
        if loss_event_rate <= 0.0:
            return self.estimate_rate_bps()
        rtt = max(self.rtt_sec, 1e-3)
        p = min(loss_event_rate, 1.0)
        # Simplified TCP throughput equation
        return self.mss_bytes * 8.0 / (rtt * math.sqrt(p))

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float) -> None:
        delivered = acked_packets / max(attempted_packets, 1)
        loss_event_rate = 1.0 - delivered
        target_rate = self._throughput_equation(loss_event_rate)
        self.send_rate_bps = 0.9 * self.send_rate_bps + 0.1 * target_rate

    def on_loss(self, lost_packets: int, attempted_packets: int) -> None:
        if attempted_packets <= 0:
            return
        loss_fraction = lost_packets / attempted_packets
        target_rate = self._throughput_equation(max(loss_fraction, 1e-3))
        self.send_rate_bps = max(0.5 * self.send_rate_bps, target_rate)


def jain_fairness(values: Sequence[float]) -> float:
    squared_sum = sum(values) ** 2
    count = len(values)
    sum_squares = sum(v * v for v in values)
    if count == 0 or sum_squares == 0.0:
        return 0.0
    return squared_sum / (count * sum_squares)


def allocate_packets(
    flows: Sequence[TCPFlow],
    rng: np.random.Generator,
    scenario: Scenario,
) -> Tuple[List[int], List[int]]:
    capacity_packets = max(
        1,
        int(round(scenario.bandwidth_mbps * 1e6 * DT / (8.0 * MSS_BYTES))),
    )
    weights = np.array([flow.effective_cwnd() for flow in flows], dtype=float)
    total = float(weights.sum())
    if total == 0.0:
        weights[:] = 1.0
        total = float(len(flows))
    weights /= total
    raw_attempts = rng.multinomial(capacity_packets, weights)
    attempts = raw_attempts.astype(int)
    leftover = 0
    for idx, flow in enumerate(flows):
        cap = int(max(flow.effective_cwnd(), 1.0))
        if attempts[idx] > cap:
            leftover += attempts[idx] - cap
            attempts[idx] = cap
    if leftover > 0:
        room = [max(0, int(max(flow.effective_cwnd(), 1.0)) - attempts[i]) for i, flow in enumerate(flows)]
        total_room = sum(room)
        if total_room > 0:
            share = np.array(room, dtype=float) / total_room
            redistrib = rng.multinomial(leftover, share)
            attempts += redistrib
    attempted_packets = attempts.tolist()
    max_losses = attempts
    losses = [int(rng.binomial(max(0, packets), scenario.loss_rate / 100.0)) for packets in max_losses]
    return attempted_packets, losses


def simulate_scenario(
    scenario: Scenario, rng: np.random.Generator
) -> Tuple[List[Dict[str, float]], float, Dict[str, List[float]]]:
    flows: List[TCPFlow] = [
        TCPTahoe(scenario.rtt_ms),
        TCPReno(scenario.rtt_ms),
        TCPNewReno(scenario.rtt_ms),
        TCPCubic(scenario.rtt_ms),
        TCPFriendlyRateControl(scenario.rtt_ms),
    ]

    flow_stats: Dict[str, Dict[str, List[float]]] = {
        flow.name: {"cwnd": [], "throughput": [], "attempted": [], "acked": [], "lost": []}
        for flow in flows
    }

    for step in range(STEPS):
        attempted_packets, losses = allocate_packets(flows, rng, scenario)
        for flow, attempted, lost in zip(flows, attempted_packets, losses):
            acked = max(attempted - lost, 0)
            throughput_mbps = acked * flow.mss_bytes * 8.0 / (DT * 1e6)
            flow.on_ack(acked, attempted, DT)
            if lost > 0:
                flow.on_loss(lost, attempted)

            flow_stats[flow.name]["cwnd"].append(flow.effective_cwnd())
            flow_stats[flow.name]["throughput"].append(throughput_mbps)
            flow_stats[flow.name]["attempted"].append(attempted)
            flow_stats[flow.name]["acked"].append(acked)
            flow_stats[flow.name]["lost"].append(lost)

    summary: List[Dict[str, float]] = []
    throughput_means = []
    for flow in flows:
        stats = flow_stats[flow.name]
        total_acked = sum(stats["acked"])
        total_attempted = sum(stats["attempted"])
        avg_throughput = (total_acked * flow.mss_bytes * 8.0) / (SIM_DURATION * 1e6)
        avg_cwnd = float(np.mean(stats["cwnd"]))
        loss_ratio = (total_attempted - total_acked) / max(total_attempted, 1)
        throughput_means.append(avg_throughput)
        summary.append(
            {
                "Algorithm": flow.name,
                "Avg Throughput (Mbps)": avg_throughput,
                "Avg CWND (packets)": avg_cwnd,
                "Loss Ratio": loss_ratio,
            }
        )

    fairness = jain_fairness(throughput_means)
    for row in summary:
        row["Fairness"] = fairness
    cwnd_series = {flow.name: flow_stats[flow.name]["cwnd"] for flow in flows}
    return summary, fairness, cwnd_series


def build_results_dataframe(
    scenarios: Sequence[Scenario],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, List[float]]]]:
    records: List[Dict[str, float]] = []
    fairness_by_scenario: Dict[str, float] = {}
    cwnd_timeseries: Dict[str, Dict[str, List[float]]] = {}

    for scenario in scenarios:
        scenario_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        summary, fairness, cwnd_series = simulate_scenario(scenario, scenario_rng)
        fairness_by_scenario[scenario.name] = fairness
        cwnd_timeseries[scenario.name] = cwnd_series
        for row in summary:
            records.append({"Scenario": scenario.name, **row})

    df = pd.DataFrame.from_records(records)
    return df, fairness_by_scenario, cwnd_timeseries


def plot_throughput(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    agg = df.groupby(["Scenario", "Algorithm"])["Avg Throughput (Mbps)"].mean()
    scenarios = sorted(df["Scenario"].unique())
    algorithms = ALGORITHM_COLOR_ORDER
    indices = np.arange(len(scenarios))
    width = 0.8 / max(len(algorithms), 1)
    fig, ax = plt.subplots(figsize=(11.5, 5.75), dpi=190)
    bar_containers = []
    for offset, algorithm in enumerate(algorithms):
        values = [agg.get((scenario, algorithm), 0.0) for scenario in scenarios]
        color = ALGORITHM_COLORS.get(algorithm, DEFAULT_COLORS[offset % len(DEFAULT_COLORS)])
        container = ax.bar(
            indices + offset * width,
            values,
            width=width,
            label=algorithm,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        bar_containers.append(container)
    ax.set_xticks(indices + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=0)
    ax.set_ylabel("Average Throughput (Mbps)")
    ax.set_title("Average Throughput by Scenario and Algorithm")
    ax.legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_facecolor("#fdfdfd")
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    for container in bar_containers:
        for bar in container:
            height = bar.get_height()
            if height <= 0:
                continue
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_vs_algorithm.png", bbox_inches="tight")
    plt.close(fig)


def plot_fairness(fairness: Dict[str, float], output_dir: pathlib.Path) -> None:
    scenarios = list(fairness.keys())
    values = [fairness[name] for name in scenarios]
    colors = [DEFAULT_COLORS[idx % len(DEFAULT_COLORS)] for idx in range(len(scenarios))]
    fig, ax = plt.subplots(figsize=(9, 4.75), dpi=190)
    bars = ax.bar(scenarios, values, color=colors, edgecolor="black", linewidth=0.7)
    upper = max(1.0, max(values) * 1.05)
    ax.set_ylim(0, min(1.05, upper))
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Fairness Across Scenarios")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_facecolor("#fdfdfd")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(output_dir / "fairness_vs_scenario.png", bbox_inches="tight")
    plt.close(fig)


def plot_cwnd_example(
    cwnd_timeseries: Dict[str, Dict[str, List[float]]],
    output_dir: pathlib.Path,
    scenario_name: str,
) -> None:
    series = cwnd_timeseries[scenario_name]
    time_axis = np.linspace(0.0, SIM_DURATION, num=STEPS)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    for idx, (algo, values) in enumerate(series.items()):
        color = ALGORITHM_COLORS.get(algo, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
        ax.plot(
            time_axis,
            values,
            label=algo,
            linewidth=2.2,
            color=color,
            alpha=0.95,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CWND (packets)")
    ax.set_title(f"Congestion Window Evolution — {scenario_name}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.set_facecolor("#fdfdfd")
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    fig.tight_layout()
    fig.savefig(output_dir / "cwnd_evolution_example.png", bbox_inches="tight")
    plt.close(fig)


def plot_cwnd_per_algorithm(
    cwnd_timeseries: Dict[str, Dict[str, List[float]]],
    output_dir: pathlib.Path,
    scenario_name: str,
) -> None:
    series = cwnd_timeseries[scenario_name]
    time_axis = np.linspace(0.0, SIM_DURATION, num=STEPS)
    algorithms = [algo for algo in ALGORITHM_COLOR_ORDER if algo in series]
    n_algos = len(algorithms)
    cols = 2
    rows = math.ceil(n_algos / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 3.8 * rows), dpi=180, sharex=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx, algo in enumerate(algorithms):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        values = series[algo]
        color = ALGORITHM_COLORS.get(algo, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
        ax.plot(time_axis, values, color=color, linewidth=2.2)
        ax.set_title(algo)
        ax.set_ylabel("CWND")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_facecolor("#fdfdfd")
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")
    for idx in range(n_algos, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis("off")
    for col in range(cols):
        ax = axes[rows - 1][col]
        if ax.has_data():
            ax.set_xlabel("Time (s)")
    fig.suptitle(f"Per-Algorithm CWND Evolution — {scenario_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "cwnd_evolution_panels.png", bbox_inches="tight")
    plt.close(fig)


SAMPLE_PARAMETER_SETS = [
    {
        "Label": "Broadband validation",
        "Bandwidth (Mbps)": 50,
        "RTT (ms)": 40,
        "Loss (%)": 0.1,
    },
    {
        "Label": "High-delay queue",
        "Bandwidth (Mbps)": 20,
        "RTT (ms)": 150,
        "Loss (%)": 0.5,
    },
    {
        "Label": "Wireless-like loss",
        "Bandwidth (Mbps)": 30,
        "RTT (ms)": 80,
        "Loss (%)": 1.2,
    },
    {
        "Label": "Data-center stress",
        "Bandwidth (Mbps)": 100,
        "RTT (ms)": 25,
        "Loss (%)": 0.2,
    },
]


def main() -> None:
    output_dir = pathlib.Path(__file__).resolve().parent
    configure_matplotlib()
    rng = np.random.default_rng(RNG_SEED)

    scenarios = [
        Scenario("Scenario 1", 50, 40, 0.1),
        Scenario("Scenario 2", 20, 150, 0.5),
        Scenario("Scenario 3", 100, 50, 1.0),
    ]

    df, fairness, cwnd_timeseries = build_results_dataframe(scenarios, rng)
    results_path = output_dir / "results.csv"
    df.to_csv(results_path, index=False)

    print("Simulation complete. Results written to:", results_path)
    print()
    print(df.to_string(index=False, formatters={
        "Avg Throughput (Mbps)": "{:.2f}".format,
        "Fairness": "{:.3f}".format,
        "Avg CWND (packets)": "{:.1f}".format,
        "Loss Ratio": "{:.3f}".format,
    }))
    print()
    print("Fairness per scenario:")
    for name, value in fairness.items():
        print(f"  {name}: {value:.3f}")
    print()

    print("Sample parameter sets for further experimentation:")
    sample_df = pd.DataFrame(SAMPLE_PARAMETER_SETS)
    print(sample_df.to_string(index=False))
    print()

    throughput_order = df.groupby("Algorithm")["Avg Throughput (Mbps)"].mean().sort_values(ascending=False)
    leading = throughput_order.index[0]
    trailing = throughput_order.index[-1]
    print(
        f"Summary: {leading} achieved the highest mean throughput across scenarios, while {trailing} trailed. "
        "Use the parameter catalog above to explore alternative network regimes."
    )

    plot_throughput(df, output_dir)
    plot_fairness(fairness, output_dir)
    plot_cwnd_example(cwnd_timeseries, output_dir, "Scenario 1")
    plot_cwnd_per_algorithm(cwnd_timeseries, output_dir, "Scenario 1")
    plt.show()


if __name__ == "__main__":
    main()
