
import math
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TCPFlow:
    def __init__(self, name: str, rtt_ms: float, mss_bytes: int = 1460):
        self.name = name
        self.rtt_sec = rtt_ms / 1000.0
        self.mss_bytes = mss_bytes
        self.cwnd = 1.0
        self.ssthresh = 64.0

    def estimate_rate_bps(self) -> float:
        """Estimate sending rate in bits per second based on cwnd."""
        cwnd_packets = max(self.cwnd, 1.0)
        return cwnd_packets * self.mss_bytes * 8.0 / max(self.rtt_sec, 1e-6)

    def effective_cwnd_packets(self) -> float:
        return max(self.cwnd, 1.0)

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float):
        raise NotImplementedError

    def on_loss(self, lost_packets: int, attempted_packets: int):
        raise NotImplementedError


class TCPTahoe(TCPFlow):
    def __init__(self, rtt_ms: float, mss_bytes: int = 1460):
        super().__init__("TCP Tahoe", rtt_ms, mss_bytes)

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float):
        if acked_packets <= 0:
            return
        if self.cwnd < self.ssthresh:
            self.cwnd += acked_packets
        else:
            self.cwnd += acked_packets / max(self.cwnd, 1.0)

    def on_loss(self, lost_packets: int, attempted_packets: int):
        if lost_packets <= 0:
            return
        self.ssthresh = max(self.cwnd / 2.0, 1.0)
        self.cwnd = 1.0


class TCPReno(TCPFlow):
    def __init__(self, rtt_ms: float, mss_bytes: int = 1460):
        super().__init__("TCP Reno", rtt_ms, mss_bytes)

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float):
        if acked_packets <= 0:
            return
        if self.cwnd < self.ssthresh:
            self.cwnd += acked_packets
        else:
            self.cwnd += acked_packets / max(self.cwnd, 1.0)

    def on_loss(self, lost_packets: int, attempted_packets: int):
        if lost_packets <= 0:
            return
        self.ssthresh = max(self.cwnd / 2.0, 1.0)
        self.cwnd = max(self.ssthresh, 1.0)


class TCPNewReno(TCPReno):
    def __init__(self, rtt_ms: float, mss_bytes: int = 1460):
        super().__init__(rtt_ms, mss_bytes)
        self.name = "TCP NewReno"
        self.in_recovery = False

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float):
        if acked_packets <= 0:
            return
        if self.cwnd < self.ssthresh:
            self.cwnd += acked_packets
            return
        increase = acked_packets / max(self.cwnd, 1.0)
        if self.in_recovery:
            self.cwnd += 1.5 * increase
            if self.cwnd >= self.ssthresh:
                self.in_recovery = False
        else:
            self.cwnd += increase

    def on_loss(self, lost_packets: int, attempted_packets: int):
        if lost_packets <= 0:
            return
        self.ssthresh = max(self.cwnd / 2.0, 1.0)
        self.cwnd = max(self.ssthresh, 1.0)
        self.in_recovery = True


class TCPCubic(TCPFlow):
    def __init__(self, rtt_ms: float, mss_bytes: int = 1460):
        super().__init__("TCP Cubic", rtt_ms, mss_bytes)
        self.C = 0.4
        self.beta = 0.3
        self.time_since_loss = 0.0
        self.W_max = self.cwnd

    def cubic_target(self) -> float:
        if self.W_max <= 0:
            return max(self.cwnd, 1.0)
        k = (self.W_max * (1 - self.beta) / self.C) ** (1.0 / 3.0)
        return self.C * (self.time_since_loss - k) ** 3 + self.W_max

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float):
        if acked_packets <= 0:
            self.time_since_loss += dt
            return
        self.time_since_loss += dt
        additive = acked_packets / max(self.cwnd, 1.0)
        target = self.cubic_target()
        self.cwnd += additive
        if target > self.cwnd:
            self.cwnd = 0.7 * self.cwnd + 0.3 * max(target, 1.0)
        self.cwnd = max(self.cwnd, 1.0)
        self.W_max = max(self.W_max, self.cwnd)

    def on_loss(self, lost_packets: int, attempted_packets: int):
        if lost_packets <= 0:
            return
        self.W_max = max(self.cwnd, 1.0)
        self.cwnd = max(self.cwnd * (1 - self.beta), 1.0)
        self.ssthresh = self.cwnd
        self.time_since_loss = 0.0


class TFRCRateFlow(TCPFlow):
    def __init__(self, rtt_ms: float, bandwidth_mbps: float, mss_bytes: int = 1460):
        super().__init__("TFRC", rtt_ms, mss_bytes)
        initial_rate = self.mss_bytes * 8.0 / max(self.rtt_sec, 1e-6)
        self.rate_bps = initial_rate
        self.bandwidth_bps = bandwidth_mbps * 1e6
        self.p_est = 0.0001
        self.gain = 0.2
        self.rto = 4.0 * self.rtt_sec

    def estimate_rate_bps(self) -> float:
        return min(self.rate_bps, self.bandwidth_bps)

    def effective_cwnd_packets(self) -> float:
        return max(self.rate_bps * self.rtt_sec / (self.mss_bytes * 8.0), 1.0)

    def on_ack(self, acked_packets: int, attempted_packets: int, dt: float):
        if attempted_packets > 0:
            loss_ratio = max(attempted_packets - acked_packets, 0) / attempted_packets
            self.p_est = 0.9 * self.p_est + 0.1 * loss_ratio
        if dt > 0:
            sample_rate = acked_packets * self.mss_bytes * 8.0 / dt
            self.rate_bps = 0.8 * self.rate_bps + 0.2 * sample_rate
        self.rate_bps = min(self.rate_bps, self.bandwidth_bps)

    def on_loss(self, lost_packets: int, attempted_packets: int):
        if attempted_packets <= 0:
            return
        p = max(self.p_est, 1e-6)
        s = self.mss_bytes * 8.0
        rtt = max(self.rtt_sec, 1e-6)
        b = 1.0
        t_rto = max(self.rto, rtt)
        sqrt_term = math.sqrt(2.0 * b * p / 3.0)
        denom = rtt * (sqrt_term + (t_rto * (3.0 * math.sqrt(3.0 * b * p / 8.0) * p * (1.0 + 32.0 * p * p))))
        if denom <= 0:
            return
        target_rate = s / denom
        self.rate_bps = (1 - self.gain) * self.rate_bps + self.gain * target_rate
        self.rate_bps = max(min(self.rate_bps, self.bandwidth_bps), s / rtt)


SCENARIOS = [
    {
        "id": 1,
        "bandwidth_mbps": 50,
        "rtt_ms": 40,
        "loss_percent": 0.1,
        "description": "Typical broadband (low delay)",
    },
    {
        "id": 2,
        "bandwidth_mbps": 20,
        "rtt_ms": 150,
        "loss_percent": 0.5,
        "description": "Long delay, moderate loss",
    },
    {
        "id": 3,
        "bandwidth_mbps": 100,
        "rtt_ms": 50,
        "loss_percent": 1.0,
        "description": "High-speed, lossy link",
    },
]

DT = 0.1  # 100 ms resolution
SIM_DURATION = 120.0  # seconds for richer time series detail
STEPS = int(SIM_DURATION / DT)
MSS_BYTES = 1460

ALGORITHM_COLORS = {
    "TCP Tahoe": "#1f77b4",
    "TCP Reno": "#ff7f0e",
    "TCP NewReno": "#2ca02c",
    "TCP Cubic": "#d62728",
    "TFRC": "#9467bd",
}

def jains_fairness(values):
    values = np.array(values, dtype=float)
    if np.all(values == 0):
        return 0.0
    numerator = values.sum() ** 2
    denominator = len(values) * np.sum(values ** 2)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def instantiate_flows(scenario):
    rtt_ms = scenario["rtt_ms"]
    bw = scenario["bandwidth_mbps"]
    return {
        "TCP Tahoe": TCPTahoe(rtt_ms, MSS_BYTES),
        "TCP Reno": TCPReno(rtt_ms, MSS_BYTES),
        "TCP NewReno": TCPNewReno(rtt_ms, MSS_BYTES),
        "TCP Cubic": TCPCubic(rtt_ms, MSS_BYTES),
        "TFRC": TFRCRateFlow(rtt_ms, bw, MSS_BYTES),
    }


def run_scenario(scenario, rng):
    flows = instantiate_flows(scenario)
    capacity_bps = scenario["bandwidth_mbps"] * 1e6
    loss_prob = scenario["loss_percent"] / 100.0

    throughput_records = {name: [] for name in flows}
    cwnd_records = {name: [] for name in flows}
    loss_records = {name: {"attempted": 0, "lost": 0} for name in flows}

    time_series = np.arange(STEPS) * DT

    for _ in range(STEPS):
        estimated_rates = {name: flow.estimate_rate_bps() for name, flow in flows.items()}
        total_demand = sum(estimated_rates.values())
        capacity_factor = 1.0
        if total_demand > 0:
            capacity_factor = min(1.0, capacity_bps / total_demand)

        for name, flow in flows.items():
            effective_success = (1.0 - loss_prob) * capacity_factor
            effective_success = max(min(effective_success, 1.0), 0.0)

            packets_attempted = max(
                1,
                int(
                    round(
                        flow.effective_cwnd_packets()
                        * DT
                        / max(flow.rtt_sec, 1e-6)
                    )
                ),
            )

            acked_packets = rng.binomial(packets_attempted, effective_success)
            lost_packets = packets_attempted - acked_packets

            flow.on_ack(acked_packets, packets_attempted, DT)
            if lost_packets > 0:
                flow.on_loss(lost_packets, packets_attempted)

            throughput_mbps = (
                acked_packets * MSS_BYTES * 8.0 / DT / 1e6
            )
            throughput_records[name].append(throughput_mbps)
            cwnd_records[name].append(flow.effective_cwnd_packets())
            loss_records[name]["attempted"] += packets_attempted
            loss_records[name]["lost"] += lost_packets

    avg_throughput = {
        name: float(np.mean(values)) for name, values in throughput_records.items()
    }
    avg_cwnd = {
        name: float(np.mean(values)) for name, values in cwnd_records.items()
    }
    loss_ratio = {
        name: (
            loss_records[name]["lost"] / loss_records[name]["attempted"]
            if loss_records[name]["attempted"] > 0
            else 0.0
        )
        for name in flows
    }

    fairness = jains_fairness(list(avg_throughput.values()))

    scenario_data = []
    for name in flows:
        scenario_data.append(
            {
                "Scenario": scenario["id"],
                "Algorithm": name,
                "Avg Throughput (Mbps)": avg_throughput[name],
                "Fairness": fairness,
                "Avg CWND (packets)": avg_cwnd[name],
                "Packet Loss Ratio": loss_ratio[name],
                "Description": scenario["description"],
            }
        )

    return scenario_data, time_series, cwnd_records


def main():
    rng = np.random.default_rng(42)
    all_results = []
    scenario_cwnd_series = {}
    scenario_time = None

    for scenario in SCENARIOS:
        data, time_points, cwnd_records = run_scenario(scenario, rng)
        all_results.extend(data)
        if scenario["id"] == 1:
            scenario_cwnd_series = {name: cwnd_records[name] for name in cwnd_records}
            scenario_time = time_points

    df = pd.DataFrame(all_results)
    output_dir = pathlib.Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "results.csv", index=False)

    print("Simulation Summary Table:\n")
    print(
        df[
            [
                "Scenario",
                "Algorithm",
                "Avg Throughput (Mbps)",
                "Fairness",
                "Avg CWND (packets)",
                "Packet Loss Ratio",
            ]
        ].to_string(index=False, formatters={
            "Avg Throughput (Mbps)": "{:.2f}".format,
            "Fairness": "{:.2f}".format,
            "Avg CWND (packets)": "{:.2f}".format,
            "Packet Loss Ratio": "{:.3f}".format,
        })
    )

    throughput_by_algorithm = (
        df.groupby("Algorithm")["Avg Throughput (Mbps)"].mean().sort_values()
    )
    plt.figure(figsize=(9, 5.5))
    colors = [ALGORITHM_COLORS.get(name, "#1f77b4") for name in throughput_by_algorithm.index]
    bars = throughput_by_algorithm.plot(kind="bar", color=colors, edgecolor="black")
    plt.ylabel("Average Throughput (Mbps)")
    plt.title("Average Throughput by Algorithm (across scenarios)")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_vs_algorithm.png", dpi=200)

    fairness_by_scenario = df.groupby("Scenario")["Fairness"].mean()
    plt.figure(figsize=(9, 5))
    fairness_colors = ["#8dd3c7", "#80b1d3", "#bebada"]
    bars = fairness_by_scenario.plot(kind="bar", color=fairness_colors[: len(fairness_by_scenario)], edgecolor="black")
    plt.ylabel("Jain's Fairness Index")
    plt.title("Fairness Across Scenarios")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "fairness_vs_scenario.png", dpi=200)

    if scenario_cwnd_series and scenario_time is not None:
        plt.figure(figsize=(11, 5.5))
        for name, series in scenario_cwnd_series.items():
            color = ALGORITHM_COLORS.get(name)
            plot_kwargs = {"label": name, "linewidth": 1.6}
            if color is not None:
                plot_kwargs["color"] = color
            plt.plot(scenario_time, series, **plot_kwargs)
        plt.xlabel("Time (s)")
        plt.ylabel("Congestion Window (packets)")
        plt.title("Congestion Window Evolution (Scenario 1)")
        plt.legend()
        plt.grid(linestyle="--", linewidth=0.6, alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "cwnd_evolution_example.png", dpi=220)

        plt.figure(figsize=(11, 5.5))
        for name, series in scenario_cwnd_series.items():
            series_array = np.asarray(series, dtype=float)
            max_value = float(series_array.max()) if series_array.size else 0.0
            if max_value > 0:
                normalized_series = series_array / max_value
            else:
                normalized_series = series_array
            color = ALGORITHM_COLORS.get(name)
            plot_kwargs = {"label": name, "linewidth": 1.6}
            if color is not None:
                plot_kwargs["color"] = color
            plt.plot(scenario_time, normalized_series, **plot_kwargs)
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Congestion Window")
        plt.title("Normalized Congestion Window Evolution (Scenario 1)")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(linestyle="--", linewidth=0.6, alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "cwnd_evolution_normalized.png", dpi=220)

    print("\nKey Observations:")
    df_sorted = df.sort_values(["Scenario", "Avg Throughput (Mbps)"], ascending=[True, False])
    for scenario_id, group in df_sorted.groupby("Scenario"):
        best = group.iloc[0]
        print(
            f"Scenario {scenario_id}: {best['Algorithm']} achieved the highest throughput at "
            f"{best['Avg Throughput (Mbps)']:.2f} Mbps with fairness {best['Fairness']:.2f}."
        )

    overall_best = df.groupby("Algorithm")["Avg Throughput (Mbps)"].mean().idxmax()
    most_fair = df.groupby("Algorithm")["Fairness"].mean().idxmax()
    print(
        f"\nOverall, {overall_best} delivered the highest average throughput, while "
        f"{most_fair} maintained the best fairness across varying network conditions."
    )

    plt.show()


if __name__ == "__main__":
    main()
