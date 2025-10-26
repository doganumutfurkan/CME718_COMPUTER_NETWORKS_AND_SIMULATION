# Congestion Avoidance Simulation Toolkit

This module implements a discrete-time simulator that contrasts five TCP congestion control variants—Tahoe, Reno, NewReno, Cubic, and TFRC—across configurable network scenarios. The goal is to generate reproducible throughput, fairness, and congestion window insights without relying on external simulators.

## Project Layout

```
congestion_avoidance_simulation/
├── tcp_compare.py           # Main entry point for running simulations
├── placeholder.txt          # Reserved for additional notes/resources
├── findings/                # Auto-generated outputs from the simulator
│   ├── results.csv          # Tabulated metrics per algorithm and scenario
│   ├── throughput_vs_algorithm.png
│   ├── fairness_vs_scenario.png
│   ├── cwnd_evolution_example_<scenario>.png
│   ├── cwnd_evolution_panels_<scenario>.png
│   └── cwnd_evolution_<algorithm>_<scenario>.png
└── README.md                # Detailed usage and interpretation guide
```

The `findings/` directory is where the script saves all derived artifacts. When you run `tcp_compare.py`, it will create any missing files in that folder and overwrite prior results.

## Getting Started

1. **Install dependencies**

   ```bash
   python -m pip install numpy pandas matplotlib
   ```

2. **Run the simulator**

   From the repository root:

   ```bash
   python congestion_avoidance_simulation/tcp_compare.py
   ```

   The script executes four default scenarios for 5 minutes of simulated time at 100 ms resolution:

   | Scenario | Bandwidth (Mbps) | RTT (ms) | Loss (%) |
   | --- | --- | --- | --- |
   | Broadband Validation | 50 | 40 | 0.1 |
   | High-Delay Queue | 20 | 150 | 0.5 |
   | Wireless-Like Loss | 30 | 80 | 1.2 |
   | Data-Center Stress | 100 | 25 | 0.2 |

   Progress messages summarize throughput, fairness, and congestion window trends for each TCP variant.

## Understanding the Outputs

### `findings/results.csv`

A comma-separated table with one row per algorithm–scenario combination:

- `Scenario`, `Algorithm`: identifiers for slicing results.
- `Avg Throughput (Mbps)`: mean throughput achieved during the simulation window.
- `Fairness`: Jain's fairness calculation across concurrent flows.
- `Avg CWND (packets)`: time-averaged congestion window (or rate proxy).
- `Loss Ratio`: observed packet loss rate.

Load the CSV in pandas or spreadsheet software to perform further analysis or compare against custom parameter sweeps.

### Throughput Bar Chart — `findings/throughput_vs_algorithm.png`

A grouped bar chart comparing the average throughput of each TCP variant across the default scenarios. The colors correspond to the algorithm legend and match every other figure for quick cross-referencing.

### Fairness Comparison — `findings/fairness_vs_scenario.png`

Displays Jain's fairness index per scenario. Higher bars indicate more equitable bandwidth sharing among the five flows. Use this plot to identify conditions where certain algorithms dominate.

### Combined Congestion Window Evolution — `findings/cwnd_evolution_example_<scenario>.png`

High-DPI line charts overlaying the congestion window progression of all algorithms. One file is produced per scenario so you can compare how the dynamics shift under different bandwidth, RTT, and loss conditions. For backward compatibility, the first scenario also emits `cwnd_evolution_example.png` without the slug suffix.

### Per-Algorithm Congestion Window Panels — `findings/cwnd_evolution_panels_<scenario>.png`

Scenario-specific grids of subplots—one per algorithm—provide a focused look at the same time series. These panels are ideal for presentations where you need to comment on the dynamics of individual protocols without clutter from others. The first scenario still produces the legacy filename `cwnd_evolution_panels.png` in addition to the suffixed version.

### Individual Congestion Window Profiles — `findings/cwnd_evolution_<algorithm>_<scenario>.png`

For deep dives into a specific TCP variant, the simulator now exports single-series plots for every algorithm within each scenario. These images isolate one flow per figure, using the same color palette so you can annotate or compare behaviors without manual cropping. Legacy filenames like `cwnd_evolution_tcp_tahoe.png` remain available for users who still reference the first scenario’s historical outputs.

## Customizing Scenarios

Near the bottom of `tcp_compare.py`, the `scenarios` list defines the bandwidth (Mbps), round-trip time (ms), and loss probability (%) for each experiment. Adjust these values or extend the list to explore alternative network conditions. Key parameters:

- `SIM_DURATION`: total simulated time in seconds (default: 300).
- `DT`: simulation time step (default: 0.1 s).
- `MSS_BYTES`: maximum segment size in bytes (default: 1460).

Run the script again after modifying these constants to regenerate outputs with the new settings.

## Tips for High-Quality Figures

- The helper `configure_matplotlib()` enforces high DPI and consistent styling; you can tweak it to match report branding.
- For publication-ready images, consider exporting to PDF by adding `plt.savefig("findings/<name>.pdf")` within the plotting functions.
- If you evaluate additional scenarios, update the README to document new artifacts and interpretations.

## Troubleshooting

- **Missing dependencies**: Ensure `numpy`, `pandas`, and `matplotlib` are installed in the active environment.
- **Permission errors**: Confirm you have write access to `congestion_avoidance_simulation/findings/`.
- **Unexpected plateaus or resets**: The model includes configurable congestion window caps derived from each scenario's bandwidth-delay product; adjust the scaling constants near the top of the script if you need more headroom.

## Next Steps

- Add unit tests for the congestion control behaviors.
- Extend the simulator with additional algorithms (e.g., BBR, Vegas).
- Automate parameter sweeps and export aggregated dashboards for batch comparisons.

Happy experimenting!
