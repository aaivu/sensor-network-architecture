# analysis — Post-Processing Scripts

## Usage

Install dependencies (once):

```bash
pip install -r ../requirements.txt
```

Run against a directory that contains one or more `*results*.csv` files:

```bash
python analysis.py --results-dir <path-to-csv-directory>
```

## What It Produces

| Output | Description |
|--------|-------------|
| `pdr_boxplot.png` | Per-ADR-type PDR box-plots |
| `energy_bar.png` | Mean total energy with 95 % CI |
| `fairness_bar.png` | Jain's fairness index |
| `snr_scatter.png` | SNR vs distance coloured by ADR type |
| `tx_power_boxplot.png` | TX power distributions |
| `summary_stats.csv` | Mann-Whitney U, Cliff's delta, bootstrap CI |

## Metrics Computed

- **Reliability**: rolling PDR, outage count/length, consecutive-loss CDF
- **Energy**: energy per packet, energy per successful packet, projected battery life
- **Airtime**: Time-on-Air (Semtech SX1276 formula), duty-cycle utilisation
- **Interference robustness**: loss-rate by RSSI bin, SNR margin distribution
- **Stability**: SF/TP churn rate, parameter distribution
- **Fairness**: Jain's fairness index across devices
