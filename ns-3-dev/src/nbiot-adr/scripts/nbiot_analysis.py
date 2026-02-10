#!/usr/bin/env python3
"""
NB-IoT RL Link Adaptation Analysis Script
Mirrors the LoRaWAN analysis.py but adapted for NB-IoT metrics.

Generates comprehensive analysis including:
- PDR comparison across link adaptation methods
- MCS distribution analysis
- Repetition usage patterns
- RACH success rate analysis
- Energy consumption comparison
- SINR vs MCS scatter
- Per-UE performance breakdown
- Coverage enhancement level analysis
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Plot style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Method colors and labels
METHOD_COLORS = {
    'no_la': '#2196F3',
    'default_la': '#4CAF50',
    'ddqn_la': '#F44336',
    'ppo_la': '#FF9800',
    'marl_la': '#9C27B0',
}

METHOD_LABELS = {
    'no_la': 'No LA',
    'default_la': 'Default (3GPP)',
    'ddqn_la': 'DDQN-PER',
    'ppo_la': 'PPO',
    'marl_la': 'MARL',
}


def load_datasets(data_dir):
    """Load all CSV datasets from the nbiot_datasets directory."""
    datasets = {}
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

    # Filter out environment files
    csv_files = [f for f in csv_files if '_environment' not in f]

    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)
        for method_key in METHOD_LABELS.keys():
            if filename.startswith(method_key):
                try:
                    df = pd.read_csv(filepath)
                    if len(df) > 0:
                        datasets[method_key] = df
                        print(f"  Loaded {method_key}: {len(df)} records from {filename}")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
                break

    return datasets


def calculate_pdr(df):
    """Calculate overall PDR from a dataset."""
    if len(df) == 0:
        return 0.0
    total = len(df)
    success = df['data_success'].sum() if 'data_success' in df.columns else 0
    return 100.0 * success / total


def calculate_per_ue_pdr(df):
    """Calculate PDR per UE."""
    if 'ue_id' not in df.columns:
        return {}
    result = {}
    for ue_id in df['ue_id'].unique():
        ue_df = df[df['ue_id'] == ue_id]
        pdr = 100.0 * ue_df['data_success'].sum() / len(ue_df) if len(ue_df) > 0 else 0.0
        result[ue_id] = pdr
    return result


def calculate_rach_success_rate(df):
    """Calculate RACH success rate."""
    if 'rach_success' not in df.columns:
        return 0.0
    return 100.0 * df['rach_success'].sum() / len(df) if len(df) > 0 else 0.0


def plot_pdr_comparison(datasets, output_dir):
    """Bar chart comparing PDR across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    pdrs = []
    colors = []

    for key in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la']:
        if key in datasets:
            methods.append(METHOD_LABELS[key])
            pdrs.append(calculate_pdr(datasets[key]))
            colors.append(METHOD_COLORS[key])

    bars = ax.bar(methods, pdrs, color=colors, edgecolor='black', linewidth=0.5, width=0.6)

    for bar, pdr in zip(bars, pdrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{pdr:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Packet Delivery Rate (%)')
    ax.set_title('NB-IoT PDR Comparison Across Link Adaptation Methods')
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% target')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'pdr_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'pdr_comparison.eps'), format='eps')
    plt.close()
    print("  ✅ PDR comparison chart saved")


def plot_mcs_distribution(datasets, output_dir):
    """Stacked bar chart showing MCS distribution per method."""
    fig, ax = plt.subplots(figsize=(12, 6))

    mcs_range = range(0, 11)  # MCS 0-10
    x = np.arange(len([k for k in METHOD_LABELS if k in datasets]))
    width = 0.6

    method_keys = [k for k in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la'] if k in datasets]

    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, 11))

    bottoms = np.zeros(len(method_keys))
    for mcs in mcs_range:
        fractions = []
        for key in method_keys:
            df = datasets[key]
            if 'mcs' in df.columns:
                frac = 100.0 * (df['mcs'] == mcs).sum() / len(df)
            else:
                frac = 0.0
            fractions.append(frac)

        ax.bar(x, fractions, width, bottom=bottoms, color=cmap[mcs],
               label=f'MCS {mcs}', edgecolor='white', linewidth=0.3)
        bottoms += np.array(fractions)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[k] for k in method_keys])
    ax.set_ylabel('Percentage (%)')
    ax.set_title('MCS Distribution by Link Adaptation Method')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mcs_distribution.png'))
    plt.savefig(os.path.join(output_dir, 'mcs_distribution.eps'), format='eps')
    plt.close()
    print("  ✅ MCS distribution chart saved")


def plot_repetition_usage(datasets, output_dir):
    """Box plot showing repetition usage distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data_list = []
    labels = []
    colors_list = []

    for key in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la']:
        if key in datasets and 'repetitions' in datasets[key].columns:
            data_list.append(datasets[key]['repetitions'].values)
            labels.append(METHOD_LABELS[key])
            colors_list.append(METHOD_COLORS[key])

    bp = ax.boxplot(data_list, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Number of Repetitions')
    ax.set_title('NB-IoT Repetition Usage Distribution')
    ax.set_yscale('log', base=2)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'repetition_usage.png'))
    plt.savefig(os.path.join(output_dir, 'repetition_usage.eps'), format='eps')
    plt.close()
    print("  ✅ Repetition usage chart saved")


def plot_energy_comparison(datasets, output_dir):
    """Bar chart comparing average energy consumption per packet."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    energies = []
    colors = []

    for key in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la']:
        if key in datasets and 'energy_consumed_mj' in datasets[key].columns:
            methods.append(METHOD_LABELS[key])
            energies.append(datasets[key]['energy_consumed_mj'].mean())
            colors.append(METHOD_COLORS[key])

    bars = ax.bar(methods, energies, color=colors, edgecolor='black', linewidth=0.5, width=0.6)

    for bar, e in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{e:.1f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Average Energy per Packet (mJ)')
    ax.set_title('NB-IoT Energy Consumption Comparison')
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'energy_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'energy_comparison.eps'), format='eps')
    plt.close()
    print("  ✅ Energy comparison chart saved")


def plot_rach_analysis(datasets, output_dir):
    """RACH success rate and average attempts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    methods = []
    rach_rates = []
    avg_attempts = []
    colors = []

    for key in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la']:
        if key in datasets:
            df = datasets[key]
            methods.append(METHOD_LABELS[key])
            colors.append(METHOD_COLORS[key])

            if 'rach_success' in df.columns:
                rach_rates.append(calculate_rach_success_rate(df))
            else:
                rach_rates.append(0)

            if 'rach_attempts' in df.columns:
                avg_attempts.append(df['rach_attempts'].mean())
            else:
                avg_attempts.append(0)

    # RACH success rate
    bars1 = ax1.bar(methods, rach_rates, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    for bar, rate in zip(bars1, rach_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('RACH Success Rate (%)')
    ax1.set_title('NPRACH Success Rate')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)

    # Average RACH attempts
    bars2 = ax2.bar(methods, avg_attempts, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    for bar, att in zip(bars2, avg_attempts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{att:.2f}', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Average RACH Attempts')
    ax2.set_title('Average NPRACH Attempts per Packet')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rach_analysis.png'))
    plt.savefig(os.path.join(output_dir, 'rach_analysis.eps'), format='eps')
    plt.close()
    print("  ✅ RACH analysis chart saved")


def plot_sinr_vs_mcs(datasets, output_dir):
    """Scatter plot: SINR vs MCS colored by success/failure."""
    if not datasets:
        return

    key = list(datasets.keys())[-1]  # Use RL method if available
    for k in ['ddqn_la', 'ppo_la', 'marl_la', 'default_la']:
        if k in datasets:
            key = k
            break

    df = datasets[key]
    if 'sinr_db' not in df.columns or 'mcs' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    success = df[df['data_success'] == 1] if 'data_success' in df.columns else df
    failure = df[df['data_success'] == 0] if 'data_success' in df.columns else pd.DataFrame()

    ax.scatter(success['sinr_db'], success['mcs'], alpha=0.3, c='green',
               s=15, label=f'Success ({len(success)})')
    if len(failure) > 0:
        ax.scatter(failure['sinr_db'], failure['mcs'], alpha=0.3, c='red',
                   s=15, label=f'Failure ({len(failure)})', marker='x')

    ax.set_xlabel('SINR (dB)')
    ax.set_ylabel('MCS Index')
    ax.set_title(f'SINR vs MCS Index ({METHOD_LABELS[key]})')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yticks(range(0, 11))

    plt.savefig(os.path.join(output_dir, 'sinr_vs_mcs.png'))
    plt.savefig(os.path.join(output_dir, 'sinr_vs_mcs.eps'), format='eps')
    plt.close()
    print("  ✅ SINR vs MCS scatter saved")


def plot_per_ue_pdr(datasets, output_dir):
    """Per-UE PDR comparison across methods."""
    fig, ax = plt.subplots(figsize=(14, 7))

    method_keys = [k for k in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la'] if k in datasets]

    # Get all UE IDs
    all_ue_ids = set()
    for key in method_keys:
        if 'ue_id' in datasets[key].columns:
            all_ue_ids.update(datasets[key]['ue_id'].unique())
    all_ue_ids = sorted(all_ue_ids)

    x = np.arange(len(all_ue_ids))
    width = 0.15
    offset = 0

    for key in method_keys:
        per_ue = calculate_per_ue_pdr(datasets[key])
        pdrs = [per_ue.get(ue_id, 0) for ue_id in all_ue_ids]
        ax.bar(x + offset, pdrs, width, color=METHOD_COLORS[key],
               label=METHOD_LABELS[key], edgecolor='white', linewidth=0.3)
        offset += width

    ax.set_xticks(x + width * (len(method_keys) - 1) / 2)
    ax.set_xticklabels([f'UE {uid}' for uid in all_ue_ids], rotation=45, ha='right')
    ax.set_ylabel('PDR (%)')
    ax.set_title('Per-UE Packet Delivery Rate Comparison')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.4, label='90% target')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_ue_pdr.png'))
    plt.savefig(os.path.join(output_dir, 'per_ue_pdr.eps'), format='eps')
    plt.close()
    print("  ✅ Per-UE PDR chart saved")


def plot_pdr_over_time(datasets, output_dir):
    """PDR evolution over time using rolling window."""
    fig, ax = plt.subplots(figsize=(12, 6))

    window = 50  # Rolling window size

    for key in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la']:
        if key in datasets:
            df = datasets[key].sort_values('timestamp')
            if 'data_success' in df.columns and len(df) > window:
                rolling_pdr = df['data_success'].rolling(window=window).mean() * 100
                ax.plot(df['timestamp'], rolling_pdr,
                        color=METHOD_COLORS[key], label=METHOD_LABELS[key],
                        linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PDR (%) - Rolling Window')
    ax.set_title(f'NB-IoT PDR Over Time (window={window} packets)')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.4)
    ax.grid(alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'pdr_over_time.png'))
    plt.savefig(os.path.join(output_dir, 'pdr_over_time.eps'), format='eps')
    plt.close()
    print("  ✅ PDR over time chart saved")


def plot_ce_level_distribution(datasets, output_dir):
    """Coverage Enhancement level distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    method_keys = [k for k in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la'] if k in datasets]

    x = np.arange(len(method_keys))
    width = 0.25

    ce_colors = ['#4CAF50', '#FF9800', '#F44336']
    ce_labels = ['CE Level 0', 'CE Level 1', 'CE Level 2']

    for ce_level in range(3):
        fractions = []
        for key in method_keys:
            df = datasets[key]
            if 'ce_level' in df.columns:
                frac = 100.0 * (df['ce_level'] == ce_level).sum() / len(df)
            else:
                frac = 0
            fractions.append(frac)

        ax.bar(x + ce_level * width, fractions, width,
               color=ce_colors[ce_level], label=ce_labels[ce_level],
               edgecolor='white', linewidth=0.3)

    ax.set_xticks(x + width)
    ax.set_xticklabels([METHOD_LABELS[k] for k in method_keys])
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Coverage Enhancement Level Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'ce_level_distribution.png'))
    plt.savefig(os.path.join(output_dir, 'ce_level_distribution.eps'), format='eps')
    plt.close()
    print("  ✅ CE level distribution chart saved")


def plot_comprehensive_dashboard(datasets, output_dir):
    """6-panel dashboard with key metrics."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.25)

    method_keys = [k for k in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la'] if k in datasets]

    # Panel 1: PDR
    ax1 = fig.add_subplot(gs[0, 0])
    pdrs = [calculate_pdr(datasets[k]) for k in method_keys]
    bars = ax1.bar([METHOD_LABELS[k] for k in method_keys], pdrs,
                   color=[METHOD_COLORS[k] for k in method_keys],
                   edgecolor='black', linewidth=0.5)
    for bar, p in zip(bars, pdrs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{p:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_ylabel('PDR (%)')
    ax1.set_title('Packet Delivery Rate')
    ax1.set_ylim(0, 105)
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.4)
    ax1.tick_params(axis='x', rotation=15)

    # Panel 2: Energy
    ax2 = fig.add_subplot(gs[0, 1])
    energies = []
    for k in method_keys:
        if 'energy_consumed_mj' in datasets[k].columns:
            energies.append(datasets[k]['energy_consumed_mj'].mean())
        else:
            energies.append(0)
    bars = ax2.bar([METHOD_LABELS[k] for k in method_keys], energies,
                   color=[METHOD_COLORS[k] for k in method_keys],
                   edgecolor='black', linewidth=0.5)
    for bar, e in zip(bars, energies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{e:.1f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Avg Energy (mJ)')
    ax2.set_title('Energy per Packet')
    ax2.tick_params(axis='x', rotation=15)

    # Panel 3: RACH success
    ax3 = fig.add_subplot(gs[1, 0])
    rach_rates = [calculate_rach_success_rate(datasets[k]) for k in method_keys]
    bars = ax3.bar([METHOD_LABELS[k] for k in method_keys], rach_rates,
                   color=[METHOD_COLORS[k] for k in method_keys],
                   edgecolor='black', linewidth=0.5)
    for bar, r in zip(bars, rach_rates):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{r:.1f}%', ha='center', va='bottom', fontsize=9)
    ax3.set_ylabel('RACH Success (%)')
    ax3.set_title('NPRACH Success Rate')
    ax3.set_ylim(0, 105)
    ax3.tick_params(axis='x', rotation=15)

    # Panel 4: MCS distribution violin
    ax4 = fig.add_subplot(gs[1, 1])
    mcs_data = []
    mcs_labels = []
    for k in method_keys:
        if 'mcs' in datasets[k].columns:
            mcs_data.append(datasets[k]['mcs'].values)
            mcs_labels.append(METHOD_LABELS[k])
    if mcs_data:
        parts = ax4.violinplot(mcs_data, showmeans=True, showmedians=True)
        ax4.set_xticks(range(1, len(mcs_labels) + 1))
        ax4.set_xticklabels(mcs_labels, rotation=15)
    ax4.set_ylabel('MCS Index')
    ax4.set_title('MCS Usage Distribution')
    ax4.set_yticks(range(0, 11))

    # Panel 5: PDR over time
    ax5 = fig.add_subplot(gs[2, 0])
    window = 30
    for key in method_keys:
        df = datasets[key].sort_values('timestamp')
        if 'data_success' in df.columns and len(df) > window:
            rolling = df['data_success'].rolling(window=window).mean() * 100
            ax5.plot(df['timestamp'], rolling, color=METHOD_COLORS[key],
                     label=METHOD_LABELS[key], linewidth=1.2, alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('PDR (%)')
    ax5.set_title('PDR Over Time')
    ax5.legend(fontsize=8)
    ax5.set_ylim(0, 105)

    # Panel 6: TX Power distribution
    ax6 = fig.add_subplot(gs[2, 1])
    tp_data = []
    tp_labels = []
    for k in method_keys:
        if 'tx_power_dbm' in datasets[k].columns:
            tp_data.append(datasets[k]['tx_power_dbm'].values)
            tp_labels.append(METHOD_LABELS[k])
    if tp_data:
        bp = ax6.boxplot(tp_data, labels=tp_labels, patch_artist=True, showmeans=True)
        for patch, key in zip(bp['boxes'], method_keys):
            patch.set_facecolor(METHOD_COLORS[key])
            patch.set_alpha(0.7)
    ax6.set_ylabel('TX Power (dBm)')
    ax6.set_title('TX Power Distribution')
    ax6.tick_params(axis='x', rotation=15)

    fig.suptitle('NB-IoT RL Link Adaptation — Comprehensive Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, 'comprehensive_dashboard.png'))
    plt.savefig(os.path.join(output_dir, 'comprehensive_dashboard.eps'), format='eps')
    plt.close()
    print("  ✅ Comprehensive dashboard saved")


def generate_summary_table(datasets, output_dir):
    """Generate a summary CSV table."""
    rows = []
    for key in ['no_la', 'default_la', 'ddqn_la', 'ppo_la', 'marl_la']:
        if key not in datasets:
            continue
        df = datasets[key]
        row = {
            'Method': METHOD_LABELS[key],
            'PDR (%)': f"{calculate_pdr(df):.2f}",
            'RACH Success (%)': f"{calculate_rach_success_rate(df):.2f}",
            'Avg MCS': f"{df['mcs'].mean():.2f}" if 'mcs' in df.columns else 'N/A',
            'Avg Repetitions': f"{df['repetitions'].mean():.2f}" if 'repetitions' in df.columns else 'N/A',
            'Avg TX Power (dBm)': f"{df['tx_power_dbm'].mean():.2f}" if 'tx_power_dbm' in df.columns else 'N/A',
            'Avg Energy (mJ)': f"{df['energy_consumed_mj'].mean():.2f}" if 'energy_consumed_mj' in df.columns else 'N/A',
            'Avg SINR (dB)': f"{df['sinr_db'].mean():.2f}" if 'sinr_db' in df.columns else 'N/A',
            'Avg RSRP (dBm)': f"{df['rsrp_dbm'].mean():.2f}" if 'rsrp_dbm' in df.columns else 'N/A',
            'Avg RACH Attempts': f"{df['rach_attempts'].mean():.2f}" if 'rach_attempts' in df.columns else 'N/A',
            'Total Packets': len(df),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, 'nbiot_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  ✅ Summary table saved: {summary_path}")

    # Print to console
    print("\n" + "=" * 80)
    print("NB-IoT RL Link Adaptation — Summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


def main():
    # Find data directory
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Try common locations
        candidates = [
            'nbiot_datasets',
            '../nbiot_datasets',
            '../../nbiot_datasets',
        ]
        data_dir = None
        for c in candidates:
            if os.path.isdir(c):
                data_dir = c
                break

        if data_dir is None:
            print("Usage: python nbiot_analysis.py [data_directory]")
            print("No nbiot_datasets directory found.")
            sys.exit(1)

    output_dir = os.path.join(data_dir, 'graphs')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("NB-IoT RL Link Adaptation Analysis")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    print("Loading datasets...")
    datasets = load_datasets(data_dir)

    if not datasets:
        print("No datasets found! Run the simulation first.")
        sys.exit(1)

    print(f"\nLoaded {len(datasets)} datasets")
    print("\nGenerating plots...")

    plot_pdr_comparison(datasets, output_dir)
    plot_mcs_distribution(datasets, output_dir)
    plot_repetition_usage(datasets, output_dir)
    plot_energy_comparison(datasets, output_dir)
    plot_rach_analysis(datasets, output_dir)
    plot_sinr_vs_mcs(datasets, output_dir)
    plot_per_ue_pdr(datasets, output_dir)
    plot_pdr_over_time(datasets, output_dir)
    plot_ce_level_distribution(datasets, output_dir)
    plot_comprehensive_dashboard(datasets, output_dir)
    generate_summary_table(datasets, output_dir)

    print(f"\n✅ All analysis complete. Output in: {output_dir}")


if __name__ == '__main__':
    main()
