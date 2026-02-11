#!/usr/bin/env python3
"""
Extended LoRaWAN ADR Metrics Analysis Tool
==========================================
Computes additional paper-grade metrics from existing CSV datasets:
- Reliability metrics (rolling PDR, outage analysis, consecutive-loss CDF)
- Energy efficiency (energy per successful packet, goodput per Joule, battery life)
- Airtime and channel occupancy (Time-on-Air calculations)
- Interference robustness (loss rate vs pre-TX RSSI, SNR margin)
- Stability metrics (SF/TP churn, parameter distributions)
- Fairness metrics (Jain's fairness index, worst-device analysis)
- Enhanced statistical reporting (non-parametric tests, effect sizes, bootstrap CI)

Usage:
    python extended_metrics_analysis.py [--datasets-folder ./lorawan_datasets] [--output-dir ./extended_metrics]
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import argparse

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

# LoRa Time-on-Air parameters (for BW=125kHz, CR=1)
# SNR thresholds for each SF (dB) - minimum required SNR for successful demodulation
SNR_THRESHOLDS = {
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0
}

# Symbol time (ms) for each SF at BW=125kHz
SYMBOL_TIME_MS = {
    7: 1.024,
    8: 2.048,
    9: 4.096,
    10: 8.192,
    11: 16.384,
    12: 32.768
}

# Default payload size (bytes) - matches ns-3 simulation
PAYLOAD_SIZE_BYTES = 20

# Preamble symbols
PREAMBLE_SYMBOLS = 8

# ADR type display names
ADR_NAMES = {
    'no_adr': 'No ADR',
    'classical_adr': 'Classical ADR',
    'ddqn_adr': 'DDQN-PER ADR',
    'ppo_adr': 'PPO ADR',
    'marl_adr': 'MARL ADR',
    'hybrid_adr': 'Hybrid ADR'
}

ADR_ORDER = ['no_adr', 'classical_adr', 'ddqn_adr', 'ppo_adr', 'marl_adr', 'hybrid_adr']

# Plotting style
BAR_FILL = '#A8C8E8'
BAR_EDGE = '#2B5D8A'


# ============================================================================
# TIME-ON-AIR CALCULATION
# ============================================================================

def calculate_time_on_air_ms(sf: int, payload_bytes: int = PAYLOAD_SIZE_BYTES,
                              bw_khz: int = 125, cr: int = 1) -> float:
    """
    Calculate LoRa Time-on-Air in milliseconds.
    
    Formula based on Semtech SX1276 datasheet.
    Assumes explicit header mode, CRC enabled, low data rate optimization for SF11/12.
    """
    # Symbol duration
    t_sym = (2 ** sf) / (bw_khz * 1000) * 1000  # ms
    
    # Preamble time
    t_preamble = (PREAMBLE_SYMBOLS + 4.25) * t_sym
    
    # Low data rate optimization (required for SF11, SF12 at BW125)
    de = 1 if sf >= 11 else 0
    
    # Header mode (1 = explicit header enabled)
    h = 0  # explicit header
    
    # CRC (1 = enabled)
    crc = 1
    
    # Payload symbol count
    pl = payload_bytes
    numerator = 8 * pl - 4 * sf + 28 + 16 * crc - 20 * h
    denominator = 4 * (sf - 2 * de)
    
    if denominator <= 0:
        denominator = 4 * sf
    
    n_payload = 8 + max(np.ceil(numerator / denominator) * (cr + 4), 0)
    
    # Total time on air
    t_payload = n_payload * t_sym
    toa = t_preamble + t_payload
    
    return toa


def add_toa_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add Time-on-Air column to dataframe."""
    df = df.copy()
    df['toa_ms'] = df.apply(
        lambda row: calculate_time_on_air_ms(
            sf=int(row['sf']),
            payload_bytes=PAYLOAD_SIZE_BYTES,
            bw_khz=int(row['bw']) // 1000 if 'bw' in row else 125,
            cr=int(row['cr']) if 'cr' in row else 1
        ),
        axis=1
    )
    return df


# ============================================================================
# RELIABILITY METRICS
# ============================================================================

@dataclass
class ReliabilityMetrics:
    """Container for reliability metrics."""
    # Basic
    total_packets: int
    successful_packets: int
    pdr: float
    
    # Rolling PDR stats
    rolling_pdr_mean: float
    rolling_pdr_std: float
    rolling_pdr_min: float
    
    # Outage analysis
    outage_count: int
    avg_outage_length: float
    max_outage_length: int
    total_outage_packets: int
    
    # Consecutive loss distribution
    consecutive_loss_lengths: List[int]
    
    # PDR over time stability
    pdr_coefficient_of_variation: float


def compute_consecutive_runs(packet_received: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Compute consecutive runs of successes and failures.
    Returns (success_runs, failure_runs) as lists of run lengths.
    """
    if len(packet_received) == 0:
        return [], []
    
    success_runs = []
    failure_runs = []
    
    current_value = packet_received[0]
    current_length = 1
    
    for i in range(1, len(packet_received)):
        if packet_received[i] == current_value:
            current_length += 1
        else:
            if current_value == 1:
                success_runs.append(current_length)
            else:
                failure_runs.append(current_length)
            current_value = packet_received[i]
            current_length = 1
    
    # Don't forget the last run
    if current_value == 1:
        success_runs.append(current_length)
    else:
        failure_runs.append(current_length)
    
    return success_runs, failure_runs


def compute_reliability_metrics(df: pd.DataFrame, window_size: int = 20,
                                 outage_threshold: int = 3) -> ReliabilityMetrics:
    """
    Compute comprehensive reliability metrics for a device.
    
    Args:
        df: DataFrame with 'packet_received' (0/1) and 'timestamp' columns
        window_size: Rolling window size for PDR calculation
        outage_threshold: Minimum consecutive losses to count as "outage"
    """
    packet_received = df['packet_received'].values
    total = len(packet_received)
    successes = int(packet_received.sum())
    pdr = 100.0 * successes / total if total > 0 else 0.0
    
    # Rolling PDR
    rolling_pdr = pd.Series(packet_received).rolling(window=window_size, min_periods=1).mean() * 100
    rolling_mean = rolling_pdr.mean()
    rolling_std = rolling_pdr.std()
    rolling_min = rolling_pdr.min()
    
    # Coefficient of variation of rolling PDR (stability measure)
    cv = rolling_std / rolling_mean if rolling_mean > 0 else float('inf')
    
    # Consecutive loss analysis
    _, failure_runs = compute_consecutive_runs(packet_received)
    
    # Outage analysis (consecutive losses >= threshold)
    outages = [r for r in failure_runs if r >= outage_threshold]
    outage_count = len(outages)
    avg_outage_length = np.mean(outages) if outages else 0.0
    max_outage_length = max(outages) if outages else 0
    total_outage_packets = sum(outages)
    
    return ReliabilityMetrics(
        total_packets=total,
        successful_packets=successes,
        pdr=pdr,
        rolling_pdr_mean=rolling_mean,
        rolling_pdr_std=rolling_std,
        rolling_pdr_min=rolling_min,
        outage_count=outage_count,
        avg_outage_length=avg_outage_length,
        max_outage_length=max_outage_length,
        total_outage_packets=total_outage_packets,
        consecutive_loss_lengths=failure_runs,
        pdr_coefficient_of_variation=cv
    )


# ============================================================================
# ENERGY METRICS
# ============================================================================

@dataclass
class EnergyMetrics:
    """Container for energy efficiency metrics."""
    total_energy_mJ: float
    energy_per_packet_mJ: float
    energy_per_success_mJ: float  # Key metric!
    
    # Goodput efficiency
    goodput_bytes_per_second: float
    energy_efficiency_bytes_per_mJ: float
    
    # Battery life projection (2400mAh @ 3.0V = 25920 J)
    projected_battery_life_days: float
    transmissions_per_day: float


def compute_energy_metrics(df: pd.DataFrame, payload_bytes: int = PAYLOAD_SIZE_BYTES,
                           battery_capacity_mAh: float = 2400,
                           battery_voltage: float = 3.0) -> EnergyMetrics:
    """
    Compute energy efficiency metrics.
    """
    total_energy = df['energy_consumed_mJ'].sum()
    total_packets = len(df)
    successful_packets = df['packet_received'].sum()
    
    # Basic energy metrics
    energy_per_packet = total_energy / total_packets if total_packets > 0 else 0.0
    energy_per_success = total_energy / successful_packets if successful_packets > 0 else float('inf')
    
    # Time span
    duration_seconds = df['timestamp'].max() - df['timestamp'].min()
    if duration_seconds <= 0:
        duration_seconds = 1.0
    
    # Goodput (successful bytes per second)
    goodput_bps = (successful_packets * payload_bytes) / duration_seconds
    
    # Energy efficiency (bytes delivered per mJ)
    energy_efficiency = (successful_packets * payload_bytes) / total_energy if total_energy > 0 else 0.0
    
    # Battery life projection
    battery_capacity_J = battery_capacity_mAh * battery_voltage * 3.6  # mAh * V * 3.6 = J
    battery_capacity_mJ = battery_capacity_J * 1000
    
    # Energy per day (extrapolate from simulation)
    energy_per_day = (total_energy / duration_seconds) * 86400
    projected_battery_life_days = battery_capacity_mJ / energy_per_day if energy_per_day > 0 else float('inf')
    
    transmissions_per_day = (total_packets / duration_seconds) * 86400
    
    return EnergyMetrics(
        total_energy_mJ=total_energy,
        energy_per_packet_mJ=energy_per_packet,
        energy_per_success_mJ=energy_per_success,
        goodput_bytes_per_second=goodput_bps,
        energy_efficiency_bytes_per_mJ=energy_efficiency,
        projected_battery_life_days=projected_battery_life_days,
        transmissions_per_day=transmissions_per_day
    )


# ============================================================================
# AIRTIME METRICS
# ============================================================================

@dataclass
class AirtimeMetrics:
    """Container for airtime/channel occupancy metrics."""
    total_airtime_ms: float
    avg_toa_ms: float
    toa_per_success_ms: float
    airtime_per_minute_ms: float
    channel_utilization_percent: float  # Assumes duty cycle limit of 1%


def compute_airtime_metrics(df: pd.DataFrame) -> AirtimeMetrics:
    """
    Compute airtime and channel occupancy metrics.
    """
    # Ensure ToA column exists
    if 'toa_ms' not in df.columns:
        df = add_toa_column(df)
    
    total_airtime = df['toa_ms'].sum()
    avg_toa = df['toa_ms'].mean()
    successful_packets = df['packet_received'].sum()
    
    # ToA per successful packet (penalizes retransmissions conceptually)
    toa_per_success = total_airtime / successful_packets if successful_packets > 0 else float('inf')
    
    # Airtime per minute
    duration_seconds = df['timestamp'].max() - df['timestamp'].min()
    if duration_seconds <= 0:
        duration_seconds = 1.0
    airtime_per_minute = (total_airtime / duration_seconds) * 60
    
    # Channel utilization (relative to 1% duty cycle limit = 600ms per minute)
    duty_cycle_limit_ms_per_min = 600  # 1% of 60000ms
    channel_utilization = (airtime_per_minute / duty_cycle_limit_ms_per_min) * 100
    
    return AirtimeMetrics(
        total_airtime_ms=total_airtime,
        avg_toa_ms=avg_toa,
        toa_per_success_ms=toa_per_success,
        airtime_per_minute_ms=airtime_per_minute,
        channel_utilization_percent=channel_utilization
    )


# ============================================================================
# INTERFERENCE ROBUSTNESS METRICS
# ============================================================================

@dataclass
class InterferenceMetrics:
    """Container for interference robustness metrics."""
    # Loss rate by pre-TX RSSI bins
    loss_rate_by_rssi: Dict[str, float]
    
    # SNR margin analysis (received packets only)
    median_snr_margin: float
    p10_snr_margin: float  # 10th percentile
    pct_near_threshold: float  # % with margin < 3dB
    
    # Loss classification (heuristic)
    pct_collision_likely: float
    pct_weak_signal_likely: float


def compute_interference_metrics(df: pd.DataFrame,
                                  collision_rssi_threshold: float = -105.0) -> InterferenceMetrics:
    """
    Compute interference robustness metrics.
    
    Args:
        df: DataFrame with pre_tx_rssi_dBm, snr_db, sf, packet_received
        collision_rssi_threshold: If loss occurs with RSSI above this, classify as collision-likely
    """
    # Loss rate by pre-TX RSSI bins
    rssi_bins = [(-130, -115), (-115, -105), (-105, -95), (-95, -85), (-85, -70)]
    loss_rate_by_rssi = {}
    
    for low, high in rssi_bins:
        bin_label = f"{low} to {high}"
        mask = (df['pre_tx_rssi_dBm'] >= low) & (df['pre_tx_rssi_dBm'] < high)
        bin_data = df[mask]
        if len(bin_data) > 0:
            loss_rate = 100.0 * (1 - bin_data['packet_received'].mean())
            loss_rate_by_rssi[bin_label] = loss_rate
        else:
            loss_rate_by_rssi[bin_label] = np.nan
    
    # SNR margin analysis (for received packets only)
    received = df[df['snr_db'] != -999.0].copy()
    if len(received) > 0:
        received['snr_threshold'] = received['sf'].map(SNR_THRESHOLDS)
        received['snr_margin'] = received['snr_db'] - received['snr_threshold']
        
        median_margin = received['snr_margin'].median()
        p10_margin = received['snr_margin'].quantile(0.10)
        pct_near_threshold = 100.0 * (received['snr_margin'] < 3.0).mean()
    else:
        median_margin = np.nan
        p10_margin = np.nan
        pct_near_threshold = np.nan
    
    # Loss classification heuristic
    lost = df[df['packet_received'] == 0]
    if len(lost) > 0:
        collision_likely = lost[lost['pre_tx_rssi_dBm'] > collision_rssi_threshold]
        weak_signal_likely = lost[lost['pre_tx_rssi_dBm'] <= collision_rssi_threshold]
        
        pct_collision = 100.0 * len(collision_likely) / len(lost)
        pct_weak = 100.0 * len(weak_signal_likely) / len(lost)
    else:
        pct_collision = 0.0
        pct_weak = 0.0
    
    return InterferenceMetrics(
        loss_rate_by_rssi=loss_rate_by_rssi,
        median_snr_margin=median_margin,
        p10_snr_margin=p10_margin,
        pct_near_threshold=pct_near_threshold,
        pct_collision_likely=pct_collision,
        pct_weak_signal_likely=pct_weak
    )


# ============================================================================
# STABILITY METRICS (PARAMETER CHURN)
# ============================================================================

@dataclass
class StabilityMetrics:
    """Container for algorithm stability metrics."""
    sf_change_rate: float  # Fraction of packets where SF changed
    tp_change_rate: float  # Fraction of packets where TX power changed
    
    # Distribution stats
    sf_distribution: Dict[int, float]  # SF -> percentage
    tp_distribution: Dict[int, float]  # TP bin -> percentage
    
    avg_sf: float
    std_sf: float
    avg_tp: float
    std_tp: float


def compute_stability_metrics(df: pd.DataFrame) -> StabilityMetrics:
    """
    Compute algorithm stability metrics (how often parameters change).
    """
    sf_values = df['sf'].values
    tp_values = df['tx_power_dBm'].values
    
    # Change rates
    if len(sf_values) > 1:
        sf_changes = np.sum(sf_values[1:] != sf_values[:-1])
        sf_change_rate = sf_changes / (len(sf_values) - 1)
        
        tp_changes = np.sum(np.abs(tp_values[1:] - tp_values[:-1]) > 0.5)  # 0.5 dB tolerance
        tp_change_rate = tp_changes / (len(tp_values) - 1)
    else:
        sf_change_rate = 0.0
        tp_change_rate = 0.0
    
    # SF distribution
    sf_series = pd.Series(sf_values)
    sf_counts = sf_series.value_counts()
    sf_total = sf_counts.sum()
    sf_distribution = {int(sf): (count / sf_total * 100) for sf, count in sf_counts.items()}
    
    # TP distribution (bin into 2, 5, 8, 11, 14 dBm)
    tp_bins = pd.cut(tp_values, bins=[0, 4, 7, 10, 13, 16], labels=['2-4', '5-7', '8-10', '11-13', '14+'])
    tp_counts = tp_bins.value_counts()
    tp_total = tp_counts.sum()
    tp_distribution = {str(bin_label): (count / tp_total * 100) if tp_total > 0 else 0 for bin_label, count in tp_counts.items()}
    
    return StabilityMetrics(
        sf_change_rate=sf_change_rate,
        tp_change_rate=tp_change_rate,
        sf_distribution=sf_distribution,
        tp_distribution=tp_distribution,
        avg_sf=np.mean(sf_values),
        std_sf=np.std(sf_values),
        avg_tp=np.mean(tp_values),
        std_tp=np.std(tp_values)
    )


# ============================================================================
# FAIRNESS METRICS
# ============================================================================

@dataclass
class FairnessMetrics:
    """Container for fairness metrics across devices."""
    jains_fairness_pdr: float
    jains_fairness_energy_efficiency: float
    
    min_pdr: float
    p5_pdr: float  # 5th percentile
    max_pdr: float
    
    worst_device_id: str
    best_device_id: str


def compute_jains_fairness(values: np.ndarray) -> float:
    """
    Compute Jain's Fairness Index.
    J = (sum(x))^2 / (n * sum(x^2))
    Range: [1/n, 1] where 1 is perfectly fair.
    """
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    
    sum_x = np.sum(values)
    sum_x2 = np.sum(values ** 2)
    
    return (sum_x ** 2) / (n * sum_x2)


def compute_fairness_metrics(device_stats: Dict[str, Dict]) -> FairnessMetrics:
    """
    Compute fairness metrics across devices for a given ADR type.
    
    Args:
        device_stats: Dict mapping device_id -> stats dict with 'pdr' and 'energy_per_success_mJ'
    """
    if not device_stats:
        return FairnessMetrics(
            jains_fairness_pdr=0.0,
            jains_fairness_energy_efficiency=0.0,
            min_pdr=0.0,
            p5_pdr=0.0,
            max_pdr=0.0,
            worst_device_id='N/A',
            best_device_id='N/A'
        )
    
    pdrs = np.array([s['pdr'] for s in device_stats.values()])
    device_ids = list(device_stats.keys())
    
    # Energy efficiency = 1 / energy_per_success (higher is better)
    energy_efficiencies = []
    for s in device_stats.values():
        eps = s.get('energy_per_success_mJ', float('inf'))
        if eps > 0 and eps != float('inf'):
            energy_efficiencies.append(1.0 / eps)
        else:
            energy_efficiencies.append(0.0)
    energy_efficiencies = np.array(energy_efficiencies)
    
    # Jain's fairness
    jains_pdr = compute_jains_fairness(pdrs)
    jains_energy = compute_jains_fairness(energy_efficiencies)
    
    # Worst/best devices
    min_pdr = np.min(pdrs)
    max_pdr = np.max(pdrs)
    p5_pdr = np.percentile(pdrs, 5)
    
    worst_idx = np.argmin(pdrs)
    best_idx = np.argmax(pdrs)
    
    return FairnessMetrics(
        jains_fairness_pdr=jains_pdr,
        jains_fairness_energy_efficiency=jains_energy,
        min_pdr=min_pdr,
        p5_pdr=p5_pdr,
        max_pdr=max_pdr,
        worst_device_id=device_ids[worst_idx],
        best_device_id=device_ids[best_idx]
    )


# ============================================================================
# ENHANCED STATISTICAL ANALYSIS
# ============================================================================

def compute_cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
    """
    Compute Cliff's Delta effect size (non-parametric).
    
    Interpretation:
    |d| < 0.147: negligible
    |d| < 0.33: small
    |d| < 0.474: medium
    |d| >= 0.474: large
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0, 'negligible'
    
    # Count dominance
    count = 0
    for x1 in group1:
        for x2 in group2:
            if x1 > x2:
                count += 1
            elif x1 < x2:
                count -= 1
    
    delta = count / (n1 * n2)
    
    # Interpret
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = 'negligible'
    elif abs_delta < 0.33:
        interpretation = 'small'
    elif abs_delta < 0.474:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return delta, interpretation


def bootstrap_ci(data: np.ndarray, stat_func=np.mean, n_bootstrap: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    point_estimate = stat_func(data)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(stat_func(sample))
    
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def enhanced_statistical_comparison(group1: np.ndarray, group2: np.ndarray,
                                    group1_name: str, group2_name: str) -> Dict:
    """
    Perform comprehensive statistical comparison between two groups.
    """
    results = {
        'comparison': f'{group1_name} vs {group2_name}',
        'n1': len(group1),
        'n2': len(group2),
        'mean1': np.mean(group1) if len(group1) > 0 else np.nan,
        'mean2': np.mean(group2) if len(group2) > 0 else np.nan,
    }
    
    if len(group1) < 2 or len(group2) < 2:
        return results
    
    # Mann-Whitney U test (non-parametric)
    u_stat, p_value_mw = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    results['mann_whitney_u'] = u_stat
    results['mann_whitney_p'] = p_value_mw
    results['mann_whitney_sig'] = 'significant' if p_value_mw < 0.05 else 'not significant'
    
    # t-test (parametric)
    t_stat, p_value_t = stats.ttest_ind(group1, group2)
    results['ttest_t'] = t_stat
    results['ttest_p'] = p_value_t
    results['ttest_sig'] = 'significant' if p_value_t < 0.05 else 'not significant'
    
    # Effect size (Cliff's delta)
    delta, interpretation = compute_cliffs_delta(group1, group2)
    results['cliffs_delta'] = delta
    results['effect_size'] = interpretation
    
    # Bootstrap CI for difference
    combined = np.concatenate([group1, group2])
    diff = np.mean(group1) - np.mean(group2)
    _, ci_low, ci_high = bootstrap_ci(
        group1 - np.mean(group2),  # Centered difference
        stat_func=np.mean
    )
    results['mean_diff'] = diff
    results['diff_ci_lower'] = ci_low
    results['diff_ci_upper'] = ci_high
    
    return results


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class ExtendedMetricsAnalyzer:
    """
    Extended metrics analyzer for LoRaWAN ADR comparison.
    """
    
    def __init__(self, datasets_folder: str, output_dir: str = "extended_metrics"):
        self.datasets_folder = Path(datasets_folder)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.all_metrics: Dict[str, Dict[str, Dict]] = {}
        
    def load_datasets(self):
        """Load all CSV datasets."""
        print("Loading LoRaWAN datasets...")
        
        patterns = {
            'no_adr': 'no_adr_*_device_*_dataset.csv',
            'classical_adr': 'adr_*_device_*_dataset.csv',
            'ddqn_adr': 'ddqn_adr_*_device_*_dataset.csv',
            'ppo_adr': 'ppo_adr_*_device_*_dataset.csv',
            'marl_adr': 'marl_adr_*_device_*_dataset.csv',
            'hybrid_adr': 'hybrid_adr_*_device_*_dataset.csv'
        }
        
        for adr_type, pattern in patterns.items():
            files = glob.glob(str(self.datasets_folder / pattern))
            self.device_data[adr_type] = {}
            
            for file_path in files:
                filename = os.path.basename(file_path)
                # Extract device ID (format: ..._device_XX_dataset.csv)
                try:
                    import re
                    match = re.search(r'device_(\d+)_dataset', filename)
                    if match:
                        device_id = match.group(1)
                    else:
                        continue
                except:
                    continue
                
                try:
                    df = pd.read_csv(file_path)
                    # Add ToA column
                    df = add_toa_column(df)
                    self.device_data[adr_type][device_id] = df
                    print(f"  Loaded {adr_type} device {device_id}: {len(df)} packets")
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
        
        total_devices = sum(len(d) for d in self.device_data.values())
        print(f"\nLoaded {total_devices} device datasets across {len(self.device_data)} ADR types")
    
    def compute_all_metrics(self):
        """Compute all extended metrics for each device and ADR type."""
        print("\nComputing extended metrics...")
        
        for adr_type, devices in self.device_data.items():
            self.all_metrics[adr_type] = {}
            
            for device_id, df in devices.items():
                metrics = {}
                
                # Reliability
                rel = compute_reliability_metrics(df)
                metrics['reliability'] = {
                    'total_packets': rel.total_packets,
                    'successful_packets': rel.successful_packets,
                    'pdr': rel.pdr,
                    'rolling_pdr_mean': rel.rolling_pdr_mean,
                    'rolling_pdr_std': rel.rolling_pdr_std,
                    'rolling_pdr_min': rel.rolling_pdr_min,
                    'outage_count': rel.outage_count,
                    'avg_outage_length': rel.avg_outage_length,
                    'max_outage_length': rel.max_outage_length,
                    'pdr_cv': rel.pdr_coefficient_of_variation,
                    'consecutive_loss_lengths': rel.consecutive_loss_lengths
                }
                
                # Energy
                energy = compute_energy_metrics(df)
                metrics['energy'] = {
                    'total_energy_mJ': energy.total_energy_mJ,
                    'energy_per_packet_mJ': energy.energy_per_packet_mJ,
                    'energy_per_success_mJ': energy.energy_per_success_mJ,
                    'goodput_bytes_per_s': energy.goodput_bytes_per_second,
                    'energy_efficiency_bytes_per_mJ': energy.energy_efficiency_bytes_per_mJ,
                    'projected_battery_life_days': energy.projected_battery_life_days,
                    'transmissions_per_day': energy.transmissions_per_day
                }
                
                # Airtime
                airtime = compute_airtime_metrics(df)
                metrics['airtime'] = {
                    'total_airtime_ms': airtime.total_airtime_ms,
                    'avg_toa_ms': airtime.avg_toa_ms,
                    'toa_per_success_ms': airtime.toa_per_success_ms,
                    'airtime_per_minute_ms': airtime.airtime_per_minute_ms,
                    'channel_utilization_pct': airtime.channel_utilization_percent
                }
                
                # Interference
                interference = compute_interference_metrics(df)
                metrics['interference'] = {
                    'loss_rate_by_rssi': interference.loss_rate_by_rssi,
                    'median_snr_margin': interference.median_snr_margin,
                    'p10_snr_margin': interference.p10_snr_margin,
                    'pct_near_threshold': interference.pct_near_threshold,
                    'pct_collision_likely': interference.pct_collision_likely,
                    'pct_weak_signal_likely': interference.pct_weak_signal_likely
                }
                
                # Stability
                stability = compute_stability_metrics(df)
                metrics['stability'] = {
                    'sf_change_rate': stability.sf_change_rate,
                    'tp_change_rate': stability.tp_change_rate,
                    'sf_distribution': stability.sf_distribution,
                    'tp_distribution': stability.tp_distribution,
                    'avg_sf': stability.avg_sf,
                    'std_sf': stability.std_sf,
                    'avg_tp': stability.avg_tp,
                    'std_tp': stability.std_tp
                }
                
                # Distance (from first row)
                metrics['distance_m'] = df['distance_m'].iloc[0] if len(df) > 0 else 0
                
                self.all_metrics[adr_type][device_id] = metrics
        
        print("Metrics computation complete!")
    
    def print_summary_report(self):
        """Print comprehensive summary report."""
        print("\n" + "=" * 80)
        print("EXTENDED METRICS SUMMARY REPORT")
        print("=" * 80)
        
        for adr_type in ADR_ORDER:
            if adr_type not in self.all_metrics or not self.all_metrics[adr_type]:
                continue
            
            devices = self.all_metrics[adr_type]
            n_devices = len(devices)
            
            print(f"\n{'='*60}")
            print(f"ADR TYPE: {ADR_NAMES.get(adr_type, adr_type)}")
            print(f"{'='*60}")
            print(f"Devices analyzed: {n_devices}")
            
            # Aggregate metrics
            pdrs = [d['reliability']['pdr'] for d in devices.values()]
            energy_per_success = [d['energy']['energy_per_success_mJ'] for d in devices.values()]
            toa_per_success = [d['airtime']['toa_per_success_ms'] for d in devices.values()]
            sf_churn = [d['stability']['sf_change_rate'] * 100 for d in devices.values()]
            outage_counts = [d['reliability']['outage_count'] for d in devices.values()]
            battery_life = [d['energy']['projected_battery_life_days'] for d in devices.values()]
            
            print(f"\n--- RELIABILITY ---")
            print(f"  PDR:              {np.mean(pdrs):.2f}% (std: {np.std(pdrs):.2f}%, min: {np.min(pdrs):.2f}%)")
            print(f"  Outage count:     {np.mean(outage_counts):.1f} avg, {np.max(outage_counts)} max per device")
            
            print(f"\n--- ENERGY EFFICIENCY ---")
            eps_finite = [e for e in energy_per_success if e != float('inf')]
            if eps_finite:
                print(f"  Energy/success:   {np.mean(eps_finite):.3f} mJ (std: {np.std(eps_finite):.3f})")
            print(f"  Battery life:     {np.mean(battery_life):.1f} days projected")
            
            print(f"\n--- AIRTIME ---")
            print(f"  ToA/success:      {np.mean(toa_per_success):.1f} ms avg")
            
            print(f"\n--- STABILITY ---")
            print(f"  SF change rate:   {np.mean(sf_churn):.1f}% of packets")
            
            # SF distribution aggregate
            sf_totals = {}
            for d in devices.values():
                for sf, pct in d['stability']['sf_distribution'].items():
                    sf_totals[sf] = sf_totals.get(sf, 0) + pct
            for sf in sorted(sf_totals.keys()):
                sf_totals[sf] /= n_devices
            print(f"  SF distribution:  " + ", ".join([f"SF{sf}: {pct:.0f}%" for sf, pct in sorted(sf_totals.items())]))
            
            # Fairness
            device_stats = {
                did: {'pdr': d['reliability']['pdr'], 'energy_per_success_mJ': d['energy']['energy_per_success_mJ']}
                for did, d in devices.items()
            }
            fairness = compute_fairness_metrics(device_stats)
            
            print(f"\n--- FAIRNESS ---")
            print(f"  Jain's Index (PDR): {fairness.jains_fairness_pdr:.4f}")
            print(f"  5th percentile PDR: {fairness.p5_pdr:.2f}%")
            print(f"  Worst device:       {fairness.worst_device_id} ({fairness.min_pdr:.2f}% PDR)")
            print(f"  Best device:        {fairness.best_device_id} ({fairness.max_pdr:.2f}% PDR)")
    
    def print_statistical_comparisons(self):
        """Print statistical comparisons between ADR types."""
        print("\n" + "=" * 80)
        print("STATISTICAL COMPARISONS")
        print("=" * 80)
        
        # Collect PDR arrays per ADR type
        pdr_by_type = {}
        energy_by_type = {}
        
        for adr_type, devices in self.all_metrics.items():
            if devices:
                pdr_by_type[adr_type] = np.array([d['reliability']['pdr'] for d in devices.values()])
                eps = [d['energy']['energy_per_success_mJ'] for d in devices.values()]
                energy_by_type[adr_type] = np.array([e for e in eps if e != float('inf')])
        
        # Compare each RL method vs classical ADR
        comparisons = [
            ('ddqn_adr', 'classical_adr'),
            ('ppo_adr', 'classical_adr'),
            ('marl_adr', 'classical_adr'),
            ('ddqn_adr', 'no_adr'),
            ('marl_adr', 'ddqn_adr'),
        ]
        
        print("\n--- PDR Comparisons ---")
        for g1, g2 in comparisons:
            if g1 in pdr_by_type and g2 in pdr_by_type:
                result = enhanced_statistical_comparison(
                    pdr_by_type[g1], pdr_by_type[g2],
                    ADR_NAMES.get(g1, g1), ADR_NAMES.get(g2, g2)
                )
                print(f"\n{result['comparison']}:")
                print(f"  Mean diff: {result.get('mean_diff', 0):.2f} pp")
                print(f"  Mann-Whitney p: {result.get('mann_whitney_p', 'N/A'):.4f} ({result.get('mann_whitney_sig', 'N/A')})")
                print(f"  Cliff's delta: {result.get('cliffs_delta', 0):.3f} ({result.get('effect_size', 'N/A')})")
    
    def create_visualizations(self):
        """Create extended metrics visualizations."""
        print("\nGenerating visualizations...")
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'figure.dpi': 150,
        })
        
        # Collect data for plotting
        present_types = [t for t in ADR_ORDER if t in self.all_metrics and self.all_metrics[t]]
        if not present_types:
            print("  No data to visualize")
            return
        
        # =====================================================================
        # FIGURE 1: Energy per Successful Packet
        # =====================================================================
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        
        eps_data = []
        labels = []
        for adr_type in present_types:
            devices = self.all_metrics[adr_type]
            eps = [d['energy']['energy_per_success_mJ'] for d in devices.values() if d['energy']['energy_per_success_mJ'] != float('inf')]
            if eps:
                eps_data.append(eps)
                labels.append(ADR_NAMES.get(adr_type, adr_type).replace(' ', '\n'))
        
        if eps_data:
            bp = ax1.boxplot(eps_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(BAR_FILL)
                patch.set_edgecolor(BAR_EDGE)
            ax1.set_ylabel('Energy per Successful Packet (mJ)')
            ax1.set_title('Energy Efficiency by ADR Type')
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            fig1.tight_layout()
            fig1.savefig(self.output_dir / 'energy_per_success.png', dpi=300)
            fig1.savefig(self.output_dir / 'energy_per_success.eps', format='eps')
            print(f"  Saved: energy_per_success.png/eps")
        plt.close(fig1)
        
        # =====================================================================
        # FIGURE 2: PDR vs Distance (scatter)
        # =====================================================================
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(present_types)))
        
        for idx, adr_type in enumerate(present_types):
            devices = self.all_metrics[adr_type]
            distances = [d['distance_m'] for d in devices.values()]
            pdrs = [d['reliability']['pdr'] for d in devices.values()]
            ax2.scatter(distances, pdrs, alpha=0.7, label=ADR_NAMES.get(adr_type, adr_type),
                       color=colors[idx], s=40)
        
        ax2.set_xlabel('Distance from Gateway (m)')
        ax2.set_ylabel('Packet Delivery Rate (%)')
        ax2.set_title('PDR vs Distance by ADR Type')
        ax2.legend(loc='lower left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
        fig2.tight_layout()
        fig2.savefig(self.output_dir / 'pdr_vs_distance.png', dpi=300)
        fig2.savefig(self.output_dir / 'pdr_vs_distance.eps', format='eps')
        print(f"  Saved: pdr_vs_distance.png/eps")
        plt.close(fig2)
        
        # =====================================================================
        # FIGURE 3: Consecutive Loss CDF
        # =====================================================================
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        
        for idx, adr_type in enumerate(present_types):
            devices = self.all_metrics[adr_type]
            all_losses = []
            for d in devices.values():
                all_losses.extend(d['reliability']['consecutive_loss_lengths'])
            
            if all_losses:
                sorted_losses = np.sort(all_losses)
                cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
                ax3.plot(sorted_losses, cdf, label=ADR_NAMES.get(adr_type, adr_type),
                        color=colors[idx], linewidth=1.5)
        
        ax3.set_xlabel('Consecutive Loss Length (packets)')
        ax3.set_ylabel('CDF')
        ax3.set_title('CDF of Consecutive Packet Losses')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, min(50, ax3.get_xlim()[1]))
        fig3.tight_layout()
        fig3.savefig(self.output_dir / 'consecutive_loss_cdf.png', dpi=300)
        fig3.savefig(self.output_dir / 'consecutive_loss_cdf.eps', format='eps')
        print(f"  Saved: consecutive_loss_cdf.png/eps")
        plt.close(fig3)
        
        # =====================================================================
        # FIGURE 4: SF Distribution Comparison
        # =====================================================================
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        
        sf_range = [7, 8, 9, 10, 11, 12]
        x = np.arange(len(sf_range))
        width = 0.12
        
        for idx, adr_type in enumerate(present_types):
            devices = self.all_metrics[adr_type]
            n_devices = len(devices)
            
            sf_avg = {sf: 0 for sf in sf_range}
            for d in devices.values():
                for sf, pct in d['stability']['sf_distribution'].items():
                    if sf in sf_avg:
                        sf_avg[sf] += pct / n_devices
            
            values = [sf_avg[sf] for sf in sf_range]
            offset = (idx - len(present_types) / 2 + 0.5) * width
            ax4.bar(x + offset, values, width, label=ADR_NAMES.get(adr_type, adr_type),
                   color=colors[idx], edgecolor='black', linewidth=0.5)
        
        ax4.set_xlabel('Spreading Factor')
        ax4.set_ylabel('Usage (%)')
        ax4.set_title('SF Distribution by ADR Type')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'SF{sf}' for sf in sf_range])
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(self.output_dir / 'sf_distribution.png', dpi=300)
        fig4.savefig(self.output_dir / 'sf_distribution.eps', format='eps')
        print(f"  Saved: sf_distribution.png/eps")
        plt.close(fig4)
        
        # =====================================================================
        # FIGURE 5: Jain's Fairness Index
        # =====================================================================
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        
        fairness_values = []
        fairness_labels = []
        
        for adr_type in present_types:
            devices = self.all_metrics[adr_type]
            device_stats = {
                did: {'pdr': d['reliability']['pdr'], 'energy_per_success_mJ': d['energy']['energy_per_success_mJ']}
                for did, d in devices.items()
            }
            fairness = compute_fairness_metrics(device_stats)
            fairness_values.append(fairness.jains_fairness_pdr)
            fairness_labels.append(ADR_NAMES.get(adr_type, adr_type).replace(' ', '\n'))
        
        bars = ax5.bar(fairness_labels, fairness_values, color=BAR_FILL, edgecolor=BAR_EDGE)
        ax5.set_ylabel("Jain's Fairness Index")
        ax5.set_title('PDR Fairness Across Devices')
        ax5.set_ylim(0, 1.05)
        ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect fairness')
        ax5.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, fairness_values):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=9)
        
        fig5.tight_layout()
        fig5.savefig(self.output_dir / 'jains_fairness.png', dpi=300)
        fig5.savefig(self.output_dir / 'jains_fairness.eps', format='eps')
        print(f"  Saved: jains_fairness.png/eps")
        plt.close(fig5)
        
        # =====================================================================
        # FIGURE 6: Loss Classification (Collision vs Weak Signal)
        # =====================================================================
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        
        collision_pct = []
        weak_pct = []
        
        for adr_type in present_types:
            devices = self.all_metrics[adr_type]
            coll = np.mean([d['interference']['pct_collision_likely'] for d in devices.values()])
            weak = np.mean([d['interference']['pct_weak_signal_likely'] for d in devices.values()])
            collision_pct.append(coll)
            weak_pct.append(weak)
        
        x = np.arange(len(present_types))
        width = 0.35
        
        ax6.bar(x - width/2, collision_pct, width, label='Collision-likely', color='#E57373', edgecolor='black')
        ax6.bar(x + width/2, weak_pct, width, label='Weak-signal-likely', color='#64B5F6', edgecolor='black')
        
        ax6.set_xlabel('ADR Type')
        ax6.set_ylabel('Percentage of Lost Packets (%)')
        ax6.set_title('Loss Classification by ADR Type')
        ax6.set_xticks(x)
        ax6.set_xticklabels([ADR_NAMES.get(t, t).replace(' ', '\n') for t in present_types])
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        fig6.tight_layout()
        fig6.savefig(self.output_dir / 'loss_classification.png', dpi=300)
        fig6.savefig(self.output_dir / 'loss_classification.eps', format='eps')
        print(f"  Saved: loss_classification.png/eps")
        plt.close(fig6)
        
        print(f"\nAll visualizations saved to: {self.output_dir}")
    
    def export_to_csv(self):
        """Export all metrics to CSV files."""
        print("\nExporting metrics to CSV...")
        
        # Flatten metrics for each device
        rows = []
        for adr_type, devices in self.all_metrics.items():
            for device_id, metrics in devices.items():
                row = {
                    'adr_type': adr_type,
                    'device_id': device_id,
                    'distance_m': metrics['distance_m'],
                    
                    # Reliability
                    'pdr': metrics['reliability']['pdr'],
                    'total_packets': metrics['reliability']['total_packets'],
                    'successful_packets': metrics['reliability']['successful_packets'],
                    'rolling_pdr_mean': metrics['reliability']['rolling_pdr_mean'],
                    'rolling_pdr_std': metrics['reliability']['rolling_pdr_std'],
                    'rolling_pdr_min': metrics['reliability']['rolling_pdr_min'],
                    'outage_count': metrics['reliability']['outage_count'],
                    'avg_outage_length': metrics['reliability']['avg_outage_length'],
                    'max_outage_length': metrics['reliability']['max_outage_length'],
                    'pdr_cv': metrics['reliability']['pdr_cv'],
                    
                    # Energy
                    'total_energy_mJ': metrics['energy']['total_energy_mJ'],
                    'energy_per_packet_mJ': metrics['energy']['energy_per_packet_mJ'],
                    'energy_per_success_mJ': metrics['energy']['energy_per_success_mJ'],
                    'goodput_bytes_per_s': metrics['energy']['goodput_bytes_per_s'],
                    'energy_efficiency_bytes_per_mJ': metrics['energy']['energy_efficiency_bytes_per_mJ'],
                    'projected_battery_life_days': metrics['energy']['projected_battery_life_days'],
                    
                    # Airtime
                    'total_airtime_ms': metrics['airtime']['total_airtime_ms'],
                    'avg_toa_ms': metrics['airtime']['avg_toa_ms'],
                    'toa_per_success_ms': metrics['airtime']['toa_per_success_ms'],
                    'airtime_per_minute_ms': metrics['airtime']['airtime_per_minute_ms'],
                    'channel_utilization_pct': metrics['airtime']['channel_utilization_pct'],
                    
                    # Interference
                    'median_snr_margin': metrics['interference']['median_snr_margin'],
                    'p10_snr_margin': metrics['interference']['p10_snr_margin'],
                    'pct_near_threshold': metrics['interference']['pct_near_threshold'],
                    'pct_collision_likely': metrics['interference']['pct_collision_likely'],
                    'pct_weak_signal_likely': metrics['interference']['pct_weak_signal_likely'],
                    
                    # Stability
                    'sf_change_rate': metrics['stability']['sf_change_rate'],
                    'tp_change_rate': metrics['stability']['tp_change_rate'],
                    'avg_sf': metrics['stability']['avg_sf'],
                    'std_sf': metrics['stability']['std_sf'],
                    'avg_tp': metrics['stability']['avg_tp'],
                    'std_tp': metrics['stability']['std_tp'],
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        output_file = self.output_dir / 'extended_metrics_all_devices.csv'
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # Summary by ADR type
        summary_rows = []
        for adr_type in ADR_ORDER:
            if adr_type not in self.all_metrics or not self.all_metrics[adr_type]:
                continue
            
            devices = self.all_metrics[adr_type]
            n = len(devices)
            
            # Compute aggregates
            pdrs = [d['reliability']['pdr'] for d in devices.values()]
            eps = [d['energy']['energy_per_success_mJ'] for d in devices.values() if d['energy']['energy_per_success_mJ'] != float('inf')]
            
            # Fairness
            device_stats = {
                did: {'pdr': d['reliability']['pdr'], 'energy_per_success_mJ': d['energy']['energy_per_success_mJ']}
                for did, d in devices.items()
            }
            fairness = compute_fairness_metrics(device_stats)
            
            summary_rows.append({
                'adr_type': adr_type,
                'n_devices': n,
                'pdr_mean': np.mean(pdrs),
                'pdr_std': np.std(pdrs),
                'pdr_min': np.min(pdrs),
                'pdr_max': np.max(pdrs),
                'pdr_p5': np.percentile(pdrs, 5),
                'energy_per_success_mean': np.mean(eps) if eps else np.nan,
                'energy_per_success_std': np.std(eps) if eps else np.nan,
                'jains_fairness_pdr': fairness.jains_fairness_pdr,
                'worst_device_id': fairness.worst_device_id,
                'worst_device_pdr': fairness.min_pdr,
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_file = self.output_dir / 'extended_metrics_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"  Saved: {summary_file}")
    
    def run_full_analysis(self):
        """Run the complete extended metrics analysis."""
        print("=" * 60)
        print("EXTENDED LORAWAN ADR METRICS ANALYSIS")
        print("=" * 60)
        print(f"Datasets folder: {self.datasets_folder}")
        print(f"Output directory: {self.output_dir}")
        
        self.load_datasets()
        
        if not any(self.device_data.values()):
            print("\nNo datasets found. Exiting.")
            return
        
        self.compute_all_metrics()
        self.print_summary_report()
        self.print_statistical_comparisons()
        self.create_visualizations()
        self.export_to_csv()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {self.output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extended LoRaWAN ADR Metrics Analysis')
    parser.add_argument('--datasets-folder', type=str, default='./lorawan_datasets',
                        help='Path to folder containing device CSV files')
    parser.add_argument('--output-dir', type=str, default='./extended_metrics',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Check if datasets folder exists
    if not os.path.exists(args.datasets_folder):
        print(f"Error: Datasets folder not found: {args.datasets_folder}")
        print("Please specify the correct path with --datasets-folder")
        return
    
    analyzer = ExtendedMetricsAnalyzer(args.datasets_folder, args.output_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
