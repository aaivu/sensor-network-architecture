#!/usr/bin/env python3
"""
LoRaWAN ADR Comparison Analysis Tool (Extended)
================================================
Analyzes DDQN-PER ADR vs Classical ADR vs No ADR performance
Device-by-device insights and comprehensive statistics

Extended Metrics:
- Reliability: rolling PDR, outage analysis, consecutive-loss CDF
- Energy efficiency: energy per successful packet, goodput, battery life
- Airtime: Time-on-Air, channel occupancy
- Interference robustness: loss rate vs RSSI, SNR margin
- Stability: SF/TP churn, parameter distributions
- Fairness: Jain's index, worst-device analysis
- Enhanced statistics: Mann-Whitney, Cliff's delta, bootstrap CI
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import hashlib
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import re

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS FOR EXTENDED METRICS
# ============================================================================

# SNR thresholds for each SF (dB) - minimum required SNR for successful demodulation
SNR_THRESHOLDS = {
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0
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


# ============================================================================
# TIME-ON-AIR CALCULATION
# ============================================================================

def calculate_time_on_air_ms(sf: int, payload_bytes: int = PAYLOAD_SIZE_BYTES,
                              bw_khz: int = 125, cr: int = 1) -> float:
    """Calculate LoRa Time-on-Air in milliseconds (Semtech SX1276 formula)."""
    t_sym = (2 ** sf) / (bw_khz * 1000) * 1000  # ms
    t_preamble = (PREAMBLE_SYMBOLS + 4.25) * t_sym
    de = 1 if sf >= 11 else 0
    h = 0  # explicit header
    crc = 1
    pl = payload_bytes
    numerator = 8 * pl - 4 * sf + 28 + 16 * crc - 20 * h
    denominator = 4 * (sf - 2 * de)
    if denominator <= 0:
        denominator = 4 * sf
    n_payload = 8 + max(np.ceil(numerator / denominator) * (cr + 4), 0)
    t_payload = n_payload * t_sym
    return t_preamble + t_payload


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
# EXTENDED METRICS DATACLASSES
# ============================================================================

@dataclass
class ReliabilityMetrics:
    """Container for reliability metrics."""
    total_packets: int
    successful_packets: int
    pdr: float
    rolling_pdr_mean: float
    rolling_pdr_std: float
    rolling_pdr_min: float
    outage_count: int
    avg_outage_length: float
    max_outage_length: int
    total_outage_packets: int
    consecutive_loss_lengths: List[int]
    pdr_coefficient_of_variation: float


@dataclass
class EnergyMetrics:
    """Container for energy efficiency metrics."""
    total_energy_mJ: float
    energy_per_packet_mJ: float
    energy_per_success_mJ: float
    goodput_bytes_per_second: float
    energy_efficiency_bytes_per_mJ: float
    projected_battery_life_days: float
    transmissions_per_day: float


@dataclass
class AirtimeMetrics:
    """Container for airtime/channel occupancy metrics."""
    total_airtime_ms: float
    avg_toa_ms: float
    toa_per_success_ms: float
    airtime_per_minute_ms: float
    channel_utilization_percent: float


@dataclass
class InterferenceMetrics:
    """Container for interference robustness metrics."""
    loss_rate_by_rssi: Dict[str, float]
    median_snr_margin: float
    p10_snr_margin: float
    pct_near_threshold: float
    pct_collision_likely: float
    pct_weak_signal_likely: float


@dataclass
class StabilityMetrics:
    """Container for algorithm stability metrics."""
    sf_change_rate: float
    tp_change_rate: float
    sf_distribution: Dict[int, float]
    tp_distribution: Dict[str, float]
    avg_sf: float
    std_sf: float
    avg_tp: float
    std_tp: float


@dataclass
class FairnessMetrics:
    """Container for fairness metrics across devices."""
    jains_fairness_pdr: float
    jains_fairness_energy_efficiency: float
    min_pdr: float
    p5_pdr: float
    max_pdr: float
    worst_device_id: str
    best_device_id: str


# ============================================================================
# EXTENDED METRICS COMPUTATION FUNCTIONS
# ============================================================================

def compute_consecutive_runs(packet_received: np.ndarray) -> Tuple[List[int], List[int]]:
    """Compute consecutive runs of successes and failures."""
    if len(packet_received) == 0:
        return [], []
    success_runs, failure_runs = [], []
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
    if current_value == 1:
        success_runs.append(current_length)
    else:
        failure_runs.append(current_length)
    return success_runs, failure_runs


def compute_reliability_metrics(df: pd.DataFrame, window_size: int = 20,
                                 outage_threshold: int = 3) -> ReliabilityMetrics:
    """Compute comprehensive reliability metrics for a device."""
    packet_received = df['packet_received'].values
    total = len(packet_received)
    successes = int(packet_received.sum())
    pdr = 100.0 * successes / total if total > 0 else 0.0
    rolling_pdr = pd.Series(packet_received).rolling(window=window_size, min_periods=1).mean() * 100
    rolling_mean = rolling_pdr.mean()
    rolling_std = rolling_pdr.std()
    rolling_min = rolling_pdr.min()
    cv = rolling_std / rolling_mean if rolling_mean > 0 else float('inf')
    _, failure_runs = compute_consecutive_runs(packet_received)
    outages = [r for r in failure_runs if r >= outage_threshold]
    outage_count = len(outages)
    avg_outage_length = np.mean(outages) if outages else 0.0
    max_outage_length = max(outages) if outages else 0
    total_outage_packets = sum(outages)
    return ReliabilityMetrics(
        total_packets=total, successful_packets=successes, pdr=pdr,
        rolling_pdr_mean=rolling_mean, rolling_pdr_std=rolling_std, rolling_pdr_min=rolling_min,
        outage_count=outage_count, avg_outage_length=avg_outage_length,
        max_outage_length=max_outage_length, total_outage_packets=total_outage_packets,
        consecutive_loss_lengths=failure_runs, pdr_coefficient_of_variation=cv
    )


def compute_energy_metrics(df: pd.DataFrame, payload_bytes: int = PAYLOAD_SIZE_BYTES,
                           battery_capacity_mAh: float = 2400,
                           battery_voltage: float = 3.0) -> EnergyMetrics:
    """Compute energy efficiency metrics."""
    total_energy = df['energy_consumed_mJ'].sum()
    total_packets = len(df)
    successful_packets = df['packet_received'].sum()
    energy_per_packet = total_energy / total_packets if total_packets > 0 else 0.0
    energy_per_success = total_energy / successful_packets if successful_packets > 0 else float('inf')
    duration_seconds = df['timestamp'].max() - df['timestamp'].min()
    if duration_seconds <= 0:
        duration_seconds = 1.0
    goodput_bps = (successful_packets * payload_bytes) / duration_seconds
    energy_efficiency = (successful_packets * payload_bytes) / total_energy if total_energy > 0 else 0.0
    battery_capacity_mJ = battery_capacity_mAh * battery_voltage * 3.6 * 1000
    energy_per_day = (total_energy / duration_seconds) * 86400
    projected_battery_life_days = battery_capacity_mJ / energy_per_day if energy_per_day > 0 else float('inf')
    transmissions_per_day = (total_packets / duration_seconds) * 86400
    return EnergyMetrics(
        total_energy_mJ=total_energy, energy_per_packet_mJ=energy_per_packet,
        energy_per_success_mJ=energy_per_success, goodput_bytes_per_second=goodput_bps,
        energy_efficiency_bytes_per_mJ=energy_efficiency,
        projected_battery_life_days=projected_battery_life_days,
        transmissions_per_day=transmissions_per_day
    )


def compute_airtime_metrics(df: pd.DataFrame) -> AirtimeMetrics:
    """Compute airtime and channel occupancy metrics."""
    if 'toa_ms' not in df.columns:
        df = add_toa_column(df)
    total_airtime = df['toa_ms'].sum()
    avg_toa = df['toa_ms'].mean()
    successful_packets = df['packet_received'].sum()
    toa_per_success = total_airtime / successful_packets if successful_packets > 0 else float('inf')
    duration_seconds = df['timestamp'].max() - df['timestamp'].min()
    if duration_seconds <= 0:
        duration_seconds = 1.0
    airtime_per_minute = (total_airtime / duration_seconds) * 60
    duty_cycle_limit_ms_per_min = 600
    channel_utilization = (airtime_per_minute / duty_cycle_limit_ms_per_min) * 100
    return AirtimeMetrics(
        total_airtime_ms=total_airtime, avg_toa_ms=avg_toa, toa_per_success_ms=toa_per_success,
        airtime_per_minute_ms=airtime_per_minute, channel_utilization_percent=channel_utilization
    )


def compute_interference_metrics(df: pd.DataFrame,
                                  collision_rssi_threshold: float = -105.0) -> InterferenceMetrics:
    """Compute interference robustness metrics."""
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
    received = df[df['snr_db'] != -999.0].copy()
    if len(received) > 0:
        received['snr_threshold'] = received['sf'].map(SNR_THRESHOLDS)
        received['snr_margin'] = received['snr_db'] - received['snr_threshold']
        median_margin = received['snr_margin'].median()
        p10_margin = received['snr_margin'].quantile(0.10)
        pct_near_threshold = 100.0 * (received['snr_margin'] < 3.0).mean()
    else:
        median_margin, p10_margin, pct_near_threshold = np.nan, np.nan, np.nan
    lost = df[df['packet_received'] == 0]
    if len(lost) > 0:
        pct_collision = 100.0 * len(lost[lost['pre_tx_rssi_dBm'] > collision_rssi_threshold]) / len(lost)
        pct_weak = 100.0 * len(lost[lost['pre_tx_rssi_dBm'] <= collision_rssi_threshold]) / len(lost)
    else:
        pct_collision, pct_weak = 0.0, 0.0
    return InterferenceMetrics(
        loss_rate_by_rssi=loss_rate_by_rssi, median_snr_margin=median_margin,
        p10_snr_margin=p10_margin, pct_near_threshold=pct_near_threshold,
        pct_collision_likely=pct_collision, pct_weak_signal_likely=pct_weak
    )


def compute_stability_metrics(df: pd.DataFrame) -> StabilityMetrics:
    """Compute algorithm stability metrics (parameter churn)."""
    sf_values = df['sf'].values
    tp_values = df['tx_power_dBm'].values
    if len(sf_values) > 1:
        sf_changes = np.sum(sf_values[1:] != sf_values[:-1])
        sf_change_rate = sf_changes / (len(sf_values) - 1)
        tp_changes = np.sum(np.abs(tp_values[1:] - tp_values[:-1]) > 0.5)
        tp_change_rate = tp_changes / (len(tp_values) - 1)
    else:
        sf_change_rate, tp_change_rate = 0.0, 0.0
    sf_series = pd.Series(sf_values)
    sf_counts = sf_series.value_counts()
    sf_total = sf_counts.sum()
    sf_distribution = {int(sf): (count / sf_total * 100) for sf, count in sf_counts.items()}
    tp_bins = pd.cut(tp_values, bins=[0, 4, 7, 10, 13, 16], labels=['2-4', '5-7', '8-10', '11-13', '14+'])
    tp_counts = tp_bins.value_counts()
    tp_total = tp_counts.sum()
    tp_distribution = {str(b): (c / tp_total * 100) if tp_total > 0 else 0 for b, c in tp_counts.items()}
    return StabilityMetrics(
        sf_change_rate=sf_change_rate, tp_change_rate=tp_change_rate,
        sf_distribution=sf_distribution, tp_distribution=tp_distribution,
        avg_sf=np.mean(sf_values), std_sf=np.std(sf_values),
        avg_tp=np.mean(tp_values), std_tp=np.std(tp_values)
    )


def compute_jains_fairness(values: np.ndarray) -> float:
    """Compute Jain's Fairness Index: (sum(x))^2 / (n * sum(x^2))"""
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    sum_x = np.sum(values)
    sum_x2 = np.sum(values ** 2)
    return (sum_x ** 2) / (n * sum_x2)


def compute_fairness_metrics(device_stats: Dict[str, Dict]) -> FairnessMetrics:
    """Compute fairness metrics across devices for a given ADR type."""
    if not device_stats:
        return FairnessMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 'N/A', 'N/A')
    pdrs = np.array([s['pdr'] for s in device_stats.values()])
    device_ids = list(device_stats.keys())
    energy_efficiencies = []
    for s in device_stats.values():
        eps = s.get('energy_per_success_mJ', float('inf'))
        if eps > 0 and eps != float('inf'):
            energy_efficiencies.append(1.0 / eps)
        else:
            energy_efficiencies.append(0.0)
    energy_efficiencies = np.array(energy_efficiencies)
    jains_pdr = compute_jains_fairness(pdrs)
    jains_energy = compute_jains_fairness(energy_efficiencies)
    min_pdr, max_pdr = np.min(pdrs), np.max(pdrs)
    p5_pdr = np.percentile(pdrs, 5)
    worst_idx, best_idx = np.argmin(pdrs), np.argmax(pdrs)
    return FairnessMetrics(
        jains_fairness_pdr=jains_pdr, jains_fairness_energy_efficiency=jains_energy,
        min_pdr=min_pdr, p5_pdr=p5_pdr, max_pdr=max_pdr,
        worst_device_id=device_ids[worst_idx], best_device_id=device_ids[best_idx]
    )


def compute_cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
    """Compute Cliff's Delta effect size (non-parametric)."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0, 'negligible'
    count = 0
    for x1 in group1:
        for x2 in group2:
            if x1 > x2:
                count += 1
            elif x1 < x2:
                count -= 1
    delta = count / (n1 * n2)
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
    """Compute bootstrap confidence interval."""
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    point_estimate = stat_func(data)
    bootstrap_stats = [stat_func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return point_estimate, ci_lower, ci_upper

class LoRaWANADRAnalyzer:
    def __init__(self, datasets_folder, output_root="graphs"):
        self.datasets_folder = Path(datasets_folder)
        self.output_root = Path(output_root)
        self.run_hash = self._generate_run_hash()
        self.output_dir = self.output_root / self.run_hash
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_data = {}
        self.summary_stats = {}
        self.extended_metrics = {}  # Store extended metrics per ADR type and device

    def _generate_run_hash(self):
        """Generate a short, unique hash for this analysis run."""
        seed = f"{self.datasets_folder.resolve()}|{datetime.utcnow().isoformat(timespec='seconds')}"
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
        
    def load_datasets(self):
        """Load all CSV datasets for analysis"""
        print("🔍 Loading LoRaWAN datasets...")
        
        # Find all device datasets (including PPO, MARL, and HYBRID)
        patterns = {
            'no_adr': 'no_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv',
            'classical_adr': 'adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv',
            'ddqn_adr': 'ddqn_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv',
            'ppo_adr': 'ppo_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv',
            'marl_adr': 'marl_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv',
            'hybrid_adr': 'hybrid_adr_advanced_pre_tx_rssi_dataset_device_*_dataset.csv'
        }
        
        for adr_type, pattern in patterns.items():
            files = glob.glob(str(self.datasets_folder / pattern))
            self.device_data[adr_type] = {}
            
            for file_path in files:
                # Extract device ID from filename using regex
                filename = os.path.basename(file_path)
                match = re.search(r'device_(\d+)_dataset', filename)
                if match:
                    device_id = match.group(1)
                else:
                    device_id = filename.split('device')[1].split('_dataset')[0]
                
                try:
                    df = pd.read_csv(file_path)
                    # Add ToA column for extended metrics
                    df = add_toa_column(df)
                    self.device_data[adr_type][device_id] = df
                    print(f"  ✅ Loaded {adr_type} data for Device {device_id}: {len(df)} records")
                    # Debug: print columns for first file to verify
                    if len(self.device_data[adr_type]) == 1:
                        print(f"    Debug - Columns: {df.columns.tolist()}")
                except Exception as e:
                    print(f"  ❌ Error loading {file_path}: {e}")
        
        print(f"\n📊 Datasets loaded successfully!")
        
    def analyze_device_performance(self, device_id, adr_type):
        """Analyze performance metrics for a specific device and ADR type"""
        if adr_type not in self.device_data or device_id not in self.device_data[adr_type]:
            return None
            
        df = self.device_data[adr_type][device_id]
        
        # Debug: Check if snr_db column exists
        if 'snr_db' not in df.columns:
            print(f"❌ DEBUG: 'snr_db' column missing for {adr_type} device {device_id}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Filter out invalid SNR values (-999 indicates failed reception)
        successful_packets = df[df['snr_db'] != -999.0]
        failed_packets = df[df['snr_db'] == -999.0]
        
        # Basic stats (original)
        stats = {
            'device_id': device_id,
            'adr_type': adr_type,
            'total_packets': len(df),
            'successful_packets': len(successful_packets),
            'failed_packets': len(failed_packets),
            'pdr': len(successful_packets) / len(df) * 100 if len(df) > 0 else 0,
            'avg_snr': successful_packets['snr_db'].mean() if len(successful_packets) > 0 else 0,
            'std_snr': successful_packets['snr_db'].std() if len(successful_packets) > 0 else 0,
            'min_snr': successful_packets['snr_db'].min() if len(successful_packets) > 0 else 0,
            'max_snr': successful_packets['snr_db'].max() if len(successful_packets) > 0 else 0,
            'avg_tx_power': df['tx_power_dBm'].mean(),
            'std_tx_power': df['tx_power_dBm'].std(),
            'avg_sf': df['sf'].mean(),
            'std_sf': df['sf'].std(),
            'avg_rssi': df['pre_tx_rssi_dBm'].mean(),
            'distance': df['distance_m'].iloc[0] if len(df) > 0 else 0,
            'position_x': df['tx_x'].iloc[0] if len(df) > 0 else 0,
            'position_y': df['tx_y'].iloc[0] if len(df) > 0 else 0,
            'total_energy': df['energy_consumed_mJ'].sum(),
            'energy_per_packet': df['energy_consumed_mJ'].mean()
        }
        
        # Compute extended metrics
        try:
            reliability = compute_reliability_metrics(df)
            energy = compute_energy_metrics(df)
            airtime = compute_airtime_metrics(df)
            interference = compute_interference_metrics(df)
            stability = compute_stability_metrics(df)
            
            # Add extended metrics to stats
            stats.update({
                # Reliability
                'rolling_pdr_mean': reliability.rolling_pdr_mean,
                'rolling_pdr_std': reliability.rolling_pdr_std,
                'rolling_pdr_min': reliability.rolling_pdr_min,
                'outage_count': reliability.outage_count,
                'avg_outage_length': reliability.avg_outage_length,
                'max_outage_length': reliability.max_outage_length,
                'pdr_cv': reliability.pdr_coefficient_of_variation,
                'consecutive_loss_lengths': reliability.consecutive_loss_lengths,
                
                # Energy
                'energy_per_success_mJ': energy.energy_per_success_mJ,
                'goodput_bytes_per_s': energy.goodput_bytes_per_second,
                'energy_efficiency_bytes_per_mJ': energy.energy_efficiency_bytes_per_mJ,
                'projected_battery_life_days': energy.projected_battery_life_days,
                
                # Airtime
                'total_airtime_ms': airtime.total_airtime_ms,
                'avg_toa_ms': airtime.avg_toa_ms,
                'toa_per_success_ms': airtime.toa_per_success_ms,
                'channel_utilization_pct': airtime.channel_utilization_percent,
                
                # Interference
                'median_snr_margin': interference.median_snr_margin,
                'p10_snr_margin': interference.p10_snr_margin,
                'pct_near_threshold': interference.pct_near_threshold,
                'pct_collision_likely': interference.pct_collision_likely,
                'pct_weak_signal_likely': interference.pct_weak_signal_likely,
                
                # Stability
                'sf_change_rate': stability.sf_change_rate,
                'tp_change_rate': stability.tp_change_rate,
                'sf_distribution': stability.sf_distribution,
            })
        except Exception as e:
            print(f"⚠ Extended metrics computation failed for {adr_type} device {device_id}: {e}")
        
        return stats
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis for all devices and ADR types"""
        print("\n🔬 Generating comprehensive analysis...")
        
        all_stats = []
        device_comparison = {}
        
        # Get all unique device IDs
        all_devices = set()
        for adr_type in self.device_data:
            all_devices.update(self.device_data[adr_type].keys())
        
        all_devices = sorted(all_devices)
        
        # Analyze each device for each ADR type
        for device_id in all_devices:
            device_comparison[device_id] = {}
            for adr_type in ['no_adr', 'classical_adr', 'ddqn_adr', 'ppo_adr', 'marl_adr', 'hybrid_adr']:
                stats = self.analyze_device_performance(device_id, adr_type)
                if stats:
                    all_stats.append(stats)
                    device_comparison[device_id][adr_type] = stats
        
        return all_stats, device_comparison
    
    def print_executive_summary(self, all_stats):
        """Print executive summary of the analysis"""
        df_stats = pd.DataFrame(all_stats)
        
        print("\n" + "="*80)
        print("🚀 LORAWAN ADR COMPARISON - EXECUTIVE SUMMARY")
        print("="*80)
        
        # Overall performance by ADR type
        adr_summary = df_stats.groupby('adr_type').agg({
            'pdr': ['mean', 'std', 'count'],
            'avg_snr': 'mean',
            'avg_tx_power': 'mean',
            'avg_sf': 'mean',
            'total_energy': 'mean'
        }).round(3)
        
        print("\n📊 OVERALL PERFORMANCE BY ADR TYPE:")
        print("-" * 50)
        
        for adr_type in ['no_adr', 'classical_adr', 'ddqn_adr', 'ppo_adr', 'marl_adr', 'hybrid_adr']:
            if adr_type in adr_summary.index:
                row = adr_summary.loc[adr_type]
                adr_name = ADR_NAMES.get(adr_type, adr_type.replace('_', ' ').title())
                
                print(f"\n{adr_name}:")
                print(f"  📈 Packet Delivery Rate: {row[('pdr', 'mean')]:.1f}% (±{row[('pdr', 'std')]:.1f}%)")
                print(f"  📡 Average SNR: {row[('avg_snr', 'mean')]:.1f} dB")
                print(f"  ⚡ Average TX Power: {row[('avg_tx_power', 'mean')]:.1f} dBm")
                print(f"  📶 Average Spreading Factor: {row[('avg_sf', 'mean')]:.1f}")
                print(f"  🔋 Average Energy per Device: {row[('total_energy', 'mean')]:.3f} mJ")
                print(f"  🏠 Devices Analyzed: {int(row[('pdr', 'count')])}")
                
                # Extended metrics summary
                adr_data = df_stats[df_stats['adr_type'] == adr_type]
                if 'energy_per_success_mJ' in adr_data.columns:
                    eps = adr_data['energy_per_success_mJ'].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(eps) > 0:
                        print(f"  ⚡ Energy per Success: {eps.mean():.2f} mJ")
                if 'sf_change_rate' in adr_data.columns:
                    print(f"  🔄 SF Change Rate: {adr_data['sf_change_rate'].mean() * 100:.1f}%")
                if 'outage_count' in adr_data.columns:
                    print(f"  🚫 Avg Outages per Device: {adr_data['outage_count'].mean():.1f}")
        
        # Best performing ADR type
        avg_pdr_by_type = df_stats.groupby('adr_type')['pdr'].mean()
        best_adr = avg_pdr_by_type.idxmax()
        improvement = avg_pdr_by_type[best_adr] - avg_pdr_by_type.drop(best_adr).max()
        
        print(f"\n🏆 BEST PERFORMING ADR: {best_adr.replace('_', ' ').title()}")
        print(f"   Improvement: +{improvement:.1f} percentage points")
        
        # Fairness summary
        print("\n📊 FAIRNESS ANALYSIS:")
        print("-" * 30)
        for adr_type in ADR_ORDER:
            adr_data = df_stats[df_stats['adr_type'] == adr_type]
            if len(adr_data) > 0:
                device_stats = {row['device_id']: {'pdr': row['pdr'], 
                                'energy_per_success_mJ': row.get('energy_per_success_mJ', float('inf'))} 
                               for _, row in adr_data.iterrows()}
                fairness = compute_fairness_metrics(device_stats)
                print(f"  {ADR_NAMES.get(adr_type, adr_type)}: Jain's Index = {fairness.jains_fairness_pdr:.4f}, "
                      f"5th %ile PDR = {fairness.p5_pdr:.1f}%")
        
    def print_device_by_device_analysis(self, device_comparison):
        """Print detailed device-by-device analysis"""
        print("\n" + "="*80)
        print("🔍 DEVICE-BY-DEVICE DETAILED ANALYSIS")
        print("="*80)
        
        for device_id in sorted(device_comparison.keys()):
            device_data = device_comparison[device_id]
            
            print(f"\n📱 DEVICE {device_id}")
            print("-" * 40)
            
            # Device position and distance
            if 'no_adr' in device_data:
                stats = device_data['no_adr']
                print(f"📍 Position: ({stats['position_x']:.1f}, {stats['position_y']:.1f}) m")
                print(f"📏 Distance from Gateway: {stats['distance']:.1f} m")
            
            print(f"\n{'ADR Method':<15} {'PDR (%)':<8} {'Avg SNR':<10} {'TX Power':<10} {'SF':<5} {'Energy (mJ)':<12}")
            print("-" * 70)
            
            for adr_type in ADR_ORDER:
                if adr_type in device_data:
                    stats = device_data[adr_type]
                    adr_name = ADR_NAMES.get(adr_type, adr_type.replace('_', ' ').title())
                    
                    print(f"{adr_name:<15} {stats['pdr']:<8.1f} {stats['avg_snr']:<10.1f} "
                          f"{stats['avg_tx_power']:<10.1f} {stats['avg_sf']:<5.1f} {stats['total_energy']:<12.3f}")
            
            # Extended metrics table
            print(f"\n{'ADR Method':<15} {'E/Success':<10} {'SF Churn':<10} {'Outages':<10} {'Battery(d)':<12}")
            print("-" * 70)
            
            for adr_type in ADR_ORDER:
                if adr_type in device_data:
                    stats = device_data[adr_type]
                    adr_name = ADR_NAMES.get(adr_type, adr_type.replace('_', ' ').title())
                    
                    eps = stats.get('energy_per_success_mJ', float('inf'))
                    eps_str = f"{eps:.1f}" if eps != float('inf') else "∞"
                    sf_churn = stats.get('sf_change_rate', 0) * 100
                    outages = stats.get('outage_count', 0)
                    battery = stats.get('projected_battery_life_days', float('inf'))
                    battery_str = f"{battery:.0f}" if battery != float('inf') else "∞"
                    
                    print(f"{adr_name:<15} {eps_str:<10} {sf_churn:<10.1f}% {outages:<10} {battery_str:<12}")
            
            # Analysis and insights for this device
            if len(device_data) >= 2:
                pdrs = {adr: data['pdr'] for adr, data in device_data.items()}
                best_adr = max(pdrs.keys(), key=lambda x: pdrs[x])
                worst_adr = min(pdrs.keys(), key=lambda x: pdrs[x])
                
                improvement = pdrs[best_adr] - pdrs[worst_adr]
                
                print(f"\n  💡 Insights:")
                print(f"     🏆 Best: {best_adr.replace('_', ' ').title()} ({pdrs[best_adr]:.1f}% PDR)")
                print(f"     📉 Worst: {worst_adr.replace('_', ' ').title()} ({pdrs[worst_adr]:.1f}% PDR)")
                print(f"     📈 Improvement: {improvement:.1f} percentage points")
                
                # Distance-based insights
                distance = device_data[list(device_data.keys())[0]]['distance']
                if distance > 800:
                    print(f"     📡 Long-range device (>{800}m) - ADR impact more significant")
                elif distance < 300:
                    print(f"     📡 Short-range device (<{300}m) - Good signal conditions")
                else:
                    print(f"     📡 Medium-range device - Moderate signal conditions")
    
    def create_performance_visualizations(self, all_stats, device_comparison):
        """Create publication-quality visualizations saved as EPS"""
        print("\n📊 Generating publication-quality visualizations (EPS)...")
        
        df_stats = pd.DataFrame(all_stats)
        
        # Use a clean style suitable for papers
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'text.usetex': False,
        })
        
        adr_mapping = {
            'no_adr': 'No ADR',
            'classical_adr': 'Classical\nADR',
            'ddqn_adr': 'DDQN-PER\nADR',
            'ppo_adr': 'PPO\nADR',
            'marl_adr': 'MARL\nADR',
            'hybrid_adr': 'Hybrid\nADR'
        }
        adr_order = ['no_adr', 'classical_adr', 'ddqn_adr', 'ppo_adr', 'marl_adr', 'hybrid_adr']
        # Only keep ADR types that exist in data
        present_types = [t for t in adr_order if t in df_stats['adr_type'].unique()]
        present_labels = [adr_mapping[t] for t in present_types]
        
        # Professional uniform colour
        BAR_FILL  = '#A8C8E8'   # Soft steel blue
        BAR_EDGE  = '#2B5D8A'   # Darker blue edge
        ERR_COLOR = '#2B5D8A'   # Match edge for error bars
        
        # Color palette for line plots
        LINE_COLORS = {
            'no_adr': '#e74c3c',
            'classical_adr': '#3498db',
            'ddqn_adr': '#2ecc71',
            'ppo_adr': '#9b59b6',
            'marl_adr': '#f39c12',
            'hybrid_adr': '#1abc9c'
        }
        
        # =====================================================================
        # FIGURE 0: PDR Learning Curve / Convergence Over Time
        # =====================================================================
        self._plot_pdr_convergence(present_types, LINE_COLORS, adr_mapping)
        
        # =====================================================================
        # FIGURE 1: Packet Delivery Rate (PDR) by ADR Type
        # =====================================================================
        fig1, ax1 = plt.subplots(figsize=(6, 4.2))
        
        pdr_means = [df_stats[df_stats['adr_type'] == t]['pdr'].mean() for t in present_types]
        pdr_stds  = [df_stats[df_stats['adr_type'] == t]['pdr'].std()  for t in present_types]
        
        bars1 = ax1.bar(present_labels, pdr_means, yerr=pdr_stds,
                        color=BAR_FILL, edgecolor=BAR_EDGE, linewidth=0.8,
                        capsize=4, error_kw={'linewidth': 0.8, 'color': ERR_COLOR})

        # PDR is a percentage; keep the y-axis capped at 100%
        pdr_ylim = 100.0
        ax1.set_ylim(0, pdr_ylim)
        
        # Annotate each bar — place label inside bar if it would exceed the plot
        for bar, mean, std in zip(bars1, pdr_means, pdr_stds):
            top = bar.get_height() + std
            label_y = top + pdr_ylim * 0.015          # small gap above error bar
            va = 'bottom'
            color = 'black'
            if label_y + pdr_ylim * 0.04 > pdr_ylim:  # would overflow
                label_y = bar.get_height() - pdr_ylim * 0.02
                va = 'top'
                color = BAR_EDGE
            ax1.text(bar.get_x() + bar.get_width() / 2.0, label_y,
                     f'{mean:.1f}%', ha='center', va=va, fontsize=9,
                     fontweight='bold', color=color)
        
        ax1.set_ylabel('Packet Delivery Rate (%)')
        ax1.set_xlabel('ADR Method')
        ax1.set_title('Packet Delivery Rate (PDR) by ADR Type')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        fig1.tight_layout()
        pdr_eps = self.output_dir / 'pdr_by_adr_type.eps'
        pdr_png = self.output_dir / 'pdr_by_adr_type.png'
        fig1.savefig(pdr_eps, format='eps', bbox_inches='tight')
        fig1.savefig(pdr_png, format='png', dpi=300, bbox_inches='tight')
        print(f"  ✅ PDR chart saved as: {pdr_eps} / {pdr_png}")
        plt.close(fig1)
        
        # =====================================================================
        # FIGURE 2: Energy Consumption by ADR Type
        # =====================================================================
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        
        energy_means = [df_stats[df_stats['adr_type'] == t]['total_energy'].mean() for t in present_types]
        energy_stds  = [df_stats[df_stats['adr_type'] == t]['total_energy'].std()  for t in present_types]
        
        bars2 = ax2.bar(present_labels, energy_means, yerr=energy_stds,
                        color=BAR_FILL, edgecolor=BAR_EDGE, linewidth=0.8,
                        capsize=4, error_kw={'linewidth': 0.8, 'color': ERR_COLOR})
        
        # Safe y-limit
        energy_tops = [m + s for m, s in zip(energy_means, energy_stds)]
        energy_ylim = max(energy_tops) * 1.18
        ax2.set_ylim(0, energy_ylim)
        
        # Annotate — energy values can be large, use compact formatting
        for bar, mean, std in zip(bars2, energy_means, energy_stds):
            top = bar.get_height() + std
            # Format large numbers compactly: e.g. 281.3 k
            if mean >= 1000:
                label = f'{mean/1000:.1f}k'
            else:
                label = f'{mean:.1f}'
            label_y = top + energy_ylim * 0.015
            va = 'bottom'
            color = 'black'
            if label_y + energy_ylim * 0.04 > energy_ylim:
                label_y = bar.get_height() - energy_ylim * 0.02
                va = 'top'
                color = BAR_EDGE
            ax2.text(bar.get_x() + bar.get_width() / 2.0, label_y,
                     label, ha='center', va=va, fontsize=9,
                     fontweight='bold', color=color)
        
        ax2.set_ylabel('Total Energy Consumed (mJ)')
        ax2.set_xlabel('ADR Method')
        ax2.set_title('Energy Consumption by ADR Type')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        fig2.tight_layout()
        energy_eps = self.output_dir / 'energy_by_adr_type.eps'
        energy_png = self.output_dir / 'energy_by_adr_type.png'
        fig2.savefig(energy_eps, format='eps', bbox_inches='tight')
        fig2.savefig(energy_png, format='png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Energy chart saved as: {energy_eps} / {energy_png}")
        plt.close(fig2)
        
        # =====================================================================
        # FIGURE 3: Device Placement Map
        # =====================================================================
        fig3, ax3 = plt.subplots(figsize=(6.5, 6.5))
        
        # Collect unique device positions (use first available ADR type per device)
        device_positions = []
        for device_id in sorted(device_comparison.keys()):
            dd = device_comparison[device_id]
            first_key = list(dd.keys())[0]
            stats = dd[first_key]
            device_positions.append({
                'device_id': device_id,
                'x': stats['position_x'],
                'y': stats['position_y'],
                'distance': stats['distance'],
            })
        
        df_pos = pd.DataFrame(device_positions)
        
        # Gateway at origin
        ax3.scatter(0, 0, marker='^', s=220, c='#2B5D8A', edgecolors='black',
                    linewidths=1.2, zorder=5, label='Gateway')
        ax3.annotate('GW', (0, 0), textcoords='offset points', xytext=(8, 8),
                     fontsize=9, fontweight='bold', color='#2B5D8A')
        
        # End devices coloured by distance (single blue gradient)
        sc = ax3.scatter(df_pos['x'], df_pos['y'], c=df_pos['distance'],
                         cmap='Blues', s=80, edgecolors='#2B5D8A', linewidths=0.6,
                         zorder=4, label='End Device')
        
        # Label each device
        for _, row in df_pos.iterrows():
            ax3.annotate(f"D{row['device_id']}",
                         (row['x'], row['y']),
                         textcoords='offset points', xytext=(5, 5),
                         fontsize=7, color='#333333')
        
        # Draw distance rings
        max_dist = df_pos['distance'].max()
        ring_distances = [d for d in [250, 500, 750, 1000, 1250, 1500] if d <= max_dist * 1.2]
        for rd in ring_distances:
            circle = plt.Circle((0, 0), rd, fill=False, linestyle='--',
                                linewidth=0.5, color='gray', alpha=0.5)
            ax3.add_patch(circle)
            ax3.text(rd * 0.707, rd * 0.707, f'{rd}m',
                     fontsize=7, color='gray', alpha=0.7)
        
        cbar = fig3.colorbar(sc, ax=ax3, shrink=0.8, pad=0.02)
        cbar.set_label('Distance from Gateway (m)', fontsize=10)
        
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Y Position (m)')
        ax3.set_title('LoRaWAN Network Device Placement')
        ax3.set_aspect('equal')
        ax3.legend(loc='upper left', framealpha=0.9)
        ax3.grid(True, alpha=0.2, linestyle='-')
        
        # Auto-scale with some padding
        pad = max_dist * 0.15
        lim = max(abs(df_pos['x']).max(), abs(df_pos['y']).max()) + pad
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        
        fig3.tight_layout()
        placement_eps = self.output_dir / 'device_placement.eps'
        placement_png = self.output_dir / 'device_placement.png'
        fig3.savefig(placement_eps, format='eps', bbox_inches='tight')
        fig3.savefig(placement_png, format='png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Device placement saved as: {placement_eps} / {placement_png}")
        plt.close(fig3)
        
        # =====================================================================
        # FIGURE 4: Energy per Successful Packet (Extended Metric)
        # =====================================================================
        if 'energy_per_success_mJ' in df_stats.columns:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            eps_data = []
            labels = []
            for adr_type in present_types:
                eps = df_stats[df_stats['adr_type'] == adr_type]['energy_per_success_mJ']
                eps = eps.replace([np.inf, -np.inf], np.nan).dropna()
                if len(eps) > 0:
                    eps_data.append(eps.values)
                    labels.append(adr_mapping[adr_type])
            
            if eps_data:
                bp = ax4.boxplot(eps_data, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(BAR_FILL)
                    patch.set_edgecolor(BAR_EDGE)
                ax4.set_ylabel('Energy per Successful Packet (mJ)')
                ax4.set_title('Energy Efficiency by ADR Type')
                ax4.grid(axis='y', alpha=0.3, linestyle='--')
                fig4.tight_layout()
                fig4.savefig(self.output_dir / 'energy_per_success.png', dpi=300, bbox_inches='tight')
                fig4.savefig(self.output_dir / 'energy_per_success.eps', format='eps', bbox_inches='tight')
                print(f"  ✅ Energy per success saved")
            plt.close(fig4)
        
        # =====================================================================
        # FIGURE 5: PDR vs Distance (Extended Metric)
        # =====================================================================
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        colors = plt.cm.Set2(np.linspace(0, 1, len(present_types)))
        for idx, adr_type in enumerate(present_types):
            adr_data = df_stats[df_stats['adr_type'] == adr_type]
            ax5.scatter(adr_data['distance'], adr_data['pdr'], alpha=0.7, 
                       label=ADR_NAMES.get(adr_type, adr_type), color=colors[idx], s=40)
        ax5.set_xlabel('Distance from Gateway (m)')
        ax5.set_ylabel('Packet Delivery Rate (%)')
        ax5.set_title('PDR vs Distance by ADR Type')
        ax5.legend(loc='lower left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 105)
        fig5.tight_layout()
        fig5.savefig(self.output_dir / 'pdr_vs_distance.png', dpi=300, bbox_inches='tight')
        fig5.savefig(self.output_dir / 'pdr_vs_distance.eps', format='eps', bbox_inches='tight')
        print(f"  ✅ PDR vs distance saved")
        plt.close(fig5)
        
        # =====================================================================
        # FIGURE 6: Consecutive Loss CDF (Extended Metric)
        # =====================================================================
        if 'consecutive_loss_lengths' in df_stats.columns:
            fig6, ax6 = plt.subplots(figsize=(6, 4))
            for idx, adr_type in enumerate(present_types):
                adr_data = df_stats[df_stats['adr_type'] == adr_type]
                all_losses = []
                for losses in adr_data['consecutive_loss_lengths']:
                    if isinstance(losses, list):
                        all_losses.extend(losses)
                if all_losses:
                    sorted_losses = np.sort(all_losses)
                    cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
                    ax6.plot(sorted_losses, cdf, label=ADR_NAMES.get(adr_type, adr_type),
                            color=colors[idx], linewidth=1.5)
            ax6.set_xlabel('Consecutive Loss Length (packets)')
            ax6.set_ylabel('CDF')
            ax6.set_title('CDF of Consecutive Packet Losses')
            ax6.legend(loc='lower right', fontsize=8)
            ax6.grid(True, alpha=0.3)
            ax6.set_xlim(0, min(50, ax6.get_xlim()[1]))
            fig6.tight_layout()
            fig6.savefig(self.output_dir / 'consecutive_loss_cdf.png', dpi=300, bbox_inches='tight')
            fig6.savefig(self.output_dir / 'consecutive_loss_cdf.eps', format='eps', bbox_inches='tight')
            print(f"  ✅ Consecutive loss CDF saved")
            plt.close(fig6)
        
        # =====================================================================
        # FIGURE 7: SF Distribution (Extended Metric)
        # =====================================================================
        if 'sf_distribution' in df_stats.columns:
            fig7, ax7 = plt.subplots(figsize=(7, 4))
            sf_range = [7, 8, 9, 10, 11, 12]
            x = np.arange(len(sf_range))
            width = 0.12
            for idx, adr_type in enumerate(present_types):
                adr_data = df_stats[df_stats['adr_type'] == adr_type]
                n_devices = len(adr_data)
                sf_avg = {sf: 0 for sf in sf_range}
                for dist in adr_data['sf_distribution']:
                    if isinstance(dist, dict):
                        for sf, pct in dist.items():
                            if sf in sf_avg:
                                sf_avg[sf] += pct / n_devices
                values = [sf_avg[sf] for sf in sf_range]
                offset = (idx - len(present_types) / 2 + 0.5) * width
                ax7.bar(x + offset, values, width, label=ADR_NAMES.get(adr_type, adr_type),
                       color=colors[idx], edgecolor='black', linewidth=0.5)
            ax7.set_xlabel('Spreading Factor')
            ax7.set_ylabel('Usage (%)')
            ax7.set_title('SF Distribution by ADR Type')
            ax7.set_xticks(x)
            ax7.set_xticklabels([f'SF{sf}' for sf in sf_range])
            ax7.legend(loc='upper right', fontsize=8)
            ax7.grid(axis='y', alpha=0.3)
            fig7.tight_layout()
            fig7.savefig(self.output_dir / 'sf_distribution.png', dpi=300, bbox_inches='tight')
            fig7.savefig(self.output_dir / 'sf_distribution.eps', format='eps', bbox_inches='tight')
            print(f"  ✅ SF distribution saved")
            plt.close(fig7)
        
        # =====================================================================
        # FIGURE 8: Jain's Fairness Index (Extended Metric)
        # =====================================================================
        fig8, ax8 = plt.subplots(figsize=(5, 4))
        fairness_values = []
        fairness_labels = []
        for adr_type in present_types:
            adr_data = df_stats[df_stats['adr_type'] == adr_type]
            if len(adr_data) > 0:
                device_stats = {row['device_id']: {'pdr': row['pdr'], 
                                'energy_per_success_mJ': row.get('energy_per_success_mJ', float('inf'))} 
                               for _, row in adr_data.iterrows()}
                fairness = compute_fairness_metrics(device_stats)
                fairness_values.append(fairness.jains_fairness_pdr)
                fairness_labels.append(adr_mapping[adr_type])
        
        if fairness_values:
            bars8 = ax8.bar(fairness_labels, fairness_values, color=BAR_FILL, edgecolor=BAR_EDGE)
            ax8.set_ylabel("Jain's Fairness Index")
            ax8.set_title('PDR Fairness Across Devices')
            ax8.set_ylim(0, 1.05)
            ax8.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax8.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars8, fairness_values):
                ax8.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontsize=9)
            fig8.tight_layout()
            fig8.savefig(self.output_dir / 'jains_fairness.png', dpi=300, bbox_inches='tight')
            fig8.savefig(self.output_dir / 'jains_fairness.eps', format='eps', bbox_inches='tight')
            print(f"  ✅ Jain's fairness saved")
        plt.close(fig8)
        
        # =====================================================================
        # FIGURE 9: Loss Classification (Extended Metric)
        # =====================================================================
        if 'pct_collision_likely' in df_stats.columns:
            fig9, ax9 = plt.subplots(figsize=(6, 4))
            collision_pct = []
            weak_pct = []
            for adr_type in present_types:
                adr_data = df_stats[df_stats['adr_type'] == adr_type]
                collision_pct.append(adr_data['pct_collision_likely'].mean())
                weak_pct.append(adr_data['pct_weak_signal_likely'].mean())
            x = np.arange(len(present_types))
            width = 0.35
            ax9.bar(x - width/2, collision_pct, width, label='Collision-likely', color='#E57373', edgecolor='black')
            ax9.bar(x + width/2, weak_pct, width, label='Weak-signal-likely', color='#64B5F6', edgecolor='black')
            ax9.set_xlabel('ADR Type')
            ax9.set_ylabel('Percentage of Lost Packets (%)')
            ax9.set_title('Loss Classification by ADR Type')
            ax9.set_xticks(x)
            ax9.set_xticklabels(present_labels)
            ax9.legend()
            ax9.grid(axis='y', alpha=0.3)
            fig9.tight_layout()
            fig9.savefig(self.output_dir / 'loss_classification.png', dpi=300, bbox_inches='tight')
            fig9.savefig(self.output_dir / 'loss_classification.eps', format='eps', bbox_inches='tight')
            print(f"  ✅ Loss classification saved")
            plt.close(fig9)
        
        print(f"  📄 All figures saved in EPS format (LaTeX / paper ready): {self.output_dir}")
    
    def _plot_pdr_convergence(self, present_types, line_colors, adr_mapping):
        """
        Plot PDR improvement over time (learning curve).
        Shows how quickly each ADR algorithm learns and converges.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Collect time-series PDR data for all ADR types
        convergence_data = {}
        
        for adr_type in present_types:
            if adr_type not in self.device_data:
                continue
            
            # Aggregate packets across all devices, sorted by timestamp
            all_packets = []
            for device_id, df in self.device_data[adr_type].items():
                df_copy = df.copy()
                df_copy['device_id'] = device_id
                all_packets.append(df_copy)
            
            if not all_packets:
                continue
            
            combined_df = pd.concat(all_packets, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate rolling PDR over time (window = 500 packets network-wide)
            window_size = min(500, len(combined_df) // 10) if len(combined_df) > 1000 else 50
            combined_df['rolling_pdr'] = combined_df['packet_received'].rolling(
                window=window_size, min_periods=1
            ).mean() * 100
            
            # Calculate cumulative PDR over time
            combined_df['cumulative_success'] = combined_df['packet_received'].cumsum()
            combined_df['cumulative_total'] = range(1, len(combined_df) + 1)
            combined_df['cumulative_pdr'] = (combined_df['cumulative_success'] / 
                                              combined_df['cumulative_total']) * 100
            
            # Normalize time to simulation progress (0-100%)
            min_time = combined_df['timestamp'].min()
            max_time = combined_df['timestamp'].max()
            if max_time > min_time:
                combined_df['time_progress'] = ((combined_df['timestamp'] - min_time) / 
                                                 (max_time - min_time)) * 100
            else:
                combined_df['time_progress'] = 0
            
            # Convert timestamp to minutes for readability
            combined_df['time_minutes'] = (combined_df['timestamp'] - min_time) / 60
            
            convergence_data[adr_type] = combined_df
        
        # =====================================================================
        # Left plot: Rolling PDR vs Time (minutes)
        # =====================================================================
        ax1 = axes[0]
        
        for adr_type in present_types:
            if adr_type not in convergence_data:
                continue
            
            df = convergence_data[adr_type]
            color = line_colors.get(adr_type, '#333333')
            label = ADR_NAMES.get(adr_type, adr_type)
            
            # Downsample for cleaner plot (every nth point)
            n_points = min(500, len(df))
            step = max(1, len(df) // n_points)
            df_plot = df.iloc[::step]
            
            ax1.plot(df_plot['time_minutes'], df_plot['rolling_pdr'], 
                    color=color, label=label, linewidth=1.5, alpha=0.85)
        
        ax1.set_xlabel('Simulation Time (minutes)')
        ax1.set_ylabel('Rolling PDR (%)')
        ax1.set_title('PDR Evolution Over Time (Rolling Window)')
        ax1.set_ylim(0, 105)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Add annotation for convergence targets
        ax1.axhline(y=50, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        ax1.text(ax1.get_xlim()[1] * 0.02, 51, '50% threshold', fontsize=8, color='orange', alpha=0.7)
        
        # =====================================================================
        # Right plot: Cumulative PDR vs Simulation Progress (%)
        # =====================================================================
        ax2 = axes[1]
        
        for adr_type in present_types:
            if adr_type not in convergence_data:
                continue
            
            df = convergence_data[adr_type]
            color = line_colors.get(adr_type, '#333333')
            label = ADR_NAMES.get(adr_type, adr_type)
            
            # Downsample
            n_points = min(500, len(df))
            step = max(1, len(df) // n_points)
            df_plot = df.iloc[::step]
            
            ax2.plot(df_plot['time_progress'], df_plot['cumulative_pdr'],
                    color=color, label=label, linewidth=1.5, alpha=0.85)
        
        ax2.set_xlabel('Simulation Progress (%)')
        ax2.set_ylabel('Cumulative PDR (%)')
        ax2.set_title('Cumulative PDR Convergence')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 105)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        # Add shaded region for "learning phase" (first 20%)
        ax2.axvspan(0, 20, alpha=0.1, color='orange', label='_nolegend_')
        ax2.text(10, 8, 'Learning\nPhase', fontsize=8, ha='center', color='orange', alpha=0.7)
        
        fig.tight_layout()
        
        # Save
        convergence_png = self.output_dir / 'pdr_convergence.png'
        convergence_eps = self.output_dir / 'pdr_convergence.eps'
        fig.savefig(convergence_png, dpi=300, bbox_inches='tight')
        fig.savefig(convergence_eps, format='eps', bbox_inches='tight')
        print(f"  ✅ PDR convergence/learning curve saved")
        plt.close(fig)
        
        # =====================================================================
        # Per-Device Learning Curves (shows variance)
        # =====================================================================
        self._plot_per_device_convergence(present_types, line_colors)
        
        # =====================================================================
        # Time to Convergence Bar Chart
        # =====================================================================
        self._plot_time_to_convergence(convergence_data, present_types, line_colors)
    
    def _plot_per_device_convergence(self, present_types, line_colors):
        """
        Plot per-device PDR convergence - shows variance across devices during learning.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for adr_type in present_types:
            if adr_type not in self.device_data:
                continue
            
            color = line_colors.get(adr_type, '#333333')
            label = ADR_NAMES.get(adr_type, adr_type)
            
            # Collect rolling PDR per device, then compute mean ± std across devices
            device_rolling_pdrs = []
            max_len = 0
            
            for device_id, df in self.device_data[adr_type].items():
                df_sorted = df.sort_values('timestamp').reset_index(drop=True)
                window = min(20, len(df_sorted) // 5) if len(df_sorted) > 50 else 5
                rolling = df_sorted['packet_received'].rolling(window=window, min_periods=1).mean() * 100
                device_rolling_pdrs.append(rolling.values)
                max_len = max(max_len, len(rolling))
            
            if not device_rolling_pdrs:
                continue
            
            # Pad arrays to same length and compute stats
            padded = np.full((len(device_rolling_pdrs), max_len), np.nan)
            for i, arr in enumerate(device_rolling_pdrs):
                padded[i, :len(arr)] = arr
            
            mean_pdr = np.nanmean(padded, axis=0)
            std_pdr = np.nanstd(padded, axis=0)
            
            # Normalize x-axis to percentage of simulation
            x = np.linspace(0, 100, len(mean_pdr))
            
            # Downsample for cleaner plot
            step = max(1, len(x) // 200)
            x_plot = x[::step]
            mean_plot = mean_pdr[::step]
            std_plot = std_pdr[::step]
            
            ax.plot(x_plot, mean_plot, color=color, label=label, linewidth=1.5)
            ax.fill_between(x_plot, mean_plot - std_plot, mean_plot + std_plot,
                           color=color, alpha=0.15)
        
        ax.set_xlabel('Simulation Progress (%)')
        ax.set_ylabel('Rolling PDR (%) - Mean ± Std across devices')
        ax.set_title('Per-Device PDR Convergence with Variance')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'pdr_convergence_per_device.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / 'pdr_convergence_per_device.eps', format='eps', bbox_inches='tight')
        print(f"  ✅ Per-device convergence saved")
        plt.close(fig)
    
    def _plot_time_to_convergence(self, convergence_data, present_types, line_colors):
        """
        Bar chart showing time/packets to reach target PDR (e.g., 20%, 30%, 40%).
        """
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        # Use lower targets appropriate for these algorithms
        target_pdrs = [20, 30, 40]
        convergence_times = {target: {} for target in target_pdrs}
        
        for adr_type in present_types:
            if adr_type not in convergence_data:
                continue
            
            df = convergence_data[adr_type]
            final_pdr = df['cumulative_pdr'].iloc[-1]
            
            for target in target_pdrs:
                # Find first time cumulative PDR exceeds target
                above_target = df[df['cumulative_pdr'] >= target]
                if len(above_target) > 0:
                    first_idx = above_target.index[0]
                    progress_pct = (first_idx / len(df)) * 100
                    convergence_times[target][adr_type] = progress_pct
                else:
                    convergence_times[target][adr_type] = np.nan  # Never reached
        
        # Plot grouped bars
        x = np.arange(len(present_types))
        width = 0.25
        
        colors_target = ['#3498db', '#2ecc71', '#e74c3c']
        
        for i, target in enumerate(target_pdrs):
            values = [convergence_times[target].get(adr, np.nan) for adr in present_types]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=f'{target}% PDR', 
                         color=colors_target[i], edgecolor='black', linewidth=0.5)
            
            # Annotate bars
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.0f}%', ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('ADR Method')
        ax.set_ylabel('Simulation Progress to Reach Target (%)')
        ax.set_title('Time to Convergence (Lower = Faster Learning)')
        ax.set_xticks(x)
        ax.set_xticklabels([ADR_NAMES.get(t, t) for t in present_types], fontsize=9)
        ax.legend(title='Target PDR', loc='upper right', fontsize=8)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add note for algorithms that never converge
        ax.text(0.02, 0.98, '(Missing bars = target not reached)', transform=ax.transAxes,
               fontsize=7, va='top', color='gray', style='italic')
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'time_to_convergence.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / 'time_to_convergence.eps', format='eps', bbox_inches='tight')
        print(f"  ✅ Time to convergence saved")
        plt.close(fig)
    
    def generate_statistical_report(self, all_stats):
        """Generate detailed statistical report with enhanced tests"""
        print("\n" + "="*80)
        print("📊 STATISTICAL ANALYSIS REPORT (ENHANCED)")
        print("="*80)
        
        df_stats = pd.DataFrame(all_stats)
        
        # Overall statistics
        print("\n📈 OVERALL STATISTICS:")
        print("-" * 30)
        
        agg_cols = {'pdr': ['count', 'mean', 'std', 'min', 'max'],
                    'avg_snr': ['mean', 'std'],
                    'avg_tx_power': ['mean', 'std'],
                    'total_energy': ['mean', 'std']}
        
        # Add extended metrics if available
        if 'energy_per_success_mJ' in df_stats.columns:
            agg_cols['energy_per_success_mJ'] = ['mean', 'std']
        if 'goodput_bps' in df_stats.columns:
            agg_cols['goodput_bps'] = ['mean', 'std']
        
        summary_stats = df_stats.groupby('adr_type').agg(agg_cols).round(3)
        print(summary_stats)
        
        # Statistical significance tests
        print(f"\n🔬 STATISTICAL SIGNIFICANCE ANALYSIS:")
        print("-" * 40)
        
        from scipy.stats import ttest_ind, f_oneway, mannwhitneyu
        
        # Compare PDR between different ADR types
        no_adr_pdr = df_stats[df_stats['adr_type'] == 'no_adr']['pdr'].values
        classical_adr_pdr = df_stats[df_stats['adr_type'] == 'classical_adr']['pdr'].values
        ddqn_adr_pdr = df_stats[df_stats['adr_type'] == 'ddqn_adr']['pdr'].values
        ppo_adr_pdr = df_stats[df_stats['adr_type'] == 'ppo_adr']['pdr'].values
        marl_adr_pdr = df_stats[df_stats['adr_type'] == 'marl_adr']['pdr'].values
        hybrid_adr_pdr = df_stats[df_stats['adr_type'] == 'hybrid_adr']['pdr'].values
        
        comparisons = [
            ("DDQN-PER vs No ADR", ddqn_adr_pdr, no_adr_pdr),
            ("DDQN-PER vs Classical", ddqn_adr_pdr, classical_adr_pdr),
            ("PPO vs Classical", ppo_adr_pdr, classical_adr_pdr),
            ("MARL vs Classical", marl_adr_pdr, classical_adr_pdr),
            ("MARL vs DDQN-PER", marl_adr_pdr, ddqn_adr_pdr),
            ("Hybrid vs Classical", hybrid_adr_pdr, classical_adr_pdr),
        ]
        
        print("\n  📊 Parametric Tests (t-test):")
        for name, group1, group2 in comparisons:
            if len(group1) > 0 and len(group2) > 0:
                t_stat, p_value = ttest_ind(group1, group2)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"    {name}: t={t_stat:.3f}, p={p_value:.4f} ({significance})")
        
        print("\n  📊 Non-parametric Tests (Mann-Whitney U):")
        for name, group1, group2 in comparisons:
            if len(group1) > 0 and len(group2) > 0:
                try:
                    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                    delta = cliffs_delta(group1, group2)
                    effect_size = "large" if abs(delta) >= 0.474 else "medium" if abs(delta) >= 0.33 else "small" if abs(delta) >= 0.147 else "negligible"
                    print(f"    {name}: U={u_stat:.0f}, p={p_value:.4f}, Cliff's δ={delta:.3f} ({effect_size})")
                except Exception:
                    pass
        
        # Bootstrap Confidence Intervals
        print("\n  📊 Bootstrap 95% Confidence Intervals (PDR mean):")
        for adr_type in df_stats['adr_type'].unique():
            pdr_vals = df_stats[df_stats['adr_type'] == adr_type]['pdr'].values
            if len(pdr_vals) > 0:
                _, ci_lower, ci_upper = bootstrap_ci(pdr_vals, n_bootstrap=1000, ci=0.95)
                adr_name = ADR_NAMES.get(adr_type, adr_type)
                print(f"    {adr_name}: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
        
        # ANOVA test for all groups
        all_groups = [g for g in [no_adr_pdr, classical_adr_pdr, ddqn_adr_pdr, ppo_adr_pdr, marl_adr_pdr, hybrid_adr_pdr] if len(g) > 0]
        if len(all_groups) >= 3:
            f_stat, p_value = f_oneway(*all_groups)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"\n  📊 ANOVA (all ADR types): F={f_stat:.3f}, p={p_value:.4f} ({significance})")
        
        # Energy comparison
        print("\n  📊 Energy per Success Comparison (Mann-Whitney U):")
        if 'energy_per_success_mJ' in df_stats.columns:
            for baseline_name, baseline_type in [("Classical", "classical_adr")]:
                baseline_eps = df_stats[df_stats['adr_type'] == baseline_type]['energy_per_success_mJ'].replace([np.inf, -np.inf], np.nan).dropna().values
                if len(baseline_eps) == 0:
                    continue
                for adr_type in ['ddqn_adr', 'ppo_adr', 'marl_adr', 'hybrid_adr']:
                    eps = df_stats[df_stats['adr_type'] == adr_type]['energy_per_success_mJ'].replace([np.inf, -np.inf], np.nan).dropna().values
                    if len(eps) > 0:
                        try:
                            u_stat, p_value = mannwhitneyu(eps, baseline_eps, alternative='two-sided')
                            delta = cliffs_delta(eps, baseline_eps)
                            effect_size = "large" if abs(delta) >= 0.474 else "medium" if abs(delta) >= 0.33 else "small" if abs(delta) >= 0.147 else "negligible"
                            adr_name = ADR_NAMES.get(adr_type, adr_type)
                            print(f"    {adr_name} vs {baseline_name}: U={u_stat:.0f}, p={p_value:.4f}, δ={delta:.3f} ({effect_size})")
                        except Exception:
                            pass
    
    def export_results(self, all_stats, device_comparison):
        """Export results to CSV files with extended metrics"""
        print(f"\n💾 Exporting analysis results...")
        
        # Export overall statistics
        df_stats = pd.DataFrame(all_stats)
        
        # Define columns to export (core + extended)
        core_cols = ['adr_type', 'device_id', 'pdr', 'total_packets', 'received_packets', 
                     'lost_packets', 'avg_snr', 'avg_tx_power', 'avg_sf', 'total_energy', 
                     'distance', 'position_x', 'position_y']
        extended_cols = ['energy_per_success_mJ', 'goodput_bps', 'battery_life_days', 
                         'avg_toa_ms', 'channel_util_pct', 'snr_margin_mean', 'snr_margin_min',
                         'pdr_rolling_mean', 'outage_ratio', 'sf_churn_rate', 'tp_churn_rate',
                         'pct_collision_likely', 'pct_weak_signal_likely', 'max_consecutive_loss']
        
        # Keep columns that exist
        export_cols = [c for c in core_cols + extended_cols if c in df_stats.columns]
        df_export = df_stats[export_cols]
        
        output_file = self.output_dir / 'lorawan_adr_analysis_results.csv'
        df_export.to_csv(output_file, index=False)
        print(f"  ✅ Overall statistics exported to: {output_file}")
        
        # Export device comparison summary
        comparison_data = []
        for device_id, device_data in device_comparison.items():
            for adr_type, stats in device_data.items():
                row = {
                    'device_id': device_id,
                    'adr_type': adr_type,
                    'pdr': stats['pdr'],
                    'distance': stats['distance'],
                    'position_x': stats['position_x'],
                    'position_y': stats['position_y']
                }
                # Add extended metrics if available
                for ext_col in extended_cols:
                    if ext_col in stats:
                        row[ext_col] = stats[ext_col]
                comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        comparison_file = self.output_dir / 'device_comparison_summary.csv'
        df_comparison.to_csv(comparison_file, index=False)
        print(f"  ✅ Device comparison exported to: {comparison_file}")
        
        # Export extended metrics summary per ADR type
        summary_data = []
        for adr_type in df_stats['adr_type'].unique():
            adr_data = df_stats[df_stats['adr_type'] == adr_type]
            row = {'adr_type': adr_type, 'n_devices': len(adr_data)}
            row['pdr_mean'] = adr_data['pdr'].mean()
            row['pdr_std'] = adr_data['pdr'].std()
            _, row['pdr_ci_lower'], row['pdr_ci_upper'] = bootstrap_ci(adr_data['pdr'].values, n_bootstrap=1000, ci=0.95)
            row['energy_total_mean'] = adr_data['total_energy'].mean()
            if 'energy_per_success_mJ' in adr_data.columns:
                eps = adr_data['energy_per_success_mJ'].replace([np.inf, -np.inf], np.nan).dropna()
                row['energy_per_success_mean'] = eps.mean() if len(eps) > 0 else np.nan
            if 'goodput_bps' in adr_data.columns:
                row['goodput_mean'] = adr_data['goodput_bps'].mean()
            if 'battery_life_days' in adr_data.columns:
                bl = adr_data['battery_life_days'].replace([np.inf, -np.inf], np.nan).dropna()
                row['battery_life_mean_days'] = bl.mean() if len(bl) > 0 else np.nan
            if 'outage_ratio' in adr_data.columns:
                row['outage_ratio_mean'] = adr_data['outage_ratio'].mean()
            if 'sf_churn_rate' in adr_data.columns:
                row['sf_churn_mean'] = adr_data['sf_churn_rate'].mean()
            # Fairness
            device_stats_dict = {r['device_id']: {'pdr': r['pdr'], 'energy_per_success_mJ': r.get('energy_per_success_mJ', float('inf'))} 
                                 for _, r in adr_data.iterrows()}
            fairness = compute_fairness_metrics(device_stats_dict)
            row['jains_fairness_pdr'] = fairness.jains_fairness_pdr
            row['jains_fairness_energy_efficiency'] = fairness.jains_fairness_energy_efficiency
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        summary_file = self.output_dir / 'extended_metrics_summary.csv'
        df_summary.to_csv(summary_file, index=False)
        print(f"  ✅ Extended metrics summary exported to: {summary_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting LoRaWAN ADR Performance Analysis...")
        print("="*60)
        print(f"📁 Output directory: {self.output_dir}")
        
        # Load datasets
        self.load_datasets()
        
        # Generate analysis
        all_stats, device_comparison = self.generate_comprehensive_analysis()
        
        if not all_stats:
            print("❌ No data found for analysis!")
            return
        
        # Print reports
        self.print_executive_summary(all_stats)
        self.print_device_by_device_analysis(device_comparison)
        self.generate_statistical_report(all_stats)
        
        # Create visualizations
        try:
            self.create_performance_visualizations(all_stats, device_comparison)
        except ImportError:
            print("⚠  Matplotlib/Seaborn not available - skipping visualizations")
        except Exception as e:
            import traceback
            print(f"⚠  Error creating visualizations: {e}")
            traceback.print_exc()
        
        # Export results
        self.export_results(all_stats, device_comparison)
        
        print(f"\n✅ Analysis complete! Generated insights for {len(device_comparison)} devices")
        print("📋 Summary files created:")
        print(f"   - {self.output_dir / 'lorawan_adr_analysis_results.csv'}")
        print(f"   - {self.output_dir / 'device_comparison_summary.csv'}")
        print(f"   - {self.output_dir / 'pdr_by_adr_type.eps'}")
        print(f"   - {self.output_dir / 'energy_by_adr_type.eps'}")
        print(f"   - {self.output_dir / 'device_placement.eps'}")

def main():
    """Main function to run the analysis"""
    # Default datasets folder
    datasets_folder = "./lorawan_datasets"
    
    # Check if folder exists
    if not os.path.exists(datasets_folder):
        print(f"❌ Datasets folder not found: {datasets_folder}")
        print("Please update the datasets_folder path in the script.")
        return
    
    # Create analyzer and run analysis
    analyzer = LoRaWANADRAnalyzer(datasets_folder)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()