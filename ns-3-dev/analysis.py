#!/usr/bin/env python3
"""
LoRaWAN ADR Comparison Analysis Tool
Analyzes DDQN-PER ADR vs Classical ADR vs No ADR performance
Device-by-device insights and comprehensive statistics
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
warnings.filterwarnings('ignore')

class LoRaWANADRAnalyzer:
    def __init__(self, datasets_folder, output_root="graphs"):
        self.datasets_folder = Path(datasets_folder)
        self.output_root = Path(output_root)
        self.run_hash = self._generate_run_hash()
        self.output_dir = self.output_root / self.run_hash
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device_data = {}
        self.summary_stats = {}

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
                # Extract device ID from filename
                filename = os.path.basename(file_path)
                device_id = filename.split('device')[1].split('_dataset')[0]
                
                try:
                    df = pd.read_csv(file_path)
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
                adr_name = {
                    'no_adr': 'No ADR',
                    'classical_adr': 'Classical ADR', 
                    'ddqn_adr': 'DDQN-PER ADR',
                    'ppo_adr': 'PPO ADR',
                    'marl_adr': 'MARL ADR',
                    'hybrid_adr': 'HYBRID ADR'
                }[adr_type]
                
                print(f"\n{adr_name}:")
                print(f"  📈 Packet Delivery Rate: {row[('pdr', 'mean')]:.1f}% (±{row[('pdr', 'std')]:.1f}%)")
                print(f"  📡 Average SNR: {row[('avg_snr', 'mean')]:.1f} dB")
                print(f"  ⚡ Average TX Power: {row[('avg_tx_power', 'mean')]:.1f} dBm")
                print(f"  📶 Average Spreading Factor: {row[('avg_sf', 'mean')]:.1f}")
                print(f"  🔋 Average Energy per Device: {row[('total_energy', 'mean')]:.3f} mJ")
                print(f"  🏠 Devices Analyzed: {int(row[('pdr', 'count')])}")
        
        # Best performing ADR type
        avg_pdr_by_type = df_stats.groupby('adr_type')['pdr'].mean()
        best_adr = avg_pdr_by_type.idxmax()
        improvement = avg_pdr_by_type[best_adr] - avg_pdr_by_type.drop(best_adr).max()
        
        print(f"\n🏆 BEST PERFORMING ADR: {best_adr.replace('_', ' ').title()}")
        print(f"   Improvement: +{improvement:.1f} percentage points")
        
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
            
            for adr_type in ['no_adr', 'classical_adr', 'ddqn_adr', 'ppo_adr', 'marl_adr', 'hybrid_adr']:
                if adr_type in device_data:
                    stats = device_data[adr_type]
                    adr_name = {
                        'no_adr': 'No ADR',
                        'classical_adr': 'Classical ADR',
                        'ddqn_adr': 'DDQN-PER ADR',
                        'ppo_adr': 'PPO ADR',
                        'marl_adr': 'MARL ADR',
                        'hybrid_adr': 'HYBRID ADR'
                    }[adr_type]
                    
                    print(f"{adr_name:<15} {stats['pdr']:<8.1f} {stats['avg_snr']:<10.1f} "
                          f"{stats['avg_tx_power']:<10.1f} {stats['avg_sf']:<5.1f} {stats['total_energy']:<12.3f}")
            
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
        
        print(f"  📄 All figures saved in EPS format (LaTeX / paper ready): {self.output_dir}")
    
    def generate_statistical_report(self, all_stats):
        """Generate detailed statistical report"""
        print("\n" + "="*80)
        print("📊 STATISTICAL ANALYSIS REPORT")
        print("="*80)
        
        df_stats = pd.DataFrame(all_stats)
        
        # Overall statistics
        print("\n📈 OVERALL STATISTICS:")
        print("-" * 30)
        
        summary_stats = df_stats.groupby('adr_type').agg({
            'pdr': ['count', 'mean', 'std', 'min', 'max'],
            'avg_snr': ['mean', 'std'],
            'avg_tx_power': ['mean', 'std'],
            'total_energy': ['mean', 'std']
        }).round(3)
        
        print(summary_stats)
        
        # Statistical significance tests
        print(f"\n🔬 STATISTICAL SIGNIFICANCE ANALYSIS:")
        print("-" * 40)
        
        from scipy.stats import ttest_ind, f_oneway
        
        # Compare PDR between different ADR types
        no_adr_pdr = df_stats[df_stats['adr_type'] == 'no_adr']['pdr']
        classical_adr_pdr = df_stats[df_stats['adr_type'] == 'classical_adr']['pdr']
        ddqn_adr_pdr = df_stats[df_stats['adr_type'] == 'ddqn_adr']['pdr']
        ppo_adr_pdr = df_stats[df_stats['adr_type'] == 'ppo_adr']['pdr']
        marl_adr_pdr = df_stats[df_stats['adr_type'] == 'marl_adr']['pdr']
        
        if len(no_adr_pdr) > 0 and len(ddqn_adr_pdr) > 0:
            t_stat, p_value = ttest_ind(ddqn_adr_pdr, no_adr_pdr)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"DDQN-PER vs No ADR PDR: t={t_stat:.3f}, p={p_value:.3f} ({significance})")
        
        if len(classical_adr_pdr) > 0 and len(ddqn_adr_pdr) > 0:
            t_stat, p_value = ttest_ind(ddqn_adr_pdr, classical_adr_pdr)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"DDQN-PER vs Classical ADR PDR: t={t_stat:.3f}, p={p_value:.3f} ({significance})")
        
        if len(ppo_adr_pdr) > 0 and len(classical_adr_pdr) > 0:
            t_stat, p_value = ttest_ind(ppo_adr_pdr, classical_adr_pdr)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"PPO vs Classical ADR PDR: t={t_stat:.3f}, p={p_value:.3f} ({significance})")
        
        if len(marl_adr_pdr) > 0 and len(classical_adr_pdr) > 0:
            t_stat, p_value = ttest_ind(marl_adr_pdr, classical_adr_pdr)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"MARL vs Classical ADR PDR: t={t_stat:.3f}, p={p_value:.3f} ({significance})")
        
        if len(marl_adr_pdr) > 0 and len(ddqn_adr_pdr) > 0:
            t_stat, p_value = ttest_ind(marl_adr_pdr, ddqn_adr_pdr)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"MARL vs DDQN-PER ADR PDR: t={t_stat:.3f}, p={p_value:.3f} ({significance})")
        
        # ANOVA test for all groups
        all_groups = [g for g in [no_adr_pdr, classical_adr_pdr, ddqn_adr_pdr, ppo_adr_pdr, marl_adr_pdr] if len(g) > 0]
        if len(all_groups) >= 3:
            f_stat, p_value = f_oneway(*all_groups)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"ANOVA test (all ADR types): F={f_stat:.3f}, p={p_value:.3f} ({significance})")
    
    def export_results(self, all_stats, device_comparison):
        """Export results to CSV files"""
        print(f"\n💾 Exporting analysis results...")
        
        # Export overall statistics
        df_stats = pd.DataFrame(all_stats)
        output_file = self.output_dir / 'lorawan_adr_analysis_results.csv'
        df_stats.to_csv(output_file, index=False)
        print(f"  ✅ Overall statistics exported to: {output_file}")
        
        # Export device comparison summary
        comparison_data = []
        for device_id, device_data in device_comparison.items():
            for adr_type, stats in device_data.items():
                comparison_data.append({
                    'device_id': device_id,
                    'adr_type': adr_type,
                    'pdr': stats['pdr'],
                    'distance': stats['distance'],
                    'position_x': stats['position_x'],
                    'position_y': stats['position_y']
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        comparison_file = self.output_dir / 'device_comparison_summary.csv'
        df_comparison.to_csv(comparison_file, index=False)
        print(f"  ✅ Device comparison exported to: {comparison_file}")
    
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