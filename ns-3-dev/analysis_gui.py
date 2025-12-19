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
import warnings
warnings.filterwarnings('ignore')

class LoRaWANADRAnalyzer:
    def __init__(self, datasets_folder):
        self.datasets_folder = Path(datasets_folder)
        self.device_data = {}
        self.summary_stats = {}
        
    def load_datasets(self):
        """Load all CSV datasets for analysis"""
        print("🔍 Loading LoRaWAN datasets...")
        
        # Find all device datasets - use specific patterns for GUI-generated files
        patterns = {
            'no_adr': ['no_adr_simulation_results_device_*_dataset.csv'],
            'classical_adr': ['adr_simulation_results_device_*_dataset.csv'],
            'ddqn_adr': ['ddqn_adr_simulation_results_device_*_dataset.csv']
        }
        
        for adr_type, pattern_list in patterns.items():
            files = []
            # Try all patterns for this ADR type
            for pattern in pattern_list:
                files.extend(glob.glob(str(self.datasets_folder / pattern)))
            
            # Remove duplicates if any
            files = list(set(files))
            self.device_data[adr_type] = {}
            
            for file_path in files:
                # Extract device ID from filename
                filename = os.path.basename(file_path)
                device_id = filename.split('device')[1].split('_dataset')[0].strip('_')
                
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
            for adr_type in ['no_adr', 'classical_adr', 'ddqn_adr']:
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
        
        for adr_type in ['no_adr', 'classical_adr', 'ddqn_adr']:
            if adr_type in adr_summary.index:
                row = adr_summary.loc[adr_type]
                adr_name = {
                    'no_adr': 'No ADR',
                    'classical_adr': 'Classical ADR', 
                    'ddqn_adr': 'DDQN-PER ADR'
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
            
            for adr_type in ['no_adr', 'classical_adr', 'ddqn_adr']:
                if adr_type in device_data:
                    stats = device_data[adr_type]
                    adr_name = {
                        'no_adr': 'No ADR',
                        'classical_adr': 'Classical ADR',
                        'ddqn_adr': 'DDQN-PER ADR'
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
    
    def create_performance_visualizations(self, all_stats):
        """Create performance visualization charts - DISABLED FOR GUI"""
        print("\n📊 Visualization generation disabled for GUI")
        return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LoRaWAN ADR Performance Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. PDR Comparison by ADR Type
        adr_mapping = {
            'no_adr': 'No ADR',
            'classical_adr': 'Classical ADR',
            'ddqn_adr': 'DDQN-PER ADR'
        }
        df_stats['adr_type_label'] = df_stats['adr_type'].map(adr_mapping)
        
        sns.boxplot(data=df_stats, x='adr_type_label', y='pdr', ax=axes[0,0])
        axes[0,0].set_title('Packet Delivery Rate (PDR) by ADR Type')
        axes[0,0].set_ylabel('PDR (%)')
        axes[0,0].set_xlabel('ADR Method')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. SNR vs Distance scatter plot
        colors = {'No ADR': 'blue', 'Classical ADR': 'orange', 'DDQN-PER ADR': 'green'}
        for adr_type in df_stats['adr_type_label'].unique():
            data = df_stats[df_stats['adr_type_label'] == adr_type]
            axes[0,1].scatter(data['distance'], data['avg_snr'], 
                            label=adr_type, alpha=0.7, s=50, c=colors.get(adr_type, 'gray'))
        
        axes[0,1].set_title('SNR vs Distance by ADR Type')
        axes[0,1].set_xlabel('Distance from Gateway (m)')
        axes[0,1].set_ylabel('Average SNR (dB)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. TX Power Distribution
        sns.boxplot(data=df_stats, x='adr_type_label', y='avg_tx_power', ax=axes[1,0])
        axes[1,0].set_title('TX Power Distribution by ADR Type')
        axes[1,0].set_ylabel('Average TX Power (dBm)')
        axes[1,0].set_xlabel('ADR Method')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Energy Consumption
        sns.barplot(data=df_stats, x='adr_type_label', y='total_energy', ax=axes[1,1])
        axes[1,1].set_title('Energy Consumption by ADR Type')
        axes[1,1].set_ylabel('Total Energy Consumed (mJ)')
        axes[1,1].set_xlabel('ADR Method')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'lorawan_adr_performance_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Performance charts saved as: {output_file}")
        
        plt.show()
    
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
        
        if len(no_adr_pdr) > 0 and len(ddqn_adr_pdr) > 0:
            t_stat, p_value = ttest_ind(ddqn_adr_pdr, no_adr_pdr)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"DDQN-PER vs No ADR PDR: t={t_stat:.3f}, p={p_value:.3f} ({significance})")
        
        if len(classical_adr_pdr) > 0 and len(ddqn_adr_pdr) > 0:
            t_stat, p_value = ttest_ind(ddqn_adr_pdr, classical_adr_pdr)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"DDQN-PER vs Classical ADR PDR: t={t_stat:.3f}, p={p_value:.3f} ({significance})")
        
        # ANOVA test for all three groups
        if len(no_adr_pdr) > 0 and len(classical_adr_pdr) > 0 and len(ddqn_adr_pdr) > 0:
            f_stat, p_value = f_oneway(no_adr_pdr, classical_adr_pdr, ddqn_adr_pdr)
            significance = "significant" if p_value < 0.05 else "not significant"
            print(f"ANOVA test (all ADR types): F={f_stat:.3f}, p={p_value:.3f} ({significance})")
    
    def export_results(self, all_stats, device_comparison):
        """Export results to CSV files"""
        print(f"\n💾 Exporting analysis results...")
        
        # Export overall statistics
        df_stats = pd.DataFrame(all_stats)
        output_file = 'lorawan_adr_analysis_results.csv'
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
        comparison_file = 'device_comparison_summary.csv'
        df_comparison.to_csv(comparison_file, index=False)
        print(f"  ✅ Device comparison exported to: {comparison_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting LoRaWAN ADR Performance Analysis...")
        print("="*60)
        
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
        
        # Skip visualizations for GUI version
        print("📊 Visualization skipped (GUI mode)")
        
        # Export results
        self.export_results(all_stats, device_comparison)
        
        print(f"\n✅ Analysis complete! Generated insights for {len(device_comparison)} devices")
        print("📋 Summary files created:")
        print("   - lorawan_adr_analysis_results.csv")
        print("   - device_comparison_summary.csv")

def main():
    """Main function to run the analysis"""
    import sys
    
    # Check if datasets folder is provided as command line argument
    if len(sys.argv) > 1:
        datasets_folder = sys.argv[1]
    else:
        # Default datasets folder
        datasets_folder = "./lorawan_datasets"
    
    # Check if folder exists
    if not os.path.exists(datasets_folder):
        print(f"❌ Datasets folder not found: {datasets_folder}")
        print("Please provide a valid datasets folder path as an argument:")
        print("  python3 analysis_gui.py /path/to/results-csv/")
        return
    
    print(f"📂 Using datasets folder: {datasets_folder}")
    
    # Create analyzer and run analysis
    analyzer = LoRaWANADRAnalyzer(datasets_folder)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()