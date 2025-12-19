import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
import numpy as np
import os
import glob

def visualize_lorawan_environment(env_csv_path, output_dir="environment_plots"):
    """
    Visualize the LoRaWAN simulation environment from CSV data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read environment data
    df = pd.read_csv(env_csv_path)
    
    # Create figure with high DPI for crisp images
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    # Set background color
    ax.set_facecolor('#f0f8ff')  # Light blue background
    
    # Find the simulation area bounds
    all_x = df['x'].values
    all_y = df['y'].values
    margin = 50
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin
    
    # Plot urban obstacles (buildings)
    obstacles = df[df['type'] == 'obstacle']
    for _, obstacle in obstacles.iterrows():
        x, y = obstacle['x'], obstacle['y']
        width, height = obstacle['width'], obstacle['height']
        attenuation = obstacle['attenuation']
        
        # Color intensity based on attenuation (darker = more attenuation)
        color_intensity = (attenuation - 10) / 15  # Normalize 10-25 dB to 0-1
        color = plt.cm.Greys(0.3 + 0.6 * color_intensity)
        
        # Draw building as rectangle
        rect = Rectangle((x - width/2, y - width/2), width, width, 
                        facecolor=color, edgecolor='black', linewidth=0.5,
                        alpha=0.8, label='Buildings' if _ == obstacles.index[0] else "")
        ax.add_patch(rect)
        
        # Add attenuation text for some buildings
        if len(obstacles) < 50:  # Only if not too many
            ax.text(x, y, f'{attenuation:.0f}dB', ha='center', va='center', 
                   fontsize=6, color='white', weight='bold')
    
    # Plot WiFi interferers
    wifi_nodes = df[df['type'] == 'wifi_interferer']
    if not wifi_nodes.empty:
        ax.scatter(wifi_nodes['x'], wifi_nodes['y'], 
                  s=120, c='red', marker='s', alpha=0.7, 
                  edgecolors='darkred', linewidth=2, label='WiFi Interferers', zorder=5)
        
        # Add WiFi interference circles and labels
        for _, wifi in wifi_nodes.iterrows():
            circle = Circle((wifi['x'], wifi['y']), 100, 
                          fill=False, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
            
            # Add WiFi labels
            ax.annotate(f"W{wifi['id'].split('_')[1]}", 
                       (wifi['x'], wifi['y']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, color='darkred', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Plot end devices
    devices = df[df['type'] == 'end_device']
    if not devices.empty:
        # Color devices by TX power
        tx_powers = devices['attenuation'].values  # TX power stored in attenuation field
        scatter = ax.scatter(devices['x'], devices['y'], 
                           s=80, c=tx_powers, cmap='viridis', 
                           alpha=0.8, edgecolors='black', linewidth=1,
                           label='LoRa Devices', zorder=6)
        
        # Add colorbar for TX power
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('TX Power (dBm)', rotation=270, labelpad=15)
        
        # Add device IDs with better visibility
        for _, device in devices.iterrows():
            ax.annotate(f"D{device['id']}", 
                       (device['x'], device['y']), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, color='black', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Plot gateway
    gateway = df[df['type'] == 'gateway']
    if not gateway.empty:
        gw_x, gw_y = gateway['x'].iloc[0], gateway['y'].iloc[0]
        ax.scatter(gw_x, gw_y, s=300, c='gold', marker='*', 
                  edgecolors='orange', linewidth=3, label='Gateway', zorder=10)
        
        # Add gateway range circles
        for radius in [500, 1000, 1500]:
            circle = Circle((gw_x, gw_y), radius, 
                          fill=False, color='orange', linestyle=':', 
                          alpha=0.4, linewidth=1)
            ax.add_patch(circle)
        
        ax.text(gw_x, gw_y-50, 'GATEWAY', ha='center', va='top',
                fontsize=12, color='darkorange', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='orange'))
    
    # Customize plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Position (meters)', fontsize=12)
    ax.set_ylabel('Y Position (meters)', fontsize=12)
    ax.set_title('LoRaWAN Simulation Environment\nwith Urban Obstacles and Interference', 
                 fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_aspect('equal')
    
    # Add statistics text box
    stats_text = f"""Environment Statistics:
• Devices: {len(devices)}
• Buildings: {len(obstacles)}
• WiFi Interferers: {len(wifi_nodes)}
• Area: {(x_max-x_min)/1000:.1f} × {(y_max-y_min)/1000:.1f} km²"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8), fontsize=10)
    
    # Save the plot
    base_name = os.path.splitext(os.path.basename(env_csv_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_environment_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Environment map saved to: {output_path}")
    
    plt.close()  # Close figure to free memory
    
    return fig, ax

def create_interference_heatmap(env_csv_path, output_dir="environment_plots"):
    """
    Create a heatmap showing interference levels across the environment
    """
    df = pd.read_csv(env_csv_path)
    
    # Create grid for interference calculation
    margin = 100
    x_min, x_max = df['x'].min() - margin, df['x'].max() + margin
    y_min, y_max = df['y'].min() - margin, df['y'].max() + margin
    
    # Create meshgrid
    resolution = 50  # meters per grid point
    x_grid = np.arange(x_min, x_max, resolution)
    y_grid = np.arange(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate interference at each grid point
    interference_map = np.zeros_like(X)
    
    # WiFi interference
    wifi_nodes = df[df['type'] == 'wifi_interferer']
    for _, wifi in wifi_nodes.iterrows():
        distances = np.sqrt((X - wifi['x'])**2 + (Y - wifi['y'])**2)
        # Simple path loss model for WiFi interference
        wifi_interference = -130 + 20 * np.log10(np.maximum(distances, 1))
        interference_map += 10**(wifi_interference/10)  # Convert to linear, sum, convert back
    
    # Building shadowing
    obstacles = df[df['type'] == 'obstacle']
    shadowing_map = np.zeros_like(X)
    for _, obstacle in obstacles.iterrows():
        # Simple shadowing model - add attenuation in building area
        in_building = ((np.abs(X - obstacle['x']) < obstacle['width']/2) & 
                      (np.abs(Y - obstacle['y']) < obstacle['width']/2))
        shadowing_map += in_building * obstacle['attenuation']
    
    # Convert interference back to dBm
    interference_map = 10 * np.log10(interference_map + 1e-15)
    
    # Create the heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Interference heatmap
    im1 = ax1.contourf(X, Y, interference_map, levels=20, cmap='Reds', alpha=0.7)
    ax1.set_title('WiFi Interference Levels (dBm)', fontsize=14, weight='bold')
    plt.colorbar(im1, ax=ax1, label='Interference Level (dBm)')
    
    # Shadowing heatmap
    im2 = ax2.contourf(X, Y, shadowing_map, levels=20, cmap='Blues', alpha=0.7)
    ax2.set_title('Building Shadowing (dB)', fontsize=14, weight='bold')
    plt.colorbar(im2, ax=ax2, label='Additional Path Loss (dB)')
    
    # Add devices and obstacles to both plots
    for ax in [ax1, ax2]:
        # Plot buildings
        obstacles = df[df['type'] == 'obstacle']
        for _, obstacle in obstacles.iterrows():
            rect = Rectangle((obstacle['x'] - obstacle['width']/2, 
                            obstacle['y'] - obstacle['width']/2), 
                           obstacle['width'], obstacle['width'],
                           facecolor='black', alpha=0.3, edgecolor='black')
            ax.add_patch(rect)
        
        # Plot devices
        devices = df[df['type'] == 'end_device']
        ax.scatter(devices['x'], devices['y'], s=60, c='yellow', 
                  marker='o', edgecolors='black', linewidth=1, zorder=5)
        
        # Add device labels for all devices
        for _, device in devices.iterrows():
            ax.annotate(f"D{device['id']}", 
                       (device['x'], device['y']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='black', weight='bold')
        
        # Plot gateway
        gateway = df[df['type'] == 'gateway']
        ax.scatter(gateway['x'], gateway['y'], s=200, c='gold', marker='*',
                  edgecolors='orange', linewidth=2, zorder=6)
        
        # Plot WiFi
        wifi_nodes = df[df['type'] == 'wifi_interferer']
        ax.scatter(wifi_nodes['x'], wifi_nodes['y'], s=80, c='red', 
                  marker='s', edgecolors='darkred', linewidth=1, zorder=5)
        
        # Add WiFi labels
        for _, wifi in wifi_nodes.iterrows():
            ax.annotate(f"W{wifi['id'].split('_')[1]}", 
                       (wifi['x'], wifi['y']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, color='darkred', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    base_name = os.path.splitext(os.path.basename(env_csv_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_interference_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Interference heatmap saved to: {output_path}")
    
    plt.close()  # Close figure to free memory

def visualize_all_environments():
    """
    Find and visualize all environment CSV files
    """
    env_files = glob.glob("lorawan_datasets/*_environment.csv")
    
    if not env_files:
        print("No environment CSV files found!")
        print("Make sure to run your ns-3 simulation first.")
        return
    
    print(f"Found {len(env_files)} environment files:")
    
    for env_file in env_files:
        print(f"\nProcessing: {env_file}")
        try:
            # Create environment map
            visualize_lorawan_environment(env_file)
            
            # Create interference heatmap
            create_interference_heatmap(env_file)
            
        except Exception as e:
            print(f"Error processing {env_file}: {e}")

if __name__ == "__main__":
    # Visualize all environment files
    visualize_all_environments()
    
    # Or visualize a specific file
    # visualize_lorawan_environment("lorawan_datasets/no_adr_advanced_pre_tx_rssi_dataset_environment.csv")