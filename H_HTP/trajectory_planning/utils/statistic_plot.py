import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl



def bar_plot_plan(read_dir):
    # 初始化一个空列表，用于存储所有数据
    all_data = []

    # 遍历目录中的所有 CSV 文件
    for filename in os.listdir(read_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(read_dir, filename)
            # 读取 CSV 文件，假设每个文件只有一列数据
            data = pd.read_csv(file_path, header=None).values.flatten()
            all_data.extend(data)

    # 将数据乘以1000（转换为毫秒）
    data_ms = np.array(all_data) * 1000

    # 定义区间边界 - 注意数量比labels多1
    bins = [0, 45, 50, 55, 60, 65, 70, 75, 80, 300]  # 共9个边界点

    # 定义对应的labels - 共8个区间标签
    labels = range(9)
    tick_labels = ['<45', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '>80']


    # Bin the data into the defined ranges
    binned_data = pd.cut(data_ms, bins=bins, labels=labels, right=False).value_counts().sort_index()
    
    plt.bar(labels, binned_data, tick_label = tick_labels)
    
    # Calculate the total count of all data points
    total_count = len(data_ms)
    # Calculate the percentage of each bin
    bin_percentages = [(count / total_count) * 100 for count in binned_data]
    # Add the percentage of each bar to the plot
    for i, percentage in enumerate(bin_percentages):
        plt.text(i, binned_data[i], f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.title('Histogram of Data Distribution')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y')
    
    # 调整布局防止标签被截断
    plt.tight_layout()
    plt.show()

def publication_quality_bar_plot(read_dir, save_dir, dir_idx, figsize=(8, 6), dpi=300):
    """
    Generate a publication-quality histogram of latency data distribution.
    
    Parameters:
    -----------
    read_dir : str
        Directory containing CSV files with latency data (in seconds)
    figsize : tuple, optional
        Figure dimensions in inches (default: (8, 6))
    dpi : int, optional
        Resolution in dots per inch (default: 300)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Set academic style parameters
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.titlesize': 20,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

    # Initialize data collection
    all_data = []
    
    # Data collection
    for filename in os.listdir(read_dir[dir_idx]):
        if filename.endswith('.csv'):
            file_path = os.path.join(read_dir[dir_idx], filename)
            data = pd.read_csv(file_path, header=None).values.flatten()
            all_data.extend(data)
    
    # Convert to milliseconds
    data_ms = np.array(all_data) * 1000
    
    # Bin definitions
    if dir_idx == 0:
        bins = [0, 45, 50, 55, 60, 65, 70, 75, 80, 300]
        labels =  [  '[0, 45)', '[45, 50)', '[50, 55)','[55, 60)','[60, 65)',
        '[65, 70)', '[70, 75)', '[75, 80)',   '[80, ∞)'  ]

    else:
        bins = [0, 7, 9, 11, 13, 15, 17, 19, 21, 100]
        labels = [ '[0,7)', '[7,9)', '[9,11)', '[11,13)', '[13,15)',  # 第一行5个区间
            '[15,17)', '[17,19)', '[19,21)', '[21,∞)'  ]

    
    # Create figure with constrained layout
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Bin the data
    binned_data = pd.cut(data_ms, bins=bins, labels=labels, right=False).value_counts().sort_index()
    
    # Create bar plot with academic style
    bars = ax.bar(labels, binned_data, 
                 color='#1f77b4',  # Matplotlib default blue
                 edgecolor='black',
                 linewidth=0.5,
                 alpha=0.8,
                 width=0.7)
    
    # Add percentage annotations
    total_count = len(data_ms)
    bin_percentages = (binned_data / total_count) * 100
    
    for i, (rect, percentage) in enumerate(zip(bars, bin_percentages)):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.02*max(binned_data),
                f'{percentage:.2f}%',
                ha='center', va='bottom',
                fontsize=15)
    
    # Axis labels and title
    ax.set_xlabel('Running Time (ms)')   #, fontweight='bold'
    ax.set_ylabel('Frequency Count')
    # ax.set_title('Distribution of System Latency Measurements',
    #             fontweight='bold', pad=20)
    
    # Customize ticks and grid
    ax.set_xticks(labels)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Remove top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # # Add sample size annotation
    # ax.text(0.98, 0.98, f'N = {total_count:,}',
    #         transform=ax.transAxes,
    #         ha='right', va='top',
    #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add minor ticks for better readability
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    
    fig.show()
    # fig.savefig(save_dir[dir_idx]+'latency_distribution.svg', bbox_inches='tight')
    plt.close(fig)
    
    return total_count

def box_plot_plan(read_dir):
    # Initialize an empty list, used to store all data
    all_data = []

    # Traverse the directory for all CSV files
    for filename in os.listdir(read_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(read_dir[dir_idx], filename)
            # Read CSV file, assuming each file has only one column of data
            data = pd.read_csv(file_path, header=None).values.flatten()
            all_data.extend(data)

    # Convert data to milliseconds
    data_ms = np.array(all_data) * 1000
    # Calculate median and mean values
    median_value = np.median(data_ms)
    mean_value = np.mean(data_ms)
    print("median value : %.4f" % median_value)
    print("mean value : %.4f"% mean_value)

    # Plot the box plot
    plt.boxplot(data_ms, positions=[10], showfliers=False, widths=10)
    
    # Add a horizontal line for the mean value
    plt.axhline(y=mean_value, color='green', linestyle='--', label='Mean')
    # plt.axhline(y=median_value, color='red', linestyle='--', label='Mean')

    # plt.title('Box Plot of Data Distribution')
    # plt.xlabel('Time (ms)')
    plt.ylabel('Time (ms)')
    plt.xticks([10], ['recordings 53 to 55'])
    plt.grid(True, axis='y')

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    plt.show()

def publication_quality_box_plot(read_dir, save_dir, dir_idx, figsize=(6, 6), dpi=300):
    """
    Generate a publication-quality box plot of latency data.
    
    Parameters:
    -----------
    read_dir : str
        Directory containing CSV files with latency data (in seconds)
    figsize : tuple, optional
        Figure dimensions in inches (default: (6, 6))
    dpi : int, optional
        Resolution in dots per inch (default: 300)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Set academic style parameters without bold text
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 15,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.dpi': dpi,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.titlepad': 12
    })

    # Initialize data collection
    all_data = []
    
    # Data collection
    for filename in os.listdir(read_dir[dir_idx]):
        if filename.endswith('.csv'):
            file_path = os.path.join(read_dir[dir_idx], filename)
            data = pd.read_csv(file_path, header=None).values.flatten()
            all_data.extend(data)
    
    # Convert to milliseconds
    data_ms = np.array(all_data) * 1000
    
    # Calculate statistics
    median_value = np.median(data_ms)
    mean_value = np.mean(data_ms)
    
    # Create figure with constrained layout
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Create box plot with academic style
    boxprops = dict(linestyle='-', linewidth=1, color='black')
    medianprops = dict(linestyle='-', linewidth=1.5, color='#1f77b4')
    whiskerprops = dict(linestyle='-', linewidth=1, color='black')
    capprops = dict(linestyle='-', linewidth=1, color='black')
    
    
    # box = plt.boxplot(data, 
    #                 patch_artist=True,
    #                 showfliers=True,  # 显示异常值（默认即为True）
    #                 flierprops=dict(
    #                     marker='o',          # 圆形标记
    #                     markerfacecolor='r', # 红色填充
    #                     markersize=8,        # 大小
    #                     markeredgecolor='k'  # 黑色边框
    #                 )
    #                 )
    bp = ax.boxplot(data_ms, 
                   positions=[1], 
                   showfliers=True, 
                   widths=0.6,
                   patch_artist=True,
                   boxprops=boxprops,
                   medianprops=medianprops,
                   whiskerprops=whiskerprops,
                   capprops=capprops,
                    flierprops=dict(
                        marker='o',          # 圆形标记
                        markersize=8,        # 大小
                        markeredgecolor='grey'  # 黑色边框
                    )
                   )
    
    # Set box fill color
    for box in bp['boxes']:
        box.set(facecolor='#ffffff', alpha=0.8)
    
    # Add mean line with subtle styling
    ax.axhline(y=mean_value, 
              color='#d62728', 
              linestyle=':', 
              linewidth=3,
              label=f'Mean ({mean_value:.2f} ms)')
    
    # Add median line for reference (optional)
    ax.axhline(y=median_value, 
              color='steelblue', 
              linestyle='--', 
              linewidth=3,
              alpha=0.7,
              label=f'Median ({median_value:.2f} ms)')
    
    # Axis labels and title
    ax.set_ylabel('Running Time (ms)')
    ax.set_xlabel('')
    ax.set_xticks([1])
    ax.set_xticklabels(['Recordings 53-55'])
    
    # Customize grid and spines
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Add legend with transparent background
    ax.legend(frameon=True, framealpha=0.8)
    
    # Adjust y-axis limits to accommodate mean line
    if dir_idx == 0:
        y_min, y_max = min(data_ms) * 0.95 , max(data_ms) * 1.05 # 35, 75  
    else:
        y_min, y_max = 3, 23  # min(data_ms) * 0.95    #max(data_ms) * 1.05
    ax.set_ylim(y_min, y_max)
    
    fig.show()
    # fig.savefig(save_dir[dir_idx]+ 'latency_boxplot2.svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return 1


def bar_plot_solve(read_dir):
    return 1

def box_plot_solve(read_dir):
    return 1


if __name__ == '__main__':
    # # 设置 CSV 文件所在的目录
    # csv_directory = [ '/home/chwei/AutoVehicle_DataAndOther/myData/RESULTS_2024_2025/running_time/plannning_time', 
    #                  '/home/chwei/AutoVehicle_DataAndOther/myData/RESULTS_2024_2025/running_time/solving_time'  ]
    # save_directory = ['/home/chwei/AutoVehicle_DataAndOther/myData/RESULTS_2024_2025/running_time/plannning_time/plots/',
    #                     '/home/chwei/AutoVehicle_DataAndOther/myData/RESULTS_2024_2025/running_time/solving_time/plots/'  ]
    # # box_plot_plan(csv_directory)
    # # bar_plot_plan(csv_directory)
    # dir_idx = 0 #  0--- 轨迹规划时间   1---优化模型求解时间
    # # total_count = publication_quality_bar_plot(csv_directory, save_directory, dir_idx)
    # # print('Total count: %d' % total_count)
    # publication_quality_box_plot(csv_directory, save_directory, dir_idx)
    
