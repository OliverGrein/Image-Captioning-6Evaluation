import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Directory containing the CSV files
csv_dir = 'data/image_distance_final'

# Function to process each CSV and generate plots
def process_csv(file_path, file_name):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Calculate absolute difference
    df['absolute_difference'] = np.abs(df['image_distance_candidate'] - df['image_distance_reference'])
    df = df[df['percent_yes_x'] != -1]
    

    # Round percent_yes_x to the first decimal place
    df['percent_yes_x'] = (df['percent_yes_x'] * 10).round() / 10
    print(df['percent_yes_x'].unique())
    
    # 1. One plot per metric that has one bar per percent_yes_x (candidates only)
    def plot_candidate_by_percent(data, filename):
        metrics = data['metric'].unique()
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6*len(metrics)), squeeze=False)

        for i, metric in enumerate(metrics):
            metric_data = data[data['metric'] == metric]
            sns.barplot(x='percent_yes_x', y='image_distance_candidate', data=metric_data, ax=axes[i, 0])
            axes[i, 0].set_title(f'Candidate Distances for {metric} by Percent Yes')
            axes[i, 0].set_xlabel('Percent Yes')
            axes[i, 0].set_ylabel('Distance')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_candidate_by_percent(df, f'{file_name}_candidate_distances_by_percent_yes.png')

    # 3. One plot showing absolute difference per metric with one bar per percent_yes_x (candidates only)
    def plot_absolute_diff_by_percent(data, filename):
        plt.figure(figsize=(15, 8))

        sns.barplot(x='metric', y='absolute_difference', hue='percent_yes_x', data=data, ci=None)

        plt.title('Absolute Difference by Metric and Percent Yes')
        plt.xlabel('Metric')
        plt.ylabel('Average Absolute Difference')
        plt.legend(title='Percent Yes')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_absolute_diff_by_percent(df, f'{file_name}_absolute_difference_by_percent_yes.png')

    def plot_candidate_diff_by_percent(data, filename):
        plt.figure(figsize=(15, 8))

        sns.barplot(x='metric', y='image_distance_candidate', hue='percent_yes_x', data=data, ci=None)

        plt.title('Candidate Difference by Metric and Percent Yes')
        plt.xlabel('Metric')
        plt.ylabel('Average Candidate Distance')
        plt.legend(title='Percent Yes')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_candidate_diff_by_percent(df, f'{file_name}_candidate_difference_by_percent_yes.png')
    
    # Calculate average image_distance_candidate by percent_yes_x
    avg_distance = df.groupby(['percent_yes_x', 'metric'])['image_distance_candidate'].mean().reset_index()
    
    return avg_distance

# List to store all average distances
all_avg_distances = []

# Iterate over all CSV files in the directory
for file in os.listdir(csv_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(csv_dir, file)
        file_name = os.path.splitext(file)[0]
        avg_distance = process_csv(file_path, file_name)
        avg_distance['file_name'] = file_name
        all_avg_distances.append(avg_distance)

# Combine all average distances
combined_avg_distances = pd.concat(all_avg_distances, ignore_index=True)

# Save the combined average distances to a CSV file
output_csv_path = 'average_image_distances_by_percent_yes.csv'
combined_avg_distances.to_csv(output_csv_path, index=False)

print(f"Average image distances by percent yes saved to {output_csv_path}")