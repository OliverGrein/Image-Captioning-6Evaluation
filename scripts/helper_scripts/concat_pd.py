# Helper script to collect all dataframes for a dataset and merge them

import os
import re
import pandas as pd

def load_and_concatenate_csvs(directory):
    dataframes = []

    # Function to extract the numeric part of the filename
    def extract_number(filename):
        match = re.search(r'(\d+)\.csv$', filename)
        return int(match.group(1)) if match else -1

    # Get a sorted list of files based on their numeric suffix
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    csv_files.sort(key=extract_number)

    # Collect all DataFrames
    for filename in csv_files:
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    return concatenated_df

directory_path = 'openai'
result_df = load_and_concatenate_csvs(directory_path)
result_df.to_csv("openai/openai_image_distance.csv")
print(result_df)

directory_path = 'vertexai'
result_df = load_and_concatenate_csvs(directory_path)
result_df.to_csv("vertexai/vertexai_image_distance.csv")
print(result_df)

directory_path = 'voyageai'
result_df = load_and_concatenate_csvs(directory_path)
result_df.to_csv("voyageai/voyageai_image_distance.csv")
print(result_df)

directory_path = 'cohere'
result_df = load_and_concatenate_csvs(directory_path)
result_df.to_csv("cohere/cohere_image_distance.csv")
print(result_df)