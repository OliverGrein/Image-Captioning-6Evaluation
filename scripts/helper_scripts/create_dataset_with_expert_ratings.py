# Helper Script to extract the expert and crowdfunded ratings from the txt file

import pandas as pd

# Set paths to expert and crowdfunded ratings
path = "image_distance_final/old/voyageai_image_distance.csv"
path2 = "image_distance_final/voyageai_image_distance.csv"

image_df = pd.read_csv(path, index_col=0)

# Set paths to expert and crowdfunded ratings
expert_annotations_path = 'Image-Captioning-6Evaluation/data/flickr-8k/raw/flickr-8k/ExpertAnnotations.txt'
crowdflower_annotations_path = 'Image-Captioning-6Evaluation/data/flickr-8k/raw/flickr-8k/CrowdFlowerAnnotations.txt'

# Read the expert annotations
expert_df = pd.read_csv(expert_annotations_path, delimiter='\t', header=None, 
                        names=['image_caption_id', 'caption_id', 'expert_score1', 'expert_score2', 'expert_score3'])

# Extract the image id from the combined image_caption_id column
expert_df['image'] = expert_df['image_caption_id'].apply(lambda x: x.split('#')[0])

# Calculate the average and median expert score for each image-caption pair
expert_df['avg_expert_score'] = expert_df[['expert_score1', 'expert_score2', 'expert_score3']].mean(axis=1)
expert_df['median_expert_score'] = expert_df[['expert_score1', 'expert_score2', 'expert_score3']].median(axis=1)
avg_expert_scores = expert_df.groupby('image')['avg_expert_score'].mean().reset_index()
median_expert_scores = expert_df.groupby('image')['median_expert_score'].median().reset_index()
expert_scores = pd.merge(avg_expert_scores, median_expert_scores, on='image')

# Read the CrowdFlower annotations
crowdflower_df = pd.read_csv(crowdflower_annotations_path, delimiter='\t', header=None, 
                             names=['image_caption_id', 'caption_id', 'percent_yes', 'num_yes', 'num_no'])

# Extract the image id from the combined image_caption_id column
crowdflower_df['image'] = crowdflower_df['image_caption_id'].apply(lambda x: x.split('#')[0])

# Calculate the average and median expert score for each image-caption pair
avg_crowdflower_scores = crowdflower_df.groupby('image')['percent_yes'].mean().reset_index()
median_crowdflower_scores = crowdflower_df.groupby('image')['percent_yes'].median().reset_index()
crowdflower_scores = pd.merge(avg_crowdflower_scores, median_crowdflower_scores, on='image')

# Combine df to one df holding all data
merged_scores_df = pd.merge(expert_scores, crowdflower_scores, on='image', how='outer')
final_df = pd.merge(image_df, merged_scores_df, on='image', how='left')

# Set NaNs to -1
final_df = final_df.fillna(-1)

# Save data
final_df.to_csv(path2, index=False)
