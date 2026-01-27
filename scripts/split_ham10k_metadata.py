import pandas as pd
from sklearn.model_selection import train_test_split

# Load the metadata
metadata_path = '/usr/local/faststorage/datasets/zahrat/ham10000/HAM10000_metadata.csv'
seed = 42
train_split = 0.8

# # Determine the test ratio based on class size
# def get_test_ratio(class_size):
#     return 0.3 if class_size < 200 else 0.15

df = pd.read_csv(metadata_path)

# Split the df into train, test, and val df
df['split'] = None
unique_lesion_ids = list(df['lesion_id'].unique())
lesion_id_to_dx = df.set_index('lesion_id')['dx'].to_dict()

train_lesion_ids, test_lesion_ids = train_test_split(
    unique_lesion_ids, train_size=train_split, random_state=seed, 
    stratify=[lesion_id_to_dx[lesion_id] for lesion_id in unique_lesion_ids])


df.loc[df['lesion_id'].isin(train_lesion_ids), 'split'] = 'train'
df.loc[df['lesion_id'].isin(test_lesion_ids), 'split'] = 'test'

# Save the new dataframe to a CSV file
output_path = '/usr/local/data/zahrat/workshop/stable-diffusion-finetune/data/ham10k_metadata.csv'
df.to_csv(output_path, index=False)

# Print number of samples in each split for each dx class
print(df.groupby(['dx', 'split']).size())

print(f"Split metadata saved to {output_path}")
