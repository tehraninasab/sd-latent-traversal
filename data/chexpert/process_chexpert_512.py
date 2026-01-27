import os
from PIL import Image
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

# Define the base path
base_path = '/usr/local/faststorage/datasets/'

# Function to resize an image and save it
def resize_and_save(row):
    try:
        # Extract the file path and dataset name
        row_index, row_data = row
        img_path = os.path.join(base_path, row_data['Path'])
        
        # Open and resize the image
        img = Image.open(img_path)
        img = img.resize((512, 512), Image.LANCZOS)
        
        # Construct the new path and save the resized image
        new_path = os.path.join(base_path, 'CheXpert-v1.0_512x512', '/'.join(row_data['Path'].split('/')[1:]))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        img.save(new_path)
        
    except Exception as e:
        print(f"Error processing file {row_data['Path']}: {e}")

# Function to process a single dataset (train, valid, test)
def process_dataset(name, df):
    print(f"Processing {name} dataset with {len(df)} images...")
    
    # Use multiprocessing to process images in parallel
    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(resize_and_save, df.iterrows()), total=len(df)))

train_df = pd.read_csv('/usr/local/data/amarkr/lgcig/data/chexpert/chexpert_train.csv')
val_df = pd.read_csv('/usr/local/data/amarkr/lgcig/data/chexpert/chexpert_val.csv')
test_df = pd.read_csv('/usr/local/data/amarkr/lgcig/data/chexpert/chexpert_test.csv')

# Process each dataset
for name, dataset in zip(['train', 'valid', 'test'], [train_df, val_df, test_df]):
    print('Number of cpu cores:', os.cpu_count())
    process_dataset(name, dataset)

