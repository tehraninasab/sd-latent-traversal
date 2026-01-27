import os 
from os.path import join
import pandas as pd


df = pd.read_csv('../data/chexpert/chexpert_test.csv')

pe_df = df[
    (df['Cardiomegaly'] == 0) &
    (df['Lung Opacity'] == 0) &
    (df['Edema'] == 0) &
    (df['Pneumonia'] == 0) &
    (df['Pneumothorax'] == 0) &
    (df['Pleural Effusion'] == 1) & 
    (df['Support Devices'] == 0)
    ]

sd_df = df[
    (df['Cardiomegaly'] == 0) &
    (df['Lung Opacity'] == 0) &
    (df['Edema'] == 0) &
    (df['Pneumonia'] == 0) &
    (df['Pneumothorax'] == 0) &
    (df['Pleural Effusion'] == 0) & 
    (df['Support Devices'] == 1)
    ]

source_dir = "/usr/local/faststorage/datasets/"
dest_dir = "/usr/local/data/zahrat/workshop/dent/data/chexpert/subgroup_test_images"

print(len(pe_df), len(sd_df))
for i, row in pe_df.iterrows():
    
    source = os.path.join(source_dir, row['Path'].replace('CheXpert-v1.0', 'CheXpert-v1.0_512x512'))
    filename = os.path.basename(source)
    dest = os.path.join(dest_dir, 'pe', row['Path'].replace('CheXpert-v1.0', 'CheXpert-v1.0_512x512'))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.symlink(source, dest)


for i, row in sd_df.iterrows():
    
    source = os.path.join(source_dir, row['Path'].replace('CheXpert-v1.0', 'CheXpert-v1.0_512x512'))
    filename = os.path.basename(source)
    dest = os.path.join(dest_dir, 'sd', row['Path'].replace('CheXpert-v1.0', 'CheXpert-v1.0_512x512'))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.symlink(source, dest)

print("Done")


source_dir = '/usr/local/data/zahrat/workshop/dent/scripts/outputs/chexpert/swap'
dest_dir = '/usr/local/data/zahrat/workshop/dent/data/chexpert/subgroup_synthetic_images'

patient_ids = list(range(400, 821))
# Generated Images
for pid in patient_ids:
    for lambda_t_star in range(5, 31, 5):
        sd_img_path = join(source_dir, f'seed{pid}/normal-support_devices/lambda_t_star{lambda_t_star}/style.png')
        dst_img_path = join(dest_dir, 'sd', f'seed{pid}', f'lambda_t_star{lambda_t_star}.png')
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        os.symlink(sd_img_path, dst_img_path)
        

        pe_img_path = join(source_dir, f'seed{pid}/normal-pleural_effusion/lambda_t_star{lambda_t_star}/style.png')
        dst_img_path = join(dest_dir, 'pe', f'seed{pid}', f'lambda_t_star{lambda_t_star}.png')
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        os.symlink(pe_img_path, dst_img_path)
        
print("Done")
