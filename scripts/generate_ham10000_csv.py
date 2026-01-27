import os
import pandas as pd


def set_image_paths(df, image_root_path):
    image_paths = []

    for index, row in df.iterrows():
        image_id = row['image_id']
        
        base1_path = os.path.join(image_root_path, 'HAM10000_images_part_1')
        base2_path = os.path.join(image_root_path, 'HAM10000_images_part_2')
        
        image_path_1 = os.path.join(base1_path, f'{image_id}.jpg')
        image_path_2 = os.path.join(base2_path, f'{image_id}.jpg')

        if os.path.exists(image_path_1):
            image_paths.append(image_path_1)
        elif os.path.exists(image_path_2):
            image_paths.append(image_path_2)
        else:
            print(f'Image {image_id} not found in either path.')

        # Add the image paths to your DataFrame
    df['image_path'] = image_paths
    return df
        
def generate_ham10000_csv(image_root_path, output_csv_path):
    dx_dict = {
        "mel": "melanoma",
        "nv": "melanocytic nevi",
        "bkl": "benign keratosis",
        "bcc": "basal cell carcinoma",
        "akiec": "actinic keratosis",
        "vasc": "vascular lesions",
        "df": "dermatofibroma",
    }

    metadata_path = os.path.join(image_root_path, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    df = set_image_paths(df, image_root_path)

    captions = []
    for _, row in df.iterrows():
        finding = dx_dict.get(row['dx'], "unknown condition")
        caption = f"dermatoscopic image showing {finding}"
        captions.append(caption)

    df['caption'] = captions
    # df['image_path'] = df['image_id'].apply(lambda x: f"{x}.jpg")

    output_df = df[['image_path', 'caption']]
    output_df.to_csv(output_csv_path, index=False, header=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CSV file for HAM10000 dataset")
    parser.add_argument("--image_root_path", type=str, default='/usr/local/faststorage/datasets/zahrat/ham10000', help="Path to the root directory of HAM10000 images")
    parser.add_argument("--output_csv_path", type=str, default='ham10000_caption.csv', help="Path to the output CSV file")

    args = parser.parse_args()
    generate_ham10000_csv(args.image_root_path, args.output_csv_path)
