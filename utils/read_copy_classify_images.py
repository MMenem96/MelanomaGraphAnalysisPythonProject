import pandas as pd
import shutil
import os

# Load the CSV file
csv_file = "/Users/mmoniem96/Downloads/ISIC-images/metadata.csv"
df = pd.read_csv(csv_file) 

# Path to the image folder
image_folder = "/Users/mmoniem96/Downloads/ISIC-images/"

# Paths to the destination folders
benign_folder = "data/sk"
malignant_folder = "data/bcc"

# Create the destination folders if they don't exist
os.makedirs(benign_folder, exist_ok=True)
os.makedirs(malignant_folder, exist_ok=True)

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    image_filename = row['isic_id'] + '.jpg' 
    diagnosis = row['diagnosis_1'] 
    
    # Define source and destination paths
    source_image_path = os.path.join(image_folder, image_filename)
  
    
    # Depending on the diagnosis, move the image to the appropriate folder
    if diagnosis == 'Benign':  # Adjust as per your dataset
        destination_path = os.path.join(benign_folder, image_filename)
    elif diagnosis == 'Malignant':  # Adjust as per your dataset
        destination_path = os.path.join(malignant_folder, image_filename)
    else:
        continue  # Skip if diagnosis is not recognized
    
    # Copy the image to the destination folder
    if os.path.exists(source_image_path):
        shutil.copy(source_image_path, destination_path)
    else:
        print(f"Image {image_filename} not found!")
        print(f"source_image_path {source_image_path} not found!")
        print(f"destination_path {destination_path} not found!")
