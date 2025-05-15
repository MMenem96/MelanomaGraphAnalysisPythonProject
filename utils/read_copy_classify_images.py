import pandas as pd
import shutil
import os

# Load the CSV file
csv_file = "/Users/mmoniem96/Desktop/Work/Master/Images/HAM100000-Images/metadata.csv"
df = pd.read_csv(csv_file)

# Path to the image folder
image_folder = "/Users/mmoniem96/Desktop/Work/Master/Images/HAM100000-Images/"

# Paths to the destination folders
benign_folder = "data/sk"
malignant_folder = "data/bcc"

# Create the destination folders if they don't exist
os.makedirs(benign_folder, exist_ok=True)
os.makedirs(malignant_folder, exist_ok=True)

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    image_filename = row['isic_id'] + '.jpg'
    diagnosis = str(row['diagnosis_3']).lower()  # Ensure it's a string
    
    source_image_path = os.path.join(image_folder, image_filename)
    
    # Depending on the diagnosis, move the image to the appropriate folder
    if 'keratosis' in diagnosis:
        destination_path = os.path.join(benign_folder, image_filename)
    elif 'basal cell carcinoma' in diagnosis:
        destination_path = os.path.join(malignant_folder, image_filename)
    else:
        continue  # Skip if diagnosis is not recognized
    
    # Copy the image to the destination folder
    if os.path.exists(source_image_path):
        shutil.copy(source_image_path, destination_path)
    else:
        print(f"Image {image_filename} not found!")
