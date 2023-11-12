import os
import csv

# # Folder path
# root_folder = '/sddata/data/R21_AFs'

# # Create a CSV file for individual image information
# image_csv_file = "individual_images.csv"
# with open(image_csv_file, mode='w', newline='') as image_file:
#     image_writer = csv.writer(image_file)
#     image_writer.writerow(["Patient", "Date", "Eye", "Image"])

#     for patient_folder in os.listdir(root_folder):
#         for date_folder in os.listdir(os.path.join(root_folder, patient_folder)):
#             for eye_folder in os.listdir(os.path.join(root_folder, patient_folder, date_folder)):
#                 for image_file in os.listdir(os.path.join(root_folder, patient_folder, date_folder, eye_folder)):
#                     patient = patient_folder
#                     date = date_folder
#                     eye = eye_folder
#                     image = os.path.join(root_folder, patient, date, eye, image_file)
#                     image_writer.writerow([patient, date, eye, image])

# # Create a CSV file for pairwise combinations of images within the same patient and the same eye across different dates
# pairwise_csv_file = "pairwise_images.csv"
# with open(pairwise_csv_file, mode='w', newline='') as pairwise_file:
#     pairwise_writer = csv.writer(pairwise_file)
#     pairwise_writer.writerow(["Patient", "Date1", "Date2", "Eye", "Image1", "Image2"])

#     for patient_folder in os.listdir(root_folder):
#         dates = os.listdir(os.path.join(root_folder, patient_folder))
#         for i in range(len(dates)):
#             for j in range(i+1, len(dates)):
#                 date1 = dates[i]
#                 date2 = dates[j]
#                 for eye_folder in os.listdir(os.path.join(root_folder, patient_folder, date1)):
#                     if eye_folder in os.listdir(os.path.join(root_folder, patient_folder, date2)):
#                         for image_file1 in os.listdir(os.path.join(root_folder, patient_folder, date1, eye_folder)):
#                             for image_file2 in os.listdir(os.path.join(root_folder, patient_folder, date2, eye_folder)):
#                                 image1 = os.path.join(root_folder, patient_folder, date1, eye_folder, image_file1)
#                                 image2 = os.path.join(root_folder, patient_folder, date2, eye_folder, image_file2)
#                                 pairwise_writer.writerow([patient_folder, date1, date2, eye_folder, image1, image2])

import pandas as pd

# Load the CSV file into a DataFrame
csv_file = '/sddata/data/R21_AFs/pairwise_images.csv'
df = pd.read_csv(csv_file)

# Define the function to map image paths to vessel folder paths
def map_image_to_folder(image_path):
    basename = os.path.basename(image_path)
    print(basename)
    # Replace this with the actual path to your vessel folder
    vessel_folder_path = '/sddata/data/R21_AFs/soft_vessel_masks'
    return f'{vessel_folder_path}/{basename}'

# Apply the mapping function to each column and create new columns
df['VesselImage1Path'] = df['Image1'].apply(map_image_to_folder)
df['VesselImage2Path'] = df['Image2'].apply(map_image_to_folder)

# Save the updated DataFrame to a new CSV file
output_csv_file = '/sddata/data/R21_AFs/pairwise_images_wmasks.csv'
df.to_csv(output_csv_file, index=False)

print(f"CSV file with vessel paths has been saved to '{output_csv_file}'.")
