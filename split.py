import splitfolders
import os

dr = r"D:\\proj4\\sign language final copy\\Dataset"

# Check if the folder exists
if not os.path.exists(dr):
    raise ValueError(f"The folder {dr} does not exist. Please check the path.")

# Split the dataset
splitfolders.ratio(dr, output="sign language 3.0", ratio=(0.8, 0.2))
