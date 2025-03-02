import kaggle
import os
dataset = 'kritikseth/fruit-and-vegetable-image-recognition'  
path = 'KaggleData' 
kaggle.api.dataset_download_files(dataset, path=path, unzip=True)
print("Download complete")