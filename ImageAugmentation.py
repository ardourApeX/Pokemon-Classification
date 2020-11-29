from pathlib import Path
from keras_preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore") # To avoid any kind of warning

def Image_Augment(Images, label):
	
	print(Images.shape)
	datagen = image.ImageDataGenerator(
	    rotation_range = 40,
	    width_shift_range = 0.2,
	    height_shift_range = 0.2,
	    shear_range = 0.2,
	    zoom_range = 0.2,
	    horizontal_flip = True,
	    fill_mode = "nearest"
	)

	# Generating 410 more images of each pokemon

	i = 0
	for batch in datagen.flow(Images, batch_size = 1, save_to_dir = "dataset/" + label, save_prefix = label, save_format = "jpg"):
	    
	    
	    i += 1
	    if i == 430:
	        break
	    

def load_image():
	path = Path("./dataset/")
	dirs = path.glob("*")

	
	for folder in dirs:
		label = str(folder).split("\\")[-1]
		image_data = []
		for Image in folder.glob("*.jpg"):
			img = image.load_img(Image, target_size = (1000, 1000))
			img = image.img_to_array(img)
			image_data.append(img)
		image_data = np.array(image_data)
		Image_Augment(image_data, label)

load_image()