#THIS MIGHT HELP YOU RESIZE YOUR IMAGES

# Importing Image class from PIL module
from PIL import Image, ImageOps
 
counter = 0

import os

for subdir, dirs, files in os.walk("../originalImages"):
    for file in files:
        counter += 1
        print(file)
    # Opens a image in RGB mode
        im = Image.open("../originalImages/" + file)

        newsize = (50, 50)
        im1 = im.resize(newsize)
        
        im1 = im1.save("images/" + str(counter) + ".png")
