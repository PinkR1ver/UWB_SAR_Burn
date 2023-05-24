#!/usr/bin/env python3
import numpy as np
from PIL import Image

# Define width and height of image
W, H = 650, 650

# Create solid red image
im = np.random.randint (0,255,size= (H,W),dtype=np.uint16)

'''
# Choose center position
cx = 0 # x coordinate of center
cy = 0 # y coordinate of center

# Create radial alpha/transparency layer. 255 in centre, 0 at edge
Y = np.linspace (-1, 1, H) [None, :]*255
X = np.linspace (-1, 1, W) [:, None]*255
D = np.sqrt ((X-cx)**2 + (Y-cy)**2) # distance matrix
alpha = 255 - np.clip (0,255,D) # transparency matrix

# Push that radial gradient transparency onto red image and save
im.putalpha (Image.fromarray (alpha.astype (np.uint8)))
im.save ('result.png')
'''


result = 0;

for i in range(-100, 101):
    for j in range(-100, 101):
        # Choose center position
        cx = i # x coordinate of center
        cy = j # y coordinate of center

        # Create radial alpha/transparency layer. 255 in centre, 0 at edge
        Y = np.linspace (-1, 1, H) [None, :]*255
        X = np.linspace (-1, 1, W) [:, None]*255
        D = np.sqrt ((X-cx)**2 + (Y-cy)**2) # distance matrix
        alpha = 255 - np.clip (0,255,D) # transparency matrix

        result += abs(alpha)

im.putalpha (Image.fromarray (result.astype (np.uint8)))
im.save ('final.png')