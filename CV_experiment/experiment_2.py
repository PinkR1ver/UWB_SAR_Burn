#!/usr/bin/env python3
import numpy as np
import cv2

# Define width and height of array
W, H = 100, 100

# Create an empty array
arr = np.empty ((H,W))

# Generate two coordinate matrices
X, Y = np.meshgrid (np.arange (W), np.arange (H))

# Calculate the distance of each element to the center
D = np.sqrt ((X-W/2)**2 + (Y-H/2)**2)

# Clip the distance to the range [0, W/2]
D = np.clip (D, 0, W/2)

# Subtract the distance from the maximum distance
arr = W/2 - D

# Print the array
print (arr)

cv2.imshow('image', arr)
cv2.waitKey(0)
cv2.destroyAllWindows