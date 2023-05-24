import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = os.path.dirname(__file__)

# Load the data from the CSV file
data = pd.read_csv(os.path.join(base_path, 'correction_Param.csv'))

# Extract the feature values and target values
x1 = data['azimuth']
x2 = data['elevation']
y = data['correction_param']

# Create a scatter plot to visualize the relationship between x1, x2, and y
plt.figure(figsize=(8, 6))
plt.scatter(x1, y, label='azimuth')
plt.scatter(x2, y, label='elevation')
plt.xlabel('azimuth and elevation')
plt.ylabel('y')
plt.legend()
plt.title('Relationship between azimuth, elevation, and y')
plt.show()
