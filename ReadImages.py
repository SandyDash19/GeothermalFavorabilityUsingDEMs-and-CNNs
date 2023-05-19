import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Open training data
file = h5py.File("../mp02files/DEM_train.h5", "r+")
X_train = np.array(file["/images"])
y_train = np.array(file["/meta"])
file.close()

# Open test data
file = h5py.File("../mp02files/DEM_test_features.h5", "r+")
X_test = np.array(file["/images"])
file.close()

print (X_train.shape, y_train.shape, X_test.shape)
# display an arbitrary image
# Images that should be cropped. 
# 6, 36, 28, 19, 76, 85, 104, 108
# 199, 208, 221
idx = 108
im = X_train[idx, :, :]
plt.imshow(im)
plt.show()

"""
# Delete images which has very little information
indices_to_delete = [6,76,91,104,108,156]

X_train = np.delete(X_train, indices_to_delete, axis=0)
y_train = np.delete(y_train, indices_to_delete, axis=0)

print (X_train.shape, y_train.shape, X_test.shape)
"""

# To display images in two separate grids, split the images 
# into two groups
group1 = X_train[:111]
y_train1 = y_train[:111]
group2 = X_train[111:]
y_train2 = y_train[111:]

indices_to_delete = [6,76,91,104,108,156]

print (group1.shape)
#Plot half of the images
#n_images = (np.ceil(n_images / 2)).astype(int)
grid_size1 = int(np.ceil(np.sqrt(group1.shape[0])))

# Create a new figure
plt.figure(figsize=(50, 50))

# Loop over the images
# Images are filled col wise meaning fill all columns and then 
# move to the next row.
# X_train[0] is on row[0], col[0]
# X_train[1] is on row[0], col[1]
for i in range(len(group1)):
    # Create a subplot for each image
    ax = plt.subplot(grid_size1, grid_size1, i + 1)
    # Remove axes for clarity
    ax.axis('off')    

    # Add a red rectangle 
    if i in indices_to_delete:
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='r', facecolor='none', transform=ax.transAxes)
        ax.add_patch(rect)
    # Show the image
    ax.imshow(group1[i, :, :])
    # Add label on top of the image
    ax.set_title(str(y_train1[i]), fontsize=10, pad=5)
    # Print the index below the image
    ax.text(0.5, -0.2, str(i), transform=ax.transAxes,
            fontsize=8, color='red', ha='center')

# Adjust the spacing between rows
plt.subplots_adjust(hspace=0.6)
plt.show()


grid_size2 = int(np.ceil(np.sqrt(group2.shape[0])))
# Loop over the images
for i in range(len(group2)):
    # Create a subplot for each image
    ax = plt.subplot(grid_size2, grid_size2, i + 1)
    # Remove axes for clarity
    ax.axis('off')    

    # Add a red rectangle 
    if i+112 in indices_to_delete:
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='r', facecolor='none', transform=ax.transAxes)
        ax.add_patch(rect)

    # Show the image
    ax.imshow(group2[i, :, :])
    # Add label on top of the image
    ax.set_title(str(y_train2[i]), fontsize=10, pad=5)
    # Print the index below the image
    ax.text(0.5, -0.2, str(i+112), transform=ax.transAxes,
            fontsize=8, color='red', ha='center')

# Adjust the spacing between rows
plt.subplots_adjust(hspace=0.6)
# Show the plot
plt.show()

# Plot the test set 
test_size = X_test.shape[0]
grid_size3 = int(np.ceil(np.sqrt(test_size)))

for i in range(test_size):
    # Create a subplot for each image
    ax = plt.subplot(grid_size3, grid_size3, i + 1)
    # Remove axes for clarity
    ax.axis('off')    
    # Show the image
    ax.imshow(X_test[i, :, :])

    # Print the index below the image
    ax.text(0.5, -0.2, str(i), transform=ax.transAxes,
            fontsize=8, color='red', ha='center')

# Adjust the spacing between rows
plt.subplots_adjust(hspace=0.6)
plt.show()

# Save y_train in a csv
#np.savetxt('Labels.csv', y_train, delimiter=',')




