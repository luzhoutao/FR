import os
# image operation
import cv2
import numpy as np
# utils
from PCA import pca

# settings
data_root = "./lfw"

# read labeled image
image_mat = []
labels = []
for file in os.listdir(data_root):
    # on mac os
    if file == '.DS_Store':
        continue

    for imagefile in os.listdir(os.path.join(data_root, file)):
        image = cv2.imread(os.path.join(data_root, file, imagefile), 0) # grayscale
        image = np.reshape(image, -1) # reshape to a row-vector

        image_mat.append(image)
        labels.append(file)

image_mat = np.array(image_mat, dtype=np.float32).T
labels = np.array(labels)

[W_norm, v, mean] = pca(image_mat)
print("saving...")
np.save('pca/Wnorm', W_norm)
np.save('pca/eigenvalue', v)
np.save('pca/mean', mean)
print('done!')
