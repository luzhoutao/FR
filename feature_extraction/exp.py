import os
# image operation
from PIL import Image
import numpy as np
# utils
from PCA import pca
# detection and alignment
import detection

# settings
data_root = "./data"
detector = detection.Detector()

image_mat = []
labels = []
for file in os.listdir(data_root):
    # on mac os
    if file == '.DS_Store':
        continue

    for imagefile in os.listdir(os.path.join(data_root, file)):
        # read image
        image = Image.open(os.path.join(data_root, file, imagefile)).convert(mode='L')

        # detect face and landmark (must have one)
        face = np.array(detector.detect(image)[0])

        face_vector = np.reshape(face, -1) # reshape to a row-vector

        image_mat.append(face_vector)
        labels.append(file)
# get the traning matrix
image_mat = np.array(image_mat, dtype=np.float32).T
labels = np.array(labels)

# run pca and save PC
[W_norm, v, mean] = pca(image_mat)
print("saving...")
np.save('pca/Wnorm', W_norm)
np.save('pca/eigenvalue', v)
np.save('pca/mean', mean)
print('done!')