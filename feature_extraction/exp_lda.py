import os
import sys
import time
# image operation
from PIL import Image
import numpy as np
# utils
from PCA import pca
from LDA import lda
# detection and alignment
import detection

# settings
data_root = "./data"
detector = detection.Detector()
pca_w_path = "./Wnorm.npy"
result_root = "./result/w"


'''
This file is used to extract LDA feature of faces.
Data source: http://vintage.winklerbros.net/facescrub.html
First, apply the PCA reduction to the raw data; then use it as the input data of LDA

faces: 258 people, 37813 faces
training time: ______ seconds (on Ubuntu 14.04, Intel core i7 3.40GHz * 8)

Input face size: 150 * 170
90% eigenvector: ___
'''

pca_w = np.load(pca_w_path)

count = 0
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
        faces = detector.detect(image)

        if len(faces) == 0:
            print('.', end='')
            count += 1
            sys.stdout.flush()
            continue

        face = np.array(faces[0].convert(mode='L'))

        face_vector = np.reshape(face, -1) # reshape to a row-vector
        reduced_face_vector = np.dot(face_vector, pca_w)

        image_mat.append(reduced_face_vector)
        labels.append(file)

print('\nFind %d mis-detected faces!' % (count))

# get the traning matrix
image_mat = np.array(image_mat, dtype=np.float32).T
labels = np.array(labels)
print("Collect %d faces!" % (len(labels)))


print("Start LDA ...")

start_time = time.time()
[W, center, classes] = lda(image_mat, labels)
print('Finish in %s seconds!'%(time.time() - start_time))

print('saving...')
np.save(os.path.join(result_root, 'W'), W)
np.save(os.path.join(result_root, 'center'), center)
np.save(os.path.join(result_root, 'classes'), classes)
print('done!')