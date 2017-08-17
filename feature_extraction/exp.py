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

        image_mat.append(face_vector)
        labels.append(file)

print('\nFind %d mis-detected faces!' % (count))

# get the traning matrix
image_mat = np.array(image_mat, dtype=np.float32).T
labels = np.array(labels)
print("Collect %d faces!" % (len(labels)))

# run pca and save PC
print("Start PCA ...")

start_time = time.time()
[W_norm, v, mean] = pca(image_mat)
print('Finish in %s seconds!'%(time.time() - start_time))

print("saving...")
np.save('pca/Wnorm', W_norm)
np.save('pca/eigenvalue', v)
np.save('pca/mean', mean)
print('done!')

print("Start LDA ...")

start_time = time.time()
[W, center, classes] = lda(image_mat, labels)
print('Finish in %s seconds!'%(time.time() - start_time))

print('saving...')
np.save('lda/W', W)
np.save('lda/center', center)
np.save('lda/classes', classes)
print('done!')