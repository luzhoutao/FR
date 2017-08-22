from skimage.feature import local_binary_pattern
from PIL import Image
import numpy as np

import LBP



print(np.shape(LBP.lbp('../data/face.jpg')))