from skimage.feature import local_binary_pattern
from PIL import Image
import numpy as np

# settings
neighbors = 8 # ?
radius = 2 # ?
method = 'nri_uniform' # ?
regions_num = [6, 10] # ? component-based
face_size = [150, 170]

def lbp(filename):
    '''
    Extract the local binary pattern from image of filename
    :param filename: 
        filename - the path to face image (detected, aligned and cropped)
    :return: 
        feature - matrix where row is
    '''
    image = Image.open(filename).convert('L')
    imarray = np.array(image)

    # divide into regions
    [per_width, per_height] = [int(face_size[0]/regions_num[0]), int(face_size[1]/regions_num[1])]
    regions = [ imarray[r*per_height:(r+1)*per_height, c*per_width:(c+1)*per_width] for c in range(regions_num[0]) for r in range(regions_num[1])]

    patterns = [local_binary_pattern(region, neighbors, radius, method) for region in regions]

    bin_range = int(np.ceil(np.max(patterns)))
    hists = [ np.histogram(pattern.ravel(), bins=bin_range)[0] for pattern in patterns] # ? normalize
    return np.vstack(hists) # row - region , column - label