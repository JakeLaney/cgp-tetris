
import numpy as np
from PIL import Image
import matplotlib
import igen

DOG_LABEL = 5
from functions import FUNCTIONS

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

dict = unpickle('cifar-10-batches-py/data_batch_1')
labels = np.array(dict['labels'])
data = dict['data']

dogImages = data[np.where(labels == DOG_LABEL)]

igen.ImageGenerator().doggos(dogImages)
