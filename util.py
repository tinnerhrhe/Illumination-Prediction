import numpy as np
#from scipy.misc import imresize, imsave
import torch
from os import listdir
from imageio import imsave
'''
def load_image(filepath):
    image = imread(filepath)
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    min = image.min()
    max = image.max()
    image = torch.FloatTensor(image.size()).copy_(image)
    image.add_(-min).mul_(1.0 / (max - min))
    image = image.mul_(2).add_(-1)
    return image
'''
def load_data(data):
    min = data.min()
    max = data.max()
    data = torch.FloatTensor(data.size()).copy_(data)
    data.add_(-min).mul_(1.0 / (max - min))
    data = data.mul_(2).add_(-1)
    return data

def save_image(image, filename):

    image = image.add_(1).div_(2)
    image = image.numpy()
    image *= 255.0
    image = image.clip(0, 255)
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.uint8)
    imsave(filename, image)
    print ("Image saved as {}".format(filename))

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".exr"])
def is_exr(filename):
    return any(filename.endswith(extension) for extension in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
