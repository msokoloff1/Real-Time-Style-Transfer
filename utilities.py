import tensorflow as tf
import numpy as np
from scipy import misc
from PIL import Image



def mse(preds, correct):
    return (tf.reduce_sum(tf.pow(tf.sub(preds,correct), 2)) / (2))

def euclidean(preds, correct):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.sub(preds, correct))))*(0.5)




def loadImage(path, imageShape, display=False):
    subjectImage = misc.imresize(misc.imread(path), imageShape) / 255
    if(display):
        showImage(subjectImage, imageShape)
    reshaped = (1,) + imageShape
    return subjectImage.reshape(reshaped)


def showImage(image, shape):
    img = np.clip(image,0, 1)*255
    img = Image.fromarray((img.reshape(shape)).astype('uint8'), 'RGB')
    img.show()