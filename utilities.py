import tensorflow as tf
import numpy as np
from scipy import misc
from PIL import Image

import random

def mse(preds, correct):
    return (tf.reduce_sum(tf.pow(tf.sub(preds,correct), 2)) / (2))

def euclidean(preds, correct):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.sub(preds, correct))))*(0.5)



def loadImage(path, imageShape, display=False):
    subjectImage = misc.imresize(misc.imread(path), imageShape) / 255
    if(display):
        showImage(subjectImage, imageShape)
    reshaped = (1,) + imageShape
    try:
        reshaped2 = subjectImage.reshape(reshaped)
        return reshaped2
    except:
        print("ERROR")
        return np.array(loadImage('../neural_art/images/testingContent1.jpg', imageShape)) 


def showImage(image, shape, name):
    try:
        img = np.clip(image,0, 1) * 255
        img = Image.fromarray((img.reshape(shape)).astype('uint8'), 'RGB')
        img.show()
        img.save("./images/"+name+".jpg", "JPEG")
    except:
        print("Unable to display")


def convolution(inputs,filterX, filterY, numFilterOutputs, strideX, strideY,layerNum,currentlyTraining, batchNormalize = True, isPadded=False, padding="VALID", activation=tf.nn.relu):
    if(isPadded):
        padSize = 16
        inputs = tf.pad(inputs, [[0, 0], [padSize, padSize], [padSize, padSize], [0, 0]], "CONSTANT")

    filterZ = inputs.get_shape()[-1]

    filter = tf.Variable(tf.truncated_normal([filterX, filterY, int(filterZ), numFilterOutputs], stddev=0.01), name = str("filter_"+layerNum))
    if(batchNormalize):
        return activation(batch_normalization(tf.nn.conv2d(inputs, filter, strides=[1, strideX, strideY, 1], padding=padding), currentlyTraining))
    else:
        bias = tf.Variable(tf.constant(0.01, shape=[numFilterOutputs]), name=str("bias_" + layerNum))
        return activation(tf.nn.conv2d(inputs, filter, strides=[1, strideX, strideY, 1], padding=padding) + bias)


