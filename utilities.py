
##Imports##
from scipy import misc
from PIL import Image
import utilities as utils
import tensorflow as tf
from functools import reduce
import numpy as np
import vggNet
###########################

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
        a=subjectImage.reshape(reshaped)
        return a
    except:
        print("Error!")
        print(reshaped)
        print(imageShape)
        print(subjectImage.shape)
        exit(0)



def showImage(image, shape):
    img = np.clip(image,0, 1)*255
    img = Image.fromarray((img.reshape(shape)).astype('uint8'), 'RGB')
    img.show()

##Global Options##
contentPath        = '../real_time_style_transfer/images/testingContent.jpg'
stylePath          = '../real_time_style_transfer/images/testingArt.jpg'
contentLayer       = 'conv4_2'
styleLayers        = ['conv1_1','conv2_1','conv3_1', 'conv4_1', 'conv5_1']
styleWeights       = [0.2      ,0.2      , 0.2     , 0.2      , 0.2      ]
styleData          = {}
styleBalanceData   = {}
contentData        = None
errorMetricContent = utils.mse #utils.euclidean
errorMetricStyle   = utils.mse
normalizeContent   = True #Inversion paper says to use this
normalizeStyle     = False #No mention of style in inversion paper
imageShape         = (int(256),int(256),3)
#TODO : Add a real value for sigma. Should be the average euclidean norm of the vgg training images. This also requires an option for the model to change whether or not sigma is multipled by the input image
sigma = 1.0 #<- if sigma is one then it doesnt need to be included in the vgg net (because the multiplicative identity)
beta = 2.0
alpha = 6.0
a = 0.01
B = 120.0 #Pixel min/max encourages pixels to be in the range of [-B, +B]

alphaNormLossWeight = 0.0001
TVNormLossWeight    = 1.5
styleLossWeight     = 0.0001
contentLossWeight   = 0.05

learningRate  = 0.025
numIters      = 501
showEveryN   = 500
##################


with tf.Session() as sess:
    inputTensor = tf.placeholder(tf.float32, shape=[None,imageShape[0], imageShape[1], imageShape[2]])
    contentImage = np.array(utils.loadImage(contentPath, imageShape))
    styleImage = utils.loadImage(stylePath, imageShape)
    styleBalanceImage = utils.loadImage(contentPath, imageShape)
    model =vggNet.Vgg19()
    model.build(inputTensor, imageShape)
    contentData = eval('sess.run(model.' + contentLayer + ',feed_dict={inputTensor:contentImage})')
    for styleLayer in styleLayers:
        styleData[styleLayer] = np.array(eval('sess.run(model.' + styleLayer + ',feed_dict={inputTensor:styleImage})'))





def buildStyleLoss(model):
    totalStyleLoss = []
    for index, styleLayer in enumerate(styleLayers):
        normalizingConstant = 1
        if (normalizeStyle):
            normalizingConstant = (reduce(lambda x, y: x + y, (styleData[styleLayer] ** 2)) ** (0.5))

        styleLayerVar = tf.Variable(styleData[styleLayer])
        correctGrams  = buildGramMatrix(styleLayerVar)
        tensorGrams   = buildGramMatrix(eval('model.'+styleLayer))
        _, dimX, dimY, num_filters = styleLayerVar.get_shape()
        denominator   =(2*normalizingConstant)*((float(int(dimX))*float(int(dimY)))**2)*(float(int(num_filters))**2)
        error         = tf.reduce_sum(errorMetricStyle(tensorGrams, correctGrams))
        totalStyleLoss.append((tf.div(error,denominator)))


    #styleLoss = (reduce(lambda x, y: x + y, totalStyleLoss))
    styleLoss = tf.reduce_sum(totalStyleLoss)
    return styleLoss


#def buildGramMatrix(layer):
#    _, dimX, dimY, num_filters = layer.get_shape()
#    vectorized_maps = tf.reshape(layer, [int(dimX) * int(dimY), int(num_filters)])

#    if int(dimX) * int(dimY) > int(num_filters):
#        return tf.matmul(vectorized_maps, vectorized_maps, transpose_a=True)
#    else:
#        return tf.matmul(vectorized_maps, vectorized_maps, transpose_b=True)


def buildGramMatrix(v):

    dim1 = v.get_shape().as_list()
    v = tf.reshape(v, [-1,dim1[1]*dim1[2],dim1[3]])
    dim = v.get_shape().as_list()

    if dim1[1] * dim1[2] > dim1[3]:
        return tf.batch_matmul(tf.reshape(v,[-1, dim[2],dim[1]]),v)
    else:
        return tf.batch_matmul(v,tf.reshape(v,[-1, dim[2], dim[1]]))



def buildContentLoss(model, correctAnswer=contentData):

    normalizingConstant = 1
    if(normalizeContent):

        normalizingConstant = np.sum(  (correctAnswer**2))**(0.5)

    print("Normalizing Constant : %g"%(normalizingConstant))
    contentLoss = (eval('errorMetricContent(model.' + contentLayer + ', correctAnswer)') / normalizingConstant)
    return tf.reduce_sum(contentLoss)


def buildAlphaNorm(model):
    adjustedImage = model.bgr
    return tf.reduce_sum(tf.pow(adjustedImage, alpha))



def buildTVNorm(model):
    adjustedImage = model.bgr


    yPlusOne = tf.slice(adjustedImage, [0,0,1,0], [1,imageShape[0],(imageShape[1]-1),imageShape[2]])
    xPlusOne = tf.slice(adjustedImage, [0,1,0,0], [1,(imageShape[0]-1),imageShape[1],imageShape[2]])

    inputNoiseYadj = tf.slice(adjustedImage,[0,0,0,0],[1,imageShape[0],(imageShape[1]-1),imageShape[2]])
    inputNoiseXadj = tf.slice(adjustedImage, [0,0,0,0], [1,(imageShape[0]-1),imageShape[1],imageShape[2]])


    lambdaBeta = (sigma**beta) / (imageShape[0]*imageShape[1]*((a*B)**beta))
    error1 = tf.slice(tf.square(yPlusOne-inputNoiseYadj), [0,0,0,0], [1,(imageShape[0]-1),(imageShape[1]-1), imageShape[2]])
    error2 = tf.slice(tf.square(xPlusOne-inputNoiseXadj), [0,0,0,0], [1,(imageShape[0]-1),(imageShape[1]-1), imageShape[2]])

    return lambdaBeta*tf.reduce_sum( tf.pow((error1+error2),(beta/2) ))



def totalLoss(model, contentModel):
    errorComponents =[buildStyleLoss(model), buildContentLoss(contentModel), buildTVNorm(model)]
    LossWeights = [styleLossWeight, contentLossWeight,TVNormLossWeight]
    loss =[]
    for error, weights in zip(errorComponents, LossWeights):
        loss.append(error*weights)

    reducedLoss = reduce(lambda x,y: x+y, loss)
    return reducedLoss












def batch_normalization(inputs, currentlyTraining, decay = 0.999, epsilon = 1e-3):
    #https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
    #^Thanks
    ## inputs.get_shape()[-1] returns the number of filters (we want to normalize each of them)
    ### When you have more time, make this a recursive function that takes the true average instead of exponential moving avg
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    return tf.cond(currentlyTraining,lambda : train_BN(inputs,pop_mean,pop_var,scale), lambda : test_BN(inputs, pop_mean, pop_var, beta, scale))



def train_BN(inputs,pop_mean,pop_var,scale, decay=0.999,epsilon=1e-3):
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2], name='moments')
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
    with tf.control_dependencies([train_mean, train_var]):
        # Control dependencies ensures that train_mean, and train_var are up to date
        # Return the values needed to recursively update the avg batch_mean and avg batch_var
        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)


def test_BN(inputs, pop_mean, pop_var, beta, scale, epsilon=1e-3):
    return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


def convolution(inputs,filterX, filterY, numFilterOutputs, strideX, strideY,layerNum,currentlyTraining, batchNormalize = True, isPadded=False, padding="VALID", activation=tf.nn.relu):
    if(isPadded):
        inputs = tf.pad(inputs, [[0, 0], [40, 40], [40, 40], [0, 0]], "CONSTANT")

    filterZ = inputs.get_shape()[-1]

    filter = tf.Variable(tf.truncated_normal([filterX, filterY, int(filterZ), numFilterOutputs], stddev=0.01), name = str("filter_"+layerNum))
    if(batchNormalize):
        return activation(batch_normalization(tf.nn.conv2d(inputs, filter, strides=[1, strideX, strideY, 1], padding=padding), currentlyTraining))
    else:
        bias = tf.Variable(tf.constant(0.01, shape=[numFilterOutputs]), name=str("bias_" + layerNum))
        return activation(tf.nn.conv2d(inputs, filter, strides=[1, strideX, strideY, 1], padding=padding) + bias)



def residualBlock(input, currentlyTraining, layerNum, batchSize):
    conv1_BN = convolution(input, 3, 3, 128, 1, 1,layerNum,currentlyTraining, batchNormalize=True, padding="VALID")
    conv2 = convolution(conv1_BN, 3, 3, 128, 1, 1, layerNum, currentlyTraining, batchNormalize=False, padding="VALID")
    conv2_BN = batch_normalization(conv2, currentlyTraining)
    _, yDim, xDim, numChannels = conv2_BN.get_shape()

    cropped = tf.slice(input, [0,1,1,0], [batchSize,int(yDim), int(xDim),int(numChannels)])
    return cropped + conv2_BN

def deconvolution(inputs, batch_size, outputDimXY,outputDimZ, currentlyTraining):
    strides = [1, 2, 2, 1]
    filterZ = int(inputs.get_shape()[-1])
    w = tf.constant(0.1, shape=[3, 3, outputDimZ, filterZ])
    h1 = tf.nn.conv2d_transpose(inputs, w, output_shape=[batch_size, outputDimXY,outputDimXY,outputDimZ], strides=strides, padding='SAME')
    h1.set_shape([None, outputDimXY, outputDimXY, outputDimZ])
    return  tf.nn.relu(batch_normalization(h1, currentlyTraining))






