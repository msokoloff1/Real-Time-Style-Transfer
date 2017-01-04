##Imports##
import net
import utilities as utils
import tensorflow as tf
from functools import reduce
import numpy as np
import time
###########################

##Global Options##
contentPath        = '../neural_art/images/testingContent.jpg'
stylePath          = '../neural_art/images/testingArt.jpg'
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
imageShape         = (int(720/2),int(1280/2),3)
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
    model =net.Vgg19()
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


def buildGramMatrix(layer):
    _, dimX, dimY, num_filters = layer.get_shape()
    vectorized_maps = tf.reshape(layer, [int(dimX) * int(dimY), int(num_filters)])

    if int(dimX) * int(dimY) > int(num_filters):
        return tf.matmul(vectorized_maps, vectorized_maps, transpose_a=True)
    else:
        return tf.matmul(vectorized_maps, vectorized_maps, transpose_b=True)


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



def totalLoss(model):
    errorComponents =[buildStyleLoss(model), buildContentLoss(model), buildTVNorm(model)]
    LossWeights = [styleLossWeight, contentLossWeight,TVNormLossWeight]
    loss =[]
    for error, weights in zip(errorComponents, LossWeights):
        loss.append(error*weights)

    reducedLoss = reduce(lambda x,y: x+y, loss)
    return reducedLoss

def getUpdateTensor(model, inputVar):
    loss = totalLoss(model)
    optimizer = tf.train.AdamOptimizer(learningRate)
    grads = optimizer.compute_gradients(loss, [inputVar])
    clipped_grads = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in grads]
    return [optimizer.apply_gradients(clipped_grads), loss]


def train(model, inputVar, sess):
    updateTensor, lossTensor = getUpdateTensor(model, inputVar)
    sess.run(tf.initialize_all_variables())
    start_time = time.time()

    for iteration in range(numIters):
        sess.run(updateTensor)
        if(iteration%showEveryN==0):
            img = inputVar.eval()
            print("Iteration : %s | Loss : %g"%(str(iteration).zfill(4), lossTensor.eval()))
            utils.showImage(img,imageShape)
        elif(iteration%10==0):
            print("Iteration : %s | Loss : %g" % (str(iteration).zfill(4), lossTensor.eval()))

    elapsed = time.time() -start_time
    print("Experiment Took : %s"%(str(elapsed)))


with tf.Session() as sess:
    model = net.Vgg19()
    inputVar = tf.Variable(tf.random_uniform((1,)+imageShape, minval=0.25, maxval=0.75))
    model.build(inputVar, imageShape)
    train(model, inputVar, sess)
