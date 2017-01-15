##Imports##
import net
import utilities as utils
import tensorflow as tf
from functools import reduce
import numpy as np
import time
import Generator
import os
###########################




class Loss():
    def __init__(self, contentPath, stylePath, contentLayer, styleLayers
                    ,styleWeights, imageShape, TVNormLossWeight=0.000,styleLossWeight = 3.0
                    , contentLossWeight  = 0.05, learningRate = 0.0001):
        
        self.contentPath = contentPath
        self.stylePath = stylePath
        self.contentLayer = contentLayer
        self.styleLayers = styleLayers 
        self.styleWeights = styleWeights
        self.imageShape = imageShape
        self.styleData          = {}
        self.styleBalanceData   = {}
        self.errorMetricContent = utils.mse #utils.euclidean
        self.errorMetricStyle   = utils.mse
        self.normalizeContent   = True #Inversion paper says to use this
        self.normalizeStyle     = True #No mention of style in inversion paper
        self.sigma = 1.0
        self.beta = 2.0
        self.alpha = 6.0
        self.a = 0.01
        self.B = 120.0
        self.lambdaBeta = 1
        self.alphaNormLossWeight = 0.01
        self.TVNormLossWeight    = TVNormLossWeight
        self.styleLossWeight     = styleLossWeight
        self.contentLossWeight   = contentLossWeight
        self.learningRate  = learningRate

        with tf.Session() as sess:
            self.inputTensor = tf.placeholder(tf.float32, shape=[None,imageShape[0], imageShape[1], imageShape[2]])
            contentImage = np.array(utils.loadImage(contentPath, imageShape))
            styleImage = utils.loadImage(stylePath, imageShape)
            styleBalanceImage = utils.loadImage(contentPath, imageShape)
            model =net.Vgg19()
            model.build(self.inputTensor, imageShape)
            self.contentData = eval('sess.run(model.' + contentLayer + ',feed_dict={self.inputTensor:contentImage})')
            for styleLayer in styleLayers:
                self.styleData[styleLayer] = np.array(eval('sess.run(model.' + styleLayer + ',feed_dict={self.inputTensor:styleImage})'))


    def __buildStyleLoss__(self,model):
        totalStyleLoss = []
        for index, styleLayer in enumerate(self.styleLayers):
            normalizingConstant = 1
            if (self.normalizeStyle):
                normalizingConstant = (np.sum(self.styleData[styleLayer][0] ** 2))**(0.5)
                print("Style Normalizing Constant for Layer %s : %s"%(index, normalizingConstant))
    
            styleLayerVar = tf.Variable(self.styleData[styleLayer])
            correctGrams  = self.__buildGramMatrix__(styleLayerVar)
            tensorGrams   = self.__buildGramMatrix__(eval('model.'+styleLayer))
            _, dimX, dimY, num_filters = styleLayerVar.get_shape()
            
            denominator   =(2*normalizingConstant)*((float(int(dimX))*float(int(dimY)))**2)*(float(int(num_filters))**2)
            error         = tf.reduce_sum(self.errorMetricStyle(tensorGrams, correctGrams))
            totalStyleLoss.append((tf.div(error,denominator)))
    
        styleLoss = tf.reduce_sum(totalStyleLoss)
        return styleLoss

    def __buildGramMatrix__(self,layer):
        shape = tf.shape(layer)
        reshaped = tf.reshape(layer,[shape[0], shape[1]*shape[2], shape[3]])
        transposed = tf.transpose(reshaped, perm=[0,2,1])
        return tf.batch_matmul(transposed, reshaped)

    def __buildContentLoss__(self,model, contentModel):
        normalizingConstant = 1
        if(self.normalizeContent):
            normalizingConstant = np.sum(  (self.contentData**2))**(0.5)
            print("Content Normalizing Constant : %g"%(normalizingConstant))
            
        contentLoss = (eval('self.errorMetricContent(model.' + self.contentLayer+',contentModel.'+self.contentLayer+') / normalizingConstant'))
        return tf.reduce_sum(contentLoss)

    def __buildAlphaNorm__(self,model):
        adjustedImage = model.bgr
        return tf.reduce_sum(tf.pow(adjustedImage, self.alpha))


    def __buildTVNorm__(self,model):
        adjustedImage = model.bgr
    
        yPlusOne = tf.slice(adjustedImage, [0,0,1,0], [1,self.imageShape[0],(self.imageShape[1]-1),self.imageShape[2]])
        xPlusOne = tf.slice(adjustedImage, [0,1,0,0], [1,(self.imageShape[0]-1),self.imageShape[1],self.imageShape[2]])

        inputNoiseYadj = tf.slice(adjustedImage,[0,0,0,0],[1,self.imageShape[0],(self.imageShape[1]-1),self.imageShape[2]])
        inputNoiseXadj = tf.slice(adjustedImage, [0,0,0,0], [1,(self.imageShape[0]-1),self.imageShape[1],self.imageShape[2]])

        lambdaBeta = (self.sigma**self.beta) / (self.imageShape[0]*self.imageShape[1]*((self.a*self.B)**self.beta))
        error1 = tf.slice(tf.square(yPlusOne-inputNoiseYadj), [0,0,0,0], [1,(self.imageShape[0]-1),(self.imageShape[1]-1), self.imageShape[2]])
        error2 = tf.slice(tf.square(xPlusOne-inputNoiseXadj), [0,0,0,0], [1,(self.imageShape[0]-1),(self.imageShape[1]-1), self.imageShape[2]])

        return self.lambdaBeta*tf.reduce_sum( tf.pow((error1+error2),(self.beta/2) ))

    def __totalLoss__(self,model, contentModel):
        errorComponents =[self.__buildStyleLoss__(model), self.__buildContentLoss__(model,contentModel), self.__buildTVNorm__(model)]
        LossWeights = [self.styleLossWeight, self.contentLossWeight,self.TVNormLossWeight ]
        loss =[]
        for error, weights in zip(errorComponents, LossWeights):
            loss.append(error*weights)
    
        reducedLoss = reduce(lambda x,y: x+y, loss)
        return reducedLoss

    def getUpdateTensor(self,model, generatorTrainables, contentModel):
        loss = self.__totalLoss__(model, contentModel)
        #tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': 100}).minimize(session)    
        optimizer = tf.train.AdamOptimizer(self.learningRate)
        grads = optimizer.compute_gradients(loss, generatorTrainables)
        #clipped_grads = [(tf.clip_by_value(grad, -0.01,0.01), var) for grad, var in grads]
        return [optimizer.apply_gradients(grads), loss, grads]
    

