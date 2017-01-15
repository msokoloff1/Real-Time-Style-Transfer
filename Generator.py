import tensorflow as tf
import utilities as utils
import tensorflow as tf
import numpy as np
import os
import gc


class GeneratorNet():
    def __init__(self, sess, dims, trainingNet = True):
        #Different Net for predict than training, for arbitrary image sizes..
        self.sess = sess
        with tf.variable_scope("generator"):
            print("Creating Generator Network")
            self.trainingNet = trainingNet
            self.inputContent = tf.placeholder(tf.float32, shape=[None, dims[0],dims[1],dims[2]], name='Input_Content_PH')
            #ARGS: inputLayer, filterWH, numOutputFeatures, strideSize, name, activation= tf.nn.relu, padType = 'VALID'
            conv_1 = self.__convLayer__(self.inputContent, 9, 32, 1, "conv1_1")
            conv_2 = self.__convLayer__(conv_1, 3, 64, 2, "conv2_2")
            conv_3 = self.__convLayer__(conv_2, 3, 128, 2, "conv3_3")
            #ARGS: inputLayer, name, filterWH=3
            res_1 = self.__residualBlock__(conv_3, "res1_4")
            res_2 = self.__residualBlock__(res_1, "res2_5")
            res_3 = self.__residualBlock__(res_2, "res3_6")
            res_4 = self.__residualBlock__(res_3, "res4_7")
            res_5 = self.__residualBlock__(res_4, "res5_8")
            #ARGS : inputLayer, filterWH, numOutputFeatures, stride, name
            deconv_1 = self.__deconvLayer__(res_5, 3, 64,  2, "deconv1_9")
            deconv_2 = self.__deconvLayer__(deconv_1, 3, 32, 2, "deconv2_10")
            self.output = self.__convLayer__(deconv_2, 9, 3, 1, "output_11", activation=tf.nn.sigmoid)
            
            
            self.trainableVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            for element in self.trainableVars:
                print(element.name)
            
            print("Generator Network Initialized")
                
    def predict(self, inputImage):
        return self.sess.run(self.output, feed_dict={self.inputContent:inputImage})   

    def __convLayer__(self, inputLayer, filterWH, numOutputFeatures, strideSize, name, activation= tf.nn.relu, padType = 'VALID'):
        padSize = filterWH//2
        inputLayer = tf.pad(inputLayer, [[0,0],[padSize,padSize],[padSize,padSize],[0,0]], "REFLECT")
        numInputFeatures = inputLayer.get_shape().as_list()[-1]    
        convFilter = tf.Variable(tf.random_normal([filterWH, filterWH, numInputFeatures, numOutputFeatures], stddev=np.sqrt(2./( (filterWH**2)*numInputFeatures)), name="weights"))
        strideConfig =  [1, strideSize, strideSize, 1]
        if not self.trainingNet:
            padType = 'SAME'
                
        conv = tf.nn.conv2d(inputLayer, convFilter, strideConfig, padding=padType)
        bias = tf.Variable(tf.zeros([numOutputFeatures]), name="biases")
        convOutput = tf.nn.bias_add(conv, bias)

        if activation == tf.nn.relu:
            norm = self.__normalizeBatch__(convOutput)
            return activation(norm)
        else:
            return activation(convOutput)

    def __residualBlock__(self, inputLayer, name, filterWH=3):
        with tf.variable_scope(name):
            numInputFeatures = inputLayer.get_shape().as_list()[-1]
            conv = self.__convLayer__(inputLayer, filterWH, numInputFeatures, 1, "%s-a" % name)
            return inputLayer + self.__normalizeBatch__(self.__convLayer__(conv, filterWH, numInputFeatures, 1, name, activation=lambda x: x))

    def __normalizeBatch__(self,x):
        with tf.variable_scope("instance_norm"):
            eps = 1e-6
            mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
            return (x - mean) / (tf.sqrt(var) + eps)

    def __deconvLayer__(self, inputLayer, filterWH, numOutputFeatures, stride, name):
        with tf.variable_scope(name):
            _, h, w, numInputFeatures = inputLayer.get_shape().as_list()
            
            batchSize = 1
            convFilter = tf.Variable(tf.random_normal([filterWH, filterWH,numOutputFeatures,numInputFeatures], stddev=np.sqrt(2./( (filterWH**2)*numInputFeatures)), name="weights"))
            deconv = tf.nn.conv2d_transpose(inputLayer, convFilter, [batchSize, h * stride, w * stride, numOutputFeatures], [1, stride, stride, 1], padding='SAME')
            biases = tf.Variable(tf.zeros([numOutputFeatures]), name="biases")
            bias = tf.nn.bias_add(deconv, biases)
            norm = self.__normalizeBatch__(bias)
            return tf.nn.relu(norm)
        

        