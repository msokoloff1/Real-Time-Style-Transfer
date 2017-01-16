##Imports##
import net
import utilities as utils
import tensorflow as tf
from functools import reduce
import numpy as np
import time
import Generator
import os
import Loss
###########################

## This file

class Trainer():
    def __init__(self, savePath,numIters, imageShape ,contentPath,contentLayer,stylePath,styleLayers,styleWeights,TVNormLossWeight, styleLossWeight,contentLossWeight,  verbose = True, showEveryN = 100):
        #Own session so it goes away after training
        with tf.Session() as sess:
            self.sess  = sess    
            self.imageShape = imageShape
            self.Verbose = verbose
            self.showEveryN = showEveryN
            self.numIters = numIters
            model_content = net.Vgg19()
            vggContentPlaceholder = tf.placeholder(tf.float32, shape = [None, imageShape[0],imageShape[1],imageShape[2]])
            model_content.build(vggContentPlaceholder, imageShape)
    
            
            model = net.Vgg19()
            gen = Generator.GeneratorNet(self.sess, imageShape)
            inputVar = gen.output
            model.build(inputVar, imageShape)
            
            
            self.lossObj = Loss.Loss(contentPath, stylePath, contentLayer, styleLayers, styleWeights, imageShape, TVNormLossWeight,styleLossWeight, contentLossWeight)
            saver = tf.train.Saver()
            try:
                saver.restore(self.sess, savePath)  #"./modelNight.ckpt"
                print("Successfully Restored Model") 
            except:
                print("No model available for restoration")
        
            self.__train__(model, inputVar,model_content, vggContentPlaceholder, self.sess, gen)
            save_path = saver.save(self.sess, savePath) #"./modelNight.ckpt"
            
        
    def __train__(self, model, inputVar,contentModel, contentPH,  sess, gen):
        updateTensor, lossTensor, grads = self.lossObj.getUpdateTensor(model, gen.trainableVars, contentModel)
        sess.run(tf.initialize_all_variables())
        start_time = time.time()
        testImg = np.array(utils.loadImage("./images/testingContent.jpg", self.imageShape))
        
        for iteration in range(0,self.numIters):
            inputImage = self.__getBatch__(iteration)
            sess.run(updateTensor, feed_dict={gen.inputContent:inputImage,contentPH:inputImage})
            if(iteration%self.showEveryN==0 and self.Verbose):                
                img = gen.predict(testImg.reshape(1,self.imageShape[0],self.imageShape[1],self.imageShape[2]))
                print("Iteration : %s | Loss : %g"%(str(iteration).zfill(4), sess.run(lossTensor, feed_dict={gen.inputContent:inputImage,contentPH:inputImage})))
                utils.showImage(img,self.imageShape, str(iteration))
                
            elif(iteration%10==0 and self.Verbose):
                print("Iteration : %s | Loss : %g" % (str(iteration).zfill(4), sess.run(lossTensor, feed_dict={gen.inputContent:inputImage,contentPH:inputImage})))
    
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        elapsed = time.time() -start_time
        print("Experiment Complete! Elapsed Time : %s"%(str(elapsed)))


    def __getBatch__(self,iteration):
        dir="/home/matt/repositories/coco/train2014"
        filenames = os.listdir(dir)
        batch = []
        iter = iteration%80000
        #for index in range(iter, (iter+4) ):
        index = iter
        img = utils.loadImage(dir+"/"+filenames[index], (self.imageShape[0],self.imageShape[1],self.imageShape[2]))
        batch.append(img.reshape(self.imageShape[0],self.imageShape[1],self.imageShape[2]))
            
        return np.array(batch).reshape(1,self.imageShape[0],self.imageShape[1],self.imageShape[2])


        
