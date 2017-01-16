import tensorflow as tf
import Generator
import os
import utilities as utils
import time 
import numpy as np

class LoadedModel():
    def __init__(self,sess, imageShape, modelPath):
        
        self.sess  = sess
        self.imageShape = imageShape
        self.gen = Generator.GeneratorNet(self.sess, self.imageShape, trainingNet = True)
        saver = tf.train.Saver()
        try:
            
            saver.restore(sess, modelPath)  #"./modelNight.ckpt"
            print("Successfully Restored Model") 
        except:
            print("No model available for restoration")
        
    
    def predict(self, testPath,pathIsDir):
        if(pathIsDir):
            #dir = "/home/matt/repositories/fastStyleTransfer/testImages"
            for element in os.listdir(testPath):
                path = testPath + "/" + element
                start = time.time()
                testImg = np.array(utils.loadImage(path, self.imageShape))
                img = self.gen.predict(testImg.reshape(1,self.imageShape[0],self.imageShape[1],self.imageShape[2]))
                utils.showImage(img,self.imageShape, "Night"+ str(element))
                elasped = time.time() - start
                print("Made Prediction for: %s | Time taken : %s"%(element, elasped))
        else:
            start = time.time()
            testImg = np.array(utils.loadImage(testPath, self.imageShape))
            img = self.gen.predict(testImg.reshape(1,self.imageShape[0],self.imageShape[1],self.imageShape[2]))
            utils.showImage(img,self.imageShape, "NewTrials"+ str(element))
            elasped = time.time() - start
            print("Made Prediction | Time taken : %s"%(element, elasped))
            
    def predictImage(self, imgData):
        #For webcam api
        img = self.gen.predict(imgData.reshape(1,self.imageShape[0],self.imageShape[1],self.imageShape[2]))
        return img
            
    
