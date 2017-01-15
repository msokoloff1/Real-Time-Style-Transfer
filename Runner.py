import Train
import Test
import utilities as utils
import tensorflow as tf
import Generator



Train.Trainer(savePath = "./savedNets/model.ckpt"
              ,numIters = 10000
              ,imageShape = (int(1600/8),int(2560/8),3)
              ,contentPath = './images/testingContent1.jpg'
              ,contentLayer = 'conv3_3'
              ,stylePath = './images/testingArt.jpg'
              ,styleLayers =  ['conv1_2','conv2_2','conv3_3', 'conv4_1', 'conv5_1']
              ,styleWeights = [0.2      ,0.2     , 0.3     , 0.3       , 0.2      ]
              ,TVNormLossWeight =  0.0001
              ,styleLossWeight=0.8
              ,contentLossWeight=0.05)


with tf.Session() as sess:
    testImageShape = (int(1600/4), int(2560/4), 3)
    tester = Test.LoadedModel(sess,testImageShape, modelPath = "./savedNets/model.ckpt")
    
    tester.predict("./testImages", pathIsDir = True)