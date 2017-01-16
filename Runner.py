import Train
import Test
import utilities as utils
import tensorflow as tf
import Generator

train = False
if (train):
    Train.Trainer(savePath = "./savedNets/model.ckpt"
                  ,numIters = 10000
                  ,imageShape = (int(1600/4),int(2560/4),3)
                  ,contentPath = './images/testingContent1.jpg'
                  ,contentLayer = 'conv3_3'
                  ,stylePath = './images/testingArt.jpg'
                  ,styleLayers =  ['conv1_2','conv2_2','conv3_3', 'conv4_1', 'conv5_1']
                  ,styleWeights = [0.1      ,0.1     , 0.3     , 0.3       , 0.2      ]
                  ,TVNormLossWeight =  0.000001
                  ,styleLossWeight=0.0000075
                  ,contentLossWeight=0.05)


with tf.Session() as sess:
    testImageShape = (int(1600/2), int(2560/2), 3)
    tester = Test.LoadedModel(sess,testImageShape, modelPath = "./savedNets/model.ckpt")
        
    tester.predict("./testImages", pathIsDir = True)
        
    ## ADD LOGIC FOR HTTP REQUEST HERE
    #imgData = POST['image']
    #prediction = tester.predictImage(imgData)
    #message.send(prediction)
