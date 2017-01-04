import utilities as utils
import tensorflow as tf
import numpy as np


class GeneratorNet():
    def __init__(self, sess):
        with tf.variable_scope("generator"):
            ###PLACEHOLDERS
            self.inputContent             = tf.placeholder(tf.float32, shape=[None, utils.imageShape[0],utils.imageShape[1],utils.imageShape[2]])
            self.currentlyTraining        = tf.placeholder(tf.bool)

            ###DOWNSAMPLING (NON RESIDUAL BLOCKS)
            #1
            self.conv1_reflection= utils.convolution(self.inputContent,9,9,3,1,1,'1',True, padding="SAME")
            self.conv1_reflection_BN = utils.batch_normalization(self.conv1_reflection,self.currentlyTraining)
            #2
            self.conv2 =  utils.convolution(self.conv1_reflection_BN,3,3,32,1,1,'2',True, padding="SAME")
            self.conv2_BN = utils.batch_normalization(self.conv2, self.currentlyTraining)
            #3
            self.conv3 = utils.convolution(self.conv2_BN,3,3,32,1,1,'2',True, padding="SAME")




            self.trainableVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            sess.run(tf.initialize_all_variables())
            sess.run(self.reflectionPaddingLayer, feed_dict = {self.inputContent:np.array(utils.loadImage(utils.contentPath, utils.imageShape)), self.currentlyTraining:False})
        ########END OF GENERATOR########
        #Should be outside variable scope so don't indent:

        #self.vgg = utils.vggNet.Vgg19()
        #inputVar = self.imagePred
        #self.vgg.build(inputVar, utils.imageShape)



    def __initUpdateTensor__(self):
        loss = utils.totalLoss(self.vgg)
        optimizer = tf.train.AdamOptimizer(utils.learningRate)
        grads = optimizer.compute_gradients(loss, self.trainableVars)
        clipped_grads = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in grads]
        return [optimizer.apply_gradients(clipped_grads), loss]


   # def train(self, inputImage):
        #BACKWARD PROP

   # def addStyle(self, ):
        #FORWARD PROP








"""
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
"""

#Call in runner:
with tf.Session() as sess:
    generator = GeneratorNet(sess)


