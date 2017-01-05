import utilities as utils
import tensorflow as tf
import numpy as np


class GeneratorNet():
    def __init__(self, sess):
        with tf.variable_scope("generator"):
            ########################################################################################################################
            ###PLACEHOLDERS
            self.batchSize  = tf.placeholder(tf.int32)
            self.inputContent             = tf.placeholder(tf.float32, shape=[None, utils.imageShape[0],utils.imageShape[1],utils.imageShape[2]])
            self.currentlyTraining        = tf.placeholder(tf.bool)
            print(self.inputContent.get_shape())
            ########################################################################################################################
            ###DOWNSAMPLING (NON RESIDUAL BLOCKS)
            #1
            self.conv1_reflection_BN= utils.convolution(self.inputContent,9,9,3,1,1,'1_conv_1',self.currentlyTraining,isPadded=True, padding="SAME")
            print(self.conv1_reflection_BN.get_shape())
            #2
            self.conv2_BN =  utils.convolution(self.conv1_reflection_BN,3,3,32,1,1,'2_conv_2',self.currentlyTraining, padding="SAME")
            print(self.conv2_BN.get_shape())
            #3
            self.conv3_BN = utils.convolution(self.conv2_BN,3,3,64,2,2,'3_conv_3',self.currentlyTraining, padding="SAME")
            print(self.conv3_BN.get_shape())
            #4
            self.conv4_BN = utils.convolution(self.conv3_BN, 3, 3, 128, 2, 2, '4_conv_4',self.currentlyTraining, padding="SAME")
            print(self.conv4_BN.get_shape())
            ########################################################################################################################
            ###DOWNSAMPLING (RESIDUAL BLOCKS)

            self.residual1 = utils.residualBlock(self.conv4_BN, self.currentlyTraining, '5_residual_1', self.batchSize)
            print(self.residual1.get_shape())
            self.residual2 = utils.residualBlock(self.residual1, self.currentlyTraining, '6_residual_2', self.batchSize)
            print(self.residual2.get_shape())
            self.residual3 = utils.residualBlock(self.residual2, self.currentlyTraining, '7_residual_3', self.batchSize)
            print(self.residual3.get_shape())
            self.residual4 = utils.residualBlock(self.residual3, self.currentlyTraining, '8_residual_4', self.batchSize)
            print(self.residual4.get_shape())
            self.residual5 = utils.residualBlock(self.residual4, self.currentlyTraining, '9_residual_5', self.batchSize)
            print(self.residual5.get_shape())
            ########################################################################################################################
            ###UPSAMPLING

            deconv = tf.nn.conv2d_transpose(input_layer, [3, 3, 1, 1],[1, 26, 20, 1], [1, 2, 2, 1], padding='SAME', name=None)







            self.trainableVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            sess.run(tf.initialize_all_variables())
            a = sess.run(self.residual1, feed_dict = {self.inputContent:np.array(utils.loadImage(utils.contentPath, utils.imageShape)),self.batchSize:int(1),  self.currentlyTraining:False})
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


