import utilities as utils
import tensorflow as tf
import numpy as np
import os

import gc

class GeneratorNet():
    def __init__(self, sess):
        self.sess = sess
        with tf.variable_scope("generator"):
            ########################################################################################################################
            ###PLACEHOLDERS
            self.batchSize  = tf.placeholder(tf.int32, name='Batch_Size_PH')
            self.inputContent             = tf.placeholder(tf.float32, shape=[None, utils.imageShape[0],utils.imageShape[1],utils.imageShape[2]], name='Input_Content_PH')
            self.currentlyTraining        = tf.placeholder(tf.bool, name='Currently_Training_PH')
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

            self.deconv1 = utils.deconvolution(self.residual5, self.batchSize,128, 64, self.currentlyTraining)
            print(self.deconv1.get_shape())
            self.deconv2 = utils.deconvolution(self.deconv1, self.batchSize, 256, 32, self.currentlyTraining)
            print(self.deconv2.get_shape())
            self.output  = utils.convolution(self.deconv2,9,9,3,1,1,'output',self.currentlyTraining,batchNormalize=False,padding='SAME', activation=tf.nn.sigmoid)
            print(self.output.get_shape())

            self.trainableVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")


        #Different loss depending on the input
        self.vggContentLoss = utils.vggNet.Vgg19()
        self.vggContentLoss.build(self.inputContent, utils.imageShape)

        #

        self.vgg = utils.vggNet.Vgg19()
        inputVar = self.output
        self.vgg.build(inputVar, utils.imageShape)

        self.loss = utils.totalLoss(self.vgg, self.vggContentLoss)

        self.optimizer = tf.train.AdamOptimizer(utils.learningRate)
        self.grads = self.optimizer.compute_gradients(self.loss, self.trainableVars)
        self.clipped_grads = [(tf.clip_by_value(self.grad, -5.0, 5.0), var) for self.grad, var in self.grads]
        self.updateOp = self.optimizer.apply_gradients(self.clipped_grads)

    def train(self, trainingIters = 20, batchSize = 2):
        self.sess.run(tf.initialize_all_variables())
        print("HERE")
        
        
        for iteration in range(trainingIters):
            print("HERE" + str(iteration))
            imageBatch = self.getImages()
            if(iteration%10==0):

                print("Iteration : %s | Loss : %s "%(iteration, self.getLoss(np.random.random((batchSize,256,256,3)), batchSize)))
            
            feedDict = {self.inputContent : np.random.random((batchSize,256,256,3)), self.batchSize: int(batchSize), self.currentlyTraining:True }
            self.sess.run(self.updateOp, feed_dict=feedDict)
        self.sess.close()
        gc.collect()
        exit()
        #save_path = saver.save(sess, "/tmp/model.ckpt")

    def addStyle(self, inputImage):
        image = self.sess.run(self.output, feed_dict={})

    def getLoss(self, inputBatch, batchSize =4):
        return self.sess.run(self.loss, feed_dict={self.inputContent : inputBatch, self.batchSize: int(batchSize), self.currentlyTraining:False})

    def getImages(self, dir="/home/matt/repositories/neural_art/images", batchSize = 4):
        filenames = os.listdir(dir)
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        #image = tf.image.decode_png(image_file)
        image = tf.image.decode_jpeg(image_file)

        image.set_shape((256, 256, 3))
        # Generate batch
        num_preprocess_threads = 1
        min_queue_examples = 256
        batch = tf.train.shuffle_batch(
            [image],
            batch_size=batchSize,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batchSize,
            min_after_dequeue=min_queue_examples)

        return batch

"""
    def getBatch(self, images, batch_size=4):


    return batch
"""

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
    generator.train()
    saver = tf.train.Saver()

    gc.collect()



