import utilities as utils
import tensorflow as tf


class GeneratorNet():
    def __init__(self, sess):
        #TODO: Implement generator net
        self.inputContent = tf.placeholder(tf.float32, shape=[None, utils.imageShape[0],utils.imageShape[1],utils.imageShape[2]])



        ########END OF GENERATOR########
        self.vgg = utils.vggNet.Vgg19()
        inputVar = self.imagePred
        self.vgg.build(inputVar, utils.imageShape)
        self.trainableVars = []

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


