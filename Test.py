import tensorflow as tf
import Generator
    
class LoadedModel():
    def __init__(self,sess, imageShape, modelPath):
        with tf.Session() as sess:
            self.sess  = sess
            self.imageShape = imageShape
            self.gen = Generator.GeneratorNet(self.sess, self.imageShape)
            saver = tf.train.Saver()
            try:
                
                saver.restore(sess, modelPath)  #"./modelNight.ckpt"
                print("Successfully Restored Model") 
            except:
                print("No model available for restoration")
        
    
    def predict(testPath,pathIsDir):
        if(pathIsDir):
            #dir = "/home/matt/repositories/fastStyleTransfer/testImages"
            for element in os.listdir(testPath):
                path = testPath + "/" + element
                start = time.time()
                testImg = np.array(utils.loadImage(path, imageShape))
                img = self.gen.predict(testImg.reshape(1,imageShape[0],imageShape[1],imageShape[2]))
                utils.showImage(img,imageShape, "Night"+ str(element))
                elasped = time.time() - start
                print("Made Prediction for: %s | Time taken : %s"%(element, elasped))
        else:
            start = time.time()
            testImg = np.array(utils.loadImage(testPath, self.imageShape))
            img = self.gen.predict(testImg.reshape(1,self.imageShape[0],self.imageShape[1],self.imageShape[2]))
            utils.showImage(img,imageShape, "Night"+ str(element))
            elasped = time.time() - start
            print("Made Prediction | Time taken : %s"%(element, elasped))