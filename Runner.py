#import Train
import Test
import utilities as utils
import tensorflow as tf
import Generator
import numpy as np
import io
import argparse
import re
from os import listdir
from os.path import isfile, join
import time
from os.path import basename

parser = argparse.ArgumentParser(description='Control parameters for real time style transfer.')

parser.add_argument('-model_load_path' , default = './savedNets/bnightXHigh.ckpt'                          , help = 'Path to directory where the trained model will be loaded')
parser.add_argument('-model_save_path' , default = './savedNets/modelColors.ckpt'                        , help = 'Path to directory where the trained model will be saved')
parser.add_argument('-xfer_dir'        , default = './transferSource'                                    , help = 'Path to directory containing images to be stylized')
parser.add_argument('-result_dir'      , default = './results'                                           , help = 'Path to directory where stylized images will be stored')
parser.add_argument('-train'           , default = False, type=bool                                      , help = 'Flag indicating whether to train or not')
parser.add_argument('-xfer'            , default = True, type=bool                                       , help = 'Flag indicating whether to perform style transfer or not')
parser.add_argument('-train_iters'     , default = 20000, type=int                                       , help = 'Number of iterations for training operation')
parser.add_argument('-style_image_path', default = './sourceImages/testingArt.jpg'                       , help = 'Image to use for copying the style')
parser.add_argument('-style_layers'    , default = ['conv1_2','conv2_2','conv3_3', 'conv4_1', 'conv5_1'] , help = 'Which layers of the vgg network to be used for obtaining style statistics')
parser.add_argument('-style_weights'   , default = [0.2      ,0.2     , 0.3     , 0.3       , 0.2      ] , help = 'Weights for the loss between generator result and style image for each layer in the vgg network')
parser.add_argument('-tvnorm_weight'   , default = 0.00001, type=float                                   , help = 'Weight for the tv norm loss')
parser.add_argument('-style_weight'    , default = 0.000018, type=float                                  , help = 'Weight for the style loss')
parser.add_argument('-content_weight'  , default = 0.05, type=float                                      , help = 'Weight for the content loss')           
parser.add_argument('-result_shape'    , default = (int(1600/2), int(2560/2), 3)                         , help = 'Dimensions of stylized result (Height/Width/Color Channels)')
parser.add_argument('-use_all_styles'  , default = False, type=bool                                      , help = 'Flag indicating whether or not to apply all saved styles')

args = parser.parse_args()
assert(len(args.style_weights) == len(args.style_layers)), "Number of style layers and the number of style layer weights do not match"


if(args.train):
    Train.Trainer(savePath           = args.model_save_path
                  , numIters         = args.train_iters
                  , imageShape       = (int(1600/4),int(2560/4),3)
                  , contentPath      = './sourceImages/testingContent1.jpg' #This is what gets displayed during training
                  , stylePath        = args.style_image_path # This is the style that is transferred
                  , styleLayers      = args.style_layers
                  , styleWeights     = args.style_weights
                  , TVNormLossWeight = args.tvnorm_weight
                  , styleLossWeight  = args.style_weight
                  ,contentLossWeight = args.content_weight
                  )


if(args.xfer):
    testImageShape = args.result_shape
    if(args.use_all_styles):
        find = re.compile(r"^[^.]*")
        unwanted = ['checkpoint', 'vgg19', 'modelColors', 'sky', 'red', 'snightmed']
        allStyles = [re.search(find, f).group(0) for f in listdir('./savedNets')]
        uniqueStyles = [["./savedNets/%s.ckpt"%(x),x] for x in allStyles if x not in unwanted and (unwanted.append(x) or True)]
        for modelPath, name in uniqueStyles:
            print("RUNNING %s"%(modelPath))
            with tf.Session() as sess:
                tester = Test.LoadedModel(sess,testImageShape, modelPath = modelPath)
                tm = time.strftime('%Y%M%d%_H%M%S')
                tester.predict(args.xfer_dir, pathIsDir = True, destDir = args.result_dir, prefix = "%s%s"%(name,tm))
    else:
        with tf.Session() as sess:
            find = re.compile(r"^[^.]*")
            name = re.search(find, basename(args.model_load_path)).group(0)
            tester = Test.LoadedModel(sess,testImageShape, modelPath = args.model_load_path)  
            tm = time.strftime('%Y%M%d%_H%M%S')
            tester.predict(args.xfer_dir, pathIsDir = True,destDir = args.result_dir,  prefix = "%s%s"%(name,tm))
        
        
