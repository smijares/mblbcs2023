#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mblbcs2022 main script
V1.1
Sebastià Mijares i Verdú - GICI, UAB
sebastia.mijares@uab.cat

Core script to replicate the results described in the "Multiband deep learning
models for hyperspectral remote sensing images compression" paper by Sebastià
Mijares i Verdú, Valero Laparra, Johannes Ballé, Joan Bartrina-Rapesta, Miguel
Hernández-Cabronero, and Joan Serra-Sagristà, submitted for publication in the
IEEE Geoscience and Remote Sensing Letters journal in July 2022.

This script automates processes to be run by the architecture script to train
new models or test existing ones. Run 'python3 MAIN.py -h' for help on how to
use this script.

Required libraries
------------------

argparse
glob
sys
absl
os
numpy
math
random
PIL
time
tensorflow

"""

import argparse
import sys
from absl import app
from absl.flags import argparse_flags
import os
import numpy as np
import time
import tensorflow as tf
import random
from PIL import Image

def read_raw(filename,height,width,bands,endianess,datatype):
  """
  Reads a raw image file and returns a tensor of given height, width, and number of components, taking endianess and bytes-per-entry into account.

  This function is independent from the patchsize chosen for training.
  """
  string = tf.io.read_file(filename)
  vector = tf.io.decode_raw(string,datatype,little_endian=(endianess==1))
  return tf.transpose(tf.reshape(vector,[bands,height,width]),(1,2,0))

def ms_ssim(X, Y, height=512, width=680, bands=224, endianess = 1, data_type = tf.uint16, maxval = 65535):
    """
    Function imported from metrics v2.1.
    """
    x = read_raw(X,height,width,bands,endianess,data_type)
    x_hat = read_raw(Y,height,width,bands,endianess,data_type)
    return np.array(tf.squeeze(tf.image.ssim_multiscale(x, x_hat, maxval)))

def get_geometry_dataset(directory):
    D = os.listdir(directory)
    for file in D:
        if file[-4:]=='.raw':
            G = file.split('.')[1].split('_')
            bands = G[0]
            width = G[1]
            height = G[2]
            datatype = G[3]
            endianess = G[4]
            #Note these are all strings, not int numbers!
            return (bands, width, height, endianess, datatype)
    print('No RAW files were found. Assumes 8-bit PNG files instead.')
    return ('0', '0', '0', '0', '1')

def get_geometry_file(file):
    if file[-4:]=='.raw':
        G = file.split('.')[1].split('_')
        bands = G[0]
        width = G[1]
        height = G[2]
        datatype = G[3]
        endianess = G[4]
        #Note these are all strings, not int numbers!
        return (bands, width, height, endianess, datatype)
    print('No RAW files were found. Assumes 8-bit PNG files instead.')
    return ('0', '0', '0', '0', '1')
        
def dataset_path(args):
    return '../datasets/'+args.dataset

def full_model_path(args):
    return '../models/'+args.model_path
    
def baseline_path(args):
    return './'+args.architecture+'.py'

def list_command(L):
    """
    Converts a list into a string with the elements sepparated by spaces for command line purposes.
    """
    result = ''
    for x in L:
        result+=str(x)+' '
    return result [:-1]

def run_training(args):
    print('Delaying for '+str(args.delay)+' seconds...')
    time.sleep(args.delay)
    bands, width, height, endianess, datatype = get_geometry_dataset(dataset_path(args))
    
    if int(bands) > args.input_bands:
        print('Auxiliary training set needed')
        dataset_p = dataset_path(args)+'_aux'
        if not os.path.isdir(dataset_path(args)+'_aux'):
            print('Generating auxiliary training set')
            os.system('python3 ./auxiliary/bands_extractor.py --source '+args.dataset+' --destination '+args.dataset+'_aux --consecutive_bands '+str(args.input_bands))
    else:
        dataset_p = dataset_path(args)
    if not os.path.isdir(full_model_path(args)):
        os.mkdir(full_model_path(args))
        os.mkdir(full_model_path(args)+'/logs')
    os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' -V train --train_path '+full_model_path(args)+'/logs --train_glob "'+dataset_p+'/*.raw" --num_scales '+str(args.num_scales)+' --scale_min '+str(args.scale_min)+' --scale_max '+str(args.scale_max)+' --epochs '+str(args.epochs)+' --bands '+str(args.input_bands)+' --width '+width+' --height '+height+' --endianess '+endianess+' --lambda '+list_command(args.lmbda)+' --patchsize '+str(args.patchsize)+' --batchsize '+str(args.batchsize)+' --num_filters '+list_command(args.num_filters)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
    print('Training ended on '+time.asctime(time.localtime()))
    if not args.autotest == None:
        print('Running automatic testing of model on '+args.autotest+' dataset')
        test(args, dataset=args.autotest)
        print('Testing ended on '+time.asctime(time.localtime()))
    
def test(args,dataset=None):
    print('Delaying for '+str(args.delay)+' seconds...')
    time.sleep(args.delay)
    if dataset==None:
        dataset=args.dataset
        
    bands, width, height, endianess, datatype = get_geometry_dataset('../datasets/'+dataset)
    if int(bands) > args.input_bands:
        print('Auxiliary test set needed')
        dataset = dataset+'_aux'
        if not os.path.isdir(dataset+'_aux'):
            print('Generating auxiliary test set')
            os.system('python3 ./auxiliary/bands_extractor.py --source '+dataset+' --destination '+dataset+'_aux --consecutive_bands '+str(args.input_bands))
    
    try:
    	corpus = random.sample(os.listdir('../datasets/'+dataset),args.sample)
    except:
    	corpus = os.listdir('../datasets/'+dataset)
    results = open(full_model_path(args)+'/'+args.model_path+'_'+dataset+'_results.csv','w')
    results.write('Image,')
    results.write('TFCI size,Rate (bps),MSE')
    try:
        if args.SSIM:
            results.write(',MS-SSIM')
        if args.MAE:
            results.write(',MAE')
        if args.PAE:
            results.write(',PAE')
    except:
        pass
    results.write ('\n')
    
    print('Compressing...')
    os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' compress "'+'../datasets/'+dataset+'/*.raw"')
    print('Decompressing...')
    os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' decompress "'+'../datasets/'+dataset+'/*.raw.tfci"')

        
    if datatype == '0':
        D = np.bool_
        maxval = 1
    elif datatype == '1':
        D = np.uint8
        maxval = 255
    elif datatype == '2':
        D = np.uint16
        maxval = 65535
    elif datatype == '3':
        D = np.int16
        maxval = 32767
    elif datatype == '4':
        D = np.int32
        maxval = None
    elif datatype == '5':
        D = np.int64
        maxval = None
    elif datatype == '6':
        D = np.float32
        maxval = None
    else:
        D = np.float64
        maxval = None
        
    sanity_check = random.choice(corpus)
    while not os.path.splitext(sanity_check)[1] == '.raw':
        sanity_check = random.choice(corpus)
    
    for IMAGE in corpus:
        print('Testing image '+IMAGE)
        if os.path.splitext(IMAGE)[1] == '.raw':
            bands, width, height, endianess, datatype = get_geometry_file(IMAGE)
            path_to_image = '../datasets/'+dataset+'/'+IMAGE
            results.write(IMAGE+',')
            tfci_path_to_image = os.path.splitext(path_to_image)[0]+'.raw.tfci'
            raw_tfci_path_to_image = os.path.splitext(path_to_image)[0]+'.raw.tfci.raw'
            compressed_size = os.stat(tfci_path_to_image)[6]
            img0 = np.reshape(np.fromfile(path_to_image,dtype=D),(int(bands),int(height),int(width))).astype(np.float32)
            img1 = np.reshape(np.fromfile(raw_tfci_path_to_image,dtype=D),(int(bands),int(height),int(width))).astype(np.float32)
            mse = np.mean((img0-img1)**2)
            if int(bands)*int(width)*int(height)==0:
                bps = 0
            else:
                bps = compressed_size*8/(int(bands)*int(width)*int(height))
            results.write(str(compressed_size)+','+str(bps)+','+str(mse))
            try:
                if args.SSIM:
                    if maxval == None:
                        maxval = np.max(img0)
                    ssim = ms_ssim(path_to_image, raw_tfci_path_to_image, height=int(height), bands=int(bands), width=int(width), endianess=int(endianess), data_type=D, maxval=maxval)
                    results.write(','+str(ssim))
                if args.MAE:
                    mae = np.mean(abs(img0-img1))
                    results.write(','+str(mae))
                if args.PAE:
                    pae = np.max(abs(img0-img1))
                    results.write(','+str(pae))
            except:
                pass
            if datatype in '12' and IMAGE==sanity_check:
                IMG0 = Image.fromarray(img0[0,:,:].astype(D))
                IMG1 = Image.fromarray(img1[0,:,:].astype(D))
                IMG0.save(full_model_path(args)+'/'+args.model_path+'_'+dataset+'_sanity-check_org_'+sanity_check[:-4]+'.png')
                IMG1.save(full_model_path(args)+'/'+args.model_path+'_'+dataset+'_sanity-check_new_'+sanity_check[:-4]+'.png')

            os.system('rm '+tfci_path_to_image)
            os.system('rm '+raw_tfci_path_to_image)
            results.write('\n')
    results.close()

def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--model_path", default="test_model",
      help="Code under which to save/load the trained model. This will be in a predefined directory.")
  parser.add_argument(
      "--architecture", default="hyperprior-adaptive",
      help="Baseline architecture to be trained or tested. Just use the code.")
  parser.add_argument(
      "--input_bands", type=int, default=1,
      help="Number of input bands to be taken in by the model. If the number of"
      "bands in the dataset images is larger, an auxiliary training/testing"
      "dataset will be created slicing the images into clusters of k bands, where"
      "k is the value of this parameter.")
  parser.add_argument(
      "--delay", type=int, default=0,
      help="Time delay (in seconds) for the code to run. Only integer times.")
  parser.add_argument(
      "--CPU", action="store_true",
      help="Mark to force the code to only run on CPU and temporarily disable GPU devices."
      "This is necessary in some hyperprior architectures for testing.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'test' applies a trained model to all images "
           "in a repository and measures the rate-distortion performance.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
  train_cmd.add_argument(
      "--lambda", type=float, nargs='+', default=[0.01], dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--dataset", type=str, default="",
      help="Name of the training dataset. Will be searched as a directory with the same name under ../datasets.")
  train_cmd.add_argument(
      "--num_filters", type=int, nargs='+', default=[128],
      help="Number of filters per 2D convolutional layer. If more than one input is specified, it will place them in the same order."
      "In non-slimmable architectures, two inputs may indicate (1) number of filters in hidden layers and (2) number of filters in latent layers."
      "Read the specifications of each architecture.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Size of batch for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=100,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--learning_rate", type=float, default=0.0001, dest="learning_rate",
      help="Float indicating the learning rate for training.")
  train_cmd.add_argument(
      "--num_scales", type=int, default=64, dest="num_scales",
      help="Number of Gaussian scales to prepare range coding tables for.")
  train_cmd.add_argument(
      "--scale_min", type=float, default=0.11, dest="scale_min",
      help="Minimum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--scale_max", type=float, default=256.0, dest="scale_max",
      help="Maximum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--autotest", default=None,
      help="Run testing automatically at the end of training. It will use the dataset "
      "indicated in this option.")

    # 'test' subcommand.
  test_cmd = subparsers.add_parser(
      "test",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Tests a trained model on all images in a dataset."
                  "the dataset's specifications will be loaded automatically")

  # Arguments for test command.
  test_cmd.add_argument(
        "--dataset", type=str, default="LandSat8_cropRGB",
        help="Test dataset. Its geometry will be automatically loaded.")
  test_cmd.add_argument(
        "--sample", type=int,
        help="Maximum sample size.")
  test_cmd.add_argument(
        "--SSIM", action="store_true",
        help="Computes MS-SSIM distortion.")
  test_cmd.add_argument(
        "--MAE", action="store_true",
        help="Computes Mean Absolute Error (MAE, L1 norm) distortion.")
  test_cmd.add_argument(
        "--PAE", action="store_true",
        help="Computes Peak Absolute Error (PAE, L-infinity norm) distortion.")
  



  
  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.CPU:
      os.environ["CUDA_VISIBLE_DEVICES"]="-1"
  if args.command == "train":
    run_training(args)
  elif args.command == "test":
    test(args)

if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
