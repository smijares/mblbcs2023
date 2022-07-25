#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plantilla d'entrenament de models de xarxes neurals
V7.5
Sebastià Mijares i Verdú - GICI, UAB
sebastia.mijares@uab.cat

Implementació d'entrenament de model de xarxes neurals i proves de compressió.

Compatible amb xarxes convolucionals 2D i convolucionals 1D+2D.

Com utilitzar aquest codi
-------------------------

Executar acompanyat dels mòduls requerits i indicant els paràmetres desitjats a
les comandes. La comanda 'train' entrena un model de xarxa neural. La comanda
'test' el prova sobre totes les imatges d'un repositori. La geometria de les
imatges dels repositoris es carrega automàticament.

Paràmetres
----------

No hi ha paràmetres generals hard-coded.

Requisits
---------

-Mòdul argparse
-Mòdul glob
-Mòdul sys
-Mòdul absl
-Mòdul os
-Mòdul numpy
-Mòdul math
-Mòdul random
-Mòdul PIL

Desenvolupament
---------------

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

def sam(X, Y, height=512, width=680, bands=224, endianess = 1, data_type = tf.uint16):
    """
    Function imported from metrics v2.1.
    """
    x = tf.cast(read_raw(X,height,width,bands,endianess,data_type),tf.float32)
    x_hat = tf.cast(read_raw(Y,height,width,bands,endianess,data_type),tf.float32)
    return np.array(tf.reduce_mean(tf.acos(tf.reduce_sum(x*x_hat,axis=2)/tf.sqrt(tf.reduce_sum(x*x,axis=2)*tf.reduce_sum(x_hat*x_hat,axis=2)))))

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
    subarchitecture = 'miscellaneous-architectures'
    if args.architecture[0:4] == 'NLTC':
        subarchitecture = 'non-linear'
    elif args.architecture[0:4] == 'LLTC':
        subarchitecture = 'linear-learned'
    elif args.architecture[0:4] == 'IMIC':
        subarchitecture = 'importance-map'
    elif args.architecture[0:4] == 'DFLC':
        subarchitecture = 'defined-features'
    elif args.architecture[0:4] == 'CSMR':
        subarchitecture = 'multi-rate-CS'
    elif args.architecture[0:4] == 'MDAE':
        subarchitecture = 'modulated-AE'

    return './'+subarchitecture+'/'+args.architecture+'.py'

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
    if not os.path.isdir(full_model_path(args)):
        os.mkdir(full_model_path(args))
        os.mkdir(full_model_path(args)+'/logs')
    if args.hyperprior:
        os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' -V train --train_path '+full_model_path(args)+'/logs --train_glob "'+dataset_path(args)+'/*.'+args.extension+'" --num_scales '+str(args.num_scales)+' --scale_min '+str(args.scale_min)+' --scale_max '+str(args.scale_max)+' --epochs '+str(args.epochs)+' --bands '+bands+' --width '+width+' --height '+height+' --endianess '+endianess+' --lambda '+list_command(args.lmbda)+' --patchsize '+str(args.patchsize)+' --batchsize '+str(args.batchsize)+' --num_filters '+list_command(args.num_filters)+' --num_filters_1D '+str(args.num_filters_1D)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
    else:
        os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' -V train --train_path '+full_model_path(args)+'/logs --train_glob "'+dataset_path(args)+'/*.'+args.extension+'" --epochs '+str(args.epochs)+' --bands '+bands+' --width '+width+' --height '+height+' --endianess '+endianess+' --lambda '+list_command(args.lmbda)+' --patchsize '+str(args.patchsize)+' --batchsize '+str(args.batchsize)+' --num_filters '+list_command(args.num_filters)+' --num_filters_1D '+str(args.num_filters_1D)+' --learning_rate '+str(args.learning_rate)+' --steps_per_epoch '+str(args.steps_per_epoch))
    print('Training ended on '+time.asctime(time.localtime()))
    if not args.autotest == None:
        print('Running automatic testing of model on '+args.autotest[0]+' dataset')
        if len(args.autotest)>1:
            for i in args.autotest[1:]:
                test(args, dataset=args.autotest[0], qual=i)
            full_results = open(full_model_path(args)+'/'+args.model_path+'_'+args.autotest[0]+'_results.csv','w')
            lines = []
            for qual in args.autotest[1:]:
                partial_results = open(full_model_path(args)+'/'+args.model_path+'_'+args.autotest[0]+'_qual-'+str(qual)+'_results.csv','r')
                if qual == args.autotest[1:][1]:
                    lines += partial_results.readlines()
                else:
                    lines += partial_results.readlines()[1:]
                os.system('rm '+full_model_path(args)+'/'+args.model_path+'_'+args.autotest[0]+'_qual-'+str(qual)+'_results.csv')
            for line in lines:
                full_results.write(line)
            full_results.close()
        else:
            test(args, dataset=args.autotest[0])
        print('Testing ended on '+time.asctime(time.localtime()))
    
def test(args,dataset=None,qual=None):
    print('Delaying for '+str(args.delay)+' seconds...')
    time.sleep(args.delay)
    if not qual == None and float(qual) == int(float(qual)):
        qual = int(qual)
    if dataset==None:
        dataset=args.dataset
    try:
    	corpus = random.sample(os.listdir('../datasets/'+dataset),args.sample)
    except:
    	corpus = os.listdir('../datasets/'+dataset)
    if qual == None:
        results = open(full_model_path(args)+'/'+args.model_path+'_'+dataset+'_results.csv','w')
    else:
        results = open(full_model_path(args)+'/'+args.model_path+'_'+dataset+'_qual-'+str(qual)+'_results.csv','w')
    results.write('Image,')
    if not qual == None:
        results.write('Quality parameter,')
    results.write('TFCI size,Rate (bps),MSE')
    try:
        if args.SSIM:
            results.write(',MS-SSIM')
        if args.SAM:
            results.write(',SAM (radians)')
        if args.MAE:
            results.write(',MAE')
        if args.PAE:
            results.write(',PAE')
    except:
        pass
    results.write ('\n')
    bands, width, height, endianess, datatype = get_geometry_dataset('../datasets/'+dataset)
    
    if qual==None:
        print('Compressing...')
        os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' compress "'+'../datasets/'+dataset+'/*.'+args.extension+'"')
    else:
        print('Compressing...')
        os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' compress "'+'../datasets/'+dataset+'/*.'+args.extension+'" --quality '+str(qual))
    print('Decompressing...')
    os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' decompress "'+'../datasets/'+dataset+'/*.'+args.extension+'.tfci"')

        
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
    while not os.path.splitext(sanity_check)[1] == '.raw' and args.extension=='raw':
        sanity_check = random.choice(corpus)
    
    for IMAGE in corpus:
        print('Testing image '+IMAGE)
        if os.path.splitext(IMAGE)[1] == '.raw' and args.extension == 'raw':
            bands, width, height, endianess, datatype = get_geometry_file(IMAGE)
            path_to_image = '../datasets/'+dataset+'/'+IMAGE
            results.write(IMAGE+',')
            if not qual == None:
                results.write(str(qual)+',')
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
                if args.SAM:
                    sam_res = sam(path_to_image, raw_tfci_path_to_image, height=int(height), bands=int(bands), width=int(width), endianess=int(endianess), data_type=D)
                    results.write(','+str(sam_res))
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
                IMG0.save(full_model_path(args)+'/'+args.model_path+'_'+dataset+'_sanity-check_org_'+sanity_check[:-4]+'_quality-'+str(qual)+'.png')
                IMG1.save(full_model_path(args)+'/'+args.model_path+'_'+dataset+'_sanity-check_new_'+sanity_check[:-4]+'_quality-'+str(qual)+'.png')
                os.system('cp '+path_to_image+' ./')
                os.system('mv ./'+sanity_check+' sanity-check.raw')
                os.system('java -Xmx4096m -classpath ../BOI/dist/boi.jar boi.Encoder -i "./sanity-check.raw" -ig '+bands+' '+height+' '+width+' '+datatype+' 0 '+str(8*int(datatype))+' '+endianess+' 0 0 -r '+str(bps)+' -f 1')
                os.system('java -Xmx4096m -classpath ../BOI/dist/boi.jar boi.Decoder -i "./sanity-check.jpc" -og '+datatype+' '+endianess+' 0 -o "./sanity-check.jpc.raw"')
                jpg2k = Image.fromarray(np.reshape(np.fromfile('./sanity-check.jpc.raw',dtype=D),(int(bands),int(height),int(width)))[0,:,:])
                jpg2k.save(full_model_path(args)+'/'+args.model_path+'_'+dataset+'_sanity-check_JPEG2000_'+sanity_check[:-4]+'_quality-'+str(qual)+'.png')
                os.system('rm ./sanity-check.raw')
                os.system('rm ./sanity-check.jpc')
                os.system('rm ./sanity-check.jpc.raw')

            os.system('rm '+tfci_path_to_image)
            os.system('rm '+raw_tfci_path_to_image)
            results.write('\n')
        elif os.path.splitext(IMAGE)[1] == '.png' and args.extension == 'png':
            path_to_image = '../datasets/'+dataset+'/'+IMAGE
            results.write(IMAGE+',')
            tfci_path_to_image = os.path.splitext(path_to_image)[0]+'.png.tfci'
            png_tfci_path_to_image = os.path.splitext(path_to_image)[0]+'.png.tfci.png'
            compressed_size = os.stat(tfci_path_to_image)[6]
            img0 = np.asarray(Image.open(path_to_image)).astype(np.float32)
            img1 = np.asarray(Image.open(png_tfci_path_to_image)).astype(np.float32)
            mse = np.mean((img0-img1)**2)
            bps = compressed_size*8/(img0.size)
            results.write(str(compressed_size)+','+str(bps)+','+str(mse))
            if args.MAE:
                mae = np.mean(abs(img0-img1))
                results.write(','+str(mae))
            if not IMAGE==sanity_check:
                os.system('rm '+png_tfci_path_to_image)
            os.system('rm '+tfci_path_to_image)
            results.write('\n')
            os.system('mv '+'../datasets/'+dataset+'/'+sanity_check+' '+full_model_path(args)+'/'+args.model_path+'_'+dataset+'_sanity-check_org.png')
            os.system('mv '+'../datasets/'+dataset+'/'+sanity_check+'.tfci.png '+full_model_path(args)+'/'+args.model_path+'_'+dataset+'_sanity-check_new.png')
    results.close()
    
def visualise(args):
    print('Delaying for '+str(args.delay)+' seconds...')
    time.sleep(args.delay)
    if not args.quality == None and float(args.quality) == int(float(args.quality)):
        args.quality = int(args.quality)
    corpus = random.sample(os.listdir('../datasets/'+args.dataset),args.sample)    
    if args.quality==None:
        print('Compressing...')
        os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' compress "'+'../datasets/'+args.dataset+'/*.'+args.extension+'"')
    else:
        print('Compressing...')
        os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' compress "'+'../datasets/'+args.dataset+'/*.'+args.extension+'" --quality '+str(args.quality))
    print('Decompressing...')
    os.system('python3 '+baseline_path(args)+' --model_path '+full_model_path(args)+' decompress "'+'../datasets/'+args.dataset+'/*.'+args.extension+'.tfci"')
    for IMAGE in corpus:
        print('Recovering image '+IMAGE)
        if os.path.splitext(IMAGE)[1] == '.raw' and args.extension == 'raw':
            bands, width, height, endianess, datatype = get_geometry_file(IMAGE)
            
            if datatype == '0':
                D = np.bool_
            elif datatype == '1':
                D = np.uint8
            elif datatype == '2':
                D = np.uint16
            elif datatype == '3':
                D = np.int16
            elif datatype == '4':
                D = np.int32
            elif datatype == '5':
                D = np.int64
            elif datatype == '6':
                D = np.float32
            else:
                D = np.float64
            
            path_to_image = '../datasets/'+args.dataset+'/'+IMAGE
            tfci_path_to_image = os.path.splitext(path_to_image)[0]+'.raw.tfci'
            raw_tfci_path_to_image = os.path.splitext(path_to_image)[0]+'.raw.tfci.raw'
            if args.keep_originals:
                img0 = np.reshape(np.fromfile(path_to_image,dtype=D),(int(bands),int(height),int(width)))
            img1 = np.reshape(np.fromfile(raw_tfci_path_to_image,dtype=D),(int(bands),int(height),int(width)))
            if args.all_bands:
                for b in range(int(bands)):
                    IMG1 = Image.fromarray(img1[b,:,:])
                    IMG1.save(full_model_path(args)+'/'+args.model_path+'_'+IMAGE[:-4]+'_model-reconstruction_quality-'+str(args.quality)+'_band-'+str(b)+'.png')
                    if args.keep_originals:
                        IMG0 = Image.fromarray(img0[b,:,:])
                        IMG0.save(full_model_path(args)+'/'+args.model_path+'_'+IMAGE[:-4]+'_original_band-'+str(b)+'.png')
                    if args.keep_JPEG2000:
                        bps = os.stat(tfci_path_to_image)[6]*8/(img1.size)
                        os.system('cp '+path_to_image+' ./')
                        os.system('java -Xmx4096m -classpath ../BOI/dist/boi.jar boi.Encoder -i "./'+IMAGE+'" -ig '+bands+' '+height+' '+width+' '+datatype+' 0 '+str(8*int(datatype))+' '+endianess+' 0 0 -r '+str(bps)+' -f 1')
                        os.system('java -Xmx4096m -classpath ../BOI/dist/boi.jar boi.Decoder -i "./'+IMAGE[:-4]+'.jpc" -og '+datatype+' '+endianess+' 0 -o "./'+IMAGE[:-4]+'.jpc.raw"')
                        jpg2k = Image.fromarray(np.reshape(np.fromfile('./'+IMAGE[:-4]+'.jpc.raw',dtype=D),(int(bands),int(height),int(width)))[0,:,:])
                        jpg2k.save(full_model_path(args)+'/'+args.model_path+'_'+IMAGE[:-4]+'_JPEG2000-reconstruction_quality-'+str(args.quality)+'.png')
                        os.system('rm ./'+IMAGE)
                        os.system('rm ./'+IMAGE[:-4]+'.jpc')
                        os.system('rm ./'+IMAGE[:-4]+'.jpc.raw')

            else:
                b = random.randint(0,int(bands)-1)
                IMG1 = Image.fromarray(img1[b,:,:])
                IMG1.save(full_model_path(args)+'/'+args.model_path+'_'+IMAGE[:-4]+'_model-reconstruction_quality-'+str(args.quality)+'.png')
                if args.keep_originals:
                    IMG0 = Image.fromarray(img0[b,:,:])
                    IMG0.save(full_model_path(args)+'/'+args.model_path+'_'+IMAGE[:-4]+'_original.png')
                if args.keep_JPEG2000:
                    bps = os.stat(tfci_path_to_image)[6]*8/(img1.size)
                    os.system('cp '+path_to_image+' ./')
                    os.system('java -Xmx4096m -classpath ../BOI/dist/boi.jar boi.Encoder -i "./'+IMAGE+'" -ig '+bands+' '+height+' '+width+' '+datatype+' 0 '+str(8*int(datatype))+' '+endianess+' 0 0 -r '+str(bps)+' -f 1')
                    os.system('java -Xmx4096m -classpath ../BOI/dist/boi.jar boi.Decoder -i "./'+IMAGE[:-4]+'.jpc" -og '+datatype+' '+endianess+' 0 -o "./'+IMAGE[:-4]+'.jpc.raw"')
                    jpg2k = Image.fromarray(np.reshape(np.fromfile('./'+IMAGE[:-4]+'.jpc.raw',dtype=D),(int(bands),int(height),int(width)))[0,:,:])
                    jpg2k.save(full_model_path(args)+'/'+args.model_path+'_'+IMAGE[:-4]+'_JPEG2000-reconstruction_quality-'+str(args.quality)+'.png')
                    os.system('rm ./'+IMAGE)
                    os.system('rm ./'+IMAGE[:-4]+'.jpc')
                    os.system('rm ./'+IMAGE[:-4]+'.jpc.raw')

            os.system('rm ../datasets/'+args.dataset+'/*.tfci*')

def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--hyperprior", "-H", action="store_true",
      help="This is a hyperprior model and the train command will include the scales parameters.")
  parser.add_argument(
      "--model_path", default="test_model",
      help="Code under which to save/load the trained model. This will be in a predefined directory.")
  parser.add_argument(
      "--architecture", default="CSFL2022012501",
      help="Baseline architecture to be trained or tested. Just use the code.")
  parser.add_argument(
      "--extension", default="raw",
      help="Extension of the data the network processes.")
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
      "--num_filters_1D", type=int, default=4,
      help="Number of filters in the 1D layer, if any.")
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
      help="Only for hyperprior models. Number of Gaussian scales to prepare range coding tables for.")
  train_cmd.add_argument(
      "--scale_min", type=float, default=0.11, dest="scale_min",
      help="Only for hyperprior models. Minimum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--scale_max", type=float, default=256.0, dest="scale_max",
      help="Only for hyperprior models. Maximum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--autotest", nargs='+', default= None,
      help="Run testing automatically at the end of training. It will use the dataset "
      "indicated in this option, as well as the quality parameters indicated after it. "
      "Example usage: --autotest some-data-testset 1 2 4")

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
        "--quality", type=float, nargs="+", default=None,
        help="Quality parameters to be used (n>0 floats or integers) in testing. They will be all written in a single results file while saving multiple sanity check images.")
  test_cmd.add_argument(
        "--SSIM", action="store_true",
        help="Computes MS-SSIM distortion.")
  test_cmd.add_argument(
        "--SAM", action="store_true",
        help="Computes Spectral Angle Mapper (SAM) distortion.")
  test_cmd.add_argument(
        "--MAE", action="store_true",
        help="Computes Mean Absolute Error (MAE, L1 norm) distortion.")
  test_cmd.add_argument(
        "--PAE", action="store_true",
        help="Computes Peak Absolute Error (PAE, L-infinity norm) distortion.")
  
  visualise_cmd = subparsers.add_parser(
      "visualise",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Applies a trained model on a random image (or sample) from a dataset"
                  "and saves a visual example in the model's folder.")

  # Arguments for test command.
  visualise_cmd.add_argument(
        "--dataset", type=str, default="LandSat8_cropRGB",
        help="Source dataset. Its geometry will be automatically loaded.")
  visualise_cmd.add_argument(
        "--sample", type=int, default=1,
        help="Maximum sample size.")
  visualise_cmd.add_argument(
        "--quality", type=float,  default=None,
        help="Quality parameter to be used in compression. Only one parameter.")
  visualise_cmd.add_argument(
        "--keep_originals", action="store_true",
        help="Save the original image as well as the model's reconstruction to be visualised.")
  visualise_cmd.add_argument(
        "--keep_JPEG2000", action="store_true",
        help="Save the JPEG 2000 reconstruction as well as the model's reconstruction to be visualised.")
  visualise_cmd.add_argument(
        "--all_bands", action="store_true",
        help="Save visualisations from all bands in the image. If false, only saves a single band chosen at random.")

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
    if args.quality:
        for i in args.quality:
            test(args,qual=i)
        full_results = open(full_model_path(args)+'/'+args.model_path+'_'+args.dataset+'_results.csv','w')
        lines = []
        for qual in args.quality:
            partial_results = open(full_model_path(args)+'/'+args.model_path+'_'+args.dataset+'_qual-'+str(qual)+'_results.csv','r')
            if qual == args.quality[0]:
                lines += partial_results.readlines()
            else:
                lines += partial_results.readlines()[1:]
            os.system('rm '+full_model_path(args)+'/'+args.model_path+'_'+args.dataset+'_qual-'+str(qual)+'_results.csv')
        for line in lines:
            full_results.write(line)
        full_results.close()
        
    else:
      test(args)
  elif args.command == "visualise":
      visualise(args)

if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
