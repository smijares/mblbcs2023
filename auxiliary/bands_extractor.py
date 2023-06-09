#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consecutive bands generator
V1.1
Sebastià Mijares i Verdú - GICI, UAB
sebastia.mijares@uab.cat

Slices images with our standard naming convention in .raw format stored in ../../datasets/<some_repo> into auxiliary images with a fixed number of consecutive channels.

Requiremed libraries
--------------------

os
numpy
argparse
sys
absl

"""

import numpy as np
import os
import argparse
import sys
from absl import app
from absl.flags import argparse_flags

def get_geometry(file):
    if file[-4:]=='.raw':
        G = file.split('.')[1].split('_')
        bands = G[0]
        width = G[1]
        height = G[2]
        datatype = G[3]
        endianess = G[4]
        #Note these are all strings, not int numbers!
        return (bands, width, height, endianess, datatype)

def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--source", "-H", default="AVIRIS",
      help="Source directory to be sliced. To be stored as .raw images using our standard naming format.")
  parser.add_argument(
      "--destination", default="AVIRIS_aux",
      help="Destination of the exrtracted bands. If the directory doesn't exist, it will be created.")
  parser.add_argument(
      "--consecutive_bands", type=int, default=1,
      help="Number of consecutive bands to be sliced.")
  parser.add_argument(
      "--min_band", type=int, default=0,
      help="Minimum band (included) to be extracted.")
  parser.add_argument(
      "--max_band", type=int, default=None,
      help="Maximum band (included) to be extracted.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="Use 'extract' to run the extractor with the corresponding script arguments.")

  # 'train' subcommand.
  extract_cmd = subparsers.add_parser(
      "extract",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Runs script.")
  
  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  corpus = os.listdir('../datasets/'+args.source)
  if not os.path.isdir('../datasets/'+args.destination):
      os.mkdir('../datasets/'+args.destination)
  for image in corpus:
      if image[-4:] =='.raw':
        print(image)
        bands, width, height, endianess, datatype = get_geometry(image)
        
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
        
        name = image.split('.')[0]
        img = np.reshape(np.fromfile('../datasets/'+args.source+'/'+image,dtype=D),(int(bands),int(height),int(width)))
        
        if args.min_band:
            b = args.min_band
        else:
            b = 0
        if args.max_band:
            max_band = args.max_band
        else:
            max_band = int(bands)-1
            
        while b+args.consecutive_bands<=max_band+1:
            arr = img[b:b+args.consecutive_bands,:,:]
            arr.tofile('../datasets/'+args.destination+'/'+name+'_b'+str(b+1)+'-'+str(b+args.consecutive_bands)+'.'+str(args.consecutive_bands)+'_'+width+'_'+height+'_'+datatype+'_1_0.raw')
            b+=args.consecutive_bands
            
        if (max_band+1-args.min_band)%args.consecutive_bands > 0:
            arr0 = img[-((max_band+1-args.min_band)%args.consecutive_bands):,:,:]
            null_arr = np.zeros((args.consecutive_bands-((max_band+1-args.min_band)%args.consecutive_bands), arr0.shape[1], arr0.shape[2]), dtype=D)
            arr = np.concatenate((arr0,null_arr),axis=0)
            arr.tofile('../datasets/'+args.destination+'/'+name+'_b'+str(max_band+1-((max_band+1-args.min_band)%args.consecutive_bands)+1)+'-'+str((max_band+1-((max_band+1-args.min_band)%args.consecutive_bands))+args.consecutive_bands)+'.'+str(args.consecutive_bands)+'_'+width+'_'+height+'_'+datatype+'_1_0.raw')

if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)