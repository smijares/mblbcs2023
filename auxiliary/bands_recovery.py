#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consecutive bands clusters reconstructer
V1.2
Sebastià Mijares i Verdú - GICI, UAB
sebastia.mijares@uab.cat

Reconstructs images from slices with our standard naming convention in .raw format stored in ../../datasets/<some_repo>_aux .

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
    if file[-13:]=='.raw.tfci.raw':
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
      "--source", "-H", default="AVIRIS_aux",
      help="Auxiliary directory to be recovered. To be stored as .raw.tfci.raw images using our standard naming format.")
  parser.add_argument(
      "--destination", default="AVIRIS",
      help="Destination of the recovered bands.")
  parser.add_argument(
      "--consecutive_bands", type=int, default=1,
      help="Number of consecutive bands that were sliced.")
  parser.add_argument(
      "--min_band", type=int, default=0,
      help="Minimum band (included) to be extracted.")
  parser.add_argument(
      "--max_band", type=int,
      help="Maximum band (included) to be recovered.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="Use 'recover' to run the recoverer with the corresponding script arguments.")

  # 'train' subcommand.
  recover_cmd = subparsers.add_parser(
      "recover",
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
      if image[-13:] == '.raw.tfci.raw':
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
        
        name_aux = image.split('.')[0]
        
        name = ''
        if len(name_aux.split('_b'))>2:
            for x in range(len(name_aux.split('_b'))-1):
                name += name_aux.split('_b')[x]+'_b'
            name = name[:-2]
        else:
            name += name_aux.split('_b')[0]
        
        b = args.min_band
        
        if not os.path.exists('../datasets/'+args.destination+'/'+name+'.'+str(args.max_band-args.min_band+1)+'_'+width+'_'+height+'_'+datatype+'_1_0.raw.tfci.raw'):
                
            if args.min_band:
                b = args.min_band
            else:
                b = 0
            first = True
            while b+args.consecutive_bands<=args.max_band+1:
                arr = np.reshape(np.fromfile('../datasets/'+args.source+'/'+name+'_b'+str(b+1)+'-'+str(b+int(args.consecutive_bands))+'.'+str(args.consecutive_bands)+'_'+width+'_'+height+'_'+datatype+'_1_0.raw.tfci.raw',dtype=D),(int(bands),int(height),int(width)))
                if first:
                    img = arr
                    first = not first
                else:
                    img = np.concatenate((img, arr), axis=0)
                b+=args.consecutive_bands
            
            if (args.max_band+1-args.min_band)%args.consecutive_bands > 0:
                arr = np.reshape(np.fromfile('../datasets/'+args.source+'/'+name+'_b'+str(b+1)+'-'+str(b+int(args.consecutive_bands))+'.'+str(args.consecutive_bands)+'_'+width+'_'+height+'_'+datatype+'_1_0.raw.tfci.raw',dtype=D),(int(bands),int(height),int(width)))
                img = np.concatenate((img, arr), axis=0)
                
            img = img[:args.max_band-args.min_band+1,:,:]
            img.tofile('../datasets/'+args.destination+'/'+name+'.'+str(args.max_band-args.min_band+1)+'_'+width+'_'+height+'_'+datatype+'_1_0.raw.tfci.raw')
            
if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)