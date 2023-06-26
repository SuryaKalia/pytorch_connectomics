import sys
import h5py
import numpy as np
from pathlib import Path
from tifffile import imsave

usage = "<input hdf5 filepath> <hdf internal path>  <output_file_path> <dtype>\n\
        This script will convert an image matrix stored within an hdf5 file into a tiff stack \n"

def main():
  if (len(sys.argv) != 5):
    print(usage)
    exit()

  input_file_path = sys.argv[1]
  internal_path = sys.argv[2]
  output_file_path = sys.argv[3]
  dtype_str = sys.argv[4]
  
  f = h5py.File(input_file_path, 'r')
  
  dtype = np.uint64
  
  if (dtype_str == "uint8"):
    dtype = np.uint8
  elif (dtype_str == "uint16"):
    dtype = np.uint16
  elif (dtype_str == "uint32"):
    dtype = np.uint32
    
  raw_data = np.array(f[internal_path]).astype(dtype)
  
  imsave(output_file_path, raw_data)


if __name__ == "__main__":
    main()