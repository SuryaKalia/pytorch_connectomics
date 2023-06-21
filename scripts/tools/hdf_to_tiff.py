import sys
import h5py
import numpy as np
from pathlib import Path
from tifffile import imsave

usage = "<input hdf5 filepath> <hdf internal path>  <output_file_path>\n\
        This script will convert an image matrix stored within an hdf5 file into a tiff stack \n"

def main():
  if (len(sys.argv) != 4):
    print(usage)
    exit()

  input_file_path = sys.argv[1]
  internal_path = sys.argv[2]
  output_file_path = sys.argv[3]
  
  f = h5py.File(input_file_path, 'r')
  raw_data = np.array(f[internal_path])
  
  imsave(output_file_path, raw_data)


if __name__ == "__main__":
    main()