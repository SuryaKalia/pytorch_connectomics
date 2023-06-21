import sys
import h5py
import numpy as np
from pathlib import Path

usage = "<input hdf5 filepath> <output_dir>  <output_filename_prefix>\n\
        This script will display the internal file structure of the given hdf file\n"

def print_details(name, obj):
  print(name, "|", obj)

def main():
  if (len(sys.argv) != 4):
    print(usage)
    exit()

  input_file_path = sys.argv[1]
  output_dir = sys.argv[2]
  output_filename_prefix = sys.argv[3]
  
  Path(output_dir + "/images/").mkdir(parents=True, exist_ok=True)
  Path(output_dir + "/labels/").mkdir(parents=True, exist_ok=True)
  

  f = h5py.File(input_file_path, 'r')
  f.visititems(print_details)
  
  img_output = h5py.File(output_dir + "/images/im_" + output_filename_prefix + ".h5" , 'w')
  label_output = h5py.File(output_dir + "/labels/label_" + output_filename_prefix + ".h5" , 'w')
  
  
  # Replace backgorund voxel label value from 2^64-1 to 0 for processing by pytorch connectomics
  raw_label = np.array(f["volumes/labels/clefts"])
  raw_label[raw_label == int(18446744073709551615)] = int(0)
  
  img_output.create_dataset('images', data=np.array(f["volumes/raw"]), compression="gzip", compression_opts=7)
  img_output.close()
  
  label_output.create_dataset('labels', data=raw_label, compression="gzip", compression_opts=7)
  label_output.close()


if __name__ == "__main__":
    main()