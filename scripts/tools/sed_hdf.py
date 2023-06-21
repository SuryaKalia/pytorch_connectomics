import sys
import h5py
import numpy as np
from pathlib import Path

usage = "<input hdf5 filepath> <output_dir>  <output_filename> <source value> <target value>\n\
        This script will replace given source value to target value in provided hdf file and write to new file\n"

def main():
  if (len(sys.argv) != 6):
    print(usage)
    exit()

  input_file_path = sys.argv[1]
  output_dir = sys.argv[2]
  output_filename = sys.argv[3]
  source_value = sys.argv[4]
  target_value = sys.argv[5]
  
  Path(output_dir).mkdir(parents=True, exist_ok=True)
  
  f = h5py.File(input_file_path, 'r')
  raw_data = np.array(f['labels'])
  raw_data[raw_data == int(source_value)] = int(target_value)
  
  label_output = h5py.File(output_dir + output_filename, 'w')

  label_output.create_dataset('labels', data=raw_data, compression="gzip", compression_opts=7)
  label_output.close()


if __name__ == "__main__":
    main()