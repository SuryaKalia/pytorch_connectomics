import sys
import h5py

usage = "python3 ls_hdf.py <filename>\n This script will display the internal file structure of the given hdf file"

def print_details(name, obj):
  print(name, "|", obj)

def main():
  if (len(sys.argv) != 2):
    print(usage)
    exit()

  file_path = sys.argv[1]
  f = h5py.File(file_path, 'r')
  f.visititems(print_details)

if __name__ == "__main__":
    main()
