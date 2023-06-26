import numpy as np
import sys
import cv2
from tifffile import imsave
from skimage import io
import matplotlib.pyplot as plt

usage = "<input tiff filepath>  <output_file_path> <filter threshold (0-255)>\n\
        This script will identify 26-connected components (3D) from tiff image stack and return a connected component tiff file \n"

def connect_3d(img, kernel_size):
  # connected_img = np.zeros(img.shape)
  # kernel = np.ones((kernel_size, kernel_size, kernel_size))
  padding_width = kernel_size//2
  padded_img = np.pad(img, ((padding_width,padding_width), (padding_width,padding_width), (padding_width,padding_width)), 'minimum')
  print("Padded img", padded_img.shape)
  output = np.zeros(padded_img.shape).astype(np.uint16)
  component_id = 0
  labels = np.arange(img.shape[0]*img.shape[1]*img.shape[2]).astype(np.uint16)
  
  def union (x, y) :
      labels[find(x)] = find(y)


  def find(x):
    y = x
    while (labels[y] != y):
      # print("y = ", y)
      y = labels[y]

    while (labels[x] != x):
      # print("x = ", x)
      z = labels[x]
      labels[x] = y
      x = z
    return y

  print(img.shape)
  
  for z in range (padding_width, img.shape[0] + padding_width):
    for y in range (padding_width, img.shape[1] + padding_width):
      for x in range (padding_width, img.shape[2] + padding_width):
        # img_crop = padded_img[z-padding_width:z+padding_width+1, y-padding_width:y+padding_width+1, x-padding_width:x+padding_width+1]
        # print("X:", x, "Y:", y, "Z:", z, img_crop.shape)
        # output_crop = output[z-padding_width:z+padding_width+1, y-padding_width:y+padding_width+1, x-padding_width:x+padding_width+1]
        # Using an optimized kernel of only 12 values instead of 27 values
        output_crop = output[z-padding_width:z+padding_width, y-padding_width:y+padding_width, x-padding_width:x+padding_width+1]

        
        if (padded_img[z, y, x] > 0):
          # Non zero pixel found. Assign a component value
          max_val = output_crop.max()
          if (max_val == 0):
            # No neighbors in vicinity. Start new component
            component_id += 1
            output[z,y,x] = component_id
          else :
            # Assign the max value among neighbors to the 
            output[z,y,x] = max_val
            for val in np.nditer(output_crop):
              if (val != 0):
                # print("Union between ", val, " ", max_val)
                # print("Labels: ", labels[0:10])
                union(int(val), int(max_val))
    print("X:", x, "Y:", y, "Z:", z)
  
  # Second pass for assigning final label values
  label_map = {}
  curr_label = 1
  for z in range (padding_width, img.shape[0] + padding_width):
    for y in range (padding_width, img.shape[1] + padding_width):
      for x in range (padding_width, img.shape[2] + padding_width):
        if (output[z, y, x] != 0):
          # Re-map the union-find label values to a smaller coontinuous range of label values from 1 onwards
          original_label = find(int(output[z, y, x]))
          if (original_label in label_map):
            output[z, y, x] = label_map[original_label]
          else:
            label_map[original_label] = curr_label
            curr_label += 1
            output[z, y, x] = label_map[original_label]

  return output[padding_width:-padding_width, padding_width:-padding_width, padding_width:-padding_width]

def main():
  if (len(sys.argv) != 4):
    print(usage)
    exit()

  input_file_path = sys.argv[1]
  output_file_path = sys.argv[2]
  filter_threshold = int(sys.argv[3])
  # img = cv2.imread(input_file_path)
  # sklearm IO returns (chan, z, y, x) shaped output here
  img = io.imread(input_file_path)
  img = img[0, :, :, :]
  
  img[img < filter_threshold] = 0
  
  # plt.imshow(img[0, :, :], interpolation='nearest')
  # plt.savefig("source_img_for_connected_comp.jpg")
  
  # # Slice for debugging
  # img = img[[0], :, :]
  
  # Create a mask so that all non zero values become 1
  img[img > 0]  = 1
  kernel_size = 3
  connected_img = connect_3d(img, kernel_size)  
  
  # plt.clf()
  # plt.imshow(connected_img[0, :, :], interpolation='nearest')
  # plt.savefig("output_img_for_connected_comp.jpg")
  
  imsave(output_file_path, connected_img)

if __name__ == "__main__":
    main()