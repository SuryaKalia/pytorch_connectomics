from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats
import pickle

usage = "<predicted synapse component filepath> <labelled synapse groundtruth filepath> <IOU threshold> <voxel threshold> <checkpoint_path> <load_maps>\n\
        Calculate prediction accuracy using provided IOU and volume threshold \n"

def load_pkl(filepath):
  with open(filepath, "rb") as pkl_handle:
    output = pickle.load(pkl_handle)
    return output
      
def store_pkl(dict, filepath):
  with open(filepath, "wb") as pkl_handle:
    pickle.dump(dict, pkl_handle)

def good_overlap(pred_gt_overlap_volume, pred_volume, gt_volume, iou_threshold):
  intersection = pred_gt_overlap_volume
  union = pred_volume + gt_volume - pred_gt_overlap_volume
  assert union > 0 , "Negative union volume!"
  assert intersection <= union, "Invalid intersection volume!"
  return (intersection/union) >= iou_threshold

def calculate_threshold_accuracy(pred_label_count, gt_label_count, pred_label_mapping, gt_label_mapping, iou_threshold, voxel_threshold):
  # Clean out labels with volume lesser than voxel_threshold
  for pred_label in list(pred_label_count.keys()):
    volume = pred_label_count[pred_label]
    if volume < voxel_threshold:
      del pred_label_count[pred_label]
      del pred_label_mapping[pred_label]
      for gt_label in list(gt_label_mapping.keys()):
        map = gt_label_mapping[gt_label]
        if pred_label in map:
          del gt_label_mapping[gt_label][pred_label]
  
  # Calculate true positive and false positive rate by inspecting pred_label_mapping
  # True Positive (TP): A correct detection. Detection with IOU ≥ threshold
  # False Positive (FP): A wrong detection. Detection with IOU < threshold
  tp = 0
  fp = 0
  for pred_label, pred_volume in pred_label_count.items():
    # Each pred label could have overlap with multiple gt labels. Will count it as 1 tp/fp only depending on if at-least one overlap satisfies IOU threshold
    if pred_label == 0 :
      # Ignore background voxel
      continue
    found_match = False
    for gt_label, pred_gt_overlap_volume in pred_label_mapping[pred_label].items():
      if gt_label == 0 :
        # Ignore backgorund voxels
        continue
      gt_volume = gt_label_count[gt_label]
      if (good_overlap(pred_gt_overlap_volume, pred_volume, gt_volume, iou_threshold)):
        # Found good IOU overlap. Count this pred_label as true positive 
        tp += 1
        found_match = True
        break
    # Check if match was found
    if (not found_match):
      # It was a false positive label
      fp += 1
  
  print("TP:", tp)
  print("FP:", fp)
  print("pred_label_count:", pred_label_count)
  print("gt_label_count:", gt_label_count)
  print("")
  
  
  assert tp + fp == len(pred_label_count)-1 , "Mismatch in tp + fp and total number of filtered predictions (except background)"
  
  fn = 0
  # Calculate false negative rate by inspecting gt_label_mapping
  for gt_label, gt_mapping_volumes in gt_label_mapping.items():
    # Check if gt label overlaps with any valid prediction labels
    # False negative if no overlap found
    pred_label_list = list(gt_mapping_volumes.keys())
    print("GT_Label: ", gt_label, " Mapping Volumes: ", gt_mapping_volumes)
    if (len(pred_label_list) == 1 and pred_label_list[0] == 0):
      # Only overlap was with background pixels. This is a false negative
      fn += 1
  
  return tp, fp, fn, len(pred_label_count)-1, len(gt_label_count)-1
  
def calculate_accuracy(pred_img, gt_img, iou_threshold, voxel_threshold, checkpoint_path, load_maps):
  pred_label_count = {}
  gt_label_count = {}
  pred_label_mapping = {}
  gt_label_mapping = {}
  if load_maps :
    # Load pkl label maps from checkpoint path
    pred_label_count = load_pkl(checkpoint_path + "/pred_label_count.pkl")
    gt_label_count = load_pkl(checkpoint_path + "/gt_label_count.pkl")
    pred_label_mapping = load_pkl(checkpoint_path + "/pred_label_mapping.pkl")
    gt_label_mapping = load_pkl(checkpoint_path + "/gt_label_mapping.pkl")
    
  else:
    # Compute own label maps and store them at checkpoint path
    print("Construct voxel count maps")
    dims = pred_img.shape
    for z in range (dims[0]):
      print("Z layer num:", z)
      for y in range (dims[1]):
        for x in range (dims[2]):
          pred_label = pred_img[z, y, x]
          gt_label = gt_img[z, y, x]
          
          if (pred_label == 0 and gt_label == 0):
            # Both background voxels. Ignore
            continue
          
          # Update label counts for pred and gt
          if (pred_label in pred_label_count):
            pred_label_count[pred_label] += 1
          else:
            pred_label_count[pred_label] = 1
            pred_label_mapping[pred_label] = {}

          if (gt_label in gt_label_count):
            gt_label_count[gt_label] += 1
          else:
            gt_label_count[gt_label] = 1
            gt_label_mapping[gt_label] = {}
            
          # Update mappings of overlapping labels
          if (gt_label in pred_label_mapping[pred_label]):
            pred_label_mapping[pred_label][gt_label] += 1
          else:
            pred_label_mapping[pred_label][gt_label] = 1
          
          if (pred_label in gt_label_mapping[gt_label]):
            gt_label_mapping[gt_label][pred_label] += 1
          else:
            gt_label_mapping[gt_label][pred_label] = 1
    
    # Save maps to disk
    store_pkl(pred_label_count, checkpoint_path + "/pred_label_count.pkl")
    store_pkl(gt_label_count, checkpoint_path + "/gt_label_count.pkl")
    store_pkl(pred_label_mapping, checkpoint_path + "/pred_label_mapping.pkl")
    store_pkl(gt_label_mapping, checkpoint_path + "/gt_label_mapping.pkl")

    
    
    
  print("Num predicted labels = ", len(pred_label_count))
  print("Num groundtruth labels = ", len(gt_label_count))
  
  # Print stats of ground truth and prediction volumes
  print("Predicted stats: ", stats.describe(list(pred_label_count.values())))
  print("Ground Truth stats: ", stats.describe(list(gt_label_count.values())))
  
    
  # Refer to https://github.com/rafaelpadilla/Object-Detection-Metrics for variable name explanation
  tp, fp, fn, num_detections, num_truths = calculate_threshold_accuracy(pred_label_count, gt_label_count, pred_label_mapping, gt_label_mapping, iou_threshold, voxel_threshold)
  
  print("TP = ", tp)
  print("FP = ", fp)
  print("FN = ", fn)
  print("Num Detections = ", num_detections)
  print("Num Truths = ", num_truths)
  

def main():
  if (len(sys.argv) != 7):
    print(usage)
    exit()

  
  predicton_file_path = sys.argv[1]
  groundtruth_file_path = sys.argv[2]
  iou_threshold = float(sys.argv[3])
  voxel_threshold = int(sys.argv[4])
  checkpoint_path = sys.argv[5]
  load_maps = sys.argv[6] == "True"
  
  if load_maps :
    # Images are dummy
    pred_img = None
    gt_img = None
  else :
    pred_img = io.imread(predicton_file_path)
    gt_img = io.imread(groundtruth_file_path)
    
    assert pred_img.shape == gt_img.shape, "Shape mismatch between prediction and ground truth"
    
    # # Slice for debugging
    # pred_img = pred_img[:1, :, :]
    # gt_img = gt_img[:1, :, :]
    
    # plt.imshow(pred_img[0, :, :], interpolation='nearest')
    # plt.savefig("pred_slice.jpg")
    
    # plt.clf()
    # plt.imshow(gt_img[0, :, :], interpolation='nearest')
    # plt.savefig("gt_slice.jpg")
    
    print("Input shape = ", pred_img.shape)
    
  calculate_accuracy(pred_img, gt_img, iou_threshold, voxel_threshold, checkpoint_path, load_maps)
  # print("Overall accuracy = ", accuracy)

if __name__ == "__main__":
    main()
    
    
# References:
# Accuracy metrics for detection: https://github.com/rafaelpadilla/Object-Detection-Metrics