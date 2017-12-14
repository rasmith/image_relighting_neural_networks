import sys
import time
import glob
import os
from scipy import misc
import numpy as np
from multiprocessing import Pool
from model import ModelMaker
import tensorflow as tf
from keras import backend as K
import matplotlib.image as mpimg

from cluster import *

# width = 696
# height = 464
def get_parameters(directory):
  os.chdir(directory)
  png_files = glob.glob("*.png")
  img = mpimg.imread(png_files[0])
  height, width, channels = img.shape
  num_images = len(png_files)
  return width, height, num_images


dirname = sys.argv[1]
width, height, num_images = get_parameters(dirname)

print("width = %d, height = %d, num_images = %d\n" % (width, height, num_images))
quit()
num_levels = 4
num_hidden_nodes = 15 
light_dim = 1
level = 0
ensemble_size = 5
tolerance = 1e-3
save_data_file_name = "pixel_assignments.cfg"

max_clusters =  int(\
    maximum_clusters(width, height, num_hidden_nodes, num_images))

timed = True

pixel_clusters = \
    PixelClusters(dirname, num_levels, max_clusters, ensemble_size, timed)

flagged = np.ones((height, width), dtype  = bool)
errors = np.ndarray((height, width), dtype = 'float', order='C')

once = False

def get_flagged_clusters(cluster_ids, closest, flagged):
  flagged_ids = closest[np.where(flagged)]
  flagged_ids = np.unique(flagged_ids)
  flagged_ids.sort()
  return flagged_ids

def get_flagged_pixels(relative_error, tolerance):
  return relative_error > tolerance
  
def save_pixel_assignments(file_name, model_directory, assignments):
  height, width, assignment_size = assignments.shape
  ensemble_size = assignment_size - 1
  # Open file.
  with open(file_name, "w") as f:
    f.write("%s\n" % ((model_directory))) # write model directory 
    f.write("%d\n" % ((width))) # write width
    f.write("%d\n" % ((height))) # write height
    f.write("%d\n" % ((ensemble_size))) # ensemble_size
    for x in range(0, height):
      for y in range(0, width):
          for i in range(0, assignment_size):
            f.write("%d" % assignments[y,x,i])

def load_pixel_assignments(file_name):
  with open(file_name, "r") as f:
    lines = f.readlines()
    model_directory = lines[0]
    width = int(lines[1])
    height = int(lines[2])
    ensemble_size = int(lines[3])
    assignment_size = ensemble_size + 1
    assignments = np.zeros((height, width, assignment_size))
    j = 4
    for x in range(0, width):
      for y in range(0, height):
        values = np.array(lines[j].split(" ")).astype(int)
        for i in range(0, assignment_size):
          assignments[y, x, i] = int(values[i])
        j = j + 1
  return model_directory, width, height, assignments


def predict_image(image, average, model_directory, assignments):
  test, batch_sizes = get_test_data(image, average, cluster_assignments) 
  with tf.device('/cpu:0'):
    checkpoint_file = model_directory+'/model_'\
                    +str(level)+'-'+str(cluster_id)+'.hdf5'
    model = ModelMaker(light_dim, num_hidden_nodes)
    model.set_checkpoint_file(checkpoint_file)
    model.compile()
    model.load_weights()
    predictions = model.predict(test, batch_size) 
#    kmeans2d.predictions_to_image(test, predictions, predicted_image)

# y  x level clusters 
assignments = np.zeros((height, width, 1, ensemble_size), dtype = 'int')

for indices, order, centers, labels, closest, train_data, train_labels, batch_sizes\
    in reversed(pixel_clusters):

  if not once:
    kmeans2d.total_norm(train_data, train_labels, totals)
    totals = np.sqrt(totals)
    
  print("closest.shape = %s\n" % str(closest.shape))
  num_samples = len(indices)
  starts = np.zeros(len(batch_sizes), dtype=np.int32)
  ends = np.zeros(len(batch_sizes), dtype=np.int32)
  for i in range(0, len(batch_sizes)):
    if i > 0:
      starts[i] = batch_sizes[i] * num_samples + starts[i-1]
  ends = starts + np.array(batch_sizes) * num_samples
  # get number of clusters
  num_clusters = len(centers)
  print("num_centers = %d\n" % len(centers))
  print ("len(batch_sizes) = %d\n" % len(batch_sizes))
  cluster_ids = get_flagged_clusters(range(0, len(centers)), closest, flagged)

  for cluster_id in cluster_ids:
    checkpoint_file = 'models/model_'+str(level)+'-'+str(cluster_id)+'.hdf5'
    print("[%d] %d/%d checkpoint_file = %s" %
          (level, cluster_id, len(centers) - 1, checkpoint_file))
    if not os.path.exists(checkpoint_file):
      start = time.time()
      with tf.device('/cpu:0'):
        model = ModelMaker(light_dim, num_hidden_nodes)
        model.set_checkpoint_file(checkpoint_file)
        model.compile()
        model.train(train_data[starts[cluster_id]:ends[cluster_id], :], \
            train_labels[starts[cluster_id]:ends[cluster_id], :], 
            batch_sizes[cluster_id])
        K.clear_session()
      end = time.time();
      print("[%d] %d/%d time to train %f\n" % \
          (level, cluster_id, len(centers), end - start))
  
  predicted_images = np.zeros((channels, height, width, num_samples), \
                    dtype = 'float') 

  # Compute all of the predicted images.
  for cluster_id in cluster_ids:
    batch_size = batch_sizes[cluster_id]
    for k in range(0, ensemble_size):
      test, target = kmeans2d.closest_test_target(k, cluster_id, closest,\
                                                  train_data, target_data) 
      checkpoint_file = 'models/model_'+str(level)+'-'+str(cluster_id)+'.hdf5'
      print("[%d] %d/%d checkpoint_file = %s" %
            (level, cluster_id, len(centers) - 1, checkpoint_file))
      with tf.device('/cpu:0'):
        model = ModelMaker(light_dim, num_hidden_nodes)
        model.set_checkpoint_file(checkpoint_file)
        model.compile()
        model.load_weights()
        predictions = model.predict(test, batch_size) 
        kmeans2d.predictions_to_images(order, test, target, predictions, \
                                      predicted_images)
      del test
      del target

  # Compute error.
  kmeans2d.compute_errors(ensemble_size, order, train_data, target_data, \
      predicted_images, errors)

  # Compute relative error.
  kmeans2d.compute_total_values(train_data, target_data, totals)
  relative_error = errors / totals

  # Assign pixels that are approximated well enough.
  flagged = get_flagged_pixels(relative_error, tolerance)

  # Save pixel data.
  save_pixel_assignments(save_data_file_name, model_directory, assignments)

  del train_data
  del train_labels
  del closest
  
  level = level + 1

# Predict all images.
