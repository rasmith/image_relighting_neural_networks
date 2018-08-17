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
import config

from cluster import *

# width = 696
# height = 464
def get_parameters(directory):
  current_dir = os.getcwd()
  os.chdir(directory)
  png_files = glob.glob("*.png")
  img = mpimg.imread(png_files[0])
  height, width, channels = img.shape
  num_images = len(png_files)
  os.chdir(current_dir)
  return width, height, num_images

def init(dirname):
  models_dir = dirname + '/models'
  cfg_dir = dirname + '/cfg'
  if not os.path.exists(models_dir):
    os.mkdir(models_dir) 
  if not os.path.exists(cfg_dir):
    os.mkdir(cfg_dir) 
  return (models_dir, cfg_dir)

def get_flagged_clusters(cluster_ids, closest, flagged):
  flagged_ids = closest[np.where(flagged)]
  flagged_ids = np.unique(flagged_ids)
  flagged_ids.sort()
  return flagged_ids

dirname = sys.argv[1]

width, height, num_images = get_parameters(dirname)

print("width = %d, height = %d, num_images = %d\n" % (width, height, num_images))
num_levels = 4
num_hidden_nodes = 15 
light_dim = 1
level = 0
ensemble_size = 5
tolerance = float(1e-3)

max_clusters =  int(\
    maximum_clusters(width, height, num_hidden_nodes, num_images))

timed = True

pixel_clusters = \
    PixelClusters(dirname, num_levels, max_clusters, ensemble_size, timed)

flagged = np.ones((height, width), dtype  = bool)
errors = np.ndarray((height, width), dtype = np.float32, order='C')

(models_dir, cfg_dir) = init(dirname)
print ("models_dir = %s\n" % (models_dir))
print ("cfg_dir = %s\n" % (cfg_dir))

# y  x level clusters 
assignments = np.zeros((height, width, ensemble_size + 1), dtype = 'int')

for indices, cxx_order, centers, labels, closest, average, train_data, \
    train_labels, batch_sizes\
    in reversed(pixel_clusters):
  print("level = %d" % ((level)))
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
  print("len(cluster_ids) = %d\n" % len(cluster_ids))

  for cluster_id in cluster_ids:
    (checkpoint_file_name, checkpoint_file) = \
        config.get_checkpoint_file_info(models_dir, level, cluster_id)
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
  
  channels = 3

  # Compute all of the predicted images.
  for cluster_id in cluster_ids:
    batch_size = batch_sizes[cluster_id]
    print("batch_size = %d\n" % (batch_size))
    for k in range(0, ensemble_size):
      (checkpoint_file_name, checkpoint_file) = \
          config.get_checkpoint_file_info(models_dir, level, cluster_id)
      test, target = kmeans2d.closest_k_test_target(int(k), int(cluster_id),\
                                              closest, train_data, train_labels) 
      print("[%d] %d/%d checkpoint_file = %s, ensemble %d" %
            (level, cluster_id, len(centers) - 1, checkpoint_file, k))
      print("train_data.shape = %s" % str(train_data.shape))
      print("train_labels.shape = %s" % str(train_labels.shape))
      print("test.shape = %s" % str(test.shape))
      print("target.shape = %s" % str(target.shape))
      with tf.device('/cpu:0'):
        model = ModelMaker(light_dim, num_hidden_nodes)
        model.set_checkpoint_file(checkpoint_file)
        model.compile()
        model.load_weights()
        predictions = model.predict(test, batch_size) 
        kmeans2d.predictions_to_errors(cxx_order, ensemble_size,\
            test, target, predictions, errors);
      del test
      del target

  # TODO: need to update flagged and propagate already flagged clusters
  # flagged = errors > tolerance
  current_flagged = errors > tolerance
  flagged = np.logical_and(flagged, current_flagged)

  # Update assignments.
  for x in range(0, width):
    for y in range(0, height):
      if flagged[y, x]:
        assignments[y, x, 0] = level
        assignments[y, x, 1:ensemble_size + 1] = closest[y, x, :]
        
  # Save pixel assignments to file.
  config.save_cfg(cfg_dir, average, indices, assignments, num_images, level)

  del train_data
  del train_labels
  del closest
  del average
  
  level = level + 1
