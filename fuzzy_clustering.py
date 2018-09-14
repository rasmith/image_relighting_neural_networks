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
import config

from cluster import *

def save_assignment_map(level, cluster_id, width, height, test_data,\
                        network_data):
  image_out = np.zeros((height, width, 3), dtype = np.float32)
  image_file_name = "render_images/map_%04d_%04d.png" % (level, cluster_id)
  level, network_id, start, count= network_data 
  values = test_data[start:start + count]
  coords = [(x[0] * (width - 1), x[1] * (height - 1)) for x in values]
  coords = np.round(np.array(coords)).astype(int)
  if level == 0 and network_id == 0:
    np.set_printoptions(threshold=np.nan)
    print("coords = %s" % str(coords))
  for x in coords:
    image_out[x[1], x[0], :]  = [255.0, 0.0, 0.0]
  misc.imsave(image_file_name, image_out)

# width = 696
# height = 464
def get_parameters(directory):
  print ("directory = %s" % directory)
  current_dir = os.getcwd()
  os.chdir(directory)
  png_files = glob.glob("*.png")
  img = misc.imread(png_files[0], mode = 'RGB').astype('float32') / 255.0
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
num_levels = 1 
max_levels = 1 
num_hidden_nodes = 15
light_dim = 1
level = 0
ensemble_size = 1 
tolerance = .03

max_clusters =  int(\
    maximum_clusters(width, height, num_hidden_nodes, num_images))

timed = True

pixel_clusters = \
    PixelClusters(dirname, num_levels, max_clusters, ensemble_size, timed)


flagged = np.ones((height, width), dtype  = bool)
used = np.zeros((height, width), dtype  = bool)
errors = np.ndarray((height, width), dtype = np.float32, order='C')

(models_dir, cfg_dir) = init(dirname)
print ("models_dir = %s\n" % (models_dir))
print ("cfg_dir = %s\n" % (cfg_dir))

# y  x level clusters 
assignments = np.zeros((height, width, ensemble_size + 1), dtype = 'int')

for indices, cxx_order, centers, labels, closest, average, train_data, \
    train_labels, batch_sizes\
    in reversed(pixel_clusters):
  if level >=  max_levels:
    break
  palette = generate_palette(len(centers))
  cluster_map = np.array(render_clusters(width, height, labels, palette))
  cluster_map = np.transpose(np.reshape(cluster_map, (width, height, 3)),
      (1, 0, 2))
  cluster_map_file = "render_images/cluster_map_%d.png" % (level)
  print("cluster_map.shape = %s " % str(cluster_map.shape))
  misc.imsave(cluster_map_file, cluster_map)

  # print("level = %d" % ((level)))
  # print("closest.shape = %s\n" % str(closest.shape))
  num_samples = len(indices)
  starts = np.zeros(len(batch_sizes), dtype=np.int32)
  ends = np.zeros(len(batch_sizes), dtype=np.int32)
  for i in range(0, len(batch_sizes)):
    if i > 0:
      starts[i] = batch_sizes[i-1] * num_samples + starts[i-1]
  ends = starts + np.array(batch_sizes) * num_samples
  # get number of clusters
  num_clusters = len(centers)
  # print("num_centers = %d\n" % len(centers))
  # print ("len(batch_sizes) = %d\n" % len(batch_sizes))
  cluster_ids = get_flagged_clusters(range(0, len(centers)), closest, flagged)
  # print("len(cluster_ids) = %d\n" % len(cluster_ids))

  for cluster_id in cluster_ids:
    (checkpoint_file_name, checkpoint_file) = \
        config.get_checkpoint_file_info(models_dir, level, cluster_id)
    # print("[%d] %d/%d checkpoint_file = %s" %
          # (level, cluster_id, len(centers) - 1, checkpoint_file))
    count = ends[cluster_id] - starts[cluster_id]
    network_data = np.array([level, cluster_id, starts[cluster_id], count])
    # print("level = %d cluster_id = %d  start = %d count = %d\n"%\
      # (level, cluster_id, starts[cluster_id], count))
    save_assignment_map(level, cluster_id, width, height,\
      train_data, network_data)
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
          (level, cluster_id, len(centers) - 1, end - start))
  
  channels = 3

  # Compute all of the predicted images.
  cluster_index = 0
  for cluster_id in cluster_ids:
    if level == max_levels - 1:
      break;
    batch_size = batch_sizes[cluster_id]
    print("batch_size = %d\n" % (batch_size))
    (checkpoint_file_name, checkpoint_file) = \
        config.get_checkpoint_file_info(models_dir, level, cluster_id)
    model = ModelMaker(light_dim, num_hidden_nodes)
    model.set_checkpoint_file(checkpoint_file)
    model.compile()
    model.load_weights()
    for k in range(0, ensemble_size):
      test, target = kmeans2d.closest_k_test_target(int(k), int(cluster_id),\
                                              closest, train_data, train_labels) 
      # print("[%d] %d/%d, %d/%d checkpoint_file = %s, ensemble %d" %
            # (level, cluster_index, len(cluster_ids) - 1, cluster_id, \
                # len(centers) - 1, checkpoint_file, k))
      with tf.device('/cpu:0'):
        predictions = model.predict(test, batch_size) 
        kmeans2d.predictions_to_errors(cxx_order, ensemble_size,\
            test, target, predictions, errors);
      del test
      del target
    cluster_index = cluster_index + 1
  current_flagged = errors > tolerance
  flagged = np.logical_and(flagged, current_flagged)
  if level == max_levels - 1:
    flagged = np.zeros(flagged.shape)

  # Update assignments.
  for x in range(0, width):
    for y in range(0, height):
      if flagged[y, x] == 0 and not used[y,x]:
        assignments[y, x, 0] = level
        assignments[y, x, 1:] = closest[y, x, :]
        used[y, x] = 1
        
  np.set_printoptions(threshold=np.nan)
  print("closest[0, 0, :] = %s" % str(closest[0, 0, :]))
  print("closest[1, 0, :] = %s" % str(closest[1, 0, :]))
  print("closest[2, 0, :] = %s" % str(closest[2, 0, :]))
  print("closest[0, 1, :] = %s" % str(closest[0, 1, :]))
  print("closest[1, 1, :] = %s" % str(closest[1, 1, :]))
  print("closest[2, 1, :] = %s" % str(closest[2, 1, :]))
  print("closest[0, 2, :] = %s" % str(closest[0, 2, :]))
  print("closest[1, 2, :] = %s" % str(closest[1, 2, :]))
  print("closest[2, 2, :] = %s" % str(closest[2, 2, :]))
  print("assignments[0, 0, :] = %s" % str(assignments[0, 0, :]))
  print("assignments[1, 0, :] = %s" % str(assignments[1, 0, :]))
  print("assignments[2, 0, :] = %s" % str(assignments[2, 0, :]))
  print("assignments[0, 1, :] = %s" % str(assignments[0, 1, :]))
  print("assignments[1, 1, :] = %s" % str(assignments[1, 1, :]))
  print("assignments[2, 1, :] = %s" % str(assignments[2, 1, :]))
  print("assignments[0, 2, :] = %s" % str(assignments[0, 2, :]))
  print("assignments[1, 2, :] = %s" % str(assignments[1, 2, :]))
  print("assignments[2, 2, :] = %s" % str(assignments[2, 2, :]))


  # Save pixel assignments to file.
  config.save_cfg(cfg_dir, average, indices, assignments, num_images, level)

  del train_data
  del train_labels
  del closest
  del average
  
  level = level + 1
