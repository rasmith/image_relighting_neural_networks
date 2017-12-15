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

def init(dirname):
  models_dir = dirname + '/models'
  cfg_dir = dirname + '/cfg'
  os.mkdir(models_dir) if not os.path.exists(models_dir)
  os.mkdir(cfg_dir) if not os.path.exists(cfg_dir)

dirname = sys.argv[1]
destdir = sys.argv[2] if len(sys.argv) > 1 else ''

width, height, num_images = get_parameters(dirname)

print("width = %d, height = %d, num_images = %d\n" % (width, height, num_images))
quit()
num_levels = 4
num_hidden_nodes = 15 
light_dim = 1
level = 0
ensemble_size = 5
tolerance = 1e-3

max_clusters =  int(\
    maximum_clusters(width, height, num_hidden_nodes, num_images))

timed = True

pixel_clusters = \
    PixelClusters(dirname, num_levels, max_clusters, ensemble_size, timed)

flagged = np.ones((height, width), dtype  = bool)
errors = np.ndarray((height, width), dtype = 'float', order='C')

once = False

init(dirname)

def get_flagged_clusters(cluster_ids, closest, flagged):
  flagged_ids = closest[np.where(flagged)]
  flagged_ids = np.unique(flagged_ids)
  flagged_ids.sort()
  return flagged_ids

def get_flagged_pixels(relative_error, tolerance):
  return relative_error > tolerance
  
def save_cfg(dirname, average, sampled, assignments):
  cfg = dirname + '/cfg/relighting.cfg'
  height, width, assignment_size = assignments.shape
  ensemble_size = assignment_size - 1
  # Open file.
  with open(cfg, "w") as f:
    f.write("%d\n" % (width)) # write width
    f.write("%d\n" % (height)) # write height
    f.write("%d\n" % (num_images)) # write num_images
    f.write("%d\n" % (ensemble_size)) # ensemble_size
    f.write("%s\n" % (' '.join([str(i) for i in x]))) # sampled images
    for x in range(0, height):
      for y in range(0, width):
          for i in range(0, assignment_size):
            f.write("%d" % assignments[y,x,i])
  mpimg.imsave(cfg + '/average.png', average)

def load_cfg(dirname):
  cfg = dirname + '/cfg/relighting.cfg'
  with open(cfg, "r") as f:
    lines = f.readlines()
    width = int(lines[0])
    height = int(lines[1])
    num_images = int(lines[2])
    ensemble_size = int(lines[3])
    sampled = [int(i) for i in lines[4].split(' ')]
    assignment_size = ensemble_size + 1
    assignments = np.zeros((height, width, assignment_size))
    j = 5
    for y in range(0, height):
     for x in range(0, width):
        values = np.array(lines[j].split(" ")).astype(np.int)
        for i in range(0, assignment_size):
          assignments[y, x, i] = int(values[i])
        j = j + 1
  img_dir = dirname + '/img'
  model_dir= dirname + '/models'
  return model_dir, img_dir, width, height, num_images, sampled, assignments


def predict_images(dirname, dest_dir):
  model_dir, img_dir, width, height, num_images, sampled, assignments = \
    load_cfg(dirname)
  img = mpimg.imread(cfg_dir + "/average.png")
  for i in range(0, images):
    image = predict_image(i, average, model_dir, assignments)
    mpimpg.imsave("%03d_out.png", image)
    
def predict_img(i, average, model_dir, assignments):
  with tf.device('/cpu:0'):
    test, batch_sizes, starts, ends, levels, cluster_ids = \
      kmeans2d.assignments_to_predict_data( assignments)
    for i in range(0, len(cluster_ids)):
      level = level[i]
      cluster_id = cluster_ids[i]
      batch_size = batch_size[i]
      start = starts[i]
      end = ends[i]
      checkpoint_file = model_dir+'/model_'\
                      +str(level)+'-'+str(cluster_id)+'.hdf5'
      model = ModelMaker(light_dim, num_hidden_nodes)
      model.set_checkpoint_file(checkpoint_file)
      model.compile()
      model.load_weights()
      predictions = model.predict(test[start:end], batch_size) 
      kmeans2d.predictions_to_img(test[start:end], predictions, predicted_img)

# y  x level clusters 
assignments = np.zeros((height, width, 1, ensemble_size), dtype = 'int')

for indices, order, centers, labels, closest, average, train_data, \
    train_labels, batch_sizes\
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
  flagged = relative_error > tolerance

  # Update assignments.
  for x in range(0, width):
    for y in range(0, height):
      if flagged[y, x]:
        assignments[y, x, 0] = level
        assignments[y, x, 1:ensemble_size] = closest[x, y]
        
  # Save pixel assignments to file.
  save_cfg(dirname, average, sampled, assignments)

  del train_data
  del train_labels
  del closest
  del average
  
  level = level + 1

# Predict all images.
if not destdir == '':
  predict_images(dirname, destdir)

