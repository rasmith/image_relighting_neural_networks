import sys
import time
import glob
import os
from scipy import misc
import numpy as np
from multiprocessing import Pool as ThreadPool
from threading import Lock
import config
import concurrent.futures

from cluster import *

def predict_thread(predict_data):
    start_time = time.time()
    level, cluster_id, k, ensemble_size, closest, errors,\
      train_data, train_labels, cxx_order, models_dir, lock = predict_data
    test, target = kmeans2d.closest_k_test_target(int(k), int(cluster_id),\
                                         closest, train_data, train_labels) 
    (checkpoint_file_name, checkpoint_file) = \
        config.get_checkpoint_file_info(models_dir, level, cluster_id)
    predictions = kmeans2d.predict(checkpoint_file_name, test)
    lock.acquire()
    kmeans2d.predictions_to_errors(cxx_order, ensemble_size,\
          test, target, predictions, errors);
    lock.release()
    del test
    del target
    end_time = time.time()
    print("predict: [%d] %d %f" % (level, cluster_id, end_time - start_time))

def train_thread(thread_data):
  cluster_id, start, end, models_dir, level, train_data, train_labels\
    = thread_data
  print("[%d] %d begin training" % (level, cluster_id))
  return train(cluster_id, models_dir, level,\
                    train_data[start:end],\
                    train_labels[start:end])

def train(cluster_id, models_dir, level, train_data, train_labels):
  (checkpoint_file_name, checkpoint_file) = \
      config.get_checkpoint_file_info(models_dir, level, cluster_id)
  start = time.time()
  accuracy = kmeans2d.train_network(checkpoint_file, train_data,\
      train_labels, num_hidden_nodes)
  end = time.time();
  print("[%d] %d/%d time to train %f with accuracy %f\n" % \
      (level, cluster_id, len(centers) - 1, end - start, accuracy))
  return (cluster_id, accuracy, end - start)

def save_assignment_map(level, cluster_id, width, height, test_data,\
                        network_data):
  image_out = np.zeros((height, width, 3), dtype = np.float64)
  image_file_name = "render_images/map_%04d_%04d.png" % (level, cluster_id)
  level, network_id, start, count= network_data 
  values = test_data[start:start + count]
  coords = [(x[0] * (width - 1), x[1] * (height - 1)) for x in values]
  coords = np.round(np.array(coords)).astype(int)
  for x in coords:
    image_out[x[1], x[0], :]  = [255.0, 0.0, 0.0]
  misc.imsave(image_file_name, image_out)

def update_input_map(level, cluster_id, width, height, test_data,\
                        network_data, input_map):
  level, network_id, start, count= network_data 
  for t in test_data[start:start + count]:
    x = int(np.round(t[0] * (width - 1)))
    y = int(np.round(t[1] * (height - 1)))
    input_map[y, x, :]  = 255.0*t[3:]

def update_label_map(level, cluster_id, width, height, test_data,\
                        label_data, network_data, label_map):
  level, network_id, start, count= network_data 
  for i in range(start, start + count):
    t = test_data[i]
    l = label_data[i]
    x = int(np.round(t[0] * (width - 1)))
    y = int(np.round(t[1] * (height - 1)))
    label_map[y, x, :]  = 255.0 * l

def update_accuracy_map(network_data, test_data, accuracy, accuracy_map):
  level, network_id, start, count= network_data 
  for t in test_data[start:start + count]:
    x = int(np.round(t[0] * (width - 1)))
    y = int(np.round(t[1] * (height - 1)))
    accuracy_map[y, x, 0] = 255.0 * accuracy
    accuracy_map[y, x, 1] = 255.0 * accuracy
    accuracy_map[y, x, 2] = 255.0 * accuracy

# width = 696
# height = 464
def get_parameters(directory):
  print ("directory = %s" % directory)
  current_dir = os.getcwd()
  os.chdir(directory)
  png_files = glob.glob("*.png")
  img = misc.imread(png_files[0], mode = 'RGB').astype('float64') / 255.0
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
max_levels = 4 
num_hidden_nodes = 15
level = 0
ensemble_size = 5 
tolerance = .03

max_clusters =  int(\
    maximum_clusters(width, height, num_hidden_nodes, num_images))

pixel_clusters = \
    PixelClusters(dirname, num_levels, max_clusters, ensemble_size, True)

flagged = np.ones((height, width), dtype  = bool)
used = np.zeros((height, width), dtype  = bool)

(models_dir, cfg_dir) = init(dirname)
print ("models_dir = %s\n" % (models_dir))
print ("cfg_dir = %s\n" % (cfg_dir))

# y  x level clusters 
assignments = np.zeros((height, width, ensemble_size + 1), dtype = 'int')

for indices, cxx_order, centers, labels, closest, average, train_data, \
    train_labels, batch_sizes\
    in reversed(pixel_clusters):
  print ("level %d" % (level))
  if level >=  max_levels:
    break
  errors = np.zeros((height, width), dtype = np.float64, order='C')
  accuracy_map = np.zeros((height, width, 3), dtype = np.float64)
  input_map = np.zeros((height, width, 3), dtype = np.float64)
  label_map = np.zeros((height, width, 3), dtype = np.float64)

  print ("Compute start and end locations")
  num_samples = len(indices)
  starts = np.zeros(len(batch_sizes), dtype=np.int32)
  ends = np.zeros(len(batch_sizes), dtype=np.int32)
  for i in range(0, len(batch_sizes)):
    if i > 0:
      starts[i] = batch_sizes[i-1] * num_samples + starts[i-1]
  ends = starts + np.array(batch_sizes) * num_samples

  print ("Get clusters to train.")
  # get number of clusters
  num_clusters = len(centers)
  cluster_ids = get_flagged_clusters(range(0, len(centers)), closest, flagged)
  clusters_to_train = [c for c in cluster_ids if not os.path.exists(
    config.get_checkpoint_file_info(models_dir, level, c)[1])]

  thread_data = \
        [(i, starts[i],  ends[i], models_dir, level, train_data, train_labels)
        for i in clusters_to_train]

  print ("Launch thread pool for training.")
  pool = ThreadPool(8)
  results = pool.map(train_thread, thread_data)
  pool.close()
  pool.join()

  print ("Update debug data.")
  for result in results:
    (cluster_id, accuracy, execution_time) = result
    (checkpoint_file_name, checkpoint_file) = \
        config.get_checkpoint_file_info(models_dir, level, cluster_id)
    count = ends[cluster_id] - starts[cluster_id]
    network_data = np.array([level, cluster_id, starts[cluster_id], count])
    print ("%d save_assignment_map" % (cluster_id))
    save_assignment_map(level, cluster_id, width, height,\
      train_data, network_data)
    print ("%d update_input_map" % (cluster_id))
    update_input_map(level, cluster_id, width, height,\
      train_data, network_data, input_map)
    print ("%d update_label_map" % (cluster_id))
    update_label_map(level, cluster_id, width, height,\
      train_data, train_labels, network_data, label_map)
    print ("%d update_accuracy_map" % (cluster_id))
    update_accuracy_map(network_data, train_data, accuracy, accuracy_map)
    
  # Compute all of the predicted images and error.
  if level < max_level -1:
    print ("Launch thread pool for error computation.")
    lock = Lock()
    pool = ThreadPool(8)
    predict_data = [(level, cluster_id, k, ensemble_size, closest, errors,\
        train_data, train_labels, cxx_order, models_dir, lock)\
        for cluster_id in cluster_ids for k in range(0, ensemble_size)]
    results = pool.map(predict_thread, predict_data)
    pool.close()
    pool.join()

  # Compute flagged pixels.
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
        
  # Debug images.
  palette = generate_palette(len(centers))
  cluster_map = np.array(render_clusters(width, height, labels, palette))
  cluster_map = np.transpose(np.reshape(cluster_map, (width, height, 3)),
      (1, 0, 2))
  misc.imsave("render_images/cluster_map_%d.png" % (level), cluster_map)
  misc.imsave("render_images/input_map_%d.png" % (level), input_map)
  misc.imsave("render_images/label_map_%d.png" % (level), label_map)
  misc.imsave("render_images/accuracy_map_%d.png" % (level), accuracy_map)

  # Save pixel assignments to file.
  config.save_cfg(cfg_dir, average, indices, assignments, num_images, level)

  del train_data
  del train_labels
  del closest
  del average
  
  level = level + 1
