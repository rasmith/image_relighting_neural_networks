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

import concurrent.futures

import config
from cluster import *

import threading

lock = threading.Lock()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def save_assignment_map(level, cluster_id, width, height, test_data,\
                        network_data):
  image_out = np.zeros((height, width, 3), dtype = np.float64)
  image_file_name = "render_images/amap_%04d_%04d.png" % (level, cluster_id)
  # print("network_data = %s" % str(network_data))
  level, network_id, start, count= network_data
  values = test_data[start:start + count]
  coords = [(x[0] * (width - 1), x[1] * (height - 1)) for x in values]
  coords = np.round(np.array(coords)).astype(int)
  if level == 0 and network_id == 0:
    np.set_printoptions(threshold=np.nan)
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

def init(models_dir, img_dir):
  assert os.path.exists(models_dir)
  if not os.path.exists(img_dir):
    os.mkdir(img_dir) 
  return (models_dir, img_dir)

dirname = sys.argv[1]
image_number = int(sys.argv[2])

(model_dir, img_dir, width, height, num_images, ensemble_size,  max_levels, \
    sampled, assignments, average_img) = config.load_cfg(dirname)

init(model_dir, img_dir)

# Need to get [[L, i, i, i, i, i],
#              [L, i, i, i, i, i], ...
# to [x, y, j, r, g, b], ...
# load model file, execute
               
print("assignments.shape = %s, average_img.shape = %s"\
  % (assignments.shape, average_img.shape))

print("image_number = %s, num_images = %s" % (str(image_number), str(num_images)))

print("assignments.dtype= %s, average_img.dtype = %s"\
  % (str(assignments.dtype), str(average_img.dtype)))

test_data, network_data = kmeans2d.assignment_data_to_test_data(\
  assignments, image_number, num_images, average_img)

print ("test_data.shape = %s, network_data.shape = %s" % \
  (test_data.shape, network_data.shape))

# # predict image pixels
light_dim = 1
num_hidden_nodes = 15 
start = 0
end = 0
input_map = np.zeros((height, width, 3), dtype = np.float64)
  
def predict(arg):
  level, network_id, start, batch_size = arg
  save_assignment_map(level, network_id, width, height, test_data,\
                        [level, network_id, start, batch_size])
  update_input_map(level, network_id, width, height, test_data,\
                        [level, network_id, start, batch_size],\
                        input_map)
  end = start + batch_size 
  (checkpoint_file_name, checkpoint_file) = \
      config.get_checkpoint_file_info(model_dir, level, network_id)
  graph = tf.Graph()
  with graph.as_default():
    model = ModelMaker(light_dim, num_hidden_nodes)
    model.set_checkpoint_file(checkpoint_file)
    model.compile()

  with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    model.load_weights()
    predictions = model.predict(test_data[start:end], batch_size) 
  return (checkpoint_file_name, level, network_id, predictions, \
          test_data[start:end])

def main():
  image_out = np.zeros((height, width, 3), dtype = np.float64)

  start = time.clock()
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    for result in executor.map(predict, network_data):
      (checkpoint_file_name, level, network_id, predictions, test_out) = result 
      # print("====================================")
      # print ("type(image_out) = %s" % type(image_out))
      # print ("type(test_out) = %s" % type(test_out))
      # print("type(predictions) = %s" % type(predictions))
      # print("predictions = %s" % predictions)
      # print("type(predictions[0]) = %s" % type(predictions[0]))
      # predictions = np.asarray(predictions, order='C', dtype='float64')
      # print("--predictions = %s" % predictions)
      # print("--type(predictions) = %s" % type(predictions))
      # print("--type(predictions[0]) = %s" % type(predictions[0]))
      # print("predictions.shape = %s" % (str(predictions.shape)))
      # print("test_out.shape = %s" % (str(test_out.shape)))
      print("isfortran = %s" % (str(np.isfortran(predictions))))
      kmeans2d.predictions_to_image(image_out, test_out, predictions)
      prediction_out = np.zeros((height, width, 3), dtype = np.float64)
      kmeans2d.predictions_to_image(prediction_out, test_out, predictions)
      prediction_file_name = "render_images/prediction_%04d_%04d.png" %\
        (level, network_id)
      misc.imsave(prediction_file_name, prediction_out);

      print("checkpoint_file_name=%s" % (checkpoint_file_name))
      # print("predictions = %s" % (predictions))
      # print("test = %s" % (test_out))
  # image_out = np.divide(image_out, float(ensemble_size))
  end = time.clock()
  image_file_name = "render_images/" + str(sys.argv[2]) + '.png'
  print("time = %5.5f" % (end - start))
  print("saved %s" % (image_file_name))
  # image_out = image_out - np.min(image_out)
  # image_out = np.divide(image_out, np.max(image_out))
  image_out = np.clip(255.0 * image_out, 0.0, 255.0)
  misc.imsave("render_images/render_input_map.png", input_map)
  misc.imsave(image_file_name, image_out)
  # for data in network_data:
    # test(data)

if __name__ == '__main__':
  main()



  






