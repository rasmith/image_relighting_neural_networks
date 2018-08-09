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

import concurrent.futures

import config
from cluster import *

import threading

lock = threading.Lock()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

test_data, ensemble_data = kmeans2d.assignment_data_to_test_data(\
  assignments, image_number, num_images, average_img)

print ("test_data.shape = %s, ensemble_data.shape = %s" % \
  (test_data.shape, ensemble_data.shape))

# # predict image pixels
light_dim = 1
num_hidden_nodes = 15 
start = 0
end = 0
# for data in ensemble_data:
  # level, ensemble_id, count =  data
  # batch_size = count 
  # start = end
  # end = start + count
  # print("level = %d, ensemble_id = %d, count = %d, start = %d, end = %d\n"\
    # % (level, ensemble_id, count, start, end)) 
  # (checkpoint_file_name, checkpoint_file) = \
      # config.get_checkpoint_file_info(model_dir, level, ensemble_id)
  # with tf.device('/cpu:0'):
    # model = ModelMaker(light_dim, num_hidden_nodes)
    # model.set_checkpoint_file(checkpoint_file)
    # model.compile()
    # model.load_weights()
    # predictions = model.predict(test_data[start:end], batch_size) 
  
def predict(arg):
  level, ensemble_id, start, end, batch_size = arg
  (checkpoint_file_name, checkpoint_file) = \
      config.get_checkpoint_file_info(model_dir, level, ensemble_id)
  # print("checkpoint_file = %s\n" % (checkpoint_file))
  graph = tf.Graph()
  with graph.as_default():
    model = ModelMaker(light_dim, num_hidden_nodes)
    model.set_checkpoint_file(checkpoint_file)
    model.compile()

  with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    model.load_weights()
    predictions = model.predict(test_data[start:end], batch_size) 
  # lock.acquire()
  # tid = threading.get_ident()
  # print("=============================")
  # print("tid = %d" % (tid))
  # print("len=%d" % (len(test_data)))
  # print("start = %d, end = %d, batch_size = %d" % (start, end, batch_size))
  # print("checkpoint_file_name =%s" % (checkpoint_file_name))
  # print("predictions = %s" % (predictions))
  # print("test = %s" % (test_data[start:end]))
  # kmeans2d.predictions_to_image(image_out, test_out[start:end], predictions)
  # print("=============================")
  # lock.release()
  return (checkpoint_file_name, predictions, test_data[start:end])

def main():
  image_out = np.zeros((height, width, 3), dtype = np.float32)
  start = time.clock()
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    for result in executor.map(predict, ensemble_data):
      (checkpoint_file_name, predictions, test_out)  = result 
      kmeans2d.predictions_to_image(image_out, test_out, predictions)
      # print("checkpoint_file_name =%s" % (checkpoint_file_name))
      # print("predictions = %s" % (predictions))
      # print("test = %s" % (test_out))
  image_out = np.divide(image_out, float(ensemble_size))
  end = time.clock()
  image_file_name = "render_images/" + str(sys.argv[2]) + '.png'
  print("time = %5.5f" % (end - start))
  print("saved %s" % (image_file_name))
  mpimg.imsave(image_file_name, image_out)
  # for data in ensemble_data:
    # test(data)

if __name__ == '__main__':
  main()



  






