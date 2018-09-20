import sys
import time
import glob
import os
from scipy import misc
import numpy as np
from multiprocessing import Pool as ThreadPool
import config
from cluster import *
import threading

lock = threading.Lock()

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
  level, network_id, start, count = network_data 
  for t in test_data[start:start + count]:
    x = int(np.round(t[0] * (width - 1)))
    y = int(np.round(t[1] * (height - 1)))
    input_map[y, x, :]  = 255.0*t[3:]

def init(models_dir, img_dir):
  assert os.path.exists(models_dir)
  if not os.path.exists(img_dir):
    os.mkdir(img_dir) 
  return (models_dir, img_dir)

  
def predict_thread(arg):
  network_data, test_data, models_dir = arg
  level, cluster_id, start, count = network_data
  (checkpoint_file_name, checkpoint_file) = \
      config.get_checkpoint_file_info(models_dir, level, cluster_id)
  predictions = kmeans2d.predict(checkpoint_file, test_data[start:start+count])
  return (level, cluster_id, start, count, predictions,\
          test_data[start:start+count]) 

def main():
  dirname = sys.argv[1]
  image_number = int(sys.argv[2])

  (model_dir, img_dir, width, height, num_images, ensemble_size,  max_levels, \
      sampled, assignments, average_img) = config.load_cfg(dirname)

  init(model_dir, img_dir)
                 
  print("image_number = %s, num_images = %s"\
  % (str(image_number), str(num_images)))

  test_data, network_data = kmeans2d.assignment_data_to_test_data(\
    assignments, image_number, num_images, average_img)

  input_map = np.zeros((height, width, 3), dtype = np.float64)
  image_out = np.zeros((height, width, 3), dtype = np.float64)
  thread_data = [(data, test_data, model_dir) for data in network_data]
  start_time = time.clock()
  pool = ThreadPool(8)
  results = pool.map(predict_thread, thread_data)
  pool.close()
  pool.join()
  for result in results:
    # predictions
    (level, network_id, start, count, predictions, test_out) = result 
    kmeans2d.predictions_to_image(image_out, test_out, predictions)

    # debug data
    prediction_out = np.zeros((height, width, 3), dtype = np.float64)
    update_input_map(level, network_id, width, height, test_data,\
                        [level, network_id, start, count], input_map)
    kmeans2d.predictions_to_image(prediction_out, test_out, predictions)
    prediction_file_name = "render_images/prediction_%04d_%04d.png" %\
      (level, network_id)
    misc.imsave(prediction_file_name, prediction_out);

  image_out = np.divide(image_out, float(ensemble_size))
  image_out = np.clip(255.0 * image_out, 0.0, 255.0)
  end_time = time.clock()
  image_file_name = "render_images/" + str(sys.argv[2]) + '.png'
  misc.imsave("render_images/render_input_map.png", input_map)
  misc.imsave(image_file_name, image_out)
  print("time = %5.5f" % (end_time - start_time))
  print("saved %s" % (image_file_name))

if __name__ == '__main__':
  main()



  






