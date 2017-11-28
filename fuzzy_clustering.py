import sys
import time
import glob
import os
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import threading
from multiprocessing.dummy import Pool as ThreadPool

from cluster import *
from model import *

class PartitionData:
  def __init__(self, images, num_clusters, cluster_to_pixels, pixel_counts):
    self.images = images
    self.num_clusters = num_clusters
    self.cluster_to_pixels = cluster_to_pixels
    self.pixel_counts = pixel_counts
    self.iteration = 0
    self.max_iteration = len(images) * num_clusters;
    self.cluster = 0;

  def __iter__(self):
    return self

  def __reversed__(self):
    self.counts.reverse()
    return self

  def __next__(self):
    return self.next()

  def next(self):
    if self.iteration >= self.max_iteration:
      raise StopIteration
    for i in range(0, len(self.images)):
      image = self.images[i]
      tuples = [tuple(xy) for xy in self.cluster_to_pixels[self.cluster]]
      batch_size = self.pixel_counts[self.cluster]
      train_labels = []
      for xy in tuples:
          rgb = image[xy]
          train_labels.append([rgb[0], rgb[1], rgb[2]])
      train_data = []
      for xy in tuples:
          avg = image_avg[xy]
          train_data.append([xy[0], xy[1], i, avg[0], avg[1], avg[2]])
    return (train_labels, train_data, batch_size)

class GatherLabels(threading.Thread):
  def __init__(self, tid, num_threads, image, tuples, labels):
    threading.Thread.__init__(self)
    self.tid = tid
    self.num_threads = num_threads
    self.image = image
    self.tuples = tuples
    self.labels = labels

  def run(self):
    block_size = int(len(self.tuples)/self.num_threads)
    start = self.tid * block_size
    end = min(start + block_size, len(self.tuples))
    for i in range(start, end):
      xy = tuples[i]
      rgb = self.image[xy]
      self.labels[i] = [rgb[0], rgb[1], rgb[2]]
      
class GatherData(threading.Thread):
  def __init__(self, tid, num_threads, pos, average, tuples, data):
    threading.Thread.__init__(self)
    self.tid = tid
    self.num_threads = num_threads
    self.pos = pos
    self.average = average
    self.tuples = tuples
    self.data =  data

  def run(self):
    block_size = int(len(self.tuples)/self.num_threads)
    start = self.tid * block_size
    end = min(start + block_size, len(self.tuples))
    for i in range(start, end):
      xy = tuples[i]
      rgb = self.average[xy]
      self.data[i] = [xy[0], xy[1], self.pos, rgb[0], rgb[1], rgb[2]]

class GatherImages(threading.Thread):
  def __init__(self, tid, num_threads, files, images):
    threading.Thread.__init__(self)
    self.tid = tid
    self.num_threads = num_threads
    self.files = files
    self.images = images

  def run(self):
    block_size = int(len(self.images)/self.num_threads)
    start = self.tid * block_size
    end = min(start + block_size, len(self.images))
    for i in range(start, end):
      self.images[i] = misc.imread(self.files[i])

class ImageAverage(threading.Thread):
	def __init__(self, tid, num_threads, image, average):
		threading.Thread.__init__(self)
		self.tid = tid
		self.num_threads = num_threads
		self.image = image
		self.average = average

	def run(self):
		w, h, c = self.image.shape
		block_size = int(h / self.num_threads)
		start = self.tid * block_size
		end = min(start + block_size, h)
		self.average[:,start:end,:] += self.image[:,start:end,:]
		# print("tid=%d, start=%d, end=%d" % (self.tid, start, end))
		
def gather_images(files):
  peek_image = misc.imread(files[0])
  w, h, c = peek_image.shape
  images = len(files)*[np.ndarray(shape=(w,h,c), dtype = float, order = 'C')]
  num_threads = 4
  threads = [GatherImages(i, num_threads, files, images) for i in \
      range(0, num_threads)]
  for i in range(0, num_threads):
    threads[i].start()
  for i in range(0, num_threads):
    threads[i].join()
  return images

def average_image(images):
	for image in images:
		try:
			average 
		except NameError:
			average = image.astype('float')
		else:
			num_threads = 2
			threads = [ImageAverage(i, num_threads, image, average) for i in \
					range(0, num_threads)]
			for i in range(0, num_threads):
				threads[i].start()
			for i in range(0, num_threads):
				threads[i].join()
	average /= float(len(files))
	return average.astype('uint8')

def gather_labels(images, tuples, labels):
  num_threads = 2
  threads = [GatherLabels(i, num_threads, images, tuples, labels) for i in \
      range(0, num_threads)]
  for i in range(0, num_threads):
    threads[i].start()
  for i in range(0, num_threads):
    threads[i].join()

def gather_data(pos, average, tuples, data):
  num_threads = 2 
  threads = [GatherData(i, num_threads, pos, average, tuples, data) for i in \
      range(0, num_threads)]
  for i in range(0, num_threads):
    threads[i].start()
  for i in range(0, num_threads):
    threads[i].join()

directory = sys.argv[1]
print("input_directory = %s" % directory)
os.chdir(directory)
files = glob.glob('*.png')
num_images = len(files)
image  = misc.imread(files[0])
print(image.shape)
print(image.dtype)
width, height, channels  = image.shape
num_hidden_nodes = 15 
num_levels = -1
light_dim = 1

# get average image
print("Gathering images...")
start = time.time()
images = gather_images(files)
end = time.time()
print("Time to gather images = %s" % (str(end - start)))
print("Computing average image...")
start = time.time()
image_avg = average_image(images)
end = time.time()
print("Time to compute average = %s" % (str(end - start)))

show = True 

if show:
  plt.figure()
  image_plot = plt.imshow(image_avg)
  plt.show(image_plot)


# create clusters
print("Cluster...")
max_clusters =  int(\
    maximum_clusters(width, height, num_hidden_nodes, num_images))
print("width = %d, height = %d, channels = %d, num_images = %d" % \
    (width, height, channels, num_images))

queue = []
extents = [width, height]
timed = True
level = 0
pixel_clusters = PixelClusters(num_levels, max_clusters, extents, timed)
model = ModelMaker(light_dim, num_hidden_nodes)
model.compile()
for centers, labels in reversed(pixel_clusters):
  # get number of clusters
  num_clusters = len(centers)
  print("num_centers = %d" % len(centers))

  # partition pixels to clusters
  cluster_to_pixels = {c: [] for c in range(0, num_clusters)}
  for y in range(0, height):
    for x in range(0, width):
      c  = labels[y * width + x]
      cluster_to_pixels[c].append([x, y])
      
  pixels_to_errors = [[0 for x in range(0, width)] for y in range(0, height)] 
  
  # get pixel counts 
  pixel_counts = [len(cluster_to_pixels[k]) for k in cluster_to_pixels.keys()]

  model.reset()

  pool = ThreadPool(4);
  results = pool.map(train_model, model_data);

  # train neural networks at each level
  for i in range(0, len(images)):
    image = images[i]
    start = time.time()
    for c in range(0, num_clusters):
      tuples = [tuple(xy) for xy in cluster_to_pixels[c]]

      # print("Train neural network at cluster %d at level %d on %d x %d pixels" \
          # % (c, level, pixel_counts[c], num_images))
      batch_size = pixel_counts[c]
      train_labels = []
      for xy in tuples:
          rgb = image[xy]
          train_labels.append([rgb[0], rgb[1], rgb[2]])

      train_data = []
      for xy in tuples:
          avg = image_avg[xy]
          train_data.append([xy[0], xy[1], i, avg[0], avg[1], avg[2]])

      queue.append([train_data, train_labels, batch_size])
      #model.train(train_data, train_labels, batch_size)
    end = time.time()
    print ("time = %s" % (str(end - start)))

  level +=1

  min_pixels = min(pixel_counts)
  max_pixels = max(pixel_counts)
  print ("level = %d, min_pixels = %d, max_pixels = %d" % \
      (level, min_pixels, max_pixels))
  if show:
    palette = generate_palette(len(centers))
    image = render_clusters(width, height, labels, palette)
    image = np.array(image)
    image = image.reshape((width, height, channels))
    plt.figure()
    image_plot = plt.imshow(image)
    plt.show(image_plot)

