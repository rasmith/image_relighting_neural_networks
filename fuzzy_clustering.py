import sys
import time
import glob
import os
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

from cluster import *
from model import *

def gather_images(files):
  images = []
  for file in files:
    images.append(misc.imread(file))
  return images

def average_image(images):
  for image in images:
    try:
      average 
    except NameError:
      average = image.astype('float')
    else:
      average += image.astype('float')
  average /= float(len(files))
  return average.astype('uint8')

def gather_training_data(cluster_pixels, images, average):  
  tuples = [tuple(xy) for xy in cluster_pixels]
  l = len(cluster_pixels)

  train_data = (len(images)*len(cluster_pixels))*[6*[float]]
  for i in range(0, len(images)):
    pos = float(i)/float(len(images))
    train_data[i*l:(i+1)*l] = [[xy[0], xy[1], pos, average[xy][0],\
        average[xy][1], average[xy][2]] for xy in tuples]

  train_labels = (len(images)*len(cluster_pixels))*[3*[float]]
  for i in range(0, len(images)):
    image = images[i]
    train_labels[i*l:(i+1)*l] = [image[xy] for xy in tuples]

  return train_data, train_labels

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
num_levels = 4
light_dim = 1

# get average image
print("Computing average image...")
images = gather_images(files)
image_avg = average_image(images)

show = False

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

  # train neural networks at each level
  for c in range(0, num_clusters):
    model.reset()
    print("Train neural network at cluster %d at level %d on %d x %d pixels" \
         % (c, level, pixel_counts[c], num_images))
    batch_size = pixel_counts[c]

    start = time.time()
    train_data, train_labels = \
        gather_training_data(cluster_to_pixels[c], images, image_avg)
    end = time.time()
    print ("time = %s" % (str(end - start)))
    # model.train(train_data, train_labels, batch_size)

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

