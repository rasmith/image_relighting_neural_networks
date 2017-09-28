import sys
import glob
import os
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

from cluster import *
from model import *

def average_image(files):
  for file in files:
    image = misc.imread(file)
    try:
      average 
    except NameError:
      average = image.astype('float')
    else:
      average += image.astype('float')
  average /= float(len(files))
  return average.astype('uint8')

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

# get average image
print("Computing average image...")
image_avg = average_image(files)

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

extents = [width, height]
timed = True
level = 0
pixel_clusters = PixelClusters(num_levels, max_clusters, extents, timed)
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
   pixels_to_clusters = [[0 for x in range(0, width)] for y in range(0, height)] 

  for c in range(0, num_clusters):

  
  level +=1

  # print and show
  pixel_counts = [len(cluster_to_pixels[k]) for k in cluster_to_pixels.keys()]
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

