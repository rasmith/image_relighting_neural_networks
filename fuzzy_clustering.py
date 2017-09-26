import sys
import glob
import os
import cluster
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

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
num_hidden_nodes = 10

# get average image
print("Computing average image...")
image_avg = average_image(files)

if len(sys.argv) >=3 and sys.argv[2] == "show":
  #show it
  plt.figure()
  image_plot = plt.imshow(image_avg)
  plt.show(image_plot)

# create clusters
print("Cluster...")
max_num_clusters =  int(\
    cluster.maximum_clusters(width, height, num_hidden_nodes, num_images))
print("width = %d, height = %d, channels = %d, num_images = %d" % \
    (width, height, channels, num_images))
pc = cluster.PixelClusters(max_num_clusters, [width, height])
pc.cluster()

if len(sys.argv) >=3 and sys.argv[2] == "show":
  #show it
  for i in range(0, len(pc.levels)):
    centers, labels = pc.levels[i]
    palette = cluster.generate_palette(len(centers[0]))
    print("num_centers = %d" % len(centers[0]))
    image = cluster.render_clusters(width, height, labels[0], palette)
    image = np.array(image)
    image = image.reshape((width, height, channels))
    plt.figure()
    image_plot = plt.imshow(image)
    plt.show(image_plot)
