import sys
import glob
import os
import cluster
from scipy import misc

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
image_avg = average_image(files)

# create clusters
max_num_clusters =  int(\
    cluster.maximum_clusters(width, height, num_hidden_nodes, num_images))
print("width = %d, height = %d, channels = %d, num_images = %d" % \
    (width, height, channels, num_images))
pc = cluster.PixelClusters(max_num_clusters, [width, height])
pc.cluster()

