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

dirname = sys.argv[1]

(model_dir, img_dir, width, height, num_images, sampled, assignments) \
    = config.load_cfg(dirname)


